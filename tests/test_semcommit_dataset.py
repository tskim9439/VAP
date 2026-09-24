"""semcommit_dataset — 가짜 tokenizer·합성 행으로 버킷팅 래퍼·SEM/턴 종료 배치·B/N 결정 위치·패딩·collate 를 검사(CPU, 오디오 없이; pcm 조립만 합성 파일).
데이터셋 기본은 <SEM_END> 만(turn_end=False). 턴 종료를 켜면 토큰은 Phase 2 <EOT> — 아래 SEM/TURN 상수는 순수 함수용 가짜 id."""
import json, random
import numpy as np, pytest, torch
from vapasr.data.interleave import build_interleaved, Specials
from vapasr.data.semcommit_dataset import (build_interleaved_events, build_semcommit_tokens, build_semcommit_sequence, SemCommitDataset, collate_semcommit,
                                           add_semcommit_specials, last_emit_chunk, semcommit_round_robin, SEM_SPECIALS, semcommit_fingerprint, fingerprint_diff,
                                           checkpoint_fingerprint_diff, vocab_sha256, FP_PROTOCOL)
from vapasr.data.dialogue_tokens import PHASE2_SPECIALS
from vapasr.uslm.interleave_data import SPECIAL_TOKENS

class FakeTok:
    """add_tokens / convert_tokens_to_ids / __call__ 만 흉내. 특수 토큰은 base 부터 추가 순서대로, 텍스트는 문자 코드."""
    def __init__(self, base=150000): self.v = {"<|audio_pad|>": 149999}; self.base = base
    def add_tokens(self, toks, special_tokens=True):
        for t in toks: self.v.setdefault(t, self.base + len(self.v) - 1)
    def convert_tokens_to_ids(self, t): return self.v[t]
    def __call__(self, text, add_special_tokens=False, **kw): return {"input_ids": [ord(c) % 1000 + 1 for c in text]}

SP = Specials(next_audio=90, empty_audio=91, spk=(92, 93))
SEM, TURN = 94, 95

# 단어: (text, [(tid, t_end)])  — w1 은 두 토큰 같은 시각, w2/w3 은 같은 종료 시각(동률), w6 은 시각이 다른 두 토큰(KO 부분 어절형)
WORDS = [("hello", [(1001, 0.62)]), ("world", [(1002, 0.94), (1003, 0.94)]), ("and", [(1004, 1.26)]), ("then", [(1005, 1.26)]),
         ("um", [(1006, 1.58)]), ("so", [(1007, 1.90)]), ("done", [(1008, 2.22), (1009, 2.30)])]

def make_row(sid="ls-1", words=WORDS, tail=0.3, lang="English", path="/x/a.pcm"):
    toks, ws = [], []
    for i, (text, tt) in enumerate(words):
        a = len(toks); toks += [[tid, t] for tid, t in tt]; ws.append(dict(i=i, text=text, a=a, b=len(toks), end_time=max(t for _, t in tt), seg=0, tags=[]))
    dur = round(toks[-1][1] + tail, 3)
    return dict(id=sid, set="librispeech-test", lang=lang, K=int(round(dur * 12.5)), duration_s=dur, text=" ".join(w for w, _ in words), tokens=toks,
                segments=[dict(path=path, offset_s=0.3, silence_before_s=0.3, dur_s=round(dur - 0.6, 3), utt_id="u1", raw_text="x")], words=ws)

def make_labels(sid="ls-1", cands=((1, "A"), (2, "A"), (4, "N"), (5, "B"), (6, "A")), turn_end=True, lang="English"):
    return dict(id=sid, lang=lang, candidates=[dict(after_word=i, grade=g, why="t", stageA=True) for i, g in cands], turn_end=turn_end)

def seq_of(row, lab, delay, K=None, hardneg=1.0, turn_end=True):
    ev, marks = build_semcommit_tokens(row["words"], row["tokens"], lab, SEM, TURN, turn_end=turn_end)
    return ev, marks, build_semcommit_sequence(ev, marks, K or row["K"], [1, 2, 3], 89, SP, delay, hardneg_weight=hardneg, sem_id=SEM, turn_id=TURN)

def chunk_at(s, p):
    """시퀀스 위치 p 가 속한 청크(앞쪽 가장 가까운 audio 자리의 chunk_of; flush 면 -1)."""
    for q in range(p, -1, -1):
        if s["is_input"][q] and s["chunk_of"][q] >= 0: return s["chunk_of"][q]
        if s["is_input"][q] and s["ids"][q] == SP.empty_audio: return -1
    return None

# ── 래퍼 = build_interleaved (이벤트 없음)
def test_wrapper_equals_build_interleaved_without_events():
    rng = random.Random(0)
    for trial in range(400):
        streams = []
        for s in range(rng.choice([1, 1, 2])):
            n = rng.randint(0, 30); ts = sorted(rng.choice([round(rng.uniform(0, 4), 3), round(rng.randint(0, 50) * 0.08, 2)]) for _ in range(n))
            streams.append([(rng.randint(1, 500), t) for t in ts])
        dur, d, M, tags = rng.uniform(0.1, 4.0), rng.randint(0, 6), rng.choice([0, 0, 1, 2, 3]), rng.choice([True, False])
        ref = build_interleaved(streams, dur, SP, delay_frames=d, max_per_chunk=M, add_spk_tags=tags)
        got = build_interleaved_events(streams, dur, SP, delay_frames=d, max_per_chunk=M, add_spk_tags=tags)
        assert got[0] == ref[0] and got[1] == ref[1], (trial, streams, dur, d, M, tags)
        got3 = build_interleaved_events(streams, dur, SP, delay_frames=d, max_per_chunk=M, add_spk_tags=tags, with_src=True)
        assert got3[0] == ref[0] and all(len(e) == len(sr) for (_, e), sr in zip(got3[0], got3[2]))
    chunks, _ = build_interleaved_events([[(3200, 1.0), (13, 1.0)]], .95, SP, delay_frames=2, add_spk_tags=False)   # 19a2594 flush 동률 순서
    assert chunks[-1][1] == [3200, 13, 91]

def test_explicit_event_waits_for_predecessors_and_min_chunk():
    toks = [(7, 0.10), (8, 0.50), (9, 0.0, 3)]                          # 명시 이벤트: k_min 3, 앞선 원소(청크 0+δ, 6+δ) 뒤
    for d in (0, 2):
        ch, _, sr = build_interleaved_events([toks], 2.0, SP, delay_frames=d, add_spk_tags=False, with_src=True)
        k9 = [k for k, e in ch if 9 in e][0]; assert k9 == max(3, int(0.50 / 0.08) + d)
        e = dict(ch)[k9]; assert e[e.index(9) - 1] == 8 and e[-1] == SP.next_audio
    ch, _ = build_interleaved_events([toks], 0.3, SP, delay_frames=2, add_spk_tags=False)   # 스트림이 짧으면 flush 에서 EMPTY 앞
    assert ch[-1][1][-3:] == [8, 9, SP.empty_audio]
    two = [[(7, 0.10), (8, 0.50), (9, 0.0, 8)], [(5, 0.70)]]            # 다른 화자 토큰과 같은 청크: 화자 태그를 다시 붙이고 뒤에
    ch, _ = build_interleaved_events(two, 2.0, SP, delay_frames=0, add_spk_tags=True)
    assert dict(ch)[8] == [SP.spk[1], 5, SP.spk[0], 9, SP.next_audio] and dict(ch)[9] == [SP.next_audio]

# ── SEM 배치
@pytest.mark.parametrize("delay", [2, 3, 4, 6])
def test_sem_right_after_last_token_same_chunk(delay):
    row = make_row(); ev, marks, s = seq_of(row, make_labels(), delay, K=40)
    assert s["n_sem"] == 3 and s["sem_in_flush"] == 0
    ids = s["ids"]
    for p, want in zip(s["sem_pos"], (1003, 1004, 1009)):
        assert ids[p] == SEM and ids[p - 1] == want and not s["is_input"][p - 1]            # 바로 앞 = 단어의 마지막 토큰(오디오 자리 없음 → 같은 청크)
        assert chunk_at(s, p) == int({t: tt for t, tt in row["tokens"]}[want] / 0.08) + delay
    p2 = s["sem_pos"][1]; assert ids[p2 + 1] == 1005                                    # 동률(and/then 같은 시각): and SEM then
    assert s["labels"][p2] == SEM and all(s["labels"][p] == SEM for p in s["sem_pos"])
    got = [t for t, inp in zip(ids, s["is_input"]) if not inp and t not in (SP.next_audio, SEM, TURN)]
    assert got == [t for t, _ in row["tokens"]]                                         # 텍스트 순서 불변

# ── TURN_END 청크·패딩
@pytest.mark.parametrize("delay", [2, 3, 4, 6])
def test_turn_chunk_padding_and_never_in_flush(tmp_path, delay):
    row = make_row(); wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"
    wp.write_text(json.dumps(row) + "\n"); lp.write_text(json.dumps(make_labels()) + "\n")
    ds = SemCommitDataset(str(wp), str(lp), FakeTok(), delays=(2, 3, 4, 6), online=False, turn_end=True)
    assert ds.turn_id == ds.sp_ids["<EOT>"] and ds.turn_id < ds.sem_id                        # 턴 종료 = Phase 2 <EOT>
    it = ds.items[0]; t_last = 2.30; want_max = max(int(t_last / 0.08) + 6, int((t_last + 0.48) / 0.08))
    assert it["K0"] == row["K"] and it["K"] == max(row["K"], want_max + 2) and it["K"] > it["K0"] and abs(it["duration_s"] - it["K"] * 0.08) < 1e-9
    s = ds.sequence(0, delay); k_turn = max(int(t_last / 0.08) + delay, int((t_last + 0.48) / 0.08))
    assert s["n_turn"] == 1 and s["turn_chunk"] == k_turn < it["K"] - 1 and s["turn_in_flush"] == 0 and s["sem_in_flush"] == 0
    p = s["turn_pos"][0]; ids = s["ids"]; first_empty = ids.index(ds.sp.empty_audio)
    assert p < first_empty and chunk_at(s, p) == k_turn and ids[p + 1] == ds.sp.next_audio and s["labels"][p] == ds.turn_id
    assert ids[first_empty:] == [ds.sp.empty_audio, ds.sp.next_audio]                    # flush = 빈 라운드 하나뿐(모든 방출이 실제 오디오 청크 안)
    assert ds.check_targets(0, delay)
    x = ds[0]; assert x["wav"].shape[0] == it["K"] * 1280 and int(x["is_audio"].sum()) == it["K"] and x["K"] == it["K"]

def test_no_turn_end_and_label_flag():
    row = make_row(); ev, _ = build_semcommit_tokens(row["words"], row["tokens"], make_labels(turn_end=False), SEM, TURN)
    assert all(len(x) == 2 for x in ev) and TURN not in [x[0] for x in ev]
    ev, _ = build_semcommit_tokens(row["words"], row["tokens"], make_labels(), SEM, TURN, turn_end=False)
    assert TURN not in [x[0] for x in ev]
    ev, _ = build_semcommit_tokens(row["words"], row["tokens"], make_labels(), SEM, TURN, hangover_s=0.48)
    assert ev[-1][0] == TURN and ev[-1][2] == int((2.30 + 0.48) / 0.08) and ev[-2] == (SEM, 2.30)

# ── B / N 결정 위치
@pytest.mark.parametrize("delay", [2, 4, 6])
def test_B_mark_masks_exactly_decision_position(delay):
    row = make_row(); base = seq_of(row, make_labels(cands=((1, "A"),)), delay, K=40)[2]
    ev, marks, s = seq_of(row, make_labels(cands=((1, "A"), (5, "B"))), delay, K=40)
    assert marks == [(ev.index((1007, 1.90)), "B")] and s["ids"] == base["ids"]
    p = s["pos_of"][marks[0][0]] + 1; assert s["ids"][p - 1] == 1007 and s["decision"] == [(p, "B")]
    diff = [q for q in range(len(s["labels"])) if s["labels"][q] != base["labels"][q]]
    assert diff == [p] and s["labels"][p] == -100 and base["labels"][p] == s["ids"][p] and not s["is_input"][p]
    assert sum(s["pos_weight"]) == 0

def test_B_mark_before_same_chunk_text_masks_that_text_label():
    row = make_row(); ev, marks, s = seq_of(row, make_labels(cands=((2, "B"),)), 2, K=40)       # and|then 같은 시각 → 결정 위치 라벨은 then 의 텍스트
    p = s["decision"][0][0]; assert s["ids"][p - 1] == 1004 and s["ids"][p] == 1005 and s["labels"][p] == -100

@pytest.mark.parametrize("hw", [1.0, 2.5])
def test_N_mark_sets_pos_weight_only(hw):
    row = make_row(); base = seq_of(row, make_labels(cands=()), 3, K=40)[2]
    ev, marks, s = seq_of(row, make_labels(cands=((4, "N"),)), 3, K=40, hardneg=hw)
    p = s["decision"][0][0]; assert s["ids"][p - 1] == 1006 and s["ids"][p] == SP.next_audio
    assert [q for q, w in enumerate(s["pos_weight"]) if w] == [p] and s["pos_weight"][p] == hw and s["labels"] == base["labels"]

def test_label_row_errors_are_rejected(tmp_path):
    row = make_row()
    for bad in (make_labels(cands=((1, "A"), (1, "N"))), make_labels(cands=((99, "A"),)), make_labels(cands=((1, "C"),))):
        with pytest.raises(ValueError): build_semcommit_tokens(row["words"], row["tokens"], bad, SEM, TURN)
    r2 = make_row("ls-2"); r2["words"][1]["end_time"] = 0.90                                  # end_time ≠ 마지막 토큰 시각
    r3 = make_row("ls-3")                                                                     # 라벨 없음 → 건너뜀
    wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"
    wp.write_text("".join(json.dumps(r) + "\n" for r in (make_row(), r2, r3)))
    lp.write_text("".join(json.dumps(make_labels(sid)) + "\n" for sid in ("ls-1", "ls-2")))
    ds = SemCommitDataset(str(wp), str(lp), FakeTok(), online=False)
    assert len(ds) == 1 and ds.stats["skipped_unlabeled"] == 1 and ds.bad == {"word_end_time": 1}
    ds = SemCommitDataset([str(wp)], [str(lp)], FakeTok(), online=False, allow_unlabeled=True, langs=["English"])
    assert len(ds) == 2 and SemCommitDataset(str(wp), str(lp), FakeTok(), online=False, langs=["Korean"]).items == []

# ── 데이터셋 항목·prefix·collate
def test_dataset_item_prefix_and_collate(tmp_path):
    rows = [make_row("ls-1"), make_row("ls-2", WORDS[:3], tail=1.5)]
    wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"
    wp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lp.write_text(json.dumps(make_labels("ls-1")) + "\n" + json.dumps(make_labels("ls-2", cands=((0, "N"), (2, "A")))) + "\n")
    tok = FakeTok(); ds = SemCommitDataset(str(wp), str(lp), tok, delays=(4,), online=False, hardneg_weight=2.0, turn_end=True)
    pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n")["input_ids"] + tok("language English<asr_text>")["input_ids"] + [tok.v["<DELAY_4>"]]
    assert ds.prefix("English", 4) == pre
    xs = [ds[0], ds[1]]
    for x in xs:
        assert x["ids"][: len(pre)].tolist() == pre and (x["labels"][: len(pre)] == -100).all() and x["pos_weight"].dtype == torch.float32
        assert x["ids"].shape == x["labels"].shape == x["pos_weight"].shape == x["chunk_of"].shape and x["delay"] == 4
        assert {"wav", "is_audio", "lang", "name", "id", "n_text", "overflow", "n_flush", "K", "rounds", "n_sem", "n_turn", "n_B", "n_N"} <= set(x)
    b = collate_semcommit(xs); B, L = b["ids"].shape
    assert B == 2 and L == max(len(x["ids"]) for x in xs) and b["pos_weight"].shape == (2, L) and b["pos_weight"].dtype == torch.float32
    for i, x in enumerate(xs):
        n = len(x["ids"]); assert torch.equal(b["pos_weight"][i, :n], x["pos_weight"]) and (b["pos_weight"][i, n:] == 0).all() and (b["labels"][i, n:] == -100).all()
    assert b["n_sem"] == xs[0]["n_sem"] + xs[1]["n_sem"] and b["n_N"] == 2 and b["wav"].shape[0] == 2 and b["K"].tolist() == [xs[0]["K"], xs[1]["K"]]
    assert set(b["pos_weight"][b["pos_weight"] > 0].tolist()) == {2.0} and b["n_turn"] == 2

def test_online_pcm_padding_synthesizes_tail(tmp_path):
    x = (0.2 * np.sin(np.arange(int(2.0 * 16000)) / 7.0) * 32767).astype("<i2"); pcm = tmp_path / "a.pcm"; x.tofile(pcm)
    row = make_row(path="/remote/root/a.pcm"); row["segments"][0]["dur_s"] = 2.0
    wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"; wp.write_text(json.dumps(row) + "\n"); lp.write_text(json.dumps(make_labels()) + "\n")
    ds = SemCommitDataset(str(wp), str(lp), FakeTok(), delays=(6,), online=True, path_map={"/remote/root/": str(tmp_path) + "/"})
    it = ds.items[0]; w = ds[0]["wav"].numpy(); end = int(round((0.3 + 2.0) * 16000))
    assert w.shape[0] == it["K"] * 1280 > int(round(row["duration_s"] * 16000)) and np.abs(w[end:]).max() > 0     # 늘린 끝 = 합성 배경(디지털 0 아님)

def test_audio_failure_falls_back_to_neighbor(tmp_path):
    x = np.zeros(int(2.0 * 16000), "<i2"); x.tofile(tmp_path / "ok.pcm")
    rows = [make_row("ls-1", path=str(tmp_path / "missing.pcm")), make_row("ls-2", path=str(tmp_path / "ok.pcm"))]
    wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"; wp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lp.write_text("".join(json.dumps(make_labels(r["id"])) + "\n" for r in rows))
    ds = SemCommitDataset(str(wp), str(lp), FakeTok(), delays=(2,), online=True, seed=0)
    i_bad = [i for i, it in enumerate(ds.items) if it["id"] == "ls-1"][0]
    if i_bad == len(ds) - 1: ds.items = ds.items[::-1]; i_bad = 0                            # 이웃(i+1)이 정상 항목이 되게
    x = ds[i_bad]; it2 = next(it for it in ds.items if it["id"] == "ls-2")
    assert x["id"] == "ls-2" and x["K"] == it2["K"] and x["wav"].shape[0] == int(round(it2["duration_s"] * 16000))   # 안 늘린 항목은 원래 길이·manifest K(학습·평가 공통 규약)

def test_max_per_chunk_padding_uses_simulated_last_chunk():
    toks = [(i, 1.0) for i in range(12)] + [(99, 0.0, 0)]                               # 동일 시각 12 토큰 + 명시 이벤트, M=2 → 이월 6 청크
    for d in (2, 6):
        k = last_emit_chunk(toks, d, SP, M=2); assert k == int(1.0 / 0.08) + d + 5 and last_emit_chunk(toks, d, SP, M=0) == int(1.0 / 0.08) + d
        ch, _ = build_interleaved_events([toks], (k + 2 - 0.5) * 0.08, SP, delay_frames=d, max_per_chunk=2, add_spk_tags=False)
        assert dict(ch)[k][-2:] == [99, SP.next_audio] and all(c < k + 2 for c, _ in ch)          # flush 항목 없음

def test_add_semcommit_specials_appends_after_phase2():
    tok = FakeTok(base=151705); ids = add_semcommit_specials(tok)
    assert ids["<NEXT_AUDIO>"] == 151705 and ids["<EOT>"] == 151722 and ids["<SEM_END>"] == 151723 and "<TURN_END>" not in ids and "<TURN_END>" not in tok.v
    assert list(ids) == SPECIAL_TOKENS + PHASE2_SPECIALS + SEM_SPECIALS

def test_round_robin_loader_alternates_languages_with_pos_weight(tmp_path):
    rows = [make_row(f"ls-{i}") for i in range(4)] + [make_row(f"ks-{i}", WORDS[:3], lang="Korean") for i in range(6)]
    wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"; wp.write_text("".join(json.dumps(r) + "\n" for r in rows))
    lp.write_text("".join(json.dumps(make_labels(r["id"], cands=((0, "N"), (1, "A")), lang=r["lang"])) + "\n" for r in rows))
    sets = {k: SemCommitDataset(str(wp), str(lp), FakeTok(), langs=[lang], online=False) for k, lang in (("en", "English"), ("ko", "Korean"))}
    rr = semcommit_round_robin(sets, lambda m: 2 if m == "en" else 3, num_workers=0)
    got = [(b["lang"][0], tuple(b["ids"].shape), tuple(b["pos_weight"].shape), int((b["pos_weight"] > 0).sum()), b["n_N"]) for b in rr]
    assert len(rr) == len(got) == 4 and [g[0] for g in got] == ["English", "Korean"] * 2
    assert all(g[1] == g[2] and g[3] == g[4] == g[1][0] for g in got)                      # 스트림마다 N 1 개 → pos_weight>0 도 배치 크기만큼

# ── 마지막 단어 B/N + TURN: δ=6 이면 결정 위치 = TURN 타깃 → TURN 라벨·가중 유지, '@TURN' 으로 센다(리뷰 finding 1)
@pytest.mark.parametrize("g", ["B", "N"])
def test_last_word_B_or_N_keeps_turn_target(g):
    row = make_row(); last = len(row["words"]) - 1
    for d in (2, 6):
        ev, marks, s = seq_of(row, make_labels(cands=((1, "A"), (last, g))), d, K=60, hardneg=2.5)
        tp = s["turn_pos"][0]; p = s["pos_of"][marks[0][0]] + 1
        assert s["ids"][tp] == TURN and s["labels"][tp] == TURN and s["pos_weight"][tp] == 0            # TURN 타깃·turn_weight(덮어쓰기 없음) 유지
        if d == 6:                                                                                     # int(2.30/.08)+6 = 34 = int((2.30+.48)/.08) → 같은 청크 바로 뒤
            assert p == tp and s["decision"] == [(tp, f"{g}@TURN")] and s["n_decision_on_event"] == 1 and s["n_B"] == s["n_N"] == 0
            assert sum(s["pos_weight"]) == 0 and all(s["labels"][q] != -100 for q in range(len(s["ids"])) if not s["is_input"][q])
        else:
            assert p != tp and s["ids"][p] == SP.next_audio and s["decision"] == [(p, g)] and s["n_decision_on_event"] == 0
            assert (s["labels"][p] == -100) if g == "B" else (s["pos_weight"][p] == 2.5)

def test_dataset_counts_decision_on_turn_and_checks(tmp_path):
    row = make_row(); last = len(row["words"]) - 1; wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"
    wp.write_text(json.dumps(row) + "\n"); lp.write_text(json.dumps(make_labels(cands=((1, "A"), (last, "B")))) + "\n")
    ds = SemCommitDataset(str(wp), str(lp), FakeTok(), delays=(2, 3, 4, 6), online=False, turn_end=True)
    assert ds.stats["decision_on_event"] == 1 and ds.stats["decision_on_event_items"] == 1                  # δ=6 만
    assert all(ds.target_problems(0, d) == [] for d in ds.delays)
    ds.delays = (6,); x = ds[0]; assert x["n_decision_on_event"] == 1 and x["n_B"] == 0 and collate_semcommit([x])["n_decision_on_event"] == 1
    s = ds.sequence(0, 6); tp = s["turn_pos"][0]; s["labels"][tp] = -100; s["decision"] = [(tp, "B")]      # 고치기 전 동작을 흉내 → 관문이 잡는다
    ds.sequence = lambda i, d: s; pr = ds.target_problems(0, 6)
    assert f"event_target_altered@{tp}" in pr and f"decision@{tp}:B" in pr and not ds.check_targets(0, 6)

# ── M>0: SEM 이 청크 한도에 세어져 단어와 다른 청크로 밀린다 → check_targets 가 잡는다(finding 2)
def test_M_cap_detaches_sem_and_check_targets_catches_it(tmp_path):
    row = make_row(); wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"
    wp.write_text(json.dumps(row) + "\n"); lp.write_text(json.dumps(make_labels(cands=((1, "A"),))) + "\n")
    ok = SemCommitDataset(str(wp), str(lp), FakeTok(), delays=(2,), online=False)
    assert ok.target_problems(0, 2) == [] and ok.check_targets(0, 2)
    ds = SemCommitDataset(str(wp), str(lp), FakeTok(), delays=(2,), online=False, max_per_chunk=1)
    s = ds.sequence(0, 2); p = s["sem_pos"][0]; assert s["ids"][p - 1] != 1003                              # world 의 두 번째 토큰과 SEM 사이에 NEXT·오디오
    assert "sem_detached@3" in ds.target_problems(0, 2) and not ds.check_targets(0, 2)

# ── words ↔ labels 짝 통계(finding 5)
def test_label_pairing_stats_and_lang_mismatch(tmp_path):
    rows = [make_row("ls-1"), make_row("ks-1", WORDS[:3], lang="Korean"), make_row("ks-2", WORDS[:3], lang="Korean")]
    labs = [make_labels("ls-1"), make_labels("ks-1", cands=((1, "A"),)), make_labels("zz-9")]              # ks-1 label 의 lang = English(불일치), zz-9 = words 없음
    wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"
    wp.write_text("".join(json.dumps(r) + "\n" for r in rows)); lp.write_text("".join(json.dumps(r) + "\n" for r in labs))
    ko = SemCommitDataset(str(wp), str(lp), FakeTok(), online=False, langs=["Korean"])
    assert len(ko) == 0 and ko.stats["words_rows"] == 2 and ko.stats["labeled"] == 1 and ko.stats["skipped_unlabeled"] == 1
    assert ko.bad == {"lang_mismatch": 1} and ko.stats["labels_without_words"] == 1
    en = SemCommitDataset(str(wp), str(lp), FakeTok(), online=False, langs=["English"])
    assert len(en) == 1 and en.stats["words_rows"] == en.stats["labeled"] == 1 and not en.bad

# ── tokenizer 내용 관문: 빌더와 같은 split_words 로 다시 나눠 단어가 같은가(finding 4)
PIECES = {1001: "hello", 1002: "Ġwor", 1003: "ld", 1004: "Ġand", 1005: "Ġthen", 1006: "Ġum", 1007: "Ġso", 1008: "Ġdo", 1009: "ne"}
class PieceTok(FakeTok):
    def __init__(self, pieces=PIECES): super().__init__(); self.pieces = dict(pieces)
    def convert_ids_to_tokens(self, ids): return [self.pieces.get(int(i)) for i in ids]

def test_tokenizer_content_check(tmp_path):
    wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"; wp.write_text(json.dumps(make_row()) + "\n"); lp.write_text(json.dumps(make_labels()) + "\n")
    ds = SemCommitDataset(str(wp), str(lp), PieceTok(), online=False); assert ds.stats["tok_check"] == 1 and "tok_check_skipped" not in ds.stats and len(ds) == 1
    for bad in ({**PIECES, 1004: "Ġthen", 1005: "Ġand"}, {**PIECES, 1003: "Ġld"}, {**PIECES, 1006: None}):     # 다른 단어 / 다른 경계 / 모르는 id
        with pytest.raises(ValueError, match="tokenizer_mismatch"): SemCommitDataset(str(wp), str(lp), PieceTok(bad), online=False)
    assert SemCommitDataset(str(wp), str(lp), FakeTok(), online=False).stats["tok_check_skipped"] == 1

# ── 실행 지문: 설정·파일·어휘 → 재개 대조(finding 4)
class VocabTok(FakeTok):
    def get_vocab(self): return dict(self.v)

def test_fingerprint_and_checkpoint_resume_diff(tmp_path):
    from vapasr.hf.configuration_vapasr import VapAsrConfig
    wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"; wp.write_text(json.dumps(make_row()) + "\n"); lp.write_text(json.dumps(make_labels()) + "\n")
    tok = VocabTok(); add_semcommit_specials(tok)
    fpx = lambda tk=tok, **kw: semcommit_fingerprint(str(wp), str(lp), tk, **{**dict(turn_end=True, hangover_s=0.48, tail_margin=2, pad_tail=True, M=0, delays=(2, 3, 4, 6)), **kw})
    fp = fpx(); assert fp["protocol"] == FP_PROTOCOL and fp["delays"] == [2, 3, 4, 6] and len(fp["words_sha256"]) == len(fp["labels_sha256"]) == 1 and fp["vocab_sha256"]
    assert fingerprint_diff(json.loads(json.dumps(fp)), fp) == [] and fingerprint_diff(None, fp) == ["<fingerprint 없음>"]
    keys = lambda d: [x.split(":")[0] for x in d]
    assert keys(fingerprint_diff(fp, fpx(hangover_s=0.40))) == ["hangover_s"] and keys(fingerprint_diff(fp, fpx(turn_end=False, delays=(2,)))) == ["delays", "turn_end"]
    tok2 = VocabTok(); add_semcommit_specials(tok2); tok2.v["<extra>"] = 7; assert keys(fingerprint_diff(fp, fpx(tok2))) == ["vocab_sha256"]
    assert vocab_sha256(FakeTok()) is None
    ck = tmp_path / "checkpoint-5"; c = VapAsrConfig(); c.semcommit = fp; c.save_pretrained(str(ck))    # config.json 왕복 → 재개 대조
    assert VapAsrConfig.from_pretrained(str(ck)).semcommit == fp and checkpoint_fingerprint_diff(str(ck), fp) == []
    lp.write_text(json.dumps(make_labels(cands=((1, "A"),))) + "\n")                                     # 같은 경로, 다른 labels 내용
    assert keys(checkpoint_fingerprint_diff(str(ck), fpx())) == ["labels_sha256"] and checkpoint_fingerprint_diff(str(tmp_path / "none"), fp) == ["<fingerprint 없음>"]

def test_random_rows_pass_target_problems(tmp_path):
    """무작위 단어 시각·등급(마지막 단어 A/B/N 포함) → 모든 δ 에서 불변식 통과, B/N 이 TURN 에 떨어진 수 = 데이터셋 통계."""
    rng = random.Random(1); rows, labs = [], []
    for n in range(60):
        t, ws = rng.uniform(0.1, 1.0), []
        for i in range(rng.randint(1, 12)):
            t += rng.choice([0.0, 0.04, 0.08, round(rng.uniform(0.05, 0.9), 2)]); tt = [(1000 + i * 3 + k, round(t, 2)) for k in range(rng.randint(1, 3))]; ws.append((f"w{i}", tt))
        r = make_row(f"r-{n}", ws, tail=rng.choice([0.0, 0.1, 1.5])); rows.append(r)
        labs.append(make_labels(f"r-{n}", cands=[(i, rng.choice("ABN")) for i in range(len(ws)) if rng.random() < 0.6], turn_end=rng.random() < 0.8))
    wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"
    wp.write_text("".join(json.dumps(r) + "\n" for r in rows)); lp.write_text("".join(json.dumps(r) + "\n" for r in labs))
    ds = SemCommitDataset(str(wp), str(lp), FakeTok(), online=False, hardneg_weight=2.0, turn_end=True)
    assert len(ds) == 60 and not ds.bad
    got = 0
    for i in range(len(ds)):
        for d in ds.delays:
            assert ds.target_problems(i, d) == [], (ds.items[i]["id"], d, ds.target_problems(i, d)); got += ds.sequence(i, d)["n_decision_on_event"]
    assert got == ds.stats["decision_on_event"] > 0


def test_default_is_sem_only_no_turn_token(tmp_path):
    """Phase 1 기본 = <SEM_END> 만: 턴 이벤트·<EOT> 라벨 없음, 마지막 B/N 결정 위치는 평소처럼 가림/가중(TURN 에 떨어지지 않음), 패딩은 pad_tail 로만."""
    row = make_row(); last = len(row["words"]) - 1; wp, lp = tmp_path / "w.jsonl", tmp_path / "l.jsonl"
    wp.write_text(json.dumps(row) + "\n"); lp.write_text(json.dumps(make_labels(cands=((1, "A"), (last, "B")))) + "\n")
    ds = SemCommitDataset(str(wp), str(lp), FakeTok(), delays=(2, 3, 4, 6), online=False)
    assert not ds.turn_end and ds.stats["turn"] == 0 and ds.stats["decision_on_event"] == 0 and ds.stats["sem"] == 1
    for d in ds.delays:
        s = ds.sequence(0, d); assert s["n_turn"] == 0 and ds.turn_id not in s["ids"] and s["n_B"] == 1 and ds.target_problems(0, d) == []
    it = ds.items[0]; assert it["K"] == max(row["K"], int(2.30 / 0.08) + 6 + 2)            # 마지막 방출(δ=6) + tail_margin

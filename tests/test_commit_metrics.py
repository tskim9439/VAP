"""Semantic commit 평가(vapasr/hf/commit_metrics.py, experiments/semcommit_eval.py) — 최적 매칭·비대칭 창·지연 산술·텍스트 위치 PCR 범주·
이벤트 추출·참조 직렬화(oracle) 상한·가짜 모델 디코드(bias/threshold/guard/trace/guard_blocked)·CLI 종단(이어하기 지문·words 행 지문·K' 검사·
학습 설정 복원·δ 검사·오디오 선읽기 상한). 합성 데이터, CPU 수 초."""
import importlib.util, json, random, sys, tempfile, types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vapasr.hf import commit_metrics as cm
from vapasr.hf.commit_metrics import (SEM_END_ID as SEM, TURN_END_ID as TURN, aggregate, align_pairs, emit_time, events_from_emits, finalize, map_events, match_optimal,
                                      oracle_hyp, ref_chunk, score_stream, split_hyp, text_position_metrics, timing_metrics, turn_metrics, train_pad_K, eval_audio_duration, turn_ref_chunk,
                                      eval_audio_len, edit_counts, turn_flag)

ROOT = Path(__file__).resolve().parents[1]

def _cli():
    spec = importlib.util.spec_from_file_location("semcommit_eval", ROOT / "experiments/semcommit_eval.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def mk_row(sid, lang, spec, duration_s, set_="toy"):
    """spec: [(단어, end_time, tags, n_tok)] → words.jsonl 행(토큰 id = 1000+10·i+j, 토큰 시각 = 단어 끝)."""
    toks, words = [], []
    for i, (w, t, tags, n) in enumerate(spec):
        a = len(toks); toks += [[1000 + 10 * i + j, t] for j in range(n)]
        words.append(dict(i=i, text=w, a=a, b=len(toks), end_time=t, seg=0, tags=list(tags)))
    return dict(id=sid, set=set_, lang=lang, K=int(round(duration_s * 12.5)), duration_s=duration_s, text=" ".join(w for w, *_ in spec), tokens=toks,
                segments=[dict(path="/nonexistent.flac", offset_s=0.3, silence_before_s=0.3, dur_s=duration_s - 0.6, utt_id=sid, raw_text="")], words=words)

def cand(i, g, why="", B=None, C=None):
    return dict(after_word=i, grade=g, why=why, stageA=True, B=B or {}, B_margin={}, C=C or {"relation": "STABLE", "type": ""}, future_unobserved=False, resolved_by_judge=False)

# ───────────────────────────── 매칭 ─────────────────────────────
def test_optimal_beats_greedy_counterexample():
    from vapasr.hf.lane_metrics import match_events
    hyp, ref = [1.0, 1.25], [1.15, 0.8]
    assert match_events(hyp, ref, 0.2)[0] == 1                                       # 탐욕: 1.0 이 가까운 1.15 를 가져가 1.25 가 짝을 잃는다
    for use_scipy in (True, False):
        m = match_optimal(hyp, ref, early=0.2, late=0.2, use_scipy=use_scipy)
        assert m["hits"] == 2 and m["pairs"] == [(0, 1), (1, 0)] and m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0

def test_asymmetric_window_chunks():
    r = [10]
    assert match_optimal([8], r, early=1, late=5)["hits"] == 0                        # 2 청크 이른 commit 은 거절
    assert match_optimal([9], r, early=1, late=0)["hits"] == 1
    assert match_optimal([11], r, early=1, late=0)["hits"] == 0 and match_optimal([11], r, early=1, late=2)["hits"] == 1
    assert match_optimal([12], r, early=1, late=2)["hits"] == 1 and match_optimal([13], r, early=1, late=2)["hits"] == 0
    m = match_optimal([9, 10, 30], [10, 20], early=1, late=2); assert m["hits"] == 1 and m["pairs"] == [(1, 0)]     # 동률이면 |Δ| 최소 쌍
    assert m["precision"] == 1 / 3 and m["recall"] == 0.5 and abs(m["f1"] - 0.4) < 1e-12
    e = match_optimal([], [1, 2], 1, 2); assert e["hits"] == 0 and e["precision"] is None and e["recall"] == 0.0 and e["f1"] is None

def test_dp_equals_linear_sum_assignment_random():
    pytest.importorskip("scipy")
    rng = random.Random(0)
    for _ in range(300):
        hyp = [rng.randint(0, 40) for _ in range(rng.randint(0, 9))]; ref = [rng.randint(0, 40) for _ in range(rng.randint(0, 9))]
        early, late = rng.randint(0, 2), rng.choice([0, 2, 5])
        a, b = match_optimal(hyp, ref, early, late, use_scipy=True), match_optimal(hyp, ref, early, late, use_scipy=False)
        cost = lambda m: sum(abs(hyp[i] - ref[j]) for i, j in m["pairs"])
        assert a["hits"] == b["hits"] and cost(a) == cost(b)
        assert all(ref[j] - early <= hyp[i] <= ref[j] + late for i, j in a["pairs"] + b["pairs"])
        assert len({i for i, _ in a["pairs"]}) == a["hits"] == len({j for _, j in a["pairs"]})

# ───────────────────────────── 시각 ─────────────────────────────
def test_events_from_emits():
    em = [(0, 5), (3, SEM), (3, 7), (9, SEM), (9, TURN), (12, SEM)]
    assert events_from_emits(em, SEM) == [3, 9, 12] and events_from_emits(em, SEM, K=10) == [3, 9, 10] and events_from_emits(em, TURN) == [9]
    assert events_from_emits(em, SEM, K=10, times=True) == [0.32, 0.8, 0.8]            # (k+1)·0.08, flush(k ≥ K) 는 K·0.08
    assert events_from_emits(em, SEM, times=True) == [0.32, 0.8, 1.04] and events_from_emits([], SEM) == []

def test_ref_chunk_matches_build_interleaved():
    from vapasr.data.interleave import build_interleaved, Specials
    sp = Specials(next_audio=1, empty_audio=2, spk=(3, 4)); K = 40
    for t in (0.0, 0.08, 0.16, 0.24, 0.4, 0.72, 1.0, 2.96, 3.1):
        for d in (0, 2, 4, 6):
            chunks, _ = build_interleaved([[(77, t)]], (K - 0.5) * 0.08, sp, delay_frames=d, add_spk_tags=False)
            got = next(k for k, e in chunks if 77 in e)
            assert ref_chunk(t, d, K) == got, (t, d)                                    # ε 없는 int(t/0.08) — 0.24/0.08 = 2.999… → 2 까지 학습과 같다

def test_latency_math():
    row = mk_row("s", "English", [("go", 1.00, [], 1), ("now", 2.00, [], 1)], 3.0); K = row["K"]; cands = [cand(0, "A"), cand(1, "A")]
    assert ref_chunk(1.0, 4) == 16 and ref_chunk(2.0, 4) == 29 and emit_time(16) == 1.36
    t = timing_metrics([16, 31], K, row["words"], cands, delta=4, early=1, lates=(0, 2, 5))
    assert t["ideal_lat"] == [0.36, 0.4]                                                # (int(t/0.08)+δ+1)·0.08 − t
    assert t["late0"]["hits"] == 1 and t["late0"]["lat"] == [0.36] and t["late0"]["lag"] == [0]
    assert t["late2"]["hits"] == 2 and t["late2"]["lat"] == [0.36, 0.56] and t["late2"]["lag"] == [0, 2]
    f = finalize(dict(streams=1, audio_s=3.0, events=dict(sem=2, turn=0, sem_in_flush=0), timing=t, text=text_position_metrics(row["words"], cands, [], []),
                      asr={"wer": dict(substitutions=0, deletions=0, insertions=0, n_ref=2, n_hyp=2, errors=0)}))
    assert f["timing"]["late2"]["latency_s"] == dict(n=2, mean=0.46, p50=0.46, p90=0.54) and f["timing"]["late0"]["miss"] == 1 and f["timing"]["late0"]["recall"] == 0.5
    tf = timing_metrics([40], 28, row["words"], cands, delta=4)                          # flush: ref_k·hyp_k 모두 K 로 자르고 시각은 K·0.08
    assert tf["late0"]["lag"] == [0] and tf["late0"]["lat"] == [round(28 * 0.08 - 2.0, 4)] == [0.24]

def test_b_grade_not_counted_as_false_in_timing():
    row = mk_row("s", "English", [("a", 0.5, [], 1), ("b", 1.0, [], 1), ("c", 1.5, [], 1)], 2.0); cands = [cand(0, "A"), cand(1, "B"), cand(2, "N", "filler")]
    t = timing_metrics([ref_chunk(0.5, 2), ref_chunk(1.0, 2), ref_chunk(1.5, 2)], row["K"], row["words"], cands, delta=2)
    w = finalize(dict(streams=1, audio_s=2.0, events=dict(sem=3, turn=0, sem_in_flush=0), timing=t, text=text_position_metrics(row["words"], cands, [], []),
                      asr={}))["timing"]["late0"]
    assert (w["hits"], w["b_hits"], w["false"], w["precision"], w["precision_excl_B"]) == (1, 1, 1, 1 / 3, 0.5)

# ───────────────────────────── 텍스트 위치 ─────────────────────────────
def _fake_tok(pieces):
    """조각 목록 → (ids, is_text, word_start, decode). 공백으로 시작하는 조각이 단어 시작."""
    vocab = {p: 100 + i for i, p in enumerate(dict.fromkeys(pieces))}; inv = {v: p for p, v in vocab.items()}
    return vocab, (lambda t: t in inv), (lambda t: inv[t].startswith(" ")), (lambda ids: "".join(inv[i] for i in ids))

def test_split_hyp_words_events_mid_word():
    vocab, is_text, ws, dec = _fake_tok([" he", "llo", " world"])
    em = [(0, SEM), (1, vocab[" he"]), (1, SEM), (2, vocab["llo"]), (2, 151705), (3, SEM), (3, vocab[" world"]), (5, TURN)]
    h = split_hyp(em, is_text=is_text, word_start=ws, decode=dec)
    assert h["words"] == ["hello", "world"]
    assert [(e["id"], e["k"], e["after"], e["mid_word"]) for e in h["events"]] == [(SEM, 0, -1, False), (SEM, 1, 0, True), (SEM, 3, 0, False), (TURN, 5, 1, False)]

def test_text_position_pcr_categories():
    spec = [("uh", 0.3, ["filler"], 1), ("i", 0.5, [], 1), ("went", 0.8, [], 1), ("home", 1.1, ["punct_final"], 1), ("and", 1.5, [], 1), ("then", 1.8, [], 1),
            ("i", 2.0, ["rep"], 1), ("i", 2.2, [], 1), ("slept", 2.6, ["punct_final"], 2)]
    row = mk_row("s", "English", spec, 3.2)
    cands = [cand(0, "N", "filler"), cand(2, "N", "stageC", C={"relation": "REVISION", "type": "correction"}), cand(3, "A"), cand(5, "B"),
             cand(7, "N", "repetition"), cand(8, "A"), cand(4, "N", "judges", B={"qwen": "WAIT", "exaone": "WAIT"})]
    vocab, is_text, ws, dec = _fake_tok([" uh", " i", " went", " home", " and", " then", " sl", "ept"])
    v = vocab
    em = [(0, SEM), (1, v[" uh"]), (1, SEM), (2, v[" i"]), (2, SEM), (3, v[" went"]), (3, SEM), (4, v[" home"]), (4, SEM), (4, SEM), (5, v[" and"]), (6, v[" then"]), (6, SEM),
          (7, v[" i"]), (7, SEM), (8, v[" i"]), (8, SEM), (9, v[" sl"]), (9, SEM), (9, v["ept"]), (10, SEM)]
    h = split_hyp(em, is_text=is_text, word_start=ws, decode=dec)
    assert h["words"] == [w for w, *_ in spec]
    x = text_position_metrics(row["words"], cands, h["words"], [e for e in h["events"] if e["id"] == SEM])
    assert (x["n_hyp"], x["correct"], x["ambiguous"], x["duplicate"], x["premature"]) == (11, 2, 1, 1, 7)
    assert {c: n for c, n in x["cat"].items() if n} == dict(no_word=1, filler=1, other_inside=1, revision=1, rep=2, mid_word=1)
    assert (x["n_A"], x["n_B"], x["n_N"], x["N_hit"]) == (2, 1, 4, 3) and x["N_cat"] == dict(filler=1, revision=1, rep=1, all_wait=1) and x["N_hit_cat"] == dict(filler=1, revision=1, rep=1)
    f = finalize(dict(streams=1, audio_s=3.2, events=dict(sem=11, turn=0, sem_in_flush=0), timing=dict(ideal_lat=[]), text=x, asr={}))["text"]
    assert f["precision"] == 2 / 11 and f["recall"] == 1.0 and f["pcr"] == 7 / 11 and f["pcr_excl_B"] == 0.7 and f["hardneg_commit_rate"] == 0.75
    assert f["hardneg_by_category"]["all_wait"] == dict(n=1, committed=0, rate=0.0) and f["B_commit_rate"] == 1.0

def test_deletion_attribution_by_emission_chunk():
    row = mk_row("s", "English", [("a", 0.5, [], 1), ("b", 1.0, [], 1), ("c", 1.5, [], 1), ("d", 2.0, [], 1)], 2.6); cands = [cand(2, "A"), cand(3, "A")]
    rk = [ref_chunk(w["end_time"], 2) for w in row["words"]]; assert rk == [8, 14, 20, 27]
    hw = ["a", "b", "d"]                                                                # ASR 가 c 를 빠뜨림
    late = dict(id=SEM, k=21, after=1, mid_word=False); early = dict(id=SEM, k=15, after=1, mid_word=False)
    assert map_events([w["text"] for w in row["words"]], hw, [late, early], rk) == [2, 1]   # c 의 ref_k(20) 이후 방출이면 c 뒤로
    x = text_position_metrics(row["words"], cands, hw, [late], rk); assert x["correct"] == 1 and x["premature"] == 0
    x = text_position_metrics(row["words"], cands, hw, [early], rk); assert x["correct"] == 0 and x["cat"]["other_inside"] == 1
    ins = map_events(["a", "b"], ["a", "x", "b"], [dict(k=0, after=1)]); assert ins == [0]  # 삽입 단어 뒤 = 직전 참조 단어 뒤
    assert map_events(["Hello,", "world"], ["hello", "World."], [dict(k=0, after=0)]) == [0]   # 대소문자·구두점 무시

def _cost_matches(r, h, al):
    return sum(1 for i, j in al if i is None or j is None or r[i] != h[j]), sum(1 for i, j in al if i is not None and j is not None and r[i] == h[j])

def _best_cost_matches(r, h):
    """독립 기준: (편집 수 최소, 그중 일치 최대) 튜플 DP."""
    n, m = len(r), len(h); D = [[(j, 0) for j in range(m + 1)]] + [[(i, 0)] + [None] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            c, mm = D[i - 1][j - 1]; eq = int(r[i - 1] == h[j - 1])
            D[i][j] = min(((c + 1 - eq, mm + eq), (D[i - 1][j][0] + 1, D[i - 1][j][1]), (D[i][j - 1][0] + 1, D[i][j - 1][1])), key=lambda x: (x[0], -x[1]))
    return D[n][m]

def test_align_min_edits_then_max_matches():
    r, h = "e c c d c b e a d a".split(), "a e c b b d".split()                       # 리뷰 반례: 옛 DP(대각 우선) 3 일치, rapidfuzz 4 일치 — 둘 다 편집 7
    assert _cost_matches(r, h, align_pairs(r, h)) == (7, 4)
    assert align_pairs(["a", "b"], ["b", "c"]) == [(0, None), (1, 0), (None, 1)]       # 치환 2 대신 삭제·일치·삽입(같은 편집 2, 일치 1)
    rng = random.Random(1)
    for _ in range(400):
        r = [rng.choice("abcd") for _ in range(rng.randint(0, 9))]; h = [rng.choice("abcd") for _ in range(rng.randint(0, 9))]; al = align_pairs(r, h)
        assert [i for i, _ in al if i is not None] == list(range(len(r))) and [j for _, j in al if j is not None] == list(range(len(h)))
        assert _cost_matches(r, h, al) == _best_cost_matches(r, h)
        assert edit_counts(r, h, use_rapidfuzz=False)["errors"] == _best_cost_matches(r, h)[0]

def _asr_error_cases():
    """(ref, hyp, events, ref_k, 기대 참조 위치): 치환·삭제·삽입이 섞인 작은 ASR 오류."""
    ref = "the cat sat on the mat".split(); rk = [3, 6, 9, 12, 15, 18]
    return [(["a", "b"], ["b", "c"], [dict(k=5, after=0), dict(k=5, after=1)], None, [1, 1]),               # hyp b 는 ref b 와 일치(치환 a→b 아님)
            (ref, "the bat sat the mat".split(), [dict(k=9, after=2), dict(k=12, after=2), dict(k=6, after=1)], rk, [2, 3, 1]),   # on 삭제: k ≥ ref_k(on) 이면 on 뒤
            (ref, "the cat cat sat on the mat".split(), [dict(k=6, after=2), dict(k=9, after=3)], rk, [1, 2]),  # 삽입(cat 반복) 뒤 = 직전 참조 단어 뒤
            ("e c c d c b e a d a".split(), "a e c b b d".split(), [dict(k=0, after=j) for j in range(6)], None, None)]

@pytest.mark.parametrize("backend", ["absent", "stub", "real"])
def test_map_events_independent_of_rapidfuzz(monkeypatch, backend):
    """이벤트 사상은 rapidfuzz 유무와 무관(정렬기 하나): 없을 때·쓰면 터지는 가짜가 있을 때·진짜(rack4) 모두 같은 결과."""
    if backend == "absent":
        monkeypatch.setitem(sys.modules, "rapidfuzz", None); monkeypatch.setitem(sys.modules, "rapidfuzz.distance", None)
    elif backend == "stub":
        def boom(*a, **k): raise AssertionError("정렬에 rapidfuzz 를 쓰면 안 된다")
        rf = types.ModuleType("rapidfuzz"); dist = types.ModuleType("rapidfuzz.distance"); dist.Levenshtein = SimpleNamespace(opcodes=boom, editops=boom); rf.distance = dist
        monkeypatch.setitem(sys.modules, "rapidfuzz", rf); monkeypatch.setitem(sys.modules, "rapidfuzz.distance", dist)
    else: pytest.importorskip("rapidfuzz")
    got = []
    for ref, hyp, evs, rk, want in _asr_error_cases():
        p = map_events(ref, hyp, evs, rk); got.append(p)
        if want is not None: assert p == want, (ref, hyp, p)
    assert got[-1] == [-1, 0, 4, 5, 7, 8]                                               # 반례: a 삽입(−1), e=e(0), c=ref4, b=ref5, b↔a 치환(7), d=ref8 — 일치 4 정렬 위의 사상
    words = [dict(i=i, text=w, end_time=0.24 * (i + 1), tags=[]) for i, w in enumerate("the cat sat on the mat".split())]
    x = text_position_metrics(words, [cand(2, "N", "other"), cand(3, "A"), cand(5, "A")], "the bat sat the mat".split(),
                              [dict(id=SEM, k=12, after=2, mid_word=False), dict(id=SEM, k=20, after=4, mid_word=False)], [3, 6, 9, 12, 15, 18])
    assert (x["correct"], x["premature"], x["N_hit"]) == (2, 0, 0)

@pytest.mark.parametrize("use_rf", [False, True])
def test_edit_counts_backends_same_errors(use_rf):
    if use_rf: pytest.importorskip("rapidfuzz")
    rng = random.Random(2)
    for _ in range(200):
        r = [rng.choice("abcde") for _ in range(rng.randint(0, 12))]; h = [rng.choice("abcde") for _ in range(rng.randint(0, 12))]
        c = edit_counts(r, h, use_rapidfuzz=use_rf); assert c["errors"] == _best_cost_matches(r, h)[0] and (c["n_ref"], c["n_hyp"]) == (len(r), len(h))
    if use_rf:                                                                           # rack4: S/D/I 도 single_turn_eval.score_pair 와 같다
        from vapasr.data.single_turn_eval import score_pair
        w = score_pair("the cat sat on the mat", "the bat sat the mat mat", "English")["wer"]; c = cm.asr_counts("the cat sat on the mat", "the bat sat the mat mat", "English")["wer"]
        assert all(c[k] == w[k] for k in ("substitutions", "deletions", "insertions", "errors", "n_ref"))

def test_textnorm_drops_event_tags():
    from vapasr.data.textnorm import score_en, score_ko
    assert score_en("hello <SEM_END> world <TURN_END>") == "hello world" and score_ko("안녕 <SEM_END>하세요", False) == "안녕하세요"
    assert score_en("hel<SEM_END>lo") == "hel lo"                                        # 단어 중간 태그는 공백이 된다 → 채점 전에 이벤트 id 를 뺀다
    a = cm.asr_counts("the cat sat", "the cat sad", "English")["wer"]; assert (a["substitutions"], a["errors"], a["n_ref"]) == (1, 1, 3)
    k = cm.asr_counts("오늘 날씨", "오늘날씨", "Korean"); assert k["cer_nospace"]["errors"] == 0 and k["cer_space"]["deletions"] == 1

# ───────────────────────────── TURN_END ─────────────────────────────
def test_turn_ref_and_padding():
    assert turn_ref_chunk(2.0, 4) == max(int(2.0 / 0.08) + 4, int(2.48 / 0.08)) == 31 and turn_ref_chunk(2.0, 8) == 33
    assert train_pad_K(29, 2.0, (2, 3, 4, 6), True) == 31 + 2 and train_pad_K(29, 2.0, (2, 4), False) == 25 + 4 + 2 and train_pad_K(100, 2.0, (6,), True) == 100
    assert all(turn_ref_chunk(2.0, d) < train_pad_K(29, 2.0, (2, 3, 4, 6), True) for d in (2, 3, 4, 6))       # TURN 은 flush 가 아닌 실제 청크
    w = mk_row("s", "English", [("a", 1.0, [], 1), ("b", 2.0, [], 1)], 2.3); assert eval_audio_duration(w, (2, 3, 4, 6), True) == round(33 * 0.08, 6)
    w2 = dict(w, K=200, duration_s=16.0); assert eval_audio_duration(w2, (2, 3, 4, 6), True) == 16.0      # 이미 길면 원래 길이
    row = mk_row("s", "English", [("a", 1.0, [], 1), ("b", 2.0, [], 1)], 2.3)
    t = turn_metrics([10, 31, 40], 33, row["words"], True, delta=4, lates=(0, 2))
    assert (t["n_ref"], t["n_hyp"], t["early_turn"], t["in_flush"]) == (1, 3, 1, 1) and t["late0"]["hits"] == 1 and t["late0"]["lat"] == [round(32 * 0.08 - 2.0, 4)]
    assert turn_metrics([31], 33, row["words"], False, delta=4)["late0"]["hits"] == 0

def test_pad_matches_semcommit_dataset_rule():
    """평가 오디오 길이 = 학습 길이: SemCommitDataset 의 need = max_δ last_emit_chunk(ev, δ) + tail_margin (M=0) 과 train_pad_K 가 같다."""
    sd = pytest.importorskip("vapasr.data.semcommit_dataset")
    if not all(hasattr(sd, f) for f in ("build_semcommit_tokens", "last_emit_chunk")): pytest.skip("semcommit_dataset API 변경")
    rng = random.Random(3)
    for _ in range(200):
        n = rng.randint(1, 6); t = sorted(round(rng.uniform(0.1, 6.0), 2) for _ in range(n)); row = mk_row("s", "English", [(f"w{i}", x, [], rng.randint(1, 2)) for i, x in enumerate(t)], t[-1] + 0.3)
        lab = dict(candidates=[cand(i, rng.choice("ABN")) for i in rng.sample(range(n), rng.randint(0, n))], turn_end=rng.random() < 0.7)
        delays = tuple(sorted(rng.sample([1, 2, 3, 4, 5, 6, 8], rng.randint(1, 4)))); turn = rng.random() < 0.8; h = rng.choice([0.24, 0.48, 0.8]); tm = rng.randint(0, 3)
        ev, _ = sd.build_semcommit_tokens(row["words"], row["tokens"], lab, SEM, TURN, turn_end=turn, hangover_s=h)
        has_turn = any(len(x) > 2 for x in ev); want = max(row["K"], max(sd.last_emit_chunk(ev, d, None, 0) for d in delays) + tm)
        assert train_pad_K(row["K"], t[-1], delays, has_turn, h, tm) == want and has_turn == (turn and lab["turn_end"])

def test_turn_end_missing_defaults_true():
    """labels 행에 turn_end 가 없으면 True — 패딩(eval_len)·학습(build_semcommit_tokens)·채점(turn_metrics)·oracle 이 모두 같은 기본."""
    row = mk_row("s", "English", [("a", 1.0, [], 1), ("b", 2.0, [], 1)], 2.3); lab = dict(id="s", lang="English", candidates=[cand(1, "A")])
    assert turn_flag(lab) and turn_flag(None) and not turn_flag(dict(turn_end=False))
    K = eval_audio_len(row, (2, 4), turn_flag(lab))[1]; assert K == 33                    # TURN 규칙(없으면 31)
    h = oracle_hyp(row, lab, 4, K, eval_turn=True); assert [t for _, t in h["emitted"]].count(TURN) == 1
    m = score_stream(row, lab, h, 4, K, eval_turn=True); assert m["turn"]["n_ref"] == 1 and m["turn"]["late0"]["hits"] == 1
    sd = pytest.importorskip("vapasr.data.semcommit_dataset")
    ev, _ = sd.build_semcommit_tokens(row["words"], row["tokens"], lab, SEM, TURN, turn_end=True); assert any(len(x) > 2 for x in ev)

def test_eval_len_keeps_manifest_K():
    """안 늘린 행은 words K(manifest) 를 그대로 — 길이에서 다시 반올림하면 d·12.5 = x.5 에서 1 청크 모자란다(ls-test-clean 1580-141083-0019: d 4.36, K 55)."""
    row = dict(mk_row("s", "English", [("a", 0.5, [], 1), ("b", 1.0, [], 1)], 4.36), K=55)
    assert int(round(round(4.36 * 16000) / 16000 / 0.08)) == 54                            # 옛 평가 유도(은행가 반올림)
    assert eval_audio_len(row, (2, 3, 4, 6), True) == (4.36, 55) and eval_audio_duration(row, (2, 3, 4, 6), True) == 4.36
    short = dict(mk_row("s", "English", [("a", 2.0, [], 1)], 2.1), K=26); d, K = eval_audio_len(short, (2, 3, 4, 6), True); assert K == 33 and d == round(33 * 0.08, 6)
    assert int(round(round(d * 16000) / 16000 / 0.08)) == K                                 # 늘린 행은 K'·0.08 이라 어느 쪽이든 같다
    assert train_pad_K(26, 2.0, (2, 4), False, pad_tail=False) == 26 and train_pad_K(26, 2.0, (2, 4), False) == 25 + 4 + 2

# ───────────────────────────── 종단: oracle ─────────────────────────────
def _toy_set():
    en = mk_row("en1", "English", [("i", 0.6, [], 1), ("saw", 0.9, [], 1), ("him", 1.2, ["punct_final"], 2), ("but", 1.8, [], 1), ("he", 2.0, [], 1), ("left", 2.4, ["punct_final"], 1)], 2.8, "librispeech-test")
    ko = mk_row("ko1", "Korean", [("어", 0.4, ["filler"], 1), ("저는", 0.9, [], 2), ("괜찮아요", 1.6, ["punct_final"], 3)], 2.2, "kspon-dev")
    lab = {"en1": dict(id="en1", lang="English", candidates=[cand(2, "A"), cand(3, "N", "all wait", B={"q": "WAIT"}), cand(5, "A")], turn_end=True),
           "ko1": dict(id="ko1", lang="Korean", candidates=[cand(0, "N", "filler"), cand(1, "B"), cand(2, "A")], turn_end=True)}
    return {"en1": en, "ko1": ko}, lab

def test_oracle_scores_perfect_and_aggregate_by_lang():
    words, labels = _toy_set(); recs = []
    for sid, w in words.items():
        K = int(round(eval_audio_duration(w, (2, 3, 4, 6), True) * 12.5))
        h = oracle_hyp(w, labels[sid], 4, K, eval_turn=True)
        recs.append(dict(id=sid, lang=w["lang"], set=w["set"], metrics=score_stream(w, labels[sid], h, 4, K, eval_turn=True)))
    rep = aggregate(recs)
    o = rep["overall"]; assert o["streams"] == 2 and o["events"]["sem"] == 3 and o["events"]["turn"] == 2
    for L in ("late0", "late2", "late5"): assert o["timing"][L]["precision"] == 1.0 and o["timing"][L]["recall"] == 1.0 and o["timing"][L]["lag_chunks"]["mean"] == 0
    assert o["timing"]["late0"]["latency_s"] == o["timing"]["ideal_latency_s"]
    assert o["text"]["precision"] == 1.0 and o["text"]["recall"] == 1.0 and o["text"]["pcr"] == 0.0 and o["text"]["n_N"] == 2 and o["text"]["hardneg_commit_rate"] == 0.0
    assert o["turn"]["late0"]["recall"] == 1.0 and o["turn"]["in_flush"] == 0 and o["turn"]["early_turn"] == 0
    assert rep["by_lang"]["Korean"]["asr"]["cer_nospace"]["rate"] == 0.0 and rep["by_lang"]["English"]["asr"]["wer"]["rate"] == 0.0 and set(rep["by_set"]) == {"kspon-dev", "librispeech-test"}
    assert rep["by_lang"]["Korean"]["text"]["n_B"] == 1 and rep["by_lang"]["Korean"]["text"]["B_commit_rate"] == 0.0

def test_cli_oracle_end_to_end_and_resume():
    cli = _cli(); words, labels = _toy_set()
    with tempfile.TemporaryDirectory() as td:
        wp, lp, out = Path(td) / "w.jsonl", Path(td) / "l.jsonl", Path(td) / "rep" / "report.json"
        wp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in words.values())); lp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in labels.values()))
        args = ["--words", str(wp), "--labels", str(lp), "--delay", "4", "--turn-end", "--oracle", "--out", str(out)]
        rep = cli.main(args); c = rep["configs"]["oracle"]
        assert c["complete"] and c["overall"]["text"]["precision"] == 1.0 and c["overall"]["timing"]["late2"]["f1"] == 1.0 and c["overall"]["turn"]["late0"]["recall"] == 1.0
        assert {p["group"] for p in rep["pr_curve"]["oracle"]} == {"overall", "English", "Korean"}
        streams = Path(str(out.with_suffix("")) + ".streams.jsonl"); n = len(streams.read_text().splitlines())
        cli.main(args); assert len(streams.read_text().splitlines()) == n == 2                       # 이어하기: 이미 한 (id, 설정) 은 다시 안 씀
        rep2 = cli.main(["--words", str(wp), "--labels", str(lp), "--delay", "4", "--turn-end", "--score-only", "--out", str(out)])
        assert rep2["configs"]["oracle"]["overall"]["text"] == c["overall"]["text"]
        with pytest.raises(AssertionError): cli.main(["--words", str(wp), "--labels", str(lp), "--delay", "2", "--turn-end", "--oracle", "--out", str(out)])   # 설정 불일치
        ids, info = cli.select_streams(words, labels, ["Korean"], 0); assert ids == ["ko1"] and info["selected"] == 1
        dc = json.loads(Path(str(streams) + ".config.json").read_text())
        assert dc["words_sha256"] == cli.sha256_file(wp) and set(dc["code"]) >= {"experiments/semcommit_eval.py", "vapasr/hf/commit_metrics.py", "vapasr/hf/batch_decode.py"}
        assert dc["pad_delays"] == [2, 3, 4, 6] and dc["turn_end"] is True and all(r["words_digest"] == cli.row_digest(words[r["id"]]) for r in map(json.loads, streams.read_text().splitlines()))
        w2 = json.loads(json.dumps(words)); w2["en1"]["tokens"][0][1] = 0.55                                        # 같은 경로에 다시 만든 words(토큰 시각 변경)
        wp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in w2.values()))
        with pytest.raises(AssertionError, match="이어하기 지문"): cli.main(args)                                  # 이어하기: words sha256 불일치
        with pytest.raises(AssertionError, match="words_digest"): cli.main(["--words", str(wp), "--labels", str(lp), "--score-only", "--out", str(out)])

def test_cli_score_only_rejects_changed_padding():
    """labels.turn_end 가 디코드 뒤에 바뀌면(δ 2 4 패딩에서 K' 가 달라짐) 재채점이 멈춘다 — 다른 길이의 오디오로 만든 방출."""
    cli = _cli(); words, labels = _toy_set()
    with tempfile.TemporaryDirectory() as td:
        wp, lp, out = Path(td) / "w.jsonl", Path(td) / "l.jsonl", Path(td) / "report.json"
        dump = lambda path, rows: path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()))
        dump(wp, words); dump(lp, labels)
        rep = cli.main(["--words", str(wp), "--labels", str(lp), "--delay", "4", "--pad-delays", "2", "4", "--oracle", "--out", str(out)])
        assert rep["decode"]["pad_delays"] == [2, 4] and rep["decode"]["turn_end"] is True and rep["configs"]["oracle"]["overall"]["turn"]["late0"]["recall"] == 1.0
        labels["en1"]["turn_end"] = False; dump(lp, labels)
        with pytest.raises(AssertionError, match="K'"): cli.main(["--words", str(wp), "--labels", str(lp), "--score-only", "--out", str(out)])
        del labels["en1"]["turn_end"]; dump(lp, labels)                                                             # 키 없음 = True → 디코드 때와 같다
        assert cli.main(["--words", str(wp), "--labels", str(lp), "--score-only", "--out", str(out)])["configs"]["oracle"]["overall"]["turn"]["n_ref"] == 2

def _args(cli, model, *extra):
    return cli.parse_args(["--model", str(model), "--words", "w.jsonl", "--labels", "l.jsonl", "--out", "r.json", *extra])

def test_resolve_settings_from_training_config():
    """평가 패딩·TURN 설정은 학습 설정에서: config.json semcommit > config.delays > run.json args; 명시값이 다르면 멈춤; 평가 δ ∉ 학습 δ 면 멈춤."""
    cli = _cli()
    with tempfile.TemporaryDirectory() as td:
        run = Path(td) / "run"; fin = run / "final"; fin.mkdir(parents=True)
        (run / "run.json").write_text(json.dumps(dict(args=dict(no_turn_end=False, hangover=0.48, tail_margin=2, M=0, delays="2,3,4,6"))))
        a = cli.resolve_settings(_args(cli, fin))
        assert (a.turn_end, a.hangover_s, a.tail_margin, a.pad, a.pad_tail) == (True, 0.48, 2, [2, 3, 4, 6], True) and a.train["src"] == [str(run / "run.json")]
        (fin / "config.json").write_text(json.dumps(dict(delays=[2, 3, 4], semcommit=dict(turn_end=False, hangover_s=0.4, tail_margin=3, pad_tail=True, M=0, delays=[2, 3, 4]))))
        a = cli.resolve_settings(_args(cli, fin)); assert (a.turn_end, a.hangover_s, a.tail_margin, a.pad) == (False, 0.4, 3, [2, 3, 4])
        with pytest.raises(AssertionError, match="turn_end"): cli.resolve_settings(_args(cli, fin, "--turn-end"))
        a = cli.resolve_settings(_args(cli, fin, "--turn-end", "--allow-train-mismatch")); assert a.turn_end is True
        assert cli.resolve_settings(_args(cli, fin, "--no-turn-end", "--tail-margin", "3")).turn_end is False  # 같은 값을 명시하면 통과
        with pytest.raises(AssertionError, match="δ=5"): cli.resolve_settings(_args(cli, fin, "--delay", "5"))
        assert cli.resolve_settings(_args(cli, fin, "--delay", "6", "--allow-untrained-delay")).pad == [2, 3, 4, 6]   # 평가 δ 도 패딩에 포함
        with pytest.raises(AssertionError, match="pad_delays"): cli.resolve_settings(_args(cli, fin, "--pad-delays", "2", "4"))
        (fin / "config.json").write_text(json.dumps(dict(delays=[2, 4], semcommit=dict(M=2))))
        with pytest.raises(AssertionError, match="M"): cli.resolve_settings(_args(cli, fin))
        empty = Path(td) / "bare" / "m"; empty.mkdir(parents=True)
        a = cli.resolve_settings(_args(cli, empty, "--delay", "8")); assert (a.turn_end, a.hangover_s, a.tail_margin, a.pad, a.train) == (True, 0.48, 2, [2, 3, 4, 6, 8], {})
        o = cli.resolve_settings(cli.parse_args(["--oracle", "--words", "w", "--labels", "l", "--out", "r", "--delay", "8"])); assert o.pad == [2, 3, 4, 6, 8] and o.turn_end is True

# ───────────────────────────── 가짜 모델 디코드 ─────────────────────────────
def _fake_model(V=12, D=16, cap=4, speech=None):
    """다음 토큰 분포가 마지막 입력에만 달린 전이표 모델. 0‥V-1 = 토큰 one-hot, 12 = 발화 청크, 13 = 무음 청크.
    발화 → 텍스트 1 (speech={tid: p} 로 바꿈), 텍스트 1 → p(SEM)=0.4/p(NEXT)=0.6, SEM → p(SEM)=0.9/p(NEXT)=0.1 (가드 없으면 SEM 폭주), 그 밖 → NEXT."""
    import math
    NEXT, EMPTY, S, T = 5, 6, 7, 8
    W = torch.full((V, D), -30.0)
    def col(d, probs):
        for t, p in probs.items(): W[t, d] = math.log(p)
    for d in range(D): col(d, {NEXT: 1.0})
    W[:, 12] = -30.0; W[1, 12] = 10.0
    if speech: W[:, 12] = -30.0; col(12, speech)
    W[:, 1] = -30.0; col(1, {S: 0.4, NEXT: 0.6})
    W[:, S] = -30.0; col(S, {S: 0.9, NEXT: 0.1})
    emb = torch.nn.Embedding(V, D); emb.weight.data.zero_(); emb.weight.data[:, :V] = torch.eye(V)
    thinker = SimpleNamespace(model=lambda inputs_embeds, past_key_values=None, use_cache=True: SimpleNamespace(last_hidden_state=inputs_embeds), lm_head=lambda h: h @ W.T)
    m = SimpleNamespace(thinker=thinker, get_input_embeddings=lambda: emb, chunk_embed=lambda x: x[:, 0], blocked=torch.tensor([9]), next_audio=NEXT, empty_audio=EMPTY,
                        config=SimpleNamespace(max_flush_rounds=8, runaway_cap=cap))
    audio = torch.zeros(1, 3, D); audio[0, 0, 12] = 1; audio[0, 1:, 13] = 1                     # K=3: 발화, 무음, 무음
    return m, audio, S, T

def test_decode_rows_bias_threshold_guard_trace():
    cli = _cli(); m, f, S, T = _fake_model()
    pol = [dict(mode="bias", value=0.0), dict(mode="bias", value=1.0), dict(mode="threshold", value=0.3), dict(mode="threshold", value=0.5)]
    st = cli.decode_rows(m, [f] * 4, [0, 0], pol, sem_id=S, turn_id=T, cache=[])
    assert [s.emitted for s in st] == [[(0, 1)], [(0, 1), (0, S)], [(0, 1), (0, S)], [(0, 1)]]      # bias +1·θ 0.3 만 commit, SEM 뒤 SEM 은 가드가 막음
    assert all(s.done and s.flush_rounds == 1 and s.forced == 0 for s in st) and [s.guard_blocked for s in st] == [0, 1, 1, 0]   # 막은 SEM 뒤 SEM 을 센다
    tr = st[0].trace; assert [k for k, _, _ in tr] == [0, 0, 1, 2, 3] and abs(tr[1][1] - 0.4) < 1e-6 and tr[0][1] < 1e-6       # 결정 step 마다 p(SEM)
    ng = cli.decode_rows(m, [f], [0], [dict(mode="bias", value=1.0)], sem_id=S, turn_id=T, sem_guard=False, cache=[])[0]
    assert ng.emitted == [(0, 1), (0, S), (0, S), (0, S)] and ng.forced == 1                     # 가드 끄면 runaway_cap(4) 까지 SEM 반복
    assert cm.events_from_emits(st[1].emitted, S) == [0]

def test_guard_blocked_counts_premature_sem_before_text():
    """발화 청크 argmax 가 텍스트 전 SEM(p=0.7): 가드 끄면 no_word 조기 commit, 가드 켜면 방출은 없지만 guard_blocked 로 세고 pcr_raw 가 조기 commit 으로 센다."""
    cli = _cli(); m, f, S, T = _fake_model(speech={7: 0.7, 1: 0.3}); assert S == 7
    pol = [dict(mode="bias", value=0.0), dict(mode="threshold", value=0.5), dict(mode="threshold", value=0.8)]
    on = cli.decode_rows(m, [f] * 3, [0], pol, sem_id=S, turn_id=T, cache=[])
    assert [s.emitted for s in on] == [[(0, 1)]] * 3 and [s.guard_blocked for s in on] == [1, 1, 0] and abs(on[0].trace[0][1] - 0.7) < 1e-6
    off = cli.decode_rows(m, [f], [0], pol[:1], sem_id=S, turn_id=T, sem_guard=False, cache=[])[0]
    assert off.emitted[0] == (0, S) and off.guard_blocked == 0
    row = mk_row("s", "English", [("la", 0.1, [], 1)], 0.5); lab = dict(candidates=[cand(0, "A")])
    fns = dict(is_text=lambda t: t == 1, word_start=lambda t: True, decode=lambda ids: "la" * len(ids), event_ids=(S, T))
    score = lambda s: finalize(score_stream(row, lab, dict(emitted=s.emitted, text="la", **split_hyp(s.emitted, **fns)), 2, 3, sem_id=S, turn_id=T, guard_blocked=s.guard_blocked))
    g, o = score(on[0]), score(off)
    assert (g["text"]["n_hyp"], g["text"]["premature"], g["text"]["pcr"], g["text"]["guard_blocked"], g["text"]["pcr_raw"], g["events"]["guard_blocked"]) == (0, 0, None, 1, 1.0, 1)
    assert o["text"]["premature_by_category"]["no_word"] >= 1
    assert o["text"]["pcr"] == 1.0 and o["text"]["guard_blocked"] == 0 and o["text"]["pcr_raw"] == 1.0

class _FakeTok:
    """byte-BPE 흉내: 'Ġ' = 앞 공백. 151643 이상은 특수/추가 토큰."""
    P = {1: "Ġthe", 2: "Ġca", 3: "t", 4: "Ġsat", 151705: "<NEXT_AUDIO>", SEM: "<SEM_END>", TURN: "<TURN_END>"}
    all_special_ids = [151643]; added_tokens_decoder = {151705: None, SEM: None, TURN: None}
    def convert_ids_to_tokens(self, i): return self.P[i]
    def decode(self, ids, skip_special_tokens=True): return "".join(self.P[i].replace("Ġ", " ") for i in ids if i < 151643)

def test_hyp_from_emitted_with_tokenizer_fns():
    cli = _cli(); em = [(0, 1), (1, 2), (1, SEM), (2, 3), (3, SEM), (3, 4), (4, SEM), (6, TURN)]
    h = cli.hyp_from_emitted(em, _FakeTok(), SEM, TURN)
    assert h["words"] == ["the", "cat", "sat"] and h["text"] == "the cat sat"
    assert [(e["id"], e["after"], e["mid_word"]) for e in h["events"]] == [(SEM, 1, True), (SEM, 1, False), (SEM, 2, False), (TURN, 2, False)]
    w = mk_row("s", "English", [("the", 0.3, [], 1), ("cat", 0.6, [], 2), ("sat", 0.9, [], 1)], 1.5); lab = dict(candidates=[cand(1, "N", "other"), cand(2, "A")])
    m = score_stream(w, lab, dict(emitted=em, **h), 2, w["K"])
    assert m["text"]["cat"]["mid_word"] == 1 and m["text"]["N_hit"] == 1 and m["text"]["correct"] == 1 and m["asr"]["wer"]["errors"] == 0 and m["events"]["sem"] == 3

def test_rowstate_follows_decodestate_protocol():
    cli = _cli(); s = cli.RowState(K=2, cap=2, max_flush=8, max_total=20)
    assert s.consume(11, 99) == ("text", 11) and s.consume(99, 99) == ("next", 0) and s.consume(11, 99) == ("audio", 1)
    assert s.consume(12, 99) == ("text", 12); s.consume(99, 99); assert s.consume(0, 99) == ("empty", 0)
    s.consume(13, 99); s.consume(99, 99); s.consume(0, 99); s.consume(99, 99)
    assert s.done and s.emitted == [(0, 11), (1, 12), (2, 13)] and s.flush_rounds == 2
    z = cli.RowState(K=1, cap=1, max_flush=0, max_total=10); z.consume(11, 99); assert z.consume(12, 99) == ("next", 0) and z.done and z.forced == 1

def _patch_fake_run(monkeypatch, cli, encs=None, loads=None, peak=None):
    """run() 용 가짜 모델·tokenizer·오디오. encs: encode 호출 (len(wav), K) 기록, loads/peak: 오디오 로드 수·(로드 − 인코드) 최대."""
    import numpy as np
    m, _, S, T = _fake_model()
    def encode(w, wl, K):
        K = int(K[0]); x = torch.zeros(1, 1, K, 16); x[0, 0, : K // 2, 12] = 1; x[0, 0, K // 2:, 13] = 1
        if encs is not None: encs.append((int(wl[0]), K))
        return x
    m.encode = encode; m.parameters = lambda: iter([m.get_input_embeddings().weight]); m.config.prefix_ids = [0]; m.config.sp_ids = {"<DELAY_4>": 0}
    class Tok(_FakeTok):
        P = {1: "Ġla", S: "<SEM_END>", T: "<TURN_END>"}; added_tokens_decoder = {S: None, T: None}
        def __call__(self, text, add_special_tokens=False): return {"input_ids": [0, 0]}
    monkeypatch.setattr(cli, "load_sem_model", lambda *a, **k: (m, Tok()))
    monkeypatch.setattr(cli, "check_sem_ids", lambda *a, **k: (S, T))
    def audio(wrow, dur, remaps):
        if loads is not None: loads.append(wrow["id"]); peak[0] = max(peak[0], len(loads) - len(encs))
        if wrow["id"] == "bad": raise OSError("missing file")
        return np.zeros(int(round(dur * 16000)), np.float32)
    monkeypatch.setattr(cli, "load_stream_audio", audio)
    orig = cli.decode_rows; monkeypatch.setattr(cli, "decode_rows", lambda *a, **k: orig(*a, **dict(k, cache=[])))     # 가짜 모델은 KV 캐시를 안 쓴다
    return m, S, T

def _dump(path, rows): path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows.values()))

def test_run_loop_with_fake_model(monkeypatch):
    """run(): 언어별 배치·스트림×설정 행·특징 캐시·기록·이어하기·오디오 실패 건너뛰기·guard_blocked·K'=words K·체크포인트 지문 (모델·tokenizer·오디오는 가짜)."""
    cli = _cli(); encs = []; m, S, T = _patch_fake_run(monkeypatch, cli, encs)
    words, labels = _toy_set(); bad = mk_row("bad", "English", [("x", 0.5, [], 1)], 1.0); words["bad"] = bad; labels["bad"] = dict(id="bad", lang="English", candidates=[], turn_end=False)
    words["en2"] = dict(mk_row("en2", "English", [("go", 0.5, [], 1), ("now", 1.0, [], 1)], 4.36, "librispeech-test"), K=55)       # d·12.5 = 54.5 — manifest K 55
    labels["en2"] = dict(id="en2", lang="English", candidates=[cand(1, "A")], turn_end=True)
    with tempfile.TemporaryDirectory() as td:
        wp, lp, out, md = Path(td) / "w.jsonl", Path(td) / "l.jsonl", Path(td) / "report.json", Path(td) / "model"; md.mkdir(); _dump(wp, words); _dump(lp, labels)
        args = ["--model", str(md), "--words", str(wp), "--labels", str(lp), "--sem-bias", "0", "1", "--theta", "0.3", "--batch-size", "2", "--out", str(out)]
        rep = cli.main(args); streams = Path(td) / "report.streams.jsonl"; recs = [json.loads(l) for l in streams.read_text().splitlines()]
        assert sorted((r["id"], r["config"]["name"]) for r in recs) == sorted((i, c) for i in ("en1", "en2", "ko1") for c in ("bias=0", "bias=1", "theta=0.3"))
        assert json.loads((Path(td) / "report.streams.jsonl.audio-failures.json").read_text()) == {"bad": "OSError: missing file"}
        c = rep["configs"]; assert set(c) == {"bias=0", "bias=1", "theta=0.3"} and not c["bias=0"]["complete"] and c["bias=0"]["n_streams"] == 3
        r0 = next(r for r in recs if r["id"] == "en1" and r["config"]["name"] == "bias=1"); n_txt = sum(t == 1 for r in recs if r["config"]["name"] == "bias=1" for _, t in r["emitted"])
        assert c["bias=0"]["overall"]["events"]["sem"] == 0 and c["bias=1"]["overall"]["events"]["sem"] == n_txt == c["theta=0.3"]["overall"]["events"]["sem"] > 0   # 텍스트마다 1 회, 연속 SEM 은 가드가 막음
        for name in ("bias=1", "theta=0.3"):                                                                                    # 막은 연속 SEM 은 guard_blocked 로 보고
            t = c[name]["overall"]["text"]; assert t["guard_blocked"] == n_txt == sum(r["guard_blocked"] for r in recs if r["config"]["name"] == name) and t["pcr_raw"] >= t["pcr"]
        assert c["bias=0"]["overall"]["text"]["guard_blocked"] == 0 and {p["guard_blocked"] for p in rep["pr_curve"]["bias"] if p["value"] == 1.0 and p["group"] == "overall"} == {n_txt}
        assert r0["emitted"][:3] == [[0, 1], [0, S], [1, 1]] and r0["trace"]["steps"] == len(r0["trace"]["k"]) and abs(r0["trace"]["p_sem"][1] - 0.4) < 1e-4
        e2 = [r for r in recs if r["id"] == "en2"]; assert {r["K"] for r in e2} == {55} and (69760, 55) in encs                  # 인코더·채점 K = words K(55), 길이에서 다시 유도한 54 가 아님
        assert {p["value"] for p in rep["pr_curve"]["bias"]} == {0.0, 1.0} and {p["group"] for p in rep["pr_curve"]["threshold"]} == {"overall", "English", "Korean"}
        assert rep["decode"]["turn_end"] is True and rep["decode"]["train"] == {} and rep["decode"]["checkpoint_weights"] == {}  # 학습 설정 모름 → 학습 기본(TURN 켬)
        cli.main(args); assert len(streams.read_text().splitlines()) == 9                                                           # 이어하기: 새로 쓴 행 없음
        (md / "model.safetensors").write_bytes(b"new weights")                                                                     # 같은 경로에 다시 저장된 체크포인트
        with pytest.raises(AssertionError, match="checkpoint_weights"): cli.main(args + ["--sem-bias", "0", "1", "2"])
        (md / "model.safetensors").unlink(); (md / "config.json").write_text(json.dumps(dict(delays=[2, 3, 4, 6])))
        with pytest.raises(AssertionError, match="checkpoint_config_sha256"): cli.main(args)

def test_run_prefetch_is_bounded(monkeypatch):
    """오디오 선읽기는 창(--prefetch) 만큼만: 로드 시작 수 − 인코드 수 ≤ 창 + 1 (옛 코드는 스트림 전부를 한꺼번에 제출)."""
    cli = _cli(); encs, loads, peak = [], [], [0]; _patch_fake_run(monkeypatch, cli, encs, loads, peak)
    words = {f"s{i:02d}": mk_row(f"s{i:02d}", "English", [("la", 0.4 + 0.01 * i, [], 1)], 1.0 + 0.01 * i) for i in range(12)}
    labels = {k: dict(id=k, lang="English", candidates=[cand(0, "A")], turn_end=True) for k in words}
    with tempfile.TemporaryDirectory() as td:
        wp, lp, md = Path(td) / "w.jsonl", Path(td) / "l.jsonl", Path(td) / "model"; md.mkdir(); _dump(wp, words); _dump(lp, labels)
        rep = cli.main(["--model", str(md), "--words", str(wp), "--labels", str(lp), "--batch-size", "1", "--io-workers", "2", "--prefetch", "2", "--out", str(Path(td) / "r.json")])
    assert len(loads) == len(encs) == 12 and sorted(loads) == sorted(words) and peak[0] <= 3 and rep["configs"]["bias=0"]["complete"]

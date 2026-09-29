"""Semantic commit 평가(vapasr/hf/commit_metrics.py, experiments/semcommit_eval.py) — 최적 매칭·비대칭 창·지연 산술·텍스트 위치 PCR 범주·
이벤트 추출·참조 직렬화(oracle) 상한·가짜 모델 디코드(bias/threshold/guard/trace/guard_blocked)·CLI 종단(이어하기 지문·words 행 지문·K' 검사·
학습 설정 복원·δ 검사·오디오 선읽기 상한)·창 없는 지연(A 경계별 commit·표시·첫 토큰·after-text lag·옛 jsonl word_k backfill). 합성 데이터, CPU 수 초."""
import importlib.util, json, random, sys, tempfile, types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vapasr.hf import commit_metrics as cm
from vapasr.hf.commit_metrics import (SEM_END_ID as SEM, EOT_ID as TURN, aggregate, align_pairs, emit_time, events_from_emits, finalize, map_events, match_optimal,
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
    assert f["timing"]["late2"]["latency_s"] == dict(n=2, mean=0.46, p50=0.46, p90=0.54, p95=0.55, p99=0.558, max=0.56) and f["timing"]["late0"]["miss"] == 1 and f["timing"]["late0"]["recall"] == 0.5
    assert "commit" not in f and "display" not in f                                     # 손으로 만든 m(지연 블록 없음)도 finalize 통과
    assert cm.latency_stats([]) == dict(n=0, mean=None, p50=None, p90=None, p95=None, p99=None, max=None)
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
    assert h["word_k"] == [[1, 2], [3, 3]]                                              # 가설 단어별 [첫, 마지막] 텍스트 토큰 청크(비텍스트 토큰은 무시)

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
    c, d = o["commit"], o["display"]                                                     # 창 없는 지연: oracle 은 이상 지연 그대로·전부 확정·lag 0
    assert c["latency_s"] == o["timing"]["ideal_latency_s"] and c["coverage"] == 1.0 and c["exact_rate"] == 1.0 and c["excess_chunks"]["max"] == 0
    assert c["n_A"] == o["text"]["n_A"] == 3 and c["streams_no_A"] == 0 and c["B"]["committed"] == 0 and c["N"]["rate"] == 0.0 and c["prefix"]["latency_s"] == c["latency_s"]
    assert d["coverage"] == 1.0 and d["lag_chunks"]["max"] == 0 and d["lag_chunks"]["mean"] == 0 and d["missing"] == 0 and d["first_token_s"]["n"] == 2
    assert o["after_text"]["n"] == 3 and o["after_text"]["zero_rate"] == 1.0 and o["after_text"]["A"]["n"] == 3

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
        rep = cli.main(["--words", str(wp), "--labels", str(lp), "--delay", "4", "--pad-delays", "2", "4", "--oracle", "--turn-end", "--out", str(out)])   # 턴 종료는 선택(기본 끔)
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
        a = cli.resolve_settings(_args(cli, empty, "--delay", "8")); assert (a.turn_end, a.hangover_s, a.tail_margin, a.pad, a.train) == (False, 0.48, 2, [2, 3, 4, 6, 8], {})   # 학습 기본 = <SEM_END> 만
        o = cli.resolve_settings(cli.parse_args(["--oracle", "--words", "w", "--labels", "l", "--out", "r", "--delay", "8"])); assert o.pad == [2, 3, 4, 6, 8] and o.turn_end is False
        new = Path(td) / "new"; nf = new / "final"; nf.mkdir(parents=True)                                       # semcommit_train v1 인자(dest turn_end)
        (new / "run.json").write_text(json.dumps(dict(args=dict(turn_end=True, hangover=0.48, tail_margin=2, M=0, delays="2,3,4,6"))))
        assert cli.resolve_settings(_args(cli, nf)).turn_end is True

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

def test_decode_rows_without_turn_token():
    """<SEM_END> 만 학습한 모델(turn_id=None): 턴 토큰 없이 디코드, trace 의 p(턴) = 0, 턴 이벤트 없음 — 결과는 턴 토큰이 있을 때와 같다(이 가짜 모델은 턴을 안 낸다)."""
    cli = _cli(); m, f, S, T = _fake_model()
    pol = [dict(mode="bias", value=1.0), dict(mode="threshold", value=0.3)]
    st = cli.decode_rows(m, [f] * 2, [0, 0], pol, sem_id=S, turn_id=None, cache=[]); ref = cli.decode_rows(m, [f] * 2, [0, 0], pol, sem_id=S, turn_id=T, cache=[])
    assert [s.emitted for s in st] == [s.emitted for s in ref] and all(pt == 0.0 for s in st for _, _, pt in s.trace)
    assert cm.events_from_emits(st[0].emitted, None) == [] and cm.split_hyp(st[0].emitted, is_text=lambda t: t != S, word_start=lambda t: True, decode=str, event_ids=(S, None))["events"][0]["id"] == S

def test_check_sem_ids_turn_token_from_registry():
    """턴 종료 id = config.sem_registry 의 학습 이벤트: SEM 만 → None(--turn-end 거부), mono 턴 → <EOT> 151722, v0.2 → <TURN_END> 151724. 차단된 턴 토큰은 거부."""
    from vapasr.data.semcommit_tokens import add_semantic_specials
    cli = _cli()
    class Tok:
        def __init__(s): s.v = {}
        def add_tokens(s, toks, special_tokens=True):
            for t in toks: s.v.setdefault(t, 151705 + len(s.v))
        def convert_tokens_to_ids(s, t): return s.v.get(t, 3)
    def model(reg, blocked=(0,)):
        return SimpleNamespace(config=SimpleNamespace(sp_ids={}, sem_registry=reg, lanes=0), get_input_embeddings=lambda: SimpleNamespace(weight=torch.zeros(151936, 1)),
                               blocked=torch.tensor(list(blocked)))
    tok = Tok(); add_semantic_specials(tok)
    assert cli.check_sem_ids(model({"<SEM_END>": SEM}), tok) == (SEM, None)
    with pytest.raises(AssertionError, match="턴 종료 토큰을 학습하지"): cli.check_sem_ids(model({"<SEM_END>": SEM}), tok, turn_end=True)
    assert cli.check_sem_ids(model({"<SEM_END>": SEM, "<EOT>": TURN}), tok, turn_end=True) == (SEM, TURN) == (151723, 151722)
    with pytest.raises(AssertionError, match="차단"): cli.check_sem_ids(model({"<SEM_END>": SEM, "<EOT>": TURN}, blocked=[TURN]), tok, turn_end=True)
    old = Tok(); add_semantic_specials(old); old.add_tokens(["<TURN_END>"])                          # v0.2 체크포인트 tokenizer
    assert cli.check_sem_ids(model({"<SEM_END>": SEM, "<TURN_END>": cm.LEGACY_TURN_END_ID}), old, turn_end=True) == (SEM, 151724)

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
    assert h["word_k"] == [[0, 0], [1, 2], [3, 3]]
    rec = cli.make_record(dict(id="s", lang="English", K=10), dict(name="bias=0"), 2, 10, 0.8, em, h)
    assert rec["hyp"]["word_k"] == [[0, 0], [1, 2], [3, 3]]                             # 스트림 jsonl 에 저장(표시·첫 토큰 지연 재채점용)
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
        assert all(len(r["hyp"]["word_k"]) == len(r["hyp"]["words"]) > 0 for r in recs)                                            # 표시 지연 입력 저장
        assert all(c[n]["overall"]["commit"]["n_A"] == 4 and c[n]["overall"]["display"]["missing"] == 0 for n in c)                  # 설정별 창 없는 지연 요약
        assert {p["value"] for p in rep["pr_curve"]["bias"]} == {0.0, 1.0} and {p["group"] for p in rep["pr_curve"]["threshold"]} == {"overall", "English", "Korean"}
        assert rep["decode"]["turn_end"] is False and rep["decode"]["train"] == {} and rep["decode"]["checkpoint_weights"] == {}  # 학습 설정 모름 → 학습 기본(<SEM_END> 만)
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

def test_pc_punct_after_and_commit_counts():
    """구두점 전사 정렬: 대소문자·구두점 무시 정렬, 전사에 없는 단어는 빠짐, 커밋 위치별 SENT/CLAUSE/NONE 과 문장 끝 재현."""
    words = [dict(i=i, text=w, end_time=0.5 * (i + 1)) for i, w in enumerate("i went home then i slept well".split())]
    pa = cm.pc_punct_after(words, "I went home, then I slept well.")
    assert pa == {0: "NONE", 1: "NONE", 2: "CLAUSE", 3: "NONE", 4: "NONE", 5: "NONE", 6: "SENT"}
    assert cm.pc_punct_after(words[:3] + [dict(i=3, text="uh", end_time=2.0)], "I went home.") == {0: "NONE", 1: "NONE", 2: "SENT"}
    hyp = "i went home then i slept well".split()
    ev = [dict(id=SEM, k=10, after=2), dict(id=SEM, k=20, after=6), dict(id=SEM, k=21, after=6)]      # 같은 위치 중복은 한 번
    c = cm.pc_commit_counts(words, "I went home, then I slept well.", hyp, ev)
    assert c == dict(commits=2, SENT=1, CLAUSE=1, NONE=0, n_sent=1, sent_hit=1)
    assert cm.pc_commit_counts(words, "I went home, then I slept well.", hyp, [])["commits"] == 0

def test_gold_label_and_commit_counts():
    """골드(전수 주석) 대조: 교사 A/B/N 을 골드 COMMIT/AMBIG/NO 에 대어 세고, 모델 확정은 map_events 위치로 hit/ambig/no·중복·지연을 센다."""
    words = [dict(i=i, text=w, end_time=0.4 * (i + 1)) for i, w in enumerate("i went home and then i slept well".split())]
    gold = dict(commit=[2, 7], ambig=[4])
    cands = [dict(after_word=2, grade="A"), dict(after_word=3, grade="N"), dict(after_word=4, grade="B"), dict(after_word=7, grade="N")]
    c = cm.gold_label_counts(gold, cands); f = cm.finalize_gold_labels(c)
    assert (c["gold_commit"], c["cand_at_commit"], c["A_at_commit"], c["N_at_commit"], c["N_at_no"], c["B_at_ambig"]) == (2, 2, 1, 1, 1, 1)
    assert f["A_precision"] == 1.0 and f["A_recall"] == 0.5 and f["cand_recall"] == 1.0 and f["N_precision"] == 0.5
    hyp = "i went home and then i slept well".split(); K = 40
    ev = [dict(id=SEM, k=10, after=2), dict(id=SEM, k=11, after=2), dict(id=SEM, k=14, after=4), dict(id=SEM, k=16, after=5), dict(id=SEM, k=30, after=7)]
    g = cm.gold_commit_counts(words, gold, hyp, ev, K=K); r = cm.finalize_gold_commits(g)
    assert (g["hit"], g["dup"], g["at_ambig"], g["at_no"]) == (2, 1, 1, 1) and g["lat"] == [round(cm.emit_time(10, K) - 1.2, 4), round(cm.emit_time(30, K) - 3.2, 4)]
    assert r["P"] == 0.5 and r["R"] == 1.0 and r["PCR_gold"] == 0.4                       # P = 2/(5−1), 오류 = no 1 + 중복 1

def test_gold_cli_merge_adjudication_and_scoring(tmp_path):
    """merge: 일치 → 그대로, 불일치 → 판정(없으면 AMBIG·판정 패킷), κ·COMMIT F1; score-labels·score-eval 이 골드를 읽는다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("semcommit_gold", ROOT / "experiments/semcommit_gold.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    row = mk_row("s1", "English", [("i", 0.4, [], 1), ("went", 0.8, [], 1), ("home", 1.2, [], 1), ("then", 1.6, [], 1), ("slept", 2.0, [], 1)], 3.0)
    wp = tmp_path / "w.jsonl"; wp.write_text(json.dumps(row) + "\n")
    a = {"s1": {"commit": [2, 4], "ambig": [], "why": {"2": "done", "4": "end"}}}; b = {"s1": {"commit": [4], "ambig": [2], "why": {"2": "maybe"}}}
    (tmp_path / "x.rules.json").write_text(json.dumps(a)); (tmp_path / "x.consumer.json").write_text(json.dumps(b))
    out = tmp_path / "gold.jsonl"; pk = tmp_path / "adj.txt"
    st = g.main(["merge", "--words", str(wp), "--ann", str(tmp_path / "x.rules.json"), str(tmp_path / "x.consumer.json"), "--out", str(out), "--adj-packet", str(pk)])
    gold = json.loads(out.read_text()); assert gold["commit"] == [4] and gold["ambig"] == [2] and gold["how"]["2"] == "unresolved" and "i=2 (home)" in pk.read_text()
    (tmp_path / "adj.json").write_text(json.dumps({"s1": {"2": "COMMIT"}}))
    g.main(["merge", "--words", str(wp), "--ann", str(tmp_path / "x.rules.json"), str(tmp_path / "x.consumer.json"), "--adj", str(tmp_path / "adj.json"), "--out", str(out)])
    gold = json.loads(out.read_text()); assert gold["commit"] == [2, 4] and gold["ambig"] == [] and gold["how"] == {"2": "adjudicated", "4": "agree"}
    lp = tmp_path / "l.jsonl"; lp.write_text(json.dumps(dict(id="s1", lang="English", candidates=[cand(2, "A"), cand(3, "N")])) + "\n")
    res = g.main(["score-labels", "--words", str(wp), "--labels", str(lp), "--gold", str(out)]); assert res["A_precision"] == 1.0 and res["A_recall"] == 0.5 and res["N_precision"] == 1.0
    rec = dict(id="s1", config=dict(name="bias=0"), K=40, delta=4, event_ids=[SEM, None], hyp=dict(words=["i", "went", "home", "then", "slept"], events=[dict(id=SEM, k=20, after=2, mid_word=False)]))
    sp = tmp_path / "r.streams.jsonl"; sp.write_text(json.dumps(rec) + "\n")
    ev = g.main(["score-eval", "--words", str(wp), "--gold", str(out), "--streams", str(sp)])["r"]["bias=0"]; assert ev["hit"] == 1 and ev["P"] == 1.0 and ev["R"] == 0.5

def test_gold_cli_tune_picks_thresholds_meeting_floor(tmp_path):
    """tune: 후보 특징(평균 P(SAFE)·P(REVISION)·구두점)으로 dev 절반에서 정밀도 floor 이상 중 재현율 최대 임계값 — 분리 가능한 합성 데이터면 dev·test 모두 P 1."""
    import importlib.util, hashlib
    spec = importlib.util.spec_from_file_location("semcommit_gold", ROOT / "experiments/semcommit_gold.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    words, gold, A, B, C = [], [], [], [], []
    for n in range(24):
        sid = f"k{n:02d}"; good = n % 3 != 0
        words.append(dict(id=sid, lang="Korean", duration_s=2.0, K=25, segments=[], words=[dict(i=i, text=f"w{i}", end_time=0.5 * (i + 1), seg=0, tags=[]) for i in range(3)]))
        gold.append(dict(id=sid, commit=[1] if good else [], ambig=[]))
        A.append(dict(stage="A", id=sid, status="ok", out=dict(semantic_boundaries=[1])))
        for j in ("exaone35", "qwen3"):
            B.append(dict(stage="B", id=sid, after_word=1, judge=j, status="ok", decision="SAFE" if good else "WAIT",
                          logprobs={"SAFE": 0.0 if good else -4.0, "WAIT": -4.0 if good else 0.0, "UNCERTAIN": -4.0}))
        C.append(dict(stage="C", id=sid, after_word=1, judge="qwen3", status="ok", relation="STABLE", logprobs={"REVISION": -5.0, "STABLE": 0.0}))
    assert {g.split_half(r["id"]) for r in words} == {"dev", "test"}
    paths = {}
    for name, rows in (("w", words), ("g", gold), ("a", A), ("b", B), ("c", C)):
        paths[name] = tmp_path / f"{name}.jsonl"; paths[name].write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep = g.main(["tune", "--words", str(paths["w"]), "--gold", str(paths["g"]), "--stageA", str(paths["a"]), "--stageB", str(paths["b"]), "--stageC", str(paths["c"]),
                  "--extra-candidates", "", "--floor", "0.9", "--min-branch-a", "1", "--out", str(tmp_path / "th.json")])
    k = rep["Korean"]; th = json.loads((tmp_path / "th.json").read_text())["Korean"]
    assert k["dev"]["P"] == 1.0 and k["dev"]["R"] == 1.0 and k["test"]["P"] == 1.0 and th["other"]["p_safe"] > 0.1   # WAIT 쪽(P(SAFE)≈0.02)은 A 가 안 된다


def test_gold_cli_tune_floors_each_branch(tmp_path):
    """tune: floor 는 가지마다 — 구두점 가지(전부 COMMIT)의 여유로 전체 정밀도가 floor 를 넘어도, 특징이 같은 C/NO 반반인 그 외 가지는 꺼진다."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("semcommit_gold", ROOT / "experiments/semcommit_gold.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    ids = [f"s{n:03d}" for n in range(100)]; half = {sid: g.split_half(sid) for sid in ids}
    other = {sid for h in ("dev", "test") for sid in [x for x in ids if half[x] == h][:4]}             # 절반마다 그 외 후보 4 개(C 2 · NO 2)
    other_c = {sid for h in ("dev", "test") for sid in [x for x in ids if half[x] == h][:2]}
    words, gold, A, B, C = [], [], [], [], []
    for sid in ids:
        words.append(dict(id=sid, lang="Korean", duration_s=2.0, K=25, segments=[],
                          words=[dict(i=i, text=f"w{i}", end_time=0.5 * (i + 1), seg=0, tags=["punct_final"] if i == 0 else []) for i in range(3)]))
        cands = [0, 1] if sid in other else [0]
        gold.append(dict(id=sid, commit=[0] + ([1] if sid in other_c else []), ambig=[]))
        A.append(dict(stage="A", id=sid, status="ok", out=dict(semantic_boundaries=cands)))
        for i in cands:
            for j in ("exaone35", "qwen3"):
                B.append(dict(stage="B", id=sid, after_word=i, judge=j, status="ok", decision="SAFE", logprobs={"SAFE": 0.0, "WAIT": -4.0, "UNCERTAIN": -4.0}))
            C.append(dict(stage="C", id=sid, after_word=i, judge="qwen3", status="ok", relation="STABLE", logprobs={"REVISION": -5.0, "STABLE": 0.0}))
    paths = {}
    for name, rows in (("w", words), ("g", gold), ("a", A), ("b", B), ("c", C)):
        paths[name] = tmp_path / f"{name}.jsonl"; paths[name].write_text("".join(json.dumps(r) + "\n" for r in rows))
    n_dev = sum(1 for sid in ids if half[sid] == "dev"); assert (n_dev + 2) / (n_dev + 4) >= 0.9     # 전체 floor 였다면 그 외 가지가 켜졌을 구성
    rep = g.main(["tune", "--words", str(paths["w"]), "--gold", str(paths["g"]), "--stageA", str(paths["a"]), "--stageB", str(paths["b"]), "--stageC", str(paths["c"]),
                  "--extra-candidates", "", "--floor", "0.9", "--out", str(tmp_path / "th.json")])
    th = json.loads((tmp_path / "th.json").read_text())["Korean"]; k = rep["Korean"]
    assert th["other"] is None and th["punct"] is not None
    assert k["branches"]["punct"]["dev"]["P"] == 1.0 and k["branches"]["other"]["dev"] is None and k["dev"]["P"] == 1.0 and k["test"]["P"] == 1.0

    # 최소 지지(--min-branch-a): dev A 가 그보다 적으면 정밀도가 floor 를 넘어도 가지를 켜지 않는다 — 구두점 가지의 dev A 는 스트림 수만큼
    n_dev_a = k["branches"]["punct"]["dev"]["n_A"]
    rep2 = g.main(["tune", "--words", str(paths["w"]), "--gold", str(paths["g"]), "--stageA", str(paths["a"]), "--stageB", str(paths["b"]), "--stageC", str(paths["c"]),
                   "--extra-candidates", "", "--floor", "0.9", "--min-branch-a", str(n_dev_a + 1), "--out", str(tmp_path / "th2.json")])
    assert json.loads((tmp_path / "th2.json").read_text())["Korean"]["punct"] is None and rep2["Korean"]["branches"]["punct"]["dev"] is None

# ───────────────────────────── 창 없는 지연(commit·표시·첫 토큰·after-text) ─────────────────────────────
def _abc(n=6, step=0.5):
    """참조 단어 a, b, c, … (끝 시각 step·(i+1))."""
    return [dict(i=i, text=w, end_time=round(step * (i + 1), 4), tags=[]) for i, w in enumerate("abcdefghij"[:n])]

def _sem(k, after, mid=False): return dict(id=SEM, k=k, after=after, mid_word=mid)

def _hyp(words, events, word_k=None, text=None):
    """score_stream 용 가설: 방출은 이벤트만(텍스트 토큰은 지표에 안 쓰인다), word_k 를 주면 표시 지연까지."""
    h = dict(emitted=sorted([[e["k"], e["id"]] for e in events]), words=list(words), events=list(events), text=" ".join(words) if text is None else text)
    if word_k is not None: h["word_k"] = word_k
    return h

def test_commit_latency_window_free_worked_example():
    """A1·A4 경계, B2·N3: 첫 확정만 세고(A1 중복 무시), mid_word(36)는 빼며, 창 밖의 늦은 commit(45, f 뒤)은 A4 를 overshoot 1 로 확정한다."""
    words = _abc(); cands = [cand(1, "A"), cand(2, "B"), cand(3, "N", "other"), cand(4, "A")]; K, d = 60, 4
    rk = [ref_chunk(w["end_time"], d, K) for w in words]; assert rk == [10, 16, 22, 29, 35, 41]
    hw = [w["text"] for w in words]; ev = [_sem(16, 1), _sem(17, 1), _sem(23, 2), _sem(36, 4, True), _sem(45, 5)]
    assert map_events(hw, hw, ev, rk, K) == [1, 1, 2, 4, 5]
    c = cm.commit_latency_metrics(words, cands, hw, ev, rk, K); conf = c.pop("_events")
    assert (c["n_A"], c["n_last"], c["committed"], c["exact"], c["in_flush"], c["streams_no_A"]) == (2, 0, 2, 1, 0, 0) and conf == [0, 4]
    assert c["lat"] == [0.36, 1.18] and c["excess"] == [0, 10] and c["overshoot"] == [0, 1] and c["prefix_lat"] == [0.36, 1.18] and c["last"] == [0, 0]
    assert c["B"] == dict(n=1, committed=1, lat=[0.42]) and c["N"] == dict(n=1, committed=0, lat=[])
    x = text_position_metrics(words, cands, hw, ev, rk, K)                                  # 항등식: 정확 위치 = text.correct, B/N 확정 = ambiguous/N_hit
    assert c["exact"] == x["correct"] == 1 and c["B"]["committed"] == x["ambiguous"] == 1 and c["N"]["committed"] == x["N_hit"] == 0
    t = timing_metrics([e["k"] for e in ev], K, words, cands, d)                             # 창 매칭은 청크만 봐서 mid_word(36)를 A4 에 맞히고 늦은 45 는 못 본다
    assert t["late5"]["hits"] == 2 and t["late5"]["lat"] == [0.36, 0.46]
    f = cm._finalize_commit(c, cm.LATE_THRESHOLDS_S, [0, 2, 5])
    assert f["coverage"] == 1.0 and f["exact_rate"] == 0.5 and f["late_rate"] == {"0.32": 1.0, "0.64": 0.5, "1": 0.5} and f["recall_at"] == {"0.32": 0.0, "0.64": 0.5, "1": 0.5}
    assert f["recall_within_chunks"] == {"0": 0.5, "2": 0.5, "5": 0.5} and f["excess_chunks"]["max"] == 10 and f["overshoot_words"]["mean"] == 0.5
    assert f["latency_s"] == dict(n=2, mean=0.77, p50=0.77, p90=1.098, p95=1.139, p99=1.1718, max=1.18)
    assert f["B"]["rate"] == 1.0 and f["N"]["rate"] == 0.0 and f["mid"]["committed"] == 2 and f["last"]["n_A"] == 0 and f["last"]["coverage"] is None
    a = cm.after_text_metrics(ev, map_events(hw, hw, ev, rk, K), [[k, k] for k in rk], K, conf)   # after-text: commit 청크 − 앞 단어 마지막 토큰 청크
    assert (a["n"], a["mid_word"], a["no_word"], a["no_text"], a["in_flush"], a["lag"], a["lag_A"]) == (4, 1, 0, 0, 0, [0, 1, 1, 4], [0, 4])
    fa = cm._finalize_after_text(a); assert fa["zero_rate"] == 0.25 and fa["ge2_rate"] == 0.25 and fa["A"]["zero_rate"] == 0.5 and fa["lag_chunks"]["max"] == 4

def test_commit_latency_next_A_cut_and_prefix():
    """다음 A 이후의 commit 은 그 A 몫: A1 미확정(coverage 감소), prefix 는 A1 을 포함한 텍스트가 확정된 시각(A4 commit)까지."""
    words = _abc(); K, d = 60, 4; rk = [ref_chunk(w["end_time"], d, K) for w in words]; hw = [w["text"] for w in words]
    c = cm.commit_latency_metrics(words, [cand(1, "A"), cand(4, "A")], hw, [_sem(35, 4)], rk, K)
    assert (c["committed"], c["lat"], c["prefix_lat"], c["prefix_uncovered"]) == (1, [0.38], [1.88, 0.38], 0)
    f = cm._finalize_commit(c, cm.LATE_THRESHOLDS_S, [0, 2, 5])
    assert f["coverage"] == 0.5 and f["prefix"]["coverage"] == 1.0 and f["prefix"]["recall_at"]["1"] == 0.5 and f["recall_at"]["1"] == 0.5

def test_commit_latency_deletion_insertion():
    words = _abc(); K, d = 60, 4; rk = [ref_chunk(w["end_time"], d, K) for w in words]
    dele = ["a", "b", "c", "d", "f"]                                                         # 가설이 e(A4) 를 빠뜨림
    c = cm.commit_latency_metrics(words, [cand(4, "A")], dele, [_sem(34, 3)], rk, K)         # e 의 ref_k(35) 전 방출 → d 뒤(p=3): 미확정
    assert c["committed"] == 0 and c["prefix_uncovered"] == 1
    c = cm.commit_latency_metrics(words, [cand(4, "A")], dele, [_sem(35, 3)], rk, K)         # ref_k(e) 이후 → e 뒤: 정확 확정
    assert (c["committed"], c["exact"], c["lat"]) == (1, 1, [0.38])
    ins = ["a", "b", "x", "c", "d", "e", "f"]                                                # b 뒤 삽입 x 뒤 commit = b 뒤(p=1)
    c = cm.commit_latency_metrics(words, [cand(1, "A"), cand(2, "B"), cand(3, "N"), cand(4, "A")], ins, [_sem(16, 2)], rk, K)
    assert (c["committed"], c["exact"], c["lat"], c["overshoot"], c["prefix_lat"], c["prefix_uncovered"]) == (1, 1, [0.36], [0], [0.36], 1)

def test_commit_latency_flush_is_censored():
    """flush 라운드(k ≥ K) 방출은 K·0.08 로 잘린 하한 — in_flush 로 따로 센다. 마지막 단어 A 는 last.
    excess 는 emit_time 과 같은 척도(flush = 청크 K−1): 44 − 41 = 3 = (0.6 − 이상 0.36)/0.08. after-text lag 는 원래 청크 차이(47 − 41 = 6)를 넣는다(스트림 끝 늦은 확정)."""
    words = _abc(); K, d = 45, 4; rk = [ref_chunk(w["end_time"], d, K) for w in words]; assert rk[5] == 41
    c = cm.commit_latency_metrics(words, [cand(5, "A")], [w["text"] for w in words], [_sem(47, 5)], rk, K)
    assert (c["lat"], c["excess"], c["in_flush"], c["last"], c["n_last"]) == ([0.6], [3], 1, [1], 1)             # 45·0.08 − 3.0
    assert abs(c["excess"][0] * cm.CHUNK_S - (c["lat"][0] - (emit_time(rk[5], K) - 3.0))) < 1e-9
    f = cm._finalize_commit(c, cm.LATE_THRESHOLDS_S, [0]); assert f["last"]["coverage"] == 1.0 and f["mid"]["n_A"] == 0 and f["in_flush"] == 1
    a = cm.after_text_metrics([_sem(47, 5)], [5], [[k, k] for k in rk], K, [0]); assert (a["n"], a["in_flush"], a["lag"], a["lag_A"]) == (1, 1, [6], [6])
    a = cm.after_text_metrics([_sem(47, 5)], [5], [[k, k] for k in rk[:5]] + [[46, 46]], K, [0])                    # 단어도 flush(라운드 46) — 원래 번호라 0 착시 없음
    assert (a["n"], a["in_flush"], a["lag"]) == (1, 1, [1])

def test_flush_commit_excess_matches_latency_scale():
    """목표 청크가 K−1 인 단어를 flush 첫 라운드(k=K)에서 확정: 시각은 oracle 과 같고(K·0.08) excess 도 0 — recall_within_chunks['0'] 과 recall_at 이 어긋나지 않는다.
    창 지표(timing)의 lag 는 청크 순서라 1 이다(창 정의는 그대로)."""
    w = mk_row("s", "English", [("a", 0.5, [], 1), ("b", 1.0, [], 1)], 1.2); d = 2; K = ref_chunk(1.0, d) + 1; k0 = ref_chunk(0.5, d)
    h = dict(emitted=[[k0, 1000], [K, 1010], [K, SEM]], words=["a", "b"], events=[dict(id=SEM, k=K, after=1, mid_word=False)], text="a b", word_k=[[k0, k0], [K, K]])
    m = score_stream(w, dict(candidates=[cand(1, "A")]), h, d, K)
    assert m["commit"]["excess"] == [0] and m["commit"]["lat"] == m["timing"]["ideal_lat"] and m["commit"]["in_flush"] == 1 and m["timing"]["late2"]["lag"] == [1]
    assert m["display"]["lag"] == [0, 0] and m["display"]["in_flush"] == 1 and m["after_text"]["lag"] == [0] and m["after_text"]["in_flush"] == 1
    f = finalize(m)["commit"]; assert f["recall_within_chunks"]["0"] == 1.0 and f["recall_at"]["1"] == 1.0

def test_commit_latency_streams_without_A_or_events_and_micro_aggregate():
    r1 = mk_row("s1", "English", [("go", 0.5, [], 1), ("now", 1.0, [], 1)], 2.0); K1 = r1["K"]
    r2 = mk_row("s2", "English", [("hi", 0.5, [], 1), ("there", 1.0, [], 1)], 2.0)
    r3 = dict(mk_row("s3", "English", [("x", 0.5, [], 1)], 1.0), words=[], tokens=[], text="")                   # 단어 없는 스트림
    wk = lambda n: [[5 + 3 * i, 5 + 3 * i] for i in range(n)]
    m1 = score_stream(r1, dict(candidates=[cand(0, "A"), cand(1, "A")]), _hyp(["go", "now"], [_sem(9, 0)], wk(2)), 2, K1)     # A 2 개 중 1 개 확정
    m2 = score_stream(r2, dict(candidates=[cand(0, "B")]), _hyp(["hi", "there"], [_sem(9, 0)], wk(2)), 2, r2["K"])          # A 없음 — B 확정만
    m3 = score_stream(r3, dict(candidates=[]), _hyp([], [], []), 2, r3["K"])
    m4 = score_stream(r1, dict(candidates=[cand(1, "A")]), _hyp(["go", "now"], [], wk(2)), 2, K1)                             # 이벤트 없음
    assert (m1["commit"]["n_A"], m1["commit"]["committed"], m1["commit"]["lat"], m2["commit"]["streams_no_A"], m2["commit"]["B"]["committed"]) == (2, 1, [0.3], 1, 1)
    assert m3["commit"]["n_A"] == 0 and m3["commit"]["streams_no_A"] == 1 and m3["display"]["n_ref"] == 0 and m3["display"]["no_text"] == 1 and m3["display"]["first_any_lat"] == []
    assert m4["commit"]["committed"] == 0 and m4["commit"]["prefix_uncovered"] == 1 and m4["after_text"]["n"] == 0
    f2 = finalize(m2)["commit"]; assert f2["coverage"] is None and f2["latency_s"]["n"] == 0 and f2["recall_at"]["1"] is None
    o = aggregate([dict(id=i, lang="English", set="t", metrics=m) for i, m in enumerate((m1, m2, m3, m4))])["overall"]["commit"]
    assert (o["n_A"], o["committed"], o["coverage"], o["streams_no_A"], o["B"]["rate"]) == (3, 1, 1 / 3, 2, 1.0)        # micro: Σ확정 / ΣA

def _rand_row(rng):
    """단어 1–9 개, 단어당 토큰 1–3 개(첫 토큰 시각 ≤ 단어 끝 = 마지막 토큰 시각)."""
    ends = sorted({round(rng.uniform(0.1, 6.0), 2) for _ in range(rng.randint(1, 9))}); toks, words = [], []
    for i, t in enumerate(ends):
        lo = ends[i - 1] if i else 0.0; a = len(toks)
        toks += [[1000 + 10 * i + j, x] for j, x in enumerate(sorted(round(rng.uniform(lo, t), 2) for _ in range(rng.randint(0, 2))) + [t])]
        words.append(dict(i=i, text=f"w{i}", a=a, b=len(toks), end_time=t, tags=[]))
    return dict(id="r", set="rand", lang="English", words=words, tokens=toks, text=" ".join(w["text"] for w in words), K=int(round((ends[-1] + 0.3) / 0.08)),
                duration_s=round(ends[-1] + 0.3, 3), segments=[])

def test_oracle_latency_properties_random():
    """oracle: commit 지연 = δ 이상 지연(timing.ideal_lat), 전부 정확 확정·excess 0, 표시 지연 = 이상 지연·lag 0, 첫 토큰 = 첫 토큰의 이상 지연, after-text lag 0."""
    rng = random.Random(7)
    for _ in range(400):
        w = _rand_row(rng); n = len(w["words"]); ends = [x["end_time"] for x in w["words"]]
        lab = dict(candidates=[cand(i, rng.choice("ABN")) for i in rng.sample(range(n), rng.randint(0, n))])
        d = rng.choice([2, 3, 4, 6, 8]); K0 = w["K"]
        K = rng.choice([K0, train_pad_K(K0, ends[-1], (2, 3, 4, 6, 8), False), max(1, K0 - rng.randint(0, 5))])            # 늘림·원래·잘림(flush)
        h = oracle_hyp(w, lab, d, K); m = score_stream(w, lab, h, d, K); c, dp, at = m["commit"], m["display"], m["after_text"]
        assert sorted(c["lat"]) == sorted(m["timing"]["ideal_lat"]) and c["committed"] == c["exact"] == c["n_A"] == m["text"]["n_A"]
        assert set(c["excess"]) <= {0} and set(c["overshoot"]) <= {0} and c["prefix_lat"] == c["lat"] and c["prefix_uncovered"] == 0
        assert dp["n_aligned"] == n and set(dp["lag"]) <= {0} and dp["lat"] == [round(emit_time(ref_chunk(e, d, K), K) - e, 4) for e in ends]
        assert dp["first_lat"] == [round(emit_time(ref_chunk(w["tokens"][0][1], d, K), K) - ends[0], 4)] and dp["first_skip"] == 0
        assert set(at["lag"]) <= {0} and at["n"] == c["n_A"] and at["lag_A"] == at["lag"] and at["no_word"] == at["no_text"] == 0 and at["in_flush"] == c["in_flush"]
        f = finalize(m); assert f["commit"]["latency_s"] == f["timing"]["ideal_latency_s"] and f["commit"]["recall_within_chunks"]["0"] == (1.0 if n and c["n_A"] else None)

def test_commit_identities_random_asr_errors():
    """ASR 오류(치환·삭제·삽입)가 있어도: 유효 commit 위치는 방출 순서에 단조 → commit.exact = text.correct, B/N 확정 = ambiguous/N_hit, coverage·exact_rate = text.recall."""
    rng = random.Random(5)
    for _ in range(400):
        n = rng.randint(1, 8); ref = [rng.choice("abcd") for _ in range(n)]; ends = [round(0.3 * (i + 1), 2) for i in range(n)]
        words = [dict(i=i, text=x, end_time=e, tags=[]) for i, (x, e) in enumerate(zip(ref, ends))]
        hyp = [x if rng.random() < 0.7 else rng.choice("abcdx") for x in ref if rng.random() < 0.85] + ([rng.choice("abx")] if rng.random() < 0.3 else [])
        K = int(ends[-1] / 0.08) + 12; d = rng.choice([2, 4]); rk = [ref_chunk(e, d, K) for e in ends]; ev, k = [], 0
        for j in range(-1, len(hyp)):
            k += rng.randint(0, 3)
            for _ in range(rng.choice([0, 0, 1, 2])): ev.append(_sem(k, j, 0 <= j < len(hyp) - 1 and rng.random() < 0.15)); k += rng.randint(0, 1)
        cands = [cand(i, rng.choice("ABN")) for i in rng.sample(range(n), rng.randint(0, n))]
        pos = map_events(ref, hyp, ev, rk, K); el = [p for e, p in zip(ev, pos) if not e["mid_word"] and p >= 0]; assert el == sorted(el)
        c = cm.commit_latency_metrics(words, cands, hyp, ev, rk, K); x = text_position_metrics(words, cands, hyp, ev, rk, K)
        assert c["exact"] == x["correct"] and c["B"]["committed"] == x["ambiguous"] and c["N"]["committed"] == x["N_hit"] and c["n_A"] == x["n_A"]
        f = cm._finalize_commit(c, (1.0,), [0])
        if c["committed"]: assert abs(f["coverage"] * f["exact_rate"] - x["correct"] / x["n_A"]) < 1e-12

def test_excess_and_display_lag_on_latency_scale_random():
    """flush 가 잦은 무작위 스트림(K 를 마지막 단어보다 짧게도): 확정 A 마다 excess·0.08 = lat − 이상 지연, 참조 단어마다 display lag·0.08 = lat − 이상 표시 지연."""
    rng = random.Random(13)
    for _ in range(500):
        n = rng.randint(1, 6); t, ends = 0.0, []
        for _ in range(n): t = round(t + rng.uniform(0.1, 0.8), 2); ends.append(t)
        d = rng.choice([2, 4, 8]); K = max(1, int(ends[-1] / 0.08) + rng.choice([-4, 0, 2, 9])); rk = [ref_chunk(e, d, K) for e in ends]
        w = mk_row("s", "English", [(f"w{i}", e, [], 1) for i, e in enumerate(ends)], ends[-1] + 0.5); hw = [f"w{i}" for i in range(n)]
        a = rng.randrange(n); k = rk[a] + rng.randint(-2, 6); wk = [[x, x + rng.randint(0, 3)] for x in rk]
        m = score_stream(w, dict(candidates=[cand(a, "A")]), _hyp(hw, [_sem(k, a)], wk), d, K)
        ideal = emit_time(rk[a], K) - ends[a]
        for lat, ex in zip(m["commit"]["lat"], m["commit"]["excess"]): assert abs(ex * cm.CHUNK_S - (lat - ideal)) < 2e-4
        for i, (lat, lg) in enumerate(zip(m["display"]["lat"], m["display"]["lag"])): assert abs(lg * cm.CHUNK_S - (lat - (emit_time(rk[i], K) - ends[i]))) < 2e-4
        assert m["after_text"]["lag"] == [k - wk[a][1]] and m["after_text"]["in_flush"] == int(k >= K)

def test_after_text_no_word_vs_no_text():
    """lag 에서 빼는 두 경우를 나눠 센다: 첫 참조 단어 앞 삽입('uh') 뒤 commit = no_word(p < 0, text.cat.no_word 와 같음),
    첫 참조 단어가 빠진 가설에서 텍스트 전 commit 이 빠진 단어 뒤로 사상 = no_text(유효 commit — A 를 확정한다)."""
    words = _abc(2); K, d = 60, 4; rk = [ref_chunk(w["end_time"], d, K) for w in words]; assert rk == [10, 16]
    w = mk_row("s", "English", [("a", 0.5, [], 1), ("b", 1.0, [], 1)], 1.5)
    m = score_stream(w, dict(candidates=[cand(0, "A")]), _hyp(["uh", "a", "b"], [_sem(3, 0)], [[2, 2], [10, 10], [16, 16]]), d, K)
    at = m["after_text"]; assert (at["no_word"], at["no_text"], at["n"]) == (1, 0, 0) and m["text"]["cat"]["no_word"] == 1 and m["commit"]["committed"] == 0
    m = score_stream(w, dict(candidates=[cand(0, "A")]), _hyp(["b"], [_sem(10, -1)], [[16, 16]]), d, K)
    at = m["after_text"]; assert (at["no_word"], at["no_text"], at["n"]) == (0, 1, 0) and m["commit"]["committed"] == m["commit"]["exact"] == m["text"]["correct"] == 1
    f = finalize(m)["after_text"]; assert (f["no_word"], f["no_text"], f["n"]) == (0, 1, 0)

def test_out_of_range_candidates_rejected():
    """단어 범위 밖 후보(after_word ∉ [0, n))는 학습처럼 거부 — text 와 commit 의 n_A 가 어긋나 coverage·exact_rate = text.recall 이 깨지기 때문.
    score_stream 은 ValueError, CLI 는 디코드 전(run)·재채점 전(build_report)에 멈춘다."""
    w = mk_row("s", "English", [("a", 0.5, [], 1), ("b", 1.0, [], 1)], 1.5); good = dict(candidates=[cand(1, "A")])
    assert cm.after_word_range_errors(w["words"], [cand(1, "A"), cand(2, "A"), cand(-1, "N"), cand(5, "B")]) == [-1, 2, 5]
    with pytest.raises(ValueError, match="after_word_range"): score_stream(w, dict(candidates=[cand(1, "A"), cand(2, "A")]), oracle_hyp(w, good, 2, w["K"]), 2, w["K"])
    cli = _cli(); words, labels = _toy_set()
    with tempfile.TemporaryDirectory() as td:
        wp, lp, out = Path(td) / "w.jsonl", Path(td) / "l.jsonl", Path(td) / "report.json"; _dump(wp, words); _dump(lp, labels)
        args = ["--words", str(wp), "--labels", str(lp), "--delay", "4", "--oracle", "--out", str(out)]; cli.main(args)
        bad = dict(labels, en1=dict(labels["en1"], candidates=labels["en1"]["candidates"] + [cand(6, "A")])); _dump(lp, bad)
        with pytest.raises(AssertionError, match="after_word_range"): cli.main(["--words", str(wp), "--labels", str(lp), "--score-only", "--out", str(out)])
        with pytest.raises(AssertionError, match="after_word_range"): cli.main(args[:-1] + [str(Path(td) / "other.json")])
        assert not (Path(td) / "other.streams.jsonl").exists()                                   # 디코드 전에 멈춤

def test_late_threshold_keys_must_be_distinct():
    """보고서 키는 %g(유효숫자 6 자리) — 0.32 와 0.3200001 은 같은 키 '0.32' 라 항목이 조용히 덮어써지므로 거부한다."""
    cli = _cli(); base = ["--oracle", "--words", "w", "--labels", "l", "--out", "o", "--late-thresholds"]
    with pytest.raises(AssertionError, match="late-thresholds"): cli.parse_args(base + ["0.32", "0.3200001"])
    assert cli.parse_args(base + ["0.32", "0.33"]).late_thresholds == [0.32, 0.33]

def test_commit_latency_equals_window_latency_for_exact_commits():
    """가설 = 참조이고 excess ∈ [0, 2] 인 정확 commit 이면 창 지표(late2)의 지연과 같다(A 사이 ≥ 4 청크라 창이 겹치지 않음)."""
    rng = random.Random(11)
    for _ in range(300):
        n = rng.randint(1, 8); t, ends = 0.0, []
        for _ in range(n): t = round(t + rng.uniform(0.4, 1.0), 2); ends.append(t)
        w = mk_row("s", "English", [(f"w{i}", e, [], 1) for i, e in enumerate(ends)], ends[-1] + 1.0); K = w["K"] + 10; d = rng.choice([2, 4, 6])
        A = sorted(rng.sample(range(n), rng.randint(0, n))); rk = [ref_chunk(e, d, K) for e in ends]
        m = score_stream(w, dict(candidates=[cand(a, "A") for a in A]), _hyp([f"w{i}" for i in range(n)], [_sem(rk[a] + rng.randint(0, 2), a) for a in A], [[k, k] for k in rk]), d, K)
        assert sorted(m["timing"]["late2"]["lat"]) == sorted(m["commit"]["lat"]) and m["commit"]["committed"] == len(A) == m["timing"]["late2"]["hits"]

def test_display_latency_alignment_example():
    """치환·삭제·삽입이 섞인 가설: 정렬된 참조 단어만 표시 지연(삭제는 coverage), 첫 토큰 = 첫 정렬 쌍(앞 삽입 'uh' 는 first_any 에만)."""
    ref = "the cat sat on the mat".split(); words = [dict(i=i, text=w, end_time=round(0.3 * (i + 1), 2), tags=[]) for i, w in enumerate(ref)]; K, d = 40, 2
    assert [ref_chunk(w["end_time"], d, K) for w in words] == [5, 9, 13, 17, 20, 24]
    hyp = "uh the bat sat the mat".split(); wk = [[1, 1], [5, 5], [8, 9], [12, 12], [20, 20], [24, 25]]
    assert align_pairs([cm.norm_word(w) for w in ref], [cm.norm_word(h) for h in hyp]) == [(None, 0), (0, 1), (1, 2), (2, 3), (3, None), (4, 4), (5, 5)]
    x = cm.display_metrics(words, hyp, wk, K, d)
    assert (x["n_aligned"], x["n_exact"], x["n_del"], x["n_ins"], x["in_flush"]) == (5, 4, 1, 1, 0)
    assert x["lat"] == [0.18, 0.2, 0.14, 0.18, 0.28] and x["lag"] == [0, 0, -1, 0, 1] and x["exact"] == [1, 0, 1, 1, 1]
    assert x["first_lat"] == [0.18] and x["first_any_lat"] == [-0.14] and x["first_skip"] == 0 and x["no_text"] == 0
    f = cm._finalize_display(x, cm.LATE_THRESHOLDS_S)
    assert f["coverage"] == 5 / 6 and f["exact_rate"] == 0.8 and f["latency_exact_s"] == cm.latency_stats([0.18, 0.14, 0.18, 0.28]) and f["late_rate"]["0.32"] == 0.0
    assert f["lag_chunks"]["max"] == 1 and f["first_token_s"]["n"] == 1 and f["first_any_s"]["p50"] == -0.14
    y = cm.display_metrics(words, ref[1:], [[9, 9], [13, 13], [17, 17], [20, 20], [24, 24]], K, d)                   # 첫 참조 단어 삭제 → 첫 토큰 지연에서 뺀다
    assert y["first_skip"] == 1 and y["first_lat"] == [] and y["first_any_lat"] == [round(emit_time(9, K) - 0.3, 4)] and y["n_del"] == 1
    with pytest.raises(ValueError): cm.display_metrics(words, hyp, wk[:-1], K, d)

def test_display_missing_without_word_k_and_mixed_aggregate():
    """옛 행(hyp.word_k 없음): display·after_text 는 missing, commit 은 저장된 이벤트만으로 같다. 섞어 합산해도 missing 이 드러난다."""
    w = mk_row("s", "English", [("go", 0.5, [], 1), ("now", 1.0, [], 1)], 2.0); lab = dict(candidates=[cand(1, "A")])
    h = oracle_hyp(w, lab, 2, w["K"]); old = {k: v for k, v in h.items() if k != "word_k"}
    mo, mn = score_stream(w, lab, old, 2, w["K"]), score_stream(w, lab, h, 2, w["K"])
    assert mo["display"] == dict(streams=0, missing=1) == mo["after_text"] and mo["commit"] == mn["commit"] and mn["display"]["streams"] == 1
    o = aggregate([dict(id="a", lang="English", metrics=mo), dict(id="b", lang="English", metrics=mn)])["overall"]
    assert (o["display"]["streams"], o["display"]["missing"], o["display"]["n_ref"], o["after_text"]["missing"], o["commit"]["n_A"]) == (1, 1, 2, 1, 2)
    only_old = aggregate([dict(id="a", lang="English", metrics=mo)])["overall"]
    assert only_old["display"]["streams"] == 0 and only_old["display"]["missing"] == 1 and only_old["display"]["latency_s"]["n"] == 0 and only_old["display"]["coverage"] is None
    assert only_old["after_text"]["n"] == 0 and only_old["after_text"]["zero_rate"] is None

def test_pr_point_latency_keys_and_old_summary():
    words, labels = _toy_set(); recs = []
    for sid, w in words.items():
        K = eval_audio_len(w, (2, 3, 4, 6), True)[1]; h = oracle_hyp(w, labels[sid], 4, K)
        recs.append(dict(id=sid, lang=w["lang"], set=w["set"], metrics=score_stream(w, labels[sid], h, 4, K)))
    s = aggregate(recs)["overall"]; p = cm.pr_point(s)
    assert p["commit_coverage"] == 1.0 and p["commit_exact_rate"] == 1.0 and p["commit_latency_p90_s"] == s["commit"]["latency_s"]["p90"] and p["commit_excess_p90_chunks"] == 0
    assert {"commit_recall_at_0.32s", "commit_recall_at_0.64s", "commit_recall_at_1s", "prefix_latency_p90_s", "display_latency_p99_s", "first_token_p90_s"} <= set(p)
    assert p["display_coverage"] == 1.0 and p["first_token_p50_s"] == s["display"]["first_token_s"]["p50"] and p["after_text_zero_rate"] == 1.0
    old = {k: v for k, v in s.items() if k not in ("commit", "display", "after_text")}; q = cm.pr_point(old)           # 옛 요약: 새 키는 None, 예외 없음
    assert q["commit_coverage"] is None and q["display_latency_p90_s"] is None and q["commit_recall_at_1s"] is None and q["text_precision"] == p["text_precision"]
    assert "commit_recall_at_0.5s" in cm.pr_point(old, late_thresholds=(0.5,))

def test_cli_latency_report_thresholds_table_and_legacy_score_only(capsys):
    """CLI: 보고서에 창 없는 지연(oracle = 이상 지연)·임계값 기록·[latency] 표, --late-thresholds 로 키 변경, word_k 없는 옛 jsonl 도 --score-only 로 같은 보고서."""
    cli = _cli(); words, labels = _toy_set()
    with tempfile.TemporaryDirectory() as td:
        wp, lp, out = Path(td) / "w.jsonl", Path(td) / "l.jsonl", Path(td) / "report.json"; _dump(wp, words); _dump(lp, labels)
        rep = cli.main(["--words", str(wp), "--labels", str(lp), "--delay", "4", "--oracle", "--out", str(out)]); o = rep["configs"]["oracle"]["overall"]
        assert o["commit"]["latency_s"] == o["timing"]["ideal_latency_s"] and o["commit"]["coverage"] == 1.0 and o["display"]["lag_chunks"]["max"] == 0
        assert rep["scoring"]["late_thresholds_s"] == [0.32, 0.64, 1.0] and rep["scoring"]["word_k"]["legacy"] == 0 and {p["commit_coverage"] for p in rep["pr_curve"]["oracle"]} == {1.0}
        lines = capsys.readouterr().out.splitlines(); i = next(i for i, l in enumerate(lines) if l.startswith("[latency]"))
        assert "r@0.32" in lines[i + 1] and [l.split()[:2] for l in lines[i + 2:i + 5]] == [["oracle", "all"], ["oracle", "English"], ["oracle", "Korean"]]
        r2 = cli.main(["--words", str(wp), "--labels", str(lp), "--score-only", "--late-thresholds", "0.5", "1.0", "--out", str(out)])
        assert set(r2["configs"]["oracle"]["overall"]["commit"]["recall_at"]) == {"0.5", "1"} and "commit_recall_at_0.5s" in r2["pr_curve"]["oracle"][0]
        with pytest.raises(AssertionError, match="late-thresholds"): cli.parse_args(["--oracle", "--words", "w", "--labels", "l", "--out", "o", "--late-thresholds", "1.0", "0.5"])
        streams = Path(td) / "report.streams.jsonl"; recs = [json.loads(l) for l in streams.read_text().splitlines()]
        for r in recs:                                                                          # 옛 코드가 쓴 행: word_k·새 지표 블록 없음
            del r["hyp"]["word_k"]
            for k in ("commit", "display", "after_text"): (r.get("metrics") or {}).pop(k, None)        # oracle 행은 metrics 를 안 쓴다
        streams.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        r3 = cli.main(["--words", str(wp), "--labels", str(lp), "--score-only", "--out", str(out)])
        assert r3["scoring"]["word_k"] == dict(legacy=2, oracle=2, tokenizer=0, missing=0, src=None, error=None) and r3["configs"] == rep["configs"] and r3["pr_curve"] == rep["pr_curve"]
        assert "word_k" not in json.loads(streams.read_text().splitlines()[0])["hyp"]                                    # backfill 은 메모리에서만(파일 불변)

def test_cli_score_only_backfills_word_k_with_tokenizer(monkeypatch):
    """모델 행의 옛 jsonl: tokenizer 로 방출을 다시 나눠 word_k 를 채우면 원래 보고서와 같고, tokenizer 가 없으면 display missing(commit 은 그대로), 어긋나면 멈춘다."""
    cli = _cli(); orig = cli._backfill_fns; _patch_fake_run(monkeypatch, cli); tok = cli.load_sem_model()[1]; words, labels = _toy_set()
    assert cli._backfill_fns("oracle")[0] is None and cli._backfill_fns(None)[0] is None and "none" in cli._backfill_fns("none")[1]
    with tempfile.TemporaryDirectory() as td:
        wp, lp, out, md = Path(td) / "w.jsonl", Path(td) / "l.jsonl", Path(td) / "report.json", Path(td) / "model"; md.mkdir(); _dump(wp, words); _dump(lp, labels)
        assert cli._backfill_fns(str(md))[0] is None                                          # tokenizer 파일·thinker 경로 없음 → (None, 이유)
        rep = cli.main(["--model", str(md), "--words", str(wp), "--labels", str(lp), "--sem-bias", "0", "1", "--out", str(out)])
        streams = Path(td) / "report.streams.jsonl"; recs = [json.loads(l) for l in streams.read_text().splitlines()]; assert len(recs) == 4
        for r in recs: del r["hyp"]["word_k"]
        streams.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        so = ["--model", str(md), "--words", str(wp), "--labels", str(lp), "--score-only", "--out", str(out)]
        monkeypatch.setattr(cli, "_backfill_fns", lambda path: (None, "없음"))
        r1 = cli.main(so); o = r1["configs"]["bias=1"]["overall"]
        assert (o["display"]["streams"], o["display"]["missing"], o["after_text"]["missing"]) == (0, 2, 2) and o["commit"] == rep["configs"]["bias=1"]["overall"]["commit"]
        assert r1["scoring"]["word_k"]["missing"] == 4 and r1["scoring"]["word_k"]["error"] == "없음"
        seen = []
        monkeypatch.setattr(cli, "_backfill_fns", lambda path: (seen.append(path), ((lambda s, t: cli.tokenizer_fns(tok, s, t)), None))[1])
        r2 = cli.main(so); assert r2["configs"] == rep["configs"] and r2["pr_curve"] == rep["pr_curve"] and r2["scoring"]["word_k"]["tokenizer"] == 4
        assert seen == [str(md)] and r2["scoring"]["word_k"]["src"] == str(md)                  # 기본 tokenizer = 디코드 설정의 model, 한 번만 읽음
        cli.main(so + ["--tokenizer", "/elsewhere/tok"]); assert seen[-1] == "/elsewhere/tok"
        monkeypatch.setattr(cli, "_backfill_fns", lambda path: ((lambda s, t: (lambda x: x not in (s, t), lambda x: False, lambda ids: "la" * len(ids))), None))
        with pytest.raises(AssertionError, match="word_k backfill"): cli.main(so)
        monkeypatch.setattr(cli, "_backfill_fns", orig)                                         # --tokenizer none: backfill 끔 → display·after_text missing, commit 그대로
        r4 = cli.main(so + ["--tokenizer", "none"]); o4 = r4["configs"]["bias=1"]["overall"]
        assert r4["scoring"]["word_k"]["missing"] == 4 and "none" in r4["scoring"]["word_k"]["error"] and o4["commit"] == rep["configs"]["bias=1"]["overall"]["commit"]
        assert (o4["display"]["missing"], o4["after_text"]["missing"]) == (2, 2)

def test_backfill_tokenizer_load_any_exception_is_missing(monkeypatch, tmp_path):
    """tokenizer 로드가 OSError/ValueError 밖의 예외(tokenizers 바인딩의 Exception·KeyError 등)를 내도 --score-only 가 멈추지 않고 (None, 이유) — missing 으로 센다."""
    cli = _cli(); (tmp_path / "tokenizer_config.json").write_text("{}")
    import transformers
    def boom(*a, **k): raise KeyError("model_type")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom)
    fns, why = cli._backfill_fns(str(tmp_path)); assert fns is None and why.startswith("KeyError")
    def boom2(*a, **k): raise Exception("data did not match any variant of untagged enum")
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", boom2)
    fns, why = cli._backfill_fns(str(tmp_path)); assert fns is None and why.startswith("Exception")

"""vapasr/hf/twopass.py — <SEM_END> 트리거 2-pass 세그먼트화·합성·지연(모델 없음)."""
import pytest

from vapasr.hf.commit_metrics import CHUNK_S, SEM_END_ID, emit_time
from vapasr.hf.twopass import (build_segments, compose, display_latencies, final_latencies, latency_summary, segment_stats, unfinalized_tail,
                               valid_commits)

D, K = 4, 100                                                                              # δ4, 8 s 패딩 오디오
T_EOS = 7.5
WORDS = ["a", "b", "c", "d", "e", "f"]
WK = [[10, 10], [20, 21], [30, 30], [40, 41], [50, 50], [60, 62]]


def sem(k, after, mid=False): return dict(id=SEM_END_ID, k=k, after=after, mid_word=mid)


def hyp(events, words=WORDS, wk=WK): return dict(words=list(words), events=list(events), word_k=[list(x) for x in wk])


def _check_contiguous(segs, t_eos=T_EOS):
    assert segs[0]["start"] == 0.0
    for a, b in zip(segs, segs[1:]):
        assert a["end"] == pytest.approx(b["start"]) and a["w1"] == b["w0"]              # 오디오·1차 단어가 빈틈·겹침 없이 이어진다
        assert a["t_trig"] <= b["t_trig"] + 1e-9                                           # 확정 시각은 방출 순
    for s in segs:
        assert s["start"] < s["end"] <= t_eos + 1e-9 and s["w0"] <= s["w1"]
        assert s["kind"] == "eos" or s["end"] <= s["t_trig"] + 1e-9                        # 인과: 절단점은 트리거 시각보다 늦지 않다


def test_no_commit_gives_single_eos_segment():
    segs = build_segments(hyp([]), D, K, T_EOS)
    assert segs == [dict(start=0.0, end=T_EOS, t_trig=T_EOS, kind="eos", w0=0, w1=6, k_sem=None, held=0)]


def test_commits_cut_at_word_end_upper_bound_and_mid_word_ignored():
    h = hyp([sem(21, 1), sem(41, 3), sem(50, 4, mid=True)])
    segs = build_segments(h, D, K, T_EOS)
    assert [s["kind"] for s in segs] == ["sem", "sem", "eos"]
    assert segs[0]["end"] == pytest.approx((21 - D + 1) * CHUNK_S) and segs[0]["t_trig"] == emit_time(21, K)   # cut = (k_last − δ + 1)·0.08
    assert segs[1]["end"] == pytest.approx((41 - D + 1) * CHUNK_S) and (segs[1]["w0"], segs[1]["w1"]) == (2, 4)
    assert (segs[2]["w0"], segs[2]["w1"], segs[2]["t_trig"]) == (4, 6, T_EOS)            # mid_word commit 은 트리거가 아니다
    _check_contiguous(segs)
    st = segment_stats(h, segs, K, T_EOS)
    assert st == dict(commits=2, after_eos=0, held=0, seg_sem=2, seg_eos=1, mid_word=1)


def test_commit_at_last_word_leaves_no_silence_segment():
    h = hyp([sem(41, 3), sem(62, 5)])
    segs = build_segments(h, D, K, T_EOS)
    assert [s["kind"] for s in segs] == ["sem", "sem"] and segs[-1]["w1"] == len(WORDS)  # 무음 꼬리만 남으면 EOS 세그먼트를 만들지 않는다
    _check_contiguous(segs)
    ends = [0.7, 3.0, 4.7, 7.2]                                                            # 7.2 는 마지막 절단점(4.72) 뒤 — 확정되지 않은 꼬리
    lat = final_latencies(ends, segs)
    assert lat[0] == pytest.approx(emit_time(41, K) - 0.7) and lat[2] == pytest.approx(emit_time(62, K) - 4.7) and lat[3] is None
    assert unfinalized_tail(ends, segs) == [7.2]


def test_short_or_one_word_segment_is_held_and_merged():
    h = hyp([sem(12, 0), sem(41, 3)], wk=[[10, 12]] + WK[1:])                            # 첫 commit: cut 0.72 s < min_seg 0.8 → 보류
    segs = build_segments(h, D, K, T_EOS)
    assert [s["kind"] for s in segs] == ["sem", "eos"] and (segs[0]["w0"], segs[0]["w1"], segs[0]["held"]) == (0, 4, 1)
    _check_contiguous(segs)
    segs2 = build_segments(hyp([sem(21, 1), sem(30, 2)]), D, K, T_EOS, min_words=2)    # 두 번째 commit: 새 1차 단어 1 개 → 보류, EOS 와 합친다
    assert [(s["w0"], s["w1"], s["held"]) for s in segs2] == [(0, 2, 0), (2, 6, 1)]
    assert build_segments(hyp([sem(12, 0)], wk=[[10, 12]] + WK[1:]), D, K, T_EOS, min_seg_s=0.0, min_words=1)[0]["end"] == pytest.approx(0.72)


def test_commit_after_eos_is_dropped_in_favour_of_eos():
    h = hyp([sem(41, 3), sem(98, 5)])                                                     # emit 7.92 s > t_eos 7.5 → EOS 가 먼저 확정
    segs = build_segments(h, D, K, T_EOS)
    assert [s["kind"] for s in segs] == ["sem", "eos"] and segs[-1]["w1"] == 6
    assert segment_stats(h, segs, K, T_EOS)["after_eos"] == 1


def test_cut_clamped_to_eos_for_late_flush_commit():
    k0 = 94; t_eos = k0 * CHUNK_S                                                         # 패딩 없는 스트림(K = K0): flush 라운드 방출 시각 = 오디오 끝
    wk = WK[:5] + [[60, 98]]                                                              # 마지막 단어 토큰이 flush 5 번째 라운드 — (98−δ+1)·0.08 = 7.6 > t_eos
    segs = build_segments(hyp([sem(98, 5)], wk=wk), D, k0, t_eos)
    assert len(segs) == 1 and segs[0]["kind"] == "sem" and segs[0]["end"] == pytest.approx(t_eos) and segs[0]["t_trig"] == pytest.approx(t_eos)
    assert segs[0]["w1"] == 6
    _check_contiguous(segs, t_eos)


def test_empty_hypothesis_and_compose_and_display():
    segs = build_segments(hyp([], words=[], wk=[]), D, K, T_EOS)
    assert segs == [dict(start=0.0, end=T_EOS, t_trig=T_EOS, kind="eos", w0=0, w1=0, k_sem=None, held=0)]
    assert compose([" Hello there. ", "", None, "general Kenobi"]) == "Hello there. general Kenobi"
    lat = display_latencies(["A", "b", "x"], [0.5, 1.5, 2.0], ["a", "B"], [[10, 10], [20, 21]], K)
    assert lat[0] == pytest.approx(emit_time(10, K) - 0.5) and lat[1] == pytest.approx(emit_time(21, K) - 1.5) and lat[2] is None
    s = latency_summary([0.1, None, 0.3])
    assert s["n"] == 2 and s["none"] == 1 and s["p50"] == pytest.approx(0.2)


def test_word_k_required():
    with pytest.raises(AssertionError, match="word_k"):
        build_segments(dict(words=["a"], events=[], word_k=None), D, K, T_EOS)
    assert valid_commits([sem(5, -1), sem(6, 0, mid=True), sem(7, 0), dict(id=1, k=8, after=0)]) == [sem(7, 0)]


def test_twopass_eval_end_to_end_with_fake_decoder(tmp_path, monkeypatch):
    """experiments/semcommit_twopass_eval.py: 1차 스트림 행 + words 행 → 세그먼트 → (가짜) 2차 디코드 → 합성·채점·지연·요약·이어하기(모델·오디오 없음)."""
    import importlib.util, json, sys
    from pathlib import Path
    import numpy as np
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("semcommit_twopass_eval", root / "experiments/semcommit_twopass_eval.py")
    tp = importlib.util.module_from_spec(spec); spec.loader.exec_module(tp)
    se = tp._semcommit_eval()
    ref = ["i", "saw", "a", "cat", "it", "ran", "away"]; ends = [0.4, 0.9, 1.1, 1.6, 3.2, 3.6, 4.1]
    wrow = dict(id="s1", set="toy", lang="English", K=60, duration_s=4.8, text=" ".join(ref), tokens=[[1, e] for e in ends],
                segments=[dict(path="x.flac", offset_s=0.0, silence_before_s=0.0, dur_s=4.8, utt_id="u", raw_text="")],
                words=[dict(i=i, text=w, a=i, b=i + 1, end_time=e, seg=0, tags=[]) for i, (w, e) in enumerate(zip(ref, ends))])
    D4 = 4; kk = [int(e / CHUNK_S) + D4 for e in ends]                                     # 1차 = 참조 그대로(단어 끝 청크 + δ), 'cat' 뒤 commit
    hyp_words = list(ref); wk = [[k, k] for k in kk]
    events = [dict(id=SEM_END_ID, k=kk[3], after=3, mid_word=False)]
    emitted = [[k, 100 + i] for i, k in enumerate(kk)]; emitted.insert(4, [kk[3], SEM_END_ID])
    rec = dict(id="s1", set="toy", lang="English", config=dict(name="bias=0", mode="bias", value=0.0), delta=D4, K=66, K0=60, duration_s=5.28,
               event_ids=[SEM_END_ID, None], words_digest=se.row_digest(wrow), emitted=emitted, hyp=dict(words=hyp_words, events=events, text=" ".join(ref), word_k=wk))
    ev_dir, w_dir, out_dir = tmp_path / "ev", tmp_path / "w", tmp_path / "out"
    for d in (ev_dir, w_dir): d.mkdir()
    (w_dir / "words-toy.jsonl").write_text(json.dumps(wrow) + "\n")
    (ev_dir / "toy-d4.streams.jsonl").write_text(json.dumps(rec) + "\n")
    (ev_dir / "toy-d4.streams.jsonl.config.json").write_text(json.dumps(dict(model="oracle")))
    monkeypatch.setattr(se, "load_stream_audio", lambda w, dur, remaps: np.zeros(int(round(dur * 16000)), np.float32))
    calls = []
    class Fake:
        def run(self, wavs, langs, ctxs):
            calls.append((len(wavs), list(ctxs), [round(len(x) / 16000, 2) for x in wavs]))
            c = round((kk[3] - D4 + 1) * CHUNK_S, 2)                                      # 'cat' 끝 청크 상한 = 1.68 s
            texts = {c: "I saw a cat.", round(4.8 - c, 2): "It ran away!", 4.8: "i saw a cat it ran away"}   # 세그먼트 길이 → 텍스트
            return [texts.get(round(len(x) / 16000, 2), "?") for x in wavs], [0.1] * len(wavs)
    a = tp.argparse.Namespace(eval_dir=str(ev_dir), words_dir=str(w_dir), sets=["toy"], delay=4, config="bias=0", decoder=str(tmp_path / "Qwen3-ASR-fake"),
                              modes=["iso", "text", "s5"], out_dir=str(out_dir), min_seg=0.8, min_words=2, context_chars=600, path_remap=[], tokenizer=None, io_workers=2,
                              max_new_tokens=512, batch_size=4)
    monkeypatch.setattr(tp, "_semcommit_eval", lambda: se)
    out = tp.run_set(a, se, Fake(), "toy")
    rows = {r["mode"]: r for r in se.read_jsonl(out)}
    cut = (kk[3] - D4 + 1) * CHUNK_S                                                      # 'cat' 끝 청크 상한 = 1.68 s
    assert [round(s["end"], 2) for s in rows["iso"]["segs"]] == [round(cut, 2), 4.8] and rows["iso"]["segs"][-1]["kind"] == "eos"
    assert rows["iso"]["final_text"] == rows["text"]["final_text"] == "I saw a cat. It ran away!"
    assert rows["iso"]["asr_final"]["wer"]["errors"] == 0 and rows["s5"]["asr_final"]["wer"]["errors"] == 0
    assert calls[1][1] == ["", ""] or calls[1][1] == [""]                                 # iso: context 없음
    text_calls = [c for c in calls if any(c[1])]; assert text_calls and text_calls[0][1] == ["I saw a cat."]   # text: 앞 세그먼트 최종 텍스트
    lat = rows["iso"]["lat_final"]
    assert lat[3] == pytest.approx(emit_time(kk[3], 66) - 1.6) and lat[6] == pytest.approx(4.8 - 4.1)          # SEM 확정 / EOS 확정
    assert all(x == pytest.approx(4.8 - e) for x, e in zip(rows["s5"]["lat_final"], ends))
    n_calls = len(calls); tp.run_set(a, se, Fake(), "toy"); assert len(calls) == n_calls                           # 이어하기: 다 된 행은 다시 디코드하지 않는다
    summ = tp.summarize(a, se, [out])
    assert summ["toy|iso"]["final"] == 0 and summ["toy|iso"]["first"] == 0 and summ["toy|iso"]["n_seg"] == 2 and summ["toy|s5"]["n_seg"] == 1

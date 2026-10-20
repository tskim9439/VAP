"""비-TAXI dev 변환(experiments/sim_to_sessions.py) — 합성 교차 언어 대화 → cstbench 세션·mix wav: 턴 = episode, wav 길이 = duration, 턴 언어 = 화자 언어,
전사 = 정렬 토큰, 번역 참조 = CstDialogueDataset.trans_of 규칙, cst_stage4_taxi(run 의 창·슬롯, score)·cst_attr_turns 와의 호환. 모델 없이 CPU."""
import argparse, json, os, random, sys
import numpy as np, pytest
sf = pytest.importorskip("soundfile")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments")); sys.path.insert(0, os.path.join(ROOT, "cst-bench", "src"))
import sim_to_sessions as S2S  # noqa: E402
from sim_build_dialogues import write_targets  # noqa: E402
from test_cst_attr_turns import WordTok  # noqa: E402
from vapasr.data.dialogue import build_episodes  # noqa: E402
from vapasr.data.dialogue_dataset import mono_cache_path  # noqa: E402
from vapasr.data.sim_dialogue import Unit, TimingStats, SimConfig, synth_session  # noqa: E402

EN = "hello i need a cab to the airport please yes thank you where are we going now".split()
DE = "guten tag ich brauche ein taxi zum flughafen bitte ja danke wohin fahren wir jetzt".split()
KEYS = {"session_id", "corpus", "redistributable", "speakers", "channels", "sample_rate", "config", "seed", "duration_s", "overlap_ratio", "skipped", "mix", "two_channel", "turns"}
TKEYS = {"turn_id", "speaker", "lang", "start_s", "end_s", "gap_before_s", "transcript", "translation", "translation_lang", "trimmed_s"}   # cstbench render 의 timeline 턴 키


def units(tok, key, words, tgt_words, tgt, n, rng, pieces=False):
    """발화 n 개(단어 = 토큰, 번역 토큰도 단어). pieces: 둘씩 한 행의 정제 조각(#0·#1) — 행 번역이 episode 둘에 걸칠 수 있다."""
    out = []
    for i in range(n):
        k = rng.randint(2, 5); ws = [rng.choice(words) for _ in range(k)]; d = round(0.35 * k, 3)
        toks = [(tok.convert_tokens_to_ids(w), round(d * (j + 1) / k, 3)) for j, w in enumerate(ws)]
        tr = [(tok.convert_tokens_to_ids(rng.choice(tgt_words)), round(d * (j + 1) / k, 3)) for j in range(k)]
        out.append(Unit(key=key, order=i, path=f"/{key}/{i}.wav", src_offset=None, dur=d, tokens=toks, text=" ".join(ws), uid=f"{key}{i // 2}#{i % 2}" if pieces else f"{key}{i}",
                        pool="t", trans={tgt: tr}, cont=pieces and i % 2 == 1))
    return out


def make(tmp_path, tok, trans=True, n=2):
    """sim_build_dialogues 와 같은 산출 배치: <data>/sim_en-de.refined.dialogues.jsonl, mono/sim_en-de/<conv>.npy(float16), targets-sim_en-de/en-de/u-*.jsonl.gz."""
    data = tmp_path / "data"; data.mkdir(); lines, D, Y, TA = [], [], [], {}
    for i in range(n):
        rng = random.Random(i); A = units(tok, "a", EN, DE, "German", 30, rng, pieces=True); B = units(tok, "b", DE, EN, "English", 30, rng)
        st = TimingStats(fto=[0.1, 0.3, 0.6], hold=[0.4], p_hold=0.25, turn=[2.0, 4.0], bc_rate=0.0, bc_pos=[0.5], n_dialogues=1)
        d = synth_session([A, B], st, rng, 40.0, lambda u: (0.0, u.dur), f"sim_en-de_{i:06d}", "sim_en-de", "English+German", cfg=SimConfig(p_hold=0.25, bc_scale=0.0),
                          meta=dict(spk_lang={"a": "English", "b": "German"}))
        TA[i] = (d.conv_id, d.meta.pop("trans")); lines.append(d.to_json()); D.append(d)
        f = mono_cache_path(str(data / "mono"), d); os.makedirs(os.path.dirname(f), exist_ok=True)
        y = (0.3 * np.sin(np.arange(int(round(d.duration_s * 16000))) / 7.0)).astype(np.float16); np.save(f, y); Y.append(y)
    (data / "sim_en-de.refined.dialogues.jsonl").write_text("\n".join(lines) + "\n")
    if trans: write_targets(str(data), "en-de", TA)
    return data, D, Y


def args(data, out, **kw):
    return argparse.Namespace(**dict(dict(data=str(data), corpus="sim_en-de", tokenizer="fake", out=str(out), mono=None, targets=None, no_trans=False, limit=0), **kw))


def test_turns_are_episodes_audio_length_and_languages(tmp_path):
    tok = WordTok(); data, D, Y = make(tmp_path, tok); out = tmp_path / "dev"
    info = S2S.convert(args(data, out), tok=tok); recs = [json.loads(l) for l in open(out / "sessions.jsonl")]
    assert info["sessions"] == len(recs) == 2 and info["stats"]["xl"] == 2 and info["stats"]["no_mono"] == 0 and info["stats"]["untranscribed"] == 0
    for r, d, y in zip(recs, D, Y):
        eps = build_episodes(d); assert set(r) == KEYS and r["session_id"] == d.conv_id and len(r["turns"]) == len(eps) and all(set(t) == TKEYS for t in r["turns"])
        x, sr = sf.read(str(out / r["mix"]), dtype="float32"); assert sr == 16000 and x.ndim == 1 and len(x) == len(y) == round(r["duration_s"] * 16000)
        assert np.abs(x - y.astype(np.float32)).max() < 1e-4                                                   # mono 캐시 그대로(PCM16 양자화만)
        assert {k: v["lang"] for k, v in r["speakers"].items()} == d.meta["spk_lang"]
        for t, ep, prev in zip(r["turns"], eps, [None] + eps[:-1]):
            assert t["speaker"] == ep.speaker and t["lang"] == r["speakers"][ep.speaker]["lang"] and (t["start_s"], t["end_s"]) == (round(ep.start, 4), round(ep.end, 4))
            assert tok(" " + t["transcript"])["input_ids"] == [tid for tid, _ in ep.tokens] and t["transcript"] == tok.decode([tid for tid, _ in ep.tokens]).strip()
            assert t["gap_before_s"] == (None if prev is None else round(ep.start - prev.end, 4))
        assert [t["start_s"] for t in r["turns"]] == sorted(t["start_s"] for t in r["turns"]) and any(t["speaker"] == p["speaker"] for t, p in zip(r["turns"][1:], r["turns"]))   # 같은 화자 다음 episode(cont)도 턴


def test_translation_reference_follows_dataset_rule(tmp_path):
    from vapasr.data.ast_dataset import load_ast_targets
    from vapasr.data.cst_dialogue import CstDialogueDataset
    tok = WordTok(); data, D, Y = make(tmp_path, tok); out = tmp_path / "dev"; S2S.convert(args(data, out), tok=tok)
    recs = [json.loads(l) for l in open(out / "sessions.jsonl")]; TR, _, _ = load_ast_targets([str(data / "targets-sim_en-de")])
    ds = CstDialogueDataset([str(data / "sim_en-de.refined.dialogues.jsonl")], tok, delays=(4,), trans=TR, windows=[(d.conv_id, 0.0, d.duration_s) for d in D])
    split_rows = 0
    for r, d in zip(recs, D):
        f = ds.trans_of(d.conv_id, {"a": "German", "b": "English"}, 0.0); split_rows += sum(len(x["eps"]) > 1 for x in ds._rows(d.conv_id).values())
        for t, ep in zip(r["turns"], ds.eps[d.conv_id]):
            want = tok.decode([i for i, _, _ in f(ep)[0]]).strip() or None                                      # 학습 번역 흐름과 같은 토큰(episode 몫)
            assert t["translation"] == want and t["translation_lang"] == ({"a": "German", "b": "English"}[ep.speaker] if want else None)
    assert split_rows > 0 and sum(t["translation"] is not None for r in recs for t in r["turns"]) > 0            # 행이 episode 둘에 걸친 경우를 실제로 거쳤다


def test_no_translation_and_out_must_be_new(tmp_path):
    tok = WordTok(); data, D, Y = make(tmp_path, tok, trans=False); out = tmp_path / "dev"
    S2S.convert(args(data, out, limit=1), tok=tok); recs = [json.loads(l) for l in open(out / "sessions.jsonl")]
    assert len(recs) == 1 and all(t["translation"] is None and t["translation_lang"] is None for t in recs[0]["turns"])
    keep = sorted(os.listdir(out))
    with pytest.raises(SystemExit): S2S.convert(args(data, out), tok=tok)                                      # 이미 있으면 멈춘다
    assert sorted(os.listdir(out)) == keep                                                                    # 아무것도 지우거나 덮지 않았다
    with pytest.raises(SystemExit): S2S.convert(args(data, tmp_path / "dev2", targets=str(tmp_path / "none")), tok=tok)


@pytest.mark.parametrize("trans", [True, False])
def test_stage4_taxi_run_windows_and_score(tmp_path, trans):
    """run 의 창·슬롯이 세션 기록에서 나오고, score 가 번역 참조 유무와 상관없이 받아쓰기·split·턴 지표를 낸다(번역이 없으면 BLEU 방향만 빈다)."""
    pytest.importorskip("sacrebleu")
    import types, contextlib, io
    import cst_stage4_taxi as T
    tok = WordTok(); data, D, Y = make(tmp_path, tok, trans=trans); out = tmp_path / "dev"; S2S.convert(args(data, out), tok=tok)
    recs = [json.loads(l) for l in open(out / "sessions.jsonl")]
    for r in recs:
        sp, langs = T.slots_of(r); assert sorted(langs) == ["English", "German"] and [r["speakers"][s]["lang"] for s in sp] == langs
        assert T.windows_of(r, 90.0) == [(0.0, r["duration_s"])] and len(T.windows_of(r, 15.0)) > 1
    res = tmp_path / "res"; res.mkdir(); asr, hyp = [], []
    for r in recs:                                                                                            # 완벽한 받아쓰기(턴 단어를 턴 구간에 고르게 + 0.7 s)와 참조 번역 그대로의 번역
        for t in r["turns"]:
            ws = t["transcript"].split(); a, b = t["start_s"], t["end_s"]
            asr += [dict(session=r["session_id"], lang=t["lang"], t=round(a + (b - a) * (j + 1) / len(ws) + 0.7, 3), text=w) for j, w in enumerate(ws)]
            if t["translation"]: hyp += [dict(session=r["session_id"], lang=t["translation_lang"], t=round(b + 1.0, 3), text=w) for w in t["translation"].split()]
    (res / "asr-d10.jsonl").write_text("".join(json.dumps(x) + "\n" for x in asr)); (res / "hyp-d10.jsonl").write_text("".join(json.dumps(x) + "\n" for x in hyp))
    with contextlib.redirect_stdout(io.StringIO()): T.score(types.SimpleNamespace(sessions=str(out / "sessions.jsonl"), out=str(res), turn_metrics=True))
    rep = json.loads((res / "report-d10.json").read_text()); ar = json.loads((res / "asr-report-d10.json").read_text()); tm = json.loads((res / "turn-metrics-d10.json").read_text())
    assert bool(rep["directions"]) == trans and all(x["bleu"] == 100.0 for x in rep["directions"].values())
    assert ar["split"]["cp"] == ar["split"]["agn"] == 0.0 and ar["English"]["wer"] == ar["German"]["wer"] == 0.0
    assert tm["toggle_recall"]["pct"] == 100.0 and tm["changes"]["ratio"] == 1.0 and tm["sessions"] == 2 and tm["skipped_same_lang"] == 0

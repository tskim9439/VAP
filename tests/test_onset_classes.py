"""온셋 분류(cst_attr_turns.onset_class·onset_classes, V3 S0 D1)와 taxi_v3_probe --classify-onsets — 합성 원출력 [(k, tid)] 으로 다섯 클래스 ×
이전 슬롯 <EOT> 있음/없음, 정렬 창 W 3 대 5 에서 같은 클래스, OK 수 = toggle(turn_metrics), 끄면 probe 산출 바이트 동일. 모델 없이 CPU(가짜 model·decode_batch)."""
import importlib.util, json, os, sys, types, zlib
import numpy as np, pytest, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments")); sys.path.insert(0, os.path.join(ROOT, "cst-bench", "src"))
import cst_attr_turns as A  # noqa: E402
import cst_stage4_taxi as T  # noqa: E402
from cst_stage3_eval import words_of  # noqa: E402
from test_cst_attr_turns import WordTok, session, base_module, fresh, OFF  # noqa: E402
from vapasr.data.cst_eval import parse_emitted4  # noqa: E402

PROBE = "raw/sources/experiments/2026-10-21-v3-attribution/taxi_v3_probe.py"
SPK, TR, EOT = (900, 901), 902, 903


# ── 상태기계 단위: seg = 앞 턴 마지막 정렬 단어(슬롯 0 전사) 뒤 ~ 새 턴 첫 정렬 단어 앞의 tid
@pytest.mark.parametrize("seg,kb,lost,trw,want", [
    ([TR, 5, SPK[1]], 1, (), (), ("OK", False, False)),                                    # 번역 뒤 다른 슬롯
    ([TR, 5, EOT, SPK[1]], 1, (), (), ("OK", True, False)),                                # <EOT> 뒤 다른 슬롯
    ([TR, 5, SPK[0]], 0, (), (), ("RESUME", False, True)),                                 # 번역 뒤 같은 슬롯 재개
    ([SPK[0]], 0, (), (), ("RESUME", False, False)),                                       # 번역 없이 같은 슬롯 태그(resume_without_trans)
    ([TR, 5, EOT, SPK[0]], 0, (), (), ("REASSERT", True, True)),                           # <EOT> 뒤 같은 슬롯
    ([EOT, SPK[0]], 0, (), (), ("REASSERT", True, False)),
    ([], 0, (), (), ("SILENT", False, False)),                                             # 태그 없이 전사 이어 씀
    ([EOT], 0, (), (), ("SILENT", True, False)),
    ([TR, 5, 6, 7, SPK[0]], 0, ("to", "the"), ("where", "to", "to", "the"), ("ABSORBED", False, False)),   # 새 턴 앞머리가 번역으로
    ([TR, 5, 6, 7, EOT, SPK[0]], 0, ("to", "the"), ("to", "the"), ("ABSORBED", True, False)),
    ([TR, 5, SPK[0]], 0, ("to", "the"), ("where", "now"), ("RESUME", False, True)),        # 잃은 단어가 번역에 없으면 흡수가 아니다
    ([SPK[1], 4, EOT, SPK[0]], 0, (), (), ("RESUME", False, False)),                       # 다른 슬롯의 <EOT> 는 이전 슬롯 <EOT> 가 아니다
])
def test_onset_class_state_machine(seg, kb, lost, trw, want):
    assert A.onset_class(seg, 0, kb, SPK, TR, EOT, lost, trw) == want


# ── 정렬을 거친 분류: 두 턴 세션(앞 화자 → 새 화자)마다 원출력을 각본으로 만든다
A_TXT, B_TXT, TRA = "guten tag taxi zentrale", "to the airport please", "where to"
SCEN = {"ok": "OK", "ok_eot": "OK", "resume": "RESUME", "resume_bare": "RESUME", "reassert": "REASSERT", "reassert_bare": "REASSERT",
        "silent": "SILENT", "silent_eot": "SILENT", "absorbed": "ABSORBED", "absorbed_eot": "ABSORBED"}
EOT_OF = {k: k.endswith("_eot") or k.startswith("reassert") for k in SCEN}


def emit(events):
    """[(시각 s, tid)] → [(k, tid)] — 방출 시각 (k+1)·80 ms 가 그 시각에 가깝게(단조)."""
    out, k = [], 0
    for t, tid in events: k = max(k, int(round(t / 0.08)) - 1); out.append((k, tid))
    return out


def scripted(s, kind, tok):
    """세션 s(두 턴: 앞 화자 → 새 화자)의 원출력. 전사 단어 = 참조 단어 끝 + OFF, 태그·<TRANS>·<EOT> 는 이웃 단어 시각."""
    sp, langs = T.slots_of(s); spk = tuple(tok.convert_tokens_to_ids(t) for t in ("<SPK_A>", "<SPK_B>")); tr, eot = tok.convert_tokens_to_ids("<TRANS>"), tok.convert_tokens_to_ids("<EOT>")
    W = lambda w: tok.convert_tokens_to_ids(w)
    (a, b) = s["turns"]; ka, kb = sp.index(a["speaker"]), sp.index(b["speaker"])
    wt = lambda tu: [(tu["start_s"] + (tu["end_s"] - tu["start_s"]) * (j + 1) / len(tu["transcript"].split()) + OFF, W(w)) for j, w in enumerate(tu["transcript"].split())]
    aw, bw = wt(a), wt(b); ta = aw[-1][0] + 0.08; tb = bw[0][0] - 0.01
    ev = [(aw[0][0], spk[ka])] + aw; trn = [(ta, tr)] + [(ta, W(w)) for w in TRA.split()]
    if kind == "ok": ev += trn + [(tb, spk[kb])] + bw
    elif kind == "ok_eot": ev += trn + [(ta, eot), (tb, spk[kb])] + bw
    elif kind == "resume": ev += trn + [(tb, spk[ka])] + bw
    elif kind == "resume_bare": ev += [(tb, spk[ka])] + bw
    elif kind == "reassert": ev += trn + [(ta, eot), (tb, spk[ka])] + bw
    elif kind == "reassert_bare": ev += [(ta, eot), (tb, spk[ka])] + bw
    elif kind == "silent": ev += bw
    elif kind == "silent_eot": ev += [(ta, eot)] + bw
    else:                                                                                 # absorbed: 새 턴 첫 두 단어가 앞 슬롯 번역으로 나오고, 같은 슬롯 태그 뒤 나머지
        ev += trn + [(t, w) for t, w in bw[:2]] + ([(bw[1][0], eot)] if kind == "absorbed_eot" else []) + [(bw[2][0] - 0.01, spk[ka])] + bw[2:]
    return emit(ev)


def fixture(tok, reverse=()):
    """시나리오마다 세션 하나. reverse 에 든 시나리오는 영어 화자가 먼저(새 화자 = 독일어 — 방향)."""
    S, runs = [], {}
    for kind in SCEN:
        turns = [("CL", B_TXT), ("DP", A_TXT)] if kind in reverse else [("DP", A_TXT), ("CL", B_TXT)]
        s = session(f"on-{kind}", turns=turns); S.append(s); runs[s["session_id"]] = [(0.0, scripted(s, kind, tok))]
    return S, runs


def ids(tok):
    return tuple(tok.convert_tokens_to_ids(t) for t in ("<SPK_A>", "<SPK_B>")), tok.convert_tokens_to_ids("<TRANS>"), tok.convert_tokens_to_ids("<EOT>")


def asr_like_run(S, runs, tok):
    """cst_stage4_taxi.run 과 같은 asr 행(parse_emitted4 → words_of, 창 시작 + 시각, 슬롯 0·1)."""
    spk, tr, eot = ids(tok); out = []
    for s in S:
        _, langs = T.slots_of(s)
        for w0, em in runs[s["session_id"]]:
            src, _, _ = parse_emitted4(em, spk, tr, eot)
            for k in range(2): out += [dict(session=s["session_id"], lang=langs[k], t=round(w0 + t, 3), text=w) for w, t in words_of(tok, src[k])]
    return out


def test_onset_classes_all_five_with_and_without_eot():
    tok = WordTok(); S, runs = fixture(tok); spk, tr, eot = ids(tok)
    r = A.onset_classes(S, runs, tok, spk, tr, eot)
    assert r["changes"] == r["classified"] == len(SCEN) and r["noalign"] == r["cross_window"] == 0 and abs(r["offset_s"] - OFF) < 0.05
    want = {c: dict(eot=sum(1 for k, v in SCEN.items() if v == c and EOT_OF[k]), no_eot=sum(1 for k, v in SCEN.items() if v == c and not EOT_OF[k])) for c in A.ONSET}
    assert {c: dict(eot=v["eot"], no_eot=v["no_eot"]) for c, v in r["classes"].items()} == want
    assert all(v["eot"] >= 1 and v["no_eot"] >= 1 for c, v in want.items() if c in ("OK", "SILENT", "ABSORBED")) and r["resume_without_trans"] == 1
    assert list(r["by_new_lang"]) == ["English"] and "on-" not in json.dumps(r) and "airport" not in json.dumps(r)        # 집계만
    for W in (2.0, 3.0): assert A.onset_classes(S, runs, tok, spk, tr, eot, W=W)["classes"] == r["classes"]             # 정렬 창을 좁혀도 같은 클래스


def test_onset_direction_and_ok_equals_toggle():
    tok = WordTok(); S, runs = fixture(tok, reverse={"ok", "resume", "silent"}); spk, tr, eot = ids(tok)
    r = A.onset_classes(S, runs, tok, spk, tr, eot)
    assert r["by_new_lang"]["German"]["OK"]["all"] == r["by_new_lang"]["German"]["RESUME"]["all"] == r["by_new_lang"]["German"]["SILENT"]["all"] == 1
    assert r["by_new_lang"]["English"]["OK"]["all"] == 1 and r["by_new_lang"]["English"]["ABSORBED"]["all"] == 2
    asr = asr_like_run(S, runs, tok)
    for s in S:                                                                           # 원출력으로 만든 가설 단어 = run 의 asr 행
        _, langs = T.slots_of(s); hw, _ = A.emitted_words(tok, runs[s["session_id"]], spk, tr, eot, langs)
        assert [x[:3] for x in hw] == A.hyp_words(asr)[s["session_id"]]
    tm = A.turn_metrics(S, asr, B=10)
    assert tm["toggle_recall"]["toggled"] == r["classes"]["OK"]["all"] == 2 and tm["toggle_recall"]["n"] == r["classified"]


def test_cross_window_and_noalign_are_counted_not_classified():
    tok = WordTok(); spk, tr, eot = ids(tok); s = session("cw", turns=[("DP", A_TXT), ("CL", B_TXT), ("DP", "ja")])
    em = scripted(dict(s, turns=s["turns"][:2]), "ok", tok); cut = s["turns"][0]["end_s"] + 1.0                      # 앞 턴 번역 뒤·새 턴 태그 앞(방출 시각)
    k0 = int(round(cut / 0.08)); runs = {"cw": [(0.0, [x for x in em if x[0] < k0]), (cut, [(k - k0, t) for k, t in em if k >= k0])]}   # 교대 1 이 창 경계에 걸침
    r = A.onset_classes([s], runs, tok, spk, tr, eot)
    assert r["changes"] == 2 and r["cross_window"] == 1 and r["noalign"] == 1 and r["classified"] == 0                   # 'ja' 턴은 가설이 없다


# ── taxi_v3_probe --classify-onsets: 실제 cst_stage4_taxi.run 을 가짜 model·decode_batch 로 돌린다
class FakeModel:
    def __init__(self, sp): self.config = types.SimpleNamespace(sp_ids=sp)
    def parameters(self): return iter([torch.zeros(1)])
    def encode(self, w, n, K): return torch.full((1, int(K[0]), 1), float(round(float(w[0, 0]) * 100) - 1))   # 세션 번호(wav 의 직류 값)를 특징으로


def probe_inputs(tmp_path, tok):
    sf = pytest.importorskip("soundfile")
    S, runs = fixture(tok, reverse={"ok", "silent_eot"}); d = tmp_path / "taxi"; d.mkdir()
    for i, s in enumerate(S):
        s["mix"] = f"{s['session_id']}/mix.wav"; (d / s["session_id"]).mkdir(); sf.write(str(d / s["mix"]), np.full(int(s["duration_s"] * 16000), (i + 1) * 0.01, np.float32), 16000, subtype="PCM_16")
    (d / "sessions.jsonl").write_text("".join(json.dumps(s) + "\n" for s in S))
    return d / "sessions.jsonl", [runs[s["session_id"]][0][1] for s in S]


def run_probe(mod, monkeypatch, sess, out, tok, emitted, flag):
    import vapasr.data.cst_eval as E, vapasr.hf.infer as I
    sp = {"<SPK_A>": tok.convert_tokens_to_ids("<SPK_A>"), "<SPK_B>": tok.convert_tokens_to_ids("<SPK_B>"), "<EOT>": tok.convert_tokens_to_ids("<EOT>")}
    calls = []
    def decode(model, feats, prefixes, max_flush_rounds=None):
        calls.append(len(feats)); return [types.SimpleNamespace(emitted=list(emitted[int(f[0, 0])])) for f in feats]
    BD = types.ModuleType("vapasr.hf.batch_decode"); BD.decode_batch = decode                     # 실제 모듈은 transformers DynamicCache 가 필요해 가짜 모듈로 바꿔 끼운다
    BD.SwitchBias, BD.word_init_mask, BD.logits_size = object, None, None                        # run() 이 이름만 가져온다(--switch-bias 없으면 쓰지 않음)
    monkeypatch.setitem(sys.modules, "vapasr.hf.batch_decode", BD)
    monkeypatch.setattr(I, "load_model", lambda *x, **kw: (FakeModel(sp), tok)); monkeypatch.setattr(E, "parse_emitted4", parse_emitted4)
    monkeypatch.setattr(sys, "argv", ["probe", "--checkpoint", "ck", "--sessions", str(sess), "--out", str(out), "--batch-size", "4"] + (["--classify-onsets"] if flag else []))
    mod.main(); assert sum(calls) == len(emitted)
    return {f: open(os.path.join(out, f), "rb").read() for f in sorted(os.listdir(out))}


def test_probe_classify_onsets_off_is_byte_identical(tmp_path, monkeypatch):
    monkeypatch.setenv("CODE", ROOT); tok = WordTok(); sess, emitted = probe_inputs(tmp_path, tok)
    spec = importlib.util.spec_from_file_location("taxi_v3_probe_new", os.path.join(ROOT, PROBE)); new = importlib.util.module_from_spec(spec); spec.loader.exec_module(new)
    d = tmp_path / "out"
    off = run_probe(new, monkeypatch, sess, fresh(d, {}), tok, emitted, False); on = run_probe(new, monkeypatch, sess, fresh(d, {}), tok, emitted, True)
    assert "onset-classes-d10.json" not in off and sorted(set(on) - set(off)) == ["onset-classes-d10.json"] and all(on[f] == b for f, b in off.items())   # 켜도 기존 산출은 그대로
    oc = json.loads(on["onset-classes-d10.json"]); assert oc["changes"] == len(SCEN) and oc["classes"]["OK"]["all"] == 2 and oc["classes"]["ABSORBED"]["all"] == 2
    assert all(f"on-{k}" not in on["onset-classes-d10.json"].decode() for k in SCEN)
    asr = [json.loads(l) for l in off["asr-d10.jsonl"].decode().splitlines()]; S = [json.loads(l) for l in open(sess)]
    assert A.turn_metrics(S, asr, B=10)["toggle_recall"]["toggled"] == oc["classes"]["OK"]["all"]
    base = base_module(PROBE, "classify_onsets", "taxi_v3_probe_base")
    if base is None: pytest.skip("git 기준판 없음")
    assert run_probe(base, monkeypatch, sess, fresh(d, {}), tok, emitted, False) == off                                          # 바꾸기 전 판과 산출 바이트 동일

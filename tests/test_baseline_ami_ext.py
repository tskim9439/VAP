"""experiments/baseline_ami_ext.py vapasr-spk(2026-11-01 화자마다 세션 — Diarization Masked Input): 가짜 모델로 세션마다 오디오 가리기·슬롯 A 활동·화자 이름·
단어 시각(방출 청크 − δ)·--spk-keep 을 본다. 실제 모델·디코드는 서버 스모크가 본다."""
import argparse, importlib.util, pathlib, sys, types

import numpy as np
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _mod():
    sys.path.insert(0, str(_ROOT / "experiments")); sys.path.insert(0, str(_ROOT))
    spec = importlib.util.spec_from_file_location("_t_baseline_ami_ext", _ROOT / "experiments" / "baseline_ami_ext.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_vapasr_spk_sessions_mask_act_and_word_times(monkeypatch):
    be = _mod()
    import cst_stage3_eval, vapasr.hf.long_session as ls
    K, HOP, delta = 120, 1280, 10; SPK, BEGIN, EOT = (101, 102, 103, 104), 200, 201
    act = torch.zeros(1, K, 4); act[0, 0:50, 0] = 1; act[0, 60:100, 1] = 1                          # 화자 0·1 만 말한다

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1)); self.n3d = object()
        def n3d_act(self, w, wl, Kt): assert w.shape[1] == K * HOP and int(Kt[0]) == K; return act.clone()
    calls = []

    def enc_long(model, ws, Kk, act=None, **kw):
        calls.append((ws.copy(), np.asarray(act).copy())); return torch.zeros(2, Kk, 4)

    def blk_decode(model, feats, prefix, D, context_s=0.0, **kw):
        s = len(calls) - 1                                                                        # 세션 0: 슬롯 A 단어 하나, 세션 1: 슬롯 A 하나 + 슬롯 B 하나
        em = [(5, SPK[0]), (5, BEGIN), (20 + s, 7), (30 + s, EOT)] + ([(40, SPK[1]), (41, 8)] if s == 1 else [])
        return em, 0, 0, 0, 0, [1.0]
    monkeypatch.setattr(ls, "encode_long", enc_long); monkeypatch.setattr(ls, "long_block_decode", blk_decode)
    monkeypatch.setattr(cst_stage3_eval, "words_of", lambda tok, toks: [(f"w{t}", te) for te, t in toks])

    class DS:
        def slots(self, i): return ["a", "b", "c", "d"], ["en"] * 4
        def cst_prefix(self, langs, d): assert d == delta and len(langs) == 4; return [1, 2]

    for keep, n_words in (("all", 3), ("slotA", 2)):
        calls.clear()
        o = object.__new__(be.OursSpk)
        o.a = argparse.Namespace(spk_margin=2, spk_keep=keep, context_s=30.0); o.model = Model(); o.tok = None
        o.spk, o.EOT, o.BEGIN, o.N, o.delta, o.where = SPK, EOT, BEGIN, 4, delta, {"ami-sdm:M@0.0": (DS(), 0)}
        wav = np.ones(K * HOP, np.float32)
        hw, info = o([("ami-sdm:M@0.0", wav)])["ami-sdm:M@0.0"]
        assert [s["spk"] for s in info["sessions"]] == [0, 1] and len(calls) == 2                  # 말한 화자만 세션
        (w0, a0), (w1, a1) = calls
        assert a0[:, 0].tolist() == act[0, :, 0].tolist() and not a0[:, 1:].any() and a1[:, 0].tolist() == act[0, :, 1].tolist()   # 슬롯 A = 그 화자
        f0 = w0.reshape(K, HOP)[:, 0]; assert f0[:52].all() and not f0[52:].any()                     # 화자 0: 0–49 ± 2 프레임만 남는다
        f1 = w1.reshape(K, HOP)[:, 0]; assert not f1[:58].any() and f1[58:102].all() and not f1[102:].any()
        assert len(hw) == n_words and {s for s, _, _ in hw} == ({"spk0", "spk1"})
        assert ("spk0", "w7", round((20 - delta) * 0.08 + 0.04, 3)) in hw                              # 오디오 시각 = 방출 청크 − δ
        assert (("spk1", "w8", round((41 - delta) * 0.08 + 0.04, 3)) in hw) == (keep == "all")         # slotA 면 슬롯 B 단어는 버린다

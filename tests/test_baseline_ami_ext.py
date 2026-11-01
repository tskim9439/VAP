"""experiments/baseline_ami_ext.py vapasr-spk(2026-11-01 화자마다 세션 — Diarization Masked Input): 가짜 모델로 세션마다 오디오 가리기·슬롯 A 활동·화자 이름·
단어 시각(방출 청크 − δ)·--spk-keep·세션 단위 이어 하기(2026-11-02 — 선점)를 본다. 실제 모델·디코드는 서버 스모크가 본다."""
import argparse, importlib.util, json, pathlib, sys

import numpy as np
import pytest
import torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
K, HOP, DELTA = 120, 1280, 10; SPK, BEGIN, EOT = (101, 102, 103, 104), 200, 201
ACT = torch.zeros(1, K, 4); ACT[0, 0:50, 0] = 1; ACT[0, 60:100, 1] = 1                              # 화자 0·1 만 말한다
KEY = "ami-sdm:M@0.0"


def _mod():
    sys.path.insert(0, str(_ROOT / "experiments")); sys.path.insert(0, str(_ROOT))
    spec = importlib.util.spec_from_file_location("_t_baseline_ami_ext", _ROOT / "experiments" / "baseline_ami_ext.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.w = torch.nn.Parameter(torch.zeros(1)); self.n3d = object()
    def n3d_act(self, w, wl, Kt): assert w.shape[1] == K * HOP and int(Kt[0]) == K; return ACT.clone()


class _DS:
    def slots(self, i): return ["a", "b", "c", "d"], ["en"] * 4
    def cst_prefix(self, langs, d): assert d == DELTA and len(langs) == 4; return [1, 2]


def _fake(monkeypatch, crash_at=None):
    """encode_long·long_block_decode·words_of 를 가짜로. 세션 화자는 슬롯 A 활동으로 안다 — 화자 0: 슬롯 A 단어 하나, 화자 1: 슬롯 A 하나 + 슬롯 B 하나.
    crash_at = n 이면 n 번째 디코드에서 멈춘다(선점)."""
    import cst_stage3_eval, vapasr.hf.long_session as ls
    calls = []

    def enc_long(model, ws, Kk, act=None, **kw):
        a = np.asarray(act); s = next(j for j in range(2) if a[:, 0].tolist() == ACT[0, :, j].tolist())
        calls.append((ws.copy(), a.copy(), s)); return torch.zeros(2, Kk, 4)

    def blk_decode(model, feats, prefix, D, context_s=0.0, **kw):
        if len(calls) == crash_at: raise RuntimeError("선점")
        s = calls[-1][2]
        em = [(5, SPK[0]), (5, BEGIN), (20 + s, 7), (30 + s, EOT)] + ([(40, SPK[1]), (41, 8)] if s == 1 else [])
        return em, 0, 0, 0, 0, [1.0]
    monkeypatch.setattr(ls, "encode_long", enc_long); monkeypatch.setattr(ls, "long_block_decode", blk_decode)
    monkeypatch.setattr(cst_stage3_eval, "words_of", lambda tok, toks: [(f"w{t}", te) for te, t in toks])
    return calls


def _spk(be, keep, side):
    o = object.__new__(be.OursSpk)
    o.a = argparse.Namespace(spk_margin=2, spk_keep=keep, context_s=30.0); o.model = _Model(); o.tok = None
    o.spk, o.EOT, o.BEGIN, o.N, o.delta, o.where = SPK, EOT, BEGIN, 4, DELTA, {KEY: (_DS(), 0)}
    o.side = str(side); o.done = be.done_sessions(o.side)
    return o


def _go(o): return o([(KEY, np.ones(K * HOP, np.float32))])[KEY]


def test_vapasr_spk_sessions_mask_act_and_word_times(monkeypatch, tmp_path):
    be = _mod()
    for keep, n_words in (("all", 3), ("slotA", 2)):
        calls = _fake(monkeypatch)
        hw, info = _go(_spk(be, keep, tmp_path / f"s-{keep}.jsonl"))
        assert [s["spk"] for s in info["sessions"]] == [0, 1] and len(calls) == 2 and info["resumed"] == 0   # 말한 화자만 세션
        (w0, a0, _), (w1, a1, _) = calls
        assert a0[:, 0].tolist() == ACT[0, :, 0].tolist() and not a0[:, 1:].any() and a1[:, 0].tolist() == ACT[0, :, 1].tolist()   # 슬롯 A = 그 화자
        f0 = w0.reshape(K, HOP)[:, 0]; assert f0[:52].all() and not f0[52:].any()                     # 화자 0: 0–49 ± 2 프레임만 남는다
        f1 = w1.reshape(K, HOP)[:, 0]; assert not f1[:58].any() and f1[58:102].all() and not f1[102:].any()
        assert len(hw) == n_words and {s for s, _, _ in hw} == ({"spk0", "spk1"})
        assert ("spk0", "w7", round((20 - DELTA) * 0.08 + 0.04, 3)) in hw                              # 오디오 시각 = 방출 청크 − δ
        assert (("spk1", "w8", round((41 - DELTA) * 0.08 + 0.04, 3)) in hw) == (keep == "all")         # slotA 면 슬롯 B 단어는 버린다


def test_vapasr_spk_resumes_finished_sessions(monkeypatch, tmp_path):
    """선점 뒤 다시 돌면 끝난 화자 세션은 기록(sessions-rank{r}.jsonl)에서 꺼내고 남은 세션만 디코드한다 — 결과는 끊기지 않은 실행과 같다."""
    be = _mod()
    _fake(monkeypatch); full, _ = _go(_spk(be, "all", tmp_path / "full.jsonl"))
    side = tmp_path / "s.jsonl"
    _fake(monkeypatch, crash_at=2)
    with pytest.raises(RuntimeError): _go(_spk(be, "all", side))                                      # 화자 1 세션 디코드 중 선점
    assert [json.loads(l)["spk"] for l in side.read_text().splitlines()] == [0]
    with open(side, "a", encoding="utf-8") as f: f.write('{"key": "ami-sdm:M@0.0", "spk": 1, "wo')  # 쓰는 중 끊긴 줄
    calls = _fake(monkeypatch); hw, info = _go(_spk(be, "all", side))
    assert [c[2] for c in calls] == [1] and info["resumed"] == 1 and sorted(hw) == sorted(full)      # 화자 1 만 다시 디코드
    assert [s.get("resumed", False) for s in info["sessions"]] == [True, False]
    assert set(be.done_sessions(str(side))[KEY]) == {0, 1}                                           # 끊긴 줄 뒤 기록도 읽힌다

"""회의 통째 디코드 시각화 자료(2026-10-28, 사용자 "실제 추론 결과를 시각화와 함께 보여줘"): AMI 회의 하나의 정답 단어·시각, 디코드 결과 셋(태그 그대로 ·
디코드 때 태그 이끌기 · 사후 재귀속)의 단어·슬롯, E7a 헤드 AOSC 예측 활동과 정답 활동을 JSON 하나로 낸다(시각화 페이지가 읽는다). 읽기만 한다.
  PYTHONPATH=. python export_meeting_viz.py --cid ami-ihmmix:IS1009a --delta 10 --base <태그 그대로 결과 폴더> --guide <태그 이끌기 결과 폴더>
      --probs <diar_eval --session --save-probs 의 aosc/probs> --checkpoint <토크나이저·sp_ids> --out <json>
시각: 정답 단어 = 오디오 시각(토큰 정렬), 가설 단어 = 방출 시각 − δ·80 ms(오디오 시각 근사). 단어마다 그 시각 정답 활동이 가장 큰 화자를 'true' 로 붙인다.
시퀀스(2026-10-28, 사용자 "현재 사용하는 시퀀스나 예측 시퀀스도 함께 보여줘"): prefix 문자열, 정답 타깃 시퀀스(cst_sequence — 청크 k 오디오 다음에 오는 출력 토큰),
예측 시퀀스(emitted — 청크 k 다음에 낸 토큰, 태그 그대로·태그 이끌기), LLM 이 받은 슬롯 활동 입력(E7a AOSC 확률 > 임계)을 함께 낸다."""
import argparse, glob, json, os, sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
FRAME = 0.08


def row_of(d, rid, delta):
    for f in glob.glob(os.path.join(d, "predictions-rank*.jsonl")):
        for l in open(f, encoding="utf-8"):
            r = json.loads(l)
            if r["id"] == rid and r["delta"] == delta: return r
    raise SystemExit(f"행 없음 {rid} δ{delta}: {d}")


def main():
    ap = argparse.ArgumentParser()
    for k in ("cid", "base", "guide", "probs", "checkpoint", "out"): ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--delta", type=int, default=10); ap.add_argument("--data", default="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    from experiments.cst_stage3_eval import windows_of, window_ref, words_of
    from vapasr.data.cst_dialogue import SLOTS
    from vapasr.data.cst_eval import parse_emitted, relabel_by_activity, score_window, set_en_norm
    set_en_norm("whisper")
    tok = AutoTokenizer.from_pretrained(a.checkpoint); cfg = json.load(open(os.path.join(a.checkpoint, "config.json"))); sp = cfg["sp_ids"]
    corpus = a.cid.split(":", 1)[0]
    ns = SimpleNamespace(corpora=corpus, data=a.data, split=None, deltas=[a.delta], limit_windows=0, backchannel=(1.0, 3), max_speakers=4, window="full", hop=10.0,
                         window_draws=0, declare=4, xl_order="lang", context_s=30.0)
    sets, items = windows_of(ns, tok); ds = sets[corpus]
    i = next(i for c, i in items if ds.items[i][0] == a.cid); cid, t0w, L = ds.items[i]; K = int(round(L / FRAME))
    EOT = sp["<EOT>"]; spk = tuple(sp[t] for t in SLOTS[:4])
    ref_w, _, mask, sp_order = window_ref(ds, i, a.delta, EOT, tok)
    z = np.load(os.path.join(a.probs, f"{cid}@full.npz")); prob = z["prob"].astype(np.float32); ref_act = z["ref"].astype(np.float32)
    assert prob.shape[0] == K, (prob.shape, K)
    true_spk = lambda t: int(ref_act[max(0, min(K - 1, int(t / FRAME)))].argmax()) if ref_act[max(0, min(K - 1, int(t / FRAME)))].any() else -1
    rid = f"{cid}@{t0w}"
    dec = lambda t: tok.decode([int(t)], skip_special_tokens=False)
    NA = sp["<NEXT_AUDIO>"]; q = ds.cst_sequence(i, a.delta); pre_ids, ref_seq, cur = [], {}, None
    for p_, (tid, inp) in enumerate(zip(q["ids"], q["is_input"])):
        if inp: cur = int(q["chunk_of"][p_]); continue
        if cur is None: pre_ids.append(int(tid)); continue
        if int(tid) != NA: ref_seq.setdefault(cur, []).append(dec(tid))
    prefix = tok.decode(ds.cst_prefix(ds.slots(i)[1][:4], a.delta), skip_special_tokens=False)
    thr = float(cfg.get("act_threshold", 0.45))
    runs = {}
    for name, d in (("base", a.base), ("guide", a.guide)):
        r = row_of(d, rid, a.delta)
        hs, _ = parse_emitted(r["emitted"], spk, EOT, drop_chunks=mask); hw = [words_of(tok, h) for h in hs]
        seq = {}
        for k, tid in r["emitted"]: seq.setdefault(int(k), []).append(dec(tid))
        runs[name] = dict(hw=hw, retag=r.get("retag"), seq=seq)
    runs["relabel"] = dict(hw=relabel_by_activity(runs["base"]["hw"], prob, a.delta), retag=None)
    out_runs = {}
    for name, x in runs.items():
        sc = score_window(ref_w, x["hw"], "English"); n = max(1, sc["n_ref"])
        words = []
        for s, sl in enumerate(x["hw"]):
            for w, t in sl:
                ta = max(0.0, t - a.delta * FRAME); words.append([round(ta, 2), s, w, true_spk(ta)])
        words.sort(key=lambda z: z[0])
        out_runs[name] = dict(cp=round(100 * sc["cp"] / n, 2), agn=round(100 * sc["agn"] / n, 2), perm=sc["perm"], retag=x["retag"], words=words,
                              **({"seq": {str(k): v for k, v in x["seq"].items()}} if "seq" in x else {}))
    ref_words = sorted([[round(t, 2), s, w] for s, sl in enumerate(ref_w) for w, t in sl], key=lambda z: z[0])
    qz = lambda A: ["".join(str(min(9, int(v * 10))) for v in A[:, s]) for s in range(A.shape[1])]
    json.dump(dict(cid=cid, delta=a.delta, duration=round(L, 2), frame=FRAME, K=K, speakers=list(sp_order), n_ref_words=len(ref_words),
                   ref_words=ref_words, runs=out_runs, act_pred=qz(prob), act_ref=["".join("1" if v > 0 else "0" for v in ref_act[:, s]) for s in range(ref_act.shape[1])],
                   act_in=["".join("1" if v > thr else "0" for v in prob[:, s]) for s in range(prob.shape[1])], act_threshold=thr, prefix=prefix,
                   prompt_ids=len(pre_ids), ref_seq={str(k): v for k, v in ref_seq.items()}),
              open(a.out, "w"), ensure_ascii=False, separators=(",", ":"))
    print("VIZ_OK", a.out, {k: (v["cp"], v["agn"]) for k, v in out_runs.items()}, "ref words", len(ref_words), flush=True)


if __name__ == "__main__":
    main()

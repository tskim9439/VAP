#!/usr/bin/env python
"""E1 화자 분리 가지 평가(2026-10-25): AMI 창 평가(cst_stage3_eval 의 ami-bench-v1 창 — 20–90 s, draw 1, 화자 4, seed 0)와 같은 창에서 80 ms 프레임 DER(collar 0)을 잰다.
  python experiments/diar_eval.py run --diar <out/final 또는 checkpoint-N> --nemotron-dir <dir> --out <dir> [--data ami-bench-v1] [--corpora ami-ihmmix,ami-sdm]
      [--threshold 0.5] [--batch 8] [--rank r --world-size n] [--limit-windows 0]
  python experiments/diar_eval.py summarize --out <dir>
참조 = slot_activity(창 안 먼저 말한 순서), 셈에서 빼는 프레임 = 전사 없는 음성 구간(_bad_spans). DER = (miss + fa + conf)/참조 화자 프레임.
  conf_arrival = 슬롯 순서(도착 순서) 그대로 짝, conf_best = 창마다 혼동이 가장 적은 순열. 공식 분할(manifest dev·eval)·조건·창 화자 수별로 합친다.
  주의: Nemotron 3 Diarization 의 AMI DER(SDM 12.8 @1.04 s)은 세션 단위·강제 정렬 참조라 이 창 단위 값과 바로 비교되지 않는다."""
import argparse, collections, glob, json, math, os, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNK_S, SR = 0.08, 16000


def run(a):
    from experiments.cst_stage3_eval import windows_of
    from vapasr.data.cst_dialogue import slot_activity, window_speakers
    from vapasr.hf import load_tokenizer
    from vapasr.hf.diar import DiarBranch, der_counts, load_encoder
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    x = torch.load(os.path.join(a.diar, "diar.pt"), map_location="cuda"); n_spk = x["cfg"]["n_spk"]
    model = DiarBranch(**x["cfg"]).cuda().eval(); model.load_state_dict(x["state"])
    enc, _ = load_encoder(x["init"], a.nemotron_dir, "cuda"); taps = tuple(x["taps"]); tok = load_tokenizer(x["init"])
    man = json.load(open(os.path.join(a.data, "manifest.json"))) if os.path.exists(os.path.join(a.data, "manifest.json")) else {}
    sub = {**{m: "dev" for m in man.get("dev", [])}, **{m: "eval" for m in man.get("eval", [])}}
    ns = argparse.Namespace(corpora=a.corpora, data=a.data, split=None, deltas=[10], limit_windows=a.limit_windows, backchannel=(1.0, 3), max_speakers=n_spk,
                            window=tuple(float(v) for v in a.window.split(",")), hop=10.0, window_draws=a.window_draws, declare=n_spk)
    sets, items = windows_of(ns, tok); items = [it for j, it in enumerate(items) if j % a.world_size == a.rank]
    os.makedirs(a.out, exist_ok=True); dest = os.path.join(a.out, f"windows-rank{a.rank}.jsonl")
    done = {json.loads(l)["id"] for l in open(dest)} if os.path.exists(dest) else set(); t0 = time.time(); n = 0
    with open(dest, "a", buffering=1) as fo, torch.inference_mode():
        for b0 in range(0, len(items), a.batch):
            batch = [(c, i) for c, i in items[b0: b0 + a.batch] if f"{sets[c].items[i][0]}@{sets[c].items[i][1]}" not in done]
            if not batch: continue
            wavs, Ks, meta = [], [], []
            for c, i in batch:
                ds = sets[c]; cid, t0w, L = ds.items[i]; d = ds.dlgs[cid]; K = int(round(L / CHUNK_S))
                wavs.append(np.asarray(ds.mono(cid)[int(round(t0w * SR)): int(round((t0w + L) * SR))], dtype=np.float32)); Ks.append(K)
                sp = window_speakers(ds.eps[cid], t0w, L)[:n_spk]
                ref = torch.tensor(slot_activity(ds.eps[cid], sp, t0w, L, K, n_slots=n_spk), dtype=torch.float32).reshape(K, n_spk)
                m = torch.ones(K, dtype=torch.bool)
                for s, e in ds._bad_spans(d):
                    x0, x1 = s - t0w, e - t0w
                    if x1 > 0 and x0 < L: m[max(0, int(x0 / CHUNK_S)): min(K, int(math.ceil(x1 / CHUNK_S)))] = False
                meta.append((c, cid, t0w, L, K, len(sp), ref, m))
            Nw = max(len(w) for w in wavs); Km = max(Ks); W = torch.zeros(len(wavs), Nw)
            for j, w in enumerate(wavs): W[j, : len(w)] = torch.from_numpy(w)
            Kt = torch.tensor(Ks); pad = torch.arange(Km)[None] >= Kt[:, None]
            feats = enc(W.cuda(), torch.tensor([len(w) for w in wavs]).cuda(), Kt.cuda(), taps=taps)
            prob = torch.sigmoid(model(feats, pad.cuda()).float()).cpu()
            for j, (c, cid, t0w, L, K, nsp, ref, m) in enumerate(meta):
                pr = (prob[j, :K] > a.threshold).float(); dc = der_counts(pr, ref, m)
                row = dict(id=f"{cid}@{t0w}", corpus=c, sub=sub.get(cid.split(":", 1)[1], "?"), L=L, K=K, n_spk=nsp, n_hyp_spk=int((pr.sum(0) > 0).sum()), **dc)
                fo.write(json.dumps(row) + "\n"); n += 1
            print(f"  {min(b0 + a.batch, len(items))}/{len(items)} 창 · {n} 행 · {time.time() - t0:.0f}s", flush=True)
    print(f"DONE rank={a.rank} rows={n}", flush=True)


def summarize(a):
    rows = [json.loads(l) for f in sorted(glob.glob(os.path.join(a.out, "windows-rank*.jsonl"))) for l in open(f)]
    res = {}; der = lambda g, k: round(100 * (g["miss"] + g["fa"] + g[k]) / max(1, g["ref"]), 2)
    keys = sorted({(r["corpus"], r["sub"]) for r in rows})
    for c, s in keys:
        rr = [r for r in rows if r["corpus"] == c and r["sub"] == s]; g = collections.Counter()
        for r in rr: g.update({k: r[k] for k in ("ref", "miss", "fa", "conf_arrival", "conf_best")})
        by = {}
        for k in sorted({r["n_spk"] for r in rr}):
            h = collections.Counter()
            for r in rr:
                if r["n_spk"] == k: h.update({q: r[q] for q in ("ref", "miss", "fa", "conf_arrival", "conf_best")})
            by[k] = dict(der_arrival=der(h, "conf_arrival"), der_best=der(h, "conf_best"), ref_share=round(100 * h["ref"] / max(1, g["ref"]), 1))
        spk_ok = sum(r["n_hyp_spk"] == r["n_spk"] for r in rr) / max(1, len(rr))
        res[f"{c}/{s}"] = dict(windows=len(rr), der_arrival=der(g, "conf_arrival"), der_best=der(g, "conf_best"), miss=round(100 * g["miss"] / max(1, g["ref"]), 2),
                               fa=round(100 * g["fa"] / max(1, g["ref"]), 2), conf_arrival=round(100 * g["conf_arrival"] / max(1, g["ref"]), 2),
                               conf_best=round(100 * g["conf_best"] / max(1, g["ref"]), 2), spk_count_acc=round(spk_ok, 3), by_nspk=by)
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for k, v in res.items():
        print(f"{k:16s} 창 {v['windows']:4d} · DER 도착 순서 {v['der_arrival']:6.2f} · 최적 짝 {v['der_best']:6.2f}(miss {v['miss']} · fa {v['fa']} · conf {v['conf_arrival']}/{v['conf_best']}) · "
              f"화자 수 맞춤 {v['spk_count_acc']} · " + " · ".join(f"{s}인 {b['der_best']}" for s, b in v["by_nspk"].items()))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run"); q.add_argument("--diar", required=True); q.add_argument("--nemotron-dir", default=None); q.add_argument("--out", required=True)
    q.add_argument("--data", default="/soundai/users/tskim/VAPKT-data/data/ami-bench-v1"); q.add_argument("--corpora", default="ami-ihmmix,ami-sdm")
    q.add_argument("--window", default="20,90"); q.add_argument("--window-draws", type=int, default=1); q.add_argument("--threshold", type=float, default=0.5)
    q.add_argument("--batch", type=int, default=8); q.add_argument("--rank", type=int, default=0); q.add_argument("--world-size", type=int, default=1); q.add_argument("--limit-windows", type=int, default=0)
    q = sub.add_parser("summarize"); q.add_argument("--out", required=True)
    a = ap.parse_args(); {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()

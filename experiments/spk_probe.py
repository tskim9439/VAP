#!/usr/bin/env python
"""V2 M0 — 인코더 층별 화자 분리도 측정(2026-10-20): 학습 단계를 거치며 인코더가 화자를 가르는 정보를 잃었는지, 어느 층에 남아 있는지 잰다.

  python experiments/spk_probe.py run --ckpt 이름=체크포인트|orig[,…] --data <phase2> --split <v2.json> --out <dir> [--limit-windows 200] [--attr <stage3-attr-errors 폴더>]
  python experiments/spk_probe.py summarize --out <dir> [--attr …]

창: Stage 3 회귀 평가와 같은 held-out 창(cst_stage3_eval.windows_of, δ10·맞장구 1.0,3·코퍼스마다 200) — mono 혼합 그대로(모델이 보는 오디오).
턴: 창 안에 통째로 든 발화 중 1.0 s 이상이고 다른 화자와 겹치는 비율이 0.2 이하인 것(전사 유무와 무관 — 화자 라벨만 쓴다).
특징: 인코더 층마다(NeMo ConformerEncoder.layers 의 출력 hook, 'pre' = 서브샘플링 출력, 'adapter' = LLM 입력) 턴 구간 프레임 평균.
  orig = Nemotron 원본 인코더(Stage 0 는 인코더를 얼렸으므로 같다).
지표(층마다, raw 와 창 평균을 뺀 centered 두 가지):
  EER     — 같은 창 안 턴 쌍(같은 화자 = 양성, 다른 화자 = 음성)의 코사인으로 잰 등오류율
  loo_acc — 턴마다 자기를 뺀 각 화자 턴 평균(중심)과 코사인을 재 가까운 쪽이 자기 화자인 비율(슬롯 기억으로 판정할 때의 정확도에 해당)
  --attr 를 주면 V1 평가의 오류 분해(창별 연속 뒤바뀜 단위)와 묶어 '뒤바뀐 창 대 아닌 창' 의 loo_acc 를 따로 낸다.
산출: <out>/pairs-<이름>.npz(층별 점수·라벨), <out>/windows-<이름>.jsonl(창별 loo), summary.json. 아무것도 지우지 않는다."""
import argparse, collections, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def turns_of(d, t0, L, min_dur=1.0, max_ov=0.2):
    out = []
    for k, u in enumerate(d.utterances):
        if not (u.start >= t0 and u.end <= t0 + L and u.end - u.start >= min_dur): continue
        ov = sum(max(0.0, min(u.end, v.end) - max(u.start, v.start)) for v in d.utterances if v.speaker != u.speaker)
        if ov / (u.end - u.start) <= max_ov: out.append((k, u.speaker, u.start - t0, u.end - t0))
    return out


def load_encoder(spec, dev):
    import torch
    if spec == "orig":
        from vapasr.features.online import NemotronOnline
        enc = NemotronOnline(right_context=3).to(dev).eval(); return enc, None
    from vapasr.hf.infer import load_model
    model, _ = load_model(spec, encoder_path=os.environ.get("MXC_NEMOTRON_DIR"), dtype=torch.float32); model.eval()
    return model.encoder, model


def run(a):
    import numpy as np, torch
    from transformers import AutoTokenizer
    from cst_stage3_eval import windows_of
    dev = "cuda"; os.makedirs(a.out, exist_ok=True)
    tok_ck = next(v for _, v in (x.split("=", 1) for x in a.ckpt.split(",")) if v != "orig")
    tok = AutoTokenizer.from_pretrained(tok_ck)
    sets, items = windows_of(a, tok)
    plan = []                                                                                  # (코퍼스, 창, 턴들)
    for c, i in items:
        ds = sets[c]; cid, t0, L = ds.items[i]; tt = turns_of(ds.dlgs[cid], t0, L)
        if len({sp for _, sp, _, _ in tt}) == 2 and len(tt) >= 3: plan.append((c, i, tt))
    print(f"창 {len(items)} 중 화자 둘·턴 3 개 이상 {len(plan)}", flush=True)
    for name, spec in (x.split("=", 1) for x in a.ckpt.split(",")):
        dest = os.path.join(a.out, f"pairs-{name}.npz")
        if os.path.exists(dest): print(f"{name}: 이미 있음"); continue
        enc, model = load_encoder(spec, dev); layers = enc.enc.layers; feats = {}
        hooks = [layers[j].register_forward_hook(lambda m, i_, o, j=j: feats.__setitem__(j, (o[0] if isinstance(o, tuple) else o).detach())) for j in range(len(layers))]
        names = [f"L{j:02d}" for j in range(len(layers))] + (["adapter"] if model is not None else [])
        S = collections.defaultdict(list); Y = []; W = open(os.path.join(a.out, f"windows-{name}.jsonl"), "w"); t1 = time.time()
        with torch.inference_mode():
            for n, (c, i, tt) in enumerate(plan):
                ds = sets[c]; cid, t0, L = ds.items[i]; K = int(round(L / 0.08))
                wav = np.asarray(ds.mono(cid)[int(round(t0 * 16000)): int(round((t0 + L) * 16000))], dtype=np.float32)
                w = torch.from_numpy(wav).to(dev); feats.clear()
                out = enc(w[None], torch.tensor([len(wav)], device=dev), torch.tensor([K], device=dev))
                reps = {f"L{j:02d}": feats[j][0] for j in range(len(layers))}                  # (T, D) — NeMo 층 출력은 (B, T, D)
                if model is not None: reps["adapter"] = model.adapter(out[0].float())
                spk = [sp for _, sp, _, _ in tt]; lab = np.array([[spk[p] == spk[q] for q in range(len(tt))] for p in range(len(tt))])
                iu = np.triu_indices(len(tt), 1); Y.append(lab[iu]); rec = dict(id=f"{cid}@{t0}", corpus=c, lang=ds.dlgs[cid].lang, turns=len(tt))
                for nm in names:
                    R = reps[nm]; T = R.shape[0]
                    E = torch.stack([R[min(T - 1, int(s / 0.08)): max(min(T, int(np.ceil(e / 0.08))), min(T - 1, int(s / 0.08)) + 1)].mean(0) for _, _, s, e in tt]).float()
                    for mode, X in (("raw", E), ("cen", E - E.mean(0, keepdim=True))):
                        Xn = torch.nn.functional.normalize(X, dim=-1); C = (Xn @ Xn.T).cpu().numpy(); S[f"{nm}/{mode}"].append(C[iu])
                        ok = 0; m = 0
                        for p in range(len(tt)):
                            cs = {}
                            for sp in set(spk):
                                idx = [q for q in range(len(tt)) if q != p and spk[q] == sp]
                                if idx: cs[sp] = float(torch.nn.functional.normalize(X[idx].mean(0), dim=-1) @ Xn[p])
                            if len(cs) == 2: m += 1; ok += max(cs, key=cs.get) == spk[p]
                        rec[f"{nm}/{mode}"] = [ok, m]
                W.write(json.dumps(rec) + "\n")
                if n % 200 == 0: print(f"  {name}: {n}/{len(plan)} 창 · {time.time() - t1:.0f}s", flush=True)
        for h in hooks: h.remove()
        W.close(); np.savez_compressed(dest, y=np.concatenate(Y), **{k.replace("/", "__"): np.concatenate(v) for k, v in S.items()})
        del enc, model; torch.cuda.empty_cache(); print(f"{name}: 끝 {time.time() - t1:.0f}s", flush=True)


def eer(s, y):
    import numpy as np
    o = np.argsort(-s); y = y[o]; P, N = y.sum(), (~y).sum()
    fn = P - np.cumsum(y); fp = np.cumsum(~y); i = int(np.argmin(np.abs(fn / P - fp / N)))
    return float((fn[i] / P + fp[i] / N) / 2)


def summarize(a):
    import numpy as np
    flip = {}
    if a.attr:
        for l in open(os.path.join(a.attr, "windows.jsonl")):
            w = json.loads(l)
            if w["delta"] == 10: flip[w["id"]] = w.get("whole_streak_units", 0) > 0
    res = {}
    for p in sorted(glob.glob(os.path.join(a.out, "pairs-*.npz"))):
        name = os.path.basename(p)[6:-4]; z = np.load(p); y = z["y"].astype(bool)
        wins = [json.loads(l) for l in open(os.path.join(a.out, f"windows-{name}.jsonl"))]
        r = {}
        for k in z.files:
            if k == "y": continue
            nm = k.replace("__", "/"); acc = sum(w[nm][0] for w in wins) / max(1, sum(w[nm][1] for w in wins))
            e = dict(eer=round(100 * eer(z[k], y), 2), loo_acc=round(100 * acc, 2))
            for lg in ("Korean", "English", "German"):
                ww = [w for w in wins if w["lang"] == lg]; e[f"loo_{lg[:2]}"] = round(100 * sum(w[nm][0] for w in ww) / max(1, sum(w[nm][1] for w in ww)), 2)
            if flip:
                for f, tag in ((True, "flip"), (False, "noflip")):
                    ww = [w for w in wins if flip.get(w["id"]) is f]; e[f"loo_{tag}"] = round(100 * sum(w[nm][0] for w in ww) / max(1, sum(w[nm][1] for w in ww)), 2); e[f"n_{tag}"] = len(ww)
            r[nm] = e
        res[name] = dict(pairs=int(len(y)), pos=int(y.sum()), windows=len(wins), layers=r)
        best = sorted(((v["eer"], k) for k, v in r.items() if k.endswith("/cen")))[:3]
        print(f"== {name}: 쌍 {len(y)}(같은 화자 {int(y.sum())}) · 창 {len(wins)} · EER 최저 층 {best}")
        for k in sorted(r, key=lambda k: (k.split('/')[1], k)):
            v = r[k]; print(f"   {k:14s} EER {v['eer']:5.1f} loo {v['loo_acc']:5.1f} (KO {v['loo_Ko']:5.1f} EN {v['loo_En']:5.1f} DE {v['loo_Ge']:5.1f})"
                           + (f" | 뒤바뀐 창 {v['loo_flip']:5.1f}({v['n_flip']}) · 아닌 창 {v['loo_noflip']:5.1f}" if flip else ""))
    json.dump(res, open(os.path.join(a.out, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run"); q.add_argument("--ckpt", required=True); q.add_argument("--data", required=True); q.add_argument("--split", required=True); q.add_argument("--out", required=True)
    q.add_argument("--corpora", default="aihub134-2,aihub134-1,aihub71631,otoSpeech,kec,ztc,nsc"); q.add_argument("--limit-windows", type=int, default=200)
    q.add_argument("--deltas", type=lambda s: [int(x) for x in s.split(",")], default=[10])
    q.add_argument("--backchannel", default=(1.0, 3), type=lambda s: (float(s.split(",")[0]), int(s.split(",")[1])))
    q = sub.add_parser("summarize"); q.add_argument("--out", required=True); q.add_argument("--attr", default=None)
    a = ap.parse_args(); {"run": run, "summarize": summarize}[a.cmd](a)


if __name__ == "__main__":
    main()

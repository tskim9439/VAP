#!/usr/bin/env python
"""Stage 1 probe 학습·평가 (task-stage1-encoder-probing).

  python experiments/train_probe.py --encoder cpc [--frame-hz 12.5] [--train otoSpeech,aihub-ts01-5] [--val aihub-vs02] [--epochs 4]

- 캐시된 frozen 특징 위에 고정 용량 causal probe head (VAP 256 + VAD) 를 학습한다.
- 검증: val 창의 CE·VAP top-1·VAD F1.
- TurnBench dev: 캐시된 전체 대화 특징을 20 s context + 5 s step 으로 슬라이딩 추론 → p_now → probs-eot/int.json →
  turnbench.sweep 로 fp ≤ 0.1 operating point → predictions.json → turnbench.score. 결과는 CKPT_EXP_DIR/probe/<run>/ 에.
"""
import os, sys, json, time, math, argparse, random
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.probe import ProbeWindowDataset, FeatureIndex, ProbeHead, resample_feats, collate_probe

ap = argparse.ArgumentParser()
ap.add_argument("--encoder", required=True); ap.add_argument("--frame-hz", type=float, default=None, help="None=특징 고유율")
ap.add_argument("--train", default="otoSpeech,aihub-ts01-5"); ap.add_argument("--val", default="aihub-vs02"); ap.add_argument("--val-frac", type=float, default=0.05, help="val 매니페스트가 없을 때 train 에서 떼는 비율")
ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--d-model", type=int, default=256); ap.add_argument("--layers", type=int, default=2); ap.add_argument("--window-s", type=float, default=20.0)
ap.add_argument("--max-train-windows", type=int, default=None); ap.add_argument("--max-steps", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--no-turnbench", action="store_true"); ap.add_argument("--tag", default="")
ap.add_argument("--eval-only", default=None, help="기존 run 디렉토리: 학습 생략, probe.pt 로드 후 TurnBench 만 재실행")
ap.add_argument("--force", action="store_true", help="results.json 이 있어도 다시 학습")
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
if a.eval_only:   # 저장된 설정으로 인자 복원
    _ck = torch.load(os.path.join(a.eval_only, "probe.pt"), map_location="cpu"); a.encoder = _ck["encoder"]; a.frame_hz = _ck["frame_hz"]; a.d_model = _ck["d_model"]; a.layers = _ck["layers"]
FEAT = os.environ.get("DATA_FEATURE_CACHE_DIR", "/data3/tskim/features"); MAN = os.environ.get("DATA_MANIFEST_DIR", "/data3/tskim/manifests")
dev = "cuda"

def mani(names): return [(os.path.join(MAN, n), n) for n in names.split(",") if n]
train_ds = ProbeWindowDataset(FEAT, a.encoder, mani(a.train), a.window_s, 10.0, a.frame_hz, max_windows=(1 if a.eval_only else a.max_train_windows), seed=a.seed)
assert len(train_ds), f"train 창 없음: {a.encoder} / {a.train} 캐시 확인"
hz = train_ds.frame_hz; D = train_ds.dim
val_ds = ProbeWindowDataset(FEAT, a.encoder, mani(a.val), a.window_s, 20.0, hz, seed=a.seed) if a.val else None
if not val_ds or len(val_ds) == 0:     # train 에서 대화 단위로 분리
    ids = sorted({it[0] for it in train_ds.items}); rng = random.Random(a.seed); rng.shuffle(ids); vids = set(ids[: max(1, int(len(ids) * a.val_frac))])
    val_items = [it for it in train_ds.items if it[0] in vids]; train_ds.items = [it for it in train_ds.items if it[0] not in vids]
    val_ds = ProbeWindowDataset.__new__(ProbeWindowDataset); val_ds.__dict__.update(train_ds.__dict__); val_ds.items = val_items[:2000]
run = f"{a.encoder}-{hz:g}hz-d{a.d_model}L{a.layers}{('-' + a.tag) if a.tag else ''}"
out = a.eval_only or os.path.join(os.environ.get("CKPT_EXP_DIR", "/tmp"), "probe", run); os.makedirs(out, exist_ok=True)
if not a.eval_only and not a.force and os.path.exists(os.path.join(out, "results.json")):
    print(f"skip {run}: results.json 존재 (--force 로 재실행)", flush=True); sys.exit(0)
print(f"run {run}: D={D} hz={hz} | train {len(train_ds)} windows, val {len(val_ds)} | out {out}", flush=True)

model = ProbeHead(D, hz, a.d_model, a.layers).to(dev)
print(f"probe params: head {model.n_params()/1e6:.2f}M (+proj {model.n_params(True)/1e6:.2f}M total)", flush=True)
dl = torch.utils.data.DataLoader(train_ds, a.bs, shuffle=True, num_workers=6, collate_fn=collate_probe, drop_last=True, persistent_workers=True)
vdl = torch.utils.data.DataLoader(val_ds, a.bs, shuffle=False, num_workers=4, collate_fn=collate_probe)
steps_total = max(1, a.max_steps or a.epochs * len(dl)); opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 200) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total))))

def run_val():
    model.eval(); tot = dict(ce=0.0, acc=0.0, vad_f1=0.0, n=0)
    with torch.inference_mode():
        for b in vdl:
            vl, vd = model(b["feats"].to(dev)); y = b["vap_label"].to(dev); m = y >= 0
            tot["ce"] += F.cross_entropy(vl[m], y[m]).item() * m.sum().item(); tot["acc"] += (vl.argmax(-1)[m] == y[m]).float().sum().item(); tot["n"] += m.sum().item()
            p = (torch.sigmoid(vd) > 0.5).float(); t = b["vad"].to(dev); tp = (p * t).sum(); tot["vad_f1"] += (2 * tp / (p.sum() + t.sum() + 1e-8)).item() * m.sum().item()
    model.train(); n = max(1, tot["n"]); return dict(ce=tot["ce"] / n, acc=tot["acc"] / n, vad_f1=tot["vad_f1"] / n)

step = 0; t0 = time.time(); hist = []; model.train(); best = (1e9, None)
if a.eval_only:
    _prev = json.load(open(os.path.join(out, "results.json"))) if os.path.exists(os.path.join(out, "results.json")) else {}
    hist = _prev.get("val", []); best = (0, _prev.get("best_epoch")); a.epochs = 0
for ep in range(a.epochs):
    for b in dl:
        vl, vd = model(b["feats"].to(dev)); y = b["vap_label"].to(dev); m = y >= 0
        loss_vap = F.cross_entropy(vl[m], y[m]); loss_vad = F.binary_cross_entropy_with_logits(vd, b["vad"].to(dev)); loss = loss_vap + loss_vad
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        if step % 100 == 0: print(f"  ep {ep} step {step}/{steps_total} loss {loss.item():.3f} (vap {loss_vap.item():.3f} vad {loss_vad.item():.3f}) {time.time()-t0:.0f}s", flush=True)
        if step >= steps_total: break
    v = run_val(); hist.append(dict(epoch=ep, step=step, **v)); print(f"[val] ep {ep} ce {v['ce']:.4f} acc {v['acc']:.4f} vad_f1 {v['vad_f1']:.4f}", flush=True)
    if v["ce"] < best[0]: best = (v["ce"], ep); torch.save(dict(state=model.state_dict(), d_in=D, frame_hz=hz, d_model=a.d_model, layers=a.layers, encoder=a.encoder), os.path.join(out, "probe.pt"))
    if step >= steps_total: break
model.load_state_dict(torch.load(os.path.join(out, "probe.pt"), map_location=dev)["state"]); model.eval()
results = dict(run=run, encoder=a.encoder, frame_hz=hz, d_in=D, train_windows=(_prev.get("train_windows") if a.eval_only else len(train_ds)), val=hist, best_epoch=best[1], train_sec=(_prev.get("train_sec") if a.eval_only else time.time() - t0))

# ───────────────────────────── TurnBench dev ─────────────────────────────
def infer_full(f_np, ctx_s=20.0, step_s=5.0):
    """(2, T, D) 전체 대화 → p_now (T, 2): 20 s context + 5 s step 슬라이딩 (baselines/vap 와 동일 규약)."""
    f = torch.from_numpy(f_np.astype(np.float32)); T = f.shape[1]; ctx, stp = int(round(ctx_s * hz)), int(round(step_s * hz)); chunk = ctx + stp
    if T <= chunk: return model.probs(f[None].to(dev))["p_now"][0].numpy()
    outs = []; start = 0
    while True:
        end = min(T, start + chunk); p = model.probs(f[None, :, start:end].to(dev))["p_now"][0].numpy()
        outs.append(p if start == 0 else p[-(end - start - ctx):] if end - start > ctx else p[-1:]); start += stp
        if end >= T: break
    p = np.concatenate(outs)[:T]
    return p if len(p) == T else np.concatenate([p, np.repeat(p[-1:], T - len(p), 0)])

# TurnBench causality 규약: 타임스탬프는 "판단에 쓰인 오디오를 모두 들은 시각". 인코더 lookahead 를 확률 시계열에 접어 넣는다.
#   cpc / nemotron-c0 / qwen-aut-causal / fbank : ≤1 frame — VAP baseline 과 같은 프레임 규약이므로 그대로
#   qwen-aut-cc1s : 1 s 블록 내 양방향 → 블록 안의 모든 프레임 값은 블록 끝에서야 관측 가능 → 블록 끝으로 밀되 블록 내 max 유지
#   wavlm-*       : 20 s 창 내 양방향 → causal 화 불가. 비인과 참조로만 보고 (results 에 표기)
def fold_lookahead(p, enc_name, hz):
    if enc_name.startswith("qwen-aut-cc1s"):
        blk = int(round(1.0 * hz)); out = p.copy(); T = len(p)
        for b in range(0, T, blk):
            e = min(T, b + blk); m = p[b:e].max(0); out[b:e] = p[b:e].min(0)   # 블록 끝 이전엔 '아직 모름' → 보수적으로 최소
            out[e - 1] = m                                                    # 블록 끝 프레임에 블록 내 최대값
        return out
    return p
NONCAUSAL = a.encoder.startswith("wavlm")
if not a.no_turnbench:
    tb = FeatureIndex(FEAT, a.encoder, "turnbench-dev")
    if not tb.rows: print("turnbench-dev 특징 캐시 없음 → 건너뜀", flush=True)
    else:
        sys.path.insert(0, "/data3/tskim/third_party/turnbench")
        from turnbench.sweep import ProbsFile, ConversationProbs, SpeakerProbs, load_probs, sweep, operating_point, commit_events, frame_count
        from turnbench.submission import Submission, ConversationPrediction, SpeakerEvents
        from turnbench.data import resolve_dataset, DEV_DATASET
        man = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(MAN, "turnbench-dev", "manifest.jsonl"))}
        scores = []
        for cid, fr in tb.rows.items():
            f = np.load(fr["npy"], mmap_mode="r"); f = np.asarray(f)
            if abs(tb.frame_hz - hz) > 1e-6: f = resample_feats(f.astype(np.float32), tb.frame_hz, hz)
            p = fold_lookahead(infer_full(f), a.encoder, hz); dur = man[cid]["duration"]; n = frame_count(dur, hz)
            p = p[:n] if len(p) >= n else np.concatenate([p, np.repeat(p[-1:], n - len(p), 0)])
            scores.append((cid.split("tb-")[1], p[:, 0], p[:, 1], dur))
        ds = resolve_dataset(source=DEV_DATASET); tbres = {}
        for task in ("eot", "int"):
            pf = ProbsFile(schema_version=1, task=task, frame_rate_hz=hz, probs=[ConversationProbs(conversation_id=c, speaker_1=SpeakerProbs(prob=((1 - p1) if task == "eot" else p1).clip(0, 1).tolist()), speaker_2=SpeakerProbs(prob=((1 - p2) if task == "eot" else p2).clip(0, 1).tolist())) for c, p1, p2, _ in scores])
            pp = os.path.join(out, f"probs-{task}.json"); open(pp, "w").write(pf.model_dump_json()); rows = sweep(load_probs(pp), ds); op = operating_point(rows)
            tbres[task] = dict(theta=op.theta if op else None, recall=op.recall if op else None, fp_rate=op.fp_rate if op else None, lat_p50=op.lat_p50 if op else None,
                               sweep=[dict(theta=r.theta, recall=r.recall, fp_rate=r.fp_rate, lat_p50=r.lat_p50) for r in rows])
            print(f"[turnbench dev] {task.upper()} @fp≤0.1: theta {tbres[task]['theta']} recall {tbres[task]['recall']} fp {tbres[task]['fp_rate']} lat_p50 {tbres[task]['lat_p50']}", flush=True)
        te, ti = tbres["eot"]["theta"] or 0.9, tbres["int"]["theta"] or 0.9
        sub = Submission(schema_version=1, predictions=[ConversationPrediction(conversation_id=c, speaker_1=SpeakerEvents(eot=commit_events((1 - p1).clip(0, 1).tolist(), hz, te), interruption=commit_events(p1.clip(0, 1).tolist(), hz, ti)),
                                                                                speaker_2=SpeakerEvents(eot=commit_events((1 - p2).clip(0, 1).tolist(), hz, te), interruption=commit_events(p2.clip(0, 1).tolist(), hz, ti))) for c, p1, p2, _ in scores])
        sp = os.path.join(out, "predictions-dev.json"); open(sp, "w").write(sub.model_dump_json()); results["turnbench_dev"] = tbres
        results["causality"] = "non-causal reference (20 s window)" if NONCAUSAL else ("block-end folded (1 s)" if a.encoder.startswith("qwen-aut-cc1s") else "causal (≤1 frame)")
        rc = os.system(f"cd /data3/tskim/third_party/turnbench && python -m turnbench.score {sp} > {out}/score-dev.txt 2>&1")
        txt = open(os.path.join(out, "score-dev.txt")).read(); print("\n".join(l for l in txt.splitlines() if "│  EOT" in l or "│  INT" in l or "Error" in l or "missing" in l) or txt[-600:], flush=True)
        results["score_rc"] = rc
json.dump(results, open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False); print("saved", os.path.join(out, "results.json"))

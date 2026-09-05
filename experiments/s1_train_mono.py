#!/usr/bin/env python
"""Stage 1 — 단일 화자 mono 스트리밍 ASR 파일럿 학습 + 평가 (plans/stage1-mono-pilot.md §6·§7).

  overfit  : python experiments/s1_train_mono.py --overfit 16 --steps 300 --tag overfit      # EN 16 + KO 16 고정, δ=2, 같은 32 개로 학습·디코드
  pilot    : python experiments/s1_train_mono.py --tag pilot                                  # random init 6 k step, 1 k 마다 sentinel dev 평가·ckpt
  select   : python experiments/s1_train_mono.py --tag pilot --select 4000,5000,6000          # 큰 dev 표본으로 ckpt·bias 선택 → best.pt, results.json
  final    : python experiments/s1_train_mono.py --tag pilot --final                          # best.pt 로 보고 세트 전량 1 회 + δ 추종(dev, δ=2,3,4)
평가 = 스트리밍 greedy 디코드(flush 라운드 포함) → WER / CER-official·CER-nospace, 토큰 지연 p50/90/99, evidence 위반, tok/chunk vs 참조율, M 강제, backlog, tick p99.
언어 교대 배치: EN/KO 를 optimizer step 마다 번갈아 쓰되 배치 크기를 언어별로 둔다(--bs-en 2, --bs-ko 8 ≈ 프레임 예산 균형; LibriSpeech 스트림이 KsponSpeech 의 4–5 배 길다).
"""
import os, sys, json, time, math, argparse, random, difflib, subprocess, re, unicodedata, shutil
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--train", default="librispeech-100,kspon-100"); ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--bs-en", type=int, default=2); ap.add_argument("--bs-ko", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lr-adapter", type=float, default=5e-4); ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--delays", default="2,3,4,6"); ap.add_argument("--M", type=int, default=4); ap.add_argument("--next-weight", type=float, default=0.3); ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--eval-bias", default="0,1,2"); ap.add_argument("--eval-delay", type=int, default=2)
ap.add_argument("--sentinel-stream", type=int, default=20, help="학습 중 sentinel dev: 스트림 세트당 수"); ap.add_argument("--sentinel-utt", type=int, default=200, help="〃 발화 세트당 수")
ap.add_argument("--select", default=None, help="쉼표 구분 step 목록: 큰 dev 표본으로 ckpt·bias 선택"); ap.add_argument("--select-stream", type=int, default=200); ap.add_argument("--select-utt", type=int, default=1000)
ap.add_argument("--final", action="store_true", help="best.pt 로 보고 세트 전량 평가"); ap.add_argument("--smoke-eval", action="store_true", help="final/select 를 sentinel 크기로 축소(디버그)")
ap.add_argument("--overfit", type=int, default=0, help="N>0: 언어별 N 개 고정, δ=2, 같은 표본으로 학습·평가"); ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--ckpt", default=None)
ap.add_argument("--tag", default="pilot"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=50); ap.add_argument("--gpu", default=None)
a = ap.parse_args()
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if a.gpu is None:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        a.gpu = str(max([[int(v) for v in l.split(",")] for l in q.stdout.strip().splitlines()], key=lambda r: r[2] - r[1])[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import numpy as np, torch, torch.nn as nn, jiwer
from vapasr.uslm.data import normalize_text, _TAG, _PUNCT
from vapasr.uslm.mono_data import MonoStreamDataset, BucketBatchSampler, collate_streams, CHUNK_S
from vapasr.uslm.mono_model import MonoInterleavedASR
from vapasr.uslm.model import Adapter
torch.manual_seed(a.seed); random.seed(a.seed); dev = "cuda"
out = os.path.join(os.environ.get("MXC_CKPT_EXP_DIR", os.environ.get("CKPT_EXP_DIR", "/tmp")), "uslm", f"s1-mono-{a.tag}"); os.makedirs(os.path.join(out, "eval"), exist_ok=True)
QWEN = os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B")
from qwen_asr import Qwen3ASRModel
qm = Qwen3ASRModel.from_pretrained(QWEN, dtype=torch.bfloat16, device_map=dev, max_new_tokens=8)
root = next(v for v in vars(qm).values() if isinstance(v, nn.Module)); thinker = root.thinker; tok = qm.processor.tokenizer; del root.thinker.audio_tower
delays = (2,) if a.overfit else tuple(int(x) for x in a.delays.split(","))

DEV = [("dev-clean", "librispeech-dev", "dev-clean", "stream"), ("dev-other", "librispeech-dev", "dev-other", "stream"), ("kspon-dev", "kspon-dev", "dev", "utt")]
TEST = [("test-clean/stream", "librispeech-test", "test-clean", "stream"), ("test-clean/utt", "librispeech-test", "test-clean", "utt"),
        ("test-other/stream", "librispeech-test", "test-other", "stream"), ("test-other/utt", "librispeech-test", "test-other", "utt"),
        ("eval_clean", "kspon-eval", "eval_clean", "utt"), ("eval_other-partial[E03314-E06000,n=2687]", "kspon-eval", "eval_other", "utt")]
def make_sets(spec, cap_stream, cap_utt, seed=1):
    return {lab: MonoStreamDataset([m], tok, mode=mode, subsets=[sub], delays=(a.eval_delay,), max_per_chunk=a.M, seed=seed, max_items=(cap_stream if mode == "stream" else cap_utt))
            for lab, m, sub, mode in spec}
training = not (a.eval_only or a.final or a.select)
train_ds = {m: MonoStreamDataset([m], tok, mode="stream", delays=delays, max_per_chunk=a.M, seed=a.seed, max_items=(a.overfit or None)) for m in a.train.split(",")} if training else {}
if a.overfit:   # 같은 표본으로 학습·디코드. 타깃 보존 assert.
    for m, ds in train_ds.items():
        assert all(ds.check_targets(i, 2) for i in range(len(ds))), f"{m}: 시퀀스 라벨이 참조 토큰열과 다름 (flush/이월 규약 오류)"
    dev_sets = {f"overfit/{m}": ds for m, ds in train_ds.items()}; print(f"overfit: 타깃 보존 OK, " + ", ".join(f"{m}:{len(ds)}" for m, ds in train_ds.items()), flush=True)
else:
    dev_sets = make_sets(DEV, a.sentinel_stream, a.sentinel_utt)
sp_ids = next(iter(dev_sets.values())).sp_ids
print("train " + ", ".join(f"{k}:{len(v)} (drop {v.dropped}, no-align {v.no_align})" for k, v in train_ds.items()) + " | dev " + ", ".join(f"{k}:{len(v)}" for k, v in dev_sets.items()), flush=True)

model = MonoInterleavedASR(thinker, tok, Adapter(), sp_ids, lora_r=a.lora_r).to(dev); model.adapter.float()
if a.ckpt: model.load_trainable_state(torch.load(a.ckpt, map_location="cpu")); print("ckpt ←", a.ckpt, flush=True)
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) - model._embed().weight.numel() + len(model.special_rows) * model._embed().weight.shape[1]
print(f"trainable {n_tr/1e6:.1f}M (random init: adapter + LoRA r{a.lora_r} + 특수 토큰 행 {len(model.special_rows)})", flush=True)

# ───────────────────────────── 평가 ─────────────────────────────
def norm_ko(t, spaces: bool):
    t = _TAG.sub(" ", t); t = unicodedata.normalize("NFKC", t).lower(); t = _PUNCT.sub(" ", t); t = re.sub(r"\s+", " ", t).strip()
    return t if spaces else t.replace(" ", "")
def latency_stats(hyp, ref):
    h_ids = [t for _, t in hyp]; r_ids = [t for t, _ in ref]; lat = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, h_ids, r_ids, autojunk=False).get_opcodes():
        if tag == "equal":
            for di in range(i2 - i1): lat.append((hyp[i1 + di][0] + 1) * CHUNK_S - ref[j1 + di][1])
    return lat
def pct(x, p): return float(np.percentile(x, p)) if len(x) else None

@torch.no_grad()
def eval_set(ds, bias, delay):
    lang = ds.items[0]["lang"] if ds.items else "English"; R, H, lat, forced, chunks, n_tok, n_ref, backlog, ticks, rounds, top1 = [], [], [], 0, 0, 0, 0, [], [], [], []
    for i in range(len(ds)):
        f, ref, _, st, it = ds.stream(i, delay)
        with torch.autocast("cuda", dtype=torch.bfloat16): emitted, fc, tk, rd = model.stream_decode(torch.from_numpy(f).to(dev), ds.prefix(lang, delay), a.M, next_bias=bias)
        ticks += tk; rounds.append(rd); forced += fc; chunks += it["K"]; n_tok += len(emitted); n_ref += len(ref); backlog.append(st.max_backlog)
        R.append(tok.decode([t for t, _ in ref])); H.append(tok.decode([t for _, t in emitted])); lat += latency_stats(emitted, ref)
    lat = np.array(lat); ticks = np.array(ticks); m = int(len(lat))
    r = dict(bias=bias, delay=delay, n=len(ds), matched=m, latency_available=m > 0,
             lat_p50=pct(lat, 50), lat_p90=pct(lat, 90), lat_p99=pct(lat, 99), viol=(float((lat < 0).mean()) if m else None), viol_80ms=(float((lat < -0.08).mean()) if m else None),
             tok_per_chunk=n_tok / max(1, chunks), ref_per_chunk=n_ref / max(1, chunks), forced_frac=forced / max(1, chunks), backlog_p99=pct(np.array(backlog), 99),
             flush_rounds_mean=float(np.mean(rounds)) if rounds else None, tick_ms_p50=pct(ticks, 50), tick_ms_p99=pct(ticks, 99), example=(R[0][:60], H[0][:60]) if R else None)
    if lang == "Korean":
        r["cer_official"] = jiwer.cer([norm_ko(x, True) for x in R], [norm_ko(x, True) for x in H]); r["cer_nospace"] = jiwer.cer([norm_ko(x, False) for x in R], [norm_ko(x, False) for x in H]); r["err"] = r["cer_nospace"]
    else:
        r["wer"] = jiwer.wer([normalize_text(x, "English") for x in R], [normalize_text(x, "English") for x in H]); r["err"] = r["wer"]
    return r

def evaluate(sets, biases, delay, label):
    model.eval()
    if a.lora_r > 0: model.thinker.merge_adapter()
    res = {}
    for name, ds in sets.items():
        runs = {b: eval_set(ds, b, delay) for b in biases}; best_b = min(runs, key=lambda b: runs[b]["err"]); b0 = runs.get(0.0, runs[min(runs)])
        res[name] = dict(bias0=b0, best=runs[best_b], best_bias=best_b)
        v80 = "n/a" if b0["viol_80ms"] is None else f"{b0['viol_80ms']:.4f}"; p50 = "n/a" if b0["lat_p50"] is None else f"{b0['lat_p50']*1000:.0f}ms"
        print(f"  [{label}] {name}: bias0 err {b0['err']:.3f} tok/chunk {b0['tok_per_chunk']:.3f} (ref {b0['ref_per_chunk']:.3f}) matched {b0['matched']} viol80 {v80} p50 {p50} "
              f"tick p99 {b0['tick_ms_p99']:.1f}ms | best bias {best_b} err {runs[best_b]['err']:.3f}", flush=True)
    if a.lora_r > 0: model.thinker.unmerge_adapter()
    model.train(); torch.cuda.empty_cache(); return res
biases = [float(x) for x in a.eval_bias.split(",")]

if a.eval_only:
    r = evaluate(dev_sets, biases, a.eval_delay, "dev"); json.dump(r, open(os.path.join(out, "eval", "eval-only.json"), "w"), indent=1, ensure_ascii=False); sys.exit(0)
if a.select:   # 큰 dev 표본(고정 seed)으로 ckpt·bias 선택 → best.pt
    sel = make_sets(DEV, a.sentinel_stream if a.smoke_eval else a.select_stream, a.sentinel_utt if a.smoke_eval else a.select_utt, seed=7); table = {}
    for s in a.select.split(","):
        model.load_trainable_state(torch.load(os.path.join(out, f"ckpt-{s}.pt"), map_location="cpu")); r = evaluate(sel, biases, a.eval_delay, f"select@{s}")
        table[s] = dict(score=float(np.mean([v["best"]["err"] for v in r.values()])), best_bias=float(np.median([v["best_bias"] for v in r.values()])), sets=r)
    best = min(table, key=lambda s: table[s]["score"]); shutil.copy(os.path.join(out, f"ckpt-{best}.pt"), os.path.join(out, "best.pt"))
    res = json.load(open(os.path.join(out, "results.json"))) if os.path.exists(os.path.join(out, "results.json")) else {}
    res.update(selection=table, best_step=best, best_bias=table[best]["best_bias"], select_sizes=dict(stream=a.select_stream, utt=a.select_utt))
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False); print(f"select → best step {best} score {table[best]['score']:.4f} bias {table[best]['best_bias']}"); sys.exit(0)
if a.final:    # best.pt 로 보고 세트 **전량** 1 회 + δ 추종(dev)
    if not a.ckpt: model.load_trainable_state(torch.load(os.path.join(out, "best.pt"), map_location="cpu")); print("best.pt 로드", flush=True)
    test_sets = make_sets(TEST, a.sentinel_stream if a.smoke_eval else None, a.sentinel_utt if a.smoke_eval else None)
    res = json.load(open(os.path.join(out, "results.json"))) if os.path.exists(os.path.join(out, "results.json")) else {}; devb = float(res.get("best_bias", 0.0))
    final = dict(note="eval_other 는 서버 결손(E03001–E03313 없음)으로 partial. 공개 3,000 개 수치와 직접 비교하지 말 것; 대조군도 같은 2,687 개로.",
                 sizes={k: len(v) for k, v in test_sets.items()}, dev_best_bias=devb, test=evaluate(test_sets, sorted({0.0, devb}), a.eval_delay, "test"),
                 delta_sweep={d: evaluate(dev_sets, [0.0], d, f"dev δ={d}") for d in (2, 3, 4)})
    json.dump(final, open(os.path.join(out, "eval", "final.json"), "w"), indent=1, ensure_ascii=False); print("final →", os.path.join(out, "eval", "final.json")); sys.exit(0)

# ───────────────────────────── 학습 ─────────────────────────────
def bs_of(name): return a.bs_ko if name.startswith("kspon") else a.bs_en
loaders = {m: torch.utils.data.DataLoader(ds, batch_sampler=BucketBatchSampler(ds, min(bs_of(m), len(ds)), seed=a.seed, drop_last=not a.overfit), num_workers=4, collate_fn=collate_streams) for m, ds in train_ds.items()}
def cycle(dl):
    while True:
        for b in dl: yield b
its = [cycle(dl) for dl in loaders.values()]
emb_w = model._embed().weight   # 임베딩 행렬: 특수 토큰 행만 grad(마스크) — weight decay 를 걸면 전 행(tied lm_head 포함)이 감쇠하므로 wd=0
groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("adapter")], "lr": a.lr_adapter, "weight_decay": a.wd},
          {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("adapter") and p is not emb_w], "lr": a.lr, "weight_decay": a.wd},
          {"params": [emb_w], "lr": a.lr, "weight_decay": 0.0}]
assert sum(len(g["params"]) for g in groups) == sum(1 for p in model.parameters() if p.requires_grad)
opt = torch.optim.AdamW(groups)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / max(1, a.warmup)) * 0.5 * (1 + math.cos(math.pi * min(1, s / max(1, a.steps)))))
hist, best_score, best_bias = [], None, 0.0; step = 0; t0 = time.time(); acc = {}; model.train(); n_lab = n_next = n_flush = 0
while step < a.steps:
    b = next(its[step % len(its)])                                         # 언어 교대 (EN, KO, EN, …) — optimizer step 기준 1:1, 배치 크기는 언어별
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, parts = model(b["feats"].to(dev), b["ids"].to(dev), b["is_audio"].to(dev), b["chunk_of"].to(dev), b["labels"].to(dev), b["mask"].to(dev), next_weight=a.next_weight)
    loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
    for k, v in parts.items(): acc[k] = acc.get(k, 0) + v
    lab = b["labels"]; n_lab += int((lab != -100).sum()); n_next += int((lab == sp_ids["<NEXT_AUDIO>"]).sum()); n_flush += b["n_flush"]
    if step % a.log_every == 0:
        print(f"  step {step}/{a.steps} loss {loss.item():.3f} next {acc['loss_next']/a.log_every:.3f} text {acc['loss_text']/a.log_every:.3f} top1 {acc['top1_text']/a.log_every:.3f} "
              f"NEXT비율 {n_next/max(1,n_lab):.3f} flush/{a.log_every}step {n_flush} L {b['ids'].shape[1]} K {b['feats'].shape[2]} {b['lang'][0][:2]} {time.time()-t0:.0f}s", flush=True)
        acc = {}; n_lab = n_next = n_flush = 0
    if step % a.eval_every == 0 or step == a.steps:
        r = evaluate(dev_sets, biases, a.eval_delay, ("overfit" if a.overfit else "sentinel") + f"@{step}"); score = float(np.mean([v["best"]["err"] for v in r.values()]))
        hist.append(dict(step=step, score=score, **{k: dict(bias0_err=v["bias0"]["err"], best_err=v["best"]["err"], best_bias=v["best_bias"], tok_per_chunk=v["bias0"]["tok_per_chunk"], matched=v["bias0"]["matched"]) for k, v in r.items()}))
        json.dump(r, open(os.path.join(out, "eval", f"{'overfit' if a.overfit else 'sentinel'}-{step}.json"), "w"), indent=1, ensure_ascii=False)
        st = dict(**model.trainable_state(), step=step, args=vars(a)); torch.save(st, os.path.join(out, f"ckpt-{step}.pt"))
        if best_score is None or score < best_score: best_score, best_bias = score, float(np.median([v["best_bias"] for v in r.values()]))
        json.dump(dict(args=vars(a), hist=hist, sentinel_best_score=best_score, best_bias=best_bias, trainable_m=n_tr / 1e6, train_streams={k: len(v) for k, v in train_ds.items()},
                       note="sentinel 은 추세 관찰용(작은 고정 표본). ckpt 선택은 --select 로 큰 표본에서."), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
print("saved", out)

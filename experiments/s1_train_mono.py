#!/usr/bin/env python
"""Stage 1 — 단일 화자 mono 스트리밍 ASR 학습 + 평가 (plans/stage1-mono-pilot.md §6·§7). 단일 GPU 와 torchrun DDP(노드당 8 GPU) 모두 지원.

  overfit : python experiments/s1_train_mono.py --overfit 16 --steps 900 --tag overfit
  pilot   : python experiments/s1_train_mono.py --tag pilot
  DDP     : torchrun --nproc_per_node=8 experiments/s1_train_mono.py --tag A --epochs 15 --bs-en 12 --bs-ko 48 --out-dir /soundai/Model/VAPASR/s1-A ...
  select  : python experiments/s1_train_mono.py --tag A --out-dir … --select 3000,4000,5000    (단일 GPU)
  final   : python experiments/s1_train_mono.py --tag A --out-dir … --final                     (단일 GPU, 보고 세트 전량)
재개: --resume auto(기본) 는 out-dir 의 ckpt-last.pt(모델·옵티마이저·스케줄러·step) 에서 이어 간다. SLURM 선점: out-dir 에 PREEMPT 파일이 생기거나
SIGUSR1/SIGTERM 을 받으면 다음 step 경계에서 ckpt-last 를 저장하고 0 으로 종료 → --requeue 로 재시작하면 이어서 학습.
언어 교대 배치(EN/KO 를 optimizer step 마다 번갈아, 배치 크기는 언어별 — bs 2/8 ≈ 프레임 예산 균형). --epochs 를 주면 steps = epochs × (EN 배치 수 + KO 배치 수).
"""
import os, sys, json, time, math, argparse, random, difflib, subprocess, signal, glob, shutil
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--train", default="librispeech-100,kspon-100"); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--epochs", type=float, default=0, help=">0 이면 steps 를 epoch 로 계산")
ap.add_argument("--bs-en", type=int, default=2); ap.add_argument("--bs-ko", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-4, help="LoRA lr (full FT 면 thinker lr, 권장 2e-5~5e-5)"); ap.add_argument("--lr-adapter", type=float, default=5e-4); ap.add_argument("--warmup", type=int, default=300); ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--delays", default="2,3,4,6"); ap.add_argument("--M", type=int, default=0, help="청크당 텍스트 토큰 상한(0 = 제한 없음, 기본)"); ap.add_argument("--next-weight", type=float, default=0.3); ap.add_argument("--lora-r", type=int, default=16)
ap.add_argument("--next-weight-ko", type=float, default=None, help="KO 배치의 <NEXT_AUDIO> 가중치(기본 --next-weight). KO 과소 방출 대응")
ap.add_argument("--full-ft", action="store_true", help="thinker 0.6B 전체 학습(LoRA 없음)"); ap.add_argument("--features", default="online", help="online: 학습 중 오디오→Nemotron(기본) | cache: 특징 캐시(index.jsonl)"); ap.add_argument("--train-encoder", action="store_true", help="Nemotron encoder 도 학습(동결 해제)"); ap.add_argument("--lr-encoder", type=float, default=1e-5); ap.add_argument("--no-grad-ckpt", action="store_true", help="thinker gradient checkpointing 끄기(기본 켬: 최장 KO 배치 48×~1k 토큰이 140 GB 를 넘김)"); ap.add_argument("--init-adapter", default=None, help="s1_distill_adapter.py 의 adapter.pt 로 adapter 초기화(증류 init)")
ap.add_argument("--eval-every", type=int, default=1000); ap.add_argument("--eval-bias", default="0"); ap.add_argument("--eval-delay", type=int, default=2)
ap.add_argument("--sentinel-stream", type=int, default=20); ap.add_argument("--sentinel-utt", type=int, default=200)
ap.add_argument("--select", default=None); ap.add_argument("--select-stream", type=int, default=200); ap.add_argument("--select-utt", type=int, default=1000)
ap.add_argument("--final", action="store_true"); ap.add_argument("--smoke-eval", action="store_true"); ap.add_argument("--overfit", type=int, default=0); ap.add_argument("--eval-only", action="store_true"); ap.add_argument("--ckpt", default=None)
ap.add_argument("--out-dir", default=None, help="산출물 디렉토리(기본 $MXC_CKPT_EXP_DIR/uslm/s1-mono-<tag>)"); ap.add_argument("--resume", default="auto", help="auto | none | <ckpt-last.pt>")
ap.add_argument("--save-every", type=int, default=300, help="ckpt-last.pt(재개용, 옵티마이저 포함) 저장 주기")
ap.add_argument("--tag", default="pilot"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=50); ap.add_argument("--gpu", default=None)
a = ap.parse_args()
# ───────────────────────────── 분산 설정 ─────────────────────────────
rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
if world == 1 and "CUDA_VISIBLE_DEVICES" not in os.environ:
    if a.gpu is None:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        a.gpu = str(max([[int(v) for v in l.split(",")] for l in q.stdout.strip().splitlines()], key=lambda r: r[2] - r[1])[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import numpy as np, torch, torch.nn as nn, jiwer
if world > 1:
    import torch.distributed as dist; from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(hours=3)); torch.cuda.set_device(local)   # rank 0 의 sentinel 평가(수십 분) 동안 다른 rank 가 barrier 에서 기다린다 — 기본 10 분이면 죽는다
dev = f"cuda:{local}" if world > 1 else "cuda"; main = rank == 0
def log(*s):
    if main: print(*s, flush=True)
def barrier():
    if world > 1: dist.barrier()
from vapasr.data.textnorm import score_en, score_ko, TEXTNORM_VERSION
from vapasr.uslm.mono_data import MonoStreamDataset, BucketBatchSampler, collate_streams, CHUNK_S
from vapasr.uslm.mono_model import MonoInterleavedASR
from vapasr.uslm.model import Adapter
torch.manual_seed(a.seed + rank); random.seed(a.seed + rank); torch.backends.cuda.matmul.allow_tf32 = True
out = a.out_dir or os.path.join(os.environ.get("MXC_CKPT_EXP_DIR", os.environ.get("CKPT_EXP_DIR", "/tmp")), "uslm", f"s1-mono-{a.tag}")
if main: os.makedirs(os.path.join(out, "eval"), exist_ok=True)
barrier()
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
    return {lab: MonoStreamDataset([m], tok, mode=mode, subsets=[sub], delays=(a.eval_delay,), max_per_chunk=a.M, seed=seed, max_items=(cap_stream if mode == "stream" else cap_utt), online=a.features == "online")
            for lab, m, sub, mode in spec}
training = not (a.eval_only or a.final or a.select)
if world > 1 and not main: barrier()            # rank 0 이 정렬 항목 캐시(_items.json.gz)를 먼저 만들고, 나머지는 그것을 읽는다 (Lustre 소파일 7.5만 개 ×8 회피)
train_ds = {m: MonoStreamDataset([m], tok, mode="stream", delays=delays, max_per_chunk=a.M, seed=a.seed, online=a.features == "online") for m in a.train.split(",")} if training else {}
if a.overfit:   # 같은 표본으로 학습·디코드. 표적 사례(KO 숫자·라틴 이중표기 / EN 발화 경계)를 절반 이상 포함시키고 ID·커버리지를 출력. 타깃 보존 assert.
    import re
    from vapasr.data.kspon import read_trn, DUAL
    MANROOT = os.environ.get("MXC_DATA_MANIFEST_DIR", os.environ.get("DATA_MANIFEST_DIR", "/tmp"))
    for m, ds in train_ds.items():
        rows = {}
        for l in open(os.path.join(MANROOT, m, "streams.jsonl"), encoding="utf-8"): r = json.loads(l); rows[r["id"]] = r
        if m.startswith("kspon"):
            raw = {os.path.splitext(os.path.basename(rel))[0]: t for rel, t in read_trn(os.path.join(os.environ["MXC_KSPONSPEECH_DIR"], "train.trn"))}
            flag = lambda it: any(re.search(r"[0-9A-Za-z]", x) for x, _ in DUAL.findall(raw.get(rows[it["id"]]["segments"][0]["utt_id"], ""))); label = "숫자·라틴 이중표기→발음형"
        else:
            flag = lambda it: rows[it["id"]]["n_utts"] >= 2; label = "발화 경계(선행 공백)"
        tgt = [it for it in ds.items if flag(it)]; rest = [it for it in ds.items if not flag(it)]
        n_t = min(len(tgt), max(a.overfit // 2, 1)); ds.items = (tgt[:n_t] + rest[: a.overfit - n_t])[: a.overfit]
        cov = sum(flag(it) for it in ds.items); assert cov > 0, f"{m}: 표적 사례({label})가 표본에 없음 — 별도 회귀 표본 필요"
        log(f"overfit 표본 {m}: {len(ds.items)} 개 · {label} {cov} 개 (후보 {len(tgt)}/{len(tgt)+len(rest)})\n  ids: {[it['id'] for it in ds.items]}")
        for it in ds.items[:2] if flag(ds.items[0]) else []: log(f"  예: {it['id']} → {it['text'][:70]}")
        assert all(ds.check_targets(i, 2) for i in range(len(ds))), f"{m}: 시퀀스 라벨이 참조 토큰열과 다름 (flush/이월 규약 오류)"
    dev_sets = {f"overfit/{m}": ds for m, ds in train_ds.items()}; log(f"overfit: 타깃 보존 OK, " + ", ".join(f"{m}:{len(ds)}" for m, ds in train_ds.items()))
else:
    dev_sets = make_sets(DEV, a.sentinel_stream, a.sentinel_utt) if main or not training else {}
if world > 1 and main: barrier()                # 캐시 생성 완료 → 다른 rank 진행
sp_ids = (next(iter(dev_sets.values())) if dev_sets else next(iter(train_ds.values()))).sp_ids
log("train " + ", ".join(f"{k}:{len(v)} (drop {v.dropped}, no-align {v.no_align})" for k, v in train_ds.items()) + " | dev " + ", ".join(f"{k}:{len(v)}" for k, v in dev_sets.items()) + f" | world {world}")

encoder = None
if a.features == "online":
    from vapasr.features.online import NemotronOnline
    encoder = NemotronOnline().set_trainable(a.train_encoder); log(f"온라인 인코더 Nemotron [56,0] ({'학습' if a.train_encoder else '동결'})")
model = MonoInterleavedASR(thinker, tok, Adapter(), sp_ids, lora_r=a.lora_r, full_ft=a.full_ft, encoder=encoder).to(dev); model.adapter.float()
if not a.no_grad_ckpt: model._lm().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})   # 활성화 메모리 ≈5.9 MB/token → 수 배 절감, 연산 +30 %
if a.init_adapter:
    st0 = torch.load(a.init_adapter, map_location="cpu"); model.adapter.load_state_dict(st0["adapter"]); log(f"adapter 증류 init ← {a.init_adapter} (val cos {st0.get('hist', [{}])[-1].get('cos', float('nan')):.3f})")
if a.ckpt: model.load_trainable_state(torch.load(a.ckpt, map_location="cpu")); log("ckpt ←", a.ckpt)
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) if a.full_ft else sum(p.numel() for p in model.parameters() if p.requires_grad) - model._embed().weight.numel() + len(model.special_rows) * model._embed().weight.shape[1]
log(f"trainable {n_tr/1e6:.1f}M ({'thinker full FT + adapter' if a.full_ft else f'adapter + LoRA r{a.lora_r} + 특수 토큰 행 {len(model.special_rows)}'}; adapter {'증류 init' if a.init_adapter else 'random'}) → {out}")

# ───────────────────────────── 평가 (rank 0) ─────────────────────────────
def latency_stats(hyp, ref):
    h_ids = [t for _, t in hyp]; r_ids = [t for t, _ in ref]; lat = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, h_ids, r_ids, autojunk=False).get_opcodes():
        if tag == "equal":
            for di in range(i2 - i1): lat.append((hyp[i1 + di][0] + 1) * CHUNK_S - ref[j1 + di][1])
    return lat
def pct(x, p): return float(np.percentile(x, p)) if len(x) else None
@torch.no_grad()
def eval_set(ds, bias, delay):
    lang = ds.items[0]["lang"] if ds.items else "English"; R, H, lat, forced, chunks, n_tok, n_ref, backlog, ticks, rounds = [], [], [], 0, 0, 0, 0, [], [], []
    for i in range(len(ds)):
        f, ref, _, st, it = ds.stream(i, delay)
        with torch.autocast("cuda", dtype=torch.bfloat16): emitted, fc, tk, rd = model.stream_decode(torch.from_numpy(f).to(dev), ds.prefix(lang, delay), a.M, next_bias=bias)
        ticks += tk; rounds.append(rd); forced += fc; chunks += it["K"]; n_tok += len(emitted); n_ref += len(ref); backlog.append(st.max_backlog)
        R.append(tok.decode([t for t, _ in ref])); H.append(tok.decode([t for _, t in emitted])); lat += latency_stats(emitted, ref)
    lat = np.array(lat); ticks = np.array(ticks); m = int(len(lat))
    r = dict(bias=bias, delay=delay, n=len(ds), matched=m, latency_available=m > 0, lat_p50=pct(lat, 50), lat_p90=pct(lat, 90), lat_p99=pct(lat, 99),
             viol=(float((lat < 0).mean()) if m else None), viol_80ms=(float((lat < -0.08).mean()) if m else None), tok_per_chunk=n_tok / max(1, chunks), ref_per_chunk=n_ref / max(1, chunks),
             forced_frac=forced / max(1, chunks), backlog_p99=pct(np.array(backlog), 99), flush_rounds_mean=float(np.mean(rounds)) if rounds else None, tick_ms_p50=pct(ticks, 50), tick_ms_p99=pct(ticks, 99), example=(R[0][:60], H[0][:60]) if R else None)
    if lang == "Korean":
        r["cer_official"] = jiwer.cer([score_ko(x, True) for x in R], [score_ko(x, True) for x in H]); r["cer_nospace"] = jiwer.cer([score_ko(x, False) for x in R], [score_ko(x, False) for x in H]); r["err"] = r["cer_nospace"]
    else:
        r["wer"] = jiwer.wer([score_en(x) for x in R], [score_en(x) for x in H]); r["err"] = r["wer"]
    return r
def evaluate(sets, biases, delay, label):
    model.eval()
    if a.lora_r > 0 and not a.full_ft: model.thinker.merge_adapter()
    res = {}
    for name, ds in sets.items():
        runs = {b: eval_set(ds, b, delay) for b in biases}; best_b = min(runs, key=lambda b: runs[b]["err"]); b0 = runs.get(0.0, runs[min(runs)])
        res[name] = dict(bias0=b0, best=runs[best_b], best_bias=best_b)
        v80 = "n/a" if b0["viol_80ms"] is None else f"{b0['viol_80ms']:.4f}"; p50 = "n/a" if b0["lat_p50"] is None else f"{b0['lat_p50']*1000:.0f}ms"
        print(f"  [{label}] {name}: bias0 err {b0['err']:.3f} tok/chunk {b0['tok_per_chunk']:.3f} (ref {b0['ref_per_chunk']:.3f}) matched {b0['matched']} viol80 {v80} p50 {p50} tick p99 {b0['tick_ms_p99']:.1f}ms | best bias {best_b} err {runs[best_b]['err']:.3f}", flush=True)
    if a.lora_r > 0 and not a.full_ft: model.thinker.unmerge_adapter()
    model.train(); torch.cuda.empty_cache(); return res
biases = [float(x) for x in a.eval_bias.split(",")]

if a.eval_only:
    r = evaluate(dev_sets, biases, a.eval_delay, "dev"); json.dump(r, open(os.path.join(out, "eval", "eval-only.json"), "w"), indent=1, ensure_ascii=False); sys.exit(0)
if a.select:
    sel = make_sets(DEV, a.sentinel_stream if a.smoke_eval else a.select_stream, a.sentinel_utt if a.smoke_eval else a.select_utt, seed=7); table = {}
    for s in a.select.split(","):
        model.load_trainable_state(torch.load(os.path.join(out, f"ckpt-{s}.pt"), map_location="cpu")); r = evaluate(sel, biases, a.eval_delay, f"select@{s}")
        table[s] = dict(score=float(np.mean([v["best"]["err"] for v in r.values()])), best_bias=float(np.median([v["best_bias"] for v in r.values()])), sets=r)
    best = min(table, key=lambda s: table[s]["score"]); shutil.copy(os.path.join(out, f"ckpt-{best}.pt"), os.path.join(out, "best.pt"))
    res = json.load(open(os.path.join(out, "results.json"))) if os.path.exists(os.path.join(out, "results.json")) else {}
    res.update(selection=table, best_step=best, best_bias=table[best]["best_bias"], select_sizes=dict(stream=a.select_stream, utt=a.select_utt))
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False); print(f"select → best step {best} score {table[best]['score']:.4f} bias {table[best]['best_bias']}"); sys.exit(0)
if a.final:
    if not a.ckpt: model.load_trainable_state(torch.load(os.path.join(out, "best.pt"), map_location="cpu")); print("best.pt 로드", flush=True)
    test_sets = make_sets(TEST, a.sentinel_stream if a.smoke_eval else None, a.sentinel_utt if a.smoke_eval else None)
    res = json.load(open(os.path.join(out, "results.json"))) if os.path.exists(os.path.join(out, "results.json")) else {}; devb = float(res.get("best_bias", 0.0))
    final = dict(note="eval_other 는 서버 결손(E03001–E03313 없음)으로 partial. 공개 3,000 개 수치와 직접 비교하지 말 것; 대조군도 같은 2,687 개로.",
                 sizes={k: len(v) for k, v in test_sets.items()}, dev_best_bias=devb, test=evaluate(test_sets, sorted({0.0, devb}), a.eval_delay, "test"),
                 delta_sweep={d: evaluate(dev_sets, [0.0], d, f"dev δ={d}") for d in (2, 3, 4)})
    json.dump(final, open(os.path.join(out, "eval", "final.json"), "w"), indent=1, ensure_ascii=False); print("final →", os.path.join(out, "eval", "final.json")); sys.exit(0)

# ───────────────────────────── 학습 ─────────────────────────────
def bs_of(name): return a.bs_ko if name.startswith("kspon") else a.bs_en
samplers = {m: BucketBatchSampler(ds, min(bs_of(m), len(ds)), seed=a.seed, drop_last=not a.overfit, rank=rank, world=world) for m, ds in train_ds.items()}
loaders = {m: torch.utils.data.DataLoader(ds, batch_sampler=samplers[m], num_workers=4, collate_fn=collate_streams, persistent_workers=False) for m, ds in train_ds.items()}
steps_per_epoch = sum(len(s) for s in samplers.values())
if a.epochs > 0: a.steps = int(round(a.epochs * steps_per_epoch))
log(f"rank 당 배치/epoch: " + ", ".join(f"{m}:{len(s)}" for m, s in samplers.items()) + f" → steps/epoch {steps_per_epoch}, 총 {a.steps} step (epochs {a.steps/max(1,steps_per_epoch):.1f}), 유효 배치 EN {a.bs_en*world} / KO {a.bs_ko*world}")
def cycle(m):
    ep = 0
    while True:
        samplers[m].set_epoch(ep)
        for b in loaders[m]: yield b
        ep += 1
its = [cycle(m) for m in loaders]
emb_w = model._embed().weight   # 임베딩: LoRA 모드는 특수 토큰 행만 grad(마스크). weight decay 는 전 행(tied lm_head)에 걸리므로 wd=0 그룹
groups = [{"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("adapter")], "lr": a.lr_adapter, "weight_decay": a.wd},
          {"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(("adapter", "encoder")) and p is not emb_w], "lr": a.lr, "weight_decay": a.wd},
          {"params": [emb_w], "lr": a.lr, "weight_decay": 0.0},
          {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder")], "lr": a.lr_encoder, "weight_decay": a.wd}]   # --train-encoder 일 때만 비어 있지 않음
groups = [g for g in groups if g["params"]]
assert sum(len(g["params"]) for g in groups) == sum(1 for p in model.parameters() if p.requires_grad)
opt = torch.optim.AdamW(groups)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, s / max(1, a.warmup)) * 0.5 * (1 + math.cos(math.pi * min(1, s / max(1, a.steps)))))
ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=False) if world > 1 else model
hist, best_score, best_bias, step = [], None, 0.0, 0
# ── 재개
last = os.path.join(out, "ckpt-last.pt"); rp = None if a.resume == "none" else (last if a.resume == "auto" else a.resume)
if rp and os.path.exists(rp):
    try: ck = torch.load(rp, map_location="cpu")
    except Exception as e:                        # 저장 도중 선점되어 잘린 파일 → 직전 ckpt-last.prev.pt 로 폴백
        prev = rp + ".prev"; assert os.path.exists(prev), f"ckpt-last 손상({type(e).__name__})이고 .prev 도 없음: {rp}"
        log(f"!! {rp} 로드 실패({type(e).__name__}) → {prev} 로 재개"); ck = torch.load(prev, map_location="cpu"); rp = prev
    model.load_trainable_state(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step = ck["step"]; hist = ck.get("hist", []); best_score, best_bias = ck.get("best_score"), ck.get("best_bias", 0.0)
    for _ in range(step): pass   # 데이터 순서는 epoch 시드로 결정되므로 step 만 복원해도 충분(같은 epoch 안의 정확한 위치 복원은 생략)
    log(f"재개 ← {rp} (step {step})")
# ── 선점 신호
stop = {"flag": False}
def _sig(*_): stop["flag"] = True
signal.signal(signal.SIGUSR1, _sig); signal.signal(signal.SIGTERM, _sig)
def save_last():
    if not main: return
    st = dict(model=model.trainable_state(), opt=opt.state_dict(), sched=sched.state_dict(), step=step, hist=hist, best_score=best_score, best_bias=best_bias, args=vars(a))
    if os.path.exists(last):                       # 직전 저장본은 .prev 로 보존(6 GB 저장 중 선점되어 잘리면 재개 시 .prev 로 폴백)
        try: os.replace(last, last + ".prev")
        except OSError: pass
    try:                                           # 원자적 교체 시도. /soundai(Azure Blob NFS) 는 방금 쓴 파일의 rename 이 실패할 수 있어 직접 쓰기로 폴백
        torch.save(st, last + ".tmp"); os.replace(last + ".tmp", last)
    except OSError as e:
        print(f"  (ckpt-last rename 실패 {type(e).__name__} → 직접 저장)", flush=True); torch.save(st, last)
        try: os.remove(last + ".tmp")
        except OSError: pass
def should_stop():
    f = stop["flag"] or os.path.exists(os.path.join(out, "PREEMPT"))
    if world > 1:
        t = torch.tensor([1 if f else 0], device=dev); dist.all_reduce(t); f = bool(t.item())
    return f
t0 = time.time(); acc = {}; model.train(); n_lab = n_next = n_flush = 0; fr = 0
while step < a.steps:
    b = next(its[step % len(its)])
    nw = a.next_weight_ko if (a.next_weight_ko is not None and b["lang"][0] == "Korean") else a.next_weight
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = dict(wav_len=b["wav_len"].to(dev), K=b["K"].to(dev)) if "wav" in b else {}
        loss, parts = ddp((b["wav"] if "wav" in b else b["feats"]).to(dev), b["ids"].to(dev), b["is_audio"].to(dev), b["chunk_of"].to(dev), b["labels"].to(dev), b["mask"].to(dev), next_weight=nw, **x)
    loss.backward(); torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
    for k, v in parts.items(): acc[k] = acc.get(k, 0) + v
    lab = b["labels"]; n_lab += int((lab != -100).sum()); n_next += int((lab == sp_ids["<NEXT_AUDIO>"]).sum()); n_flush += b["n_flush"]; fr += b["frames"]
    if step % a.log_every == 0:
        log(f"  step {step}/{a.steps} loss {loss.item():.3f} next {acc['loss_next']/a.log_every:.3f} text {acc['loss_text']/a.log_every:.3f} top1 {acc['top1_text']/a.log_every:.3f} "
            f"NEXT비율 {n_next/max(1,n_lab):.3f} flush {n_flush} L {b['ids'].shape[1]} K {int(b['K'].max()) if 'K' in b else b['feats'].shape[2]} {b['lang'][0][:2]} {fr*world*0.08/3600:.2f}h/{a.log_every}step lr {sched.get_last_lr()[1]:.1e} {time.time()-t0:.0f}s")
        acc = {}; n_lab = n_next = n_flush = 0; fr = 0
    if step % a.eval_every == 0 or step == a.steps:
        if main:
            r = evaluate(dev_sets, biases, a.eval_delay, ("overfit" if a.overfit else "sentinel") + f"@{step}"); score = float(np.mean([v["best"]["err"] for v in r.values()]))
            hist.append(dict(step=step, score=score, **{k: dict(bias0_err=v["bias0"]["err"], best_err=v["best"]["err"], best_bias=v["best_bias"], tok_per_chunk=v["bias0"]["tok_per_chunk"], matched=v["bias0"]["matched"]) for k, v in r.items()}))
            json.dump(r, open(os.path.join(out, "eval", f"{'overfit' if a.overfit else 'sentinel'}-{step}.json"), "w"), indent=1, ensure_ascii=False)
            torch.save(dict(**model.trainable_state(), step=step, args=vars(a)), os.path.join(out, f"ckpt-{step}.pt"))
            if best_score is None or score < best_score: best_score, best_bias = score, float(np.median([v["best_bias"] for v in r.values()]))
            json.dump(dict(args=vars(a), textnorm_version=TEXTNORM_VERSION, hist=hist, sentinel_best_score=best_score, best_bias=best_bias, trainable_m=n_tr / 1e6, train_streams={k: len(v) for k, v in train_ds.items()}, steps_per_epoch=steps_per_epoch, world=world,
                           note="sentinel 은 추세 관찰용(작은 고정 표본). ckpt 선택은 --select 로 큰 표본에서."), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
        barrier()
    if step % a.save_every == 0 or step == a.steps: save_last(); barrier()
    if should_stop():
        save_last(); log(f"선점/종료 신호 → ckpt-last 저장 후 종료 (step {step}). 재시작하면 이어서 학습."); barrier(); sys.exit(0)
log("saved", out)
if world > 1: dist.destroy_process_group()

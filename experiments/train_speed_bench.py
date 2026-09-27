#!/usr/bin/env python
"""학습 처리량 벤치 — 같은 모델·데이터로 배치 방식(고정 bs 패딩 vs packing)·Liger·fused AdamW 를 바꿔 가며 optimizer step 을 돌리고
오디오 시간/초·토큰/초·최대 메모리를 잰다(단일 GPU). semcommit_train.py 와 같은 데이터 경로(SemCommitDataset + semcommit_round_robin)를 쓴다.

  python experiments/train_speed_bench.py --init <HF vapasr ckpt> --nemotron-dir <dir> --train-words @w.list --train-labels @l.list \\
      --configs base,liger,pack:16384,pack:24576 --steps 40 --warmup 8 --gpu 0 --out bench.json
설정: base = 고정 bs(EN --bs-en / KO --bs-ko) · Liger 끔 · adamw_torch(현행 semcommit 기본) / liger = base + Liger + fused AdamW /
      budget:N = liger + 최대 길이 기반 동적 배치(샘플 수 × 최장 추정 길이 ≤ N, 패딩) / pack:N = liger + packing(추정 토큰 합 ≤ N).
      샘플 수 상한 --pack-max-bs. Liger 는 되돌릴 수 없어 base 를 먼저 돈다.
측정: 배치를 먼저 메모리로 읽어(data_sec·data_audio_h_per_h = 로더 처리량, 오디오 I/O 병목 확인) 그 배치로 warmup 뒤 steps 의 GPU 계산만 잰다 —
      오디오 초(Σ K·0.08)/초, 라벨 토큰/초, 행 토큰(패딩 포함) 대비 실토큰 비율, torch.cuda.max_memory_allocated.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True); ap.add_argument("--nemotron-dir", default=None)
ap.add_argument("--train-words", action="append", required=True); ap.add_argument("--train-labels", action="append", required=True)
ap.add_argument("--configs", default="base,liger,pack:16384"); ap.add_argument("--steps", type=int, default=40); ap.add_argument("--warmup", type=int, default=8)
ap.add_argument("--bs-en", type=int, default=6); ap.add_argument("--bs-ko", type=int, default=16); ap.add_argument("--pack-max-bs", type=int, default=64)
ap.add_argument("--max-items", type=int, default=0, help="언어별 항목 상한(0 = 전부)"); ap.add_argument("--num-workers", type=int, default=6)
ap.add_argument("--gpu", default=None); ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch
from transformers import AutoTokenizer
from vapasr.hf import VapAsrForStreamingASR
from vapasr.hf.packing import budget_sampler_factory
from vapasr.data.semcommit_dataset import SemCommitDataset, add_semcommit_specials, semcommit_round_robin
from vapasr.speedup import set_attn_impl
def log(*s): print(*s, flush=True)
torch.backends.cuda.matmul.allow_tf32 = True

def _paths(vals): return [q for x in vals for q in ([l.strip() for l in open(x[1:]) if l.strip()] if x.startswith("@") else x.split(",")) if q]
t0 = time.time(); model = VapAsrForStreamingASR.from_pretrained(a.init, encoder_path=a.nemotron_dir); tok = AutoTokenizer.from_pretrained(a.init)
model.add_semantic_tokens(tok, turn=False); sp_ids = add_semcommit_specials(tok)
model.encoder.set_trainable(False)
for p_ in model.encoder.parameters(): p_.requires_grad_(False)
model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.thinker.config.use_cache = False
model = model.cuda().train(); model.encoder.eval(); set_attn_impl(model, "sdpa", log)
log(f"model {time.time()-t0:.0f}s")
W, L = _paths(a.train_words), _paths(a.train_labels); sets = {}
for key, lang in (("en", "English"), ("ko", "Korean")):
    ds = SemCommitDataset(W, L, tok, sp_ids, delays=(2, 3, 4, 6), turn_end=False, hangover_s=0.48, hardneg_weight=1.0, max_items=a.max_items or None, seed=0, online=True,
                          max_per_chunk=0, langs=[lang], tail_margin=2, source_pool="main")
    if len(ds): ds.lang = lang; sets[key] = ds
log("sets " + ", ".join(f"{k}:{len(v)}" for k, v in sets.items()))
bs_of = lambda m: a.bs_ko if sets[m].lang == "Korean" else a.bs_en
res = {}; liger_on = False
for conf in a.configs.split(","):
    pack = int(conf.split(":")[1]) if conf.startswith("pack:") else 0; budget = int(conf.split(":")[1]) if conf.startswith("budget:") else 0
    if conf != "base" and not liger_on:
        from vapasr.hf.liger import apply_liger_to_thinker; log(f"liger: {apply_liger_to_thinker(model)}"); liger_on = True
    fused = conf != "base"
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-8, fused=fused)
    dl = semcommit_round_robin(sets, bs_of, seed=0, num_workers=a.num_workers, sampler_factory=budget_sampler_factory(pack, budget, a.pack_max_bs))
    it = iter(dl); t_ = time.time(); cache = [next(it) for _ in range(a.warmup + a.steps)]; t_data = time.time() - t_   # 데이터 읽기(오디오 I/O)와 GPU 계산을 따로 잰다
    data_audio_s = sum(float(b["K"][:b["pack_n"]].sum() if "pack_n" in b else b["K"].sum()) for b in cache) * 0.08
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    audio_s = lab = row_tok = real_tok = n_samp = 0; tw = None
    for step, b in enumerate(cache):
        if step == a.warmup: torch.cuda.synchronize(); tw = time.time(); audio_s = lab = row_tok = real_tok = n_samp = 0
        x = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        nw = model.config.next_weight_ko if b["lang"][0] == "Korean" else model.config.next_weight
        inp = {k: x[k] for k in ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask", "pos_weight", "position_ids") if k in x}
        with torch.autocast("cuda", dtype=torch.bfloat16): out = model(**inp, next_weight=nw)
        out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        audio_s += float(b["K"].sum()) * 0.08; lab += int(out.n_labels); n_samp += int(b["K"].numel())
        row_tok += int(b["ids"].numel()); real_tok += int(b["mask"].sum()) if "mask" in b else int(b["ids"].numel())
    torch.cuda.synchronize(); dt = time.time() - tw
    r = dict(steps=a.steps, sec=round(dt, 1), sec_per_step=round(dt / a.steps, 3), audio_h_per_gpu_h=round(audio_s / dt, 1), samples_per_s=round(n_samp / dt, 2),
             label_tok_per_s=round(lab / dt), row_tok_per_step=round(row_tok / a.steps), real_tok_frac=round(real_tok / max(1, row_tok), 3),
             data_audio_h_per_h=round(data_audio_s / t_data, 1), data_sec=round(t_data, 1), num_workers=a.num_workers,
             samples_per_step=round(n_samp / a.steps, 1), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), liger=liger_on, fused_adamw=fused, pack_max_tokens=pack, batch_max_tokens=budget,
             batches={m: s.describe() for m, s in dl.samplers.items()} if (pack or budget) else {m: dict(bs=bs_of(m), batches=len(s)) for m, s in dl.samplers.items()})
    res[conf] = r; log(conf, json.dumps(r, ensure_ascii=False))
    del opt, dl, it, cache; torch.cuda.empty_cache()
base = res.get("base")
if base:
    for k, r in res.items(): r["speedup_vs_base"] = round(r["audio_h_per_gpu_h"] / base["audio_h_per_gpu_h"], 2)
json.dump(dict(args=vars(a), results=res), open(a.out, "w"), indent=1, ensure_ascii=False); log("→", a.out)

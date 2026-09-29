#!/usr/bin/env python
"""학습 처리량 벤치 — 같은 모델·데이터로 배치 방식(고정 bs 패딩 vs packing)·Liger·fused AdamW 를 바꿔 가며 optimizer step 을 돌리고
오디오 시간/초·토큰/초·최대 메모리를 잰다(단일 GPU). semcommit_train.py 와 같은 데이터 경로(SemCommitDataset + semcommit_round_robin)를 쓴다.

  python experiments/train_speed_bench.py --init <HF vapasr ckpt> --nemotron-dir <dir> --train-words @w.list --train-labels @l.list \\
      --configs base,liger,budget:16384,budget:16384@varlen,pack:16384@varlen --steps 40 --warmup 8 --gpu 0 --out bench.json
설정: base = 고정 bs(EN --bs-en / KO --bs-ko) · Liger 끔 · adamw_torch(현행 semcommit 기본) / liger = base + Liger + fused AdamW /
      budget:N = liger + 최대 길이 기반 동적 배치(샘플 수 × 최장 추정 길이 ≤ N, 패딩) / pack:N = liger + packing(추정 토큰 합 ≤ N).
      샘플 수 상한 --pack-max-bs. Liger 는 되돌릴 수 없어 base 를 먼저 돈다.
      이름 뒤 @attn 은 thinker attention(vapasr/speedup.set_attn_impl: sdpa 기본 | varlen | eager | flex_attention) — 예 budget:16384@varlen 은
      패딩 배치를 모델 안에서 unpad 해 torch FA2 varlen 으로, pack:16384@varlen 은 packed 행을 cu_seqlens 로 돈다. 결과 키는 설정 문자열 그대로.
측정: 배치를 먼저 메모리로 읽어(data_sec·data_audio_h_per_h = 로더 처리량, 오디오 I/O 병목 확인) 그 배치로 warmup 뒤 steps 의 GPU 계산만 잰다 —
      오디오 초(Σ K·0.08)/초, 라벨 토큰/초, 행 토큰(패딩 포함) 대비 실토큰 비율, torch.cuda.max_memory_allocated.
수치 대조: 배치를 읽기 전에 random·torch 시드를 고정해(워커의 δ 추첨까지) 같은 설정 문자열이면 배치가 비트 단위로 같다. 첫 warmup step 의 loss
      (first_loss, 이 설정의 optimizer step 전)와 그 배치 지문(first_batch = ids·labels·pos_weight·K 의 md5)을 남기고, attention 만 다른 설정
      (budget:N@varlen ↔ budget:N 또는 budget:N@sdpa, 지문 같음)끼리 first_loss_rel_vs_sdpa 를 계산한다 — Liger 를 켠 실제 모델에서 varlen 이 sdpa 와
      bf16 오차 안에서 같은 loss 를 내는지 확인. optimizer 는 lr 0 으로 돈다(step 계산량은 같고 가중치는 설정 사이에 변하지 않는다 — lr 1e-8 이던 이전 판의
      2026-09-29 첫 대조에서는 AdamW 의 부호형 갱신이 6 step 마다 loss 를 0.16–0.24 % 씩 내려 설정 간 loss 비교를 가렸다). 같은 sdpa 설정을 @sdpa 로 한 번 더 넣으면 수치 바닥을 잰다.
"""
import os, sys, json, time, argparse, random, hashlib
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
for spec in a.configs.split(","):
    conf, _, attn = spec.partition("@")
    pack = int(conf.split(":")[1]) if conf.startswith("pack:") else 0; budget = int(conf.split(":")[1]) if conf.startswith("budget:") else 0
    attn_got = set_attn_impl(model, attn or "sdpa", log)
    if conf != "base" and not liger_on:
        from vapasr.hf.liger import apply_liger_to_thinker; log(f"liger: {apply_liger_to_thinker(model)}"); liger_on = True
    fused = conf != "base"
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.0, fused=fused)             # lr 0: 계산은 그대로, 가중치 고정(설정 간 loss 비교)
    dl = semcommit_round_robin(sets, bs_of, seed=0, num_workers=a.num_workers, sampler_factory=budget_sampler_factory(pack, budget, a.pack_max_bs))
    random.seed(0); torch.manual_seed(0)                                                   # 워커 시드(= torch 기본 생성기에서 뽑는 base_seed)와 δ 추첨을 설정마다 같게
    it = iter(dl); t_ = time.time(); cache = [next(it) for _ in range(a.warmup + a.steps)]; t_data = time.time() - t_   # 데이터 읽기(오디오 I/O)와 GPU 계산을 따로 잰다
    fp = hashlib.md5(); [fp.update(cache[0][k].numpy().tobytes()) for k in ("ids", "labels", "pos_weight", "K") if k in cache[0]]; first_loss = None
    data_audio_s = sum(float(b["K"][:b["pack_n"]].sum() if "pack_n" in b else b["K"].sum()) for b in cache) * 0.08
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    audio_s = lab = row_tok = real_tok = n_samp = 0; tw = None
    for step, b in enumerate(cache):
        if step == a.warmup: torch.cuda.synchronize(); tw = time.time(); audio_s = lab = row_tok = real_tok = n_samp = 0
        x = {k: (v.cuda(non_blocking=True) if torch.is_tensor(v) else v) for k, v in b.items()}
        nw = model.config.next_weight_ko if b["lang"][0] == "Korean" else model.config.next_weight
        inp = {k: x[k] for k in ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask", "pos_weight", "position_ids") if k in x}
        with torch.autocast("cuda", dtype=torch.bfloat16): out = model(**inp, next_weight=nw)
        if step == 0: first_loss = float(out.loss)
        out.loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad(set_to_none=True)
        audio_s += float(b["K"].sum()) * 0.08; lab += int(out.n_labels); n_samp += int(b["K"].numel())
        row_tok += int(b["ids"].numel()); real_tok += int(b["mask"].sum()) if "mask" in b else int(b["ids"].numel())
    torch.cuda.synchronize(); dt = time.time() - tw
    r = dict(steps=a.steps, sec=round(dt, 1), sec_per_step=round(dt / a.steps, 3), audio_h_per_gpu_h=round(audio_s / dt, 1), samples_per_s=round(n_samp / dt, 2),
             label_tok_per_s=round(lab / dt), row_tok_per_step=round(row_tok / a.steps), real_tok_frac=round(real_tok / max(1, row_tok), 3),
             data_audio_h_per_h=round(data_audio_s / t_data, 1), data_sec=round(t_data, 1), num_workers=a.num_workers,
             samples_per_step=round(n_samp / a.steps, 1), peak_mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), liger=liger_on, fused_adamw=fused, pack_max_tokens=pack, batch_max_tokens=budget,
             attn=attn_got, first_loss=first_loss, first_batch=fp.hexdigest()[:12],
             batches={m: s.describe() for m, s in dl.samplers.items()} if (pack or budget) else {m: dict(bs=bs_of(m), batches=len(s)) for m, s in dl.samplers.items()})
    res[spec] = r; log(spec, json.dumps(r, ensure_ascii=False))
    del opt, dl, it, cache; torch.cuda.empty_cache()
for spec, r in res.items():                                                                   # 같은 배치(지문), attention 만 다른 sdpa 설정 대비 첫 step loss
    conf = spec.partition("@")[0]
    ref = next((k for k, q in res.items() if k != spec and k.partition("@")[0] == conf and q["attn"] == "sdpa" and q["first_batch"] == r["first_batch"]), None)
    if ref: r["first_loss_ref"] = ref; r["first_loss_rel_vs_sdpa"] = abs(r["first_loss"] - res[ref]["first_loss"]) / abs(res[ref]["first_loss"])
base = res.get("base")
if base:
    for k, r in res.items(): r["speedup_vs_base"] = round(r["audio_h_per_gpu_h"] / base["audio_h_per_gpu_h"], 2)
json.dump(dict(args=vars(a), results=res), open(a.out, "w"), indent=1, ensure_ascii=False); log("→", a.out)

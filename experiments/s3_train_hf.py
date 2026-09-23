#!/usr/bin/env python
"""Stage 2/3 — HF Trainer 기반 스트리밍 ASR 학습 (vapasr.hf). experiments/s1_train_mono.py 의 후속(같은 데이터·시퀀스·손실 규약).

  단일:  python experiments/s3_train_hf.py --train librispeech-100,kspon-100 --out-dir /soundai/Model/VAPASR/hf-smoke --max-steps 60 --eval-every 30 --save-every 30
  DDP :  torchrun --nproc_per_node=8 experiments/s3_train_hf.py --train librispeech-960,kspon-full --epochs 30 --out-dir /soundai/Model/VAPASR/hf-C ...
초기화(--init): Qwen3-ASR 디렉토리(새 모델, 기본 $MXC_QWEN_ASR_DIR) | HF 산출물 디렉토리(config.json model_type=vapasr) | 기존 ckpt-last.pt(full FT, from_legacy)
재개: out-dir 의 checkpoint-<step>/ 가 있으면 자동(Trainer resume: 모델·옵티마이저·스케줄러·RNG·step). 선점: PREEMPT 파일 또는 SIGUSR1/SIGTERM → 저장 후 종료.
완료: out-dir/DONE + out-dir/final/ (HF 형식 save_pretrained + tokenizer). TensorBoard: out-dir/tb/ (tensorboard --logdir /soundai/Model/VAPASR/hf-<RUN>/tb)"""
import os, sys, json, math, time, argparse, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--train", default="librispeech-100,kspon-100"); ap.add_argument("--epochs", type=float, default=0); ap.add_argument("--max-steps", type=int, default=0)
ap.add_argument("--bs-en", type=int, default=12); ap.add_argument("--bs-ko", type=int, default=48)
ap.add_argument("--lr", type=float, default=6e-5); ap.add_argument("--lr-adapter", type=float, default=1e-3); ap.add_argument("--lr-encoder", type=float, default=1e-5); ap.add_argument("--warmup", type=int, default=500); ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--delays", default="2,3,4,6"); ap.add_argument("--M", type=int, default=0); ap.add_argument("--next-weight", type=float, default=0.3); ap.add_argument("--next-weight-ko", type=float, default=0.15)
ap.add_argument("--init", default=None, help="Qwen3-ASR 디렉토리 | HF 산출물 디렉토리 | 기존 ckpt-last.pt"); ap.add_argument("--init-adapter", default=None, help="증류 adapter.pt (새 모델일 때)")
encoder_group = ap.add_mutually_exclusive_group()
encoder_group.add_argument("--train-encoder", dest="train_encoder", action="store_true", help="encoder 해동(기본값)")
encoder_group.add_argument("--freeze-encoder", dest="train_encoder", action="store_false", help="encoder 동결 ablation")
ap.set_defaults(train_encoder=True)
ap.add_argument("--no-grad-ckpt", action="store_true"); ap.add_argument("--liger", action="store_true"); ap.add_argument("--no-liger", action="store_true")
ap.add_argument("--eval-every", type=int, default=2000); ap.add_argument("--save-every", type=int, default=500); ap.add_argument("--eval-bias", default="0"); ap.add_argument("--eval-delay", type=int, default=2)
ap.add_argument("--sentinel-stream", type=int, default=10); ap.add_argument("--sentinel-utt", type=int, default=100); ap.add_argument("--eval-only", action="store_true")
ap.add_argument("--eval-tag", default="offline", help="--eval-only 결과 이름 eval/<tag>-<step>.json (예: select)"); ap.add_argument("--eval-seed", type=int, default=1, help="dev 표본 추출 seed (sentinel 1, select 7)")
ap.add_argument("--dev-extra", default="", help="추가 dev 셋 '라벨=manifest:subset:mode' 쉼표 구분 (예: ah71-dev=aihub71631-dev:dev:utt) — 새 도메인 평가용, 기본 3 셋 뒤에 붙는다")
ap.add_argument("--out-dir", required=True); ap.add_argument("--resume", default="auto", help="auto | none | <checkpoint dir>"); ap.add_argument("--save-total-limit", type=int, default=2)
ap.add_argument("--train-man-root", default=None, help="학습 manifest 전용 루트(dev/eval 기본 루트와 분리)")
ap.add_argument("--train-align-root", default=None, help="학습 alignment 전용 루트")
ap.add_argument("--language-schedule", choices=("balanced", "proportional"), default="balanced",
                help="balanced=언어 step 1:1(작은 셋 반복), proportional=각 스트림을 epoch당 정확히 한 번")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--log-every", type=int, default=50); ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--gpu", default=None)
a = ap.parse_args()

rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
_cache_root = os.path.join(os.environ.get("VAPASR_LOCAL_CACHE", "/tmp"), f"vapasr-{os.getuid()}-{os.environ.get('SLURM_JOB_ID', 'local')}")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, f"triton-{local}")); os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache_root, f"inductor-{local}"))
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True); os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
if world == 1 and a.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch, torch.distributed as dist
gloo_pg = None
if world > 1:
    from datetime import timedelta; import faulthandler, signal as _sig; faulthandler.enable(); faulthandler.register(_sig.SIGUSR2, all_threads=True, chain=False)   # kill -USR2 <pid> → 전 스레드 파이썬 스택을 stderr 로(행 진단)
    if int(os.environ.get("NCCL_IB_RETRY_CNT", "7")) > 7: os.environ["NCCL_IB_RETRY_CNT"] = "7"
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    torch.cuda.set_device(local); dist.init_process_group("nccl", timeout=timedelta(hours=3), device_id=torch.device("cuda", local))
    gloo_pg = dist.new_group(backend="gloo", timeout=timedelta(hours=1))   # 평가 gather·선점 합의가 1 h 를 넘으면 행(hang)으로 보고 실패 → requeue          # 긴 대기(평가 gather·선점 합의) 는 이더넷 gloo 로
main = rank == 0
def log(*s):
    if main: print(*s, flush=True)
def barrier():
    if world > 1: dist.barrier(group=gloo_pg)
torch.manual_seed(a.seed + rank); random.seed(a.seed + rank); torch.backends.cuda.matmul.allow_tf32 = True
out = a.out_dir; os.makedirs(os.path.join(out, "eval"), exist_ok=True) if main else None

from vapasr.hf import VapAsrForStreamingASR, load_tokenizer
from vapasr.hf.trainer import VapAsrTrainer, PreemptCallback
from vapasr.uslm.mono_data import MonoStreamDataset
from vapasr.data.textnorm import TEXTNORM_VERSION
from transformers import TrainingArguments, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
QWEN = os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B"); init = a.init or QWEN

# ── 모델 (tokenizer 는 모델과 함께)
t0 = time.time()
if os.path.isfile(init):                                                          # 기존 ckpt-last.pt
    model, tok = VapAsrForStreamingASR.from_legacy(init, QWEN, next_weight=a.next_weight, next_weight_ko=a.next_weight_ko, delays=[int(x) for x in a.delays.split(",")]); src = f"legacy {init}"
elif os.path.exists(os.path.join(init, "config.json")) and json.load(open(os.path.join(init, "config.json"))).get("model_type") == "vapasr":
    model = VapAsrForStreamingASR.from_pretrained(init); tok = load_tokenizer(init); src = f"hf {init}"
else:
    model, tok = VapAsrForStreamingASR.from_qwen(init, next_weight=a.next_weight, next_weight_ko=a.next_weight_ko, delays=[int(x) for x in a.delays.split(",")]); src = f"qwen {init}"
    if a.init_adapter: st0 = torch.load(a.init_adapter, map_location="cpu"); model.adapter.load_state_dict(st0["adapter"]); src += f" + adapter {a.init_adapter}"
if a.train_encoder: model.encoder.set_trainable(True); model.config.encoder_trainable = True
model._keys_to_ignore_on_save = [k for k in model.state_dict() if k.startswith("encoder.")] if not model.config.encoder_trainable else None   # 동결 인코더는 저장·재개 경고 제외
if not a.no_grad_ckpt: model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
use_liger = a.liger or not a.no_liger
if use_liger:
    try:
        from vapasr.hf.liger import apply_liger_to_thinker; n = apply_liger_to_thinker(model); log(f"liger: {n}")
    except ImportError as e: log(f"liger 미적용({e})"); use_liger = False
model.config.use_liger = use_liger
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"model ← {src} ({time.time()-t0:.0f}s) · trainable {n_tr/1e6:.1f}M · encoder {'학습' if model.config.encoder_trainable else '동결'} · grad ckpt {not a.no_grad_ckpt}")

# ── 데이터 (rank 0 이 항목 캐시를 먼저 만들고 나머지가 읽는다)
delays = tuple(int(x) for x in a.delays.split(",")); manifests = a.train.replace(":", ",").split(",")
DEV = [("dev-clean", "librispeech-dev", "dev-clean", "stream"), ("dev-other", "librispeech-dev", "dev-other", "stream"), ("kspon-dev", "kspon-dev", "dev", "utt")]
DEV_CAP = {}
for spec in [x for x in a.dev_extra.split(",") if x]:                            # '라벨=manifest:subset:mode[:최대 항목 수]'
    lab, rest = spec.split("="); parts_ = rest.split(":"); m_, sub_, mode_ = parts_[:3]; DEV.append((lab, m_, sub_, mode_))
    if len(parts_) > 3: DEV_CAP[lab] = int(parts_[3])
def make_sets(spec, cap_stream, cap_utt, seed=1):
    return {lab: MonoStreamDataset([m], tok, mode=mode, subsets=[sub], delays=(a.eval_delay,), max_per_chunk=a.M, seed=seed, max_items=min(DEV_CAP.get(lab, 10**9), cap_stream if mode == "stream" else cap_utt), online=True) for lab, m, sub, mode in spec}
if world > 1 and not main: barrier()
train_sets = {m: MonoStreamDataset([m], tok, mode="stream", delays=delays, max_per_chunk=a.M, seed=a.seed,
                                   manifest_root=a.train_man_root, align_root=a.train_align_root, online=True)
              for m in manifests} if not a.eval_only else {}
dev_sets = make_sets(DEV, a.sentinel_stream, a.sentinel_utt, seed=a.eval_seed)
if world > 1 and main: barrier()
log("train " + ", ".join(f"{k}:{len(v)} (drop {v.dropped}, no-align {v.no_align})" for k, v in train_sets.items()) + " | dev " + ", ".join(f"{k}:{len(v)}" for k, v in dev_sets.items()) + f" | world {world}")

# ── Trainer
targs = TrainingArguments(output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=1, num_train_epochs=a.epochs if a.epochs > 0 else 1, max_steps=a.max_steps if a.max_steps > 0 else -1,
                          learning_rate=a.lr, weight_decay=a.wd, warmup_steps=a.warmup, lr_scheduler_type="cosine", max_grad_norm=1.0, bf16=True,
                          logging_strategy="steps", logging_steps=a.log_every, logging_first_step=True, eval_strategy=("no" if a.eval_every >= 10**6 else "steps"), eval_steps=a.eval_every, save_strategy="steps", save_steps=a.save_every,   # 재개 시 checkpoint 의 eval_steps 가 복원되므로 끄려면 strategy 자체를 no 로
                          save_total_limit=a.save_total_limit, save_safetensors=True, save_on_each_node=False, seed=a.seed, data_seed=a.seed, report_to=["tensorboard"], logging_dir=os.path.join(out, "tb"), remove_unused_columns=False,
                          disable_tqdm=True, ignore_data_skip=True, ddp_find_unused_parameters=False, ddp_broadcast_buffers=False, ddp_timeout=10800, dataloader_num_workers=a.num_workers,
                          label_names=["labels"], log_level="warning" if main else "error")
preempt_cb = PreemptCallback(out, gloo_pg)
trainer = VapAsrTrainer(model=model, args=targs, train_sets=train_sets, dev_sets=dev_sets, tokenizer=tok, bs_en=a.bs_en, bs_ko=a.bs_ko, lr_adapter=a.lr_adapter, lr_encoder=a.lr_encoder,
                        eval_delay=a.eval_delay, eval_biases=[float(x) for x in a.eval_bias.split(",")], max_per_chunk=a.M, gloo_pg=gloo_pg, num_workers=a.num_workers,
                        language_schedule=a.language_schedule, callbacks=[preempt_cb], processing_class=tok)                    # checkpoint-N 에 tokenizer 도 저장
if a.eval_only:                                                                # 오프라인 평가: --init <checkpoint-N 디렉토리> → out-dir/eval/offline-N.json
    import re; mstep = re.search(r"checkpoint-(\d+)", init or ""); trainer.state.global_step = int(mstep.group(1)) if mstep else 0
    r = trainer.evaluate(metric_key_prefix=a.eval_tag); log(json.dumps(r, ensure_ascii=False))
    if mstep:                                                                   # TensorBoard 에도 기록(학습 곡선과 같은 step 축)
        try:
            from torch.utils.tensorboard import SummaryWriter
            w = SummaryWriter(os.path.join(out, "tb")); [w.add_scalar(k, v, int(mstep.group(1))) for k, v in r.items() if isinstance(v, (int, float))]; w.close()
        except Exception as e: log(f"tensorboard 기록 실패: {e}")
    sys.exit(0)
def last_complete_checkpoint(d):
    """checkpoint-N 중 trainer_state.json·model.safetensors 가 모두 있는(저장이 끝난) 가장 큰 N. 선점이 저장 도중 끊으면 trainer_state.json 없는 디렉토리가 남는다(66523 checkpoint-3000)."""
    import re
    cands = []
    for x in os.listdir(d) if os.path.isdir(d) else []:
        m = re.fullmatch(r"checkpoint-(\d+)", x)
        if m and all(os.path.exists(os.path.join(d, x, f)) for f in ("trainer_state.json", "model.safetensors")): cands.append((int(m.group(1)), os.path.join(d, x)))
    return max(cands)[1] if cands else None
last = None
if a.resume == "auto": last = last_complete_checkpoint(out)
elif a.resume != "none": last = a.resume
if main:
    tdl = trainer.get_train_dataloader(); log(f"rank 당 배치/epoch: " + ", ".join(f"{m}:{len(s)}" for m, s in tdl.samplers.items()) + f" → steps/epoch {len(tdl)} · 유효 배치 EN {a.bs_en*world} / KO {a.bs_ko*world}" + (f" · 재개 ← {last}" if last else ""))
    json.dump(dict(args=vars(a), textnorm_version=TEXTNORM_VERSION, trainable_m=n_tr / 1e6, train_streams={k: len(v) for k, v in train_sets.items()}, steps_per_epoch=len(tdl), init=src, world=world),
              open(os.path.join(out, "run.json"), "w"), indent=1, ensure_ascii=False)
if os.path.exists(os.path.join(out, "PREEMPT")) and main: os.remove(os.path.join(out, "PREEMPT"))
barrier()
res = trainer.train(resume_from_checkpoint=last)
st = trainer.state; preempted = preempt_cb.fired
done = st.global_step >= st.max_steps and not preempted
log(f"train 종료: step {st.global_step}/{st.max_steps} {'완료' if done else '(선점/중단 → 재시작 시 이어서)'} · {res.metrics}")
if done:
    trainer.save_model(os.path.join(out, "final")); tok.save_pretrained(os.path.join(out, "final")) if main else None
    if main:
        json.dump(dict(args=vars(a), textnorm_version=TEXTNORM_VERSION, hist=trainer.eval_hist, log_history=st.log_history[-200:], trainable_m=n_tr / 1e6, steps=st.global_step), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
        open(os.path.join(out, "DONE"), "w").write(time.strftime("%F %T")); log("final →", os.path.join(out, "final"))
if world > 1: dist.destroy_process_group()

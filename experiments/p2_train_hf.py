#!/usr/bin/env python
"""Phase 2 D1 학습 — 보정본 창(DialogueWindowDataset) 으로 lane 태그·ONSET·EOT(soft)·lane 활동 헤드를 학습한다 (정본 output-phase2-lane-plan §9·§13).

  python experiments/p2_train_hf.py --init /soundai/Model/VAPASR/hf-E2/final --data <phase2 dir> --corpora aihub71631,otoSpeech,ami --out-dir <run> [--windows <overfit32.windows.jsonl>] [--max-steps 300]
초기화: E2 HF 체크포인트 → add_phase2_tokens(registry 동결·<SPK_3..6>/<ONSET>/<EOT> 임베딩 초기화·활동 헤드·A/B 차단 해제). 이전 정본 §7.1 recipe: lr 6e-5(adapter 1e-3), cosine, bf16, grad ckpt, Liger.
데이터: 코퍼스별 DialogueWindowDataset(창 20–40 s, hop 10 s, δ∈{2,3,4,6}, R=6, untranscribed=mask) 을 라운드로빈. --windows 를 주면 그 창 목록만(overfit).
  배치: --max-tokens N 이면 동적 길이 배치(TokenBudgetSampler: 창 수 × 최장 추정 길이 ≤ N, 짧은 창은 많이·긴 창은 적게 → step 당 토큰 균일), 0 이면 고정 bs-en/bs-ko.
  코퍼스 비중 --mix: equal(균등 라운드로빈; 작은 코퍼스 반복)·sqrt·prop(창 수 비례). 실측 예산은 experiments/p2_mem_probe.py 로 잰다(GPU 메모리 80 %).
평가: D1 은 학습 손실 분해(text/next/eot/act)와 32창 check 로 시작하며 자유실행 lane 디코드 평가는 lane_state parser 연결 뒤 붙인다. 단일 프로세스(--gpu) 또는 torchrun(노드당 GPU 8, slurm/p2_train_d1.sbatch) 모두 지원; 완료 시 out-dir/DONE."""
import os, sys, json, time, random, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="E2 HF 체크포인트 디렉토리(또는 Phase 2 산출물)"); ap.add_argument("--data", required=True, help="<corpus>.refined.dialogues.jsonl 이 있는 디렉토리")
ap.add_argument("--corpora", default="aihub71631,aihub134-1,aihub134-2,otoSpeech,ami,notsofar,icsi"); ap.add_argument("--windows", default=None, help="창 목록 jsonl(corpus, conv, t0, L) — overfit/고정 셋")
ap.add_argument("--out-dir", required=True); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--bs-en", type=int, default=8); ap.add_argument("--bs-ko", type=int, default=8); ap.add_argument("--lr", type=float, default=6e-5); ap.add_argument("--lr-adapter", type=float, default=1e-3); ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--delays", default="2,3,4,6"); ap.add_argument("--R", type=int, default=6); ap.add_argument("--hop", type=float, default=10.0); ap.add_argument("--window", type=float, nargs=2, default=(20.0, 40.0))
ap.add_argument("--act-weight", type=float, default=1.0); ap.add_argument("--eot-weight", type=float, default=2.0); ap.add_argument("--next-weight", type=float, default=0.3); ap.add_argument("--next-weight-ko", type=float, default=0.15)
ap.add_argument("--no-liger", action="store_true"); ap.add_argument("--no-grad-ckpt", action="store_true"); ap.add_argument("--save-every", type=int, default=500); ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--max-tokens", type=int, default=0, help="동적 길이 배치: 배치의 (창 수 × 최장 추정 길이) ≤ 이 토큰 수(0 이면 고정 bs-en/bs-ko)"); ap.add_argument("--max-bs", type=int, default=48, help="동적 배치의 창 수 상한")
ap.add_argument("--mix", default="equal", choices=["equal", "sqrt", "prop"], help="코퍼스 배치 비중: equal(라운드로빈 균등)·sqrt(배치 수^0.5 비례)·prop(배치 수 비례)")
ap.add_argument("--check-save", action="store_true", help="학습 뒤 메모리 모델과 final 재로드 모델의 손실·가중치를 같은 배치로 대조(save/load 진단)")
ap.add_argument("--train-encoder", action="store_true", help="인코더도 학습(기본 동결 — 정본 §1)"); ap.add_argument("--mono-cache", default=None, help="대화별 mono 혼합 캐시 디렉토리(float16 npy, mmap)"); ap.add_argument("--resume", default="auto"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--num-workers", type=int, default=2); ap.add_argument("--gpu", default=None); ap.add_argument("--check", action="store_true", help="학습 전 창별 라운드트립(라벨 토큰 = 참조 토큰) 검사")
a = ap.parse_args()
rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
_cache_root = os.path.join(os.environ.get("VAPASR_LOCAL_CACHE", "/tmp"), f"vapasr-{os.getuid()}-{os.environ.get('SLURM_JOB_ID', 'local')}")
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_cache_root, f"triton-{local}")); os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(_cache_root, f"inductor-{local}"))
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True); os.makedirs(os.environ["TORCHINDUCTOR_CACHE_DIR"], exist_ok=True)
if world == 1 and a.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch, torch.distributed as dist
from torch.utils.data import DataLoader
gloo_pg = None
if world > 1:                                                     # torchrun(노드당 GPU 8): s3_train_hf.py 와 같은 초기화(NCCL 학습 pg + 선점 합의용 gloo pg)
    from datetime import timedelta; import faulthandler, signal as _sig; faulthandler.enable(); faulthandler.register(_sig.SIGUSR2, all_threads=True, chain=False)
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    torch.cuda.set_device(local); dist.init_process_group("nccl", timeout=timedelta(hours=3), device_id=torch.device("cuda", local))
    gloo_pg = dist.new_group(backend="gloo", timeout=timedelta(hours=1))
main = rank == 0
def barrier():
    if world > 1: dist.barrier(group=gloo_pg)
from transformers import TrainingArguments
from vapasr.hf import VapAsrForStreamingASR, load_tokenizer
from vapasr.hf.trainer import VapAsrTrainer, PreemptCallback
from vapasr.uslm.mono_data import BucketBatchSampler

class WindowBucketSampler(BucketBatchSampler):
    """창 길이(L) 기준 버킷 배치 — DialogueWindowDataset.items 는 (conv, t0, L) 튜플이라 MonoStreamDataset 의 K 키를 쓰는 부모 순서를 대체한다."""
    def __init__(self, ds, bs, seed=0, drop_last=True, rank=0, world=1):
        order = sorted(range(len(ds)), key=lambda i: ds.items[i][2]); self.batches = [order[i: i + bs] for i in range(0, len(order), bs)]
        if drop_last and self.batches and len(self.batches[-1]) < bs: self.batches = self.batches[:-1]
        self.seed, self.rank, self.world, self.epoch = seed, rank, world, 0; self.n = len(self.batches) // world
from vapasr.data.dialogue_dataset import DialogueWindowDataset, collate_dialogue, TokenBudgetSampler
from vapasr.hf.data import RoundRobinLoader

class WeightedRoundRobin(RoundRobinLoader):
    """코퍼스 배치 비중을 가중(equal/sqrt/prop) 으로 두는 라운드로빈. epoch 당 step 수는 코퍼스 배치 수의 합(부모와 같음)이고, 그 step 을 가중치대로 나눠 셔플한 일정(seed+epoch 로 결정적)으로 코퍼스를 고른다.
    equal 은 부모와 같은 균등(작은 코퍼스가 여러 epoch 반복), prop 은 창 수 비례(모두 ≈1 epoch), sqrt 는 그 중간."""
    def __init__(self, train_sets, samplers, num_workers, mix="equal", seed=0):
        empty = [m for m in train_sets if len(samplers[m]) == 0]                     # rank 당 배치가 0 인 코퍼스(배치 수 < world) — 남겨 두면 그 rank 의 iterator 가 비어 라운드로빈이 멈추고 다른 rank 는 all_reduce 에서 영원히 기다린다
        if empty: log(f"  ! 배치 수 < world 라 제외: {empty}")
        train_sets = {m: ds for m, ds in train_sets.items() if m not in empty}; samplers = {m: samplers[m] for m in train_sets}; assert train_sets, "모든 코퍼스의 배치 수가 world 보다 작음(창·예산을 늘리거나 GPU 를 줄인다)"
        self.names = list(train_sets); self.samplers = samplers
        self.loaders = {m: DataLoader(ds, batch_sampler=samplers[m], num_workers=num_workers, collate_fn=collate_dialogue, persistent_workers=False) for m, ds in train_sets.items()}
        self.epoch = 0; self._n = sum(len(s) for s in samplers.values()); self.mix = mix; self.seed = seed
        alpha = {"equal": 0.0, "sqrt": 0.5, "prop": 1.0}[mix]; w = {m: max(1, len(samplers[m])) ** alpha for m in self.names}; tot = sum(w.values())
        self.counts = {m: int(round(self._n * w[m] / tot)) for m in self.names}
        diff = self._n - sum(self.counts.values()); self.counts[max(self.counts, key=self.counts.get)] += diff
    def _cycle(self, m, ep0):
        ep = ep0
        while True:
            self.samplers[m].set_epoch(ep); n = 0
            for b in self.loaders[m]: n += 1; yield b
            assert n > 0, f"{m}: rank {self.samplers[m].rank} 에 배치 없음"; ep += 1
    def __iter__(self):
        its = {m: self._cycle(m, self.epoch) for m in self.names}; sched = [m for m in self.names for _ in range(self.counts[m])]
        random.Random(f"{self.seed}:{self.epoch}:mix").shuffle(sched)
        for m in sched: yield next(its[m])
from vapasr.data.dialogue_tokens import load_frozen_registry
torch.manual_seed(a.seed); random.seed(a.seed + rank); torch.backends.cuda.matmul.allow_tf32 = True      # torch seed 는 rank 공통(신규 토큰 임베딩 초기화가 rank 간 같아야 함)
out = a.out_dir; os.makedirs(out, exist_ok=True)
def log(*s):
    if main: print(*s, flush=True)

# ── 모델
t0 = time.time(); model = VapAsrForStreamingASR.from_pretrained(a.init); tok = load_tokenizer(a.init)
cfg = model.config
if cfg.lanes == 0: ids = model.add_phase2_tokens(tok, R=a.R); log(f"Phase 2 토큰 추가: { {k: ids[k] for k in ('<SPK_3>', '<SPK_6>', '<ONSET>', '<EOT>')} }")
else: log(f"Phase 2 모델 로드(lanes={cfg.lanes})"); model.add_phase2_tokens(tok, R=cfg.lanes)
cfg.act_weight, cfg.eot_weight, cfg.next_weight, cfg.next_weight_ko, cfg.delays = a.act_weight, a.eot_weight, a.next_weight, a.next_weight_ko, [int(x) for x in a.delays.split(",")]
assert dict(cfg.phase2_registry) == {k: v for k, v in load_frozen_registry().items() if k in cfg.phase2_registry}, "registry 불일치"
if not a.no_grad_ckpt: model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.thinker.config.use_cache = False
if not a.no_liger:
    try:
        from vapasr.hf.liger import apply_liger_to_thinker; log(f"liger: {apply_liger_to_thinker(model)}")
    except ImportError as e: log(f"liger 미적용({e})")
model.encoder.eval(); init_has_encoder = bool(cfg.encoder_trainable or getattr(cfg, "encoder_saved", False))
if not a.train_encoder:
    for p_ in model.encoder.parameters(): p_.requires_grad_(False)
    model.config.encoder_trainable = False
    model.config.encoder_saved = init_has_encoder                 # 초기화 체크포인트(E2)가 학습된 인코더를 갖고 있으면 동결해도 함께 저장(재로드 시 .nemo 원본이 붙는 사고 방지)
    log(f"인코더 동결 · 체크포인트에 인코더 저장: {model.config.encoder_saved}")
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad); log(f"모델 준비 {time.time()-t0:.0f}s · 학습 파라미터 {n_tr/1e6:.1f} M · lanes {cfg.lanes} · registry {cfg.phase2_registry}")

# ── 데이터
LANG = {"aihub71631": "Korean", "aihub134-1": "Korean", "aihub134-2": "Korean", "otoSpeech": "English", "ami": "English", "notsofar": "English", "icsi": "English"}
wins = None
if a.windows:
    wins = {}
    for l in open(a.windows): r = json.loads(l); wins.setdefault(r["corpus"], set()).add((r["conv"], round(r["t0"], 3), round(r["L"], 3)))
delays = tuple(int(x) for x in a.delays.split(",")); train_sets = {}
for c in a.corpora.split(","):
    p = os.path.join(a.data, f"{c}.refined.dialogues.jsonl")
    if not os.path.exists(p): log(f"!! {p} 없음 — 건너뜀"); continue
    if wins is not None and c not in wins: continue
    ds = DialogueWindowDataset([p], tok, R=a.R, window_s=tuple(a.window), hop_s=a.hop, delays=delays, seed=a.seed, mono_cache_dir=a.mono_cache)
    if wins is not None:
        want = wins[c]; ds.items = [it for it in ds.items if (it[0], round(it[1], 3), round(it[2], 3)) in want]
        if len(ds.items) != len(want): log(f"  ! {c}: 창 목록 {len(want)} 중 {len(ds.items)} 만 일치(창 규약이 바뀌었으면 목록을 다시 만든다)")
    ds.name = c; train_sets[c] = ds; log(f"  {c}: 창 {len(ds)} (stats {ds.stats})")
assert train_sets, "학습 셋 없음"
if a.check:
    for c, ds in train_sets.items():
        bad = 0
        for i in range(min(len(ds), 32)):
            s = ds.sequence(i, 4); eps, _ = ds.window_episodes(ds.dlgs[s["cid"]], s["t0"], s["L"]); ref = [t for e in sorted(eps, key=lambda e: e.ep_id) if e.lane for t, _ in e.tokens]
            got = [t for t, k in zip(s["ids"], s["kinds"]) if k == "text"]; bad += int(sorted(got) != sorted(ref))
        log(f"  check {c}: 창 {min(len(ds), 32)} 중 라운드트립 불일치 {bad}")

class P2Trainer(VapAsrTrainer):
    """VapAsrTrainer 의 데이터·손실만 Phase 2 로: 창 데이터셋 라운드로빈(collate_dialogue), forward 에 soft/활동 타깃 전달, 손실 분해 로그. 평가는 D1 초기엔 없음(lane parser 연결 뒤)."""
    def bs_of(self, name): return self.bs_ko if LANG.get(name, "English") == "Korean" else self.bs_en
    def get_train_dataloader(self):
        if a.max_tokens > 0: samplers = {m: TokenBudgetSampler(ds, a.max_tokens, max_bs=a.max_bs, seed=self.args.seed, rank=self.args.process_index, world=self.args.world_size, drop_last=len(ds) > a.max_bs) for m, ds in self.train_sets.items()}
        else: samplers = {m: WindowBucketSampler(ds, min(self.bs_of(m), len(ds)), seed=self.args.seed, drop_last=len(ds) > self.bs_of(m), rank=self.args.process_index, world=self.args.world_size) for m, ds in self.train_sets.items()}
        return WeightedRoundRobin(self.train_sets, samplers, self.num_workers, mix=a.mix, seed=self.args.seed)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        cfg_ = getattr(model, "module", model).config                                   # DDP 래퍼면 .module
        lang = inputs.get("lang", ["English"])[0]; nw = cfg_.next_weight_ko if lang == "Korean" else cfg_.next_weight
        x = {k: inputs[k] for k in ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask", "labels_alt", "activity", "activity_mask") if k in inputs}; x["soft_w"] = inputs["soft_w_full"]
        out = model(**x, next_weight=nw)
        for k in ("loss_next", "loss_text", "top1_text", "loss_eot", "loss_act", "act_acc"):
            v = getattr(out, k)
            if v is not None: self._parts[k] = self._parts.get(k, 0.0) + float(v)
        self._parts["n_labels"] = self._parts.get("n_labels", 0) + int(out.n_labels); self._parts["n_soft"] = self._parts.get("n_soft", 0) + int(out.n_soft or 0); self._n_parts += 1
        return (out.loss, out) if return_outputs else out.loss
    def log(self, logs: dict, start_time=None):
        if self.args.process_index != 0: self._parts = {}; self._n_parts = 0; return
        if self._n_parts and "loss" in logs:
            for k in ("loss_next", "loss_text", "top1_text", "loss_eot", "loss_act", "act_acc"):
                if k in self._parts: logs[k] = round(self._parts[k] / self._n_parts, 4)
            logs["labels_per_step"] = round(self._parts["n_labels"] / self._n_parts); logs["soft_per_step"] = round(self._parts["n_soft"] / self._n_parts, 1); self._parts = {}; self._n_parts = 0
        from transformers import Trainer
        return Trainer.log(self, logs, start_time) if start_time is not None else Trainer.log(self, logs)
    def evaluate(self, *args, **kw): return {}

targs = TrainingArguments(output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=1, num_train_epochs=a.epochs, max_steps=a.max_steps if a.max_steps > 0 else -1,
                          learning_rate=a.lr, weight_decay=a.wd, warmup_steps=a.warmup, lr_scheduler_type="cosine", max_grad_norm=1.0, bf16=True, logging_strategy="steps", logging_steps=a.log_every, logging_first_step=True,
                          eval_strategy="no", save_strategy="steps", save_steps=a.save_every, save_total_limit=2, save_safetensors=True, seed=a.seed, data_seed=a.seed, report_to=["tensorboard"], logging_dir=os.path.join(out, "tb"),
                          remove_unused_columns=False, disable_tqdm=True, ignore_data_skip=True, dataloader_num_workers=a.num_workers, label_names=["labels"], log_level="warning")
preempt_cb = PreemptCallback(out, gloo_pg)
trainer = P2Trainer(model=model, args=targs, train_sets=train_sets, dev_sets={}, tokenizer=tok, bs_en=a.bs_en, bs_ko=a.bs_ko, lr_adapter=a.lr_adapter, num_workers=a.num_workers, gloo_pg=gloo_pg, callbacks=[preempt_cb], processing_class=tok)
tdl = trainer.get_train_dataloader(); log("배치/epoch: " + ", ".join(f"{m}:{len(s)}" for m, s in tdl.samplers.items()) + f" → steps/epoch {len(tdl)} · 비중({a.mix}) " + ", ".join(f"{m}:{n}" for m, n in tdl.counts.items()))
if a.max_tokens > 0:
    for m, sp in tdl.samplers.items(): log(f"  동적 배치 {m}: {sp.describe()}")
if main: json.dump(dict(args=vars(a), world=world, registry=cfg.phase2_registry, train_windows={k: len(v) for k, v in train_sets.items()}, steps_per_epoch=len(tdl), mix_counts=tdl.counts, batching={m: sp.describe() for m, sp in tdl.samplers.items()} if a.max_tokens > 0 else None, trainable_m=n_tr / 1e6), open(os.path.join(out, "run.json"), "w"), indent=1, ensure_ascii=False)
last = None
if a.resume == "auto":
    import re; c = [(int(m.group(1)), os.path.join(out, x)) for x in os.listdir(out) for m in [re.fullmatch(r"checkpoint-(\d+)", x)] if m and os.path.exists(os.path.join(out, x, "trainer_state.json"))]
    last = max(c)[1] if c else None
elif a.resume != "none": last = a.resume
barrier(); res = trainer.train(resume_from_checkpoint=last); st = trainer.state; preempted = preempt_cb.fired
done = st.global_step >= st.max_steps and not preempted
log(f"train 종료: step {st.global_step}/{st.max_steps} {'완료' if done else '(선점/중단 → 재시작 시 이어서)'} · {res.metrics}")
if done and a.check_save and main:
    def batch_loss(m, x):
        m.eval(); xx = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in x.items()}; xin = {k: xx[k] for k in ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask", "labels_alt", "activity", "activity_mask")}; xin["soft_w"] = xx["soft_w_full"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): o = m(**xin, next_weight=0.15)
        return dict(loss_text=round(float(o.loss_text), 4), top1=round(float(o.top1_text), 4), loss_next=round(float(o.loss_next), 4), loss_eot=(round(float(o.loss_eot), 4) if o.loss_eot is not None else None))
    xb = next(iter(tdl)); mem = getattr(trainer.model, "module", trainer.model); log("check-save 메모리(trainer.model):", batch_loss(mem, xb), "| 스크립트 model 객체 동일:", mem is model)
    trainer.save_model(os.path.join(out, "final")); tok.save_pretrained(os.path.join(out, "final"))
    re = VapAsrForStreamingASR.from_pretrained(os.path.join(out, "final")).cuda(); log("check-save 재로드(final):", batch_loss(re, xb))
    sd1 = {k: v.detach().float().cpu() for k, v in mem.state_dict().items() if not k.startswith("encoder.")}; sd2 = {k: v.detach().float().cpu() for k, v in re.state_dict().items() if not k.startswith("encoder.")}
    miss = sorted(set(sd1) - set(sd2)); extra = sorted(set(sd2) - set(sd1)); diff = [(k, float((sd1[k] - sd2[k]).abs().max())) for k in sd1 if k in sd2 and sd1[k].shape == sd2[k].shape and float((sd1[k] - sd2[k]).abs().max()) > 1e-6]
    shape = [k for k in sd1 if k in sd2 and sd1[k].shape != sd2[k].shape]
    log(f"check-save state_dict: 메모리 {len(sd1)} 키 · 재로드 {len(sd2)} 키 · 누락 {miss[:10]} · 추가 {extra[:10]} · shape 불일치 {shape[:10]} · 값 차이 {len(diff)} 키: {sorted(diff, key=lambda x: -x[1])[:12]}")
    enc1 = {k: v.detach().float().cpu() for k, v in mem.state_dict().items() if k.startswith("encoder.")}; enc2 = {k: v.detach().float().cpu() for k, v in re.state_dict().items() if k.startswith("encoder.")}
    ediff = [(k, float((enc1[k] - enc2[k]).abs().max())) for k in enc1 if k in enc2 and enc1[k].shape == enc2[k].shape and float((enc1[k] - enc2[k]).abs().max()) > 1e-6]
    log(f"check-save encoder: 메모리 {len(enc1)} · 재로드 {len(enc2)} · 값 차이 {len(ediff)} 키: {sorted(ediff, key=lambda x: -x[1])[:8]}")
if done:
    trainer.save_model(os.path.join(out, "final"))
    if main:
        tok.save_pretrained(os.path.join(out, "final")); json.dump(dict(args=vars(a), log_history=st.log_history[-300:], steps=st.global_step), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
        open(os.path.join(out, "DONE"), "w").write(time.strftime("%F %T")); log("final →", os.path.join(out, "final"))
if world > 1: dist.destroy_process_group()

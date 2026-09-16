#!/usr/bin/env python
"""Phase 2 D1 학습 — 보정본 창(DialogueWindowDataset) 으로 lane 태그·ONSET·EOT(soft)·lane 활동 헤드를 학습한다 (정본 output-phase2-lane-plan §9·§13).

  python experiments/p2_train_hf.py --init /soundai/Model/VAPASR/hf-E2/final --data <phase2 dir> --corpora aihub71631,otoSpeech,ami --out-dir <run> [--windows <overfit32.windows.jsonl>] [--max-steps 300]
초기화: E2 HF 체크포인트 → add_phase2_tokens(registry 동결·<SPK_3..6>/<ONSET>/<EOT> 임베딩 초기화·활동 헤드·A/B 차단 해제). 이전 정본 §7.1 recipe: lr 6e-5(adapter 1e-3), cosine, bf16, grad ckpt, Liger.
데이터: 코퍼스별 DialogueWindowDataset(창 20–40 s, hop 10 s, δ∈{2,3,4,6}, R=6, untranscribed=mask) 을 언어별 배치 크기로 라운드로빈. --windows 를 주면 그 창 목록만(overfit).
평가: D1 은 학습 손실 분해(text/next/eot/act)와 32창 check 로 시작하며 자유실행 lane 디코드 평가는 lane_state parser 연결 뒤 붙인다. 단일 GPU 기본(사용자 지시 2026-09-16)."""
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
ap.add_argument("--resume", default="auto"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--num-workers", type=int, default=2); ap.add_argument("--gpu", default=None); ap.add_argument("--check", action="store_true", help="학습 전 창별 라운드트립(라벨 토큰 = 참조 토큰) 검사")
a = ap.parse_args()
if a.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import torch
from torch.utils.data import DataLoader
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
from vapasr.data.dialogue_dataset import DialogueWindowDataset, collate_dialogue
from vapasr.data.dialogue_tokens import load_frozen_registry
torch.manual_seed(a.seed); random.seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True
out = a.out_dir; os.makedirs(out, exist_ok=True)
def log(*s): print(*s, flush=True)

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
model.encoder.eval()
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
    ds = DialogueWindowDataset([p], tok, R=a.R, window_s=tuple(a.window), hop_s=a.hop, delays=delays, seed=a.seed)
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
        from vapasr.hf.data import RoundRobinLoader
        rr = RoundRobinLoader.__new__(RoundRobinLoader); rr.names = list(self.train_sets)
        rr.samplers = {m: WindowBucketSampler(ds, min(self.bs_of(m), len(ds)), seed=self.args.seed, drop_last=len(ds) > self.bs_of(m), rank=self.args.process_index, world=self.args.world_size) for m, ds in self.train_sets.items()}
        rr.loaders = {m: DataLoader(ds, batch_sampler=rr.samplers[m], num_workers=self.num_workers, collate_fn=collate_dialogue, persistent_workers=False) for m, ds in self.train_sets.items()}
        rr.epoch = 0; rr._n = sum(len(s) for s in rr.samplers.values()); return rr
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        lang = inputs.get("lang", ["English"])[0]; nw = model.config.next_weight_ko if lang == "Korean" else model.config.next_weight
        x = {k: inputs[k] for k in ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask", "labels_alt", "activity", "activity_mask") if k in inputs}; x["soft_w"] = inputs["soft_w_full"]
        out = model(**x, next_weight=nw)
        for k in ("loss_next", "loss_text", "top1_text", "loss_eot", "loss_act", "act_acc"):
            v = getattr(out, k)
            if v is not None: self._parts[k] = self._parts.get(k, 0.0) + float(v)
        self._parts["n_labels"] = self._parts.get("n_labels", 0) + int(out.n_labels); self._parts["n_soft"] = self._parts.get("n_soft", 0) + int(out.n_soft or 0); self._n_parts += 1
        return (out.loss, out) if return_outputs else out.loss
    def log(self, logs: dict, start_time=None):
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
trainer = P2Trainer(model=model, args=targs, train_sets=train_sets, dev_sets={}, tokenizer=tok, bs_en=a.bs_en, bs_ko=a.bs_ko, lr_adapter=a.lr_adapter, num_workers=a.num_workers, callbacks=[PreemptCallback(out)], processing_class=tok)
tdl = trainer.get_train_dataloader(); log("배치/epoch: " + ", ".join(f"{m}:{len(s)}" for m, s in tdl.samplers.items()) + f" → steps/epoch {len(tdl)}")
json.dump(dict(args=vars(a), registry=cfg.phase2_registry, train_windows={k: len(v) for k, v in train_sets.items()}, steps_per_epoch=len(tdl), trainable_m=n_tr / 1e6), open(os.path.join(out, "run.json"), "w"), indent=1, ensure_ascii=False)
last = None
if a.resume == "auto":
    import re; c = [(int(m.group(1)), os.path.join(out, x)) for x in os.listdir(out) for m in [re.fullmatch(r"checkpoint-(\d+)", x)] if m and os.path.exists(os.path.join(out, x, "trainer_state.json"))]
    last = max(c)[1] if c else None
elif a.resume != "none": last = a.resume
res = trainer.train(resume_from_checkpoint=last); st = trainer.state
log(f"train 종료: step {st.global_step}/{st.max_steps} · {res.metrics}")
trainer.save_model(os.path.join(out, "final")); tok.save_pretrained(os.path.join(out, "final"))
json.dump(dict(args=vars(a), log_history=st.log_history[-300:], steps=st.global_step), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False); log("final →", os.path.join(out, "final"))

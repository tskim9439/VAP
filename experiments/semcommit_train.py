#!/usr/bin/env python
"""Semantic commit v0 — Phase 1 mono 스트림에 <SEM_END>/<TURN_END> 를 끼워 단일 GPU 로 학습한다 (raw/inbox/streaming_asr_semantic_commit_plan.md §21·§23).

  python experiments/semcommit_train.py --init <HF vapasr 디렉토리> --qwen-dir /data4/.../Qwen3-ASR-0.6B --nemotron-dir <*.nemo 가 든 디렉토리> \\
      --train-words en.words.jsonl,ko.words.jsonl --train-labels en.labels.jsonl,ko.labels.jsonl --out-dir /data3/tskim/semcommit/v0-smoke --max-steps 300 --gpu 0
  --init qwen : Qwen3-ASR(--qwen-dir) 에서 새 모델(+ --init-adapter). 작은 run 은 ASR 부터 수렴하지 않으므로 학습된 Phase 1 HF 체크포인트를 권한다.
  --dry-run   : 모델 없이 tokenizer + 데이터셋만 만들어 시퀀스 불변식(SEM 위치·TURN flush 밖·B/N 결정 위치)을 검사하고 샘플 하나를 찍고 끝난다(CPU).
데이터: vapasr/data/semcommit_dataset.SemCommitDataset(words.jsonl + labels.jsonl, id 로 짝). --train-words/--train-labels 는 반복 또는 쉼표 목록(같은 순서)이며
  행의 lang 으로 EN/KO 셋을 나눠 VapAsrTrainer 라운드로빈(언어별 배치 bs-en/bs-ko, collate_semcommit)으로 번갈아 낸다.
손실: soft_ce 의 NEXT=next_weight(EN 0.3 / KO 0.15)·SEM=--sem-weight·TURN=--turn-weight·결정 위치 pos_weight(N 후보 = --hardneg-weight). 모델 쪽
  add_semantic_tokens / forward(pos_weight=) 가 있어야 한다(없으면 시작 전에 멈춘다 — pos_weight 가 조용히 버려지는 것을 막는다).
인코더: 기본 동결(--freeze-encoder). HF init 의 config.encoder_trainable=True 여도 실제로 requires_grad 를 끄고 encoder_saved=True 로 체크포인트에 함께 저장한다
  (2026-09-17 D1 사고 — 동결 인코더를 저장에서 빼면 재로드 때 .nemo 원본이 붙는다; _keys_to_ignore_on_save 에 기대지 않는다).
저장: checkpoint-N 을 지우지 않는다(--save-total-limit 0). 완료 시 out-dir/final + 재로드 parity(state_dict·인코더·한 배치 손실·load_tokenizer·config.semcommit) → parity.json, PARITY OK/FAIL.
지문: config.semcommit = semcommit_fingerprint(turn_end·hangover_s·tail_margin·pad_tail·M·delays·가중·표본 설정 + words/labels 파일·tokenizer 어휘 sha256) — 모든 체크포인트와 final 에 들어간다.
  재개(auto 또는 --resume DIR)는 그 체크포인트의 config.semcommit 과 다르면 거부(--force-resume 로만 진행). 평가(semcommit_eval)는 final/config.json 의 semcommit 으로
  학습 패딩(--turn-end·--hangover-s·--tail-margin·δ)을 복원할 수 있다. 실행 기록은 run-<시각>-<pid>.json(덮어쓰지 않음), final/semcommit_run.json.
언어 셋: words 행이 있는 언어가 labels 짝이 없어 0 항목이 되면 멈춘다(--allow-missing-lang 로만 진행). words↔labels 짝 비율·짝 없는 label id 를 찍는다.
--M 은 0 만: M>0 이면 <SEM_END> 가 청크 한도에 세어져 단어와 다른 청크로 밀린다(v0 계약 위반; 평가 패딩 commit_metrics.train_pad_K 도 M=0 가정).
MXC_* 환경 변수가 없어도 돈다(모든 경로는 인자)."""
import os, sys, json, time, argparse, random, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="HF vapasr 디렉토리(config.json model_type=vapasr) | qwen (--qwen-dir 에서 새 모델)")
ap.add_argument("--qwen-dir", default=None, help="Qwen3-ASR 디렉토리: --init qwen 의 원본, 또는 init 에 tokenizer 파일이 없을 때 tokenizer 출처")
ap.add_argument("--init-adapter", default=None, help="--init qwen 일 때 증류 adapter.pt ('adapter' 또는 'state' 키)")
ap.add_argument("--nemotron-dir", default=None, help="*.nemo 가 든 디렉토리(예: HF 캐시 snapshot) — NemotronOnline(path=) 로 전달")
ap.add_argument("--train-words", action="append", required=True, help="words.jsonl (반복 또는 쉼표 목록)"); ap.add_argument("--train-labels", action="append", required=True, help="labels.jsonl (--train-words 와 같은 순서)")
ap.add_argument("--path-map", default="", help="segment 경로 접두어 치환 'old=new[,old2=new2]' (words.jsonl 이 이미 rack4 경로면 불필요)")
ap.add_argument("--allow-unlabeled", action="store_true", help="labels 없는 스트림도 학습(후보 없음 = SEM 없음) — 기본은 건너뜀")
ap.add_argument("--max-items-en", type=int, default=0); ap.add_argument("--max-items-ko", type=int, default=0)
ap.add_argument("--out-dir", required=True); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--bs-en", type=int, default=6); ap.add_argument("--bs-ko", type=int, default=16)
ap.add_argument("--lr", type=float, default=6e-5); ap.add_argument("--lr-adapter", type=float, default=1e-3); ap.add_argument("--lr-encoder", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--delays", default="2,3,4,6"); ap.add_argument("--M", type=int, default=0, help="청크당 텍스트 한도 — v0 는 0 만(SEM 이 단어와 같은 청크)")
ap.add_argument("--next-weight", type=float, default=0.3); ap.add_argument("--next-weight-ko", type=float, default=0.15)
ap.add_argument("--sem-weight", type=float, default=1.0); ap.add_argument("--turn-weight", type=float, default=1.0); ap.add_argument("--hardneg-weight", type=float, default=1.0)
ap.add_argument("--no-turn-end", action="store_true", help="<TURN_END> 없이 SEM 만"); ap.add_argument("--hangover", type=float, default=0.48, help="TURN 청크 = max(마지막 방출, int((t_last+hangover)/0.08))")
ap.add_argument("--tail-margin", type=int, default=2, help="K' = max(K, 마지막 이벤트 청크(δmax) + margin)")
enc_g = ap.add_mutually_exclusive_group()
enc_g.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true", help="인코더 동결(기본)"); enc_g.add_argument("--train-encoder", dest="freeze_encoder", action="store_false")
ap.set_defaults(freeze_encoder=True)
gc_g = ap.add_mutually_exclusive_group()
gc_g.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true", help="gradient checkpointing(기본 켬)"); gc_g.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
ap.set_defaults(grad_ckpt=True)
lg_g = ap.add_mutually_exclusive_group()
lg_g.add_argument("--no-liger", dest="liger", action="store_false", help="Liger 끔(기본 — rack4 env 에 liger 없음)"); lg_g.add_argument("--liger", dest="liger", action="store_true")
ap.set_defaults(liger=False)
ap.add_argument("--save-every", type=int, default=500); ap.add_argument("--save-total-limit", type=int, default=0, help="0 = checkpoint 를 지우지 않는다(원격 삭제 금지)")
ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--language-schedule", choices=("balanced", "proportional"), default="balanced")
ap.add_argument("--resume", default="auto", help="auto | none | <checkpoint dir>"); ap.add_argument("--gpu", default=None)
ap.add_argument("--force-resume", action="store_true", help="체크포인트의 config.semcommit 지문이 지금 설정·데이터와 달라도 재개(차이는 run 기록에 남는다)")
ap.add_argument("--allow-missing-lang", action="store_true", help="words 행이 있는 언어가 labels 짝이 없어 0 항목이 돼도 다른 언어만으로 학습")
ap.add_argument("--no-check-save", action="store_true", help="final 재로드 parity 검사 끔(기본 켬)"); ap.add_argument("--check-items", type=int, default=64, help="학습 전 셋마다 시퀀스 불변식 검사 항목 수")
ap.add_argument("--report-to", default="tensorboard"); ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()
if a.M != 0: sys.exit(f"--M {a.M}: v0 는 --M 0 만 지원 — M>0 이면 <SEM_END> 가 청크 한도에 세어져 단어와 다른 청크로 밀린다(계약 위반), 평가 패딩도 M=0 가정")
if a.gpu is not None: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
_cache_root = os.path.join(os.environ.get("VAPASR_LOCAL_CACHE", "/tmp"), f"vapasr-{os.getuid()}-semcommit")
for k_, d_ in (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")): os.environ.setdefault(k_, os.path.join(_cache_root, d_)); os.makedirs(os.environ[k_], exist_ok=True)
import torch
from vapasr.data.semcommit_dataset import SemCommitDataset, add_semcommit_specials, semcommit_fingerprint, checkpoint_fingerprint_diff
from vapasr.data.semcommit_tokens import SEM_SPECIALS
def log(*s): print(*s, flush=True)
torch.manual_seed(a.seed); random.seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True
out = a.out_dir; os.makedirs(out, exist_ok=True)
if not a.dry_run and torch.cuda.is_available() and torch.cuda.device_count() > 1: sys.exit("GPU 가 여러 장 보인다 — --gpu N 으로 한 장만 고른다(Trainer 가 DataParallel 로 감싸는 것 방지)")
IN_KEYS = ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask", "pos_weight")
LOSS_KEYS = ("loss_next", "loss_text", "top1_text", "loss_sem", "top1_sem", "sem_fp", "loss_turn", "top1_turn")

def tokenizer_src(init_dir):
    if init_dir and os.path.exists(os.path.join(init_dir, "tokenizer_config.json")): return init_dir
    assert a.qwen_dir, f"{init_dir} 에 tokenizer 파일이 없다 — --qwen-dir 로 Qwen3-ASR tokenizer 를 준다"; return a.qwen_dir

# ── 모델 + tokenizer
t0 = time.time(); model = None; is_hf = a.init != "qwen"
if is_hf:
    cfg_path = os.path.join(a.init, "config.json")
    assert os.path.exists(cfg_path) and json.load(open(cfg_path)).get("model_type") == "vapasr", f"--init 은 HF vapasr 디렉토리 또는 'qwen': {a.init}"
from transformers import AutoTokenizer
if a.dry_run:                                                                                 # 모델 없이 tokenizer 만
    tok = AutoTokenizer.from_pretrained(tokenizer_src(a.init if is_hf else None)); src = "dry-run"
else:
    from vapasr.hf import VapAsrForStreamingASR
    need = [n for n, ok in (("VapAsrForStreamingASR.add_semantic_tokens", hasattr(VapAsrForStreamingASR, "add_semantic_tokens")),
                            ("VapAsrForStreamingASR.forward(pos_weight=)", "pos_weight" in inspect.signature(VapAsrForStreamingASR.forward).parameters)) if not ok]
    if need: sys.exit(f"모델 쪽 semantic-commit 지원이 없다: {need} — 모델 모듈(add_semantic_tokens·soft_ce pos_weight/sem_weight/turn_weight)을 먼저 반영한다")
    if is_hf:
        model = VapAsrForStreamingASR.from_pretrained(a.init, encoder_path=a.nemotron_dir); tok = AutoTokenizer.from_pretrained(tokenizer_src(a.init)); src = f"hf {a.init}"
    else:
        assert a.qwen_dir, "--init qwen 은 --qwen-dir 필요"
        model, tok = VapAsrForStreamingASR.from_qwen(a.qwen_dir, encoder_path=a.nemotron_dir, next_weight=a.next_weight, next_weight_ko=a.next_weight_ko, delays=[int(x) for x in a.delays.split(",")]); src = f"qwen {a.qwen_dir}"
        if a.init_adapter:
            st0 = torch.load(a.init_adapter, map_location="cpu"); model.adapter.load_state_dict(st0["adapter"] if "adapter" in st0 else st0["state"]); src += f" + adapter {a.init_adapter}"
    old_sp = dict(model.config.sp_ids)
    ret = model.add_semantic_tokens(tok)                                                      # Phase1+Phase2+SEM 을 tokenizer 에 붙이고 새 행 초기화·config 갱신(모델 모듈)
    cfg = model.config; cfg.sem_weight, cfg.turn_weight = a.sem_weight, a.turn_weight; cfg.hardneg_weight = a.hardneg_weight
    cfg.next_weight, cfg.next_weight_ko, cfg.delays = a.next_weight, a.next_weight_ko, [int(x) for x in a.delays.split(",")]
    bad = {k: (v, tok.convert_tokens_to_ids(k)) for k, v in {**old_sp, **cfg.sp_ids}.items() if tok.convert_tokens_to_ids(k) != v}
    assert not bad, f"tokenizer 특수 토큰 id 가 config 와 다름: {bad}"
sp_ids = add_semcommit_specials(tok)                                                          # 이미 있으면 그대로(모델이 붙인 것과 같은 순서) — 순서·동결 id 대조
log(f"tokenizer ← {src} · <SEM_END>={sp_ids['<SEM_END>']} <TURN_END>={sp_ids['<TURN_END>']} · {time.time() - t0:.0f}s")
if model is not None:
    assert sp_ids["<SEM_END>"] < model.get_input_embeddings().weight.shape[0], "임베딩 행이 SEM id 보다 적다"
    reg = dict(getattr(model.config, "sem_registry", None) or {})                             # forward 가 SEM/TURN 가중·통계에 쓰는 id — 없으면 sem/turn_weight 가 조용히 무시된다
    assert reg == {k: sp_ids[k] for k in SEM_SPECIALS} and all(model.config.sp_ids.get(k) == sp_ids[k] for k in SEM_SPECIALS), f"config.sem_registry/sp_ids 불일치: {reg} (add_semantic_tokens → {ret})"
    # 인코더: 동결이면 HF init 의 encoder_trainable=True 도 실제로 끈다 + 체크포인트에 함께 저장(D1 교훈, p2_train_hf.py 와 같은 처리)
    init_had_encoder = bool(cfg.encoder_trainable or getattr(cfg, "encoder_saved", False))
    if a.freeze_encoder:
        model.encoder.set_trainable(False)
        for p_ in model.encoder.parameters(): p_.requires_grad_(False)
        cfg.encoder_trainable = False; cfg.encoder_saved = True
    else:
        model.encoder.set_trainable(True); cfg.encoder_trainable = True
    model._keys_to_ignore_on_save = None                                                      # 인코더 키를 저장에서 빼지 않는다(save_pretrained 가 encoder_saved 로 판단)
    n_enc_tr = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad); assert not a.freeze_encoder or n_enc_tr == 0
    if a.grad_ckpt: model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.thinker.config.use_cache = False
    if a.liger:
        try:
            from vapasr.hf.liger import apply_liger_to_thinker; log(f"liger: {apply_liger_to_thinker(model)}")
        except ImportError as e: log(f"liger 미적용({e})"); a.liger = False
    cfg.use_liger = a.liger
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model ← {src} ({time.time()-t0:.0f}s) · 학습 {n_tr/1e6:.1f}M · 인코더 {'동결' if a.freeze_encoder else '학습'}(init 인코더 저장됨={init_had_encoder}, 이번 저장={cfg.encoder_saved or cfg.encoder_trainable}) · "
        f"sem_w {cfg.sem_weight} turn_w {cfg.turn_weight} hardneg_w {a.hardneg_weight} · grad ckpt {a.grad_ckpt} · liger {a.liger}")

# ── 데이터
W = [p for x in a.train_words for p in x.split(",") if p]; L = [p for x in a.train_labels for p in x.split(",") if p]
assert len(W) == len(L), f"--train-words {len(W)} 개 ≠ --train-labels {len(L)} 개"
pmap = dict(x.split("=", 1) for x in a.path_map.split(",") if x)
delays = tuple(int(x) for x in a.delays.split(",")); train_sets = {}
for key, lang, cap in (("en", "English", a.max_items_en), ("ko", "Korean", a.max_items_ko)):
    ds = SemCommitDataset(W, L, tok, sp_ids, delays=delays, turn_end=not a.no_turn_end, hangover_s=a.hangover, hardneg_weight=a.hardneg_weight, max_items=cap or None,
                          seed=a.seed, online=True, allow_unlabeled=a.allow_unlabeled, max_per_chunk=a.M, langs=[lang], path_map=pmap, tail_margin=a.tail_margin)
    nw_, nl_ = ds.stats["words_rows"], ds.stats["labeled"]
    log(f"  {key}: words 행 {nw_} · labels 짝 {nl_} ({nl_ / max(1, nw_):.1%}) · {len(ds)} 스트림 · stats {dict(ds.stats)} · 제외 {dict(ds.bad)}")
    if ds.stats["labels_without_words"]: log(f"  !! {key}: words 행과 짝이 없는 label id {ds.stats['labels_without_words']} 개 — labels 가 다른 셋·표본에서 만들어졌을 수 있다")
    if not len(ds):
        if nw_ and not a.allow_missing_lang:
            sys.exit(f"{lang}: words 행 {nw_} 개가 있는데 학습 항목 0 개(labels 짝 {nl_}, 제외 {dict(ds.bad)}) — --train-words/--train-labels 짝을 확인하거나 --allow-missing-lang")
        continue
    ds.lang = lang; train_sets[key] = ds
assert train_sets, "학습 스트림 없음"
check = {}
for key, ds in train_sets.items():                                                            # 시퀀스 불변식(오디오 없이): SemCommitDataset.target_problems
    n = min(len(ds), a.check_items); bad = [(ds.items[i]["id"], d, pr[:3]) for i in range(n) for d in ds.delays for pr in [ds.target_problems(i, d)] if pr]
    check[key] = dict(n=n, bad=len(bad), examples=bad[:5], decision_on_event=ds.stats["decision_on_event"])
    log(f"  check {key}: {n} 항목 × δ{list(ds.delays)} → 불일치 {len(bad)} {bad[:3]} · 결정 위치 = TURN 타깃(유지) {ds.stats['decision_on_event']} (항목 {ds.stats['decision_on_event_items']})")
assert all(v["bad"] == 0 for v in check.values()), f"시퀀스 불변식 위반: {check}"
fp = semcommit_fingerprint(W, L, tok, turn_end=not a.no_turn_end, hangover_s=a.hangover, tail_margin=a.tail_margin, pad_tail=True, M=a.M, delays=list(delays),
                           sem_weight=a.sem_weight, turn_weight=a.turn_weight, hardneg_weight=a.hardneg_weight, next_weight=a.next_weight, next_weight_ko=a.next_weight_ko,
                           allow_unlabeled=a.allow_unlabeled, max_items_en=a.max_items_en, max_items_ko=a.max_items_ko, seed=a.seed, sem_ids={k: sp_ids[k] for k in SEM_SPECIALS})
log(f"  지문: words {[x[:12] for x in fp['words_sha256']]} labels {[x[:12] for x in fp['labels_sha256']]} vocab {(fp['vocab_sha256'] or '')[:12]}")
if a.dry_run:
    ds = next(iter(train_sets.values())); s = ds.sequence(0, max(ds.delays)); it = ds.items[0]; P = len(ds.prefix(it["lang"], max(ds.delays)))
    names = {v: k for k, v in sp_ids.items()}; audio = sum(1 for i in s["is_input"][P:] if i)
    def show(p):
        t = s["ids"][p]; lab = s["labels"][p]; w = s["pos_weight"][p]
        return ("A" if s["chunk_of"][p] >= 0 else names.get(t) or repr(tok.decode([t]))) + ("" if lab != -100 or s["is_input"][p] else "[mask]") + (f"[w={w}]" if w else "")
    log(f"dry-run 샘플 {it['id']} K {it['K0']}→{it['K']} δ{max(ds.delays)} · SEM {s['n_sem']} TURN {s['n_turn']}@{s['turn_chunk']} B {s['n_B']} N {s['n_N']} · 오디오 자리 {audio}")
    log(" ".join(show(p) for p in range(P, len(s["ids"]))))
    json.dump(dict(args=vars(a), sp_ids=sp_ids, semcommit=fp, sets={k: dict(v.stats) for k, v in train_sets.items()}, bad={k: dict(v.bad) for k, v in train_sets.items()}, check=check),
              open(os.path.join(out, "dry-run.json"), "w"), indent=1, ensure_ascii=False)
    log("dry-run 끝 →", os.path.join(out, "dry-run.json")); sys.exit(0)

# ── Trainer
from transformers import TrainingArguments
from vapasr.hf.trainer import VapAsrTrainer, PreemptCallback
from vapasr.data.semcommit_dataset import semcommit_round_robin

class SemTrainer(VapAsrTrainer):
    """데이터만 바꾼다: 언어 = 셋의 lang(이름이 manifest 가 아님), collate_semcommit. 손실·로그는 VapAsrTrainer 그대로(pos_weight 전달, loss_sem/loss_turn/top1_sem/sem_fp 키별 평균)."""
    def bs_of(self, name): return self.bs_ko if self.train_sets[name].lang == "Korean" else self.bs_en
    def get_train_dataloader(self):
        return semcommit_round_robin(self.train_sets, self.bs_of, seed=self.args.seed, num_workers=self.num_workers, schedule=self.language_schedule)
    def evaluate(self, *args, **kw): return {}                                                # v0: 자유 디코드 SEM/TURN 평가는 별도 스크립트

targs = TrainingArguments(output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=1, num_train_epochs=a.epochs, max_steps=a.max_steps if a.max_steps > 0 else -1,
                          learning_rate=a.lr, weight_decay=a.wd, warmup_steps=a.warmup, lr_scheduler_type="cosine", max_grad_norm=1.0, bf16=True,
                          logging_strategy="steps", logging_steps=a.log_every, logging_first_step=True, eval_strategy="no", save_strategy="steps", save_steps=a.save_every,
                          save_total_limit=a.save_total_limit, save_safetensors=True, seed=a.seed, data_seed=a.seed, report_to=[x for x in a.report_to.split(",") if x and x != "none"],
                          logging_dir=os.path.join(out, "tb"), remove_unused_columns=False, disable_tqdm=True, ignore_data_skip=True, dataloader_num_workers=a.num_workers,
                          label_names=["labels"], log_level="warning")
def last_complete_checkpoint(d):
    """checkpoint-N 중 trainer_state.json·model.safetensors 가 모두 있는 가장 큰 N(선점이 저장 도중 끊은 디렉토리는 건너뛴다)."""
    import re
    c = [(int(m.group(1)), os.path.join(d, x)) for x in os.listdir(d) for m in [re.fullmatch(r"checkpoint-(\d+)", x)]
         if m and all(os.path.exists(os.path.join(d, x, f)) for f in ("trainer_state.json", "model.safetensors"))]
    return max(c)[1] if c else None
model.config.semcommit = fp                                                                   # 모든 checkpoint-N·final 의 config.json 에 들어간다
last = last_complete_checkpoint(out) if a.resume == "auto" else (None if a.resume == "none" else a.resume); fdiff = []
if last:                                                                                      # 재개 = 같은 데이터·설정일 때만(지문 대조)
    fdiff = checkpoint_fingerprint_diff(last, fp)
    if fdiff and not a.force_resume: sys.exit(f"재개 거부: {last} 의 config.semcommit 이 지금 설정·데이터와 다르다 {fdiff[:8]} — 새 --out-dir 을 쓰거나(같은 out-dir 에 새로 시작하면 옛 checkpoint-N 과 섞인다), 차이를 알고 이어가려면 --force-resume")
    log(f"재개 ← {last}" + (f" · !! --force-resume: 지문 차이 {fdiff}" if fdiff else " · 지문 일치"))
preempt_cb = PreemptCallback(out, None)
trainer = SemTrainer(model=model, args=targs, train_sets=train_sets, dev_sets={}, tokenizer=tok, bs_en=a.bs_en, bs_ko=a.bs_ko, lr_adapter=a.lr_adapter, lr_encoder=a.lr_encoder,
                     max_per_chunk=a.M, num_workers=a.num_workers, language_schedule=a.language_schedule, callbacks=[preempt_cb], processing_class=tok)
tdl = trainer.get_train_dataloader()
log("배치/epoch: " + ", ".join(f"{m}:{len(s)}" for m, s in tdl.samplers.items()) + f" → steps/epoch {len(tdl)} · 배치 EN {a.bs_en} / KO {a.bs_ko}")
run_rec = dict(args=vars(a), init=src, sp_ids=sp_ids, semcommit=fp, resume_from=last, resume_fingerprint_diff=fdiff, trainable_m=n_tr / 1e6, encoder_saved=bool(model.config.encoder_saved),
               train_streams={k: len(v) for k, v in train_sets.items()}, sets={k: dict(v.stats) for k, v in train_sets.items()}, excluded={k: dict(v.bad) for k, v in train_sets.items()},
               check=check, steps_per_epoch=len(tdl), started=time.strftime("%F %T"))
run_path = os.path.join(out, time.strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid()}.json")          # 시작마다 새 파일(첫 설정을 덮어쓰지 않는다)
json.dump(run_rec, open(run_path, "w"), indent=1, ensure_ascii=False); log(f"실행 기록 → {run_path}")
res = trainer.train(resume_from_checkpoint=last); st = trainer.state
done = st.global_step >= st.max_steps and not preempt_cb.fired
log(f"train 종료: step {st.global_step}/{st.max_steps} {'완료' if done else '(선점/중단 → 재시작 시 이어서)'} · {res.metrics}")
if done:
    fin = os.path.join(out, "final"); trainer.save_model(fin); tok.save_pretrained(fin)
    json.dump(run_rec, open(os.path.join(fin, "semcommit_run.json"), "w"), indent=1, ensure_ascii=False)          # 평가가 학습 설정을 찾을 수 있게(config.json 의 semcommit 과 함께)
    json.dump(dict(args=vars(a), log_history=st.log_history[-300:], steps=st.global_step), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
    if not a.no_check_save:                                                                   # 재로드 parity: state_dict(인코더 포함)·한 배치 손실
        def batch_loss(m, x):
            m.eval(); xx = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in x.items()}
            nw = m.config.next_weight_ko if x["lang"][0] == "Korean" else m.config.next_weight
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): o = m(**{k: xx[k] for k in IN_KEYS if k in xx}, next_weight=nw)
            return {k: round(float(getattr(o, k)), 5) for k in ("loss",) + LOSS_KEYS if getattr(o, k, None) is not None}
        xb = next(iter(trainer.get_train_dataloader())); mem = trainer.model
        re_ = VapAsrForStreamingASR.from_pretrained(fin, encoder_path=a.nemotron_dir).cuda()
        sd1 = {k: v.detach().float().cpu() for k, v in mem.state_dict().items()}; sd2 = {k: v.detach().float().cpu() for k, v in re_.state_dict().items()}
        miss = sorted(set(sd1) - set(sd2)); extra = sorted(set(sd2) - set(sd1)); shape = [k for k in sd1 if k in sd2 and sd1[k].shape != sd2[k].shape]
        diff = sorted(((k, float((sd1[k] - sd2[k]).abs().max())) for k in sd1 if k in sd2 and sd1[k].shape == sd2[k].shape and float((sd1[k] - sd2[k]).abs().max()) > 1e-6), key=lambda x: -x[1])
        enc_diff = [d_ for d_ in diff if d_[0].startswith("encoder.")]; n_enc = sum(k.startswith("encoder.") for k in sd1)
        from safetensors import safe_open
        import glob
        saved_enc = sum(1 for f in glob.glob(os.path.join(fin, "*.safetensors")) for k in safe_open(f, "pt").keys() if k.startswith("encoder."))
        l1, l2 = batch_loss(mem, xb), batch_loss(re_, xb)
        try:
            from vapasr.hf import load_tokenizer; tk = load_tokenizer(fin); tok_ok = all(tk.convert_tokens_to_ids(k) == v for k, v in sp_ids.items()); tok_err = None
        except Exception as e: tok_ok, tok_err = False, f"{type(e).__name__}: {str(e)[:200]}"
        fp_ok = json.load(open(os.path.join(fin, "config.json"))).get("semcommit") == fp                # 평가가 복원할 학습 설정이 final 에 있는가
        parity = (tok_ok and fp_ok and not miss and not extra and not shape and not diff and saved_enc > 0 and abs(l1["loss"] - l2["loss"]) <= 5e-3 * max(1.0, abs(l1["loss"])))
        json.dump(dict(parity=parity, semcommit_config_ok=fp_ok, memory=l1, reloaded=l2, missing=miss[:50], extra=extra[:50], shape_mismatch=shape[:50], value_diff=diff[:50], encoder_keys=n_enc,
                       encoder_diff=enc_diff[:50], encoder_saved_keys=saved_enc, tokenizer_reload_ok=tok_ok, tokenizer_reload_error=tok_err, batch_n_sem=xb.get("n_sem")),
                  open(os.path.join(out, "parity.json"), "w"), indent=1)
        log(f"check-save: 키 {len(sd1)}/{len(sd2)} · 누락 {miss[:5]} · 추가 {extra[:5]} · shape {shape[:5]} · 값 차이 {len(diff)} {diff[:5]} · 인코더 저장 키 {saved_enc} · "
            f"손실 메모리 {l1} / 재로드 {l2} · load_tokenizer {'OK' if tok_ok else tok_err} · config.semcommit {'OK' if fp_ok else '불일치'}")
        log("PARITY OK" if parity else "PARITY FAIL — 저장된 final 이 학습 모델과 다르다(parity.json)")
        del re_; torch.cuda.empty_cache()
    open(os.path.join(out, "DONE"), "w").write(time.strftime("%F %T")); log("final →", fin)

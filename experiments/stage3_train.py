#!/usr/bin/env python
"""Stage 3 학습 — 2 인 같은 언어 대화(화자 귀속 + 스트리밍 ASR + 턴 끝 <EOT>) + Stage 2 단일 화자 replay
(wiki output-stage3-prep-plan-20261014 P5, decision-cst-training-stages-and-speaker-slots). 사용자 결정(2026-10-15): semcommit_train.py 와 별도 스크립트,
섞는 비율은 파일럿에서 비교해 정한다(--weights).

  torchrun ... experiments/stage3_train.py --init <Stage 2 final> --nemotron-dir <.nemo 디렉토리> --out-dir <run> \\
      --data <phase2 dir> --split <phase2>/splits/v2.json [--dlg-corpora …] \\
      --replay-words @<EN·KO words 목록> --replay-de-words @<독일어 목록> --ast-targets <번역 타깃> [--weights dlg_ko:1,…,rep_en:0.5] [--max-steps N]
  --dry-run: 모델 없이 tokenizer + 데이터셋만 만들어 시퀀스 불변식을 검사하고 샘플 하나를 찍는다(CPU).

데이터(셋 이름 = 가중치 키):
  dlg_ko·dlg_en·dlg_de  vapasr/data/cst_dialogue.CstDialogueDataset — Phase 2 정제 대화(split held-out 제외)의 20–40 s 창,
                        prefix 'language <SPK_A>{L} <SPK_B>{L}<asr_text><DELAY_d>', 화자 태그, 턴 끝 soft <EOT>(--eot-mode·--backchannel·--decision-weight).
  rep_ko·rep_en·rep_de  vapasr/data/ast_dataset.AstStreamDataset — Stage 2 와 같은 단일 화자 ASR·번역 항목(--ast-frac 번역 모드, --offline-frac final 모드).
                        <SEM_END>·<EOT> 둘 다 중립(sem_mask) — 단일 화자 스트림 끝은 턴 끝 라벨이 아니다.
  섞기: vapasr/hf/stage3.WeightedRoundRobinLoader — step 마다 셋을 가중치로 고른다(결정적). epoch = 대화 셋 배치를 한 번 도는 길이.
모델: Stage 2 체크포인트 + add_semantic_tokens(turn=True, reinit_turn=True) — 예약만 돼 있던 <EOT> 행을 다시 초기화하고 차단을 푼다,
  config.turn_weight = --turn-weight. 대화 행은 sem_only_mask(<SEM_END> 만 중립), forward 가 labels_alt/soft_w 로 두 점 soft CE.
나머지(인코더 학습·att_context 고정·SpecAugment·속도/잡음/음량 증강·packing·선점 저장·재개 지문·final 재로드 parity)는 semcommit_train.py 와 같은 규약.
지문: config.stage3(형식·코퍼스·split·창·EOT 규칙·가중치·δ·replay 입력·tokenizer 어휘) — 재개는 같은 지문일 때만(--force-resume 로만 진행).
저장: checkpoint-N 을 지우지 않는다(--save-total-limit 0). 완료 시 out-dir/final + parity.json(PARITY OK/FAIL) + DONE."""
import os, sys, json, time, argparse, random, hashlib, re, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="Stage 2 HF 체크포인트(config.json model_type=vapasr, tokenizer 포함)")
ap.add_argument("--nemotron-dir", default=None, help="*.nemo 가 든 디렉토리 — 체크포인트에 인코더가 없을 때만 쓰인다")
ap.add_argument("--out-dir", required=True)
g = ap.add_argument_group("대화(Stage 3)")
g.add_argument("--data", required=True, help="Phase 2 폴더(<corpus>.refined.dialogues.jsonl, mono/)")
g.add_argument("--dlg-corpora", default="aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc")
g.add_argument("--split", required=True, help="held-out split(splits/v2.json) — 그 conv_id 는 학습에서 뺀다")
g.add_argument("--mono-cache", default=None, help="mono 캐시(기본 <data>/mono, p2_build_mono.py)")
g.add_argument("--window", default="20,40", help="창 길이 범위 lo,hi(s)"); g.add_argument("--hop", type=float, default=10.0)
g.add_argument("--start-mode", choices=("grid", "mixed"), default="grid", help="mixed = 무음 격자 창 + 임의 시작 창(--random-frac)")
g.add_argument("--random-frac", type=float, default=0.5)
g.add_argument("--eot-mode", choices=("soft", "hard"), default="soft", help="soft = p_end 그대로, hard = p_end ≥ 0.5 → 1 아니면 0")
g.add_argument("--backchannel", default="off", help="맞장구 선택 'max_s,max_tokens'(예 1.0,3) | off(Phase 2 규칙)")
g.add_argument("--decision-weight", type=float, default=2.0, help="EOT 결정 위치 가중(pos_weight 덮어쓰기; Phase 2 eot_weight 2)")
g.add_argument("--turn-weight", type=float, default=2.0, help="config.turn_weight(<EOT> 타깃 가중 — 결정 위치는 --decision-weight 가 덮는다)")
g.add_argument("--max-windows", type=int, default=0, help="대화 셋마다 창 상한(0 = 전부; 스모크)")
g = ap.add_argument_group("replay(Stage 2 단일 화자)")
g.add_argument("--replay-words", action="append", default=[], help="EN·KO words.jsonl(반복·쉼표·@목록) — 행의 lang 으로 rep_en/rep_ko 에 나뉜다")
g.add_argument("--replay-de-words", action="append", default=[], help="독일어 words.jsonl(반복·쉼표·@목록) → rep_de")
g.add_argument("--replay-files-frac", type=float, default=1.0, help="words 파일 중 이 비율만 읽는다(경로 해시로 고정 선택 — 초기화 시간 절약)")
g.add_argument("--replay-max-items", type=int, default=0, help="replay 셋마다 항목 상한(0 = 전부)")
g.add_argument("--ast-targets", action="append", default=[], help="번역 타깃(ast_build_targets.py 산출, pack/ 권장; 반복·쉼표·@목록)")
g.add_argument("--ast-frac", type=float, default=0.8, help="번역 타깃이 있는 replay 항목을 이 확률로 번역 모드로")
g.add_argument("--offline-frac", type=float, default=0.0, help="replay 항목의 final(오프라인) 모드 비율")
g.add_argument("--gain-augment", default="off", help="replay 음량 증강 'p,lo_db,hi_db' | off"); g.add_argument("--gain-augment-langs", default="English")
g.add_argument("--speed-perturb", default="", help="replay 속도 변형 배율 목록(예 0.9,1.0,1.1)")
g.add_argument("--noise-bank", default=None, help="replay 잡음·잔향 뱅크"); g.add_argument("--noise-p", type=float, default=0.4)
g.add_argument("--noise-snr", default="5,30"); g.add_argument("--music-snr", default="10,30"); g.add_argument("--rir-p", type=float, default=0.2)
g = ap.add_argument_group("섞기·학습")
g.add_argument("--weights", default="dlg_ko:1,dlg_en:1,dlg_de:1,rep_ko:0.5,rep_en:0.5,rep_de:0.5", help="셋 가중치 'name:w,…' — step 비율(없는 셋 = 0)")
g.add_argument("--delays", default="4,5,6,7,8,9,10"); g.add_argument("--next-weight", type=float, default=0.3); g.add_argument("--next-weight-ko", type=float, default=0.15)
g.add_argument("--lr", type=float, default=6e-5); g.add_argument("--lr-adapter", type=float, default=1e-3); g.add_argument("--lr-encoder", type=float, default=1e-5)
g.add_argument("--warmup", type=int, default=500); g.add_argument("--wd", type=float, default=0.01)
g.add_argument("--epochs", type=float, default=1.0); g.add_argument("--max-steps", type=int, default=0)
enc_g = ap.add_mutually_exclusive_group()
enc_g.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true"); enc_g.add_argument("--train-encoder", dest="freeze_encoder", action="store_false")
ap.set_defaults(freeze_encoder=False)
ap.add_argument("--encoder-right-context", type=int, default=None); ap.add_argument("--encoder-context-sampling", choices=["multi", "fixed"], default="fixed")
ap.add_argument("--spec-augment", default="off")
gc_g = ap.add_mutually_exclusive_group()
gc_g.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true"); gc_g.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
ap.set_defaults(grad_ckpt=True)
lg_g = ap.add_mutually_exclusive_group()
lg_g.add_argument("--no-liger", dest="liger", action="store_false"); lg_g.add_argument("--liger", dest="liger", action="store_true")
ap.set_defaults(liger=True)
ap.add_argument("--save-every", type=int, default=500); ap.add_argument("--save-total-limit", type=int, default=0, help="0 = checkpoint 를 지우지 않는다(원격 삭제 금지)")
ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--resume", default="auto", help="auto | none | <checkpoint dir>"); ap.add_argument("--force-resume", action="store_true"); ap.add_argument("--gpu", default=None)
ap.add_argument("--no-check-save", action="store_true"); ap.add_argument("--check-items", type=int, default=64, help="학습 전 셋마다 시퀀스 불변식 검사 항목 수")
ap.add_argument("--report-to", default="tensorboard"); ap.add_argument("--dry-run", action="store_true")
from vapasr.speedup import add_speed_args, apply_speedups, training_args_kwargs
add_speed_args(ap)
a = ap.parse_args()

DLG_LANG = {"aihub71631": "Korean", "aihub134-1": "Korean", "aihub134-2": "Korean", "nikl2020": "Korean", "otoSpeech": "English", "kec": "German", "ztc": "German", "nsc": "German"}
SHORT = {"Korean": "ko", "English": "en", "German": "de"}
delays = tuple(int(x) for x in a.delays.split(",") if x.strip())
if not delays or not all(1 <= d <= 10 for d in delays): sys.exit(f"--delays {a.delays}: 1–10 의 쉼표 목록")
corpora = [c for c in a.dlg_corpora.split(",") if c]
if not all(c in DLG_LANG for c in corpora): sys.exit(f"--dlg-corpora: 알 수 없는 코퍼스 {[c for c in corpora if c not in DLG_LANG]} (알려진 것 {sorted(DLG_LANG)})")
win = tuple(float(x) for x in a.window.split(","))
if len(win) != 2 or not 0 < win[0] <= win[1]: sys.exit(f"--window lo,hi — {a.window}")
bc = None if a.backchannel == "off" else tuple(float(x) for x in a.backchannel.split(","))
if bc is not None and len(bc) != 2: sys.exit(f"--backchannel max_s,max_tokens | off — {a.backchannel}")
bc = (bc[0], int(bc[1])) if bc else None
from vapasr.hf.stage3 import parse_weights, WeightedRoundRobinLoader, collate_stage3, file_digest, fingerprint_diff
try: weights = parse_weights(a.weights)
except ValueError as e: sys.exit(f"--weights: {e}")
gain = None if a.gain_augment == "off" else tuple(float(x) for x in a.gain_augment.split(","))
gain_langs = tuple(x for x in a.gain_augment_langs.split(",") if x)
speeds = tuple(float(x) for x in a.speed_perturb.split(",") if x.strip())
if not 0 < a.replay_files_frac <= 1: sys.exit("--replay-files-frac 은 (0, 1]")
if not (a.batch_max_tokens or a.pack_max_tokens) and not a.dry_run: sys.exit("--batch-max-tokens 또는 --pack-max-tokens 가 필요하다(대화 창은 고정 크기 버킷 배치를 쓰지 않는다)")
from vapasr.features.online import SpecAugment
try: SpecAugment.from_spec(a.spec_augment)
except (ValueError, TypeError) as e: sys.exit(f"--spec-augment {a.spec_augment!r}: {e}")
from vapasr.data.noise_aug import from_args as noise_aug_from_args
try: naug = noise_aug_from_args(a.noise_bank, a.noise_p, a.noise_snr, a.rir_p, a.music_snr)
except (ValueError, OSError) as e: sys.exit(f"--noise-bank {a.noise_bank}: {e}")

rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
if a.gpu is not None and world == 1: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def _writable_root():
    """Triton/inductor 캐시·TMPDIR: VAPASR_LOCAL_CACHE → /tmp → out-dir/.cache/<host> (semcommit_train.py 와 같은 이유 — 일부 노드는 job step 안 /tmp 쓰기 불가)."""
    import socket
    name = f"vapasr-{os.getuid()}-stage3-{os.environ.get('SLURM_JOB_ID', 'local')}"
    for base in (os.environ.get("VAPASR_LOCAL_CACHE"), "/tmp", os.path.join(os.path.abspath(a.out_dir), ".cache", socket.gethostname())):
        if not base: continue
        root = os.path.join(base, name)
        try:
            os.makedirs(root, exist_ok=True); probe = os.path.join(root, f".probe-{os.getpid()}"); open(probe, "w").close(); os.remove(probe); return root
        except OSError: continue
    raise SystemExit("쓸 수 있는 캐시 디렉토리가 없다(VAPASR_LOCAL_CACHE·/tmp·out-dir)")
_cache_root = _writable_root()
for k_, d_ in (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")): os.environ.setdefault(k_, os.path.join(_cache_root, f"{d_}-{local}")); os.makedirs(os.environ[k_], exist_ok=True)
if not os.access(os.environ.get("TMPDIR", "/tmp"), os.W_OK):
    os.environ["TMPDIR"] = os.path.join(_cache_root, f"tmp-{local}"); os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    import tempfile; tempfile.tempdir = None
import torch, torch.distributed as dist
gloo_pg = None
if world > 1 and not a.dry_run:
    from datetime import timedelta
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    torch.cuda.set_device(local); dist.init_process_group("nccl", timeout=timedelta(hours=3), device_id=torch.device("cuda", local))
    gloo_pg = dist.new_group(backend="gloo", timeout=timedelta(hours=1))
main = rank == 0
def log(*s):
    if main: print(*s, flush=True)
torch.manual_seed(a.seed); random.seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True
out = a.out_dir; os.makedirs(out, exist_ok=True)
if not a.dry_run and world == 1 and torch.cuda.is_available() and torch.cuda.device_count() > 1: sys.exit("GPU 가 여러 장 보인다 — --gpu N 또는 torchrun")
IN_KEYS = ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask", "pos_weight", "position_ids", "labels_alt", "sem_mask", "sem_only_mask")
LOSS_KEYS = ("loss_next", "loss_text", "top1_text", "loss_turn", "top1_turn")

# ── 모델 + tokenizer
from transformers import AutoTokenizer
from vapasr.data.semcommit_dataset import add_semcommit_specials
from vapasr.data.semcommit_tokens import SEM_SPECIALS, TURN_TOKEN, add_delay_ext_specials, add_trans_special
cfg_path = os.path.join(a.init, "config.json")
assert os.path.exists(cfg_path) and json.load(open(cfg_path)).get("model_type") == "vapasr", f"--init 은 HF vapasr 디렉토리: {a.init}"
assert os.path.exists(os.path.join(a.init, "tokenizer_config.json")), f"{a.init} 에 tokenizer 가 없다 — Stage 2 final 을 준다"
t0 = time.time(); tok = AutoTokenizer.from_pretrained(a.init); model = None; speed = {}
if not a.dry_run:
    from vapasr.hf import VapAsrForStreamingASR
    from vapasr.data.semcommit_tokens import add_delay_tokens, add_ast_tokens
    model = VapAsrForStreamingASR.from_pretrained(a.init, encoder_path=a.nemotron_dir); cfg = model.config; old_sp = dict(cfg.sp_ids)
    assert cfg.lanes == 0, "Stage 3 는 mono(lanes=0) Stage 2 체크포인트에서 시작한다"
    model.add_semantic_tokens(tok, turn=True, reinit_turn=True)                              # <EOT> 를 턴 종료로 켠다(예약 행 재초기화·차단 해제)
    if max(delays) > 8: add_delay_tokens(model, tok)                                         # Stage 2 에서 이미 붙었으면 그대로
    add_ast_tokens(model, tok)                                                                # <TRANS>·<SPK_A/B> 차단 해제(이미 있으면 그대로)
    cfg.turn_weight = a.turn_weight; cfg.next_weight, cfg.next_weight_ko, cfg.delays = a.next_weight, a.next_weight_ko, list(delays)
    bad = {k: (v, tok.convert_tokens_to_ids(k)) for k, v in {**old_sp, **cfg.sp_ids}.items() if tok.convert_tokens_to_ids(k) != v}
    assert not bad, f"tokenizer 특수 토큰 id 가 config 와 다름: {bad}"
sp_ids = add_semcommit_specials(tok)
if max(delays) > 8: sp_ids = {**sp_ids, **add_delay_ext_specials(tok)}
trans_id = add_trans_special(tok)["<TRANS>"]
EOT = sp_ids[TURN_TOKEN]
if model is not None:
    reg = dict(cfg.sem_registry or {})
    assert reg == {k: sp_ids[k] for k in SEM_SPECIALS + [TURN_TOKEN]}, f"config.sem_registry 가 <SEM_END>·<EOT> 가 아니다: {reg}"
    assert EOT not in cfg.blocked_ids and sp_ids["<SPK_A>"] not in cfg.blocked_ids and sp_ids["<SPK_B>"] not in cfg.blocked_ids, "<EOT>/<SPK_A>/<SPK_B> 가 decode 차단 목록에 있다"
    init_had_encoder = bool(cfg.encoder_trainable or getattr(cfg, "encoder_saved", False))
    if a.freeze_encoder:
        model.encoder.set_trainable(False)
        for p_ in model.encoder.parameters(): p_.requires_grad_(False)
        cfg.encoder_trainable = False; cfg.encoder_saved = True
    else:
        model.encoder.set_trainable(True); cfg.encoder_trainable = True
    if a.encoder_right_context is not None and a.encoder_right_context != cfg.encoder_right_context:
        ctx = [int(cfg.encoder_left_context), int(a.encoder_right_context)]; allowed = [list(map(int, c)) for c in (getattr(model.encoder.enc, "att_context_size_all", None) or [])]
        assert not allowed or ctx in allowed, f"--encoder-right-context {a.encoder_right_context}: 학습한 att_context 가 아니다 {allowed}"
        model.encoder.enc.set_default_att_context_size(ctx); cfg.encoder_right_context = ctx[1]
    if not a.freeze_encoder and a.encoder_context_sampling == "fixed":
        ctx = [int(cfg.encoder_left_context), int(cfg.encoder_right_context)]; ee = model.encoder.enc
        ee.att_context_size_all, ee.att_context_probs = [ctx], [1.0]; ee.set_default_att_context_size(ctx)
        log(f"인코더 학습 att_context 고정 {ctx}")
    model._keys_to_ignore_on_save = None
    sa = model.encoder.set_spec_augment(a.spec_augment)
    if sa is not None: log(f"SpecAugment {sa.extra_repr()}")
    if a.grad_ckpt: model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.thinker.config.use_cache = False
    speed = apply_speedups(model, a, log, liger=a.liger); a.liger = bool(speed["liger"]); cfg.use_liger = a.liger
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model ← {a.init} ({time.time() - t0:.0f}s) · 학습 {n_tr / 1e6:.1f}M · 인코더 {'동결' if a.freeze_encoder else '학습'}(init 인코더 저장됨={init_had_encoder}) · "
        f"<EOT>={EOT} 켬(turn_weight {cfg.turn_weight}) · grad ckpt {a.grad_ckpt} · liger {a.liger}")

# ── 데이터
def _paths(vals):
    return [q for x in vals for q in ([l.strip() for l in open(x[1:]) if l.strip()] if x.startswith("@") else x.split(",")) if q]
def _subset(paths, frac):
    """경로 해시로 고정 선택(--replay-files-frac) — 같은 frac 이면 늘 같은 파일."""
    if frac >= 1.0: return list(paths)
    keep = [p for p in paths if int(hashlib.sha1(p.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac]
    return keep or list(paths[:1])
from vapasr.data.cst_dialogue import CstDialogueDataset
split = json.load(open(a.split, encoding="utf-8")); mono = a.mono_cache or os.path.join(a.data, "mono")
train_sets, dlg_files = {}, []
for lang in ("Korean", "English", "German"):
    cs = [c for c in corpora if DLG_LANG[c] == lang]; key = f"dlg_{SHORT[lang]}"
    if not cs or weights.get(key, 0) <= 0: continue
    files = [os.path.join(a.data, f"{c}.refined.dialogues.jsonl") for c in cs]; dlg_files += files
    held = {cid for c in cs for cid in split["corpora"].get(c, {}).get("heldout", [])}
    ds = CstDialogueDataset(files, tok, delays=delays, exclude=held, window_s=win, hop_s=a.hop, start_mode=a.start_mode, random_frac=a.random_frac,
                            seed=a.seed, mono_cache_dir=mono, decision_weight=a.decision_weight, backchannel=bc, eot_mode=a.eot_mode)
    if a.max_windows and len(ds) > a.max_windows: ds.items = random.Random(a.seed).sample(ds.items, a.max_windows)
    ds.lang = lang; train_sets[key] = ds
    log(f"  {key} {cs}: {len(ds)} 창 · held-out 제외 {ds.stats['cst_excluded']} 대화 · stats {dict(ds.stats)}")
RW = _subset(_paths(a.replay_words), a.replay_files_frac); RD = _subset(_paths(a.replay_de_words), a.replay_files_frac); ast_fp = None
if (RW or RD) and any(weights.get(k, 0) > 0 for k in ("rep_ko", "rep_en", "rep_de")):
    from vapasr.data.ast_dataset import AstStreamDataset, load_ast_targets
    AST, ast_st, ast_fp = load_ast_targets(_paths(a.ast_targets)) if a.ast_targets else ({}, {}, None)
    log(f"  replay 번역 타깃 {dict(ast_st)} · 번역 모드 {a.ast_frac} · words 파일 EN·KO {len(RW)} · DE {len(RD)} (frac {a.replay_files_frac})")
    for key, lang, ww in (("rep_en", "English", RW), ("rep_ko", "Korean", RW), ("rep_de", "German", RD)):
        if not ww or weights.get(key, 0) <= 0: continue
        ds = AstStreamDataset(ww, [], tok, sp_ids, delays=delays, turn_end=False, hangover_s=0.48, hardneg_weight=1.0, max_items=a.replay_max_items or None,
                              seed=a.seed, online=True, allow_unlabeled=False, max_per_chunk=0, langs=[lang], path_map={}, tail_margin=2, source_pool="main",
                              ko_textnorm=True, written_frac=0.0, sem_free_unlabeled=True, sem_free_max_ratio=None, offline_frac=a.offline_frac, ignore_labels=True,
                              speed_perturb=speeds, noise_aug=naug, ast_targets=AST, ast_frac=a.ast_frac, trans_id=trans_id, gain_aug=gain, gain_aug_langs=gain_langs)
        if not len(ds): log(f"  {key}: 항목 0 — 건너뜀"); continue
        ds.lang = lang; train_sets[key] = ds; log(f"  {key}: {len(ds)} 항목 · 번역 타깃 있는 항목 {ds.stats.get('ast_items', 0)}")
missing = sorted(k for k, w in weights.items() if w > 0 and k not in train_sets)
if missing: sys.exit(f"가중치 > 0 인데 만들어지지 않은 셋: {missing} (입력·코퍼스를 확인하거나 --weights 에서 뺀다)")
assert any(k.startswith("dlg_") for k in train_sets), "대화 셋이 없다"
check = {}
for key, ds in train_sets.items():                                                             # 시퀀스 불변식(오디오 없이)
    n = min(len(ds), a.check_items); rng = random.Random(a.seed); idx = sorted(rng.sample(range(len(ds)), n))
    if key.startswith("dlg_"): bad = [(ds.items[i], d, P[:3]) for i in idx for d in ds.delays for P in [ds.target_problems(i, d)] if P]
    else: bad = [(ds.items[i]["id"], d, P[:3]) for i in idx for d in ds.delays for P in [ds.target_problems(i, d)] if P]
    check[key] = dict(n=n, bad=len(bad), examples=[str(b)[:200] for b in bad[:5]]); log(f"  check {key}: {n} 항목 × δ{list(delays)} → 불일치 {len(bad)} {bad[:2]}")
assert all(v["bad"] == 0 for v in check.values()), f"시퀀스 불변식 위반: {check}"
vocab_sha = hashlib.sha256(json.dumps(sorted(tok.get_vocab().items())).encode()).hexdigest()
fp = dict(format="cst-v1", corpora=corpora, split_sha256=hashlib.sha256(open(a.split, "rb").read()).hexdigest(), dialogues=file_digest(dlg_files),
          window=list(win), hop=a.hop, start_mode=a.start_mode, random_frac=a.random_frac if a.start_mode == "mixed" else None, eot_mode=a.eot_mode,
          backchannel=list(bc) if bc else None, decision_weight=a.decision_weight, turn_weight=a.turn_weight, weights=weights, delays=list(delays),
          next_weight=a.next_weight, next_weight_ko=a.next_weight_ko, replay_words=file_digest(RW) if RW else None, replay_de_words=file_digest(RD) if RD else None,
          replay_files_frac=a.replay_files_frac, replay_max_items=a.replay_max_items, ast_targets_sha256=ast_fp, ast_frac=a.ast_frac, offline_frac=a.offline_frac,
          gain_augment=list(gain) if gain else None, speed_perturb=list(speeds) or None, noise_aug=naug.settings() if naug else None, spec_augment=a.spec_augment,
          max_windows=a.max_windows, seed=a.seed, vocab_sha256=vocab_sha, init=os.path.abspath(a.init))
if a.dry_run:
    names = {v: k for k, v in sp_ids.items()}
    for key, ds in train_sets.items():
        if not key.startswith("dlg_"): continue
        s = ds.cst_sequence(0, max(delays)); P = len(ds.cst_prefix(ds.dlgs[s["cid"]].lang, max(delays)))
        def show(p):
            t = s["ids"][p]; lab = s["labels"][p]
            return ("A" if s["chunk_of"][p] >= 0 else names.get(t) or repr(tok.decode([t]))) + ("" if lab != -100 or s["is_input"][p] else "[m]") + (f"[p={s['soft_w'][p]}]" if p in s["soft_pos"] else "")
        log(f"dry-run {key} {s['cid']}@{s['t0']} L {s['L']} δ{max(delays)} · 화자 {s['speakers']} · EOT {s['n_eot']} soft {s['n_soft']} · 머리 {tok.decode(s['ids'][:P])!r}")
        log(" ".join(show(p) for p in range(P, min(len(s["ids"]), P + 400))))
    json.dump(dict(args=vars(a), sp_ids=sp_ids, stage3=fp, sets={k: dict(v.stats) for k, v in train_sets.items()}, sizes={k: len(v) for k, v in train_sets.items()}, check=check),
              open(os.path.join(out, "dry-run.json"), "w"), indent=1, ensure_ascii=False, default=str)
    log("dry-run 끝 →", os.path.join(out, "dry-run.json")); sys.exit(0)

# ── Trainer
from transformers import TrainingArguments
from vapasr.hf.trainer import VapAsrTrainer, PreemptCallback
from vapasr.hf.packing import budget_sampler_factory

class Stage3Trainer(VapAsrTrainer):
    """데이터: 셋 가중 라운드로빈 + collate_stage3. 로그: 대화/replay 별 손실·턴 손실을 따로 평균(loss_dlg·loss_rep·loss_turn_dlg·soft_per_step)."""
    def bs_of(self, name): return self.bs_ko if self.train_sets[name].lang == "Korean" else self.bs_en
    def get_train_dataloader(self):
        return WeightedRoundRobinLoader(self.train_sets, self.bs_of, weights=weights, anchor=[k for k in self.train_sets if k.startswith("dlg_")], seed=self.args.seed,
                                        rank=self.args.process_index, world=self.args.world_size, num_workers=self.num_workers, collate_fn=collate_stage3,
                                        sampler_factory=budget_sampler_factory(self.pack_max_tokens, self.batch_max_tokens, self.pack_max_bs))
    def evaluate(self, *args, **kw): return {}
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, o = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        kind = "dlg" if "labels_alt" in inputs else "rep"; c = self.__dict__.setdefault("_kind", {})
        c.setdefault(f"loss_{kind}", []).append(float(loss.detach()))
        if kind == "dlg" and getattr(o, "loss_turn", None) is not None: c.setdefault("loss_turn_dlg", []).append(float(o.loss_turn))
        if kind == "dlg": c.setdefault("soft", []).append(int(inputs.get("n_soft", 0)))
        return (loss, o) if return_outputs else loss
    def log(self, logs, start_time=None):
        c = self.__dict__.get("_kind")
        if c and "loss" in logs:
            for k, v in c.items():
                if v: logs["soft_per_dlg_step" if k == "soft" else k] = round(sum(v) / len(v), 4)
            self._kind = {}
        return super().log(logs, start_time)

targs = TrainingArguments(output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=1, num_train_epochs=a.epochs, max_steps=a.max_steps if a.max_steps > 0 else -1,
                          learning_rate=a.lr, weight_decay=a.wd, warmup_steps=a.warmup, lr_scheduler_type="cosine", max_grad_norm=1.0, **{"bf16": True, **training_args_kwargs(a)},
                          logging_strategy="steps", logging_steps=a.log_every, logging_first_step=True, eval_strategy="no", save_strategy="steps", save_steps=a.save_every,
                          save_total_limit=a.save_total_limit, save_safetensors=True, seed=a.seed, data_seed=a.seed, report_to=[x for x in a.report_to.split(",") if x and x != "none"],
                          logging_dir=os.path.join(out, "tb"), remove_unused_columns=False, disable_tqdm=True, ignore_data_skip=True, dataloader_num_workers=a.num_workers,
                          label_names=["labels"], log_level="warning" if main else "error", ddp_find_unused_parameters=False, ddp_broadcast_buffers=False, ddp_timeout=10800)
def last_complete_checkpoint(d):
    c = [(int(m.group(1)), os.path.join(d, x)) for x in os.listdir(d) for m in [re.fullmatch(r"checkpoint-(\d+)", x)]
         if m and all(os.path.exists(os.path.join(d, x, f)) for f in ("trainer_state.json", "model.safetensors"))]
    return max(c)[1] if c else None
model.config.stage3 = fp
last = last_complete_checkpoint(out) if a.resume == "auto" else (None if a.resume == "none" else a.resume); fdiff = []
if last:
    fdiff = fingerprint_diff(last, fp)
    if fdiff and not a.force_resume: sys.exit(f"재개 거부: {last} 의 config.stage3 가 지금 설정·데이터와 다르다 {fdiff[:8]} — 새 --out-dir 을 쓰거나 --force-resume")
    log(f"재개 ← {last}" + (f" · !! --force-resume: 지문 차이 {fdiff}" if fdiff else " · 지문 일치"))
preempt_cb = PreemptCallback(out, gloo_pg)
trainer = Stage3Trainer(model=model, args=targs, train_sets=train_sets, dev_sets={}, tokenizer=tok, bs_en=6, bs_ko=16, lr_adapter=a.lr_adapter, lr_encoder=a.lr_encoder,
                        max_per_chunk=0, num_workers=a.num_workers, language_schedule="balanced", gloo_pg=gloo_pg,
                        pack_max_tokens=a.pack_max_tokens, pack_max_bs=a.pack_max_bs, batch_max_tokens=a.batch_max_tokens, callbacks=[preempt_cb], processing_class=tok)
tdl = trainer.get_train_dataloader(); plan = tdl.describe()
log(f"섞기: steps/epoch {plan['steps_per_epoch']} · 비중 {plan['share']} · 셋 배치/epoch {plan['batches_per_epoch']} · world {world}")
run_rec = dict(args=vars(a), init=a.init, sp_ids=sp_ids, stage3=fp, resume_from=last, resume_fingerprint_diff=fdiff, trainable_m=n_tr / 1e6, encoder_saved=bool(model.config.encoder_saved),
               sizes={k: len(v) for k, v in train_sets.items()}, sets={k: dict(v.stats) for k, v in train_sets.items()}, plan=plan, check=check, world=world, speed=speed,
               started=time.strftime("%F %T"))
run_path = os.path.join(out, time.strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid()}.json")
if main: json.dump(run_rec, open(run_path, "w"), indent=1, ensure_ascii=False, default=str); log(f"실행 기록 → {run_path}")
res = trainer.train(resume_from_checkpoint=last); st = trainer.state
done = st.global_step >= st.max_steps and not preempt_cb.fired
log(f"train 종료: step {st.global_step}/{st.max_steps} {'완료' if done else '(선점/중단 → 재시작 시 이어서)'} · {res.metrics}")
if done:
    fin = os.path.join(out, "final"); trainer.save_model(fin)
if done and main:
    tok.save_pretrained(fin)
    json.dump(run_rec, open(os.path.join(fin, "stage3_run.json"), "w"), indent=1, ensure_ascii=False, default=str)
    json.dump(dict(args=vars(a), log_history=st.log_history[-300:], steps=st.global_step), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
    if not a.no_check_save:                                                                   # 재로드 parity: state_dict(인코더 포함)·대화 배치 손실·tokenizer·지문
        def batch_loss(m, x):
            m.eval(); xx = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in x.items()}
            nw = m.config.next_weight_ko if x["lang"][0] == "Korean" else m.config.next_weight
            kw = {k: xx[k] for k in IN_KEYS if k in xx}
            if "soft_w_full" in xx: kw["soft_w"] = xx["soft_w_full"]
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): o = m(**kw, next_weight=nw)
            return {k: round(float(getattr(o, k)), 5) for k in ("loss",) + LOSS_KEYS if getattr(o, k, None) is not None}
        dk = next(k for k in train_sets if k.startswith("dlg_"))
        from vapasr.data.cst_dialogue import collate_cst
        xb = collate_cst([train_sets[dk][i] for i in range(min(4, len(train_sets[dk])))]); mem = getattr(trainer.model, "module", trainer.model)
        re_ = VapAsrForStreamingASR.from_pretrained(fin, encoder_path=a.nemotron_dir).cuda()
        sd1 = {k: v.detach().float().cpu() for k, v in mem.state_dict().items()}; sd2 = {k: v.detach().float().cpu() for k, v in re_.state_dict().items()}
        miss = sorted(set(sd1) - set(sd2)); extra = sorted(set(sd2) - set(sd1)); shape = [k for k in sd1 if k in sd2 and sd1[k].shape != sd2[k].shape]
        diff = sorted(((k, float((sd1[k] - sd2[k]).abs().max())) for k in sd1 if k in sd2 and sd1[k].shape == sd2[k].shape and float((sd1[k] - sd2[k]).abs().max()) > 1e-6), key=lambda x: -x[1])
        from safetensors import safe_open
        saved_enc = sum(1 for f in glob.glob(os.path.join(fin, "*.safetensors")) for k in safe_open(f, "pt").keys() if k.startswith("encoder."))
        l1, l2 = batch_loss(mem, xb), batch_loss(re_, xb)
        try:
            from vapasr.hf import load_tokenizer; tk = load_tokenizer(fin); tok_ok = all(tk.convert_tokens_to_ids(k) == v for k, v in sp_ids.items()); tok_err = None
        except Exception as e: tok_ok, tok_err = False, f"{type(e).__name__}: {str(e)[:200]}"
        rc = json.load(open(os.path.join(fin, "config.json"))); fp_ok = rc.get("stage3") == json.loads(json.dumps(fp))
        reg_ok = rc.get("sem_registry") == {k: sp_ids[k] for k in SEM_SPECIALS + [TURN_TOKEN]} and EOT not in rc.get("blocked_ids", [])
        parity = (tok_ok and fp_ok and reg_ok and not miss and not extra and not shape and not diff and saved_enc > 0 and abs(l1["loss"] - l2["loss"]) <= 5e-3 * max(1.0, abs(l1["loss"])))
        json.dump(dict(parity=parity, stage3_config_ok=fp_ok, eot_registry_ok=reg_ok, memory=l1, reloaded=l2, missing=miss[:50], extra=extra[:50], shape_mismatch=shape[:50],
                       value_diff=diff[:50], encoder_saved_keys=saved_enc, tokenizer_reload_ok=tok_ok, tokenizer_reload_error=tok_err, batch_n_soft=xb.get("n_soft")),
                  open(os.path.join(out, "parity.json"), "w"), indent=1)
        log(f"check-save: 누락 {miss[:5]} · 추가 {extra[:5]} · shape {shape[:5]} · 값 차이 {len(diff)} {diff[:5]} · 인코더 저장 키 {saved_enc} · 손실 메모리 {l1} / 재로드 {l2} · "
            f"tokenizer {'OK' if tok_ok else tok_err} · config.stage3 {'OK' if fp_ok else '불일치'} · <EOT> 등록 {'OK' if reg_ok else '불일치'}")
        log("PARITY OK" if parity else "PARITY FAIL — 저장된 final 이 학습 모델과 다르다(parity.json)")
        del re_; torch.cuda.empty_cache()
    open(os.path.join(out, "DONE"), "w").write(time.strftime("%F %T")); log("final →", fin)
if world > 1: dist.destroy_process_group()

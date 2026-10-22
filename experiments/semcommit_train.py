#!/usr/bin/env python
"""Semantic commit — Phase 1 mono 스트림에 <SEM_END> 를 끼워 단일 GPU 로 학습한다 (raw/inbox/streaming_asr_semantic_commit_plan.md §21·§23).
턴 종료는 Phase 2 의 <EOT> 로 통일(<TURN_END> 폐기, 2026-09-24): 기본은 <SEM_END> 만, --turn-end 면 mono 에서도 <EOT> 를 발화 끝 proxy 로 함께 학습한다.

  python experiments/semcommit_train.py --init <HF vapasr 디렉토리> --qwen-dir /data4/.../Qwen3-ASR-0.6B --nemotron-dir <*.nemo 가 든 디렉토리> \\
      --train-words en.words.jsonl,ko.words.jsonl --train-labels en.labels.jsonl,ko.labels.jsonl --out-dir /data3/tskim/semcommit/v0-smoke --max-steps 300 --gpu 0
  --init qwen : Qwen3-ASR(--qwen-dir) 에서 새 모델(+ --init-adapter). 작은 run 은 ASR 부터 수렴하지 않으므로 학습된 Phase 1 HF 체크포인트를 권한다.
  --dry-run   : 모델 없이 tokenizer + 데이터셋만 만들어 시퀀스 불변식(SEM 위치·TURN flush 밖·B/N 결정 위치)을 검사하고 샘플 하나를 찍고 끝난다(CPU).
데이터: vapasr/data/semcommit_dataset.SemCommitDataset(words.jsonl + labels.jsonl, id 로 짝). --train-words/--train-labels 는 반복 또는 쉼표 목록(같은 순서)이며
  행의 lang 으로 EN/KO 셋을 나눠 VapAsrTrainer 라운드로빈(언어별 배치 bs-en/bs-ko, collate_semcommit)으로 번갈아 낸다.
손실: soft_ce 의 NEXT=next_weight(EN 0.3 / KO 0.15)·SEM=--sem-weight·턴 종료(--turn-end 일 때)=--turn-weight·결정 위치 pos_weight(N 후보 = --hardneg-weight). 모델 쪽
  add_semantic_tokens / forward(pos_weight=) 가 있어야 한다(없으면 시작 전에 멈춘다 — pos_weight 가 조용히 버려지는 것을 막는다).
인코더: 기본 동결(--freeze-encoder). HF init 의 config.encoder_trainable=True 여도 실제로 requires_grad 를 끄고 encoder_saved=True 로 체크포인트에 함께 저장한다
  (2026-09-17 D1 사고 — 동결 인코더를 저장에서 빼면 재로드 때 .nemo 원본이 붙는다; _keys_to_ignore_on_save 에 기대지 않는다).
저장: checkpoint-N 을 지우지 않는다(--save-total-limit 0). 완료 시 out-dir/final + 재로드 parity(state_dict·인코더·한 배치 손실·load_tokenizer·config.semcommit) → parity.json, PARITY OK/FAIL.
지문: config.semcommit = semcommit_fingerprint(turn_end·hangover_s·tail_margin·pad_tail·M·delays·가중·표본 설정 + words/labels 파일·tokenizer 어휘 sha256) — 모든 체크포인트와 final 에 들어간다.
  재개(auto 또는 --resume DIR)는 그 체크포인트의 config.semcommit 과 다르면 거부(--force-resume 로만 진행). 평가(semcommit_eval)는 final/config.json 의 semcommit 으로
  학습 패딩(--turn-end·--hangover-s·--tail-margin·δ)을 복원할 수 있다. 실행 기록은 run-<시각>-<pid>.json(덮어쓰지 않음), final/semcommit_run.json.
언어 셋: words 행이 있는 언어가 labels 짝이 없어 0 항목이 되면 멈춘다(--allow-missing-lang 로만 진행). words↔labels 짝 비율·짝 없는 label id 를 찍는다.
--M 은 0 만: M>0 이면 <SEM_END> 가 청크 한도에 세어져 단어와 다른 청크로 밀린다(v0 계약 위반; 평가 패딩 commit_metrics.train_pad_K 도 M=0 가정).
출력 형식(outfmt-v1, vapasr/data/output_format.py): 기본은 정규형만(빈 system 칸 — 이 기능 전과 시퀀스·지문이 같아 진행 중 run 을 재개할 수 있다).
  --written-frac[-en|-ko] F > 0 이면 표기형이 있는 항목(KO speechlm display_text · EN pnc_text 정렬, semcommit_written)을 확률 F 로 표기형
  (system 칸 = 고정 지시문, 구두점·대소문자·숫자 타깃)으로 학습한다. 그때만 지문에 output_format 이, 모델 config 에 output_formats(지시문 머리 ids)가 들어간다.
  비용: 모든 rank 가 데이터셋을 따로 만들므로 EN 표기형 빌드(pnc 정렬·재토큰화·selfcheck)가 초기화를 행당 약 1.6–3.4 ms 늘린다(mxc 로그인 노드 실측, KO 는 ~0.1 ms).
  로그 written_share = rank 0 배치의 전체 항목 중 표기형 비율, written_share_<en|ko> = 표기형을 켠 언어 항목 중 비율.
ASR 정확도 옵션(2026-09-29, 기본은 모두 꺼짐 — 끄면 시퀀스·난수·지문이 이전과 같다):
  --asr-words: 라벨 없는 words.jsonl 을 SEM 중립 ASR 행으로(텍스트·NEXT 만 배우고 <SEM_END> 는 그 행 softmax 에서 뺀다 — sem_mask), --asr-max-ratio 로 셋마다 상한.
  --speed-perturb 0.9,1.0,1.1: 항목마다 배율을 골라 리샘플 + 이벤트 시각 1/배율. --spec-augment light|nemo: 입력 log-mel 마스크(학습 중에만).
  --noise-bank DIR: 잡음·잔향 증강(MUSAN·RIRS_NOISES·DEMAND·ETSI 뱅크, --noise-p/--noise-snr/--rir-p/--music-snr).
  인코더 학습은 --train-encoder(--lr-encoder, 기본 1e-5; E2 레시피) — 활성화 메모리가 커서 --batch-max-tokens 를 줄여야 한다.
독일어(2026-10-08): --de-words(반복·쉼표 목록·@목록파일)의 words.jsonl 을 독일어 셋 'de' 로 더한다(YODAS-Granary, experiments/yodas_de_align.py).
  라벨이 없으므로 늘 SEM 중립 ASR 행이고, 독일어 셋만 이 파일을 읽는다(EN/KO 셋은 다시 읽지 않는다). 언어 머리는 'language German<asr_text>',
  NEXT 가중은 영어와 같은 --next-weight. 셋 교대는 --language-schedule(기본 balanced = 언어별 같은 step 수). --de-words 가 없으면 지문·시퀀스가 이전과 같다.
지연 확장(2026-10-08): --delays 에 9·10 이 있으면 <DELAY_9>·<DELAY_10> 을 SEM 뒤에 붙인다(vapasr/data/semcommit_tokens.add_delay_tokens —
  새 행은 <DELAY_8> 복사, decode 차단). 예: --delays 4,5,6,7,8,9,10 + 인코더 [56,3] → 미리 듣기 320–1,040 ms. 꼬리 패딩은 가장 큰 δ 에 맞춰 항목마다 자동.
MXC_* 환경 변수가 없어도 돈다(모든 경로는 인자)."""
import os, sys, json, time, argparse, random, inspect
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="HF vapasr 디렉토리(config.json model_type=vapasr) | qwen (--qwen-dir 에서 새 모델)")
ap.add_argument("--qwen-dir", default=None, help="Qwen3-ASR 디렉토리: --init qwen 의 원본, 또는 init 에 tokenizer 파일이 없을 때 tokenizer 출처")
ap.add_argument("--init-adapter", default=None, help="--init qwen 일 때 증류 adapter.pt ('adapter' 또는 'state' 키)")
ap.add_argument("--nemotron-dir", default=None, help="*.nemo 가 든 디렉토리(예: HF 캐시 snapshot) — NemotronOnline(path=) 로 전달")
ap.add_argument("--train-words", action="append", required=True, help="words.jsonl (반복·쉼표 목록·@목록파일)"); ap.add_argument("--train-labels", action="append", default=[], help="labels.jsonl (--train-words 와 같은 순서; --asr-only 면 생략·무시)")
ap.add_argument("--short-train-words", action="append", default=[], help="승인된 speechlm <8s words.jsonl; 본 데이터와 별도 풀")
ap.add_argument("--short-train-labels", action="append", default=[], help="short words 와 같은 순서의 labels.jsonl")
ap.add_argument("--short-step-fraction", type=float, default=0.10, help="short 풀의 optimizer step 비율 상한(기본 0.10; short 입력이 없으면 0)")
ap.add_argument("--path-map", default="", help="segment 경로 접두어 치환 'old=new[,old2=new2]' (words.jsonl 이 이미 rack4 경로면 불필요)")
ap.add_argument("--allow-unlabeled", action="store_true", help="labels 없는 스트림도 학습(후보 없음 = SEM 없음) — 기본은 건너뜀")
ap.add_argument("--max-items-en", type=int, default=0); ap.add_argument("--max-items-ko", type=int, default=0)
ap.add_argument("--de-words", action="append", default=[], help="독일어 ASR words.jsonl(반복·쉼표 목록·@목록파일) — 라벨 없음(SEM 중립), 독일어 셋 'de' 만 읽는다")
ap.add_argument("--max-items-de", type=int, default=0)
ap.add_argument("--ast-targets", action="append", default=[], help="ast-v1(2026-10-09): AST 번역 타깃(experiments/ast_build_targets.py 산출 폴더·파일; 반복·쉼표·@목록) — "
                "prefix 'language <SPK_x>Lang[\\ntranslate <SPK_x>Tgt]<asr_text>', 전사·번역 함께 출력(<TRANS>), vapasr/data/ast_dataset")
ap.add_argument("--ast-frac", type=float, default=0.5, help="번역 타깃이 있는 항목을 이 확률로 번역 모드로(나머지는 화자·언어 prefix 의 ASR)")
ap.add_argument("--ast-format", action="store_true", help="번역 타깃 없이도 ast-v1 prefix(화자 태그)로 학습 — --ast-targets 가 있으면 자동")
ap.add_argument("--gain-augment", default="off", help="ast-v1 음량 증강 'p,lo_db,hi_db'(예 0.5,-70,-20): --gain-augment-langs 항목을 확률 p 로 파형 RMS [lo, hi] dBFS 에 맞춘다(off=끔)")
ap.add_argument("--gain-augment-langs", default="English", help="음량 증강을 걸 언어(쉼표 목록: English,Korean,German)")
ap.add_argument("--out-dir", required=True); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--epochs", type=float, default=1.0)
ap.add_argument("--bs-en", type=int, default=6); ap.add_argument("--bs-ko", type=int, default=16)
ap.add_argument("--lr", type=float, default=6e-5); ap.add_argument("--lr-adapter", type=float, default=1e-3); ap.add_argument("--lr-encoder", type=float, default=1e-5)
ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--delays", default="2,3,4,6"); ap.add_argument("--M", type=int, default=0, help="청크당 텍스트 한도 — v0 는 0 만(SEM 이 단어와 같은 청크)")
ap.add_argument("--next-weight", type=float, default=0.3); ap.add_argument("--next-weight-ko", type=float, default=0.15)
ap.add_argument("--sem-weight", type=float, default=1.0); ap.add_argument("--turn-weight", type=float, default=1.0); ap.add_argument("--hardneg-weight", type=float, default=1.0)
tg_g = ap.add_mutually_exclusive_group()
tg_g.add_argument("--turn-end", dest="turn_end", action="store_true", help="턴 종료 <EOT> 도 함께 학습(mono 발화 끝 proxy) — 기본 끔: Phase 1 은 <SEM_END> 만")
tg_g.add_argument("--no-turn-end", dest="turn_end", action="store_false", help="(기본) <SEM_END> 만")
ap.set_defaults(turn_end=False)
ap.add_argument("--hangover", type=float, default=0.48, help="--turn-end 일 때 턴 종료 청크 = max(마지막 방출, int((t_last+hangover)/0.08))")
ap.add_argument("--tail-margin", type=int, default=2, help="K' = max(K, 마지막 이벤트 청크(δmax) + margin)")
enc_g = ap.add_mutually_exclusive_group()
enc_g.add_argument("--freeze-encoder", dest="freeze_encoder", action="store_true", help="인코더 동결(기본)"); enc_g.add_argument("--train-encoder", dest="freeze_encoder", action="store_false")
ap.set_defaults(freeze_encoder=True)
ap.add_argument("--encoder-right-context", type=int, default=None, help="Nemotron att_context 우측 프레임: 0=[56,0](80 ms), 1, 3=[56,3](4 프레임 청크, 최대 320 ms), 6, 13. "
                "생략하면 init 설정 그대로. 바꾸면 config.encoder_right_context 에 저장돼 평가·추론도 따른다(2026-10-01 lookahead 실험)")
ap.add_argument("--encoder-context-sampling", choices=["multi", "fixed"], default="multi", help="--train-encoder 일 때 NeMo 인코더는 train 모드에서 step 마다 att_context 를 "
                "att_context_size_all([56,3]·[56,0]·[56,6]·[56,13]) 중 무작위로 고른다(multi, NeMo 기본 — 2026-10-01 이전 모든 인코더 학습 run). "
                "fixed = config 의 [56, encoder_right_context] 하나로만 학습(평가·추론과 같은 context)")
gc_g = ap.add_mutually_exclusive_group()
gc_g.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true", help="gradient checkpointing(기본 켬)"); gc_g.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
ap.set_defaults(grad_ckpt=True)
lg_g = ap.add_mutually_exclusive_group()
lg_g.add_argument("--no-liger", dest="liger", action="store_false", help="Liger 끔"); lg_g.add_argument("--liger", dest="liger", action="store_true", help="(기본) Liger — 패키지가 없으면(rack4) 경고 뒤 끈다")
ap.set_defaults(liger=True)
ap.add_argument("--save-every", type=int, default=500); ap.add_argument("--save-total-limit", type=int, default=0, help="0 = checkpoint 를 지우지 않는다(원격 삭제 금지)")
ap.add_argument("--log-every", type=int, default=10); ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--language-schedule", choices=("balanced", "proportional"), default="balanced")
ap.add_argument("--resume", default="auto", help="auto | none | <checkpoint dir>"); ap.add_argument("--gpu", default=None)
ap.add_argument("--force-resume", action="store_true", help="체크포인트의 config.semcommit 지문이 지금 설정·데이터와 달라도 재개(차이는 run 기록에 남는다)")
ap.add_argument("--allow-missing-lang", action="store_true", help="words 행이 있는 언어가 labels 짝이 없어 0 항목이 돼도 다른 언어만으로 학습")
ap.add_argument("--no-ko-textnorm", dest="ko_textnorm", action="store_false", help="KO 타깃을 asr-tn-v1 로 정규화하지 않는다(기본은 정규화: speechlm 행의 구두점 토큰 제거, "
                "digit·허용 밖 문자 행 제외 — vapasr/data/semcommit_tn.py)")
ap.add_argument("--written-frac", type=float, default=0.0, help="표기형 출력 형식 비율(표기형이 있는 항목마다 이 확률로 system 지시문 + 구두점·대소문자·숫자 타깃). "
                "기본 0 = 정규형만(이 기능 전과 동일) — vapasr/data/output_format.py, semcommit_written.py")
ap.add_argument("--written-frac-en", type=float, default=None, help="EN 표기형 비율(없으면 --written-frac)"); ap.add_argument("--written-frac-ko", type=float, default=None, help="KO 표기형 비율(없으면 --written-frac)")
ap.add_argument("--asr-words", action="append", default=[], help="labels 없이 ASR 로만 쓸 words.jsonl(반복·쉼표 목록·@목록파일) — SEM 중립(<SEM_END> 를 그 행 softmax 에서 뺀다), "
                "main/short 풀은 길이(8 s)로 나눈다. short 풀은 --short-train-words 가 있을 때만 쓴다")
ap.add_argument("--asr-max-ratio", type=float, default=None, help="셋마다 SEM 중립 항목 ≤ 이 비율 × 라벨 항목(기본: 제한 없음)")
ap.add_argument("--speed-perturb", default="", help="속도 변형 배율 목록(예 0.9,1.0,1.1; 기본 끔) — 항목마다 하나를 골라 오디오를 리샘플하고 이벤트 시각을 1/배율로")
ap.add_argument("--asr-only", action="store_true", help="Stage 1 ASR 전용: labels 를 무시하고 모든 행을 SEM 중립으로(<SEM_END> 를 학습하지 않는다)")
ap.add_argument("--offline-frac", type=float, default=0.0, help="항목마다 이 확률로 final(오프라인) 모드 — 오디오 전체 → 전사(Qwen3-ASR 원 형식). 같은 모델이 스트리밍과 final 을 함께 배운다")
ap.add_argument("--freeze-thinker", action="store_true", help="thinker 동결(adapter 워밍업; --offline-frac 1 과 함께 — 원 형식은 새 특수 토큰이 필요 없다)")
ap.add_argument("--noise-bank", default=None, help="잡음·잔향 뱅크(experiments/build_noise_bank.py 출력) — 켜면 항목마다 잔향(--rir-p)·가산 잡음(--noise-p, SNR --noise-snr) 증강")
ap.add_argument("--noise-p", type=float, default=0.4); ap.add_argument("--noise-snr", default="5,30", help="잡음 SNR dB 범위 lo,hi(음악은 --music-snr)")
ap.add_argument("--music-snr", default="10,30"); ap.add_argument("--rir-p", type=float, default=0.2)
ap.add_argument("--spec-augment", default="off", help="입력 log-mel SpecAugment(학습 중에만): off | light(주파수 2×27·시간 5×0.05) | nemo(2×27·10×0.05, Nemotron 사전학습 설정) | Fm,Fw,Tm,Tw")
ap.add_argument("--no-check-save", action="store_true", help="final 재로드 parity 검사 끔(기본 켬)"); ap.add_argument("--check-items", type=int, default=64, help="학습 전 셋마다 시퀀스 불변식 검사 항목 수")
ap.add_argument("--report-to", default="tensorboard"); ap.add_argument("--dry-run", action="store_true")
from vapasr.speedup import add_speed_args, apply_speedups, training_args_kwargs
add_speed_args(ap)                                                                            # --pack-max-tokens·--attn-impl·--optim·--deepspeed (vapasr/speedup.py)
a = ap.parse_args()
req_delays = [int(x) for x in a.delays.split(",") if x.strip()]
if not req_delays or not all(1 <= d <= 10 for d in req_delays): sys.exit(f"--delays {a.delays}: 1–10 의 쉼표 목록(<DELAY_1..8>, 9·10 은 지연 확장)")
delay_ext = max(req_delays) > 8                                                               # <DELAY_9/10> 을 붙인다(SEM 뒤)
ast_on = bool(a.ast_targets) or a.ast_format                                                  # ast-v1 출력 형식(<TRANS>, 화자·언어 prefix)
if ast_on and not (a.asr_only and delay_ext): sys.exit("--ast-targets/--ast-format 은 --asr-only 와 δ 9·10(<DELAY_10> 뒤 <TRANS> id)을 함께 쓴다")
if not 0.0 <= a.ast_frac <= 1.0: sys.exit("--ast-frac 은 [0, 1]")
gain = None if a.gain_augment == "off" else tuple(float(x) for x in a.gain_augment.split(","))
gain_langs = tuple(x for x in a.gain_augment_langs.split(",") if x)
if gain and not ast_on: sys.exit("--gain-augment 는 ast-v1(--ast-targets/--ast-format) 데이터셋에서만 동작한다")
if gain and (len(gain) != 3 or not 0.0 <= gain[0] <= 1.0 or not gain[1] <= gain[2] <= 0.0): sys.exit(f"--gain-augment p,lo_db,hi_db: 0≤p≤1, lo≤hi≤0 — {a.gain_augment}")
if gain and not set(gain_langs) <= {"English", "Korean", "German"}: sys.exit(f"--gain-augment-langs: English/Korean/German — {a.gain_augment_langs}")
if a.M != 0: sys.exit(f"--M {a.M}: v0 는 --M 0 만 지원 — M>0 이면 <SEM_END> 가 청크 한도에 세어져 단어와 다른 청크로 밀린다(계약 위반), 평가 패딩도 M=0 가정")
written_frac = {"English": a.written_frac if a.written_frac_en is None else a.written_frac_en, "Korean": a.written_frac if a.written_frac_ko is None else a.written_frac_ko,
                "German": 0.0}                                                                # 독일어는 표기형 원천이 없다(타깃이 이미 Qwen 원출력 — 구두점·대소문자 포함)
if not all(0.0 <= v <= 1.0 for v in written_frac.values()): sys.exit(f"--written-frac* 는 [0, 1] 범위: {written_frac}")
if written_frac["Korean"] > 0 and not a.ko_textnorm: sys.exit("--no-ko-textnorm 과 KO 표기형 비율 > 0 은 함께 쓸 수 없다(정규형 자체가 표기형 토큰이 된다)")
written_on = any(v > 0 for v in written_frac.values())
speeds = tuple(float(x) for x in a.speed_perturb.split(",") if x.strip())
if speeds and not all(0.5 <= x <= 2.0 for x in speeds): sys.exit(f"--speed-perturb 배율은 [0.5, 2]: {speeds}")
from vapasr.features.online import SpecAugment
try: SpecAugment.from_spec(a.spec_augment)
except (ValueError, TypeError) as e: sys.exit(f"--spec-augment {a.spec_augment!r}: off | light | nemo | Fm,Fw,Tm,Tw ({e})")
if a.asr_max_ratio is not None and a.asr_max_ratio < 0: sys.exit("--asr-max-ratio 는 0 이상")
from vapasr.data.noise_aug import from_args as noise_aug_from_args
try: naug = noise_aug_from_args(a.noise_bank, a.noise_p, a.noise_snr, a.rir_p, a.music_snr)
except (ValueError, OSError) as e: sys.exit(f"--noise-bank {a.noise_bank}: {e}")
rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))   # torchrun 이면 DDP
if a.gpu is not None and world == 1: os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def _writable_root():
    """Triton/inductor 캐시·TMPDIR 자리: VAPASR_LOCAL_CACHE → /tmp → out-dir/.cache/<host>(NFS). 일부 노드는 job step 안에서 /tmp 에 쓸 수 없다
    (job 76397: PermissionError '/tmp/sa_tskim', slurmstepd 'TMPDIR [/tmp] is not writeable'). /dev/shm 은 noexec 라 Triton 이 만든 .so 를 못 올린다."""
    import socket
    name = f"vapasr-{os.getuid()}-semcommit-{os.environ.get('SLURM_JOB_ID', 'local')}"
    for base in (os.environ.get("VAPASR_LOCAL_CACHE"), "/tmp", os.path.join(os.path.abspath(a.out_dir), ".cache", socket.gethostname())):
        if not base: continue
        root = os.path.join(base, name)
        try:
            os.makedirs(root, exist_ok=True); probe = os.path.join(root, f".probe-{os.getpid()}"); open(probe, "w").close(); os.remove(probe)
            return root
        except OSError:
            continue
    raise SystemExit("쓸 수 있는 캐시 디렉토리가 없다(VAPASR_LOCAL_CACHE·/tmp·out-dir)")
_cache_root = _writable_root()
for k_, d_ in (("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "inductor")): os.environ.setdefault(k_, os.path.join(_cache_root, f"{d_}-{local}")); os.makedirs(os.environ[k_], exist_ok=True)
if not os.access(os.environ.get("TMPDIR", "/tmp"), os.W_OK):                                # tempfile(Triton 빌드 등)도 쓸 수 있는 곳으로
    os.environ["TMPDIR"] = os.path.join(_cache_root, f"tmp-{local}"); os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    import tempfile; tempfile.tempdir = None
import torch, torch.distributed as dist
gloo_pg = None
if world > 1 and not a.dry_run:                                                               # s3_train_hf.py 와 같은 초기화: NCCL 학습 pg + 선점 합의용 gloo pg
    from datetime import timedelta
    os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "3600")
    torch.cuda.set_device(local); dist.init_process_group("nccl", timeout=timedelta(hours=3), device_id=torch.device("cuda", local))
    gloo_pg = dist.new_group(backend="gloo", timeout=timedelta(hours=1))
main = rank == 0
from vapasr.data.semcommit_dataset import SemCommitDataset, add_semcommit_specials, semcommit_fingerprint, checkpoint_fingerprint_diff
from vapasr.data.semcommit_tokens import SEM_SPECIALS, TURN_TOKEN
from vapasr.data.semcommit_written import output_format_settings
from vapasr.data.output_format import output_formats_config, WRITTEN
def log(*s):
    if main: print(*s, flush=True)
torch.manual_seed(a.seed); random.seed(a.seed); torch.backends.cuda.matmul.allow_tf32 = True
out = a.out_dir; os.makedirs(out, exist_ok=True)
if not a.dry_run and world == 1 and torch.cuda.is_available() and torch.cuda.device_count() > 1: sys.exit("GPU 가 여러 장 보인다 — --gpu N 으로 한 장만 고르거나 torchrun 으로 DDP(Trainer 가 DataParallel 로 감싸는 것 방지)")
IN_KEYS = ("wav", "wav_len", "K", "ids", "is_audio", "chunk_of", "labels", "mask", "pos_weight", "position_ids")   # position_ids: packing
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
    ret = model.add_semantic_tokens(tok, turn=a.turn_end)                                     # Phase1+Phase2+SEM 을 tokenizer 에 붙이고 새 행 초기화·config 갱신(모델 모듈)
    cfg = model.config; cfg.sem_weight, cfg.turn_weight = a.sem_weight, a.turn_weight; cfg.hardneg_weight = a.hardneg_weight
    cfg.next_weight, cfg.next_weight_ko, cfg.delays = a.next_weight, a.next_weight_ko, [int(x) for x in a.delays.split(",")]
    bad = {k: (v, tok.convert_tokens_to_ids(k)) for k, v in {**old_sp, **cfg.sp_ids}.items() if tok.convert_tokens_to_ids(k) != v}
    assert not bad, f"tokenizer 특수 토큰 id 가 config 와 다름: {bad}"
    if delay_ext:                                                                             # <DELAY_9/10>: 새 행 = <DELAY_8> 복사, config.sp_ids·blocked_ids 갱신
        from vapasr.data.semcommit_tokens import add_delay_tokens
        old_d = {k: v for k, v in model.config.sp_ids.items() if k in ("<DELAY_9>", "<DELAY_10>")}
        log(f"지연 확장 토큰 {add_delay_tokens(model, tok)} (이미 있던 것 {old_d or '없음'})")
    if ast_on:                                                                                # <TRANS> 새 행(평균+잡음), <SPK_A>/<SPK_B>/<TRANS> decode 차단 해제, runaway_cap 상향
        from vapasr.data.semcommit_tokens import add_ast_tokens
        log(f"AST 형식 토큰 {add_ast_tokens(model, tok)} · 차단 해제 <SPK_A>/<SPK_B>/<TRANS> · runaway_cap {model.config.runaway_cap}")
sp_ids = add_semcommit_specials(tok)                                                          # 이미 있으면 그대로(모델이 붙인 것과 같은 순서) — 순서·동결 id 대조
if delay_ext:                                                                                 # 데이터셋 prefix 가 <DELAY_9/10> id 를 쓴다(dry-run 은 tokenizer 에만 붙인다)
    from vapasr.data.semcommit_tokens import add_delay_ext_specials
    sp_ids = {**sp_ids, **add_delay_ext_specials(tok)}
trans_id = None
if ast_on:                                                                                    # 데이터셋에는 sp_ids 밖으로 준다(check_sem_ids 는 SEM 뒤 id 를 지연 확장만 허용)
    from vapasr.data.semcommit_tokens import add_trans_special
    trans_id = add_trans_special(tok)["<TRANS>"]
EVENTS = SEM_SPECIALS + ([TURN_TOKEN] if a.turn_end else [])                                # 학습하는 이벤트 토큰 = config.sem_registry
log(f"tokenizer ← {src} · <SEM_END>={sp_ids['<SEM_END>']} · 턴 종료 {f'{TURN_TOKEN}={sp_ids[TURN_TOKEN]}' if a.turn_end else '학습 안 함'} · {time.time() - t0:.0f}s")
if model is not None:
    assert sp_ids["<SEM_END>"] < model.get_input_embeddings().weight.shape[0], "임베딩 행이 SEM id 보다 적다"
    reg = dict(getattr(model.config, "sem_registry", None) or {})                             # forward 가 SEM(·턴) 가중·통계에 쓰는 id — 없으면 sem/turn_weight 가 조용히 무시된다
    assert reg == {k: sp_ids[k] for k in EVENTS} and all(model.config.sp_ids.get(k) == sp_ids[k] for k in EVENTS), f"config.sem_registry/sp_ids 불일치: {reg} (add_semantic_tokens → {ret})"
    assert (sp_ids[TURN_TOKEN] in model.config.blocked_ids) != a.turn_end, f"{TURN_TOKEN} decode 차단 상태가 --turn-end 와 맞지 않음"
    # 인코더: 동결이면 HF init 의 encoder_trainable=True 도 실제로 끈다 + 체크포인트에 함께 저장(D1 교훈, p2_train_hf.py 와 같은 처리)
    init_had_encoder = bool(cfg.encoder_trainable or getattr(cfg, "encoder_saved", False))
    if a.freeze_encoder:
        model.encoder.set_trainable(False)
        for p_ in model.encoder.parameters(): p_.requires_grad_(False)
        cfg.encoder_trainable = False; cfg.encoder_saved = True
    else:
        model.encoder.set_trainable(True); cfg.encoder_trainable = True
    if a.encoder_right_context is not None and a.encoder_right_context != cfg.encoder_right_context:
        ctx = [int(cfg.encoder_left_context), int(a.encoder_right_context)]; allowed = [list(map(int, c)) for c in (getattr(model.encoder.enc, "att_context_size_all", None) or [])]
        assert not allowed or ctx in allowed, f"--encoder-right-context {a.encoder_right_context}: Nemotron 이 학습한 att_context 가 아니다 {allowed}"
        log(f"인코더 att_context [{cfg.encoder_left_context},{cfg.encoder_right_context}] → {ctx} (config 에 저장 — 평가·추론도 이 값)")
        model.encoder.enc.set_default_att_context_size(ctx); cfg.encoder_right_context = ctx[1]
    if not a.freeze_encoder and a.encoder_context_sampling == "fixed":                    # NeMo ConformerEncoder.forward_internal: training 이고 후보가 2 개 이상이면 random.choices
        ctx = [int(cfg.encoder_left_context), int(cfg.encoder_right_context)]; ee = model.encoder.enc
        ee.att_context_size_all, ee.att_context_probs = [ctx], [1.0]; ee.set_default_att_context_size(ctx)
        log(f"인코더 학습 att_context 고정 {ctx} (NeMo multi-lookahead 무작위 선택 끔)")
    elif not a.freeze_encoder:
        log(f"인코더 학습 att_context: NeMo 무작위 {getattr(model.encoder.enc, 'att_context_size_all', None)} 확률 {getattr(model.encoder.enc, 'att_context_probs', None)} (평가는 [{cfg.encoder_left_context},{cfg.encoder_right_context}])")
    model._keys_to_ignore_on_save = None                                                      # 인코더 키를 저장에서 빼지 않는다(save_pretrained 가 encoder_saved 로 판단)
    sa = model.encoder.set_spec_augment(a.spec_augment)                                        # 입력 특징 증강(학습 중에만; 체크포인트·추론에는 흔적 없음)
    if sa is not None: log(f"SpecAugment {sa.extra_repr()} (인코더 {'동결' if a.freeze_encoder else '학습'} — 동결이어도 adapter·thinker 에는 증강)")
    n_enc_tr = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad); assert not a.freeze_encoder or n_enc_tr == 0
    if a.freeze_thinker:
        for p_ in model.thinker.parameters(): p_.requires_grad_(False)
        log("thinker 동결(adapter 워밍업) — 학습 파라미터는 adapter(+인코더 학습 시 인코더)")
    if a.grad_ckpt: model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.thinker.config.use_cache = False
    speed = apply_speedups(model, a, log, liger=a.liger); a.liger = bool(speed["liger"])         # attention 구현 + Liger(없으면 끔)
    cfg.use_liger = a.liger
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"model ← {src} ({time.time()-t0:.0f}s) · 학습 {n_tr/1e6:.1f}M · 인코더 {'동결' if a.freeze_encoder else '학습'}(init 인코더 저장됨={init_had_encoder}, 이번 저장={cfg.encoder_saved or cfg.encoder_trainable}) · "
        f"sem_w {cfg.sem_weight} turn_w {cfg.turn_weight} hardneg_w {a.hardneg_weight} · grad ckpt {a.grad_ckpt} · liger {a.liger}")

# ── 데이터
def _paths(vals):
    """'@list' = one path per line (semcommit_collect_done.py snapshot lists — thousands of shards exceed argv limits), else comma list."""
    return [q for x in vals for q in ([l.strip() for l in open(x[1:]) if l.strip()] if x.startswith("@") else x.split(",")) if q]
W = _paths(a.train_words); L = _paths(a.train_labels)
assert a.asr_only or len(W) == len(L), f"--train-words {len(W)} 개 ≠ --train-labels {len(L)} 개"
if a.asr_only: L = []                                                                         # ASR 전용: 라벨을 읽지 않는다
if not 0.0 <= a.offline_frac <= 1.0: sys.exit("--offline-frac 은 [0, 1]")
if a.freeze_thinker and a.offline_frac < 1.0: log("!! --freeze-thinker 인데 스트리밍 항목이 있다 — 새 특수 토큰(<NEXT_AUDIO>·<DELAY_d>) 임베딩이 학습되지 않는다")
SW = _paths(a.short_train_words); SL = _paths(a.short_train_labels)
AW = _paths(a.asr_words)                                                                       # SEM 중립 ASR words(라벨 없음)
DW = _paths(a.de_words)                                                                        # 독일어 ASR words(라벨 없음, 독일어 셋만 읽는다)
if AW and a.allow_unlabeled: sys.exit("--allow-unlabeled(라벨 없음 = SEM 음성)와 --asr-words(라벨 없음 = SEM 중립)는 함께 쓸 수 없다")
assert a.asr_only or len(SW) == len(SL), f"--short-train-words {len(SW)} 개 ≠ --short-train-labels {len(SL)} 개"
if a.asr_only: SL = []
if not 0 <= a.short_step_fraction < 0.5: sys.exit("--short-step-fraction 은 [0, 0.5) 범위")
if SW and a.short_step_fraction == 0: sys.exit("short 입력이 있는데 --short-step-fraction=0 이면 학습되지 않음")
short_fraction = a.short_step_fraction if SW else 0.0
pmap = dict(x.split("=", 1) for x in a.path_map.split(",") if x)
delays = tuple(int(x) for x in a.delays.split(",")); train_sets = {}
DSCls, ast_fp = SemCommitDataset, None
if ast_on:                                                                                    # ast-v1: 화자·언어 prefix + 번역 흐름(vapasr/data/ast_dataset)
    from functools import partial
    from vapasr.data.ast_dataset import AstStreamDataset, load_ast_targets
    AST, ast_st, ast_fp = load_ast_targets(_paths(a.ast_targets)) if a.ast_targets else ({}, {}, None)
    log(f"AST 번역 타깃 {dict(ast_st)} · 번역 모드 확률 {a.ast_frac} · <TRANS>={trans_id}" + (f" · 음량 증강 {gain} {list(gain_langs)}" if gain else ""))
    DSCls = partial(AstStreamDataset, ast_targets=AST, ast_frac=a.ast_frac, trans_id=trans_id, gain_aug=gain, gain_aug_langs=gain_langs)
for key, lang, cap, ww, ll, pool in (("en", "English", a.max_items_en, W, L, "main"), ("ko", "Korean", a.max_items_ko, W, L, "main"),
                                      ("short_en", "English", 0, SW, SL, "speechlm-short"), ("short_ko", "Korean", 0, SW, SL, "speechlm-short"),
                                      ("de", "German", a.max_items_de, DW, [], "main")):
    if not ww: continue
    de = key == "de"                                                                          # 독일어: 라벨 없음 → 늘 SEM 중립, AW(EN/KO)는 읽지 않는다
    if not de: ww = ww + AW                                                                   # 라벨 없는 ASR 행 — 풀은 데이터셋이 길이로 나눈다(short 풀은 SW 가 있을 때만 만든다)
    ds = DSCls(ww, ll, tok, sp_ids, delays=delays, turn_end=a.turn_end, hangover_s=a.hangover, hardneg_weight=a.hardneg_weight, max_items=cap or None,
                          seed=a.seed, online=True, allow_unlabeled=a.allow_unlabeled if pool == "main" else False,
                          max_per_chunk=a.M, langs=[lang], path_map=pmap, tail_margin=a.tail_margin, source_pool=pool,
                          ko_textnorm=a.ko_textnorm, written_frac=written_frac[lang], sem_free_unlabeled=bool(AW) or a.asr_only or de,
                          sem_free_max_ratio=None if (a.asr_only or de) else a.asr_max_ratio,       # 독일어는 라벨 항목이 0 이라 비율 상한을 걸면 모두 버려진다
                          offline_frac=a.offline_frac, ignore_labels=a.asr_only,
                          speed_perturb=speeds, noise_aug=naug)
    nw_, nl_ = ds.stats["words_rows"], ds.stats["labeled"]
    log(f"  {key}: words 행 {nw_} · labels 짝 {nl_} ({nl_ / max(1, nw_):.1%}) · {len(ds)} 스트림 · stats {dict(ds.stats)} · 제외 {dict(ds.bad)}")
    if ds.stats["labels_without_words"]: log(f"  !! {key}: words 행과 짝이 없는 label id {ds.stats['labels_without_words']} 개 — labels 가 다른 셋·표본에서 만들어졌을 수 있다")
    if written_frac[lang] > 0:                                                                 # 표기형 가용량·예상 비율·없음 사유
        wa, hrs = ds.stats["written_available"], sum(it["duration0"] for it in ds.items) / 3600
        why = {k[len("written_"):]: v for k, v in sorted(ds.stats.items()) if k.startswith("written_")
               and k not in ("written_available", "written_hours", "written_tokens", "written_only_skipped") and not k.startswith("written_cand_dropped_")}
        drop = {k[len("written_cand_dropped_"):]: v for k, v in sorted(ds.stats.items()) if k.startswith("written_cand_dropped_")}
        log(f"  {key} 표기형: 가용 {wa}/{len(ds)} 항목 ({wa / max(1, len(ds)):.1%}) · {ds.stats['written_hours']:.1f}/{hrs:.1f} h · 비율 {written_frac[lang]} → 예상 표기형 "
            f"항목 {written_frac[lang] * wa / max(1, len(ds)):.1%}, 시간 {written_frac[lang] * ds.stats['written_hours'] / max(1e-9, hrs):.1%} · "
            f"없음 항목 {why}(합 {sum(why.values())} = {len(ds)} - {wa}) · 버린 후보 {drop} · 정규형 실패로 버린 표기형 원천 행(절단 전) {ds.stats['written_only_skipped']}")
    if not len(ds):
        if nw_ and not a.allow_missing_lang:
            sys.exit(f"{lang}: words 행 {nw_} 개가 있는데 학습 항목 0 개(labels 짝 {nl_}, 제외 {dict(ds.bad)}) — --train-words/--train-labels 짝을 확인하거나 --allow-missing-lang")
        continue
    ds.lang = lang; train_sets[key] = ds
assert train_sets, "학습 스트림 없음"
main_ids = {it["id"] for key, ds in train_sets.items() if not key.startswith("short_") for it in ds.items}
short_ids = {it["id"] for key, ds in train_sets.items() if key.startswith("short_") for it in ds.items}
if main_ids & short_ids: sys.exit(f"main/short 중복 스트림 ID: {list(sorted(main_ids & short_ids))[:3]}")
if SW and not short_ids: sys.exit("short 입력을 지정했지만 승인·라벨 통과한 short 스트림이 없음")
check = {}
for key, ds in train_sets.items():                                                            # 시퀀스 불변식(오디오 없이): SemCommitDataset.target_problems
    n = min(len(ds), a.check_items); bad = [(ds.items[i]["id"], d if f == "normalized" else f"{d}/{f}", pr[:3]) for i in range(n) for d in ds.delays for f in ds.formats(i)
                                            for pr in [ds.target_problems(i, d) if f == "normalized" else ds.target_problems(i, d, f)] if pr]   # 형식마다(표기형은 있을 때만)
    check[key] = dict(n=n, bad=len(bad), examples=bad[:5], decision_on_event=ds.stats["decision_on_event"])
    log(f"  check {key}: {n} 항목 × δ{list(ds.delays)} → 불일치 {len(bad)} {bad[:3]} · 결정 위치 = TURN 타깃(유지) {ds.stats['decision_on_event']} (항목 {ds.stats['decision_on_event_items']})")
assert all(v["bad"] == 0 for v in check.values()), f"시퀀스 불변식 위반: {check}"
fp = semcommit_fingerprint(W + SW + AW + DW, L + SL, tok, turn_end=a.turn_end, turn_token=TURN_TOKEN if a.turn_end else None, hangover_s=a.hangover, tail_margin=a.tail_margin, pad_tail=True, M=a.M, delays=list(delays),
                           sem_weight=a.sem_weight, turn_weight=a.turn_weight, hardneg_weight=a.hardneg_weight, next_weight=a.next_weight, next_weight_ko=a.next_weight_ko,
                           allow_unlabeled=a.allow_unlabeled, max_items_en=a.max_items_en, max_items_ko=a.max_items_ko, seed=a.seed,
                           short_words_n=len(SW), short_step_fraction=short_fraction, ko_textnorm="asr-tn-v1" if a.ko_textnorm else None, sem_ids={k: sp_ids[k] for k in EVENTS},
                           **output_format_settings(tok, written_frac),                        # 비율이 모두 0 이면 키 없음 → 기존 run 지문과 같다(재개 가능)
                           **({"asr_words_n": len(AW), "asr_max_ratio": a.asr_max_ratio} if AW else {}), **({"speed_perturb": list(speeds)} if speeds else {}),
                           **({"spec_augment": a.spec_augment} if a.spec_augment != "off" else {}), **({"noise_aug": naug.settings()} if naug else {}),
                           **({"de_words_n": len(DW), "max_items_de": a.max_items_de} if DW else {}),   # 독일어도 켤 때만 키
                           **({"asr_only": True} if a.asr_only else {}), **({"offline_frac": a.offline_frac} if a.offline_frac > 0 else {}),
                           **({"freeze_thinker": True} if a.freeze_thinker else {}), **({"encoder_right_context": a.encoder_right_context} if a.encoder_right_context else {}),
                           **({"encoder_context_sampling": "fixed"} if (not a.freeze_encoder and a.encoder_context_sampling == "fixed") else {}),
                           **({"ast_format": "ast-v1", "ast_frac": a.ast_frac, "ast_targets_sha256": ast_fp} if ast_on else {}),   # 새 옵션도 켤 때만 키 — 끄면 기존 지문 그대로
                           **({"gain_augment": list(gain), "gain_augment_langs": list(gain_langs)} if gain else {}))
log(f"  지문: words {[x[:12] for x in fp['words_sha256']]} labels {[x[:12] for x in fp['labels_sha256']]} vocab {(fp['vocab_sha256'] or '')[:12]}")
if a.dry_run:
    ds = next(iter(train_sets.values())); s = ds.sequence(0, max(ds.delays)); it = ds.items[0]; P = len(ds.prefix(it["lang"], max(ds.delays)))
    names = {v: k for k, v in sp_ids.items()}; audio = sum(1 for i in s["is_input"][P:] if i)
    def show(p):
        t = s["ids"][p]; lab = s["labels"][p]; w = s["pos_weight"][p]
        return ("A" if s["chunk_of"][p] >= 0 else names.get(t) or repr(tok.decode([t]))) + ("" if lab != -100 or s["is_input"][p] else "[mask]") + (f"[w={w}]" if w else "")
    log(f"dry-run 샘플 {it['id']} K {it['K0']}→{it['K']} δ{max(ds.delays)} · SEM {s['n_sem']} TURN {s['n_turn']}@{s['turn_chunk']} B {s['n_B']} N {s['n_N']} · 오디오 자리 {audio}")
    log(" ".join(show(p) for p in range(P, len(s["ids"]))))
    for key_w, ds_w in train_sets.items():                                                     # 표기형 표본 한 줄(지시문 머리 포함)
        iw = next((i for i in range(len(ds_w)) if "written" in ds_w.items[i]), None)
        if iw is None: continue
        s = ds_w.sequence(iw, max(ds_w.delays), WRITTEN); it = ds_w.items[iw]; Pw = len(ds_w.prefix(it["lang"], max(ds_w.delays), WRITTEN))
        log(f"dry-run 표기형 샘플 [{key_w}] {it['id']} δ{max(ds_w.delays)} · 머리 {tok.decode(s['ids'][:Pw])!r} · SEM {s['n_sem']} B {s['n_B']} N {s['n_N']}")
        log(" ".join(show(p) for p in range(Pw, len(s["ids"]))))
    json.dump(dict(args=vars(a), sp_ids=sp_ids, semcommit=fp, sets={k: dict(v.stats) for k, v in train_sets.items()}, bad={k: dict(v.bad) for k, v in train_sets.items()}, check=check),
              open(os.path.join(out, "dry-run.json"), "w"), indent=1, ensure_ascii=False)
    log("dry-run 끝 →", os.path.join(out, "dry-run.json")); sys.exit(0)

# ── Trainer
from transformers import TrainingArguments
from vapasr.hf.trainer import VapAsrTrainer, PreemptCallback
from vapasr.hf.stage import last_complete_checkpoint                                         # 저장 완료 checkpoint-N(한 파일·1.7B 조각 저장 모두 — 선점이 끊은 것은 건너뛴다)
from vapasr.data.semcommit_dataset import semcommit_round_robin

class SemTrainer(VapAsrTrainer):
    """데이터만 바꾼다: 언어 = 셋의 lang(이름이 manifest 가 아님), collate_semcommit. 손실·로그는 VapAsrTrainer 그대로(pos_weight 전달, loss_sem/loss_turn/top1_sem/sem_fp 키별 평균)."""
    def bs_of(self, name): return self.bs_ko if self.train_sets[name].lang == "Korean" else self.bs_en
    def get_train_dataloader(self):
        from vapasr.hf.packing import budget_sampler_factory
        return semcommit_round_robin(self.train_sets, self.bs_of, seed=self.args.seed, num_workers=self.num_workers, rank=self.args.process_index, world=self.args.world_size,
                                     schedule=self.language_schedule, short_fraction=short_fraction,
                                     sampler_factory=budget_sampler_factory(self.pack_max_tokens, self.batch_max_tokens, self.pack_max_bs))
    def evaluate(self, *args, **kw): return {}                                                # v0: 자유 디코드 SEM/TURN 평가는 별도 스크립트
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if written_on:                                                                        # 표기형 항목 수 / 모든 배치 항목 수(비율 0 인 언어의 배치도 분모에 — n_written 없음 = 0)
            lang = list(inputs.get("lang") or ()); c = self.__dict__.setdefault("_written", {}).setdefault(lang[0] if lang else "?", [0, 0])   # 배치 = 한 셋 = 한 언어
            c[0] += int(inputs.get("n_written", 0)); c[1] += len(lang)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
    def log(self, logs, start_time=None):
        w = self.__dict__.get("_written")
        if w and "loss" in logs:                                                              # rank 0 의 배치만 센다(VapAsrTrainer 의 다른 분해 지표 loss_sem 등과 같다)
            n = sum(c[1] for c in w.values())
            if n: logs["written_share"] = round(sum(c[0] for c in w.values()) / n, 3)         # 전체 항목 중 표기형
            for lg, c in sorted(w.items()):                                                   # 표기형을 켠 언어별(그 언어 항목 중 표기형)
                short = {"English": "en", "Korean": "ko", "German": "de"}.get(lg, lg)
                if c[1] and written_frac.get(lg, 0) > 0: logs[f"written_share_{short}"] = round(c[0] / c[1], 3)
            self._written = {}
        return super().log(logs, start_time)

targs = TrainingArguments(output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=1, num_train_epochs=a.epochs, max_steps=a.max_steps if a.max_steps > 0 else -1,
                          learning_rate=a.lr, weight_decay=a.wd, warmup_steps=a.warmup, lr_scheduler_type="cosine", max_grad_norm=1.0, **{"bf16": True, **training_args_kwargs(a)},
                          logging_strategy="steps", logging_steps=a.log_every, logging_first_step=True, eval_strategy="no", save_strategy="steps", save_steps=a.save_every,
                          save_total_limit=a.save_total_limit, save_safetensors=True, seed=a.seed, data_seed=a.seed, report_to=[x for x in a.report_to.split(",") if x and x != "none"],
                          logging_dir=os.path.join(out, "tb"), remove_unused_columns=False, disable_tqdm=True, ignore_data_skip=True, dataloader_num_workers=a.num_workers,
                          label_names=["labels"], log_level="warning" if main else "error", ddp_find_unused_parameters=False, ddp_broadcast_buffers=False, ddp_timeout=10800)
model.config.semcommit = fp                                                                  # 모든 checkpoint-N·final 의 config.json 에 들어간다
if written_on: model.config.output_formats = output_formats_config(tok)                       # 평가·추론이 표기형 머리를 복원(학습한 형식만; 끄면 키 없음)
last = last_complete_checkpoint(out) if a.resume == "auto" else (None if a.resume == "none" else a.resume); fdiff = []
if last:                                                                                      # 재개 = 같은 데이터·설정일 때만(지문 대조)
    fdiff = checkpoint_fingerprint_diff(last, fp)
    if fdiff and not a.force_resume: sys.exit(f"재개 거부: {last} 의 config.semcommit 이 지금 설정·데이터와 다르다 {fdiff[:8]} — 새 --out-dir 을 쓰거나(같은 out-dir 에 새로 시작하면 옛 checkpoint-N 과 섞인다), 차이를 알고 이어가려면 --force-resume")
    log(f"재개 ← {last}" + (f" · !! --force-resume: 지문 차이 {fdiff}" if fdiff else " · 지문 일치"))
preempt_cb = PreemptCallback(out, gloo_pg)
trainer = SemTrainer(model=model, args=targs, train_sets=train_sets, dev_sets={}, tokenizer=tok, bs_en=a.bs_en, bs_ko=a.bs_ko, lr_adapter=a.lr_adapter, lr_encoder=a.lr_encoder,
                     max_per_chunk=a.M, num_workers=a.num_workers, language_schedule=a.language_schedule, gloo_pg=gloo_pg,
                     pack_max_tokens=a.pack_max_tokens, pack_max_bs=a.pack_max_bs, batch_max_tokens=a.batch_max_tokens, callbacks=[preempt_cb], processing_class=tok)
tdl = trainer.get_train_dataloader()
log("배치/epoch: " + ", ".join(f"{m}:{len(s)}" for m, s in tdl.samplers.items()) + f" → steps/epoch {len(tdl)} · world {world} · "
    + (f"토큰 예산 배치 " + ", ".join(f"{m}:{s.describe()}" for m, s in tdl.samplers.items()) if (a.pack_max_tokens or a.batch_max_tokens) else f"배치 EN {a.bs_en} / KO {a.bs_ko}"))
run_rec = dict(args=vars(a), init=src, sp_ids=sp_ids, semcommit=fp, resume_from=last, resume_fingerprint_diff=fdiff, trainable_m=n_tr / 1e6, encoder_saved=bool(model.config.encoder_saved),
               train_streams={k: len(v) for k, v in train_sets.items()}, sets={k: dict(v.stats) for k, v in train_sets.items()}, excluded={k: dict(v.bad) for k, v in train_sets.items()},
               check=check, steps_per_epoch=len(tdl), world=world, speed=speed, started=time.strftime("%F %T"))
run_path = os.path.join(out, time.strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid()}.json")          # 시작마다 새 파일(첫 설정을 덮어쓰지 않는다)
if main: json.dump(run_rec, open(run_path, "w"), indent=1, ensure_ascii=False); log(f"실행 기록 → {run_path}")
res = trainer.train(resume_from_checkpoint=last); st = trainer.state
done = st.global_step >= st.max_steps and not preempt_cb.fired
log(f"train 종료: step {st.global_step}/{st.max_steps} {'완료' if done else '(선점/중단 → 재시작 시 이어서)'} · {res.metrics}")
if done:
    fin = os.path.join(out, "final"); trainer.save_model(fin)                                  # 모든 rank 가 부른다(Trainer 가 rank 0 만 쓴다)
if done and main:
    tok.save_pretrained(fin)
    json.dump(run_rec, open(os.path.join(fin, "semcommit_run.json"), "w"), indent=1, ensure_ascii=False)          # 평가가 학습 설정을 찾을 수 있게(config.json 의 semcommit 과 함께)
    json.dump(dict(args=vars(a), log_history=st.log_history[-300:], steps=st.global_step), open(os.path.join(out, "results.json"), "w"), indent=1, ensure_ascii=False)
    if not a.no_check_save:                                                                   # 재로드 parity: state_dict(인코더 포함)·한 배치 손실
        def batch_loss(m, x):
            m.eval(); xx = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in x.items()}
            nw = m.config.next_weight_ko if x["lang"][0] == "Korean" else m.config.next_weight
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): o = m(**{k: xx[k] for k in IN_KEYS if k in xx}, next_weight=nw)
            return {k: round(float(getattr(o, k)), 5) for k in ("loss",) + LOSS_KEYS if getattr(o, k, None) is not None}
        xb = next(iter(trainer.get_train_dataloader())); mem = getattr(trainer.model, "module", trainer.model)
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
if world > 1 and not a.dry_run: dist.destroy_process_group()

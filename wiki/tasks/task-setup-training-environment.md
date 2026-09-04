---
type: task
status: done
owner: tskim
due: 2026-09-12
priority: p0
created: 2026-09-03
updated: 2026-09-03
summary: tskim_env 컨테이너에 torch·NeMo·transformers 등 학습 스택 구축 및 GPU 동작 확인
sources:
  - [[decision-compute-environment]]
---

# 컨테이너 학습 환경 구축

## 배경

→ [[decision-compute-environment]]

컨테이너에 Python 3.13.11 만 있고 **torch 가 없다.** 학습 스택 없이는 Phase 0 의
causality 감사조차 돌릴 수 없다. 두 backbone 이 서로 다른 생태계(NeMo / Transformers)를
쓰므로 둘 다 필요하다.

## 완료 조건

- [x] CUDA 드라이버 버전 확인 (`nvidia-smi`) → 호환 torch 버전 결정 — 550.54.15 / CUDA 12.4 → cu124
- [x] Python 3.13 은 NeMo 호환이 불확실 — conda env `vapasr` (3.11) 생성
- [x] torch + torchaudio 설치, `torch.cuda.is_available()` 및 4 GPU 인식 확인 — 2.6.0+cu124, A100 4장, matmul OK
- [x] NeMo toolkit 설치 → Nemotron 3.5 ASR streaming 0.6B 로드 및 샘플 추론 — git main 3.1.0, 638M, 79.4 ms/frame, ctx [56,0]/[56,3] 전사 OK. **언어는 manifest `lang` 필드로** (키워드 무효)
- [x] transformers 설치 → Qwen3-ASR-0.6B 로드 및 샘플 추론 — `qwen-asr 0.0.6`, 5 s 로드, 정확 전사. audio_tower = Qwen3ASRAudioEncoder **186M**. ForcedAligner 도 동작
- [x] 원 VAP 구현체 의존성 설치 (CPC encoder 포함) — `/data3/tskim/third_party/{VoiceActivityProjection,vap_turn_taking,datasets_turntaking}` editable 설치. 동봉 체크포인트 `VAP_3mmz3t0u_50Hz_…ckpt` 로드 OK: CPC, 5.8M, 50 Hz, probs (T,256). `load_state_dict(strict=False)` 에서 missing 8 — baseline 재현 시 확인
- [x] python-dotenv 설치, `.env` 로드 확인 — `scripts/activate-env.sh`
- [x] 환경 재현용 `requirements.txt` 또는 `environment.yml` 을 저장소에 커밋 — `env/requirements.txt` + `env/requirements-lock.txt`(174줄) + `env/README.md`
- [x] 모델 가중치 캐시를 `DATA_ROOT` 하위로 지정 (HF_HOME / NEMO_CACHE_DIR) — `/data3/tskim/cache/*`, `.env` 에 기록

## 진행 기록

- 2026-09-03: 생성. 컨테이너 실측 결과 torch 미설치 확인.
- 2026-09-03: **완료.** 스모크 테스트 4종 통과 (Nemotron / Qwen3-ASR / ForcedAligner / 원 VAP).
  `env/requirements-lock.txt` 178줄 (NeMo git ea1ebf5, VAP 계열 editable 포함). 컨테이너 재생성 시
  `scripts/setup-container-env.sh` 재실행 + third_party 재설치 필요 (가중치 캐시는 /data3 에 잔존).
- 2026-09-03: Nemotron 로드 OK(638M, d_model 1024, subsampling 8, 기본 att_context [56,3]) 이나
  `transcribe` 가 `Unknown prompt key: 'None'` 으로 실패. 원인: **NeMo 3.0.0 PyPI 의
  `audio_to_text_lhotse_prompt_index.py` 가 `_setup_transcribe_dataloader` 의 `default_lang` 을 무시**
  하고 `cut.supervisions[0].language`(None) 만 읽음. 모델 카드 권장대로 git main 을 `--no-deps` 로 시험 중.
  `qwen-asr` 가 transformers 5.16.1 → 4.57.6 다운그레이드 (NeMo 와 충돌 없음 확인).
- 2026-09-03: `scripts/setup-container-env.sh` 전 단계 통과 (torch 2.6.0+cu124, NeMo 3.0.0, transformers 5.16.1). 모델 로드 스모크 테스트 진행 중.
- 2026-09-03: 착수. 드라이버 550.54.15/CUDA 12.4 확인 → cu124. base Python 3.13 대신 conda env `vapasr`(3.11) 생성 결정. 캐시를 /data3/tskim/cache 로 지정.

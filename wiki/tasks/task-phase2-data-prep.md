---
type: task
status: open
owner: tskim
due: 2026-09-26
priority: p0
created: 2026-09-15
updated: 2026-09-15
summary: Phase 2 Q0 데이터 준비 — 확보된 DB(71631·134-1·134-2 실외·otoSpeech·AMI·ICSI·NOTSOFAR-1)만으로 대화 복원·mono 혼합·정렬·lazy-free lane 라벨·EOT soft target·manifest·QC
sources:
  - '[[output-phase2-lane-plan]]'
  - '[[output-phase2-data-inventory]]'
  - '[[output-phase2-training-db]]'
  - '[[decision-eot-immediate-soft-label]]'
---

# Phase 2 Q0 데이터 준비 (확보된 DB 한정)

## 배경
정본 [[output-phase2-lane-plan]] §8·§13 의 Q0 단계. 사용자 지시(2026-09-15): "일단 확보된 DB 로만 Phase 2 데이터 준비를 해보자." 대상은 서버에 있고 A등급이 확인된 자료뿐이며, Switchboard·CHiME-6·CANDOR 등 조건부 자료는 이 태스크에서 제외한다. 청소년 실내 조각은 대화 단위 완전본이 없어 전사·활동 감독 전용 후보로만 둔다([[output-phase2-data-inventory]] §6).

## 대상과 산출물

| 자료 | 언어 | 대화/회의 | 시간 | 입력 형태 | 이 태스크의 산출물 |
|---|---|---:|---:|---|---|
| 71631 실내 stereo 원본 | KO | 757 | 196.6 h | 2ch wav + 라벨 JSON | 채널 분리 → 화자별 정렬 → mono mix |
| 134-1 실외 조각 | KO | 1,492 | ≤344 h | 화자별 조각 + 라벨 | 시간축 복원 → 화자별 정렬 → mono mix |
| 134-2 실외 조각(완전) | KO | 523 | 90.1 h | 위와 같음 | 위와 같음 |
| otoSpeech | EN | 420 | 104.9 h | 화자별 wav + SRT | 정렬 → mono mix, actor split |
| AMI | EN | 171 | 98.1 h | Headset-N wav + NXT | 헤드셋 채널 = 화자, 정렬 → mono mix |
| NOTSOFAR-1 | EN | 237 | 24.2 h | close_talk CT_*.wav + gt_transcription(단어 시각) | CT 채널 = 화자, 정렬 검증 → mono mix |
| ICSI | EN | 75 | 71.6 h | chan*.sph + NXT | 채널↔참가자 매핑 → 정렬 → mono mix |

공통 산출물: `conversations.jsonl`(대화 단위: 화자별 채널 경로·오프셋·게인·split·품질 플래그), `segments.jsonl`(화자별 발화 구간·단어 시각·TN 텍스트), lane 라벨(lazy-free allocator, R=6), EOT 후보 위치와 p_end, 청크 단위 직렬화 결과, QC 리포트.

## 순서
1. [x] 기존 데이터 계층 파악 — 기존 streams.jsonl·interleave·targets 는 2화자 고정이라 별도 dialogue 계층 신설(`vapasr/data/dialogue*.py`, `lane_alloc.py`, `eot_soft.py`)
2. [x] 코퍼스별 파서 `vapasr/data/dialogue_corpora.py` 7종 서버 실물 검증 완료(각 2대화 빌드). ICSI 는 SPH(shorten)→FLAC 변환(sph2pipe, `VAPKT-data/data/audio/icsi/`) 진행 중
3. [x] TN 적용 `experiments/p2_build_dialogues.py` 서버 실행 확인(EN 대화 코퍼스는 숫자 규칙 `yodas` 키). 71631 quarantine 20 %(digit·anon) 는 결정 항목
4. [x] 전량 정렬 완료(잡 70988). GPU 잡은 프로세스별 메모리 상한 0.8/워커·워커 6/GPU·환경 로컬 스테이징
5. [x] mono 혼합기·창 Dataset 실물 검증(창 wav 생성)
6. [x] `lane_alloc.py`·`eot_soft.py`·`dialogue_interleave.py`(serialize/flatten/활동)·`dialogue_tokens.py` + `tests/test_lane_protocol.py` 19개 통과(D0); `dialogue_dataset.py`(창 선택·잘림 EOT mask·soft/활동·collate) + 테스트; `dialogue_stitch.py`(대화 결합 합성, EOT mask) + 테스트; `experiments/p2_qc.py`
7. [ ] 중복·노출 감사(71631 E2 노출, 134-1 원본↔조각 join), split(actor·회의 계열·공식 split)
8. [ ] 경량 QC: 밀도 p99·강제 NEXT·삭제율·lane 부족률·EOT 후보 분포·오디오 spot-check
9. [x] 32창 overfit 용 소형 셋 추출 → 300 step overfit 수렴 확인(D1 준비 절 참조)

## 완료 조건
- [x] 7 개 자료 모두 dialogues.jsonl 생성(잡 70960). 대화 수·시간은 인벤토리와 일치, 단 134-1 실외 완전 대화는 459 로 정정
- [x] 정렬 coverage: 결손 조각 외 미정렬 0–495, 실패 0 (잡 70988)
- [ ] N≤6 세션에서 `never_free` ≡ `lazy_free` 테스트 통과, 16+11 fixture 통과
- [x] EOT 결과 분포(보정본): 교대 27–42 %, 혼재 23–49 %, 유지 20–40 % — 어노테이션 실측과 같은 자릿수
- [x] 32창 overfit 셋 준비·수렴 확인(QC 리포트·manifest 버전 고정은 D1 이후)

## 전량 빌드 결과 (SLURM 70960, 2026-09-15 22:48–23:47, 사용자 제출)

| 코퍼스 | 대화/회의 | 시간 | 발화 | quarantine | 결손 조각 | 세션 화자 수 |
|---|---:|---:|---:|---:|---:|---|
| 71631 실내 stereo | 757 | 196.6 h | 238,531 | 1.2 % (anon 2,549·digit 191) | 0 | 2 |
| 134-1 실외 조각 | 1,492 | 344.2 h | 426,780 | 1.2 % | 46,020 (10.8 %) → 완전 대화 459 (77.1 h), ≥95 % 368 (63.0 h) | 2 |
| 134-2 실외 조각 | 1,649 | 399.9 h | 477,664 | 1.2 % | 60,494 (12.7 %) → 완전 523 (90.1 h), ≥95 % 427 (75.4 h) | 2 |
| otoSpeech | 420 | 104.9 h | 86,893 | 0.6 % (digit) | 0 | 2 |
| AMI | 171 | 100.0 h | 83,478 | 0.0 % | 0 | 3: 5 · 4: 163 · 5: 3 |
| NOTSOFAR-1 | 237 | 24.5 h | 54,650 | 0.0 % | 0 | 3–8 |
| ICSI (FLAC) | 75 | 71.7 h | 102,177 | 4.7 % (empty) | 0 | 3–9 |
| **합계** | 4,801 | **1,241.8 h** | 1,470,173 | | | |

주의: 134-1 실외는 인벤토리에서 "1,492 대화 완전"으로 적었으나 대화 단위로는 끝부분 조각이 빠진 대화가 많아 완전 대화는 459 (77.1 h) 다. 원자료 `raw/sources/experiments/2026-09-15-phase2-full-build/`.

## 전량 ASR 대조·보정·QC 결과 (2026-09-16)

ASR 대조 잡 71381(13:36–15:13, 노드 1개), 보정은 mxc 로그인 노드(`scripts/p2-refine-local.sh` + 재보정 스크립트, 대화 병렬 16).

| 코퍼스 | 발화 | ASR 불일치(CER≥0.2, 전사 감독 제외) | 결손 의심 | 미정렬(결손 조각) | VAD 분할 | 당김 | 발화 중앙값 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 71631 | 238,531 | 89,916 (37.7 %) | 25,182 (10.6 %) | 495 | 31,558 | 73,400 s | 1.73 → 1.34 s |
| 134-1 | 426,780 | 160,431 (37.6 %) | 29,069 (6.8 %) | 45,464 | 65,840 | 81,136 s | 1.82 → 1.50 s |
| 134-2 | 477,664 | **248,007 (51.9 %)** | 22,442 (4.7 %) | 59,827 | 22,458 | 69,284 s | 1.37 → 1.17 s |
| otoSpeech | 86,893 | — | 715 (0.8 %) | 0 | 39,678 | 1,272 s | 1.40 → 1.30 s |
| AMI | 83,478 | — | 3,626 (4.3 %) | 3 | 104,205 | 14,304 s | 1.85 → 1.14 s |
| NOTSOFAR-1 | 54,650 | — | 782 (1.4 %) | 0 | 11,845 | 3,315 s | 1.35 → 1.13 s |
| ICSI | 102,177 | — | 3,100 (3.0 %) | 0 | 41,573 | 11,735 s | 1.51 → 1.05 s |

QC(`p2_qc.py`, R=6, δ=4; 보정본 기준): lane 재배정 NOTSOFAR-1 913(부족 7)·ICSI 700(부족 1), 그 외 0. 청크당 토큰 p99 2–3. 동시 발화 2명 이상(음성 프레임): 71631 10.6 %, 134-1 10.5 %, otoSpeech 3.2 %, AMI 14.5 %, NOTSOFAR-1 29.8 %, ICSI 9.9 %. EOT 결과 분포(교대/혼재/유지 짧음/유지 김/침묵): 71631 27/33/19/13/7 %, 134-1 27/38/22/9/3, otoSpeech 27/23/30/10/9, AMI 37/36/18/7/3, NOTSOFAR-1 42/49/8/0/0, ICSI 29/36/26/7/1. 원자료 `raw/sources/experiments/2026-09-16-phase2-full-qc/`. (QC 의 aligned 계수는 분할 조각을 빼고 세는 버그가 있어 다음 실행에서 수정됨.)

## D1 준비 (2026-09-16)
- 창 통계·overfit32: `experiments/p2_build_windows.py` → 129,381 창 1,083 h(hop 10 s). AI Hub 창 마스크 비율 평균 0.30(창의 17–21 % 가 절반 이상 마스크), EN 코퍼스 ≈0. overfit32 = 코퍼스별 4–6 창, 마스크 0, 회의는 3–4 명. 원자료 `raw/sources/experiments/2026-09-16-phase2-windows/`.
- 모델: registry 동결(`<SPK_3..6>` 151717–151720, `<ONSET>` 151721, `<EOT>` 151722; 코드 상수 `FROZEN_REGISTRY`), `add_phase2_tokens`(임베딩 초기화·활동 헤드·A/B 차단 해제), forward 에 EOT soft CE(가중 2)·활동 BCE(가중 1) — `vapasr/hf/p2_losses.py`(단위 테스트 3). 인코더 동결 기본.
- 학습: `experiments/p2_train_hf.py`(창 라운드로빈·soft/활동 손실·mono 캐시), `slurm/p2_train_d1.sbatch`(GPU 1 장). 스모크(3 step, GPU 1 장): 7 코퍼스 창 라운드트립 불일치 0, loss_text 6.84→5.60, top1 0.59→0.63, loss_eot ≈10(신규 토큰), loss_act 0.75(초기). 32 창 overfit 300 step(GPU 1 장, lr 1e-4·adapter 1e-3, warmup 20, 378 s = 1.26 s/step, 창당 라벨 ≈2.2 k·soft 위치 ≈40) 수렴: loss_text 6.85→0.012(top1 0.59→0.996), loss_eot 9.92→0.27, loss_act 0.76→0.008(act_acc 0.49→0.998), loss_next 0.20→0.02. loss_eot 는 0.26–0.29 에서 평탄 — soft 두 점 목표(p_end 0.8/0.5/0.3)의 엔트로피 하한이므로 0 으로 가지 않는 것이 정상. 산출물 `/soundai/users/tskim/VAPKT-data/runs/p2-overfit32/{final,checkpoint-300,tb}`.
- mono 캐시 사전 생성(2026-09-16 21:18–21:47, mxc 로그인 노드 CPU 16 프로세스, `scripts/p2-mono-local.sh` → `experiments/p2_build_mono.py`): 4,801 대화 133 GiB(float16), 실패 0, 28 분. 학습은 같은 파일(`VAPKT-data/data/phase2/mono/<corpus>/<conv>.npy`)을 mmap 으로 읽는다.
- 동적 길이 배치(`TokenBudgetSampler`, `--max-tokens`): 창을 추정 길이(prefix + 2·청크 + 텍스트 토큰 + 3·발화; 실측 대비 −12 % 이내)순으로 정렬해 (창 수 × 최장 길이) ≤ 예산으로 묶는다. GPU 메모리 실측(`experiments/p2_mem_probe.py`, GPU 1 장, AdamW 상태·grad ckpt·Liger 포함): 최장 창(L≈1,590) 예산 28 k → 실토큰 30.1 k, bs 19, peak 97 GB 할당/106 GB 예약(143.8 GB 의 74 %), 0.80 s/step, 37.5 k tok/s; 32 k → 111/122 GB(85 %). 최단 창(L≈640) 28 k → bs 45, 82/89 GB, 50 k tok/s. → **MAX_TOKENS=28000** 채택(80 % 상한 아래, 추정 오차 여유). 전량 기준 배치 4,063/epoch(창 평균 31.7 개, 채움 0.98) — 고정 bs 8(16,170 step/epoch) 대비 step 당 창 4 배.
- 코퍼스 비중(`--mix`): equal 은 NOTSOFAR 가 epoch 당 22 회 반복·134-2 는 0.37 회라 sqrt(배치 수^0.5 비례) 채택 — epoch 당 통과 71631 1.0·134-1 0.87·134-2 0.68·oto 1.35·AMI 1.6·NOTSOFAR 5.0·ICSI 1.9.
- 로더 실측(캐시 사용): 창당 getitem 134-1 11 ms·AMI 121 ms(긴 회의의 발화 순회), collate(33 창) 0.33 s → 워커 8 이면 step 당 데이터 시간 ≈0.2–0.3 s < GPU 0.75 s.
- D1 실학습 recipe(정본 §9 Q1, 노드 1 × GPU 8 torchrun DDP — 사용자 지시 2026-09-16 "노드당 8 GPU"): `sbatch --partition=apex --export=ALL,RUN=D1,MAX_TOKENS=28000,MIX=sqrt,EPOCHS=4,LR=1e-4,WARMUP=150 slurm/p2_train_d1.sbatch` — rank 당 동적 배치(예산 28 k 토큰 ≈ 창 32 개, 8 rank 유효 배치 ≈ 254 창/step), sqrt 비중, epoch 당 505 step(코퍼스별 배치 // 8 의 합), 4 epoch ≈ 2,020 step, lr 1e-4(유효 배치 8 배라 overfit 에서 검증한 1e-4 사용)·adapter 1e-3, warmup 150, EOT 가중 2·활동 가중 1·NEXT 0.3/0.15, 인코더 동결, 200 step 마다 저장, 선점(USR1/TERM → PREEMPT 파일 → gloo 합의 저장) 시 requeue 재개, 완료 표식 `DONE`. 예상 소요 ≈30–40 분 + 로드.
- DDP 스모크(2026-09-16, GPU 2 장, overfit32 창, 예산 3 k, 8 step): 손실 하강(text 5.95→3.64, eot 8.6→2.8, act_acc 0.48→0.87), rank 0 만 로그, final·DONE 생성. 스모크에서 잡은 결함 2 건 수정 — (1) rank 당 배치 0 인 코퍼스가 라운드로빈을 멈춰 다른 rank 가 all_reduce 에서 행(샘플러 n=배치//world, 제외·검출), (2) compute_loss 가 DDP 래퍼의 config 접근 실패. 산출물 `/soundai/Model/VAPASR/p2-D1`. 이후: lane 상태 디코더(`vapasr/hf/lane_state.py`)·D1 평가.

## 진행 기록
- 2026-09-16 (10): 사용자 지시 노드당 GPU 8 → sbatch 를 torchrun DDP 로 재작성, 학습 스크립트 DDP 대응, 2-GPU 스모크 통과(결함 2 건 수정).
- 2026-09-16 (9): 사용자 요청 — mono 캐시 사전 생성(133 GiB, 28 분)·동적 길이 배치 도입(예산 실측 28 k 토큰 = 메모리 74 %)·코퍼스 sqrt 비중. D1 명령 갱신(3 epoch ≈ 12 k step ≈ 3 h).
- 2026-09-16 (8): 사용자 지시 3 단계(GPU ≤1) 완료 — (1) 창 통계·overfit32 셋, (2) forward 에 EOT soft CE·활동 헤드·registry 동결, (3) overfit32 300 step 수렴 확인 → D1 잡 스크립트 준비(제출은 사용자).
- 2026-09-16 (7): ASR 대조·보정·QC 전량 완료(134-2 QC 만 진행 중). 134-2 청소년은 ASR 불일치 52 % 로 전사 감독 절반이 빠짐 → 편입 유지하되 전사 기여는 제한적, 활동·턴 감독 위주.
- 2026-09-15 (5): 전량 빌드 완료(잡 70960, 59 분, 오류 0).
- 2026-09-16 (6): 전량 정렬 완료(잡 70988, hpc, 00:11–01:18; 선점으로 3회 requeue 됐으나 재개로 이어짐). 노드 1개 GPU 8×워커 6. 정렬 발화: 71631 235,214 · 134-1 376,010(결손 조각 45,464 건너뜀) · 134-2 412,013(59,827) · otoSpeech 86,345(proxy 64, OOM 재시도 22) · AMI 83,429 · NOTSOFAR-1 54,635 · ICSI 97,135(proxy 236). ASR 대조 잡 70991 이 01:20 에 이어 시작. 보정은 SLURM 없이 mxc 로그인 노드에서 `scripts/p2-refine-local.sh` 가 코퍼스별로 자동 시작(otoSpeech·AMI·NOTSOFAR·ICSI 진행 중, AI Hub 3종은 ASR 대조 뒤). `p2_refine.py` 를 대화 병렬(fork Pool)로 바꿈.
- 2026-09-15 (4): 사용자 결정 [[decision-aihub-transcript-policy]] 구현 — ASR 일치(CER<0.2) 전사 감독, 불일치 청크 손실 마스크(창 유지), 배치 정책 정본 §8 반영. 샘플 검증 9 항목 통과.
- 2026-09-15 (3): AI Hub 라벨 품질 실측 → [[output-phase2-aihub-label-quality]]. 채널 VAD 시각 보정(`p2_refine.py`)·결손 의심/ASR 불일치 quarantine·ASR 대조(`p2_asr_check.py`, sbatch) 추가. pseudo-label 대체는 사용자 결정 대기.
- 2026-09-15 (2): 서버 복구 → 7 DB 샘플 2대화씩 빌드·정렬(6)·시퀀스 생성·8 검사 ALL_OK → [[output-phase2-sequence-samples]]. 로더 수정(AMI 빈 segment·NOTSOFAR 태그·앞 공백 토큰화), ICSI sph2pipe 변환 시작.
- 2026-09-15: 생성. 기존 데이터 계층 탐색 → D0 모듈·테스트(8fe16cc), 코퍼스 로더·혼합기·빌더·정렬 잡(b8f59e2), Dataset·SLURM 잡(17cb27f), stitch·QC(4c404ea). 로컬 테스트 29개 통과. 서버 SSH 불가(포트 3206 timeout)로 실물 검증·빌드 대기. 남은 것: 서버 코드 동기화 → 로더 스모크(ICSI Words·mrt 매핑 확인) → `slurm/p2_build_dialogues.sbatch`(CPU) → `slurm/p2_align.sbatch`(GPU, 사용자 제출) → QC → 모델 forward 의 soft CE·활동 헤드(D1).

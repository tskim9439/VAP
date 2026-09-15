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
4. [~] `experiments/p2_align.py` 로그인 노드 GPU 로 샘플 정렬 성공(6 코퍼스). 전량은 `slurm/p2_align.sbatch` 제출 대기
5. [x] mono 혼합기·창 Dataset 실물 검증(창 wav 생성)
6. [x] `lane_alloc.py`·`eot_soft.py`·`dialogue_interleave.py`(serialize/flatten/활동)·`dialogue_tokens.py` + `tests/test_lane_protocol.py` 19개 통과(D0); `dialogue_dataset.py`(창 선택·잘림 EOT mask·soft/활동·collate) + 테스트; `dialogue_stitch.py`(대화 결합 합성, EOT mask) + 테스트; `experiments/p2_qc.py`
7. [ ] 중복·노출 감사(71631 E2 노출, 134-1 원본↔조각 join), split(actor·회의 계열·공식 split)
8. [ ] 경량 QC: 밀도 p99·강제 NEXT·삭제율·lane 부족률·EOT 후보 분포·오디오 spot-check
9. [ ] 32창 overfit 용 소형 셋 추출

## 완료 조건
- [x] 7 개 자료 모두 dialogues.jsonl 생성(잡 70960). 대화 수·시간은 인벤토리와 일치, 단 134-1 실외 완전 대화는 459 로 정정
- [ ] 정렬 coverage(단어 시각 확보 비율) 코퍼스별 보고, 실패 대화는 플래그
- [ ] N≤6 세션에서 `never_free` ≡ `lazy_free` 테스트 통과, 16+11 fixture 통과
- [ ] EOT 후보 결과 분류 분포가 어노테이션 기반 실측(§5 정본)과 같은 자릿수
- [ ] QC 리포트와 manifest 버전 고정, 32창 overfit 셋 준비

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

## 진행 기록
- 2026-09-15 (5): 전량 빌드 완료(잡 70960, 59 분, 오류 0). 다음: 정렬(`p2_align.sbatch`)·ASR 대조(`p2_asr_check.sbatch`) 제출.
- 2026-09-15 (4): 사용자 결정 [[decision-aihub-transcript-policy]] 구현 — ASR 일치(CER<0.2) 전사 감독, 불일치 청크 손실 마스크(창 유지), 배치 정책 정본 §8 반영. 샘플 검증 9 항목 통과.
- 2026-09-15 (3): AI Hub 라벨 품질 실측 → [[output-phase2-aihub-label-quality]]. 채널 VAD 시각 보정(`p2_refine.py`)·결손 의심/ASR 불일치 quarantine·ASR 대조(`p2_asr_check.py`, sbatch) 추가. pseudo-label 대체는 사용자 결정 대기.
- 2026-09-15 (2): 서버 복구 → 7 DB 샘플 2대화씩 빌드·정렬(6)·시퀀스 생성·8 검사 ALL_OK → [[output-phase2-sequence-samples]]. 로더 수정(AMI 빈 segment·NOTSOFAR 태그·앞 공백 토큰화), ICSI sph2pipe 변환 시작.
- 2026-09-15: 생성. 기존 데이터 계층 탐색 → D0 모듈·테스트(8fe16cc), 코퍼스 로더·혼합기·빌더·정렬 잡(b8f59e2), Dataset·SLURM 잡(17cb27f), stitch·QC(4c404ea). 로컬 테스트 29개 통과. 서버 SSH 불가(포트 3206 timeout)로 실물 검증·빌드 대기. 남은 것: 서버 코드 동기화 → 로더 스모크(ICSI Words·mrt 매핑 확인) → `slurm/p2_build_dialogues.sbatch`(CPU) → `slurm/p2_align.sbatch`(GPU, 사용자 제출) → QC → 모델 forward 의 soft CE·활동 헤드(D1).

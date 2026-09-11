---
type: output
status: active
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2 실행 요약 — 정본 계획(output-phase2-streaming-asr-diarization-plan)의 Q0–Q4 를 달력(09-15 → 11-14)에 매핑, mxc 보유 2 화자 데이터 실측(71631 248 h·otoSpeech 104 h·TurnBench dev), 데이터 혼합 비율·학습 비용 초기값, 사용자 결정 4 건
sources:
  - [[output-phase2-streaming-asr-diarization-plan]]
  - [[output-phase1-report]]
  - [[decision-mono-input]]
  - [[source-conversation-corpora]]
  - [[output-vap-target-pipeline]]
  - [[output-stage2-e2-final-eval]]
  - [[turn-taking-evaluation-protocol]]
---

# Phase 2 실행 요약: 일정 · 데이터 보유 현황 · 결정 사항

## 질문
Phase 2 정본 계획 [[output-phase2-streaming-asr-diarization-plan]](구조·계약·관문·평가 규약)을 **언제, 어떤 데이터로, 얼마의 비용으로** 실행하며, 착수 전에 사용자가 무엇을 정해야 하는가. 이 페이지는 정본 계획을 대체하지 않는다. 같은 날 두 세션이 계획을 따로 썼고, 설계·관문은 정본 계획으로 일원화하고 여기에는 그 문서가 "실측 전" 으로 남긴 항목(서버 가용량, 일정, 비용)과 결정 요청만 둔다.

## 요약
- **정본 계획의 Q0–Q4 를 9 주(2026-09-15 → 11-14)에 배치**한다. Q0 데이터 계약 1 주 → Q1 비중첩 2 화자 전사 2 주 → Q2 overlap 2 주 → Q3 turn 헤드·의미 ablation 3 주 → Q4 장문·실시간·보고 1 주.
- **mxc 에 이미 있는 2 화자 데이터**(2026-09-11 실측): AI Hub 71631 stereo 248 h(TS_01.실내_5 757 대화 + VS_02.실외 186 대화), otoSpeech 16 k 104.9 h, TurnBench dev 38 대화. 영어 실제 대화는 otoSpeech 뿐이며 CANDOR 는 없다. Switchboard·CallHome 서버 사본은 발화 단위로 잘려 있어 대화 시간축 복원 가능 여부를 Q0 에서 확인한다.
- **학습 비용은 병목이 아니다.** E2 가 5.6 k h × 3 epoch 을 32 GPU 에서 4.3 h 에 끝냈으므로 Q1/Q2 run 은 각 2–3 h, Q3 는 1 h 미만이다. 병목은 데이터 반입(AI Hub 는 PC 브라우저 경유)·라벨 QC·사람 라벨링(Q3 의미 라벨 파일럿)이다.
- **사용자 결정 4 건**: (1) otoSpeech 라이선스 하에 학습 사용 확정·CANDOR 확보 여부, (2) 71631 추가 반입량, (3) 단일 화자 입력에도 `<SPK_A>` 를 내는 단일 모델로 갈지, (4) turn 출력은 병렬 헤드 기본 + 상태 토큰 ablation 으로 갈지.

## 1. 정본 계획과의 대응

| 정본 단계 | 기간(제안) | 이 문서의 실행 항목 | 진입 관문(정본 §7 요약) |
|---|---|---|---|
| Q0 기준선·데이터 계약 | 09-15 → 09-19 | 71631·otoSpeech 채널별 강제 정렬(`s2_prep.sbatch`, `#chN`), VAD 라벨 mxc 재빌드(rack4 npz 는 반입 불가), mono mixer·serializer·QC pack, 고정 평가 ID | 누락·덮어쓰기·화자 누출 0, 평가 재현 |
| Q1 비중첩 2 화자 전사 | 09-22 → 10-03 | E2 init, 자연 대화 창(EN/KO 20–50 h) + replay 30 %, 태그·activity 헤드 | ASR 가드레일(≤5 % 상대), 비중첩 DER ≤10 %, 귀속 오류 ≤5 % |
| Q2 overlap 전사 | 10-06 → 10-17 | 자연 overlap + 합성 overlap(10–50 %, 음량차 3/6/12 dB), 캐스케이드·clean-channel oracle 대조 | overlap cp 오류 ≥20 % 상대 감소, 화자 소실 ≤5 % |
| Q3 미래 활동·의미 예측 | 10-20 → 11-07 | head-only probe → 저율 joint, audio-only/text-ablation/oracle-history 대조군, 의미 라벨 파일럿(EN/KO 각 500 경계) | 동일 FPR 에서 audio-only 대비 +3 %p 또는 p50 −80 ms, 가드레일 유지 |
| Q4 장문·실시간 통합 | 11-10 → 11-14 | 60–120 s carry, 10–60 분 자유실행, 데모 2 레인 + P(EOT), Phase 2 보고서 | RTF <1, backlog 비발산, ID switch ≤1 회/10 분 |

## 2. 데이터 보유 현황 (mxc, 2026-09-11 실측)

| 자원 | 위치 | 실측 | 역할 | 남은 확인 |
|---|---|---|---|---|
| AI Hub 71631 stereo | `/soundai/DB/raw/aihub/71631_audio` 943 파일 54 GB | TS_01.실내_5 757 대화 196.6 h + VS_02.실외 186 대화 51.7 h = **248 h**; 라벨 JSON 은 TL/VL 전체(11,023 개) | KO 자연 대화: Q1–Q3 학습, VS_02 는 검증(Phase 1 dev 로 이미 사용) | 추가 반입량(결정 2). untouched KO test 는 TL_01 라벨 중 미반입 대화에서 새로 골라 반입 |
| otoSpeech 16 k | `/soundai/DB/raw/otoSpeech16k` 23 GB 2,100 파일 | 420 대화 104.9 h, 화자별 wav + SRT | EN 자연 대화: Q1–Q3 | non-commercial·voice cloning 금지 조항의 학습 사용 범위(결정 1) |
| TurnBench dev | `/soundai/DB/raw/turnbench` 17 GB 41 파일 | 38 대화 7.3 h, gold EOT/INT | **평가 전용**(Q3) | 공식 scorer 설치(rack4 의 클론은 반입 불가 → 재클론) |
| NIKL 일상대화 | `/soundai/DB/raw/nikl` | ≈3,800 h, **발화 단위 mono PCM**(대화 파형 없음), `발화겹침` 발화 quarantine | 단일 화자 replay·합성 대화 원료(같은 대화의 두 화자 발화를 시간 라벨대로 배치하면 준자연 대화가 됨) | 발화 start/end 로 대화 시간축 복원 시 겹침 구간 파형은 없음을 명시 |
| Switchboard / CallHome | `…/EN/TRAIN/OPEN/{switchboard,callhome}` | 발화 단위 wav(`sw0xxxxA/B`), CallHome 19.9 h 8 kHz 2 자 통화 | 시간축 복원이 되면 EN 자연 대화 확장 | CSV 에 원본 시각이 있는지(Q0) |
| LibriSpeech · KsponSpeech | Phase 1 manifest | 1,033 / 1,189 h | 합성 대화 원료(train split 만) | — |
| CANDOR | 없음 | — | — | 확보 여부(결정 1) |

## 3. 초기값 제안 (Q0 실측 후 동결)

**혼합 비율(오디오 초 기준)**

| 구분 | Q1 | Q2 | Q3 |
|---|---|---|---|
| Phase 1 단일 화자 replay(turn 손실 마스크) | 30 % | 30 % | 30 % |
| 자연 2 화자 mono(71631·oto) | 60 % | 35 % | 70 % |
| 합성 2 화자(overlap 제어, turn 손실 마스크) | 10 %(무겹침·짧은 맞물림) | 35 %(10–50 %) | 0 % |

**학습 레시피**: 정본 §7.1(E2 init, Q1 은 encoder 동결로 시작 → 정체 시 상위층부터 해동). Q2 첫 ablation 으로 **encoder 해동(LR 1e-5, E2 레시피) 조건**을 둔다. 근거: 단일 화자로 사전학습된 Nemotron 인코더가 겹친 음성을 두 화자로 표현하려면 인코더 적응이 필요할 가능성이 크고, E2 에서 인코더 해동이 초반 붕괴를 없앤 기록이 있다([[output-stage2-e2-final-eval]]).

**학습 비용**: E2 1.02 s/step(32 GPU, BS 12/24) 기준. Q1 ~1.5 k h × 2 epoch ≈ 1.5 h, Q2 ~2.5 k h × 2 epoch ≈ 2.5 h, Q3 자연 대화 350 h 위주 <1 h. sweep 포함 단계당 1 일 이내. `apex` 파티션 권장(선점 없음).

## 4. 사용자 결정 요청

1. **영어 자연 대화**: otoSpeech 를 학습에 쓰는 것으로 확정할지(라이선스 원문 재확인 포함), CANDOR(CC BY-NC) 확보를 진행할지.
2. **71631 추가 반입**: 현재 248 h. TL_01.실내 라벨 8,306 대화 중 미반입분에서 100–200 h 를 PC 브라우저 경유로 더 가져올지. untouched KO test 용 20–30 대화는 별도로 반입 권장.
3. **단일 모델·항상 태그**: 단일 화자 입력에서도 `<SPK_A>` 를 방출하는 한 모델(데모는 태그 숨김)을 권장. 무태그 모드 토큰을 두면 데이터가 갈린다.
4. **turn 출력 형식**: 오디오 클럭 병렬 헤드(activity·VAP·hazard) 기본, SoulX-Duplug 식 상태 토큰은 Q3 ablation. 정본 계획과 같은 권장.

비판과 대안 제안: [[output-phase2-plan-critique]].

## 불확실성
- 두 세션이 같은 요청으로 계획을 동시에 작성했다. 설계 충돌 지점은 두 곳이며 여기서 정본 쪽을 따랐다: (a) Q1 인코더 동결 시작(정본) vs 처음부터 해동(이 세션 초안) → 정본 + Q2 해동 ablation, (b) 청크당 토큰 상한 8→12 고정(초안) vs 밀도 실측 후 결정(정본) → 정본.
- Switchboard·CallHome 시간축 복원 가능 여부, NIKL 준자연 대화 합성의 타당성은 Q0 에서 확인한다.
- 일정은 데이터 반입·사람 라벨링 대기를 포함하지 않는다.

## 근거
- [[output-phase2-streaming-asr-diarization-plan]] — 정본 계획(구조 §3, 계약 §4, 데이터 §5, 목표 §6, 단계 §7, 평가 §8, 구현 §9)
- [[output-phase1-report]] §7 — 이월 항목
- [[source-conversation-corpora]], `raw/sources/mxc-soundai-DB-survey.md`(2026-09-06) — 코퍼스 규모·채널·라이선스, 서버 보유 목록
- [[output-vap-target-pipeline]] — 라벨 파이프라인·코퍼스 통계
- [[output-stage2-e2-final-eval]] — E2 레시피·비용
- [[decision-mono-input]] — mono 입력·커리큘럼

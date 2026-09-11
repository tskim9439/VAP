---
type: output
status: superseded
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2 v1.2 실행 요약 — 화자별 start/end 필수 생성·블록/라벨 계약·9주 일정·ASR/의미 트랙·실측 예산
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

> **2026-09-11 폐기**: 사용자 결정으로 [[output-phase2-streaming-asr-diarization-plan]] 만 따른다. 이 문서는 기록용이며 계획으로 쓰지 않는다.

## 질문
Phase 2 정본 계획 [[output-phase2-streaming-asr-diarization-plan]](구조·계약·관문·평가 규약)을 **언제, 어떤 데이터로, 얼마의 비용으로** 실행하며, 착수 전에 사용자가 무엇을 정해야 하는가. 이 페이지는 정본 계획을 대체하지 않는다. 같은 날 두 세션이 계획을 따로 썼고, 설계·관문은 정본 계획으로 일원화하고 여기에는 그 문서가 "실측 전" 으로 남긴 항목(서버 가용량, 일정, 비용)과 결정 요청만 둔다.

## 요약
- **정본 계획의 Q0–Q4 를 9 주(2026-09-15 → 11-14)에 배치**한다. Q0 데이터 계약 1 주 → Q1 비중첩 2 화자 전사 2 주 → Q2 overlap 2 주 → Q3 turn 헤드·의미 ablation 3 주 → Q4 장문·실시간·보고 1 주.
- **mxc 에 이미 있는 2 화자 데이터**(2026-09-11 실측): AI Hub 71631 stereo 248 h(TS_01.실내_5 757 대화 + VS_02.실외 186 대화), otoSpeech 16 k 104.9 h, TurnBench dev 38 대화. 영어 실제 대화는 otoSpeech 뿐이며 CANDOR 는 없다. Switchboard·CallHome 서버 사본은 발화 단위로 잘려 있어 대화 시간축 복원 가능 여부를 Q0 에서 확인한다.
- **학습 비용은 Q0/Q1 실측 후 산정한다.** E2의 32 GPU 4.3h는 참고값이며 다화자 token 밀도·긴 문맥·rollout·약라벨 생성 비용을 포함하지 않는다. 기존 Q1/Q2 2–3h·Q3 1h 미만 추정은 확정 예산으로 사용하지 않는다.
- **v1.1 적용**: G1 그룹 전사와 G0 비교, Q1 R-E2/R-low recipe 대조, Q1 말/Q2 초 memory on/off, Q0부터 의미 S 트랙 준비, ASR 강화 A 트랙. 세부·반론은 [[output-phase2-critique-response]]를 따른다.
- **v1.2 사용자 요구 반영**: 무음·단독·겹침의 블록을 구체화하고 화자별 `<start_of_turn>`·`<end_of_turn>`을 전사와 공동 생성한다. head만 두고 상태 토큰을 후속 ablation으로 미루던 범위를 변경한다. [[output-phase2-block-and-turn-label-spec]]의 라벨·상태·손실 계약을 적용한다.
- **남은 선택**: otoSpeech/CANDOR 사용 범위, 71631 추가 반입량, 단일 화자 전사의 태그 표시 정책. 화자별 start/end 생성 자체는 사용자가 요청한 필수 목표다.

## 1. 정본 계획과의 대응

| 정본 단계 | 기간(제안) | 이 문서의 실행 항목 | 진입 관문(정본 §7 요약) |
|---|---|---|---|
| Q0 기준선·데이터 계약 | 09-15 → 09-19 | 71631·otoSpeech 채널별 강제 정렬(`s2_prep.sbatch`, `#chN`), VAD 라벨 mxc 재빌드(rack4 npz 는 반입 불가), mono mixer·serializer·QC pack, 고정 평가 ID | 누락·덮어쓰기·화자 누출 0, 평가 재현 |
| Q1 비중첩 2 화자 전사 | 09-22 → 10-03 | E2 init, 자연 대화 창(EN/KO 20–50 h) + replay 30 %, G1 그룹 태그·activity, recipe·memory 비교 | ASR 가드레일(≤5 % 상대), 비중첩 DER ≤10 %, 귀속 오류 ≤5 % |
| Q2 overlap 전사 | 10-06 → 10-17 | 자연 overlap + 합성 overlap(10–50 %, 음량차 3/6/12 dB), 캐스케이드·clean-channel oracle 대조 | overlap cp 오류 ≥20 % 상대 감소, 화자 소실 ≤5 % |
| Q3 미래 활동·의미 예측 | 10-20 → 11-07 | δ=2 첫 조건/δ=4 비교, VAP → S head 전이·semantic-event·의미 ablation → 필요 시 hazard, 자체 생성 history | 동일 FPR 에서 audio-only 대비 +3 %p 또는 p50 −80 ms, 가드레일 유지 |
| Q4 장문·실시간 통합 | 11-10 → 11-14 | 60–120 s carry, 10–60 분 자유실행, 데모 2 레인 + P(EOT), Phase 2 보고서 | RTF <1, backlog 비발산, ID switch ≤1 회/10 분 |

v1.2 추가 작업: Q0에 event token registry·18개 fixture·turn 라벨 생성기를, Q1/Q2에 신뢰도 높은 자연 대화의 per-speaker start/end 공동 학습을 포함한다. Q3는 자체 생성 이벤트 이력을 쓰고 Q4는 speaker/turn_id별 API와 false/missed/repeated end를 평가한다. 기존 일정은 계획값이며 추가 라벨 검수 시간을 반영해 실측 후 조정한다.

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

자연 train 후보는 TS_01 실내 196.6h + otoSpeech 104.9h = 약 301.5h 상한이다. 화자 분할·QC 이후 더 줄어든다. VS_02와 TurnBench dev를 학습 시간에 넣지 않는다. 복원한 NIKL 합성은 원 음향·overlap 검증 전 자연 turn supervision으로 자동 승격하지 않는다.

| 구분 | Q1 | Q2 | Q3 |
|---|---|---|---|
| Phase 1 단일 화자 replay(turn 손실 마스크) | 30 % | 30 % | 30 % |
| 자연 2 화자 mono(71631·oto) | 60 % | 35 % | 70 % |
| 합성 2 화자(overlap 제어, turn 손실 마스크) | 10 %(무겹침) | 35 %(그 안에서 자연형 70 %/stress 30 % pilot) | 0 % |

**학습 레시피**: 동결 smoke 후 Q1에서 R-E2(thinker 2e-5 / encoder 해동 1e-5 / adapter 1e-3)와 R-low(같은 해동 범위, 기존 모듈 LR 0.5배)를 같은 노출량으로 비교한다. 새 heads LR은 동일하다. E2에서는 LR 감소와 해동이 함께 적용됐으므로 해동만의 효과로 해석하지 않는다. 정본 v1.1 §7.1의 warmup 이후 sentinel·ASR guardrail로 선택한다.

**학습 비용**: E2 1.02s/step(32 GPU, BS 12/24)은 참고값이다. 100–300 step의 처리량·메모리 smoke 이후 총 노출 audio-hours와 text/speaker token 수로 재산정한다. 정렬·LLM 약라벨·rollout·평가·선점 시간을 별도로 더한다. 단계당 1일 이내나 특정 파티션의 무선점을 보장하지 않는다.

**A/S 병행 및 승격**: S 라벨 pilot은 Q0부터 준비하되 Q1을 막지 않는다. A*는 Q1 시작 전 채택하거나, 늦게 채택하면 A*에서 Q1/Q2를 재실행한다. Q2 공유 가중치를 ASR-only A*로 단순 교체하지 않는다. Q3 parent는 다화자 관문을 재검증한 checkpoint로 동결한다. MVP-ASR/SD는 Q1/Q2의 중간 결과이며 최종 완료에는 의미 기여·ASR 강화·Q4가 필요하다.

## 4. 사용자 결정 요청

1. **영어 자연 대화** — **결정(2026-09-11)**: otoSpeech 를 학습에 쓴다(비상업 연구 범위, 라이선스 원문은 Q0 에서 재확인). CANDOR 는 확보 가능하면 쓴다(CC BY-NC) → [[task-secure-english-corpora]] 진행.
2. **71631 추가 반입**: 현재 248 h. TL_01.실내 라벨 8,306 대화 중 미반입분에서 100–200 h 를 PC 브라우저 경유로 더 가져올지. untouched KO test 용 20–30 대화는 별도로 반입 권장.
3. **단일 모델·항상 태그**: 단일 화자 입력에서도 `<SPK_A>` 를 방출하는 한 모델(데모는 태그 숨김)을 권장. 무태그 모드 토큰을 두면 데이터가 갈린다.
4. **turn 출력 형식 — 사용자 요구 반영**: 화자별 start/end 토큰을 전사와 함께 생성하고 activity·VAP head를 병행한다. pause/NEXT/EOF를 EOT와 구분하며 세부 규약은 [[output-phase2-block-and-turn-label-spec]]을 따른다. 필수 start/end를 ablation으로 미루지 않는다.

비판과 대안 제안: [[output-phase2-plan-critique]]. 최종 계획안: [[output-phase2-final-plan]]. 검토 답변과 정정: [[output-phase2-critique-response]].

## 불확실성
- v1.1은 초기 동결/해동 논쟁을 Q1 R-E2/R-low 비교로 구체화했다. 청크 token 상한은 고정 확대 대신 밀도·A/B 순서 편향·wall-clock 실측 후 결정한다.
- Switchboard·CallHome 시간축 복원 가능 여부, NIKL 준자연 대화 합성의 타당성은 Q0 에서 확인한다.
- 일정은 데이터 반입·사람 라벨링 대기를 포함하지 않는다.

## 근거
- [[output-phase2-streaming-asr-diarization-plan]] — 정본 계획(구조 §3, 계약 §4, 데이터 §5, 목표 §6, 단계 §7, 평가 §8, 구현 §9)
- [[output-phase1-report]] §7 — 이월 항목
- [[source-conversation-corpora]], `raw/sources/mxc-soundai-DB-survey.md`(2026-09-06) — 코퍼스 규모·채널·라이선스, 서버 보유 목록
- [[output-vap-target-pipeline]] — 라벨 파이프라인·코퍼스 통계
- [[output-stage2-e2-final-eval]] — E2 레시피·비용
- [[decision-mono-input]] — mono 입력·커리큘럼

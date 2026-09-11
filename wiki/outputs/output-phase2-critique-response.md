---
type: output
status: active
created: 2026-09-11
updated: 2026-09-11
summary: Phase 2 비판 P1–P9에 대한 답변 — 채택·조건부 수정과 encoder 프레임·문맥 예산·의미 라벨·checkpoint 승격 오류 정정
contributors:
  - tskim
sources:
  - '[[output-phase2-plan-critique]]'
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[output-phase2-plan]]'
  - '[[output-stage2-e2-final-eval]]'
  - '[[output-vap-target-pipeline]]'
  - '[[source-soulx-duplug]]'
---

# Phase 2 비판 검토와 계획 개정 근거

검토일: 2026-09-11. 대상은 [[output-phase2-plan-critique]]의 P1–P9다. 구현 사실은 현행 코드, 연구 주장은 관련 1차 논문과 비교했다. 변경은 계획·실행 요약에 반영했으며 모델 코드나 학습 job을 변경한 것은 아니다.

## 판정

**ASR 강화가 유지 관문에 머문 점, 의미 학습 신호 준비를 늦춘 점, speaker memory 검증을 Q4까지 미룰 수 있던 점은 원안의 약점이었다.** 이를 수정한다. 다만 비판의 구체적 대안에는 비용·시간 해상도·학습 목표를 과도하게 단순화한 부분이 있어 그대로 채택하지 않는다.

| 항목 | 판정 | 반영 내용 |
|---|---|---|
| P1 그룹 직렬화 | 기본 후보로 채택, 동등성 주장은 반박 | G1 A/B 그룹 vs G0 시각순 비교; B 지연·삭제·cap 편향 확인 |
| P2 의미 사전학습·약라벨 | 독립 S 트랙으로 채택, 무조건 대량 전이 보류 | 행동 전환/완결성 분리, prefix-only QA, text→audio head 전이 대조 |
| P3 2-slot memory | 조기 옵션으로 채택, EMA 효과는 검증 | Q1 인터페이스·복귀 검사, Q1 말/Q2 초 on/off; 오염·bootstrap·ID 보존 명시 |
| P4 2 audio token | 조건부 구조 ablation | encoder 프레임 수와 총 token 예산을 정정; 단일 경로 병목 확인 뒤 실행 |
| P5 ASR 강화 트랙 | 채택 | A1 정확도·A2 환경 적응·A3 속도, 필수 산출물·승격 lineage 추가 |
| P6 자연형 overlap | sampling 후보로 채택 | 합성 내 자연형 70%/stress 30% pilot; 유형별 평가·2화자 제약·turn mask |
| P7 E2 recipe 기본 | E2를 비교 후보로 격상, 무비교 채택은 보류 | R-E2와 모든 기존 모듈 LR을 절반으로 한 R-low 비교, encoder 양쪽 해동 |
| P8 turn δ=2 | Q3 첫 실행 조건으로 채택 | δ=4 필수 비교, 늦은 lexical 이력과 현재 acoustic state를 구분 |
| P9 MVP | 중간 산출물로 채택 | MVP-ASR/SD와 MVP-Turn 구분; 의미 기여 검증을 최종 요건으로 유지 |

## P1 — 그룹화는 유용하지만 ‘정보·학습 문제가 동일’하지 않다

동일 청크에 배정된 A/B lexical 토큰 집합과 각 화자 내부 순서는 그룹화해도 복원할 수 있다. 태그가 청크당 최대 2개가 되는 이점도 있다. 따라서 첫 구현 후보를 그룹화로 바꾼다.

하지만 원래 종료 시각순은 같은 80ms 안의 상대 순서를 시퀀스로 표현하며, A→B 그룹은 이를 제거한다. `P(B 토큰 | 앞선 A 토큰)` 등 decoder의 조건도 바뀐다. 특히 B는 A의 모든 토큰 계산을 기다리고 cap에 걸리면 먼저 누락될 수 있다. ‘청크가 있으니 내부 순서는 무의미하다’는 결론은 과하다. [t-SOT](https://arxiv.org/abs/2202.00842)는 시간순 직렬화의 선례이며 이번 그룹화의 동등성을 입증하지 않는다.

G0/G1은 학습도 각각 수행해 비교한다. 시각순으로 학습한 모델을 추론 시에만 그룹화하지 않는다. A/B별 품질·지연이 깨지면 청크 홀짝 순서 교대 G2를 후속 한 조건으로 넣는다. BPE의 청크 간 byte 조각과 화자별 문자열 복원은 그룹화 후에도 필요하다.

## P2 — 라벨 확대는 맞지만, 대화 전환은 의미적 완결성의 정답이 아니다

‘500–2,000건은 학습에 부족하다’는 것은 모델 동결 범위·라벨 품질·pretraining에 따라 달라지는 가설이다. 수백 시간 자연 대화의 미래 활동 supervision도 의미 신호가 전혀 없는 것은 아니다. 반면 원안에 의미 supervision 규모별 실험이 없었던 지적은 타당하다.

전사에서 다음 화자가 바뀌었는지 알 수 있어도 문장이 의미적으로 끝났다는 뜻은 아니다. 완료된 말 뒤 같은 화자가 계속할 수도 있고, 미완료된 말을 상대가 끊을 수도 있다. 그래서 실제 다음 onset을 예측하는 **행동 label**과 prefix가 완결됐는지 판단하는 **의미 label**을 분리한다.

완성 전사 전체를 본 LLM은 학습 cutoff 뒤의 정보를 이용해 모호한 prefix를 지나치게 확실하게 평가할 수 있다. 미래 사건을 target으로 쓰는 것 자체는 적법하지만, 이를 ‘현재 prefix에서 판단 가능한 완결성’ gold와 혼동하면 안 된다. 의미 teacher는 prefix+이전 문맥만 보고 uncertain을 허용하도록 계약화했다. 사람 검증을 통과한 약라벨은 학습용으로 격상한다.

텍스트 readout에서 학습한 head가 audio 위치의 hidden state에 그대로 맞는다는 보장은 없다. 따라서 같은 E2 backbone 위 random/text-pretrained head의 비교, 자연 오디오에 대한 재적응을 추가한다. thinker 전체를 텍스트만으로 바꾸는 경우 ASR 손상이 가능해 별도 branch에서 검증한다. [SoulX-Duplug](https://arxiv.org/abs/2603.14877)는 상태 예측·약라벨링의 참고이며 이 전이 방식의 증거는 아니다.

수백만 prefix의 생성·LLM 라벨링·학습 비용은 입력 길이·teacher 크기·GPU 수가 없으면 계산할 수 없다. ‘GPU 1시간’은 채택하지 않는다. 작은 label pilot의 실제 처리량으로 확대 여부를 정한다. 이 작업은 Q0 필수 관문에 직렬로 붙이지 않고 S 트랙으로 병행한다.

## P3 — memory는 앞당기되 ‘지연 0의 작은 EMA’로 취급하지 않는다

LLM context에만 ID 유지를 맡기고 실패 후 설계를 시작하는 것은 일정 위험이다. Q1부터 memory 인터페이스와 긴 침묵 복귀 검사를 준비한다.

다만 원 encoder feature 평균은 화자 고유 표현으로 학습된 것이 아니며, 잘못 분류한 청크로 prototype을 갱신하면 ID 오류가 자기강화된다. 두 슬롯을 단순 합산하면 슬롯 교환에 불변이라 A/B 정체성이 직접 보존되지 않는다. ‘태그 반전은 거의 확실하다’와 ‘memory 지연 0’ 모두 측정 전 단정이다.

수정안은 projected speaker feature, confidence 기반 단독 구간 갱신, 불확실·overlap 갱신 중단, 슬롯별 bootstrap/유효 상태, slot ID를 보존하는 readout, 예측 기반 갱신의 훈련·평가 일치를 포함한다. state 갱신은 다음 청크부터 반영해 순환 의존을 피한다.

[Streaming Sortformer](https://arxiv.org/abs/2507.18446)의 AOSC는 예측 점수로 선택한 여러 프레임의 cache와 FIFO를 사용한다. 그것이 Nemotron ASR feature 두 개의 EMA로 그대로 재현되는 것은 아니다. no-memory/EMA 또는 소형 exemplar bank 비교로 판단한다.

## P4 — 두 가지 계산 오류를 수정해야 한다

`vapasr/features/online.py`의 encoder 출력은 `(B,K,1024)`, **12.5Hz=80ms당 1프레임**이다. 8배 subsampling 이전의 10ms mel frame과 혼동하면 안 된다. 슬롯 query가 볼 대상은 현재 1프레임과 causal 과거이며, 고해상도 feature를 원하면 별도 추출 경로가 필요하다. 두 slot query가 같은 내용을 복제할 위험도 있다.

15분 세션의 위치 수는 다음과 같다. T=lexical token 수, S=speaker tag 수, P=prefix 등 추가 위치다.

| 경로 | audio 위치 | NEXT 위치 | 전체 하한 |
|---|---:|---:|---:|
| 현재 1 audio/청크 | 11,250 | 11,250 | 22,500 + T + S + P |
| 제안 2 audio/청크 | 22,500 | 11,250 | 33,750 + T + S + P |

따라서 ‘15분 약 22k token이므로 32k 이내’는 `<NEXT_AUDIO>`부터 누락했다. 실제 context 설정을 확인해야 하며, 32,768을 가정해도 텍스트 전부터 초과한다. KV 전체도 같은 dtype에서 토큰 수 비율 `(3K+T+S+P)/(2K+T+S+P)`로 증가하므로 정확히 2배가 아니다. 단일 경로도 긴 세션이면 bounded context가 필요하다.

mono 입력 원칙과 양립하고 multi-output decoder보다 작은 변경일 가능성은 인정한다. 따라서 **안정적인 slot memory와 병목 진단 뒤**의 첫 구조 ablation으로 반영했다. 추가 token이 이미 손실된 음향 정보를 되살린다고 가정하지 않는다.

## P5 — ASR 강화는 명시적 트랙으로, checkpoint 교체는 lineage를 지킨다

원안은 regression guardrail에 집중하고 ‘실시간 ASR 강화’의 독립 완료 기준이 부족했다. A1 저지연 인식, A2 대화·잡음 적응, A3 runtime 개선과 채택 지표를 계획 §10.1에 추가했다.

다만 Q3 직전에 **ASR-only A 모델을 Q2 모델과 단순 교체하면 다화자 학습한 공유 가중치가 사라진다**. head만 붙여도 복구되지 않는다. A* 승격은 Q1 시작 전에 하거나 A*에서 Q1/Q2를 다시 수행해야 한다. 늦게 발견한 recipe는 다화자 checkpoint에서 별도 재학습·검증할 수 있지만 가중치 단순 접합은 아니다. 수치 동등 runtime 최적화는 별도로 반영 가능하다.

## P6 — 자연형 패턴은 채택하되 합성 사건을 자연 의미 label로 쓰지 않는다

원안의 overlap 비율 목록은 평가 stress strata였고 균일 sampling을 지정한 것은 아니었다. 그래도 실제 합성 sampling 비중을 비워 둔 것은 보완할 점이다. 합성 내 자연형 70%/stress 30%를 첫 후보로 추가했다.

51k BC 대 6.6k INT는 휴리스틱 event 수이며 overlap 지속시간 분포나 사람이 판단한 실제 유형 비율이 아니다. Q0 검수 후 분포를 고정한다. 임의로 삽입한 짧은 응답은 담화적으로 적절한 맞장구가 아닐 수 있어 turn loss는 계속 마스킹한다. 응답 뱅크에서 다른 사람 음성을 가져오면 3화자가 되므로 donor는 세션 B와 같은 화자로 제한한다.

## P7 — E2는 유력 recipe지만 효과 원인은 분리되지 않았다

[[output-stage2-e2-final-eval]]에는 D4→E2에서 encoder 해동과 thinker LR 감소가 동시에 적용됐고 이를 분리한 대조군은 생략됐다고 명시되어 있다. ‘encoder 해동이 붕괴를 없앴다’는 단독 원인 주장은 근거를 넘는다. 더 작은 대화 데이터·새 태그·새 heads에서도 같은 LR이 최선인지는 미정이다.

원안의 adapter LR `1e-5–5e-5` 역시 E2의 `1e-3`보다 20–100배 낮게 잡은 근거가 부족했다. 따라서 E2 recipe를 R-E2 후보로 격상하고 R-low(encoder 해동 유지, 기존 모듈 LR 0.5배)와 제한 비교하도록 수정했다. 두 recipe 묶음 비교이며 모듈별 원인 분석으로 과장하지 않는다. 동결 경로는 smoke·불안정 원인 분리에 사용한다. 실제 batch·노출량·warmup 이후 sentinel로 판정한다.

## P8 — δ=2는 좋은 첫 조건, δ=4 정보가 전부 낡은 것은 아니다

`h_audio[k]`에는 현재 audio가 들어온다. δ=4에서 낡는 것은 주로 확정 lexical history이며 전체 semantic representation을 ‘320ms 전 정보’라고 할 수 없다. δ=2는 최근 단어가 빨리 들어오지만 ASR 오류가 늘 수 있다. Q3 첫 실행을 δ=2로 바꾸고, 각 조건의 자체 생성 history로 학습/평가한 δ=4를 필수 비교한다. 같은 청크 전사 전 head 위치와 80ms 출력 주기는 유지한다.

## P9 — MVP는 필요하지만 사용자 목표를 축소해 완료 선언하면 안 된다

Q1+Q2+activity가 실제 관문을 통과하면 화자별 겹침 전사의 중간 산출물이다. 이를 **MVP-ASR/SD**로 이름 붙였다. Q3의 VAP만으로 미래 활동 예측은 가능하나 의미 정보 기여가 증명되지는 않는다. 의미 ablation을 포함한 **MVP-Turn**을 별도로 두고, 일정 부족 시 hazard·복잡한 이벤트 세분화를 미룬다. 최종 Phase 2 완료에는 의미 기여·ASR 강화·장문 실시간 결과를 유지한다.

## 함께 정정한 운영 주장

- ‘encoder 동결 시 반드시 가중치가 빠진다’는 것은 조건부다. `requires_grad=False`만으로 저장 flag가 바뀌지는 않는다. config/ignore 목록이 저장을 막는 경로에서 소실되며, `attach_encoder()`로 재부착하면 이전 가중치가 더 일찍 사라질 수도 있다. 학습 상태와 artifact 저장 정책을 분리하는 조치는 유지한다.
- 자연 대화 train은 VS/TurnBench dev를 제외하면, 후속 보유 기록상 TS 196.6h + oto 104.9h = **약 301.5h 상한**이다. 화자 분할·QC 후 더 작다. 약 350h 전체를 train으로 계산하지 않는다.
- E2 4.3시간은 기존 데이터 길이·batch·token 밀도·GPU 구성의 측정치다. 2화자·긴 문맥·head rollout 비용을 그대로 외삽해 Q1/Q2 2–3시간, Q3 1시간 미만으로 확정하지 않는다. [[output-phase2-plan]]의 이 부분을 잠정 예산으로 정정했다.

## 반영 위치와 다음 검증

정본 [[output-phase2-streaming-asr-diarization-plan]] v1.1에 P1–P9의 실행 변경을 반영하고, [[output-phase2-plan]]의 recipe·학습 비용·데이터 혼합·단계 설명을 맞췄다. 비판글의 원 주장은 검토 기록으로 보존한다. 작업 도중 추가된 [[output-phase2-final-plan]]은 검토 전 통합 기록으로 보존하고, 개정 정본을 우선 적용한다는 안내를 붙였다. 특히 그 문서의 test 표본 기반 승격은 적용하지 않으며 모델 선택은 dev에서 한다.

첫 구현 묶음은 Q0 mixer·encoder 저장 parity·평가 pack → G0/G1 serializer → 작은 overfit다. S 라벨 pilot과 A 트랙 baseline은 병행 가능하다. 이번에 추가한 설계도 실험 전 가설이며, 품질·효율·전이 효과를 얻었다고 주장하지 않는다.

외부 1차 자료 확인일: 2026-09-11. t-SOT, Streaming Sortformer, SoulX-Duplug 원문을 위 링크로 확인했다. 코드 관찰은 `vapasr/features/online.py`, `vapasr/hf/modeling_vapasr.py`, `vapasr/hf/trainer.py`, `experiments/s3_train_hf.py` 기준이다.

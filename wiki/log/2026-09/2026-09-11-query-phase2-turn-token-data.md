## [2026-09-11] query | turn 토큰·시퀀스 사례·데이터 생성 파이프라인을 별도 제안서로

- Changed: `wiki/outputs/output-phase2-turn-token-proposal.md` 생성. 정본 `output-phase2-streaming-asr-diarization-plan.md` 는 직전 커밋의 §4.4·§5.3 삽입을 되돌려 원상 복구. `output-phase2-final-plan.md` 는 superseded 유지, 링크 수정
- Reason: 사용자가 정본 계획을 기본으로 택하고 교차 직렬화를 선호했으며, turn 토큰 제안은 "정본에 바로 적용하지 말고 별도 보고서로" 요청. 제안서에 규약·사례 7 종·전자동 데이터 파이프라인(VAD → 채널별 정렬 → derive_events 확장 → 슬롯 → 혼합 → 직렬화 → QC)·데이터 종류별 손실 규칙·정본 반영 시 변경 지점을 정리
- Next: 사용자 검토 후 정본 §4.4·§5.3 반영 여부 결정
- By: tskim

## [2026-09-11] query | 제안서 개정 — 토큰 종류를 의미 라벨로, LLM 시드 → 증류 → 전량 → 사람 검증

- Changed: `output-phase2-turn-token-proposal.md` §3b·§3c 신설, 요약·반영 지점·불확실성 갱신
- Reason: 사용자 검토 — 라벨이 음향(타이밍)에만 의존하고 TurnBench 에 과도하게 맞춰졌다는 지적. 위치는 VAD, 종류는 의미(완결/미완결/맞장구)로 분리하고, 텍스트만으로 사후 문맥을 보는 LLM 시드 10 만 → 증류 분류기 → 전량 1 천만 경계 → 사람 κ ≥ 0.7 검증의 저비용 경로를 제안. 완결성 정확도·응답 기회 P/R 을 주 지표로, TurnBench 는 EN 외부 검증
- Next: 사용자 검토 → L1 프롬프트·층화 표본 설계
- By: tskim

## [2026-09-11] query | 제안서 개정 — <HOLD> 를 기한 있는 잠정 판단으로, 승격·정책 계층 추가

- Changed: `output-phase2-turn-token-proposal.md` §3d 신설(정의 변경, 포기된 발화로 승격 학습, 추론 정책 계층 reason∈{semantic,hazard,timeout}, 정체율·끊김률 지표, 대안 A/B/C)
- Reason: 사용자 지적 — <HOLD> 오예측 후 후속 발화가 없으면 정체. HOLD 를 없애도 "EOT 없는 침묵" 으로 같은 문제가 남으므로, HOLD 를 잠정으로 두고 τ_max·hazard 기반 승격을 학습·추론 양쪽에 둠
- Next: Q0 에서 포기된 발화 빈도·gap 분포 측정, τ_max 후보 1.5/2.0/3.0 s
- By: tskim

## [2026-09-11] query | 제안서 개정 — <HOLD>/<EOT> 결정표(문장 완결 vs 기여 완결 × 멈춤 길이)

- Changed: `output-phase2-turn-token-proposal.md` §3e 신설(τ_cand 0.25 s 후보, LLM 등급 INCOMPLETE / COMPLETE-CONTINUING / COMPLETE-FINAL / BC / AMBIGUOUS, τ_offer 1.0 s·τ_max 2.0 s 결정표, 여러 문장·단어 공백·긴 생각 멈춤 사례), §3b.2 표 갱신
- Reason: 사용자 질문 — 2–3 문장 연속 발화와 단어 사이 공백에서 <HOLD> 를 어떻게 구분하나
- Next: Q0 에서 멈춤 분포·등급별 κ·분당 토큰 수 측정 후 τ 확정
- By: tskim

## [2026-09-11] query | 제안서 개정 — <HOLD> 위치를 침묵 0.5 s 로, 문장 분리 불필요, 한국어 구어체 신뢰도 파일럿

- Changed: `output-phase2-turn-token-proposal.md` §3f 신설(τ_hold 0.5 s 앵커, 사람 발화 분절을 prior 로, 한국어 어미 유형별 대응, Q0 κ 파일럿과 축소 사다리), §2 표·§3e 밀도 항목 갱신
- Reason: 사용자 질문 — 문장마다 HOLD 인가, 문장 분리는 어떻게, 한국어 구어체에서 신뢰성 있는 라벨이 가능한가
- Next: Q0 파일럿 300 지점 × 3 명 설계
- By: tskim

## [2026-09-11] decision | Phase 2 turn 토큰은 <ONSET>+<EOT> 만, <HOLD>/<BC>·의미 라벨링은 Stage 3 이월

- Changed: `output-phase2-turn-token-proposal.md` §1(채택 범위)·§1b(이월) 신설, §3b–§3f 를 [Stage 3 이월] 표시; `wiki/questions/question-turn-token-label-reliability.md` 생성
- Reason: 사용자 판단 — <HOLD>·<BC> 는 신뢰성 높은 데이터를 만들 수 있을지 불확실. Phase 2 는 타이밍 라벨로 전량 자동 생성 가능한 두 토큰만 쓰고, 추론 정체는 타임아웃·hazard 승격 정책으로 처리
- Next: Stage 3 착수 전 κ 파일럿으로 재검토
- By: tskim

## [2026-09-11] decision | otoSpeech 학습 사용·CANDOR 는 확보 시 사용; <ONSET> 지연은 δ_on∈{0,1,2} sweep

- Changed: `output-phase2-plan.md` 결정 1 확정, `task-secure-english-corpora.md` 진행 기록, `output-phase2-turn-token-proposal.md` §1c(δ_on 근거·결정 방식)
- Reason: 사용자 결정(EN 코퍼스)과 질문(왜 +160 ms 고정인가). 즉시 방출은 증거 0–80 ms 라 false onset 위험, 빠른 감지는 activity 헤드가 맡으므로 토큰은 Q1 에서 precision/recall·지연으로 δ_on 을 고른다
- Next: Q1 sweep 설계에 δ_on 포함
- By: tskim

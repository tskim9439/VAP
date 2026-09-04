<!-- generated: do not edit -->
# 인덱스

마지막 생성: 2026-09-04

각 페이지의 `summary` frontmatter 로부터 생성된다. 직접 편집하지 말고 페이지의
`summary` 를 고친 뒤 병합 시 재생성한다 (wiki-merge 스킬).

## Overview
- [[overview]] — streaming ASR과 VAP를 하나의 representation으로 통합하는 연구를 축적하는 볼트

## Status
- [[status]] — 현재 유지보수 상태, 다음 액션, 린트 로테이션 담당

## Concepts
- [[acoustic-linguistic-fusion]] — ASR 내부 decoder state를 turn predictor와 공유해 cascade 없이 semantic 정보를 얻는 설계
- [[korean-turn-taking-cues]] — 한국어 어미와 clause-final prosody가 turn 완결성 신호로 작동하는 방식
- [[streaming-causality-and-latency-budget]] — 80ms frame ASR encoder로 VAP를 옮길 때의 causality 감사와 latency 회계. **최대 방법론적 위험**
- [[streaming-conversational-projection-asr]] — 하나의 streaming representation에서 transcription과 미래 대화 역학을 동시 예측하는 제안 모델
- [[turn-taking-evaluation-protocol]] — TurnBench 규약을 채택한 평가 프로토콜
- [[turn-taking-objectives]] — VAP 256-class + time-to-next-turn(생존분석) + semantic event의 다중 목표 설계
- [[voice-activity-projection]] — 미래 2초 구간의 양 화자 음성 활동을 256-class로 예측하는 turn-taking 모델

## Entities
_(없음 — 모델·시스템 정보는 현재 `wiki/sources/` 에 통합되어 있다)_

## Sources
- [[source-chatgpt-research-plan]] — ChatGPT 로 작성한 streaming ASR + VAP 통합 연구 초안. 이 볼트 연구 방향의 출발점
- [[source-conversation-corpora]] — 학습·평가용 대화 코퍼스 4종의 채널 구조, 규모, 라이선스 제약
- [[source-muse-voice-transcribe]] — Meta Muse Voice Transcribe. 80ms soft token 기반 통합 모델, closed weights
- [[source-nemotron-3-5-asr-streaming]] — NVIDIA Nemotron 3.5 ASR Streaming 0.6B. 첫 baseline backbone
- [[source-qwen3-asr]] — Qwen3-ASR-0.6B. AuT 180M, 12.5Hz/80ms, Apache 2.0. main backbone 후보
- [[source-turn-taking-related-work-2026]] — JAL-Turn, Next-Turn, DualTurn의 기여와 baseline 위치
- [[source-turnbench]] — Sesame TurnBench. 30h 영어 dyadic 벤치마크와 104h otoSpeech

## Questions
- [[question-asr-representation-vs-ssl-for-vap]] — ASR pretrained representation이 SSL/생성형보다 유리한가. **연구의 중심 질문**
- [[question-encoder-lookahead-and-causality]] — **답변됨** Nemotron ≤80ms(80ms chunk), Qwen AuT 비인과/블록 420ms
- [[question-event-label-derivation-validity]] — 유도한 EOT/HOLD/INT/BACKCHANNEL 라벨이 사람 라벨과 얼마나 일치하는가
- [[question-korean-corpus-licensing]] — 재배포 가능한 한국어 stereo 대화 코퍼스가 존재하는가
- [[question-korean-turn-cue-literature]] — 한국어 어미·prosody가 turn 완결성을 신호한다는 주장의 문헌 근거
- [[question-spokenwoz-channel-structure]] — SpokenWOZ가 화자별 분리 채널로 배포되는지

## Outputs
- [[output-stage1-encoder-probing]] — Stage 1 frozen probing 잠정 판정: 프레임율 > 인코더, H1 미지지, Qwen 실외 취약
- [[output-interleaved-streaming-slm-architecture]] — **채택 기본 구조** IS-SLM: 80 ms soft token + `<NEXT_AUDIO>` 가변 방출 + audio-clock 헤드; 말미에 볼트 검토·통합 절
- [[output-unified-slm-architecture-plan]] — 통합 스트리밍 SLM(합산 융합·시간 동기) 구조 계획 + 12항목 비판 평가 + U0–U4 검증 계획
- [[output-model-architecture-proposal]] — (superseded) 이중 프레임율 구조 제안 v1: 이중 프레임율(50 Hz CPC + 12.5 Hz FastConformer) + RNNT predictor state 융합, 학습 단계 1:1
- [[output-feature-cache-and-compute-budget]] — frozen encoder 7종 특징 캐시: RTF·용량·이어붙이기 검증, AuT 13 Hz 발견
- [[output-vap-target-pipeline]] — backbone 독립 target 파이프라인: 코퍼스 3종 → VAD@50Hz → VAP·hazard·이벤트 파생, 55,139 창
- [[output-vap-turnbench-baseline-reproduction]] — TurnBench dev VAP 재현: oto 체크포인트 공식과 완전 일치, 사전학습 원본은 recall −0.05
- [[output-encoder-causality-audit]] — 절단 실험 lookahead 측정. Nemotron 80ms chunk ≤80ms, Qwen AuT 비인과·블록 평균 420ms
- [[output-streaming-vap-research-plan]] — **개선 연구 계획 v2.** 논문 2편 분할, 5개 위험 통제, 17개 실행 태스크

## Meetings
_(없음)_

## Decisions
- [[decision-target-architecture]] — **IS-SLM 단일 주력**, 이중 프레임율+RNN-T 기각 (2026-09-04 확정)
- [[decision-compute-environment]] — 학습은 rack4의 tskim_env 컨테이너, 체크포인트 /data4, 데이터 /data3, 설정은 `.env` 단일 관리 (accepted)
- [[decision-asr-backbone]] — **확정** IS-SLM backbone = Nemotron [56,0] → adapter → Qwen3-ASR thinker; 관문 U0.5, fallback AuT+thinker
- [[decision-korean-benchmark-release-scope]] — 한국어 벤치마크는 오디오가 아닌 어노테이션·툴킷만 공개 (proposed)

## Tasks

열린 태스크 대시보드는 [[todo]] 를 본다.

- [[task-add-missing-baselines]] — VAD+threshold, cascade, DualTurn, JAL-Turn 등 baseline 4종 구축
- [[task-audit-encoder-causality-lookahead]] — 실효 lookahead를 절단 실험으로 측정. 모든 모델링의 선행 조건
- [[task-bilingual-and-qwen-port]] — KO/EN temperature sampling 학습과 Qwen3-ASR 포팅
- [[task-build-vap-target-pipeline]] — **완료** target 파이프라인 (`vapasr/data/`)
- [[task-compute-budget-and-feature-cache]] — GPU 예산 산정과 frozen encoder 출력 캐싱
- [[task-event-label-heuristics-validation]] — 유도 라벨 규칙을 otoSpeech에 대해 검증
- [[task-korean-benchmark-design]] — 한국어 어노테이션 프로토콜 설계
- [[task-korean-turn-cue-literature-review]] — 한국어 turn 단서 문헌 조사
- [[task-latency-quality-curve]] — chunk별 latency-quality 곡선
- [[task-uslm-feasibility-u0]] — USLM U0: 토큰율(KO 폭주 없음)·M=4·생성기 완료, 정렬 진행 중 (p1)
- [[task-uslm-u05-adapter-bridge]] — USLM U0.5: 4 run 완료, 최선 18.2/17.5/14.5 %, adapter 병목 아님 → 관문 판정 대기 (p0)
- [[task-uslm-u1-interleaved-asr]] — USLM U1: interleaved ASR, WER 관문 (p0)
- [[task-uslm-u2-self-conditioned]] — USLM U2: self-conditioned streaming (p1)
- [[task-uslm-u3-multitask]] — USLM U3: 멀티태스크 헤드 + 하이브리드 + H2 (p0)
- [[task-qwen-aut-causal-adaptation]] — (취소) 최종 backbone 확정으로 주 경로에서 제거; 마스크 주입 코드는 유지
- [[task-paper1-scoping]] — Paper 1 범위 확정과 아웃라인
- [[task-reproduce-vap-turnbench-baseline]] — **완료** VAP 재현, dev 0.841/0.045/463 ms
- [[task-secure-english-corpora]] — otoSpeech·CANDOR 확보
- [[task-setup-training-environment]] — 컨테이너에 torch·NeMo·transformers 구축, GPU 확인
- [[task-checkpoint-retention-policy]] — /data4 여유 575G 상황의 체크포인트 보존·정리 규칙
- [[task-stage1-encoder-probing]] — frozen encoder 비교 실험(14 run 완료, seed 반복·EN-only·DualTurn 잔여). Paper 1의 핵심 결과
- [[task-stage2-multitask-vap-with-wer-guardrail]] — (폐기) A 안 전용 → U3
- [[task-stage3-linguistic-state-fusion]] — (폐기) A 안 전용 → U3
- [[task-time-to-next-turn-survival-head]] — τ head를 discrete-time hazard로 구현
- [[task-verify-aihub-stereo-and-access]] — **완료** AI Hub 실물 검증: 16 kHz 분리 stereo, 온셋 오차 30 ms

## Templates
- [[source-note]] — 원천 자료 노트 템플릿
- [[concept]] — 개념 페이지 템플릿
- [[entity]] — 인물·조직·제품·프로젝트 페이지 템플릿
- [[meeting]] — 회의록 템플릿
- [[decision]] — 결정 기록 템플릿
- [[task]] — 태스크 템플릿
- [[question]] — 미해결 질문 템플릿
- [[output]] — 산출물 템플릿

## Maintenance
_(없음)_

---
type: overview
status: active
created: 2026-09-03
updated: 2026-09-04
summary: streaming ASR과 VAP를 하나의 representation으로 통합하는 연구를 축적하는 볼트
---

# 개요

## 이 볼트가 다루는 것

**streaming ASR 과 turn-taking 예측을 하나의 representation 에서 동시에 푸는 연구.**

목표 모델 [[streaming-conversational-projection-asr]] 는 매 시점의 streaming audio 에서
transcription 과 **미래 2초의 대화 역학**(누가 언제 말할 것인가, interruption,
backchannel)을 함께 예측한다.

핵심 구분: [[source-muse-voice-transcribe]] 같은 endpointing 은 "지금 끝났는가" 를
묻고, [[voice-activity-projection]] 은 "앞으로 누가 말할 것인가" 를 묻는다.
사람은 턴 종료 **−151 ms** 시점에 이미 움직이지만 최고 성능 VAP 는 **368 ms** 가
걸린다 ([[source-turnbench]]). 이 **약 520 ms 의 격차**가 연구 대상이다.

## 현재 상태

[[source-chatgpt-research-plan]] 초안을 사실 검증하고 수정한
**[[output-streaming-vap-research-plan]] (v2)** 가 현재의 계획이다.
17개 태스크로 분해되어 있다 → [[todo]]

초안의 인용은 전부 실재했으나, 계획을 바꾸는 사실이 나왔다:
DualTurn 누락, Muse 의 closed-weights, AI Hub 재배포 제약,
encoder causality 미검증.

## 주제 영역

### 모델과 목표
- [[voice-activity-projection]] — 출발 baseline
- [[streaming-conversational-projection-asr]] — 제안 모델과 3개 가설
- [[turn-taking-objectives]] — VAP + hazard + event 다중 목표
- [[acoustic-linguistic-fusion]] — cascade 없는 semantic 결합

### 제약과 평가
- [[streaming-causality-and-latency-budget]] — **최대 방법론적 위험**
- [[turn-taking-evaluation-protocol]] — TurnBench 규약 채택
- [[korean-turn-taking-cues]] — 한국어가 흥미로운 이유

### 자원
- [[source-conversation-corpora]] — 코퍼스 채널 구조와 라이선스
- [[source-qwen3-asr]] / [[source-nemotron-3-5-asr-streaming]] — backbone 후보
- [[source-turn-taking-related-work-2026]] — JAL-Turn, Next-Turn, **DualTurn**

### 결정
- [[decision-asr-backbone]] — 최종: Nemotron `[56,0]` → adapter → Qwen3-ASR thinker LM
- [[decision-korean-benchmark-release-scope]] — 어노테이션 레이어로 공개

## 다음 관문

causality 감사는 완료되었고 최종 backbone이 확정되었다. 다음 선행 관문은
[[task-uslm-feasibility-u0]]의 정렬·토큰율 검증과 [[task-uslm-u05-adapter-bridge]]의
Nemotron–thinker 연결 품질 검증이다.

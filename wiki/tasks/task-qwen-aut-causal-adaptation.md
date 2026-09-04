---
type: task
status: done
owner: tskim
due: 2027-01-28
priority: p3
created: 2026-09-03
updated: 2026-09-04
summary: (취소) 최종 backbone을 Nemotron [56,0] + adapter + Qwen thinker로 확정해 Qwen AuT causal 적응을 주 경로에서 제거
sources:
  - [[output-encoder-causality-audit]]
---

# Qwen AuT causal 적응 연구

## 배경

[[output-encoder-causality-audit]]: AuT 는 sdpa 경로에서 비인과이고, 의도된 블록 모드도
1 s 블록 기준 lookahead 평균 420 ms + 블록 간 좌측 context 부재라 [[decision-asr-backbone]] 의
320 ms 조건에 걸렸다. 2026-09-04 최종 backbone에서 Qwen AuT를 사용하지 않기로 확정했으므로
이 적응 연구는 더 이상 주 경로의 필수 작업이 아니다. 아래 항목은 후속 연구 아이디어로만 보존한다.

## 완료 조건

- [x] 마스크 주입 패치 — `experiments/qwen_aut_mask.py` (block / chunked-causal / causal). block 1 s 가 per-block 실측과 일치해 검증됨
- [ ] `n_window` 50 → 25 → 12 (1 s → 0.5 s → 0.25 s conv chunk) 에서 WER/CER 열화 곡선 (재학습 없이)
- [ ] 블록 간 좌측 context 를 주는 변형(sliding window / cache) 설계 — 위치 임베딩이 chunk 단위라 검토 필요
- [ ] **causal 마스크 fine-tune** — 재학습 없이도 WER 23.5 %(단일 발화)로 단어 대부분 보존. 소규모 ASR 손실 fine-tune 으로 회복 폭 측정 (핵심)
- [ ] 마스크별 WER/CER 을 제대로 된 평가셋에서 — LibriSpeech test-clean 일부 + 한국어(KsponSpeech 또는 AI Hub 서브셋). 단일 발화 수치는 신호일 뿐
- [ ] 각 변형의 lookahead 를 `experiments/causality_audit.py` 로 재측정
- [ ] 판정: Paper 2 의 main backbone 으로 유지 / Nemotron 단일 backbone 으로 전환

## 진행 기록

- 2026-09-04: **취소·종결.** [[decision-asr-backbone]]에서 Nemotron `[56,0]` → adapter → Qwen thinker를 최종 조합으로 확정했다. 불필요한 encoder 교체와 causal AuT 적응을 일정에서 제거한다.
- 2026-09-03: 마스크 주입 실험 완료 → [[output-encoder-causality-audit]] 추가 실험 절. chunked-causal 1 s 가 as-is 최선(420 ms, WER +5.9 %),
  프레임 causal 은 lookahead 80 ms 에 WER 23.5 % — fine-tune 으로 회복 가능성. 우선순위 p2 → **p1 상향 검토** (Paper 2 backbone 결정에 직결).
- 2026-09-03: 생성. 감사 결과에 따라 [[decision-asr-backbone]] 재검토의 일부.

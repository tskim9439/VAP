---
type: task
status: open
owner: tskim
due: 2027-02-11
priority: p2
created: 2026-09-03
updated: 2026-09-04
summary: 최종 Nemotron–adapter–Qwen thinker backbone의 KO/EN temperature sampling과 언어별·잡음 조건 강건성 평가
sources:
  - [[output-streaming-vap-research-plan]]
---

# Bilingual 학습 및 강건성 평가

## 배경

→ [[decision-asr-backbone]] 최종 backbone.

Paper 1 결과가 H1 을 지지할 때만 착수한다.

## 완료 조건

- [ ] KO/EN 1:1 단순 sampling 이 아니라 **temperature sampling** 적용
- [ ] monolingual KO / monolingual EN / bilingual 3조건 비교 —
      각 언어의 monolingual 성능에 근접하는지 확인
- [ ] cross-language zero-shot 성능 측정 (KO 학습 → EN 평가, 역방향)
- [ ] Qwen3-ForcedAligner 로 audio/text interleaving 시퀀스 생성.
      한국어 지원 **확인됨** (2026-09-03, 모델 카드)
- [ ] 정렬 QC — aligner 신뢰도 낮은 구간, 에너지 VAD 와 불일치 큰 구간 제외
- [ ] AI Hub 실내/실외 및 SNR 조건별 WER/CER·latency 비교

## 진행 기록

- 2026-09-04: 최종 backbone을 유지하도록 범위를 수정. Qwen AuT 포팅·encoder 교체 작업을 제거하고 bilingual/robustness 평가만 남김.
- 2026-09-03: 생성.

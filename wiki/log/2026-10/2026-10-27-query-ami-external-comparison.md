## [2026-10-27] query | AMI 외부 비교 — Multitalker Parakeet·NVIDIA SSA-v2·VibeVoice-ASR(-Streaming) 대 우리 E6

- Changed: [[output-ami-external-comparison-20261027]](새)
- Reason:
  - 사용자 "… 이것과 VibeVoice-ASR, "Multitalker Parakeet SDM" 과는 비교해서는 어떠한지 정리해줘".
  - OpenReview 문서(r6QZahgnJY)는 브라우저 확인 화면 403 으로 받지 못했다.
  - 외부 수치는 HF 카드 raw README 와 arXiv HTML 표에서 직접 뽑았다(2601.18184·2609.02812·2609.10265·2609.15218).
  - AMI SDM cpWER:
    - Parakeet v1 37.44(1.12 s), NVIDIA SSA-v2 + Nemotron 3 Diarization 20.33
    - VibeVoice-ASR-Streaming 39.01(2.0 s), VibeVoice-ASR 오프라인 28.82(다른 평가 42.19·36.09)
    - 우리 E6 45.35 — 창 cpWER, 자체 정규화
- Next:
  - 회의 통째(96172·96461)를 meeteval + 표준 정규화(채움말 제거)로 다시 채점해 같은 조건으로 맞춘다.
  - OpenReview 문서를 받으면 덧붙인다.
- By: tskim

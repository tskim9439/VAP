## [2026-10-28] ingest | VibeVoice-ASR-Streaming(ICLR 2027 투고, OpenReview r6QZahgnJY)

- Changed:
  - raw 분류: `raw/inbox/15369_VibeVoice_ASR_Streaming_.pdf` → `raw/sources/15369_VibeVoice_ASR_Streaming_.pdf`(논문 — Raw File Classification).
    - inbox 의 예전 두 파일(DataSelectionPlan.md·streaming_asr_semantic_commit_plan.md)은 이번 요청 밖이라 그대로 뒀다.
  - [[source-vibevoice-asr-streaming-iclr2027]](새)
  - [[output-ami-external-comparison-20261027]]: 1.5B·15 프레임·Azure 줄, 같은 LLM 크기 비교 절
  - `raw/sources/experiments/2026-10-27-std-rescore/external_ami.json`(10 줄, WER 열), `experiments/meeteval_cpwer.py`(WER 함께 출력)
- Reason:
  - 사용자 "raw/inbox/ 에 Openreview 논문 추가했어. 거기에 1.7 B 모델도 있는데 함께 비교할 수 있도록 해줘".
  - 문서의 작은 모델은 1.5B(Qwen2.5-1.5B)다. AMI cpWER 는 IHM 45.15·SDM 56.83(2.00 s)으로, 우리 E6(LLM 1.7B, 창) 37.68·45.35 보다 8–12 높다.
- Next:
  - 회의 통째 결과(96172·96461)로 같은 크기 비교를 다시 맞춘다. 그쪽은 480 s 이하 조각이다.
- By: tskim

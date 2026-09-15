## [2026-09-15] query | AI Hub 라벨 품질 실측과 조치

- Changed: [[output-phase2-aihub-label-quality]] 신규; `vapasr/data/dialogue_refine.py`(채널 VAD 시각 보정)·`experiments/p2_refine.py`(보정+결손 의심·ASR 불일치 quarantine)·`experiments/p2_asr_check.py`(Qwen3-ASR 대조)·`slurm/p2_asr_check.sbatch`; `raw/sources/experiments/2026-09-15-phase2-sequence-samples/{asrcheck,refined}/`.
- Reason: 사용자 관찰 "aihub 전사 품질이 별로". 실측: 71631 라벨 끝 +0.86 s(p90 1.25 s)·발화 7 s, 단음절 결손 라벨; 134-1/134-2 조각 CER≥0.3 이 37/42 %(2 s 미만 ASR 잡음 바닥 30 % 감안).
- Next: pseudo-label 대체 여부 결정(사용자), 134-2 전량 ASR 대조 후 편입 판단, 창 격자 hop 2 s, 전량 Q0 에서 asr_check·refine 실행.
- By: tskim

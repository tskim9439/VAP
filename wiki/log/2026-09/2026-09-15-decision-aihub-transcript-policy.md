## [2026-09-15] decision | AI Hub 라벨 처리 정책

- Changed: [[decision-aihub-transcript-policy]] 신규; [[output-phase2-lane-plan]] §8 라벨 처리 문단; [[output-phase2-aihub-label-quality]] §3 상태; `experiments/p2_refine.py`(asr_disagree, 임계 0.2)·`vapasr/data/dialogue_interleave.py`(flatten mask_chunks)·`vapasr/data/dialogue_dataset.py`(untranscribed="mask")·`experiments/p2_show_sequence.py`(마스크 검사)·테스트.
- Reason: 사용자 청취 "라벨·ASR 반반" → 단일 ASR pseudo-label 대신 일치 발화만 전사 감독, 불일치 구간은 손실 마스크로 창 유지, 실내 원본·otoSpeech 를 전사 주력으로.
- Next: Q0 전량 빌드·정렬·ASR 대조·보정 순서로 SLURM 제출(사용자), 두 번째 KO ASR 심판 후보 조사.
- By: tskim

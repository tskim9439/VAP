## [2026-09-22] query | 새 SpeechLM Arrow ASR·정렬 및 tar 사전 인덱스 준비

- Changed: [[output-speechlm-asr-alignment-preparation]], `experiments/speechlm_asr_align.py`, `experiments/prepare_speechlm_asr_align.sh`, `experiments/speechlm-asr-align.md`, `vapasr/data/speechlm_inference.py`, Slurm 실행·제출 쉘, 관련 테스트.
- Reason: 새 Arrow에 원출력 ASR 및 forced alignment를 준비하고, 사용자 제안에 따라 tar 인덱싱을 CPU로 선행하여 GPU의 최초 오디오 읽기 대기를 줄인다.
- Next: 4,790개 tar 인덱스와 832행 CPU 준비 완료 확인 → 1노드 8 GPU 스모크 → 처리량·품질·중복 감사 후 전량 범위 결정. GPU 작업은 아직 제출하지 않음.
- By: tskim

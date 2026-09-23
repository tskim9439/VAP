## [2026-09-18] query | VibeVoice 평가 준비

- Changed: [[output-vibevoice-baseline-evaluation-runbook]], baseline 공통부·준비/추론/채점 실험 코드·전용 환경 준비·SLURM·테스트 신규 파일. 서버 전용 baseline 경로와 개발용 smoke 데이터.
- Reason: 사용자 요청에 따라 별도 환경에서 VibeVoice 1.5B zero-shot 평가를 준비하고, 기존 학습과 격리·공통 TN·순열 불변 점수·지연 의미를 명시했다. 추가 요청한 MeetEval 0.4.3을 전용 venv에도 설치했고 CPU 테스트 11/11 통과. 코드·가중치·import 준비 완료, GPU 잡은 제출하지 않았다.
- Next: 사용자가 GPU 1장 smoke 잡 제출. D1b 동일 manifest 어댑터·단어 지연·확대 평가 별도.
- By: tskim

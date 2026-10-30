## [2026-10-30] task | 외부 기준을 우리 AMI 조건으로 직접 재기 준비 — VibeVoice-ASR-Streaming-7B·Multitalker Parakeet v1(+ Sortformer v2·N3 Diarization)

- Changed:
  - 코드: `experiments/baseline_ami_ext.py`(외부 시스템을 같은 회의·오디오·참조·채점으로 — 행 꼴이 cst_stage3_eval 과 같다), `raw/sources/experiments/2026-10-30-ami-ext/ami_ext_task.sh`, `slurm/submit-evals.sh`(ext-vibevoice·ext-parakeet)
  - 맥 T5: `vapkt-models/multitalker-parakeet-streaming-0.6b-v1/`(@8749fc7, sha256 afaefe89…)·`diar_streaming_sortformer_4spk-v2/`(@84edd51, sha256 b371afce…) — HF 와 같다
  - 위키: [[output-ami-external-comparison-20261027]] "같은 조건으로 직접 재기" 절
- Reason:
  - 사용자 "지금 우리가 사용하고 있는 베이스라인 모델들의 성능 수치는 그냥 논문에 레포트된 수치이지?" → 맞다. "이 둘을 준비해줘".
  - 외부 수치는 출처마다 평가 단위(480 s 조각·세션·회의 통째)·정규화·참조가 달라 같은 시스템도 14 pt 넘게 흔들린다.
  - 맥 CPU 배선 점검: Parakeet + Sortformer v2·+ N3D 모두 AMI ES2002a 38 s 클립에서 4 명을 가르고 받아썼다.
- Next:
  - 사용자: T5 의 두 폴더를 `VAPKT-data/vapkt-baselines/` 아래 같은 이름으로 올린 뒤 `bash slurm/submit-evals.sh ext-parakeet`. VibeVoice 는 `… ext-vibevoice` 로 바로 낸다.
  - 재현 점검: VibeVoice 480 s 조각 대 보고 27.48 / 39.01, Parakeet + Sortformer v2 회의 통째 대 카드 21.26 / 37.44.
- By: tskim

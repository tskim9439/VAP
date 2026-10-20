## [2026-10-21] task | V3 화자 귀속 보정 — 조사·진단·설계·구현·검토(V3-p1 코드 완료, 기본 끔)

- Changed:
  - raw [[2026-10-21-v3-attribution]](새): README, `v3-design-spec.txt`·`v3-research-digest.txt`·`v3-evidence.md`, `diag/`(오류 해부 스크립트), `taxi_v3_probe.py`·`.sh`(<EOT> 기록·태그 bias·`--classify-onsets`), `build_tel.sh`
  - 코드(기본 끔, 끄면 바이트 동일): `vapasr/data/cst_dialogue.py`(C1 결정 가중·C5 <EOT> 문맥 제거), `vapasr/hf/{modeling_vapasr,packing,trainer,stage3,batch_decode}.py`, `vapasr/data/{sim_dialogue,phone_aug}.py`, `experiments/{stage3_train,sim_build_dialogues,cst_stage3_eval,cst_stage4_eval,cst_stage4_taxi,cst_attr_turns,sim_to_sessions}.py`, `slurm/submit-stage3-train.sh`, 시험 7 파일
  - [[output-v3-attribution-plan-20261021]](새), [[output-vibevoice-stream-comparison-20261020]](TAXI 분해), [[status]]
- Reason: 사용자 "화자 귀속 문제를 완화해서 VibeVoice 에 비해 TAXI 에서 우위를 점하고 싶은데, 최근 연구들을 조사하고 문제를 완화할 수 있는 방안은 모색해. 그래서 이를 적용한 모델이나 기법을 V3 로 명명해서 적용해줘.", "기존의 방식에서의 문제점 등도 함께 검토하고 발전시킬 수 있는 점이 있으면 개선하고"
- 결과:
  - TAXI 분해(같은 정규화): V1 cp 26.4 · agn 18.4 · 귀속 8.0, VibeVoice 스트리밍 20.5 · 20.1 · 0.4 → 차이는 전부 귀속이다.
  - 오류 해부: 틀린 단어의 89 % 가 턴 시작에서 놓친 교대 태그 뒤, toggle recall 87.4 %(VibeVoice 98.1 %), 턴 안 오교대 0.3 %, 첫 턴 86/86.
  - 원인 측정: V1 은 교차 언어 TAXI 에서 <EOT> 0 개다. 교차 언어 학습 창의 감독 없는 합성 <EOT> 가 teacher forcing 문맥에만 있어 학습·추론이 어긋난다.
  - 워크플로: 조사(8 에이전트), 설계 패널(16 — 설계 4·인용 검증 8·심사 3·종합 1, 인용 2 개 반박 → 고친 인용만 사용), 구현(4 worktree), 적대적 검토(18 — 결함 3 확인·수정, 낮음 4 수정).
  - 로컬 시험 679 통과(남은 실패 10·오류 1 은 V3 전과 같은 환경 문제).
- Next: mxc 본 checkout 반영(서버 판이 V3 전 판과 같은지 md5 확인 뒤) → dry-run·smoke → tel 셋·dev 빌드 → V1 dev 기준·C3 bias 튜닝(dev 만) → R1 `v3-p1`·R2 `v3-p1-nodata` 제출 명령(사용자 제출). 점프 호스트가 02:10 부터 offline 이라 대기 중.
- By: tskim

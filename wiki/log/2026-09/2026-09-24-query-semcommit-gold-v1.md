## [2026-09-24] query | SEM_END 골드셋 v1 · 턴 종료 <EOT> 통일

- Changed: `wiki/outputs/output-semcommit-gold-v1.md`(신규), `wiki/decisions/decision-semcommit-turn-eot-scope.md`(신규), `wiki/outputs/output-semcommit-v0.2-rack4-small.md`(골드 대조 안내),
  `raw/sources/experiments/2026-09-24-semcommit-gold-v1/`(지침·골드·주석·판정·통계·점수), 코드 `vapasr/data/semcommit_tokens.py`·`semcommit_dataset.py`·`vapasr/hf/{modeling_vapasr,commit_metrics,__init__,configuration_vapasr}.py`·
  `experiments/semcommit_{train,eval,viewer_bundle,build_words,gold,build_gigaspeech}.py`·`experiments/semcommit_teacher_gate.sh`·tests.
  rack4: `/data4/tskim/semcommit/{data/gold, labels/v0.2-gold, eval/v0.2-gold, gold/v1}`, 로컬 T5 `/Volumes/Samsung_T5/VAPKT-DB/semcommit-gold-v1`.
- Reason: 사용자 결정(<TURN_END> 는 <EOT> 로 통일, Phase 1 은 <SEM_END> 만, 본 학습은 전 언어·전 데이터·대화체 포함, 골드셋은 Claude 가 만든다).
  코드 통일(v0.2 체크포인트 호환 유지) 후 전수 주석 골드셋 v1(599 스트림, COMMIT 1,099) 을 만들고 v0.2 교사·모델을 골드와 대조:
  교사 A 재현율 0.14–0.31(정밀도 0.78–0.92), Stage A 후보 누락 21–32 %, N 정밀도 0.42–0.68; 모델은 한국어 P 0.95 / R 0.50 으로 교사보다 낫고, 대화체 영어는 거의 확정 못 함.
- Next: mxc 에서 `semcommit_teacher_gate.sh` 로 큰 교사 후보를 골드 대조(목표: A 정밀도 ≥ 0.9 유지하며 A 재현율 상향), 라벨링 레시피 수정(마지막 단어 후보·N 축소·스트림 길이),
  대화체(Switchboard·NIKL·71631) 포함 본 라벨링·학습. 필요하면 Switchboard 평가 분할로 gold v2.
- By: tskim

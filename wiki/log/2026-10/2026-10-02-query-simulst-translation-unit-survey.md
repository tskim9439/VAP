## [2026-10-02] query | 동시 번역 단위 조사 — <SEM_END> 는 번역 단위로 적합한가

- Changed:
  - `wiki/outputs/output-simulst-translation-unit-survey-20261002.md`(신규)
  - `wiki/concepts/conversational-simultaneous-s2tt.md`(열린 문제 갱신)
  - mxc `VAPKT-data/results/sem_unit_stats.py`(단위 실측)
- Reason: 사용자가 semcommit 라벨링을 일시 중단했다. 그리고 “지금 라벨링한 `<SEM_END>` 위치가 번역에도 적합한 인자인지, 실시간 번역을 어떤 단위로 해야 하는지” 조사를 요청했다.
- Next:
  - `<SEM_END>` 대 MU(번역 접두 일관성)·문맥 정렬 겹침 실험(KO↔EN, 교사 LLM 번역)
  - 번역 단위 라벨 정의, 한국어 연결어미 규칙 음성 재검토
- By: tskim

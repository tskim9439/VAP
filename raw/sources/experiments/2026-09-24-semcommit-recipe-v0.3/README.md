# semcommit 라벨링 레시피 v0.3 (2026-09-24, rack4)

gold v1(`../2026-09-24-semcommit-gold-v1/`) 대조에서 드러난 교사 라벨 문제 — Stage A 후보 누락, “판정자 전원 SAFE” 등급의 낮은 재현율, LLM 이 만든 N 의 낮은 정밀도 — 를 고친 레시피와 그 검증 기록.
해석은 `wiki/outputs/output-semcommit-recipe-v0.3.md`.

## 레시피
- 코드: `vapasr/data/semcommit_llm.py`(RECIPE_V3 = `semcommit-recipe-v0.3`: extra_candidates · candidate_features · response_unit · branch_of · rule_negatives · grade_v3 · build_labels_v3),
  `experiments/semcommit_teacher.py`(stageB/stageC `--extra-candidates`·`--only-extra`, grade `--recipe v0.3 --thresholds --neg-rules`), `experiments/semcommit_gold.py tune`(가지별 임계값 조정),
  `experiments/semcommit_teacher_gate.sh`(기본 RECIPE=v0.3). 커밋 78cf789(v0.3·v0.3.1), 41ce71c(v0.3.2), 711f6bc(v0.3.3).
- 후보 = Stage A ∪ `last` · `seg_end` · `punct_final`(참조 전사 . ? !). 추가분만 같은 판정자로 채점(Qwen3-8B·EXAONE-3.5-7.8B Stage B, Qwen3-8B Stage C; 프롬프트 `semcommit-prompt-v0.2+ab90c940`).
- 등급 = 가지(`punct` / `punct_resp` / `other`)별 P(SAFE) 평균 ≥ t_s ∧ P(REVISION) ≤ t_r → A, 사람 전사 표지·규칙 음성 → N, 나머지 B.
- 판(모두 같은 교사 행):

  | 판 | 조정 방식 | 영어(구두점 / 대답어 / 그 외) | 한국어 | 규칙 음성 |
  |---|---|---|---|---|
  | v0.3 | 언어 전체 정밀도 ≥ 0.90 (gold dev) | 전부 A / (구두점) / t_s 0.35 · t_r 0.2 | 전부 A / (구두점) / t_s 0.95 · t_r 0.02 | – |
  | v0.3.1 | 가지마다 정밀도 ≥ 0.90 | 전부 A / (구두점) / 끔 | 전부 A / (구두점) / 끔 | – |
  | v0.3.2 | 가지마다 + 대답어 가지 | 전부 A / 전부 A / 끔 | 전부 A / 끔 / 끔 | – |
  | **v0.3.3** | v0.3.2 + 규칙 음성(조정에서 그 자리 제외) | 같음 | 같음 | reply_prefix · conn_final · conn_mid |

  “전부 A” = t_s 0.0 · t_r 1.01. “(구두점)” = `punct_resp` 키가 없어 구두점 임계값으로 채점.
  현재 코드로 이전 판을 재현하려면 tune·grade 에 `--neg-rules ''` 를 준다(v0.3–v0.3.2 를 만들 때는 이 옵션이 없었다).

## 파일
- `v0.3/`, `v0.3.1/`, `v0.3.2/`, `v0.3.3/` — `thresholds.json`(grade `--thresholds` 입력), `thresholds.report.json`(dev / test P·R, 가지별 결과, 후보 재현), `labels-<set>.stats.json`(A/B/N·why·출처별 등급·입력 지문).
  set = 골드 파트(ls-test · gs-test · ks-eval · ks-long) + 학습(ls-train · ks-train). 라벨 jsonl 은 rack4 `/data4/tskim/semcommit/labels/<판>/`.
  v0.3.1 의 why 에는 꺼진 가지가 `v3_other+p_safe_low` 로 적혀 있다(등급은 같음; v0.3.2 부터 `v3_other+off`).
- `scores/teacher-<판>-<part>.json` — `semcommit_gold.py score-labels`(후보 재현·A 정밀도/재현율·N 정밀도·B 중 실제 확정). v0.2 기준선은 gold v1 폴더의 `scores/teacher-v0.2-*`.
  `scores/models-r3-<part>.json`(v0.3 라벨), `models-r5-<part>.json`(v0.3.3 라벨) — `semcommit_gold.py score-eval`(δ4, bias −2…2; r3 은 θ 0.2/0.35/0.5 도). r1/r2 는 gold v1 폴더의 `scores/models-*`.
- `analysis/` — 판단 근거가 된 분석과 그 스크립트(rack4 에서 실행):
  `branch-v0.3.txt`(v0.3 A 의 가지별 dev/test 한계 정밀도·그 외 가지 임계값 격자·구두점 끝/중간), `fp-v0.3.txt`(v0.3 오판 예시·구두점 가지 필터의 득실),
  `ko-mid-punct-v0.3.1.txt`(한국어 문장 중간 구두점 A 중 골드 NO/AMBIG — 대답어 머리 패턴), `resp-head-v0.3.2.txt`(대답어 가지 후보의 골드 라벨·등급),
  `neg-rules-candidates.txt`(규칙 음성 후보의 골드 분포·끝말별), `rule-negatives-v0.3.3.txt`(구현된 규칙의 골드 분포, dev/test),
  `models-compare.txt`(r1·r2·r3·r5 골드 대조 P/R/F1·지연·WER/CER, `compare_models.py`), `attrib-r3.txt`·`attrib-r5.txt`(모델 확정을 v0.3 라벨 종류별로 나눈 것 + 오판 예시, `attrib.py`; attrib-r5 끝에 r5 parity).
- `nopunct/` — 구두점 순환성 점검: 구두점을 숨긴 단어열로 같은 지침을 적용한 재주석(Claude; 영어 40 = ls-test 20 · gs-test 20, 한국어 80 = ks-eval 40 · ks-long 40),
  `compare.py`(골드와 대조; 인자 = gold v1 작업 폴더 — words-*.jsonl·out/gold-*.jsonl·nopunct/ 를 둔 곳), `compare.txt`(결과).
- `scripts/` — rack4 실행 기록: `recipe_v03.sh`(추가 후보 채점 → v0.3 조정·등급 → r3 학습·평가), `retune_v031.sh`·`retune_v032.sh`·`retune_v033.sh`(재조정·재등급, CPU),
  `r4_v032.sh`(학습 전에 멈추고 r5 로 대체), `r5_v033.sh`(v0.3.3 재등급 → r5 학습·평가), `gate_dryrun.sh`(교사 관문을 기존 교사 행으로 — Stage A/B/C 는 행을 미리 두고 건너뜀; 결과 = v0.3.3 점수·임계값과 일치).
- `SHA256SUMS`.

## 핵심 수치
교사 라벨 대 골드(전체 셋, A 정밀도 / 재현율):

| 파트 | v0.2 | v0.3 | v0.3.1 | v0.3.2 = v0.3.3 | 후보 재현 v0.2 → v0.3 계열 | v0.3.3 N (정밀도) |
|---|---|---|---|---|---|---|
| ls-test | 0.78 / 0.14 | 0.90 / 0.73 | 0.99 / 0.70 | 0.99 / 0.70 | 0.68 → 0.79 | 2 (1.00) |
| gs-test | 0.89 / 0.16 | 0.88 / 0.79 | 0.90 / 0.79 | 0.90 / 0.79 | 0.77 → 0.83 | 4 (1.00) |
| ks-eval | 0.92 / 0.31 | 0.90 / 0.85 | 0.90 / 0.82 | 0.96 / 0.82 | 0.79 → 0.94 | 316 (0.99) |
| ks-long | 0.86 / 0.27 | 0.86 / 0.80 | 0.87 / 0.78 | 0.89 / 0.78 | 0.74 → 0.84 | 353 (1.00) |

test 절반(임계값 선택에 안 쓴 쪽) v0.3.2–v0.3.3 A: 영어 P 0.955 / R 0.725, 한국어 P 0.926 / R 0.814.

모델 대 골드(δ4, 최고 F1 과 그 bias; r1 = v0.2 라벨, r3 = v0.3, r5 = v0.3.3, 같은 학습 설정):

| 파트 | r1 | r3 | r5 | WER/CER 기준 → r5 |
|---|---|---|---|---|
| ls-test | 0.35 (+2) | 0.78 (0) | 0.80 (+1, P 0.81 / R 0.78) | 5.27 → 5.18 % |
| gs-test | 0.11 (+2) | 0.74 (+2) | 0.71 (+2, 0.80 / 0.64) | 14.92 → 14.66 % |
| ks-eval | 0.75 (+2) | 0.82 (−2) | 0.84 (−2, 0.85 / 0.83) | 12.72 → 12.06 % |
| ks-long | 0.71 (+2) | 0.77 (−2) | 0.80 (−2, 0.77 / 0.84) | 10.96 → 10.83 % |

# SEM_END 골드셋 v1 (2026-09-24)

스트리밍 음성인식 전사에서 **`<SEM_END>`(의미 확정)가 들어가야 할 모든 단어 경계**를 전수 판정한 평가용 골드셋.
교사 LLM 라벨(Stage A 후보에 한정)과 달리 후보 누락까지 잡으므로, 교사 선정(관문)과 모델 평가의 기준으로 쓴다.
학습에는 쓰지 않는다(전부 표준 평가 분할 또는 학습에 안 쓴 코퍼스).

## 구성

| 파트 | 원천 | 성격 | 스트림 | 단어 경계 | COMMIT | AMBIG | 주석자 일치 κ(3 분류) | 두 주석자 COMMIT F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ls-test | LibriSpeech test-clean (연속 문장 스트림 ~30 s) | 영어 낭독 | 100 | 7,044 | 503 | 140 | 0.844 / 0.933 (A·B 반) | 0.951 / 0.990 |
| gs-test | GigaSpeech test (podcast·YouTube 한 화자 구간 8–25 s) | 영어 자발 발화 | 79 | 2,674 | 192 | 54 | 0.938 | 0.982 |
| ks-eval | KsponSpeech eval_clean (발화 1 개) | 한국어 자유 대화, 짧음 | 300 | 2,153 | 247 | 65 | 0.967 | 0.973 |
| ks-long | KsponSpeech eval_clean·eval_other (18 어절 이상 발화) | 한국어 자유 대화, 여러 절 | 120 | 2,674 | 157 | 54 | 0.947 | 0.977 |
| 합계 | | | 599 | 14,545 | 1,099 | 313 | | |

- ls-test·ks-eval 은 v0.2 소규모 실험의 평가 스트림과 **같은 스트림**(같은 words 행) — 그 실험의 교사 라벨·모델 출력을 곧바로 골드와 대조할 수 있다.
- gs-test 는 이번에 새로 만든 대화체 영어 몫(`experiments/semcommit_build_gigaspeech.py`): 비음성 태그 없음·끝 말고도 문장 끝이 있는 구간, podcast·YouTube 반반, 원본 오디오당 최대 2 구간. 16 kHz wav 는 `wav/`.
  본 학습의 대화체 영어(Switchboard)와는 도메인이 다르다(대리 지표).
- ks-long 은 KsponSpeech 평가 발화 중 ks-eval 300 개와 겹치지 않는 18 어절 이상 발화 120 개(각 분할 60) — 문장 중간 확정을 재기 위한 몫.

## 주석 절차
1. **지침** `GUIDELINE.md` — 계획서 §3(정의)·§5(Rule 1–15)를 주석용으로 푼 것: COMMIT = 완결 ∧ 안정, 확신 없으면 AMBIG, 나머지 NO.
2. **독립 주석 2 회**: 같은 패킷을 서로 다른 관점의 두 Claude 주석자가 따로 판정 — `rules`(지침을 문자 그대로 적용하는 언어학 주석자),
   `consumer`(확정 즉시 행동하는 뒷단 음성 에이전트 LLM 관점). 등급·교사 라벨은 보여 주지 않았다(블라인드). 참조 전사(영어 구두점 전사, 한국어 Kspon 원 전사)는 구조 파악용으로 제공.
3. **판정**: 두 주석이 다른 경계만 판정자(Claude)가 스트림 전체를 읽고 COMMIT / AMBIG / NO 로 결정(합계 197 경계). 판정 결과: `adj/*.json`(`_why` 에 이유).
4. **병합**: `python experiments/semcommit_gold.py merge --words … --ann rules consumer --adj … --out gold-….jsonl` — 일치는 그대로, 불일치는 판정대로.

### 주석 관례(주석자·판정자가 지침의 빈틈을 일관되게 채운 방식)
- 말끝 `-는데/-던데`(요 없음)는 AMBIG. 단 의문사 의문문(`언제 먹었는데?`)은 COMMIT.
- 한국어 문장 중간 `-는데` 뒤에 같은 문장이 이어지면 NO, 담화 표지(근데·그래서)로 새 단위가 시작하면 AMBIG.
- 연결어미(`-어서 -어가지고 -니까 -려고 -면 -고 때문에`)로 끝난 발화는 NO(원 전사에 마침표가 있으면 AMBIG). 완결된 문장 뒤에 붙은 이유·목적 절(`멋이 없어 ‖ 경차니까`)은 두 자리 다 COMMIT.
- 긴 발화 머리의 대답어(네·응·어·아니야·맞아)는 NO, 그 말만으로 끝나면 COMMIT. 완결 문장 뒤에 붙은 한두 단어(`… 있지 ‖ 뭐`)는 AMBIG.
- 영어: 완결 절 뒤 `so/and/but` + 자기 주어를 가진 새 절이면 접속사 앞 COMMIT, 같은 술어·목적어가 이어지면 NO. 완결 절 뒤 수식 관계절·분사구·to 부정사구는 AMBIG.
  낭독체 직접 인용 뒤 화자 표지(`… ‖ said sir ferdinando`)는 표지 뒤에서 확정.

## 파일
- `words-<part>.jsonl` — 스트림 단어 색인(semcommit words 형식: 단어·종료 시각·토큰·segment 오디오 경로·영어 `pnc_text`). **골드 번호는 이 파일의 단어 번호**다.
  segment 경로는 rack4 기준(LibriSpeech `/data5/LibriSpeech`, Kspon `/data4/tskim/DBs/KsponSpeech/extracted`, GigaSpeech `wav/`) — 다른 서버에서 모델을 평가할 때는 `semcommit_eval.py --path-remap`.
- `gold-<part>.jsonl` — `{id, set, lang, n_words, commit:[i], ambig:[i], how:{i: agree|adjudicated}}`. 목록에 없는 경계 = NO.
- `ann/<packet>.{rules,consumer}.json` — 두 주석자의 원 판정(이유 포함), `adj/<packet>.json` — 판정, `merge-<packet>.stats.json` — 일치도.
  패킷: ls-A·ls-B(ls-test 앞·뒤 50), gs, ks-short(= ks-eval), ks-long.
- `scores/` — 골드 대조 점수: 교사 라벨 v0.2(`teacher-v0.2-<part>.json`)·레시피 v0.3 계열(`teacher-v0.3{,.1,.2,.3}-<part>.json`),
  모델 r1/r2(`models-<part>.json`, v0.2 라벨)·r3(`models-r3-<part>.json`, v0.3 라벨)·r5(`models-r5-<part>.json`, v0.3.3 라벨). `SHA256SUMS`(words·gold 파일).

## 쓰는 법
```bash
# 교사 라벨(labels.jsonl, A/B/N) 대 골드: 후보 재현(cand_recall)·A 정밀도/재현율·N 정밀도·B 중 실제 확정 비율
python experiments/semcommit_gold.py score-labels --words words-ks-long.jsonl --labels labels-ks-long.jsonl --gold gold-ks-long.jsonl
# 모델 스트리밍 방출(semcommit_eval.py 의 *.streams.jsonl) 대 골드: 설정별 P/R/F1·PCR_gold·지연
python experiments/semcommit_gold.py score-eval --words words-ks-long.jsonl --gold gold-ks-long.jsonl --streams r1-d4-ks-long.streams.jsonl
# 교사 관문(기본 레시피 v0.3): 후보 교사로 네 파트 라벨링 → 골드 dev 절반에서 가지별 임계값 조정 → 등급(규칙 음성 포함) → 골드 대조 (GPU 1 장)
# 본 라벨링은 OUT/thresholds.json 을 teacher.py grade --recipe v0.3 --thresholds 로 쓴다
GOLD=<이 디렉터리> OUT=<출력> A=<경로:kind:이름> B="<경로:kind:이름> <경로:kind:이름>" C=<경로:kind:이름> bash experiments/semcommit_teacher_gate.sh
```
P = 골드 COMMIT 에 맞은 확정 ÷ (확정 − 골드 AMBIG 자리의 확정). 단어 중간·첫 단어 전·같은 자리 중복 확정은 오류로 센다.

## 기준선(참고): v0.2 교사 라벨과 소규모 모델 r1/r2(δ4)

| 파트 | 교사 후보 재현 | 교사 A 정밀도 | 교사 A 재현율 | 교사 N 정밀도 | r1 bias 0 P / R / F1 | r1 bias +2 P / R / F1 | r2 bias +2 P / R / F1 |
|---|---:|---:|---:|---:|---|---|---|
| ls-test | 0.68 | 0.78 | 0.14 | 0.48 | 0.81 / 0.09 / 0.16 | 0.70 / 0.24 / 0.35 | 0.72 / 0.29 / 0.41 |
| gs-test | 0.77 | 0.89 | 0.16 | 0.42 | 1.00 / 0.01 / 0.01 (확정 1 개) | 0.79 / 0.06 / 0.11 | 0.83 / 0.10 / 0.18 |
| ks-eval | 0.79 | 0.92 | 0.31 | 0.60 | 0.95 / 0.50 / 0.65 | 0.84 / 0.67 / 0.75 | 0.78 / 0.68 / 0.73 |
| ks-long | 0.74 | 0.86 | 0.27 | 0.68 | 0.82 / 0.48 / 0.60 | 0.68 / 0.74 / 0.71 | 0.62 / 0.75 / 0.68 |

레시피 v0.3.3(같은 8B 교사, 2026-09-24)과 그 라벨로 학습한 r5: 교사 A 정밀도 / 재현율 ls-test 0.99 / 0.70 · gs-test 0.90 / 0.79 · ks-eval 0.96 / 0.82 · ks-long 0.89 / 0.78,
r5 최고 F1 ls-test 0.80 · gs-test 0.71 · ks-eval 0.84 · ks-long 0.80 (δ4, bias −2…+2) — VAPKT `wiki/outputs/output-semcommit-recipe-v0.3.md`.

## 한계
- **사람이 아니라 Claude 가 주석·판정했다.** 두 관점의 일치(κ 0.84–0.97)는 같은 모델 계열끼리의 일치라 신뢰도의 상한으로 읽어야 한다. 사람 표본 검수는 하지 않았다.
- 대화체 영어는 GigaSpeech(podcast·YouTube) 대리 — 본 학습의 Switchboard(전화 대화)와 다르다. mxc 에서 Switchboard 평가 분할로 같은 절차의 gold v2 를 만들 수 있다.
- 한국어는 KsponSpeech 한 코퍼스뿐(NIKL·71631 대화는 없음).
- 오디오 없이 전사만 보고 판정했다(억양·쉼은 반영 안 됨). 전사 오류가 있는 곳은 말의 구조로 판단했다.

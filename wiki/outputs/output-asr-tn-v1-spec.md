---
type: output
status: active
created: 2026-09-07
updated: 2026-09-07
summary: asr-tn-v1.0.0 동결 규약(2026-09-07 frozen) — lexical/display 이중 표기, 숫자 범위, quarantine와 버전 조건
sources:
  - [[source-asr-output-style-probe]]
  - [[source-qwen3-asr]]
  - [[source-nemotron-3-5-asr-streaming]]
  - [[asr-text-normalization]]
---

# `asr-tn-v1.0.0` 텍스트 정규화 규약

## 문서 상태와 범위

- **규약 ID:** `asr-tn-v1.0.0`
- **상태:** 동결 후보(candidate). 아래 lexical freeze와 display freeze를 각각 통과하면
  전체 규약을 frozen으로 전환한다.
- **적용 범위:** Stage 2의 LibriSpeech 960 h와 KsponSpeech 약 970 h 학습 타깃,
  영어 대소문자·문장부호 출력, LibriSpeech dev/test 및 KsponSpeech dev/eval 채점.
- **비적용 범위:** otoSpeech, AI Hub, NIKL 및 이후 추가 DB. 이들은 corpus parser와
  audit를 추가한 다음 이 규약에 편입하거나 새 버전을 만든다.

이 문서는 정규화 출력의 정본이다. Python 라이브러리의 기본 동작은 정본이 아니며,
`num2words`를 포함한 구현은 이 문서의 golden 출력을 만족할 때만 사용할 수 있다.

규범 키워드 `MUST`, `MUST NOT`, `SHOULD`, `MAY`는 각각 필수, 금지, 권장, 선택을 뜻한다.

## 목적

Qwen3-ASR과 Nemotron RNN-T의 실측 출력은 EN 숫자를 단어로, KO 숫자를 한글 읽기로
표기했다. Qwen3-ASR은 영어 대소문자와 문장부호도 적극 생성했다
([[source-asr-output-style-probe]]). 최종 제품 출력에서도 이 표기가 필요하므로 영어
대문자·문장부호를 제거 기능이 아니라 별도의 필수 출력 능력으로 다룬다.

다만 LibriSpeech의 정규화 transcript에는 신뢰할 수 있는 문장 대소문자와 문장부호가
없다. 이를 규칙으로 지어내어 정답으로 쓰면 음성·원문 근거가 없는 label noise가 된다.
따라서 이 규약은 spoken content와 최종 표기를 다음 두 view로 분리한다.

## 두 가지 영어 텍스트 view

| 필드 | 목적 | 표기 | 사용처 |
|---|---|---|---|
| `lexical_text` | 발화 내용과 방출 시점 학습 | 소문자, 문장부호 없음, 단어 내부 apostrophe 유지 | 강제 정렬, 주 streaming loss, RNN-T 비교 WER |
| `display_text` | 사용자에게 보이는 최종 전사 | 문장·고유명사 대소문자, 지원 문장부호 유지 | display 보조 loss, 실제 출력 품질 평가 |

영어 manifest 행은 원문, `lexical_text`, `display_source`를 반드시 보존해야 한다
(MUST). 신뢰할 수 있는 표기 label이 있는 행은 `display_text`와
`display_confidence`도 보존해야 한다(MUST). label이 없으면 `display_source: none`으로
기록한다. `display_text`가 없는 행은 display loss를 mask해야 하며, 무구두점
`lexical_text`를 display 정답으로 간주하면 안 된다(MUST NOT).

`display_source`의 허용값과 우선순위는 다음과 같다.

1. `gold`: 사람이 검수한 시간 정렬 원문 또는 원 corpus의 신뢰 가능한 표기
2. `reconstructed`: 원 저작물과 transcript를 정확히 word-align해 복원한 표기
3. `pseudo-qwen`: Qwen3-ASR 출력에서 표기만 이전한 pseudo-label
4. `none`: 검증 가능한 display label 없음

`pseudo-qwen`은 그 출력을 lexical 채점 규칙으로 정규화한 결과가 신뢰하는
`lexical_text`와 정확히 같을 때만 허용한다(MUST). 단어가 하나라도 치환·삭제·삽입된
출력은 표기 교사로 사용하지 않고 `none` 또는 quarantine으로 기록한다. 이 필터는
Qwen의 인식 오류를 정답 단어로 복사하지 않고 대소문자·문장부호 정보만 받기 위함이다.

## 공통 불변 조건

1. 원 전사는 수정하지 않고 manifest의 원문 필드 또는 원 DB에 보존해야 한다(MUST).
2. 모든 문자열은 Unicode NFKC를 먼저 적용해야 한다(MUST).
3. CR/LF, tab 및 연속 공백은 ASCII 공백 하나로 합치고 양끝 공백을 제거해야 한다(MUST).
4. 정규화는 결정적이고 멱등이어야 한다(MUST): `target(target(x)) == target(x)`.
5. 지원하지 않는 숫자 형식을 일부만 치환해서는 안 된다(MUST NOT). 전체 항목을
   quarantine하거나 corpus parser의 명시적 규칙으로 처리해야 한다.
6. 정규화 후 빈 문자열, control character, 허용 문자 밖 문자가 남은 항목은
   학습에 넣지 않고 quarantine해야 한다(MUST).
7. 학습 타깃 정규화와 채점 정규화는 별도 함수로 유지해야 한다(MUST).
8. `lexical_text` 또는 그 Qwen tokenizer token ID가 바뀌면 기존 alignment를
   재사용하면 안 된다(MUST NOT).
9. `display_text`의 각 label은 출처를 가져야 하며, 출처 없는 대소문자·문장부호를
   deterministic rule로 생성해 gold로 기록하면 안 된다(MUST NOT).
10. display supervision 유무가 lexical loss의 포함 여부나 가중치를 바꾸면 안 된다
    (MUST NOT). 두 목적의 sampling과 loss weight는 별도 설정으로 기록한다.

`quarantine`은 원 파일 삭제를 뜻하지 않는다. 학습 manifest에서 제외하되 ID, 원문,
제외 이유와 집계는 versioned audit 산출물에 보존하는 절차다.

## 영어 lexical 학습 타깃

### 기본 표면 규칙

처리 순서는 다음과 같다.

1. 알려진 ASR 언어·특수 태그를 제거한다.
2. NFKC 적용 후 소문자로 바꾼다.
3. `’`를 ASCII `'`로 통일하고 단어 내부 아포스트로피만 유지한다.
4. 지원되는 숫자 패턴을 canonical spoken form으로 바꾼다.
5. 하이픈, slash, en dash, em dash와 나머지 구두점을 공백으로 바꾼다.
6. 공백을 축약한다.

정규화된 EN `lexical_text`의 허용 문자는 `a-z`, 단어 내부 `'`, ASCII 공백이다.

| 입력 | canonical target |
|---|---|
| `DON'T STOP!` | `don't stop` |
| `I’VE TWENTY-ONE BOOKS` | `i've twenty one books` |
| `"HELLO"—SHE SAID` | `hello she said` |

이 소문자·무구두점 표기는 최종 출력 규약이 아니라 강제 정렬과 내용 인식을 위한
lexical view다.

## 영어 display 학습 타깃

### 지원 표기

`display_text`는 v1에서 다음을 지원한다.

- 문장 첫 단어와 고유명사의 대소문자
- source가 명시한 all-caps 약어
- 문장 종결 부호 `.`, `?`, `!`
- 쉼표 `,`
- 단어 내부 apostrophe `'`

typographic apostrophe `’`는 `'`로 통일한다. 여러 문장부호가 연속된 장식 표기,
semicolon, colon, quote, 괄호, ellipsis, en/em dash는 v1 display target에서 지원하지
않는다. 허용 목록 밖 문장부호는 source별 parser가 제거 또는 quarantine하되 그
선택을 audit에 기록해야 한다. 지원 범위를 넓히려면 TN minor 버전을 올린다.

| 입력 또는 근거 | `lexical_text` | `display_text` |
|---|---|---|
| 검수 원문 `Don't stop!` | `don't stop` | `Don't stop!` |
| 검수 원문 `Alice has five books.` | `alice has five books` | `Alice has five books.` |
| LibriSpeech label `ALICE HAS FIVE BOOKS`만 존재 | `alice has five books` | 없음 (`display_source: none`) |

### 학습과 streaming 방출

Stage 2의 주 streaming target과 forced alignment는 `lexical_text`를 사용한다. 영어
대소문자·문장부호는 prompt-conditioned display mode 또는 명시적으로 가중한 보조
목적으로 학습한다. 두 모드의 special token, sampling ratio, loss weight는 checkpoint
metadata에 기록해야 한다(MUST).

display mode에서는 같은 thinker가 대소문자와 문장부호를 직접 생성해야 한다. 별도
후처리 punctuation model은 비교용 baseline으로 둘 수 있지만 이 계약의 충족으로
간주하지 않는다. `display_text`와 `lexical_text`의 lexical form이 같으므로 단어를
일대일 대응시킨 뒤, 대소문자가 포함된 wordpiece는 해당 lexical word의 alignment와
일반 `delta`를 상속한다. 문장부호는 앞 단어를 host로 삼아
`host_word_end + delta_punct`에 배치한다. `delta_punct`는 chunk 단위 설정값이며
학습·평가 metadata에 기록한다. 문장 종결 부호는 stream flush에서도 방출할 수 있다.

대문자는 해당 단어의 표기 속성이므로 별도 acoustic evidence timestamp를 만들지 않는다.
문장부호는 뒤 문맥이 필요할 수 있으므로 앞 단어의 evidence time에 억지로 묶거나
lexical `viol80` 관문을 적용하면 안 된다(MUST NOT). 대신 다음을 별도로 보고한다.

- punctuation class별 precision, recall, F1
- capitalization accuracy와 case-sensitive WER
- 앞 단어 evidence end부터 문장부호 방출까지의 delay p50/p90/p99
- 이미 확정한 표기를 고칠 수 있는 decoder라면 revision rate와 revision delay

초기 지연 budget은 고정 규약이 아니라 실험 변수다. 80 ms chunk 기준 2–6 chunk
(160–480 ms)를 먼저 측정하되, lexical token 지연과 섞어 단일 수치로 보고하지 않는다.

### LibriSpeech corpus 정책

LibriSpeech transcript는 숫자가 이미 발음 단어로 전사되어 있어야 한다. raw label에
ASCII digit가 남으면 generic number converter로 추측하지 않고 해당 행을 quarantine한다
(MUST). 따라서 960 h audit의 LibriSpeech 잔존 숫자 목표는 0이다.

이 제한은 `1984`가 `nineteen eighty four`인지 `one thousand nine hundred eighty four`인지,
`007`이 `zero zero seven`인지 `seven`인지 음성 없이 결정할 수 없기 때문이다.

## 영어 숫자 canonical form

아래 패턴은 **채점 입력**과 숫자 표기를 명시적으로 허용한 미래 corpus parser에서만
지원한다. LibriSpeech raw target에 대한 숫자 허용을 뜻하지 않는다.

### 지원 패턴

| 유형 | 허용 문법 | canonical 출력 | 제한 |
|---|---|---|---|
| 정수 | `0` 또는 nonzero digit로 시작하는 정수 | `101` → `one hundred one` | 0–999,999,999 |
| 천 단위 쉼표 | `1,000`, `12,345` | `twelve thousand three hundred forty five` | 정확한 3자리 grouping만 |
| 소수 | `<정수>.<1–6 digits>` | `1.05` → `one point zero five` | 소수부 0과 끝 0 보존 |
| 퍼센트 | `<정수 또는 소수>%` | `1.05%` → `one point zero five percent` | 숫자와 `%` 사이 공백 0–1개 |
| USD 정수 | `$<정수>` | `$1` → `one dollar`, `$5` → `five dollars` | 소수 통화 미지원 |
| 서수 | `<정수><st|nd|rd|th>` | `21st` → `twenty first` | suffix가 수와 일치해야 함 |

Canonical 영어 수사는 American-style을 사용하며 hundred와 뒤 단위 사이에 선택적
`and`를 삽입하지 않는다. 예: `101`은 `one hundred one`이다. 변환 결과의 합성
하이픈은 공백으로 바꾼다.

소수는 binary float로 변환하면 안 된다(MUST NOT). 소수점 오른쪽을 문자 단위로 읽어
`1.10 → one point one zero`를 보장해야 한다.

서수 suffix는 11th, 12th, 13th 예외와 마지막 digit 규칙을 검증한다. `11st`, `22th`는
부분 변환하지 않고 미지원 형식으로 처리한다.

### 미지원 패턴

다음은 v1에서 발음을 추측하지 않는다. 학습 target에 나타나면 quarantine하고,
채점에서는 부분 변환 없이 원형을 남겨 불일치로 센다.

| 유형 | 예 | 이유 |
|---|---|---|
| 선행 0 정수 | `007`, `0012` | digit reading과 cardinal이 모호함 |
| 연도 | `1984`, `2026` | year reading과 cardinal이 모호함 |
| 날짜 | `09/07/2026`, `Sep. 7` | 순서·서수·locale 의존 |
| 시각 | `10:30`, `7 pm` | 문맥별 읽기 차이 |
| 전화·계좌·식별번호 | `010-1234-5678`, `A320` | digit/letter 단위 읽기 필요 |
| 부호 있는 수 | `-5`, `+3` | minus/negative/plus 문맥 의존 |
| 범위·비율 | `3-5`, `3:1` | to, through, dash, colon이 모호함 |
| 분수·혼합수 | `1/2`, `2 1/2` | half/one half 등 선택 필요 |
| 소수 통화 | `$5.50` | `five fifty`와 `five dollars fifty cents`가 모호함 |
| 기타 통화 | `€5`, `£10`, `₩1000` | 통화 단위·복수형 정책 미정 |
| 숫자+단위 결합 | `5kg`, `10km`, `3G` | 단위 발음과 token 경계 미정 |
| 수식·과학 표기 | `10^3`, `1e-5`, `2×4` | 연산자 읽기 미정 |
| Roman numeral | `IV`, `Chapter X` | 서수·cardinal·고유명사 모호함 |
| 10억 이상 또는 소수부 7자리 이상 | `1000000000`, `0.1234567` | v1 지원 범위 밖 |

연도처럼 겉모양만으로 일반 정수와 구분할 수 없는 패턴은 corpus가 숫자를 허용하는
경우에도 주변 문맥 또는 corpus metadata로 먼저 분류해야 한다. 분류 근거가 없으면
quarantine한다.

## 한국어 학습 타깃

### KsponSpeech 이중표기

KsponSpeech의 정확한 `(철자)/(발음)` 쌍마다 다음 규칙을 독립 적용한다.

1. 왼쪽 철자형에 ASCII digit 또는 Latin letter가 하나라도 있으면 오른쪽 발음형을
   선택한다(MUST).
2. 그렇지 않으면 왼쪽 철자형을 선택한다(MUST).
3. 선택한 오른쪽 발음형에도 digit가 남으면 해당 행을 quarantine한다(MUST).
4. 중첩 괄호, 닫히지 않은 괄호, 빈 왼쪽/오른쪽처럼 parser가 완전한 쌍으로 인식하지
   못하는 표기는 quarantine한다(MUST).

| raw | canonical target |
|---|---|
| `(100명)/(백 명)` | `백 명` |
| `(0.1프로)/(영 점 일 프로)` | `영 점 일 프로` |
| `(TV)/(티비)` | `티비` |
| `(3G)/(쓰리 지)` | `쓰리 지` |
| `(서울)/(서울)` | `서울` |

### 숫자·Latin 지원 범위

| 유형 | v1 처리 | 예 |
|---|---|---|
| 이중표기 안 숫자 | 오른쪽 발음형 선택 | `(26일)/(이십육 일)` → `이십육 일` |
| 이중표기 안 Latin | 오른쪽 발음형 선택 | `(PC)/(피씨)` → `피씨` |
| 이미 한글인 수사 | 그대로 유지 | `백 명`, `십만 원` |
| 이중표기 밖 Arabic digit | quarantine | `100명이 왔어` |
| 이중표기 밖 Latin | 대문자로 통일해 유지 | `tv를 봤어` → `TV를 봤어` |

독립 Latin은 v1의 명시적 예외다. 임의 사전으로 `TV→티비`를 만들어내지 않는다
(MUST NOT). 전체 audit에서 partition별 비율과 상위 100개 표현을 기록한다. 이후
발음 근거가 있는 lexicon을 채택하면 target이 달라지므로 새 TN 버전으로 다룬다.

### 표지·구두점·공백

- 단독 noise 표지 `b/`, `l/`, `o/`, `n/`, `u/`는 제거한다.
- `아/`, `그/`처럼 발화된 filler의 slash만 제거하고 단어는 유지한다.
- `+`, `*`와 구두점은 제거한다.
- 간투사와 반복어 자체는 유지한다.
- 원문의 단어 사이 띄어쓰기는 유지하고 연속 공백만 축약한다.
- standalone Latin은 한글 읽기로 바꾸지 않되 대문자로 통일한다.

정규화된 KO target의 허용 문자는 한글 음절·자모, Latin letter와 ASCII 공백이다.
digit, underscore, 다른 문자권 문자 또는 control character가 남으면 quarantine한다.

## 채점 규약

### 영어 lexical 평가

참조와 가설 양쪽에 같은 NFKC, 태그 제거, 소문자, 아포스트로피, 구두점 규칙을
적용하고 공백 분리 WER를 계산한다. 지원 숫자 패턴은 위 canonical form으로 바꾸되,
미지원 패턴을 일부만 변환하지 않는다.

`and`가 실제 발화·참조에 단어로 있으면 임의로 제거하지 않는다. 숫자 가설 `101`을
canonicalize할 때만 `one hundred one`을 생성한다.

이 normalized WER가 RNN-T 대조군 및 데이터 scaling curve의 1차 지표다. display 기능을
추가해도 이 지표의 정의를 바꾸지 않아야 기존 실험과 비교할 수 있다(MUST).

### 영어 display 평가

display label이 있는 고정 평가 subset에서 다음 지표를 함께 보고한다.

- case-sensitive WER와 exact sentence match
- capitalization accuracy
- punctuation micro/macro F1 및 `.`, `?`, `!`, `,` class별 precision/recall/F1
- punctuation을 독립 token으로 포함한 display error rate
- streaming punctuation delay p50/p90/p99

display 평가 전에 소문자화하거나 문장부호를 제거하면 안 된다(MUST NOT). 다만
`display_source`별 결과를 분리해 `gold`와 pseudo-label 점수를 같은 품질로 해석하지
않는다. display label이 없는 LibriSpeech 공식 dev/test만으로는 이 능력을 판정할 수
없으므로 별도의 고정 display 평가 subset이 필요하다.

### 한국어

- `CER-official`: 알려진 태그와 구두점을 제거하고 원문 공백을 포함해 계산한다.
- `CER-nospace`: 같은 문자열에서 공백을 제거해 내부 모델 선택에 사용한다.
- 숫자 가설을 한글 읽기로 추측 변환하지 않는다. 가설 `100명`과 참조 `백 명`은
  다른 출력으로 채점한다.
- standalone Latin도 한글로 변환하지 않는다.

학습 타깃의 강한 정규화를 위해 `score_*`를 호출하면 안 된다(MUST NOT).

## Golden regression 최소 집합

구현은 최소한 다음 검사를 통과해야 한다.

```text
EN target
DON'T STOP!                  => don't stop
I’VE TWENTY-ONE BOOKS        => i've twenty one books

EN display
Don't stop!                  => lexical: don't stop | display: Don't stop!
Alice has five books.        => lexical: alice has five books | display: Alice has five books.
Qwen: "Alice has six books." => reject when trusted lexical is: alice has five books

EN supported numeric canonicalization
0                            => zero
101                          => one hundred one
12,345                       => twelve thousand three hundred forty five
1.05                         => one point zero five
1.10                         => one point one zero
1.05%                        => one point zero five percent
$1                           => one dollar
$5                           => five dollars
21st                         => twenty first

EN unsupported — no partial conversion
007, 1984(year context), 10:30, 1/2, $5.50, 5kg, A320, 11st

KO target
(100명)/(백 명)              => 백 명
(TV)/(티비)                  => 티비
(서울)/(서울)                => 서울
b/ 안녕하세요               => 안녕하세요
아/ 그게 아니고              => 아 그게 아니고
tv를 봤어                    => TV를 봤어
100명이 왔어                 => quarantine
```

각 canonical target은 두 번 정규화해도 동일해야 하며, 고정한 Qwen tokenizer에서
token ID sequence도 golden snapshot과 같아야 한다.

## Audit 산출물과 동결 관문

전체 1,930 h transcript audit는 corpus·partition별로 다음을 기록한다.

- 전체/정규화/제외 행 수와 음성 시간
- empty target, malformed dual notation, 잔존 digit와 허용 문자 위반
- 숫자·Latin 이중표기 선택 건수
- standalone Latin 행 비율과 상위 100개 표현
- 정규화 전후 길이비와 서로 다른 원문의 target collision
- idempotence 위반과 tokenizer 오류
- 목적 표본 review TSV 및 quarantine ID 목록
- display label 보유 시간·행 수와 `display_source`별 coverage
- pseudo-label의 lexical exact-match 통과·거절 수와 punctuation class 분포
- 대문자 종류(sentence initial, proper noun, all-caps)별 표본 수

동결 관문은 alignment에 영향을 주는 lexical 계약과 display 학습 계약으로 나눈다.

### Lexical freeze — 1,930 h alignment 시작 조건

1. `num2words` 또는 숫자 변환 의존성이 설치되지 않았을 때 즉시 실패한다.
2. Golden 문자열 및 tokenizer-ID test가 모두 통과한다.
3. LibriSpeech raw 잔존 digit가 0이다.
4. KsponSpeech 이중표기 밖 digit와 malformed 표기가 모두 quarantine되며 누락이 0이다.
5. 정규화 후 빈 target과 idempotence 위반이 0이다.
6. KsponSpeech_01–05 각각의 목적 표본을 사람이 검토한다.
7. 기존 230 h target/token ID와의 diff가 전부 설명된다.
8. manifest metadata에 아래 lexical fingerprint가 기록된다.

위 여덟 조건을 통과하면 1,930 h forced alignment를 시작할 수 있다. display label은
별도 token timestamp를 만들지 않으므로 이후 display 정책 변경은 이 alignment를
무효화하지 않는다.

### Display freeze — 1,930 h full-FT 시작 조건

1. 영어 display 평가 subset과 label provenance를 고정하고 사람이 표본 검토한다.
2. `pseudo-qwen`은 lexical exact-match 필터의 통과분만 남고, display label이 없는 행의
   display loss mask가 regression test로 확인된다.
3. base Qwen과 학습 checkpoint의 display-retention 평가를 동일 subset에서 수행할
   수 있어야 한다.
4. display sampling ratio, loss weight와 metadata를 고정한다.
5. manifest metadata에 아래 display fingerprint가 기록된다.

```json
{
  "textnorm_version": "asr-tn-v1.0.0",
  "textnorm_sha256": "...",
  "lexical_manifest_sha256": "...",
  "display_policy_sha256": "...",
  "display_source_manifest_sha256": "...",
  "numeric_backend_version": "...",
  "tokenizer_json_sha256": "...",
  "source_transcript_sha256": "...",
  "quarantined_ids_sha256": "...",
  "created_from_git_commit": "..."
}
```

Aligner는 lexical fingerprint가 없거나 다르면 시작 전에 실패해야 한다. 학습기는
lexical 및 display fingerprint를 모두 검사한다. 동결 후 alignment 경로는
`align-asr-tn-v1/`처럼 규약 ID를 포함해야 한다.

## 버전 정책

- **Patch (`v1.0.x`)**: target 문자열과 token ID가 한 건도 바뀌지 않는 문서·오류 메시지·audit 개선.
- **Minor (`v1.x.0`)**: 새 corpus parser, 새 숫자 패턴 또는 새 display 문장부호 추가.
  기존 corpus의 `lexical_text`가 한 건이라도 바뀌면 새 manifest와 변경 ID alignment가
  필요하다. display-only 변경은 새 학습 manifest와 fingerprint만 필요하다.
- **Major (`v2.0.0`)**: 기존 lexical 숫자 읽기, Latin 처리, 띄어쓰기 또는 두-view
  의미처럼 기존 target을 광범위하게 바꾸는 규약 변경.

어떤 버전 변경이든 old manifest·alignment를 덮어쓰지 않는다. `lexical_text` 또는 그
token ID가 바뀐 ID는 반드시 새 alignment를 만든다. `display_text`만 바뀌면 alignment와
feature cache는 재사용할 수 있지만 학습 manifest의 display fingerprint는 바뀌어야 한다.
feature cache는 오디오가 같으면 항상 재사용할 수 있다.

## 현재 구현 상태와 남은 확인

2026-09-07 작업 트리에는 `num2words==0.5.14` 설치 고정, import 실패 시 즉시 오류,
KO standalone Latin 대문자 통일, 잔존 KO digit 학습 제외와 manifest version 필드의
부분 구현이 들어와 있다. 그러나 다음 차이가 남아 있으므로 frozen으로 판정하지 않는다.

- 코드의 `TEXTNORM_VERSION`은 `1.0`이지만 fingerprint의 canonical ID는
  `asr-tn-v1.0.0`으로 통일해야 한다.
- 현재 소수 변환은 binary float를 사용하므로 `1.10`의 끝 0을 보존하지 못한다.
- 현재 USD 변환은 `$1`에도 `dollars`를 붙이며, 서수는 1–2자리만 처리한다.
- 미지원 숫자를 전체 항목 단위로 검출하기 전에 지원 regex가 부분 변환할 수 있다.
- 코드 주석이 참조하는 `wiki/decisions/decision-textnorm-v1` 페이지는 아직 존재하지 않는다.
- `num2words==0.5.14` English backend가 이 문서의 American-style golden 출력을
  그대로 만족하는지 아직 검증하지 않았다.
- `1984` 같은 네 자리 숫자를 일반 정수와 연도로 분류할 corpus metadata가 현재 없다.
- KsponSpeech 전체에서 standalone Latin 1.18% 추정치가 유지되는지 확인하지 않았다.
- 이 문서의 quarantine 정책이 기존 230 h target을 몇 건 바꾸는지 아직 측정하지 않았다.
- manifest에 `lexical_text`, `display_text`, `display_source`, `display_confidence`가 아직
  구현되지 않았다.
- display mode, masked auxiliary loss, sampling ratio와 checkpoint metadata 규약이
  아직 구현되지 않았다.
- 영어 display 평가 subset 및 capitalization·punctuation·지연 지표가 아직 없다.
- Qwen pseudo-label을 lexical exact-match로 검증·채택하는 도구와 audit가 아직 없다.

따라서 이 규약은 문서 정본이지만 아직 **frozen 판정 전**이다. 다음 작업은 전체
transcript audit와 영어 display label audit 도구를 작성해 위 미확정 수치를 채우는 것이다.

## 동결 판정 (2026-09-07)

구현 `vapasr/data/textnorm.py` + `vapasr/data/kspon.py` (`textnorm_sha256 62e11231a2c4…`, num2words 0.5.14, tokenizer `66915f2a66d1…`).
audit 결과는 [[source-asr-tn-v1-audit]]: LibriSpeech 잔존 digit 0, KsponSpeech 620,000 중 68 quarantine(전부 이중표기 밖 digit, malformed 0),
idempotence·unk 0, 230 h diff 19 건 전부 설명. 관문 1–5·7–8 통과, 관문 6 은 `review-KsponSpeech_0X.tsv` 사용자 검토 대기.
"구현 전 남은 확인" 4 항목의 답: num2words 0.5.14 고정·canonical 은 이 문서 golden 으로 검증; 연도 분류 metadata 는 없으므로 LibriSpeech 는 digit 자체가 0 이라 문제 없음(새 코퍼스는 corpus parser 에서 처리);
standalone Latin 1.14 % 로 추정치(1.18 %) 와 일치; 기존 230 h target 변경은 EN 14 건(단어 끝 `'`)·KO 5 건 quarantine.

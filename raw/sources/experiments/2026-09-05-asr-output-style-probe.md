# Qwen3-ASR · Nemotron RNN-T 출력 표기 실측 (2026-09-05, mxc)

목적: 학습 타깃 텍스트 규약을 사전학습 모델의 출력 표기에 맞추기 위해, 두 모델이 숫자·대소문자·구두점·한글 읽기를 어떻게 내는지 확인.
스크립트: 세션 scratchpad `asr_style_probe.py` (Qwen3ASRModel 오프라인 transcribe / Nemotron `[56,0]` transcribe, manifest lang 전달).
표본: LibriSpeech test-clean 에서 숫자 단어 ≥2 + 아포스트로피 포함 발화 6 개, KsponSpeech eval_clean 에서 숫자 이중표기 발화 6 개.

## EN

| REF (원문 대문자) | Qwen3-ASR | Nemotron `[56,0]` |
|---|---|---|
| …WITHIN TEN MILES OF US AND AS FOR THE KITCHEN HE ADDED SMILING I DON'T BELIEVE THERE'S ONE… | …within ten miles of us, and as for the kitchen, he added, smiling, "I don't believe there's one in the kingdom to beat it." | …within ten miles of us, and as for the kitchen, he added, smiling, I don't believe there's one in the kingdom to beat it |
| TWO BITES ARE MADE … I HAVEN'T NEARLY FINISHED… | Two bites are made, and the bread is crumbled … "Half a mole, chap! I haven't nearly finished." … | Two bytes are made, and the bread is crumbled … **\<en-US\>** Indeed, one feels it … I have it nearly finished … |
| …I'VE SPENT ELEVEN EVENIN'S PUTTIN HIM TOGETHER | …I've spent eleven evenings putting him together. | …I've spent eleven evenings putting him together |
| …A FINE OF THREE HUNDRED DOLLARS … MILLIONS OF DOLLARS | …a fine of three hundred dollars, … might be fined millions of dollars. | …a fine of three hundred dollars, … might be fine millions of dollars |

- 숫자: 둘 다 **단어** (ten, eleven, three hundred, millions). 숫자 표기(ITN) 없음.
- 아포스트로피: 둘 다 유지 (don't, there's, I've).
- 대소문자·구두점: 둘 다 **붙임**. Qwen 은 따옴표·느낌표·em-dash 까지; Nemotron 은 쉼표·마침표가 불규칙하고 발화 중간에 `<en-US>` 언어 태그를 삽입.

## KO

| RAW (.trn) | 철자형 | 발음형 | Qwen3-ASR | Nemotron |
|---|---|---|---|---|
| 지금은 (2)/(둘) 다 후회 없+ 없이… | 2 다 | 둘 다 | 어 지금은 **둘 다** 후회 없 없이… | 어 지금은 **둘 다** 후에 없 없이… |
| 근데 (1유로)/(일 유로)도 지금 (1200원)/(천 이백 원) 하더라고요. | 1유로도 … 1200원 | 일 유로도 … 천 이백 원 | 근데 **일 유로**도 지금 천이 병원 하더라고요. | 근데 **일 유로**도 지금 천이 배운 하더라고요. \<ko-KR\> |
| 며칠 (26일)/(이십 육 일)까지니까… | 26일 | 이십 육 일 | 며칠 **이십육 일**까지니까… | 몇 칠 **이십육 개**까지니까… |
| 정원이 (100명)/(백 명)이거든 … 한 (80명)/(팔십 명) 정도 | 100명 … 80명 | 백 명 … 팔십 명 | 정원이 **백 명**이거든 … 한 **팔십 명** 정도 되잖아. | 종원이 **백 명**이거든. \<ko-KR\> … **팔십 명** 정도 되잖아. \<ko-KR\> |
| 매달 (10만 원)/(십만 원)씩 계속 모으고 있어 | 10만 원 | 십만 원 | 매달 **십만 원**씩도 모으고 있어. 태국 여행 가려고. | 매달 **십만 원**씩도 모으고 있어. \<ko-KR\> |

- 숫자: 둘 다 **한글 읽기** = KsponSpeech 발음형과 일치. 숫자·라틴 표기 없음.
- 구두점: Qwen 은 마침표를 붙이고, Nemotron 은 `<ko-KR>` 태그를 붙인다.
- 띄어쓰기: "이십육 일"(모델) vs "이십 육 일"(전사) — 수사 내부 띄어쓰기는 일치하지 않는다 → CER-nospace 로 흡수.

## 결론 → `vapasr/data/textnorm.py`

- EN 타깃: 숫자 → 단어(num2words), 단어 내부 아포스트로피 유지, 소문자, 구두점·하이픈 → 공백. 대소문자·구두점은 모델이 내지만 LibriSpeech 960 h 에 없어 타깃에서는 제거하고 채점에서 양쪽 다 지운다.
- KO 타깃: `(철자)/(발음)` 중 숫자·라틴을 포함한 이중표기만 발음형 → 한글 전용. 구두점·태그 제거, 띄어쓰기 원문.
- 채점: 가설·대조군 출력에도 같은 규칙(태그 제거 포함).

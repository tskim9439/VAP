---
type: source
status: active
created: 2026-09-05
updated: 2026-09-07
summary: Qwen3-ASR·Nemotron 실측에서 영어 숫자는 단어, 한국어 숫자는 한글 읽기로 일치함을 확인한 12개 표본 조사
raw_path: raw/sources/experiments/2026-09-05-asr-output-style-probe.md
observed: 2026-09-05
raw_authors:
  - tskim
---

# Qwen3-ASR · Nemotron 출력 표기 실측

## 무엇인가

Stage 1 학습 타깃의 숫자 표기를 정하기 위해 Qwen3-ASR과 Nemotron RNN-T
`[56,0]`을 실제 음성에 적용한 소규모 조사다. LibriSpeech test-clean에서 영어
6개, KsponSpeech eval_clean에서 숫자 이중표기가 있는 한국어 6개를 골랐다.

원본에는 각 참조 전사와 두 모델의 출력이 보존되어 있다.

## 핵심 관찰

| 항목 | Qwen3-ASR | Nemotron RNN-T `[56,0]` | 당시 lexical 타깃 |
|---|---|---|---|
| 영어 숫자 | `ten`, `three hundred dollars`처럼 단어 | 동일 | 단어형 |
| 영어 아포스트로피 | `don't`, `I've`처럼 단어 내부 유지 | 동일 | 유지 |
| 영어 대소문자·구두점 | 대소문자, 따옴표, 문장부호를 적극 생성 | 불규칙한 문장부호와 `<en-US>` 태그 | 소문자·무구두점 |
| 한국어 숫자 | `백 명`, `십만 원`처럼 한글 읽기 | 동일 | KsponSpeech 숫자·라틴 이중표기의 발음형 |
| 한국어 부가 표기 | 주로 마침표 | `<ko-KR>` 태그 | 구두점·태그 제거 |

숫자 표기에서는 두 모델이 일치했고, KsponSpeech의 `(철자)/(발음)` 중 발음형이
그 출력과 맞았다. 반면 대소문자·구두점은 모델 출력과 타깃을 의도적으로 다르게
두었다. LibriSpeech 대규모 전사에 없는 표기를 합성해 넣지 않고, 평가 때 참조와
가설 양쪽에서 제거한다.

## 후속 정책 변경

2026-09-07 사용자 요구에 따라 영어 대소문자·문장부호를 최종 출력의 필수 능력으로
채택했다. 위 표의 소문자·무구두점은 폐기된 것이 아니라 정렬과 RNN-T 비교를 위한
`lexical_text` 규약으로 남는다. 사용자 표시는 provenance가 있는 `display_text`로
분리하며, LibriSpeech label에 없는 표기를 임의로 gold화하지 않는다. 세부 계약은
[[output-asr-tn-v1-spec]]을 따른다.

## 이 볼트에 준 영향

- [[asr-text-normalization]]에 영어·한국어 공통 타깃 및 채점 규약의 근거를 추가했다.
- [[source-qwen3-asr]]와 [[source-nemotron-3-5-asr-streaming]]의 실제 출력 특성을
  문서 설명과 구분해 기록했다.
- `vapasr/data/textnorm.py`가 코퍼스별 빌더와 평가기 사이의 단일 정규화 진입점이
  되었다(커밋 `4980486`).
- 기존 KsponSpeech 철자형 정렬은 폐기하지 않고 보존하며, 발음형 규약의 manifest와
  `align2/` 정렬을 새로 만든다.

## 한계와 후속 검증

- 총 12개 발화의 목적 표본이므로, 두 모델의 모든 언어·도메인·숫자 문맥에 일반화된다고
  단정할 수 없다.
- 영어 원자료에는 숫자가 이미 단어로 적혀 있었다. 이 실측은 모델의 출력 스타일을
  확인하지만, `$5`, `1984`, `1.50`, `21st`를 어떤 읽기로 변환해야 하는지는 직접
  검증하지 않는다.
- 한국어 수사 내부 띄어쓰기는 모델과 전사 간에 달랐다. 공개 수치용 CER과 별도로
  공백 제거 CER을 내부 비교에 사용한다.
- 모델 출력에는 실제 인식 오류도 섞여 있으므로, 출력 표기와 음향 인식 정확도를
  분리해 해석해야 한다.

## 출처

- 원본: `raw/sources/experiments/2026-09-05-asr-output-style-probe.md`
- 관련 구현: `vapasr/data/textnorm.py`, 커밋 `4980486`

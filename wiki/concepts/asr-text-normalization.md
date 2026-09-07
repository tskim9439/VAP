---
type: concept
status: active
created: 2026-09-05
updated: 2026-09-07
summary: 코퍼스별 ASR lexical/display 타깃과 채점을 통합하고 TN 규약·audit·fingerprint로 재현성을 보장하는 계층
sources:
  - [[source-asr-output-style-probe]]
  - [[source-qwen3-asr]]
  - [[source-nemotron-3-5-asr-streaming]]
---

# ASR 텍스트 정규화

## TN v1 동결 상태

Stage 2의 LibriSpeech 960 h와 KsponSpeech 약 970 h를 위한 규범 문서는
[[output-asr-tn-v1-spec]]이다. 지원 숫자 문법, 미지원·quarantine 조건, 영어와
한국어의 target/score 분리, 영어 lexical/display 이중 view, golden test, manifest
fingerprint 및 버전 정책을 `asr-tn-v1.0.0`으로 명시했다.

현재 상태는 **동결 후보**다. lexical golden 규약 준수, 전체 transcript audit, 기존
230 h diff와 목적 표본 검토가 끝나면 1,930 h alignment를 시작할 수 있다. display
label provenance·loss·평가 subset은 full-FT 시작 전까지 별도로 동결한다. display만
바뀌면 lexical alignment를 다시 만들지 않는다.

## 목적

LibriSpeech, KsponSpeech, otoSpeech, AI Hub처럼 표기 관습이 다른 코퍼스를 그대로
섞으면 모델은 음성이 아니라 DB별 표기 차이까지 학습한다. 학습 타깃과 평가는
코퍼스 파서 뒤의 단일 정규화 계층을 통과시켜야 한다.

```text
원본 전사 → 코퍼스 파서 ┬→ lexical_text → 정렬·streaming loss·WER/CER
                        └→ display_text → 표기 보조 loss·display 평가
모델·대조군 출력 ─────────→ lexical/display별 채점 정규화
```

원본은 `raw/` 또는 manifest의 원문 필드에 보존하고, 정규화된 문자열만 모델 입력과
지표 계산에 사용한다. 구현의 단일 진입점은 `vapasr/data/textnorm.py`다.

## 현재 규약

[[source-asr-output-style-probe]]의 직접 관찰을 근거로 다음을 채택했다.

- 영어 `lexical_text`: 소문자, 단어 내부 아포스트로피 유지, 구두점·하이픈 제거.
  숫자 표기는 발음된 단어형을 지향한다. 강제 정렬, 주 streaming loss와 RNN-T 비교
  WER에는 이 view를 쓴다.
- 영어 `display_text`: 문장·고유명사 대소문자와 `.`, `?`, `!`, `,`, 단어 내부
  apostrophe를 유지한다. 출처가 있는 행만 보조 학습하고, label이 없는 행은 display
  loss를 mask한다.
- 한국어 타깃: KsponSpeech의 숫자·라틴 포함 `(철자)/(발음)` 이중표기는 발음형을
  선택하고 나머지는 철자형을 유지한다. 표지와 구두점을 제거하되 원문 띄어쓰기는
  보존한다.
- 영어 lexical WER: 가설과 참조에 같은 소문자·무구두점 규약을 적용한다.
- 영어 display 평가: case-sensitive WER, capitalization accuracy, 문장부호 class별
  F1과 streaming punctuation delay를 별도로 보고한다.
- 한국어 CER: 언어 태그와 구두점을 제거한 공백 포함 CER을 공개 수치 비교에 쓰고,
  공백 제거 CER을 내부 모델 비교에 함께 쓴다.

Qwen3-ASR과 Nemotron이 대소문자·구두점을 생성한다는 사실을 최종 출력 요구사항에
반영한다. 다만 LibriSpeech 전사에 없는 문장부호를 규칙으로 지어내지는 않는다.
사람 검수 원문, 정확히 복원한 원 저작물, 또는 Qwen pseudo-label 순으로 출처를
기록하고, pseudo-label은 lexical form이 신뢰 전사와 완전히 같을 때만 표기 정보로
채택한다. 따라서 lexical WER의 공정성과 display 출력 능력을 동시에 보존한다.

## 불변 조건

1. 새 DB는 자체 정규화 함수를 학습기에 직접 연결하지 않고
   `textnorm.target(text, lang, corpus)`를 거친다.
2. 학습 타깃 정규화와 채점 정규화를 구분한다. 강한 채점 정규화를 학습 라벨에
   소급 적용하지 않는다.
3. 정규화 함수와 의존성 버전을 고정하고 manifest에 규약 버전을 기록한다.
4. 규칙 변경 전후 manifest·정렬·평가 결과를 다른 경로에 보존해 비교 가능하게 한다.
5. 숫자를 발음형으로 바꿀 때는 표기만으로 실제 발음을 유일하게 결정할 수 있는지
   확인한다. 연도, 통화, 소수, 전화번호, 서수는 문맥에 따라 읽기가 달라질 수 있다.
6. 영어 display label은 provenance를 필수로 가지며, label이 없는 행을 무구두점
   display 정답으로 학습하지 않는다.
7. 대문자는 단어 표기 속성으로 평가하고, 미래 문맥이 필요한 문장부호 지연은 lexical
   evidence 위반과 분리해 측정한다.

## 현재 검증 공백

- 작업 트리에는 `num2words==0.5.14` 고정과 import 실패 정책이 추가됐지만, 소수의
  끝 0 보존, USD 단복수, 서수 범위와 미지원 패턴 무부분변환은 아직 구현되지 않았다.
- 영어 숫자 변환은 `$5` 같은 예시를 처리하지만, 연도·통화 소수·전화번호의 실제
  발음과 일치한다는 근거가 아직 없다. 숫자 문자열이 있는 새 DB마다 표본 감사를
  선행해야 한다.
- `target_ko(..., corpus != "kspon")`는 구두점만 제거하며, 이중표기 밖 숫자와 라틴을
  한글 읽기로 바꾸지 않는다. AI Hub·otoSpeech 연결 전에 잔존율 QC와 코퍼스별
  원자료 표기 검증이 필요하다.
- 현재 manifest와 학습기는 영어 `display_text` provenance, loss masking과 display
  평가 지표를 구현하지 않았다. 1,930 h full-FT 전에 display-retention 평가 subset과
  학습 비율을 고정해야 한다.

위 항목 중 Stage 2 LibriSpeech·KsponSpeech 범위의 처리 원칙은
[[output-asr-tn-v1-spec]]에서 정했다. 아직 코드와 전체 corpus 실측이 따라오지 않았으므로
검증 공백 자체는 해소되지 않았다.

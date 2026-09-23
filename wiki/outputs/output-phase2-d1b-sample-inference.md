---
type: output
status: active
created: 2026-09-18
updated: 2026-09-18
summary: D1b 최종 모델의 미학습 KO·EN 대화 2개 추론 — KO 내용은 양호하나 과분할·화자 오귀속, CHiME-6는 전사 실패
sources:
  - raw/sources/experiments/2026-09-18-phase2-d1b-sample-inference/
  - https://huggingface.co/docs/transformers/model_doc/whisper
  - '[[output-phase2-d1-lane-eval]]'
  - '[[output-phase2-independent-evaluation-plan]]'
---

# Phase 2 D1b 샘플 추론과 청취

## 체크포인트

잡 72504의 실제 산출물은 `/soundai/Model/VAPASR/p2-D1b-restart/final`이다. 2026-09-18 10:45에 12,040 step을 완료했다. 인코더도 `lr=1e-5`로 함께 학습했으며, 종료 후 재로드 검사에서 모델 321키와 encoder 638키의 누락·shape 불일치·값 차이가 모두 0이었다. `parity.json`의 teacher-forced text top-1은 메모리 0.9837, 재로드 0.9826이다.

샘플 추론은 제약 디코더, `delay=4`, `delay_onset=2`, `onset_thr=0.35`, seed 7로 실행했다. 학습에 포함되지 않은 NIKL 2020과 CHiME-6 dev에서 화자 2명 이상인 창을 하나씩 결정론적으로 골랐다.

## 결과

| 표본 | 길이·참조 | 내용 성능 | ONSET P/R | EOT P/R | 관찰 |
|---|---|---:|---:|---:|---|
| NIKL `SDRW2000001298` | 35.0초·104자 | pooled CER **12.5%** | 4.5/50.0% | 0/0% | 첫 시작은 4.28초 참조→4.48초 예측으로 첫 블록 강제 ONSET은 사라졌다. 그러나 2개 참조 episode를 22개로 과분할했고, 후반 화자 2 내용을 lane 1로 바꿨다. lane CER 103.8%. |
| CHiME-6 `S09@4490s` | 25.2초·8단어 | pooled WER **150%** | 40.0/50.0% | 40.0/66.7% | 앞 20초 무음 뒤 짧은 원거리 발화다. 전사 치환·삽입과 lane 3 허위 생성이 함께 발생했다. activity accuracy 96.7%는 긴 무음에 지배되고 F1은 34.0%. |

두 표본만으로 평균 성능을 주장할 수는 없다. 다만 첫 블록 무조건 ONSET 문제와 내용 인식 문제를 분리해 보면, NIKL 표본에서는 **첫 블록 문제는 개선됐지만 turn segmentation·화자 귀속은 여전히 실패**한다. CHiME-6에서는 내용 인식 자체도 실패한다. KO pooled CER만 보고 성공으로 판정하면 과분할과 화자 오귀속을 놓친다.

### 동일 NIKL 원음의 Whisper-large-v2 대조

공식 `openai/whisper-large-v2` revision
`ae4642769ce2ad8fc292556ccea8e901f1530655`를 한국어 전사 모드로 실행했다.
35초 원음을 30초 chunk와 5초 stride로 처리하고 D1b와 같은 한국어 TN을 적용한
결과는 CER **16.35%(17/104)**였다. 원시 출력은 다음과 같다.

> 남편분은 어떻게 만나셨어요? 정말 오래됐다. 남편 만난지. 우리가 스물여덟 살 때 만났으니까 몇 년 됐지? 17년 됐나 보다. 17년. 어떻게 만났느냐 하면, 내가 이제 23살 때부터 교회생활을 시작했거든요. 시작했거든요. 그래서 이제 나는 결혼할 생각이 없었어요.

동일 표본의 D1b pooled CER 12.5%보다 3.85%p 높다. 다만 단일 표본이고,
Whisper는 화자 lane·ONSET·EOT를 출력하지 않으므로 이는 **내용 인식만의 대조**다.
특히 숫자 표기(`17`, `23`)와 NIKL 구어 전사(`심 십 칠`, `스물세`)의 차이도
CER에 포함된다. 원시 결과와 모델 revision·오디오 hash는
`whisper-large-v2-nikl2020-0.json`에 보존했다.

### Whisper-large-v3 교차검증

`openai/whisper-large-v3` revision
`06f233fe06e710322aca913c1bc4249a0d71fce1`을 동일 원음에 실행했다.

| 모델·디코딩 | TN CER | 결과 해석 |
|---|---:|---|
| D1b pooled | **12.50%** | 화자를 무시한 내용 CER |
| Whisper-large-v2, pipeline 30초/5초 | 16.35% | 전체 내용은 전사하되 숫자·구어 표기 차이 포함 |
| Whisper-large-v3, 동일 pipeline 30초/5초 | 169.23% | 첫 문장 반복 생성으로 붕괴 |
| Whisper-large-v3, HF native long-form | 76.92% | 반복은 제거됐지만 중간 발화를 대량 삭제 |

v3의 동일 pipeline 출력은 `남편분은 어떻게 만나셨어요?`를 14회 반복했다.
Hugging Face가 seq2seq의 pipeline chunking을 실험적 경로로 경고하므로, 공식
문서 조건인 `truncation=False`, `padding=longest`, attention mask,
`return_timestamps=True`를 적용한 native long-form도 실행했다. 이때 출력은
`남편은 어떻게 만났어요? 그래서 이제 나는 결혼할 생각이 없었어요.`로,
반복 대신 중간 약 25초의 내용을 삭제했다.

따라서 이 표본에서는 v3가 v2보다 낫지 않다. 다만 이것은 **단일 표본 결과**이고
디코딩 경로에 매우 민감했으므로, Whisper-large-v3 일반 성능이나 모델 순위를
뜻하지 않는다. 전체 NIKL 비교에서는 디코더 설정을 사전 고정하고 다수 파일로
평가해야 한다.

## 청취·시각화

- 로컬 통합 뷰어: `raw/sources/experiments/2026-09-18-phase2-d1b-sample-inference/viewer.html`
- 한국어 원음: `raw/sources/experiments/2026-09-18-phase2-d1b-sample-inference/audio/nikl2020-0.wav`
- 영어 원음: `raw/sources/experiments/2026-09-18-phase2-d1b-sample-inference/audio/chime6-0.wav`
- 원시 추론 기록: 같은 디렉터리의 `report.json`, `windows/*.json`
- Whisper 대조: 같은 디렉터리의 `whisper-large-v2-nikl2020-0.json`
- Whisper v3 대조: 같은 디렉터리의 `whisper-large-v3-nikl2020-0.json`,
  `whisper-large-v3-longform-nikl2020-0.json`

뷰어는 오디오 재생 위치를 참조·예측 lane 타임라인의 세로선으로 동기화한다. 막대 양 끝은 ONSET/EOT이고, 표본 전환 시 pooled 오류·이벤트 P/R·lane별 전사가 함께 바뀐다.

## 다음 판단

1. 이 결과는 샘플 진단이다. 고정 manifest에서 NIKL·CHiME 각각 최소 50창을 평가해 분산과 bootstrap CI를 낸다.
2. 첫 ONSET 지연 오차와 전체 과분할을 따로 센다. `hyp/ref episode ratio`, 발화 중 허위 EOT, 무음 false alarm/hour를 추가한다.
3. NIKL에서 내용은 맞는데 lane이 중간에 바뀌는 현상은 pooled CER·ORC와 cp/lane-DER를 함께 보고 분리한다.
4. CHiME-6는 8단어 표본 하나로 튜닝하지 않는다. 전체 held-out의 substitution/deletion/insertion과 overlap·SNR 조건별 결과를 먼저 낸다.

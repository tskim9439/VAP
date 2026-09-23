# Phase 2 D1b sample inference

- 실행일: 2026-09-18
- 모델: `/soundai/Model/VAPASR/p2-D1b-restart/final`
- 학습 종료: 12,040 step, `parity.json` 통과(thinker·adapter·activity head·encoder 값 차이 0)
- 추론: `experiments/p2_eval_lanes.py`, constrained decoder, `delay=4`, `delay_onset=2`, `onset_thr=0.35`, seed 7
- 데이터: 학습에 쓰지 않은 NIKL 2020 한 창과 CHiME-6 dev 한 창
- 서버 원본: `/soundai/users/tskim/VAPKT-data/eval/p2-D1b-restart-listen-v1`

`report.json`은 전체 집계, `windows/*.json`은 참조·가설·이벤트·활동·토큰 방출 기록이다. `audio/*.wav`는 직접 청취용 원본이고 `audio/*.ogg`는 뷰어용 압축본이다. `viewer.html`은 두 표본을 선택해 오디오와 lane 타임라인을 함께 확인하는 로컬 뷰어다.

`whisper-large-v2-nikl2020-0.json`은 같은 NIKL 원음을 공식
`openai/whisper-large-v2` revision
`ae4642769ce2ad8fc292556ccea8e901f1530655`로 오프라인 추론한 기록이다.
한국어 TN 후 CER은 16.35%(17/104)였다. Whisper는 화자·ONSET·EOT를 예측하지
않으므로 D1b와는 pooled 내용 CER만 비교한다.

Whisper-large-v3 revision `06f233fe06e710322aca913c1bc4249a0d71fce1`도 같은
원음에 실행했다.

- `whisper-large-v3-nikl2020-0.json`: v2와 동일한 pipeline 30초 chunk/5초
  stride. 첫 문장 반복으로 CER 169.23%(176/104).
- `whisper-large-v3-longform-nikl2020-0.json`: Hugging Face 공식 long-form
  조건(`truncation=False`, attention mask, timestamps)으로 재검증. 반복은
  없어졌지만 중간 발화를 대량 삭제해 CER 76.92%(80/104).
- `whisper-large-v3-native-nikl2020-0.json`: attention mask를 빠뜨린 중간 진단
  실행으로 CER 90.38%. 최종 비교값으로 사용하지 않는다.

이 결과는 단일 표본에서 디코딩 경로 민감도를 드러낸 것이며 Whisper-large-v3의
일반 성능 추정치가 아니다.

이 두 창은 질적 진단용이며 모델의 전체 성능 추정치가 아니다. 특히 CHiME-6 표본은 참조가 8단어뿐이어서 WER 분산이 크다.

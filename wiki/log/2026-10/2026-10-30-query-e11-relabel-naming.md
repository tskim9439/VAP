## [2026-10-30] query | E11 출력 재귀속 그림 설명과 "종단 모델" 명칭 — N3.5T 원문은 "a single model", end-to-end 라 부르지 않음

- Changed:
  - 위키: [[source-nemotron-3-5-transcribe-tech-report]](명칭 — 원문 표현), [[output-n3d-branch-20261029]] 6.7 절(부르는 이름)
  - 그림 설명 페이지(Artifact, 비공개): https://claude.ai/artifact/C7kz2gStAWjAmDnPTU7geZ — E11 스트리밍 추론 흐름, 단어 하나의 재귀속 예, 결과, N3.5T·VibeVoice 와 구조 비교, 명칭 판단
- Reason:
  - 사용자 "E11 출력 재귀속을 그림과 함께 설명해줄래? Nemotron 3.5 Transcribe 는 이러한 구조도 단일 종단 모델이라고 부른다고 하지 않았나? 그럼 우리도 이걸 종단 모델이라고 볼 수 있는거 아니야?"
  - N3.5T 보고서 PDF(2026-10-28)를 다시 받아 표현을 확인했다. "a single model"·"a frozen Nemotron 3 Diarization model"·"all components except the diarization model are trained jointly" 이고, end-to-end 는 자기 모델에 쓰지 않는다("outperforms published cascaded and end-to-end systems").
  - 판단: E11 은 "고정 화자 분리 가지를 둔 단일 모델"까지 같은 근거로 말할 수 있다. "종단"은 과하고, 재귀속 결과는 "단일 모델 + 활동 기반 출력 재귀속"으로 적는다.
- Next:
  - LLM 활동 추종 학습(재귀속 없이 같은 값), U3(별도 화자 분리 가지 없애기). 화자 분리기 가중치를 체크포인트에 묶는 것은 필요할 때 한다.
- By: tskim

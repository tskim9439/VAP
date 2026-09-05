## [2026-09-05] decision | 모델 입력을 mono 단일 채널로 확정, 채널별 입력(merge·화자별 오디오 토큰) 폐기

- Changed: 신설 `decision-mono-input`(accepted). `decision-asr-backbone` 세부 3항과 `decision-target-architecture` 결정 절의
  "joint chunk token / 화자별 인코딩 → merge" 를 superseded 표시하고 새 결정으로 링크(두 페이지의 나머지 결정은 유효).
  `README.md` §2(입출력 그림·블록 그림·Stage 3 표·관문), `PLAN.md`(원칙 4 신설, Stage 0 결정 (c), Stage 3 학습 DB·3-1~3-3·실패 조건·데이터 표), `wiki/index.md`.
- Reason: 사용자 결정 — 실제 서비스 입력은 마이크 하나의 mono 오디오다. 코퍼스가 화자별 분리 채널로 배포된다는 이유로 채널별 입력을
  전제한 설계는 배포 불가이고, 화자 구분을 모델이 배우지 않는다. 분리 채널은 라벨(VAD·정렬 → `<SPK_A/B>`·VAP256·hazard)과
  overlap 제어 혼합 합성에만 쓴다. U0.5 adapter 가 단일 채널 분포로 학습된 점과도 일치(U1 v1 merge 분포 이동 가설 원천 제거).
- Next: Stage 1 mono 리더·시퀀스 생성기 신설(두 채널 코드 `interleave_data.py`/`model.py` 는 주 경로 제외). Stage 3 은 분리 채널 합산 혼합으로
  3-1 혼합(전사만) → 3-2 화자 태그 → 3-3 overlap. mono 입력 encoder-only probe 대조군 재구성. U1 v2 run 처리(완주/중단)는 사용자 결정 대기.
- By: tskim

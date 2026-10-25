## [2026-10-25] task | 화자 분리 통합 E2 결과(정답 활동 화자별 디코드)·같은 모델 E0·E3 제출

- Changed:
  - `raw/sources/experiments/2026-10-25-ami-sdm-analysis/README.md`(E2 절), `e0-q17-mt4-final.json`(출발 모델 E0, 분할별 합계), [[status]]
- Reason:
  - 사용자 "계획대로 진행하는 데 1.7b 모델로 진행해"([[output-diarization-integration-20261025]] E2).
  - E2 학습(95486)은 low_p_hpc 선점으로 step 0 부터 다시 돌아 21:25 에 끝났다(저장·재로드 대조 통과). 평가(95487)는 23:35 에 끝났다.
  - δ10 창 단위 cpWER 이고, 출발 모델 → 같은 모델 E0 → E2 순이다.
    - SDM eval 53.62 → 38.98 → 28.56, IHM-mix eval 49.91 → 34.72 → 21.20 이다.
    - E2 cp 는 출발 모델 agn(28.31·21.66)과 거의 같다. 정답 활동이면 귀속 오류가 사라지고 받아쓰기 오류만 남는다. spill 은 0 이다.
  - 사용자가 E3 평가를 냈다(95543, low_p_hpc, 임계 0.45). 화자별 디코드 구조를 설명하는 그림 페이지도 만들었다(https://claude.ai/artifact/VbHf4fuJff58Stbo1Y98Sy).
- Next:
  - E3 결과로 E2 대비 화자 분리 손실을 잰다(E1 miss 13 %, 그 절반이 겹침).
  - 받아쓰기(SDM agn 약 28–30)를 20 아래로 낮출 방안을 검토한다(원거리·겹침 데이터, 대상 화자 학습 확대).
  - 화자 분리 가지를 E2 인코더 위로 옮겨 인코더를 한 벌로 합친다. 실시간 화자 분리(AOSC)도 다룬다.
- By: tskim

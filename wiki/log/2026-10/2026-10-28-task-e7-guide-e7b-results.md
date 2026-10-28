## [2026-10-28] task | E7 태그 이끌기 회의 통째 전체·E7b 창 결과(표준 재채점)

- Changed:
  - 위키: [[output-diarization-integration-20261025]] 7 절 결과 5(태그 이끌기 96942)·결과 6(E7b 창 96963)
  - 서버 재채점(읽기만, 새 폴더): `results/stage3-eval-amifull-c30-aosc-e7a-guide-d10-q17-e6-final-wn`, `results/stage3-eval-ami-d10-q17-e7b-final-wn`(`rescore_std.sh`)
- Reason:
  - 사용자 요청 "96172 끝나면 비교 결과 정리해줘" 이후 이어진 대책 셋(사후 진단·태그 이끌기·E7b)의 결과 정리. cpWER 는 외부 기준과 함께 적는다.
  - 태그 이끌기(학습 없음): 회의 통째 eval meeteval cpWER IHM-mix 67.34 → 47.50, SDM 73.10 → 61.57(δ10). 받아쓰기(agn)는 그대로.
  - E7b 창: IHM-mix eval 37.09 → 35.74, SDM 44.73 → 44.05. 귀속 몫은 2.2–2.6 pt 줄었으나 agn 이 1.2–1.9 pt 나빠졌다.
- Next:
  - E7b 회의 통째(AOSC·E7a 헤드) 평가가 끝나면 이끌기·E9 와 같은 표에 넣는다.
  - E7b 의 받아쓰기 악화 까닭(인코더 동결·다시 데우기·global slots)을 가른다.
- By: tskim

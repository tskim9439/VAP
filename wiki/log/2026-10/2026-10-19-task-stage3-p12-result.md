## [2026-10-19] task | Stage 3 파일럿 p12 결과 — NSC 격리 발화 되살리기는 독일어 귀속을 줄이지 못했다(구간이 0 포함), Stage 3 최종은 p11 ck3000 그대로

- Changed:
  - raw [[2026-10-18-stage3-ko-revive]]: p12 결과 절, `p12-vs-p11.txt`·`p12-regress.txt`, `cmp_p12.py`·`cmp_attr_p12.py`·`boot_p12.py`
  - [[output-stage3-prep-plan-20261014]] p12 줄, [[status]]
- Reason: 사용자 "응 p12까지 진행해" — apex job 91582 학습(4000 step, PARITY OK)과 평가 큐 셋이 끝났다.
- 결과(p11 → p12, δ10 귀속 몫):
  - 독일어: ck3000 8.6 → 8.7, final 8.8 → 8.7. bootstrap +0.15 [−1.14, +1.38], −0.17 [−1.67, +1.34]. NSC 만 따로 봐도 0 을 포함한다.
  - ck3000 전체 cp +1.56 [+0.55, +2.55] 로 나빴고, final 은 모든 언어에서 차이가 없다. 단일 화자 평가는 대부분 ±1 안이다.
- Next: Stage 3 최종·Stage 4 시작점은 p11 ck3000 그대로 둔다. 독일어 귀속은 Stage 4 학습에서 다시 본다.
- By: tskim

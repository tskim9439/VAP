## [2026-10-16] task | Stage 3 본 학습 q06-st3-r1 결과 — checkpoint-2000 확정(δ10 cp 28.6·EOT 정밀도 0.63), 화자 귀속 오류는 잦은 교대·겹침에 몰림

- Changed:
  - [[output-stage3-prep-plan-20261014]]: "결과와 확정" 절, "화자 귀속 분석과 개선 후보" 절, 작업 순서 표 P6 완료, summary
  - raw `raw/sources/experiments/2026-10-16-stage3-main-r1/`([[2026-10-16-stage3-main-r1]]): 체크포인트별 대화 평가, 단일 화자 퇴행, 화자 귀속 분석, 학습 경과·parity
- Reason: 사용자 결정 "checkpoint-2000으로 확정하고 위키에 기록해줘"
- 결과:
  - 학습은 21:37 완료했고 parity OK 다.
  - step 2,000 뒤에는 cp·agn 이 함께 나빠졌다(δ10 cp 28.6 → 31.0). 영어·독일어 대화 창을 되풀이한 과적합으로 본다.
  - ck2000 퇴행은 p7 과 비슷하다. en→de BLEU 가 가장 많이 떨어졌다(s4 대비 δ4 −3.5).
  - 화자 귀속 몫(cp − agn) 11.4 pt 는 교대 ≥ 20 /분 창(19.0 pt)과 겹침 ≥ 5 % 창(약 16 pt)에 몰린다.
- Next:
  - final 단일 화자 평가가 끝나면 raw 에 덧붙인다.
  - 화자 귀속 개선 후보 중 하나를 고른다. 첫 단계는 오류 유형 분해다.
  - Stage 2 전체 평가와 Stage 4 준비로 넘어간다.
- By: tskim

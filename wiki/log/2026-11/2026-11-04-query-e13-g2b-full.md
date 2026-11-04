## [2026-11-04] query | E13 + G2b 학생(G3 형태) 회의 통째 결과

- Changed:
  - 위키: [[output-diarizer-replacement-20261031]] 14 절, [[output-ami-results-summary-20261030]] 1.1 표 두 줄·3 절 DER 표 G2b 줄
- Reason:
  - 99712 회의 통째가 11:36 에 끝나 `auto_g3b.sh` 가 11:52 에 표준 재채점을 냈다. 외부 화자 분리기 없이(우리 인코더 하나) IHM-mix eval 24.48 로 E13 + N3D(24.39)와 같고, SDM eval 은 36.10(N3D 31.33, G3 53.92)이다.
  - ASR 지킴(99505)은 통과했다(창 agn 27.93 / 32.70 대 27.80 / 32.70).
- Next:
  - E11 + G2b 회의 통째(99713)와 `g3b-report.txt`. SDM 의 남은 차이는 귀속(다른 슬롯만 열린 단어 2.8 → 6.1 %)이다.
- By: tskim

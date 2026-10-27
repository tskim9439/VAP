## [2026-10-27] query | F1 데이터 확장 결과(mt4 대비)

- Changed:
  - [[task-data-expansion-20261026]] F1 결과 절, `raw/sources/experiments/2026-10-26-data-expansion/f1_compare.sh`
- Reason:
  - 사용자 "평가 끝나면 비교 결과 정리해줘".
  - 평가 95866(45/45)이 끝나 f1_compare.sh 로 쟀다. 검정은 AMI 회의 bootstrap, TAXI 세션 짝 bootstrap 이다.
  - **AMI**
    - dev 는 IHM −3.3*·SDM −2.5*(δ10, final)로 좋아졌다. 거의 다 화자 귀속 몫이다.
    - eval SDM 은 +0.3(n.s.)로 그대로다.
  - **TAXI(실제 독↔영 전화)**
    - +2.0* 나빠졌다. 받아쓰기 쪽이고(영어 WER 23.6→26.0, 독일어 13.8→15.4), F1 체크포인트 모두 그렇다.
  - **가드**
    - final 은 통과했다.
    - ck4000·avg3 는 FLEURS ko-en δ10 에서 전사를 일찍 끊는다(CER 21 %, 삭제).
- Next:
  - TAXI 받아쓰기 회귀의 원인을 가른다. 후보는 영·독 대화의 원거리 증강, 회의형 데이터 비중, Switchboard 다.
  - E5·E6(F1 레시피 위, 비교 기준 F1)은 그대로 돈다.
- By: tskim

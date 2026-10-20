## [2026-10-21] query | V2(v2-p1) 대 VibeVoice — 같은 창 재채점, 받아쓰기는 VibeVoice 오프라인보다도 낫고 귀속·TAXI 가 남은 차이

- Changed:
  - [[output-vibevoice-stream-comparison-20261020]]: "V2 첫 학습(v2-p1 ck2000) 대비" 절, TAXI 표 V2 두 줄, 요약·summary
  - raw [[2026-10-19-stage4-dialogue-translation]] `vibevoice/dialogue-{stream,offline}-score-v2.json`(`baseline_vibevoice.py score --compare`, V1 m2a ck2000·V2 ck2000·ck2500, 새 파일)
- Reason: 사용자 "VibeVoice 와 비교해서 어떻지?"
- 결과(δ10, 같은 Stage 3 회귀 창 200 × 7, cp / agn):
  - V2 ck2000 평균 24.5 / 17.0 대 VibeVoice 스트리밍 49.0 / 44.0·오프라인 37.6 / 35.4. 스트리밍보다 7 코퍼스 모두 낫다(영어 24.0 대 26.3).
  - 받아쓰기(agn)는 V2 가 VibeVoice 오프라인보다도 7 코퍼스 모두 낫다. 마스크 없는 창에서도 같다.
  - 귀속 몫은 V2 7.4(V1 과 같음) 대 VibeVoice 스트리밍 5.0·오프라인 2.2. 그래서 VibeVoice 오프라인이 영어·kec 의 cp 에서 앞선다.
  - TAXI 세션 통째: VibeVoice 스트리밍 19.2 대 V1 26.4 → V2 30.6. 차이가 7.2 → 11.4 pt 로 벌어졌다.
- Next: 다화자 창(회의·AMI·3–4 인)의 VibeVoice 비교는 채점기의 참조 2 화자 가정을 풀어야 한다(제안).
- By: tskim

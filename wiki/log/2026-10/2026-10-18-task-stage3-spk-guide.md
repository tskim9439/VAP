## [2026-10-18] task | Stage 3 화자 헤드 안내 디코딩 평가 — 태그 삽입은 손해(δ10 cp +4.1), 삽입 + 막기는 δ10 중립·δ4 cp −1.3; 라벨 오류가 agn 을 부풀리는 몫은 최대 약 2.7 pt

- Changed:
  - `vapasr/hf/batch_decode.py`: `SpkGuide`·`_guide_slot`, `decode_batch(spk_guide=)`
  - `experiments/cst_stage3_eval.py`: `--spk-guide`·`--spk-guide-hold`·`--spk-guide-no-insert`
  - `tests/test_spk_guide.py`
  - raw [[2026-10-18-stage3-spk-guide]]: 동치 확인·결과·읽기
  - raw [[2026-10-17-asr-rejudge]]: 평가 창 영향 절
  - [[output-stage3-prep-plan-20261014]]: "정답 라벨 품질 점검"·"화자 헤드 안내 디코딩" 절
- Reason: 사용자 질문 "어떻게 더 개선할 수 있을까?", "이게 얼마나 stage 3에서의 문제를 해결할 수 있을지 모르겠네"
- 결과:
  - 안내 꺼짐(on 2.0) 디코드는 기존 p9 ck3000 출력과 200/200 창의 토큰이 같다.
  - 삽입만: δ10 cp +4.06 [+3.00, +5.19], 헛교대·이른/늦은 교대 증가
  - 삽입 + 막기: δ10 −0.15(구간 안), δ4 cp −1.27 [−2.12, −0.47]
  - 라벨 불일치 음성이 많은 창일수록 agn 이 높다(12.7 → 21.4). 전체 agn 을 부풀리는 몫은 최대 약 2.7 pt, 귀속 몫과는 무관하다.
- Next:
  - 막기만 판(`p9ck3000-guide-holdonly`)의 결과를 본다.
  - 디코딩 기본은 안내 없음이다.
- By: tskim

## [2026-10-18] task | Stage 3 최종을 st3-p10-sim checkpoint-3000 으로 확정 — 화자 헤드 + 합성 대화, δ10 cp 25.6(이전 최종 r1 ck2000 28.6)

- Changed:
  - [[output-stage3-prep-plan-20261014]]: p10 결과 절에 결정, 작업 순서 표 P6, 이전 결정 문장에 교체 표시, summary
  - raw [[2026-10-18-stage3-sim-dialogue]]: 결정 절
- Reason: 사용자 결정 "p10 checkpoint-3000으로 확정하고 위키에 기록해줘"
- 결과:
  - `/soundai/Model/VAPASR/stage3-st3-p10-sim/checkpoint-3000` 점검: 인코더 키 638, `spk_head` 저장, `<EOT>` 등록·차단 해제, tokenizer, `config.stage3` cst-v1, global step 3,000
  - 성능(δ10): cp 25.6 · agn 17.5 · EOT 재현율 0.719 · 정밀도 0.624
  - r1 ck2000 대비 cp −3.04 [−4.30, −1.77], 귀속 몫 −3.52, agn +0.48
- Next:
  - Stage 2 전체 평가와 Stage 4(대화 번역) 준비를 이 체크포인트로 한다.
  - 한국어 귀속 개선 후보를 검토한다: 실제 한국어 비중 회복, NIKL 화자 재료.
- By: tskim

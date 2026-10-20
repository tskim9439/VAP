## [2026-10-20] task | 다음 학습 st4-m2 준비 — 전화 대역 증강(--phone-aug)·CallHome 독일어 단일 화자 셋·CallHome 평가 셋, 점검 통과

- Changed: `vapasr/data/phone_aug.py`(새)·`cst_dialogue.py`(wav_aug), `experiments/stage3_train.py`(--phone-aug), `slurm/submit-stage3-train.sh`(--phone-aug·--replay-set), `tests/test_phone_aug.py`, raw [[2026-10-19-stage4-dialogue-translation]] "다음 학습 st4-m2 준비" 절·평가 큐 6 단계
- Reason: 사용자 "전화 대역 증강도 넣어서 다음 학습 설정 제안해줘"
- 결과:
  - 전화 대역 증강: 8 kHz·300–3400 Hz 대역(영위상)·μ-law, 단위 시험 4 개 통과
  - CallHome 독일어 평가 셋 671 발화. 기준점 WER δ10: st4-m1 ck1500 26.8, p11 27.6
  - 학습 dry-run 불일치 0, 스모크 PARITY OK
- Next: st4-m2a(전화 대역만)·st4-m2b(전화 대역 + rep_chde) 제출(사용자) → 평가 큐 ck1500·ck2000·final
- By: tskim

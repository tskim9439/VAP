## [2026-11-07] query | 진단 — E15b 슬롯 끄기, ESB AMI 단일 화자 받아쓰기

- Changed: [[output-ssa-papers-review-20261106]] 6 절(신규)·summary, [[output-agn-reduction-20261104]] 11 절 남은 질문 갱신. 코드 `--slot-enc-off`·`SLOTOFF=1`, `eval_single_turn_asr.py prepare --esb-ami-list`, `slurm/submit-eval-esb-ami.sh`, `esb_ami_whisper.py`.
- Reason: 사용자 "1번 진단부터 진행하자"
- 발견
  - 슬롯을 꺼도 겹친 말 놓침이 그대로다(43.0 → 43.3 / 49.5 → 49.8 %). E15b 손해는 인코더 위층 학습에서 왔다.
  - ESB AMI 단일 화자 WER(Whisper)은 E13 16.47 %·Stage 2 19.10 % 다. 논문 보고 Nemotron Speech 11.58 % 보다 약 5 pt 나쁘고, 삭제가 절반 이상이다(짧은 발화 비움, 긴 발화 기능어).
- Next: 같은 셈으로 Nemotron 계열 단일 화자 기준 재기, 삭제 원인(짧은 발화·맞장구·학습 표기) 진단 → 혼자 말 받아쓰기 데이터·학습 결정
- By: tskim

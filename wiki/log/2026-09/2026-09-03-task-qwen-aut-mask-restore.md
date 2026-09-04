## [2026-09-03] task | Qwen AuT attention 마스크 복원·변형 실험

- Changed: `experiments/qwen_aut_mask.py`, `experiments/qwen_aut_mask_eval.py`(신규),
  `raw/sources/experiments/2026-09-03-qwen-aut-mask-eval.json`(신규 raw), `output-encoder-causality-audit` 추가 실험 절,
  `task-qwen-aut-causal-adaptation` p2→p1 및 체크 항목 갱신, `decision-asr-backbone` 5항 추가, `source-qwen3-asr`,
  `todo.md`·`TODO.md`·`index.md`. 부수: AI Hub 절차를 PC 다운로드 → `scripts/aihub-upload.sh` 전송으로 변경
  (API 키 발급 불가 확인), `source-conversation-corpora`·`task-verify-aihub-stereo-and-access` 갱신.
- Reason: 사용자 질문 "`_prepare_attention_mask` 를 직접 복원할 수 없나" 에 대한 실험. 레이어 forward 를 감싸
  `cu_seqlens` 로 마스크를 주입했다. block 1 s 가 per-block 실측과 일치해 패치 검증. chunked-causal 1 s 는
  lookahead 420 ms·WER +5.9 % 로 as-is 최선이나 관문은 여전히 초과. **프레임 causal 마스크에서 lookahead 80 ms,
  WER 23.5 %(단일 발화)** — 학습에 없던 마스크에서도 단어 대부분이 보존되어 causal fine-tune 으로 Qwen 을
  살릴 가능성이 열렸다. block 8 s(배포 의도)가 sdpa 무마스크와 11.8 % 다른 점도 확인 — transformers 백엔드
  결과의 재현성 주의.
- Next: 제대로 된 평가셋에서 마스크별 WER/CER, causal 마스크 소규모 fine-tune 회복 폭 측정
  ([[task-qwen-aut-causal-adaptation]]). AI Hub 는 사용자 다운로드 대기.
- By: tskim

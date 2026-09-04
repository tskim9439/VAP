## [2026-09-04] task | IS-SLM 최종 backbone 확정: Nemotron [56,0] → adapter → Qwen3-ASR thinker

- Changed: `decision-asr-backbone` → accepted(최종 고정 조합, U0.5 관문), 신설 `task-uslm-u05-adapter-bridge`(p0, ~09-25), `task-uslm-u1-interleaved-asr` backbone 문구,
  `task-qwen-aut-causal-adaptation` 취소, bilingual 태스크에서 Qwen AuT 포팅 제거, `README.md`, `TODO.md`, `wiki/overview.md`, `wiki/status.md`, 관련 설계 outputs.
- Reason: 사용자가 "Nemotron 3.5 FastConformer [56,0] → new adapter → Qwen3-ASR-0.6B-hf LM" 조합을 최종 선택. encoder는 실측 causal ≤80 ms·
  잡음 강건·ko-KR이고 thinker는 Qwen audio embedding에서 text를 생성하도록 사전학습됐다. 분포 간극은 기존 Qwen audio tower의 thinker 입력 embedding을
  교사로 삼은 표현 증류와 짧은 ASR 미세조정으로 조기 검증한다. 실패 시 adapter 학습을 재설계하며 backbone은 자동 교체하지 않는다.
- Next: U0(토큰율·정렬)과 U0.5(adapter bridge)를 병행 착수. Stage 1 결과 페이지는 cc1s 재채점 후.
- By: tskim

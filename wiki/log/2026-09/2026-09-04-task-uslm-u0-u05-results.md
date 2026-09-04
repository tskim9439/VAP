## [2026-09-04] task | U0.5 adapter bridge 실험 완료(4 run), U0 토큰율·M 예산·정렬 진행

- Changed: `task-uslm-u05-adapter-bridge`(4 run 결과·종합 판정·관문 재검토 제안), `task-uslm-feasibility-u0`, `README.md`(§4 U0/U0.5 상태, p0 행),
  `wiki/status.md`(Git 원격 SSH 443). 코드: `experiments/u05_{distill_adapter,baselines,asr_finetune}.py`, `u0_{token_rate,align,interleave_stats}.py`,
  `vapasr/uslm/{data,model}.py`, `vapasr/data/interleave.py`. 서버 산출물 `/data4/tskim/VAPASR/experiments/uslm/{u05-asr-*,adapter-distill-*}`, 정렬 `/data3/tskim/manifests/align/`.
- Reason: 최종 backbone(Nemotron [56,0] → adapter → Qwen3-ASR thinker) 연결 품질 조기 검증. 기준선 오프라인 Qwen 13.9/13.5/13.3 %,
  Nemotron RNN-T [56,0] 25.4/23.4/24.5 %, [56,13] 19.0/16.9/17.0 %. 증류(block8s 교사, cos 0.777) init 과 random init, lr 2e-4/1e-4, 6k/12k step
  4 조합 → 모두 oto 18–20 / 실내 17–19 / 실외 15–17 % 수렴(최선 증류 lr 1e-4: 6k 18.2/19.2/16.1, 12k@8000 18.8/17.5/14.5). 증류 이득은 잡음 범위.
  관문 원안(오프라인 ×1.15)은 실외만 통과하나 동일 인코더의 RNN-T 대비 −6 pt → adapter 병목 아님, 격차는 인과 인코더 상한으로 해석.
  U0: KO 토큰율 p99 0.78 tok/80 ms(폭주 없음), chunk M=4(KO 이월 2.6 %), 생성기 완료, 정렬 aihub 완료·oto 진행. 운영: 한 run 25 GB → 순차 체인(pid 대기),
  GitHub 는 회사망 22 번 차단으로 SSH 443 경유, 첫 커밋 598a8bd 푸시.
- Next: (사용자 결정) U0.5 관문 재정의 vs Nemotron 상위 블록 unfreeze 1 회. 정렬 완료 후 `u0_interleave_stats` 전체 재실행·M 확정 → U1(interleaved ASR) 착수.
  Stage 1 잔여(seed 반복, EN-only, DualTurn).
- By: tskim

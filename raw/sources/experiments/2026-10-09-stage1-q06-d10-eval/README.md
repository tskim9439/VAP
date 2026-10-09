---
author_id: tskim
created: 2026-10-09
source_type: source-drop
---

# 0.6B 영·한·독 Stage 1(q06-s1m3-d10, job 82844) 단일 발화 평가 (mxc 호스트, 2026-10-09)

- 체크포인트: `/soundai/Model/VAPASR/semcommit-q06-s1m3-d10/final`. 학습은 2026-10-09 08:21 에 끝났고, 다시 불러와도 손실이 같았다(parity true).
- 실행: `experiments/run_single_turn_asr_mxc.sh <ckpt> <out> <gpus> 64 0 4,6,8,10 <manifest> final`. 체크포인트의 인코더 설정 [56,3] 을 그대로 썼다. batch 64, 꼬리 무음 1.0 s 다.
  - 영·한: `single-turn-vpaa-kspon-v1`(VoxPopuli-AA test 628 + Kspon eval_clean 3,000 + eval_other 2,687). 결과는 `results/single-turn-vpaa-kspon-q06-s1m3-d10/` 이고, GPU 0·1 에서 약 55 분 걸렸다.
  - 독일어: `single-turn-yodas-de-dev-v1`(4,861 발화, 7.87 h, 참조는 Qwen3-ASR-1.7B 정규화 전사로 silver). 결과는 `results/single-turn-yodas-de-q06-s1m3-d10/` 이고, GPU 2·5 에서 약 50 분 걸렸다.
  - 예측은 모두 나왔다(31,575 / 24,305). 강제 종료된 발화는 칸마다 0–5 개다.
- `comparison-bootstrap.txt`(`cmp_eval.py`):
  - 지표: 영어 WER, 한국어 CER(공백 제외), 독일어 WER
  - 비교 대상: q06-s1(0.6B 파일럿, [56,0] 평가), q17-s1m-r3(1.7B Stage 1 본학습, [56,3])
  - Δ = 새 모델 − 비교 대상이다. 짝 bootstrap 1,000 회의 95 % 구간을 냈고, `*` 는 0 을 포함하지 않는다는 뜻이다.
- `summaries.json`: 칸별 오류율·강제 종료 수·AA-WER(aa-wer-approx-v1)

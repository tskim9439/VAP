---
author_id: tskim
created: 2026-10-10
source_type: source-drop
---

# 0.6B Stage 1(q06-s1m3-d10) LibriSpeech test-clean·test-other 평가 (mxc 호스트, 2026-10-10 23:25–23:58)

- 체크포인트: `/soundai/Model/VAPASR/semcommit-q06-s1m3-d10/final`. 체크포인트에 저장된 인코더 설정 [56,3] 을 그대로 썼다.
- manifest: `data/evaluation/single-turn-librispeech-v1/manifest-test-{clean,other}.jsonl`
  - 기존 `single-turn-v1/manifest.jsonl` 에서 LibriSpeech 행만 뽑았다(2,620 / 2,939). 참조와 오디오는 같다.
- 실행: `experiments/run_single_turn_asr_mxc.sh <ckpt> <out> <gpus> 64 0 4,6,8,10 <manifest> final`
  - test-clean: GPU 0·1 → `results/single-turn-librispeech-clean-q06-s1m3-d10/`
  - test-other: GPU 2·5 → `results/single-turn-librispeech-other-q06-s1m3-d10/`
  - 각각 약 33 분 걸렸다. 예측은 13,100 / 14,695 로 모두 나왔고, 강제 종료된 발화는 0 개다. 재로드 parity 는 모두 PASS 다.
  - 규약은 single-turn v1 과 같다: 발화 + 1 s 무음, greedy, FP32, batch 64, 지표는 WER.
- `comparison-bootstrap.txt`(`libri_bootstrap.py`):
  - 같은 발화로 짝 bootstrap 2,000 회를 하고 화자 단위로 묶었다(73 명). Δ = 새 모델 − 비교 대상이고, `*` 는 95 % 구간이 0 을 포함하지 않는다는 뜻이다.
  - 비교 대상(모두 같은 0.6B thinker, 인코더 right context 0):
    - 35000(semcommit 이전, hf-approved-qwen-v1-e10-n2/checkpoint-35000)
    - E2
    - v035-d8
  - 오프라인 참고: Qwen3-ASR-0.6B·1.7B. 새 모델의 final 모드와 비교했다.
  - "4/6", "4/8" 행은 미리 듣기를 맞춘 비교다. 새 모델 δ4 는 4×80 + 240 = 560 ms 이고, 옛 모델 δ6 은 480 ms, δ8 은 640 ms 다.
- `leak-check.txt`(`libri_leak.py`): 학습 데이터에 LibriSpeech 가 얼마나 들었고 test 와 겹치는지 확인했다.
  - 학습 데이터는 v035 스냅숏(main+short)과 Stage 1 ASR words 목록이다.
  - 들어 있는 LibriSpeech 분할은 train-clean-100·360 과 train-other-500 뿐이다.
  - test 화자 73 명과 겹치는 화자는 0 명이다.
- `summaries.json`: 칸별 WER(치환·삭제·삽입 수 포함)과 완료 여부

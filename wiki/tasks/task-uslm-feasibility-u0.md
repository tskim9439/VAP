---
type: task
status: doing
owner: tskim
due: 2026-09-18
priority: p1
created: 2026-09-04
updated: 2026-09-04
summary: USLM U0(streamability+타당성) — 토큰율→M 예산, ForcedAligner 정렬·QC, interleaved target 생성기
sources:
  - [[output-unified-slm-architecture-plan]]
---

# USLM U0 — streamability 증명 + 타당성 측정

## 배경

[[output-unified-slm-architecture-plan]] 의 쟁점 2(1 프레임 = 1 토큰 상한, 한국어 BPE)와 3(정렬 감독 노이즈)은 실험 없이
결정할 수 없다. 둘 다 GPU 가 거의 필요 없다.

## 완료 조건

- [x] encoder streamability(truncation audit) — [[output-encoder-causality-audit]] 에서 완료 (Nemotron ≤80 ms, AuT 는 causal 적응 필요)

- [x] 토큰율(발화 균등 가정, `experiments/u0_token_rate.py`, 라벨 624 h KO / 71 h EN): **KO tok/s p50 4.2 · p99 9.7, tok/80ms p99 0.78** (12.5 tok/s 초과 발화 0.18 %); EN p50 4.8 · p99 20 (초과 2.6 %, 짧은 발화 꼬리). chars/token KO 1.43 / EN 3.87 — **한국어 BPE 폭주 없음**
- [x] **M 예산 (chunk 단위, 정렬 토큰 기준, δ=2, 테스트 80 발화)**: EN M=2 이월 2.8 % / M=3 0.9 %; KO M=2 **21.6 %** / M=3 6.1 % / M=4 2.6 % — 정렬기가 음절 토큰들에 같은 종료 시각을 주어 chunk 내 burst. → **M=4 기본(KO 2.6 %, EN 0.35 %)**, M=3 ablation. **전체 aihub-ts01-5 정렬(190 대화, 692k 토큰, 2.3M chunk)로 재확인(09-04): M=2 22.3 % / M=3 6.9 % / M=4 2.1 %**, max backlog 13–15 chunk(≈1.1 s, 극단 burst), chunk 의 84 % 가 무방출 → **M=4 확정**. otoSpeech 전체는 정렬 완료 후 자동 집계(bg `u0-istats-oto`)
- [~] Qwen3-ForcedAligner 정렬 — `experiments/u0_align.py` (aligner 단어/음절 시각 → BPE 토큰 종료 시각, offsets 매핑). 소규모 테스트 통과(발화당 0.33 s, 라벨 끝−마지막 토큰 끝 중앙값 EN 20 ms / KO 45 ms). **전체 실행**: aihub-ts01-5 190 완료(09-04), otoSpeech 334/420 진행, vs02 미착수(≈70 s/대화) → `$DATA_MANIFEST_DIR/align/`
- [ ] 정렬 QC: aligner 시각 vs 에너지 VAD 온셋 오차 분포, 불량 발화 마스킹 규칙
- [x] **interleaved target 시퀀스 생성기** — `vapasr/data/interleave.py` (δ, M, backlog 이월, <SPK> 태그, <NEXT_AUDIO>/<EMPTY_AUDIO>) + `experiments/u0_interleave_stats.py`
- [ ] (구) 시간 동기 PAD 스트림 — 정렬 토큰 종료 시각 + δ 이후 chunk 에 배치, `<NEXT_AUDIO>` 삽입, 지연 스케줄 무작위화 옵션 — `vapasr/data/` 에 추가 (시간 동기 PAD 스트림은 ablation 용 옵션)
- [ ] 결과를 [[output-unified-slm-architecture-plan]] 에 반영, U1 착수 여부 판단

## 진행 기록

- 2026-09-04: 착수. 토큰율·생성기 완료, 정렬 전체 실행 중. thinker 주입 경로 확인(`<|audio_pad|>` id 151676, hidden 1024).
- 2026-09-04: 생성.

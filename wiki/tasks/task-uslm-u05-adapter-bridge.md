---
type: task
status: doing
owner: tskim
due: 2026-09-25
priority: p0
created: 2026-09-04
updated: 2026-09-04
summary: USLM U0.5 — Nemotron→thinker adapter bridge test: 캐시 특징 증류 초기화 → 오프라인 ASR 미세조정 → WER 관문(AuT+thinker·RNN-T 대비 ≤10–15%)
sources:
  - [[decision-asr-backbone]]
  - [[output-interleaved-streaming-slm-architecture]]
---

# USLM U0.5 — adapter bridge test

## 배경

최종 backbone 은 Nemotron [56,0] → adapter → Qwen3-ASR thinker ([[decision-asr-backbone]]). 유일한 신규 위험은 **adapter 가 분포 간극을
메우는가** 다. 같은 205 h 에 대해 Nemotron 과 AuT 특징이 모두 캐시되어 있어 라벨 없이 초기화를 만들 수 있다.

## 완료 조건

- [x] **증류 초기화** — block8s 타깃(35 h, 129 쌍) 4 epoch: held-out cosine **0.777** (항등 0.004), mse/var 0.38 → `adapter-distill-qwen-aut-block8s/adapter.pt`
- [x] 증류-init fine-tune(6000 step, lr 2e-4/5e-4) 완료 — otoSpeech 37.3 → 21.5 → **20.0 %**, 실내 32.8 → 20.9 → **18.3 %**, 실외 24.3 → 19.5 → **16.9 %**.
      random-init 3000 step(18.7 / 20.6 / 18.3) 대비 EN 은 뒤지고 KO 는 앞섬 — **증류 초기화의 이득은 불확실**(초기 불안정이 상쇄). 관문(otoSpeech ≤ 16.0 %, KO ≤ 15.5 %) 미달
- [x] 증류-init **lr 1e-4/1e-4 run(6000 step) 완료** — otoSpeech 33.6 → 21.1 → **18.2 %**, 실내 25.3 → 18.6 → **19.2 %**, 실외 25.6 → 17.6 → **16.1 %** (`u05-asr-distill-lowlr`, 34 min). 고 lr run(20.0/18.3/16.9) 대비 EN 개선·KO 실내 소폭 악화; 손실 스파이크 없이 안정. **결론 1: 6000 step 은 수렴 전**(4000→6000 에서 EN −2.9 pt) — 관문(oto ≤16.0 %, KO ≤15.5 %) 미달은 용량 한계가 아니라 학습량 부족일 가능성이 큼. **결론 2: 증류 init 의 이득은 작다** — random-init 3000 step(18.7/20.6/18.3) ≈ 증류 6000 step. 판정은 12k run 두 개로.
- [~] **12k step long run 2 종(순차, GPU 메모리 25 GB/run)**: random-init 12k(`u05-asr-noinit-12k`, 09:34 UTC 시작, ≈70 min) → 종료 후 **증류-init lr 1e-4 12k**(`u05-asr-distill-12k`, pid 파일 대기 체인) 자동 시작. 둘 다 관문 미달이면 adapter 용량/unfreeze 범위(Nemotron 상위 블록) 재설계로 이동(backbone 교체 없음, `decision-asr-backbone`).
- [ ] (운영 메모) 이전 항목의 lr 2e-4 run 손실 스파이크(1900→2000 step 0.56→1.37)는 lowlr 에서도 step 2000 근처(0.35→1.09)에 재현 — lr 무관한 데이터 배치 효과(특정 긴 발화?)로 보임. 12k 결과 후 확인.
- [x] thinker 주입 경로 — `<|audio_pad|>`(151676) 위치에 `inputs_embeds` scatter, 프롬프트 `…assistant\nlanguage {Lang}<asr_text>` 재현 (`vapasr/uslm/model.py`)
- [~] **오프라인 ASR 미세조정** — `experiments/u05_asr_finetune.py` (캐시 Nemotron 특징 발화 슬라이스 → adapter → thinker LoRA r16, 14.3M 학습, 121,827 발화). 스모크 통과. **random-init 대조 run(3000 step) 실행 중**; 증류 init run 은 block8s 증류 후
- [x] 기준선 (val 400 발화, jiwer, KO 는 공백 제거 CER):
      | 시스템 | otoSpeech WER | AI Hub 실내 CER | AI Hub 실외 CER |
      |---|---:|---:|---:|
      | 원본 Qwen3-ASR (AuT+thinker, 오프라인) | **13.9 %** | **13.5 %** | **13.3 %** |
      | Nemotron RNN-T `[56,0]` 80 ms 스트리밍 (태그 제거 후) | **25.4 %** | **23.4 %** | **24.5 %** |
      | Nemotron RNN-T `[56,13]` 1120 ms (태그 제거 후) | 19.0 % | 16.9 % | 17.0 % |
      | random-init adapter+LoRA, 3000 step (참고) | 18.7 % | 20.6 % | 18.3 % |
      ※ 첫 Nemotron 수치(35.8/37.0/34.5)는 출력의 언어 태그 `<ko-KR>` 가 문자로 세어져 ~10 pt 과대였다(정규화에서 태그 제거). 그래도 오프라인 Qwen 의 약 2배이며, chunk 크기 탓이 아니다. Nemotron RNN-T 는 짧은 자발적 발화(맞장구 '음' 등)에서 빈 가설을 내는 경향(예시 '음'→'') — 발화 길이별 오류율·빈 가설 비율을 확인할 것. 모델 카드 7.12 % 는 정제된 읽기 음성 기준.
      → **관문 분모를 재정의**: 스트리밍 RNN-T `[56,0]` 은 35.8 % 로 오프라인 Qwen 보다 훨씬 나쁘다. 관문은 "오프라인 Qwen3-ASR 대비 ≤ +15 %" 로 두고, 스트리밍 RNN-T 는 참고선.
- [ ] **관문**: 상대 열화 ≤ 10–15 % → U1 착수. 실패 → adapter 용량·정렬·초기화·encoder/thinker unfreeze 범위를 재설계하고 재평가. backbone 자동 교체 없음
- [ ] 12.5 Hz 그대로 vs 13 Hz 리샘플 ablation(작게)

## 진행 기록

- 2026-09-04: random-init 대조 step 1000: otoSpeech WER 29.6 % / 실내 CER 40.6 % / 실외 27.7 % (사용자 지시로 참고용, 이미 실행 중이라 완주).
- 2026-09-04: block8s 타깃(35 h) 추출 완료 → 정식 증류 실행. 스크립트 4종 완성(distill / baselines / finetune / data·model 모듈).
- 2026-09-04: 착수. `experiments/u05_distill_adapter.py` 작성, 증류 타깃용 `qwen-aut-block8s`(학습 분포) 특징 35 h 추출 중; 빠른 신호로 cc1s 타깃 증류 실행 중.
- 2026-09-04: 생성 (최종 backbone 확정에 따라).

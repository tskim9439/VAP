---
type: output
status: stable
created: 2026-09-03
updated: 2026-09-03
summary: backbone 독립 VAP target 파이프라인 — 코퍼스 3종 → 채널 VAD@50Hz npz → 로드 시 VAP 256-class·hazard·이벤트 파생, 20s 창 55,139개
sources:
  - [[source-conversation-corpora]]
  - [[source-turnbench]]
  - [[voice-activity-projection]]
---

# VAP target 파이프라인

코드: `vapasr/data/` (`corpora.py`, `vad.py`, `targets.py`, `dataset.py`), 러너: `experiments/build_targets.py`,
`experiments/recompute_events.py`, `experiments/test_dataset.py`. 원본 통계·QC: `raw/sources/experiments/2026-09-03-target-pipeline/`.

## 설계

```text
코퍼스 리더 ──→ Conversation(audio 16k (2,T), vad 세그먼트, utterances, meta)
  aihub        stereo wav + JSON        VAD = 채널별 에너지 (라벨 EndTime 은 넉넉함)
  otoSpeech    spk wav 48k + SRT        VAD = 라벨(speech 범주; Channel Bleed/Noise 제외)
  turnbench-dev parquet(FLAC) + annot   VAD = 라벨
        │
        ▼  build_targets.py
  npz: vad@50Hz (T,2) · tau_bin · censored · events[(t, spk, type)] · manifest.jsonl · stats.json · qc/*.png
        │
        ▼  WindowDataset(frame_hz=50 | 12.5)  — 로드 시 파생
  vap_label (T,)  ← 원 VAP ObjectiveVAP.get_labels (50 Hz bins 10/20/30/40, 12.5 Hz bins 2/5/8/10 = 2.0 s)
  tau_bin (T,2)   ← 화자별 다음 onset 까지 시간, 경계 [0.16,0.32,0.64,1.28,2.56] s, 마지막 = censored
  events          ← SHIFT / HOLD / INTERRUPT / BACKCHANNEL (아래 규칙)
```

**backbone 독립**: VAD 를 50 Hz 로만 저장하고 라벨은 요청 frame_hz 로 파생한다. 12.5 Hz 는 any-pool.
**원 VAP 와 호환**: 256-class 라벨은 [[output-vap-turnbench-baseline-reproduction]] 의 baseline 과 같은 코드(`vap.objective`)로 만든다.

## 이벤트 휴리스틱 (검증 필요 — [[question-event-label-derivation-validity]])

| 이벤트 | 규칙 (행위자) | 시각 |
|---|---|---|
| SHIFT | A 세그먼트 종료 후 ≥0.2 s 침묵, 3.0 s(TurnBench 매칭창) 안에 첫 발화가 B | A 의 종료 |
| HOLD | 위와 같되 첫 발화가 A 이거나 아무도 없음 | A 의 종료 |
| SHIFT (terminal overlap) | B 가 A 발화 중 시작했지만 A 가 0.5 s 안에 끝나고 재개하지 않음 | A 의 종료 |
| INTERRUPT | B 가 A 발화 중 시작, A 가 0.5 s 넘게 더 말한 뒤 끝나고 1.0 s 안에 재개하지 않음 | B 의 onset |
| BACKCHANNEL | B 가 A 발화 중 시작, B 세그먼트 ≤ 1.0 s 이고 A 가 계속 말함 | B 의 onset |
| (없음) | 종료 시점에 상대가 이미 말하는 중 (방해당함 / backchannel 종료) | — |

합성 VAD 단위 테스트로 5개 케이스를 확인했다. 초기 버전의 결함 두 가지(backchannel 종료·방해당한 종료의 가짜 SHIFT,
terminal overlap 을 전부 INTERRUPT 로 판정)는 QC 검수와 통계(INT > SHIFT)로 발견해 고쳤다.

## 결과

| 코퍼스 | 대화 | 시간 | SHIFT | HOLD | INT | BC | 플래그 |
|---|---:|---:|---:|---:|---:|---:|---|
| otoSpeech | 420 | 104.9 h | 28,798 | 42,154 | 2,246 | 12,457 | 한 채널 무음 4 |
| AI Hub VS_02 | 186 | 51.7 h | 35,392 | 43,785 | 2,302 | 28,599 | 30 s 미만 7 |
| **AI Hub TS_01.실내_5** | 757 | **196.6 h** | 121,020 | 206,712 | 6,618 | 51,520 | 한 채널 무음 30, 30 s 미만 20 |
| TurnBench dev | 38 | 7.3 h | 992 | 1,658 | 318 | 753 | 한 채널 무음 1 |

- 20 s 창(hop 10 s) **55,139개** (AI Hub VS_02 + otoSpeech); TS_01.실내_5 추가 시 총 **약 360 h**.
- TS_01.실내_5 는 zip 21.8 GB → 해제 43 GB(wav 가 2× 압축됨) → **196.6 h**. zip 크기 기반 추정(95 h)은 2× 과소였다. AI Hub 전체는 그러면 ~2,700 h 로 라벨 통계와 일치.
- TS_01.실내_5 화자↔채널 뒤바뀜 26/757, 한 채널 무음(<5 %) 30 파일 — Stage 1 학습에서 플래그 파일 제외. 50 Hz: (1000,2) VAD, 라벨 유효 90 %; 12.5 Hz: (250,).
- 상위 라벨 0 / 15 / 240 = 둘 다 침묵 / A 4구간 / B 4구간 — 정상 분포.
- AI Hub 화자↔채널 매핑을 라벨–VAD 겹침으로 파일별 자동 판정: **3/186 파일이 뒤바뀜** (확신도 중앙값 0.845). 라벨 없는 14 s 파일 1개 제외.
- 에너지 VAD vs 라벨 VAD 프레임 일치: otoSpeech 0.84, TurnBench dev 0.69 (bleed·저레벨 FLOAT 오디오) — 라벨 코퍼스에서는 라벨 VAD 를 쓴다.
- TurnBench dev 의 gold 이벤트(EOT 1,904 / INT 347)와 휴리스틱(SHIFT 992 / INT 318)의 차이가 크다 → **휴리스틱 검증은 즉시 가능**
  (gold 는 scorer 데이터에 있음). INT 수는 비슷하고 SHIFT 는 절반 — gold EOT 정의가 더 넓다(같은 화자 재개도 EOT?).

## 남은 것 / 다음

- [[task-event-label-heuristics-validation]] — TurnBench dev gold 로 즉시 검증 가능 (예상보다 빨리).
- 로더 속도: otoSpeech 48 k→16 k 리샘플이 항목당 300 ms — 16 k 사본 또는 특징 캐시 ([[task-compute-budget-and-feature-cache]]).
- CANDOR 리더는 데이터 확보 후 ([[task-secure-english-corpora]]).
- AI Hub TS_01.실내_5(≈95 h) 수신 완료 시 같은 명령으로 빌드.

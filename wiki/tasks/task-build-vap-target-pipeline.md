---
type: task
status: done
owner: tskim
due: 2026-10-01
priority: p0
created: 2026-09-03
updated: 2026-09-03
summary: stereo 오디오에서 VAP 256-class·VAD·유도 이벤트 라벨을 만드는 backbone 독립 파이프라인
sources:
  - [[output-streaming-vap-research-plan]]
---

# VAP target 생성 파이프라인

## 배경

[[decision-asr-backbone]] 이 2단계(NeMo → Transformers)로 가므로
**데이터·평가 코드는 backbone 에 독립적이어야** 두 번 구현하는 비용이 줄어든다.

## 완료 조건

- [x] stereo → 화자별 VAD — **에너지 기반이 1차** (`vapasr/data/vad.py`, 히스테리시스·최소길이). AI Hub 라벨 시간은 발화 구간이 넉넉해(교대의 50 % 가 '겹침', 중앙값 1 s) VAD 경계로 쓸 수 없다 ([[source-conversation-corpora]]). 라벨은 화자·텍스트용
- [x] VAP 256-class target — 원 VAP `ObjectiveVAP.get_labels` 재사용 (baseline 과 동일 코드)
- [x] frame rate 파라미터 — 50 Hz 저장, 12.5 Hz 는 any-pool + bins 2/5/8/10 (=2.0 s)
- [x] τ target — `time_to_next_onset`: 화자별 다음 onset(≥0.2 s 침묵 뒤 시작), 경계 [0.16,0.32,0.64,1.28,2.56] s, censored 플래그
- [x] 유도 이벤트 규칙 — [[output-vap-target-pipeline]] 표. 합성 테스트 5케이스 + QC 검수로 결함 2건 수정
- [x] AI Hub / otoSpeech / TurnBench dev 동일 인터페이스 (CANDOR 는 데이터 확보 후)
- [x] QC PNG — AI Hub 50 / TurnBench 10 / otoSpeech 30 생성, 6장 직접 검수 (`raw/sources/experiments/2026-09-03-target-pipeline/qc/`)

## 진행 기록

- 2026-09-03: **완료.** 코퍼스 3종 manifest 빌드 (156 h, 20 s 창 55,139개), `WindowDataset` end-to-end 확인. → [[output-vap-target-pipeline]]
- 2026-09-03: 생성.

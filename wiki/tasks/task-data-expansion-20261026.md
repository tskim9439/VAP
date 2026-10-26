---
type: task
status: doing
owner: tskim
due: 2026-11-02
priority: p1
created: 2026-10-26
updated: 2026-10-26
summary: 학습 데이터 확장 — 원거리 회의 녹음(AMI 어레이·ICSI 탁상·NOTSOFAR 장치·CHiME-6), 화자별 원거리 증강, 겹침 늘린 합성, 한국어 회의·Switchboard → F1 → E5·E6
sources:
  - '[[output-diarization-integration-20261025]]'
  - '[[task-secure-meeting-corpora]]'
  - '[[output-phase2-data-inventory]]'
---

# 학습 데이터 확장 (2026-10-26)

사용자: "데이터도 더 확보하가나 증강해서 학습량을 늘릴순 없을까?" → 제안 → "모두 진행하는 데, 모두 진행하고 나서 E5 E6 학습 진행하자".

## 왜 이 데이터인가
- 회의 학습 오디오(AMI 77 h·ICSI 54 h·NOTSOFAR 4 h)가 전부 **근접 마이크** 혼합이다. 평가 AMI SDM 은 원거리 마이크 하나다.
  - 같은 모델의 화자 무시 오류(agn)가 IHM-mix 21.7 → SDM 28.3 이다. 정답 화자 분리 E2 에서도 21.2 → 28.6 이다.
- 잔향·잡음 증강은 단일 화자 셋에만 걸렸다(잔향 0.2). 대화·회의 창에는 전화 대역 증강만 있었다.
- 합성 다자 대화(v3-multi)는 2 인 대화의 교대 간격 통계를 써서 겹침이 말한 시간의 4.4–6.7 % 뿐이다. AMI 평가 창은 참조 발화 프레임의 24 % 가 겹친 구간이다.
  - 겹친 구간은 받아쓰기 오류의 47 %, E1 화자 분리 miss 의 46 % 를 차지한다.

## 항목과 상태
| # | 항목 | 상태 |
|---|---|---|
| D1 | NOTSOFAR train 원거리 장치(어레이 4 대 ch0 + 단일 마이크 5 대) → `notsofar-far` | 빌드 완료: 72 회의, 595 변형(지연 0–30 ms, 포락선 상관 0.81). mono 캐시 진행 중 |
| D2 | ICSI 탁상 마이크(.mrt 의 참가자 아닌 채널) → `icsi-far` | 빌드 완료: 69 회의, 258 변형(PZM c2 위주, 저가 u3 는 상관이 낮아 버림). mono 캐시 진행 중 |
| D3 | AMI Array1 8 채널(Full-corpus-ASR train 136 회의 + IB4005 중 Array1 이 있는 135 회의) → `ami-far` | 맥 → T5 받는 중(동시 4, 약 2.6 MB/s, 75 GB 어림). 업로드는 사용자 |
| D4 | CHiME-6 train(16 세션, 착용 마이크 + Kinect 6 대 CH1) → `chime6`·`chime6-far` | 압축 풀기 중(VAPKT-data/data/audio/chime6/CHiME6_train). 다음: 대화 빌드 → 정렬 → 정제 |
| D5 | 화자별 원거리 증강(`far_aug`) — 화자별 신호 캐시(stems), 같은 방의 다른 RIR | 코드·시험 완료. stems 캐시(실제 대화 약 290 GB 어림)는 다음 |
| D6 | 겹침 늘린 합성(E4) — 회의 교대 통계, 끼어들기 확률, 화자별 원거리 | 설계 중 |
| D7 | 한국어 회의 NIA23 002_Meeting(7,983 회의, 혼합 녹음의 발화 조각 — B 등급) | 경로 확인. 읽기 코드 → 정렬 → 정제 |
| D8 | Switchboard(영어 2 인 전화, 복원 시 약 230 h) | 경로 확인. 읽기 코드 → 정렬 → 정제 |
| F1 | 회의 + 4 슬롯 1.7B 레시피(같은 init·step)에 D1–D8 + 원거리 증강 → AMI·TAXI·FLEURS | 데이터가 모이면 |
| E5·E6 | 한 스트림 + 화자 활동 입력([[output-diarization-integration-20261025]] 6 절) | F1 뒤 |

## 원거리 버전 규칙
- 근접 마이크로 정렬·정제된 대화를 복사한다. 발화·토큰·플래그는 그대로이고, `meta.mono_src` = {장치 파일, 채널, offset_s} 만 더한다.
  - `dialogue_mix.far_mono` 가 장치 녹음을 대화 시간축에 놓고 발화 구간 RMS 를 −23 dBFS 로 맞춘다.
- 시각 맞춤: 변형마다 근접 mono 와 장치 녹음의 10 ms 로그 에너지 포락선 상호상관을 발화가 많은 60 s 구간 3 개에서 잰다(±2 s).
  - 세 구간이 50 ms 안에서 맞고 상관이 0.3 이상일 때만 쓴다.
  - ICSI 의 음수 지연(−50–0 ms)은 헤드셋이 무선(Transmission w1)이라 늦기 때문이다. 정답 시각은 헤드셋 기준이다.
- held-out(split v4)과 split ≠ train 회의는 넣지 않는다. 학습에서는 새 그룹 `dlg_far`(영어)로 따로 가중한다.

## 코드
- `vapasr/data/dialogue_mix.py`(far_mono·mix_stems), `vapasr/data/dialogue_dataset.py`(stems 캐시), `vapasr/data/far_aug.py`(새), `vapasr/data/cst_dialogue.py`(far_aug 훅)
- `experiments/p2_build_far.py`(새), `experiments/p2_build_mono.py`(--stems), `experiments/stage3_train.py`(원거리 코퍼스·dlg_far·--far-aug*)
- 시험: `tests/test_far_data.py`, `tests/test_far_aug.py`
- 실행: `raw/sources/experiments/2026-10-26-data-expansion/`(ami_array_download.sh, far_build.sh, 회의 목록)

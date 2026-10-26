---
type: task
status: doing
owner: tskim
due: 2026-11-02
priority: p1
created: 2026-10-26
updated: 2026-10-26
summary: 학습 데이터 확장 — 원거리 회의(AMI 어레이·ICSI·NOTSOFAR·CHiME-6 train), 화자별 원거리 증강, 겹침 합성, 한국어 회의·Switchboard → F1 → E5·E6
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
| D1 | NOTSOFAR train 원거리 장치(어레이 4 대 ch0 + 단일 마이크 5 대) → `notsofar-far` | 완료: 72 회의, 595 변형(지연 0–30 ms, 포락선 상관 0.81), mono 캐시 6.6 GiB |
| D2 | ICSI 탁상 마이크(.mrt 의 참가자 아닌 채널) → `icsi-far` | 완료: 69 회의, 258 변형(PZM c2 위주, 저가 u3 는 상관이 낮아 버림), mono 캐시 27 GiB |
| D3 | AMI Array1 8 채널(Full-corpus-ASR train 136 회의 + IB4005 중 Array1 이 있는 135 회의) → `ami-far` | 완료: T5 받기 1,080 파일 69 GB(ES2012d.Array1-02 앞 331 B 서버 오류 페이지 → 다시 받아 교체, 스크립트 `--fail`) → 사용자 azcopy 업로드 `users/tskim/VAPKT-data/data/audio/ami-array1-train`(SAS 는 `DB/raw` 쓰기 403) → 서버 대조 135·1,080·74,174,109,160 B·RIFF → 학습 회의 131 × 8 = 1,048 변형 중 1,032(버림 16, 지연 중앙 10 ms, 상관 0.79) mono 66 GiB |
| D4 | CHiME-6 train(착용 마이크 + Kinect CH1) → `chime6tr`·`chime6tr-far` — 키를 dev 평가 셋 `chime6` 과 나눈다([[decision-chime6-train-sessions-20261026]]) | 완료: 정렬 95744(16 세션, 71,010 발화) → 정제 → 죽은 착용 마이크 발화 표시 7.4 %·S05 제외 → 15 세션 mono 4.0 GiB·stems 16.1 GiB, 원거리 89 변형(버림 0, 지연 중앙 10 ms, 상관 0.84) mono 23.9 GiB |
| D5 | 화자별 원거리 증강(`far_aug`) — 화자별 신호 캐시(stems), 같은 방의 다른 RIR | 완료: 실제 대화 10 셋 5,613 대화 351 GiB(실패 8 — aihub134 의 채널 없는 화자, 그 대화는 혼합 신호 하나에 RIR), Switchboard·CHiME-6 train 도 완료(한국어 회의는 혼합 조각이라 없음) |
| D6 | 겹침 늘린 합성(E4) → `sim_<l>-meet`(sim-dialogue-v4-meet) | 완료: en·ko·de 각 10,000 세션(260·252 h …). 영어 겹침 11.7 %(v3-multi 4.4 %), 교대 13.8 회/분, 맞장구 4.4 회/분, 70 % 세션 화자별 원거리 |
| D7 | 한국어 회의 NIA23 002_Meeting → `nia23mt`(dlg_mtko) | 완료: 전체 7,971 회의 5,621 h 중 빠진 발화 없는 회의에서 종류별로 골라 891 회의 651 h(녹음 251·방송 250·라디오 150 h — 인터넷은 대부분 임시 파일만 남아 제외). 정렬(95691) → 정제(발화 362,025, 미정렬 1,349·이상 2,000 제외) → phase2 → mono 69.9 GiB |
| D8 | Switchboard(영어 2 인 전화) → `swbd`(dlg_en) | 완료: 2,435 통화, 258 h. 정렬(95683, 5 분) → 정제 → phase2 → mono·stems |
| F1 | 회의 + 4 슬롯 1.7B 레시피(같은 init·step)에 D1–D8 + 원거리 증강 → AMI·TAXI·FLEURS | 제출 준비 완료(10-26 21:19): 모든 셋으로 dry-run 18 그룹 δ4–10 불일치 0(`dlg_far` 원거리 4 코퍼스 1,974 대화 101,585 창), 20 step 스모크·저장 대조 통과(`ami-far` 앞), 서버 코드 492 파일 md5 일치. 제출은 사용자(`bash slurm/submit-q17-f1.sh`, apex 2 노드) |
| E5·E6 | 한 스트림 + 화자 활동 입력([[output-diarization-integration-20261025]] 6 절) | 코드·시험 완료(E5 adapter_slot·act_noise·--act-oracle, E6 인과 diar_head·ρ 혼합·encode 활동 스트림, E1c 인과 가지). 학습은 F1 뒤 |

## 원거리 버전 규칙
- 근접 마이크로 정렬·정제된 대화를 복사한다. 발화·토큰·플래그는 그대로이고, `meta.mono_src` = {장치 파일, 채널, offset_s} 만 더한다.
  - `dialogue_mix.far_mono` 가 장치 녹음을 대화 시간축에 놓고 발화 구간 RMS 를 −23 dBFS 로 맞춘다.
- 시각 맞춤: 변형마다 근접 mono 와 장치 녹음의 10 ms 로그 에너지 포락선 상호상관을 발화가 많은 60 s 구간 3 개에서 잰다(±2 s).
  - 세 구간이 50 ms 안에서 맞고 상관이 0.3 이상일 때만 쓴다.
  - ICSI 의 음수 지연(−50–0 ms)은 헤드셋이 무선(Transmission w1)이라 늦기 때문이다. 정답 시각은 헤드셋 기준이다.
- held-out(split v4)과 split ≠ train 회의는 넣지 않는다. 학습에서는 새 그룹 `dlg_far`(영어)로 따로 가중한다.

## F1 점검에서 고친 것 (2026-10-26)
- mxc 코드가 E5·E6 커밋 전 버전이었다(`vapasr/data/act_noise.py` 없음, 모델·트레이너·데이터셋 등 18 파일이 예전 커밋).
  - 앞선 dry-run(r3)은 서버의 예전 코드로 통과한 것이었다.
  - 63 파일을 맞추고 추적 코드 484 파일의 md5 를 대조했다. 서버에서 따로 고친 파일은 없었다.
- 번역 타깃 행이 하나도 없는 대화는 학습 중 번역 모드로 뽑지 않는다(`CstDialogueDataset.has_trans`).
  - 해당 대화: swbd 2,435·nia23mt 891·chime6tr 15·chime6tr-far 89. 기존 코퍼스는 aihub 26 대화(0.4–1.9 %)뿐이다.
  - 바꾸기 전에는 `--dlg-ast-frac 0.5` 로 창 절반이 번역 모드가 됐다. 그 창에서는 번역 없는 행 규칙으로 태그·`<NEXT_AUDIO>`·`<EOT>` 가 모두 가려져 전사 글자만 학습됐다.
- 번역 커버리지 통계가 원거리 별칭(`far_of`)을 쓰게 고쳤다. `dlg_far` 는 0 으로 보였는데 실제는 476,253/820,800 행이다(시퀀스는 원래 맞았다).

## CHiME-6 train 처리 (2026-10-26)
- 사용자 선택: "train 세션만 학습에 넣기"([[decision-chime6-train-sessions-20261026]]). dev(S02·S09)·eval(S01·S21)은 평가에만 쓴다.
- 학습 키는 `chime6tr` 이다. 예전 평가 스크립트 3 개가 `<corpus>.refined` 를 먼저 찾기 때문에, `chime6.refined` 를 만들면 dev 평가가 train 세션으로 바뀐다.
- 착용 마이크 녹음 레벨이 낮다(발화 RMS −35 ~ −48 dB). 그래서 정제 VAD(절대 하한 −40 dBFS)가 발화 51 % 에서 켜지지 않았다(AMI 17 %). 그 발화는 정렬 토큰 시각으로만 끝을 당겼다.
- 죽은 착용 마이크를 처리했다(`experiments/p2_flag_channel_level.py`).
  - 기준: 자기 채널의 발화 구간 dB 중앙값이 채널 바닥 + 6 dB 아래인 발화에 `own_ch_inaudible` 를 붙인다. 학습은 그 구간을 손실에서 가린다.
  - S05 는 P14 채널이 세션 내내 죽어(98 %, 혼합 게인 +65 dB) 뺐다.
  - 표시 비율: S23 P54 94 %, S04 P11 76 %, S03 P11 32 %, S22 세 화자 18–25 %, 전체 7.4 %.
- 실행: `raw/sources/experiments/2026-10-26-data-expansion/chime6tr_finish.sh`. 원거리 빌더는 `p2_build_far.py --corpus chime6 --key chime6tr` 다.

## 코드
- `vapasr/data/dialogue_mix.py`(far_mono·mix_stems), `vapasr/data/dialogue_dataset.py`(stems 캐시), `vapasr/data/far_aug.py`(새), `vapasr/data/cst_dialogue.py`(far_aug 훅)
- `experiments/p2_build_far.py`(새), `experiments/p2_build_mono.py`(--stems), `experiments/stage3_train.py`(원거리 코퍼스·dlg_far·--far-aug*)
- `experiments/p2_flag_channel_level.py`(새 — 자기 채널에서 들리지 않는 발화 표시), `p2_build_far.py --key`(학습용 파일 키)
- 시험: `tests/test_far_data.py`, `tests/test_far_aug.py`
- 실행: `raw/sources/experiments/2026-10-26-data-expansion/`(ami_array_download.sh, far_build.sh, 회의 목록)

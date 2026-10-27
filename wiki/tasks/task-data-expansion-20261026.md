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
| F1 | 회의 + 4 슬롯 1.7B 레시피(같은 init·학습률)에 D1–D8 + 원거리 증강 → AMI·TAXI·FLEURS | 완료(학습 95865 2.4 h, 평가 95866 45/45 — 아래 "F1 결과"): AMI dev 개선(귀속), SDM eval 그대로, TAXI 유의하게 나빠짐(+2.0, 받아쓰기), avg3 는 FLEURS ko-en δ10 삭제 회귀 |
| E5·E6 | 한 스트림 + 화자 활동 입력([[output-diarization-integration-20261025]] 6 절) | 제출(10-27 09:27, 사용자, `slurm/submit-q17-e56.sh` — F1 레시피·평가 기준): E1c 95932(1 노드, 인과 가지 10,000 step, ck3000 인코더 위, 원거리·새 코퍼스 포함) · E5 학습 95933·평가 95934(q17-e5) · E6 학습 95935(E1c 성공 뒤)·평가 95936(q17-e6 = F1 묶음 + 정답 활동 상한). 준비 중 E5 다중 GPU 버그(replay 행에서 adapter_slot 미사용 → DDP 정지)를 고쳤다. E1c 완료(11:07) — AMI 최적 짝 DER eval IHM-mix 23.08 · SDM 27.04(임계 0.45 = dev 최적, E1 양방향보다 약 +4, 주로 놓침). E6 은 F1 레시피의 LLM 쪽 화자 헤드(가중 0.5)를 그대로 둔다(계획 6 절의 "끈다"와 다르다 — E5·F1 과 같은 조건이라 비교에는 낫다). E7(AOSC)은 코드 없음. **E5 결과(10-27, amiact 정답 활동, F1 final 대비 회의 bootstrap)**: AMI δ10 SDM eval 53.94 → 29.24(−24.7*), IHM-mix eval 49.43 → 23.39(−26.0*), 귀속 몫 1.0–2.1 — E2(28.6) 수준 → 판단 기준 통과(한 스트림 형식이 화자별 디코드 상한에 닿는다), E6 그대로 |

## 원거리 버전 규칙
- 근접 마이크로 정렬·정제된 대화를 복사한다. 발화·토큰·플래그는 그대로이고, `meta.mono_src` = {장치 파일, 채널, offset_s} 만 더한다.
  - `dialogue_mix.far_mono` 가 장치 녹음을 대화 시간축에 놓고 발화 구간 RMS 를 −23 dBFS 로 맞춘다.
- 시각 맞춤: 변형마다 근접 mono 와 장치 녹음의 10 ms 로그 에너지 포락선 상호상관을 발화가 많은 60 s 구간 3 개에서 잰다(±2 s).
  - 세 구간이 50 ms 안에서 맞고 상관이 0.3 이상일 때만 쓴다.
  - ICSI 의 음수 지연(−50–0 ms)은 헤드셋이 무선(Transmission w1)이라 늦기 때문이다. 정답 시각은 헤드셋 기준이다.
- held-out(split v4)과 split ≠ train 회의는 넣지 않는다. 학습에서는 새 그룹 `dlg_far`(영어)로 따로 가중한다.

## F1 결과 (2026-10-27, 평가 95866 45/45, `raw/sources/experiments/2026-10-26-data-expansion/f1_compare.sh`)
- 기준은 mt4(q17-v3-p1-tel-xlw-mt4)다.
- 검정은 두 가지다. AMI 는 회의 단위 짝 bootstrap, TAXI 는 세션 단위 짝 bootstrap 이다(각 1,000 회).
- 아래 차이는 F1 − mt4 이고, `*` 는 95 % 구간이 0 을 포함하지 않는다는 뜻이다.

| 지표 | mt4 final | F1 final | F1 avg3 | F1 final − mt4 final [95 %] |
|---|---:|---:|---:|---|
| dev tel cp δ10(선택) | 19.74 | 19.03 | 19.19 | (점추정 −0.71) |
| TAXI cp δ10(실제 독↔영 전화) | 18.9 | 20.9 | 19.9 | +2.04 [+0.30, +4.09]* — agn +1.45*(영어 WER 23.6→26.0, 독일어 13.8→15.4) |
| AMI IHM-mix dev δ10 | 58.14 | 54.82 | 53.46 | −3.33 [−4.52, −2.13]* — 귀속 −3.84* |
| AMI IHM-mix eval δ10 | 49.91 | 49.43 | 48.34 | −0.48 n.s.(avg3 −1.57*) |
| AMI SDM dev δ10 | 61.39 | 58.95 | 57.76 | −2.45 [−3.95, −0.89]* |
| AMI SDM eval δ10 | 53.62 | 53.94 | 53.37 | +0.33 [−1.03, +1.79] n.s. |
| 같은 언어 대화 cp δ10 | 25.91 | 26.31 | 25.52 | 가드 안 |
| 대화 BLEU · xl BLEU δ10 | 40.06 · 39.83 | 39.71 · 39.88 | 39.97 · 39.94 | 가드 안 |

- **AMI**: 좋아진 것은 dev 다(IHM −2 ~ −4.7, SDM −1.3 ~ −3.6). 거의 다 화자 귀속 몫이고, agn 은 같거나 조금 나빠졌다.
  - eval 은 IHM 이 avg3 기준으로만 조금 낫고(−1.1 ~ −1.6), 목표였던 **SDM eval 은 변화가 없다**.
- **TAXI**: 모든 F1 체크포인트가 나쁘다(ck3000 19.7·ck4000 19.6·final 20.9, mt4 는 18.3–18.9).
  - 이 차이는 받아쓰기(agn) 쪽이고, 영어·독일어 모두 늘었다.
  - final 은 후보 기준(TAXI cp < 20.5)을 넘었다.
  - 원인 후보는 셋이다. 영·독 대화에도 건 원거리 증강(0.3), 2 인 전화와 다른 회의형 데이터 비중, Switchboard 다.
- **단일 화자 가드**: final 은 최대 +0.75 로 통과했다.
  - ck4000·avg3 는 FLEURS ko-en δ10 CER 이 20.8–21.1 % 다(다른 판 8.6–9.4 %).
  - 원인은 삭제다(592–628 대 180–200). 일부 발화에서 한국어 전사를 끝까지 내지 않는다. 번역 BLEU 는 거의 그대로다. avg3 는 이 가드를 넘는다.

## step 수와 그룹 가중 (2026-10-26)
- 사용자 질문: "데이터도 늘었는데, step 수를 늘려야 하는거 아닐까?"
- mt4 실측(95326): 2,500 step 에 학습 59 분(1.33 s/step), epoch 1,469 step(epoch 기준 = 실제 대화 셋 `dlg_*` 를 한 번 다 도는 step 수) → 1.70 회.
- F1 을 2,500 step 그대로 두면 새 그룹(원거리·한국어 회의·회의형 합성)이 몫을 나눠 기존 셋 노출이 준다. 실제 대화 셋은 0.79 회만 돈다.
  - 한국어 대화 397 → 327 배치(−18 %), 회의 근접 199 → 115(−42 %).
  - Switchboard 가 `dlg_en` 의 창 82 % 를 차지해, 번역 타깃이 있는 otoSpeech 가 140 → 20 배치(−85 %)로 준다.
- 사용자 선택 "5,000 step + Switchboard 분리".
  - 실제 대화 셋을 1.69 회 돌고, 기존 셋이 모두 mt4 이상 노출된다(otoSpeech 219·회의 근접 221).
  - 평가 체크포인트는 같은 상대 위치(60·80 %).
  - mt4 와의 차이에는 데이터와 step 수가 섞인다. 나누려면 mt4 레시피 5,000 step 대조 run 이 필요하다.

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

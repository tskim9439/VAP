---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-11-09
created: 2026-10-26
updated: 2026-10-26
summary: CHiME-6 train 세션(S05 제외)을 학습에 넣고 dev·eval 은 평가에만 둔다. 학습 키 chime6tr, CHiME-6 평가는 같은 도메인, 처음 보는 코퍼스 평가는 DiPCo
sources:
  - '[[task-data-expansion-20261026]]'
related:
  - '[[decision-phase2-eval-plan-v2]]'
  - '[[decision-eval-train-decontamination]]'
  - '[[output-data-selection-execution-plan]]'
---

# CHiME-6 train 세션을 학습에 넣는다 (2026-10-26)

## 맥락
- 데이터 확장([[task-data-expansion-20261026]])에서 겹침이 많은 실제 원거리 회의 데이터를 늘리려고 CHiME-6 train 을 정리했다. 16 세션, 착용 마이크(바이노럴) + Kinect 6 대, 40.1 h, 발화 71,811 개다.
- 충돌하는 기존 기록이 둘 있다.
  - 2026-09-17 평가 계획 v2([[decision-phase2-eval-plan-v2]])는 CHiME-6/DiPCo 를 객관 held-out 으로 정했다.
  - 데이터 선별 실행안([[output-data-selection-execution-plan]])은 `chime6` 를 학습 선별에서 뺐다.
- 데이터 확장 제안은 이 충돌을 적지 않았다. CHiME-6 정렬(job 95744)이 끝난 뒤, 정제 결과를 phase2 로 옮기기 전에 확인해 사용자에게 물었다.
- 평가-학습 겹침 원칙([[decision-eval-train-decontamination]])은 데이터셋 이름이 아니라 녹음 단위로 막는다. CHiME-6 의 train·dev·eval 은 세션(녹음)과 참가자가 서로 다르다.

## 선택지
1. CHiME-6 를 빼고 F1 을 진행한다 — held-out 결정을 그대로 둔다.
2. train 세션만 학습에 넣는다 — dev·eval 세션은 평가에만 쓴다(CHiME 챌린지 표준 분할).
3. F1 에서는 빼고 다음 학습부터 넣는다.

## 결정
사용자 선택(2026-10-26): **2. train 세션만 학습에 넣는다.**
- **학습 셋**
  - 근접 혼합 `chime6tr`: 영어 회의, dlg_mt 그룹.
  - Kinect 원거리 버전 `chime6tr-far`: dlg_far 그룹.
- **평가에만 쓰는 세션:** dev(S02·S09)·eval(S01·S21). 기존 평가 파일 `phase2/chime6.dialogues.jsonl`(dev)은 그대로 둔다.
- **학습 키는 `chime6tr`:**
  - 예전 평가 스크립트(p2_eval_lanes·p2_eval_sessions·s3_eval_multispeaker_mono)는 `<corpus>.refined.dialogues.jsonl` 을 먼저 찾는다. 그래서 학습 파일을 `chime6.refined` 로 두면 dev 평가가 조용히 train 세션 평가로 바뀐다.
  - `stage3_train.py` 는 `chime6` 키를 받지 않는다.
  - 원거리 빌더는 `p2_build_far.py --corpus chime6 --key chime6tr` 로 돌린다.

## 데이터 품질 처리 (2026-10-26 mxc 실측)
- **착용 마이크 레벨:** 녹음 레벨이 낮다(화자별 발화 RMS 대부분 −35 ~ −48 dB).
  - 정제의 에너지 VAD 는 절대 하한이 −40 dBFS 라, 발화 51 % 에서 켜지지 않았다(AMI 17 %, NOTSOFAR 4 %).
  - 그 발화는 정렬 토큰 시각으로만 끝을 당겼고, 시작은 전사 시각 그대로다.
- **마이크가 죽은 채널:** 자기 채널에서 발화 구간 dB 중앙값이 채널 바닥(10 백분위) + 6 dB 아래인 발화를 셌다.
  - S05 P14 는 98 % 이고 mono 혼합 게인이 +65 dB 다(잡음을 키운다) → **S05 세션 제외**.
  - S23 P54 94 %, S04 P11 76 %, S03 P11 32 %, S22 세 화자 18–25 % 이고, 나머지 채널은 대부분 5 % 아래다.
  - 이런 발화에는 flags `own_ch_inaudible` 를 붙였다(`experiments/p2_flag_channel_level.py`). 정렬이 죽은 채널 위에서 나왔기 때문이다.
  - 학습은 그 구간을 전사 없는 음성처럼 손실에서 가리고, 화자 활동 타깃은 유지한다.

## 결과
- 이 결정 뒤의 학습부터, CHiME-6 dev·eval 평가는 "처음 보는 코퍼스"가 아니라 "같은 도메인의 처음 보는 세션" 평가다. 이전 모델의 CHiME-6 수치와 비교할 때 이 차이를 적는다.
- 처음 보는 코퍼스 평가(원거리 저녁 식사 대화)는 DiPCo 가 맡는다. NIKL 2020·TurnBench dev 는 그대로 held-out 이다.

## 재검토
- F1 결과가 나오면(검토일 2026-11-09) CHiME-6 dev 수치를 DiPCo 와 함께 본다.

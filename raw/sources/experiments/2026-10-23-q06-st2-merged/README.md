---
author_id: tskim
created: 2026-10-23
source_type: source-drop
---

# 0.6B 합친 Stage 2 시험 (`q06-st2-merged`, 2026-10-23)

사용자: "화자 태그는 대화에서부터 하고, 한 단계로 통합하면 안 좋으려나?" → "Apex 노드 2대 더 사용할 수 있어"

## 질문
- 지금 Stage 2(AST)는 세 번 이어 학습했다.
  - s2: 번역 모드 0.5, 2 epoch
  - s3: 번역 모드 0.8 + 영어 음량 증강, 2 epoch
  - s4: + Common Voice 4 영어와 그 번역 타깃, 2 epoch
  - 합 43,436 step 이다.
- 2·3 차는 미리 설계한 커리큘럼이 아니다. 평가에서 드러난 문제(번역 손실, 작은 소리 침묵, 영→X 약함)를 고치려고 덧붙였다.
- 최종 설정으로 한 번만 학습해도 s4 만큼 되면, Stage 2 를 한 단계로 정리하고 시간을 줄일 수 있다.
  - 1.7B 기준으로 약 25 h → 약 12 h 다.

## 화자 태그는 그대로 둔다
- Stage 2 의 prefix(`language <SPK_x>…`, x 는 A/B 무작위)는 대화 형식(언어 고정 두 슬롯)에서 슬롯을 하나만 선언한 경우다([[decision-cst-training-stages-and-speaker-slots]]).
- 대화 단계는 Stage 2 단일 화자 데이터를 계속 섞는다(replay). 그 데이터에도 태그가 필요하다.
- 비용은 항목마다 토큰 몇 개다. 남은 TAXI 오류(목소리가 비슷할 때 교대를 놓침)와도 관계가 없다. 슬롯이 하나라 화자 구분은 원래 대화 단계에서 배운다.

## 설정 (`slurm/submit-q06-st2-merged.sh`)
- init: `semcommit-q06-s1m3-d10/final`(0.6B Stage 1)
- s4 의 최종 설정 그대로 4 epoch(29,720 step, s4 의 epoch 당 7,430 step × 4)
  - 번역 모드 0.8, 영어 음량 증강, CV4 영어와 그 번역 타깃
  - 나머지는 s4 판과 같다: δ4–10, [56,3], worker 4, lr 6e-5, warmup 500
- 학습량은 세 번 합계의 68 % 다. 같으면 시간까지 줄어드는지를 함께 본다.
- 실행: apex 2 노드, 1.7B 체인과 같은 노드 제외, MAX_RESTART 5
- 어림: 0.6B s4 는 평소 0.55–0.67 s/step, 저장소 경합 때 2–4 s(중앙 1.03) → 약 5–9 h + 시작 40 분

## 평가 (`slurm/submit-eval-ast.sh`, 학습 성공 뒤 afterok)
- s3·s4 와 같은 7 셋·설정이다.
  - 셋: FLEURS · CoVoST 2 sub7 · FLEURS 영어 작은 소리 · vpaa-kspon · yodas-de · LibriSpeech clean·other
  - 설정: δ4·6·8·10 + final, batch 64, --verify, fp32
- 산출: `results/<셋>-q06-st2-merged/`
- 비교 대상은 `q06-s4-ast/final` 이다. 방법은 2026-10-13 `cmp_s4_s3.py` 와 같다.
  - 번역: BLEU 짝 bootstrap(문장), chrF, LAAL, 빈 번역
  - 받아쓰기: 짝 bootstrap

## 제출 기록
- 2026-10-23 01:1x, 사용자: 학습 job 94850(01:12 시작, hpc-29·30) → 평가 job 94851(94850 성공 뒤 afterok, 44 작업)
- 시작 점검(01:52): init `q06-s1m3-d10/final`, 번역 모드 0.8, 영어 음량 증강, 번역 타깃 6,386,086 행(CV4 포함 6 파일), epoch 당 7,430 step × 4 = 29,720 step, worker 4.
  - 첫 손실은 step 10 4.30 → step 40 2.29 다.
  - 속도는 아직 시작 구간이다(step 40 2.1 s/step). s4 의 평소 속도는 0.55–0.67 s 였다.

## 미리 정한 판단 기준
- 다음 셋을 모두 만족하면 합친 방식을 채택한다.
  - 번역 BLEU 가 어느 방향(영→독·영→한·독→영·한→영)·δ 에서도 s4 보다 유의하게 낮지 않다.
  - 받아쓰기가 어느 셋에서도 유의하게 나빠지지 않는다.
  - 빈 번역(짧은 문장 누락)이 s4 수준(6 개 안팎)이다.
- s4 에서 생긴 한→영 δ10 하락(−2.0 BLEU)이 합친 방식에서도 생기는지 따로 본다.
- 채택하면 다음에 할 일:
  - 대화 단계(Stage 3·4)까지 다시 내서 마지막 성능(TAXI·dev 귀속·가드)을 0.6B R1x 와 비교한다(약 2.5 h).
  - 그 뒤 이후 run 과 논문은 Stage 2 를 한 단계로 정리한다.
- 지금 1.7B 체인(94809–94814)은 0.6B 와 크기만 다른 비교를 위해 세 번 판 그대로 둔다.

---
type: decision
status: active
decision_status: accepted
owner: tskim
review: 2026-10-22
created: 2026-09-05
updated: 2026-09-05
summary: 모델 입력은 두 화자가 섞인 mono 단일 채널이다. 화자별 분리 채널은 라벨·혼합 합성에만 쓰고 채널별 입력(merge·화자별 오디오 토큰)은 폐기
sources:
  - [[output-interleaved-streaming-slm-architecture]]
  - [[decision-target-architecture]]
  - [[decision-asr-backbone]]
  - [[task-uslm-u1-interleaved-asr]]
---

# 결정: 모델 입력은 mono 단일 채널

## 맥락

IS-SLM 설계([[output-interleaved-streaming-slm-architecture]])와 backbone 결정([[decision-asr-backbone]] 세부 3항)은
**두 화자를 채널별로 인코딩해 merge 한 joint chunk token** 을 입력으로 잡았고, U1 v1(merge)·v2(화자별 오디오 토큰 2 개)는
그 전제로 학습했다. 학습·평가 코퍼스(otoSpeech, AI Hub, TurnBench)가 모두 화자별 분리 채널로 배포되기 때문에 자연스러운 선택이었다.

그러나 실제 서비스 입력은 **마이크 하나의 mono 오디오**다. 화자별 채널은 배포 환경에 존재하지 않는다. 채널별 입력으로 학습한
모델은 실전에서 쓸 수 없고, 화자 구분 능력도 입력 채널에 의존하게 되어 diarization 을 배우지 않는다.

## 검토한 선택지

| 선택지 | 장점 | 단점 |
|--------|------|------|
| A. 채널별 입력 유지 (merge / 화자별 오디오 토큰) | 코퍼스 형식 그대로. 화자 귀속이 쉬움. 코드 존재 | **배포 불가**(mono 마이크). 화자 구분을 모델이 배우지 않음. U1 v1 에서 정확도 미달(EN 0.52)이 merge 분포 이동 탓이라는 가설이 남음 |
| B. **mono 혼합 입력** | 배포 조건과 일치. 화자 구분(`<SPK_A/B>`)이 진짜 능력이 됨. adapter 가 U0.5 와 같은 단일 채널 분포를 봄 | 혼합 신호에서 두 화자의 미래 활동을 각각 예측해야 함(원 VAP 보다 어려움). 분리 채널 → 혼합 합성 파이프라인 필요 |
| C. 채널별로 학습 후 mono 로 증류 | 두 단계 성능 확보 | 두 번 학습, 증류 손실 불명확. 실전 입력으로 처음부터 학습하는 B 보다 근거 약함 |

## 결정

**모델 입력은 항상 mono 한 채널이다.** 코퍼스의 화자별 분리 채널은 (a) 누가 언제 말했는지의 라벨(VAD·ForcedAligner 정렬 →
`<SPK_A/B>`·VAP256·hazard 타깃)과 (b) overlap 비율을 제어한 혼합 오디오 합성에만 쓴다. 모델에 채널별로 넣지 않는다.
화자 구분은 별도 모듈이 아니라 모델이 혼합 신호에서 `<SPK_A/B>` 토큰으로 스스로 한다.

## 근거

1. 배포 조건이 mono 다. 입력 형식이 다른 모델은 벤치마크 점수와 무관하게 쓸 수 없다.
2. 화자 구분을 입력 채널이 아니라 모델이 해야 "한 모델이 전사·화자·turn 을 낸다" 는 프로젝트 목표가 성립한다.
3. U0.5 adapter 는 단일 채널 특징으로 학습됐다. mono 입력이 adapter 의 학습 분포와 일치한다(U1 v1 의 merge 분포 이동 가설을 원천 제거).
4. 복잡도를 한 축씩 올리는 커리큘럼(`PLAN.md`)과 맞는다: 단일 화자 mono → 혼합 mono(비중첩) → 화자 태그 → overlap.

## 결과 / 파급

- **강제**: Stage 1–2(단일 화자)는 그대로 mono. Stage 3 은 분리 채널을 합산한 혼합 mono 로 학습하며, 3-1 혼합(전사만) → 3-2 화자 태그 → 3-3 overlap 순.
- **배제**: `vapasr/uslm/interleave_data.py`·`model.py` 의 두 채널 경로(`feats (B,2,K,D)`, merge, `audio_spk`)는 주 경로에서 제외. Stage 1 은 mono 리더·시퀀스 생성기를 새로 둔다.
- **보존**: U1 v1/v2 결과는 두 채널 입력 실험의 진단 기록으로 [[task-uslm-u1-interleaved-asr]] 에 남긴다. v2 는 완주 여부와 무관하게 주 경로 근거로 쓰지 않는다.
- **평가 주의**: VAP oto fine-tune(0.841 @ FP 0.045 / 463 ms)은 stereo 입력 모델이다. mono 입력 IS-SLM 과의 비교는 입력 조건이 다르다는 점을 보고서에 명시한다. 같은 조건의 대조군은 mono 입력 encoder-only probe 로 다시 만든다.
- **superseded 항목**: [[decision-asr-backbone]] 세부 3항의 "두 화자는 화자별 인코딩 → merge → adapter → joint chunk token 1개",
  [[decision-target-architecture]] 결정 절의 "joint chunk token 기본". 두 페이지의 나머지 결정(backbone, IS-SLM 단일 주력)은 유효하다.
- 영향 받는 페이지: [[output-interleaved-streaming-slm-architecture]](joint chunk token 절), [[task-uslm-u1-interleaved-asr]], [[task-uslm-u3-multitask]], `README.md` §2, `PLAN.md` 원칙 4·Stage 3.

## 재검토

2026-10-22 — Stage 3-2(화자 태그) 결과가 나오는 시점. 혼합 입력에서 화자 귀속 오류율이 수용 불가하면 선택지 C(채널별 학습 → mono 증류)를 재검토한다.

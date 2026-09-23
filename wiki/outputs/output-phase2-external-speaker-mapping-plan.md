---
type: output
status: active
created: 2026-09-15
updated: 2026-09-15
summary: 외부 화자 재매핑을 허용하면 로컬 lane·episode ASR를 우선하고 내부 정체성 메모리는 보류 — 겹침 음원·EOT·지연 계약
sources:
  - '[[output-phase2-dynamic-speaker-memory-plan]]'
  - '[[output-phase2-speaker-representation-comparison]]'
  - '[[output-phase2-streaming-asr-diarization-plan]]'
  - '[[output-phase2-real-sequence-probe]]'
---

# 외부 화자 재매핑을 허용할 때의 Phase 2 설계

전체 구조·채널 재사용·겹침 crop·지연 EOT를 그림으로 설명한 입문 보고서는 [[output-phase2-external-speaker-mapping-illustrated]]를 참고한다. R=4/6·해제 정책과 EOT 참조 방식은 공통 설계 원칙과 구별해 읽는다.

## 1. 결론: 이전 상세안을 그대로 구현하지 않는다

**재등장 인물을 이전과 같은 ID로 연결하는 책임이 외부 모듈에 있다면, 내부 동적 화자 메모리까지 학습하는 기존 세 번째 후보는 우선순위가 낮아진다.** 재사용 전사 채널(lane)과 고유한 발화 기록(episode)은 유지하되, 사람의 장기 정체성은 외부에서 붙이는 분리형을 먼저 검증하는 편이 합리적이다. 성능 우위가 실측된 것은 아니며, 변경된 요구에서 구현·진단 비용을 줄이는 권고다.

사용자의 2026-09-15 조건부 질문을 “ASR가 서로 다른 episode의 사람이 같은지 보장할 필요는 없지만, 각 episode 안의 전사 귀속은 유지해야 한다”로 해석했다. **본 문서는 대안 분석이며 정본·수락된 결정·registry·학습 코드를 변경하지 않는다.** 화자별 turn-taking까지 외부로 옮겨도 되는지는 별도 범위 결정이다.

```text
mono audio → E2 encoder / adapter / thinker
                         ↓
             lane + episode별 전사·경계
                         ↓
원 오디오·유효 음성 구간 → 외부 화자 모듈 → episode → session speaker 매핑
                         ↓
             화자별 누적 전사 / turn-taking 소비자
```

외부 모듈이 등록된 사람을 식별하는 recognizer인지, 미등록 회의 참가자를 묶는 diarization/clustering인지 구분해야 한다. 후자의 경우 미지 화자 수·신규 인물·짧은 발화·unknown 처리가 별도로 필요하다. “외부에서 처리”가 이 문제의 해결을 입증하지는 않는다.

## 2. 무엇을 유지하고 무엇을 뺄까

| 구성 | 변경안 | 이유 |
|---|---|---|
| 재사용 lane / 고유 episode ID | 유지 | 겹침 전사와 지연 이벤트의 귀속에 필요 |
| 세션 speaker memory / KNOWN·NEW matcher | ASR 내부에서 제외 | 외부 재매핑과 책임 중복 |
| speaker contrastive·memory matching loss | 초기 학습에서 제외 | 우선 ASR·경계·로컬 귀속을 검증 |
| source-conditioned speaker embedding | 필요성이 확인될 때 추가 | 외부 crop만으로 겹침 귀속이 안 될 때의 대안 |
| ONSET / SEG_END / lexical / NEXT | 유지·구체화 | 세그먼트 생성과 전사 배출을 위한 기본 계약 |
| EOT | episode 귀속은 유지 가능 | SEG_END와 의미가 다르며 지연 참조 필요 |
| 전역 사람별 activity / VAP / hazard | 기존안을 그대로 이식하지 않음 | 외부 ID 가용 시점과 query 구성부터 재설계 필요 |

기존안의 decoder state 기반 prototype을 다음 입력에 넣는 순환 경로가 없어져 **기본 ASR 경로는 기존 HF teacher-forced CE 학습에 더 가까워진다.** 다만 예측 lane/종료 상태의 자유실행 검증은 여전히 필요하다. 아래의 모델 생성 EOT pointer까지 유지하면 해당 동적 참조 경로의 학습·배치 구현도 남는다. 전체를 무조건 기존 Trainer에 바로 꽂을 수 있다는 뜻은 아니다.

## 3. A/B 교대보다 재사용 lane을 권장한다

처음에는 **R=4 lane, R=2 대조**를 제안한다. 4는 세션 총인원이 아니라 미완료 전사 흐름의 용량이며 아직 실측으로 정한 최적값이 아니다. 세 명 겹침까지 다루려면 R≥3과 해당 학습 자료가 필요하다. 전사 배출 지연 때문에 순간 겹침 수보다 점유 lane 수가 클 수도 있다.

설명용 예시이며 실제 데이터 전사가 아니다:

| 발화 순서 | 실제 사람 | 모델 출력 소유자 | 외부 매핑 |
|---|---|---|---|
| 첫 발화 | 민수 | lane_1 / ep_01 | speaker_01 |
| 두 번째 발화 | 지수 | lane_2 / ep_02 | speaker_02 |
| 앞 lane 해제 후 새 발화 | 철수 | lane_1 / ep_03 | speaker_03 |
| 민수 재등장 | 민수 | lane_2 / ep_04 | speaker_01 |

ASR는 ep_01과 ep_04가 같은 사람이라고 출력할 필요가 없다. 대신 **열린 ep_01에 철수의 말을 섞으면 안 된다.** 원 화자가 종료하지 않은 채 다른 화자가 끼어들면 별도 lane/episode를 연다. 짧은 pause 병합도 단순 시간 간격만으로 다른 사람까지 합쳐서는 안 된다.

lane은 종료 후 재사용하고 episode ID는 파서가 단조 증가시켜 재사용하지 않는다. 무한한 ID를 tokenizer에 추가할 필요가 없다. lane generation과 episode 기록을 보존하고, 용량 초과 시 기존 episode를 덮어쓰지 않고 명시적으로 보고한다. 기존 `<SPK_A/B>`를 같은 토큰 ID인 채 다른 뜻으로 조용히 바꾸지 않는다.

이 역할 분리는 t-SOT와 방향이 맞는다. t-SOT는 세션 총 화자 수와 동시 발화 채널 수를 분리하며, 기본형은 동시 두 발화, 일반화는 M개 채널을 다룬다. 다만 논문 실험은 최대 두 동시 발화였고 global identity는 제공하지 않는다. **우리 lane 종료·episode/EOT 계약까지 검증한 논문은 아니다.** [t-SOT §2.1–2.3](https://arxiv.org/html/2202.00842v5), 확인 2026-09-15.

## 4. 가장 먼저 검증할 위험: 시간 crop은 음원 분리가 아니다

`x(t)=s_A(t)+s_B(t)`인 겹침 구간에서 A와 B의 시간 범위를 각각 잘라도, 두 crop에는 같은 혼합 음성이 들어갈 수 있다. **전사 두 줄을 얻었다고 화자별 깨끗한 waveform 두 개를 얻은 것은 아니다.** 외부 모듈은 그 혼합에서 얻은 사람 ID를 어느 전사에 붙일지도 알아야 한다.

권장 초기 경로는 다음과 같다.

1. 같은 episode의 **비중첩·충분한 음성 구간**을 우선 외부 speaker embedding의 근거로 제공한다. 해당 episode의 로컬 화자 일관성이 전제다.
2. 겹침 구간은 이를 둘러싼 근거 구간과 전사 연결을 이용하되, 화자/전사 연결 신뢰도를 별도로 평가한다. 시간 경계가 같다는 이유만으로 매핑하지 않는다.
3. 전 구간이 겹치거나 매우 짧아 근거가 없으면 `unresolved`를 허용한다. 이 비율을 숨기지 않는다.
4. 실패가 크면 overlap-aware 외부 diarizer와 전사 연결, source-conditioned embedding, 음원 분리를 **별도 대안**으로 비교한다. 깨끗한 음원 출력까지 요구하면 현재 ASR 외에 새로운 과제가 추가된다.

토큰별 화자 표현으로 전사와 귀속을 연결하는 연구 경로는 t-vector가 제공한다. 이를 선택하면 “화자 정체성 메모리는 외부, source-conditioned 표현은 ASR에서 출력”하는 중간형도 가능하다. 일반 crop recognizer만으로 해결된다는 증거로 인용하지 않는다. [t-vector](https://arxiv.org/abs/2203.16685), 확인 2026-09-15.

최소 출력 계약은 `episode_id, lane_id, generation, text, estimated_start/end, activity_spans, emitted_at, overlap/quality`다. 경계의 추정치와 전사 방출 시각은 분리한다. `emitted_at−δ`를 정확한 음향 경계로 간주하지 않고, 경계/활동 추정 경로를 학습 또는 별도 인과적 정렬로 검증한다. 비중첩 유효 구간이 없으면 빈 목록을 내며 만들어내지 않는다.

외부 결과에는 `speaker_id 또는 unresolved, confidence, available_at, assignment_revision`을 붙인다. 기록에 후속 매핑을 추가하되 episode ID와 최초 출력 이력은 변경하지 않는다.

## 5. EOT는 어떻게 남기는가

**음향 offset, SEG_END, EOT는 다른 사건이다.** SEG_END는 전사 처리 종료이고, 같은 사람이 잠깐 쉬고 다시 말하면 새 episode가 생겨도 앞 episode의 의미적 turn이 끝났다고 단정할 수 없다.

두 가지 경로를 분리한다.

- **모델 생성 EOT 요구를 유지:** 기존안의 작은 pending-episode 표와 `EVENT_REF → episode pointer → EOT`를 남긴다. 장기 사람 메모리 없이도 ep_01의 EOT를 출력하고 외부에서 speaker_01로 귀속할 수 있다. lane이 다른 사람에게 재사용돼도 ep_01을 참조한다. 이 경우 줄어드는 것은 정체성 관리이지 EOT 참조·학습의 모든 복잡성이 아니다.
- **최소 ASR baseline:** 모델은 ONSET/SEG_END·전사까지만 내고 외부 매핑 이후 별도 turn 모듈이 EOT를 판단한다. 더 단순하지만, 이는 원래 “ASR 모델이 EOT를 예측”하는 목표의 변경이므로 자동 채택하지 않는다.

특히 현행 C-mode weak label은 offset 후 3초 안의 **동일 인물 재개 여부**를 본다. 새 episode/lane 번호가 달라졌다는 이유만으로 다른 사람이라고 판단하면 라벨 의미가 바뀐다. 학습 라벨은 원본 화자 ID로 만들 수 있지만, 추론 판단에는 음향 문맥의 단기 화자 구별 또는 제때 도착한 외부 매핑이 필요하다. 외부 ID를 쓸 경우 그 실제 지연·오류를 포함해 평가하며 gold ID를 추론 입력으로 대체하지 않는다. [[output-phase2-streaming-asr-diarization-plan]] §6.3.

외부 매핑이 **사후 처리만** 가능하면 최종 화자별 전사에는 쓸 수 있어도 당시의 실시간 사람별 VAP/hazard가 같은 ID를 이미 알고 있었다고 주장할 수 없다. episode 단위 종료 예측과 안정된 사람별 미래 활동 예측을 별도 성능으로 보고한다. 지연은 전사 첫 출력·EOT 출력·사람 ID 확정으로 나누고, 실시간 소비자에게 필요한 사건과 ID가 모두 가용해진 시각을 최종 지연으로 측정한다.

## 6. 다음 실험: 큰 모델 변경보다 연결 가능성부터

1. **외부 재매핑 상한 확인:** AMI와 71631의 검증된 표본에서 oracle 화자별 구간을 자르고, 원 mono 음성의 비중첩/부분 겹침/전체 겹침/짧은 발화별로 외부 모듈의 오류·unknown·확정 지연을 측정한다. clean source 채널은 oracle 비교에만 사용한다. oracle 구간에서도 실패하면 경계 모델을 먼저 키우기보다 외부 입력 계약을 수정한다.
2. **local lane ASR baseline:** 기존 TN·정렬·80ms·lexical δ 계약을 유지하고 lane/episode/ONSET/SEG_END 라벨을 만든다. 재등장 장기 ID loss는 넣지 않는다. 무음·단일 발화·빠른 교대·2/3중 겹침·lane 재사용·중간 crop·EOF·용량 초과를 fixture로 고정하고 소규모 overfit 후 자유실행을 확인한다. lane 노출 균형은 모델에도 보이는 할당 순서로 조절하되 실제 3중 겹침 학습을 대체하지 않는다.
3. **예측 구간으로 교체:** 같은 외부 모듈에 모델이 낸 구간을 전달해 oracle 대비 하락을 잰다. episode 내 혼합, 잘못된 병합, 과도한 분할, 경계 오차, 누락된 겹침 전사를 분리한다. 낮은 전사 WER만으로 추출 품질을 판정하지 않는다.
4. **EOT 경로 선택·통합:** 모델 생성 요구를 유지한다면 bounded episode pointer를 추가하고 지연 EOT·재사용·uncertain 사례를 검증한다. 외부 매핑이 충분히 빠르고 안정적인지 확인한 뒤 사람별 미래 head 필요성을 판단한다.

평가에는 로컬 전사 오류와 타이밍, 외부 매핑 후 화자 귀속 전사 오류·DER, unknown coverage와 수정률을 각각 둔다. 최종 speaker-attributed 평가는 **lane을 사람 ID로 간주하지 않는다.** oracle episode→speaker 매핑과 실제 외부 매핑 결과를 나란히 내되, 예측 episode에 여러 사람이 섞인 경우 그 오류를 oracle 매핑으로 지우지 않는다. 기존 고정 K baseline과 같은 평가 집합으로 비교한다.

**채택 판단:** 외부 모듈이 위 상한/연결 시험을 통과하면 분리형을 우선한다. 겹침·짧은 발화에서 귀속이 계속 깨지거나 매우 낮은 지연의 사람별 미래 예측이 필수라면, source-conditioned 표현을 추가하거나 기존 동적 메모리안을 재검토한다. 외부화는 문제를 제거하는 것이 아니라 검증 가능한 모듈 경계로 옮기는 선택이다.

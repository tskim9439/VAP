---
type: output
status: active
created: 2026-10-03
updated: 2026-10-03
summary: CST-Bench 합성음 평가셋 계획 — EN↔DE(사람 번역 대화)·EN↔KO(다중 LLM 합의 참조), CosyVoice3+상용 TTS, 겹침 층화, 정보 인과 스크립트
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
  - '[[cst-bench]]'
  - '[[output-simulst-translation-unit-survey-20261002]]'
related:
  - '[[conversational-simultaneous-s2tt]]'
  - '[[task-cst-timeline-generator]]'
  - '[[task-cst-bench-v0-metrics]]'
raw_authors:
  - tskim
---

# CST-Bench 합성음 세트 계획 (CST-Bench-Syn, 2026-10-03)

> **2026-10-03 갱신(사용자 결정)**:
> - 원어민 번역 검수는 하지 않는다 → 사람 번역이 이미 있는 대화를 쓰고, 나머지는 다중 LLM 합의 + 품질 추정으로 관리한다(§3·§7).
> - 상용 TTS 도 쓸 수 있다(§6).
> - 평가 전용 벤치마크라 **평가셋만** 만든다(§10).

## 0. 배경과 요구 조건

TAXI·Verbmobil 은 신청 후 답변을 기다리고 있다. 그동안 실제 대화 녹음이 없어도 만들 수 있는 **합성음 세트**를 먼저 만든다. 사용자 요구 조건은 다음과 같다.

1. **논문 제출용이다.** 리뷰어를 설득할 수 있어야 한다.
2. **언어는 기준선과 비교 가능한 공통 언어여야 한다.** 영어·독일어는 필수다(TAXI 가 DE↔EN). 한국어도 넣는다.
3. **TTS 는 리뷰어가 납득할 수 있는 검증된 모델이어야 한다.**
4. **새로움(novelty)은 서로 다른 언어 화자의 동시 발화다.** 겹친 두 음성을 모두 감지·번역해야 하므로, 겹침 구간을 일정 비율 이상 둔다.
5. **두 화자는 같은 문맥을 공유하며 서로 대화해야 한다.** 스크립트 제작 방식이 공평해야 한다.
6. **끼어든 화자는 끼어들기 전에 들은 정보만 쓸 수 있다.** 예: A 가 두 문장을 말하려는데 B 가 첫 문장만 듣고 끼어들면, B 는 A 의 둘째 문장 정보를 쓰면 안 된다.

이 문서는 위 조건에 더해 리뷰어가 물을 만한 조건(§9)까지 반영한 구축 계획이다.

## 1. 세트 구성 한눈에

| 항목 | 결정(안) |
|---|---|
| 언어쌍 | **EN↔DE**(주 비교쌍), **EN↔KO**(어순이 먼 쌍, 우리 초점). 화자 언어 배정은 양방향 모두(A=EN/B=DE 와 A=DE/B=EN) |
| 내용 | 세 언어로 **병렬화된 같은 대화**. 언어쌍끼리는 언어만 다르고 내용이 같다 |
| 스크립트 | 사람이 쓰고 사람이 번역한 병렬 대화(XDailyDialog·BConTrasT)를 뼈대로 쓴다. 상호작용 변형은 **정보 인과 규칙**으로 생성하고 자동 누설 검사로 거른다 |
| TTS | **CosyVoice 3**(공개, Apache-2.0) + **상용 TTS 1 개**(Azure Neural TTS 등). 두 엔진으로 평가셋 전체를 합성해 두 판을 함께 공개한다 |
| 입력 조건 | **단일 채널 혼합**(주 조건, novelty) + 화자별 2 채널(누화 포함, 전화 조건) |
| 상호작용 조건 | Normal · Early Turn · Backchannel · Interruption(A 멈춤 / A 계속 = 동시 발화) · Explicit Barge-in |
| 겹침 | 대화 시간 대비 겹침 비율을 **0 / 자연 수준(약 5 %) / 고겹침(≥ 25 %)** 층으로 나눈다. 고겹침 층이 novelty 평가의 중심이다 |
| 참조 | EN↔DE 는 **원천의 사람 번역**(gold), KO 와 새로 생성한 끼어들기 발화는 **다중 LLM 합의 참조**(silver, 표시해서 따로 집계). 끊긴 발화는 의미 단위 경계에서 자른 **부분 참조** |
| 라벨 | 단어 시각(강제 정렬), 화자·언어, 사건(유형·시작·난이도), 기대 행동 |

## 2. 언어쌍 선택 근거 (조건 2)

| 시스템 | EN→DE | DE→EN | EN↔KO | 비고 |
|---|---|---|---|---|
| SeamlessStreaming | ○ | ○ | ○ | 100 여 개 언어. 세 언어 모두 공통 기준선 |
| StreamSpeech | – | ○ | – | CVSS 프·스·독→영 |
| Hikari | ○ | – | – | 영→독·일·러 |
| IWSLT Simultaneous 트랙 | ○ | – | – | 영→독이 전통 쌍 |
| cascade(Whisper / Qwen3-ASR + LLM MT) | ○ | ○ | ○ | 우리가 직접 구성하는 B1–B4 |

- **EN↔DE** 는 공개 동시통역 시스템과 직접 비교할 수 있는 거의 유일한 공통 쌍이다. TAXI(실제 대화)와도 같은 쌍이라, 합성→실제 일반화 검증(§8)을 할 수 있다.
  - 독일어 종속절은 동사가 뒤에 와서 동시통역 난이도도 있다.
- **EN↔KO** 는 어순이 가장 먼 쌍이고 우리 기여의 초점이다. Seamless·LLM cascade 가 기준선이 된다.
- DE↔KO 는 비교 기준선이 거의 없어 제외한다. 영어를 축으로 두면 언어쌍 효과를 깔끔하게 비교할 수 있다.

## 3. 대화 스크립트: 같은 문맥, 공평한 제작 (조건 5)

원어민 번역 검수를 할 수 없다. 그래서 **사람 번역이 이미 있는 병렬 대화를 내용의 뼈대로 삼고**, 사람 번역이 없는 부분은 다중 LLM 합의와 품질 추정으로 관리한다.

### 3.1 원천

| 원천 | 언어 | 사람 번역 | 쓰임 |
|---|---|---|---|
| **XDailyDialog**(ACL 2023) | EN·DE·ZH·IT 병렬 | ○ 전문 번역가 50 여 명, 데이터 전문가 3 명이 번역가별 20 % 표본 검수 | 일상 대화의 주 원천. EN↔DE 참조는 gold. 13K 대화, 대화당 7.9 발화. **CC BY-NC-SA 4.0**(DailyDialog 파생) |
| **BConTrasT**(WMT20 Chat, Taskmaster-1 기반) | EN↔DE | ○ Unbabel 전문 번역 + 원어민 사후 편집 | 과제형 대화(주문·예약 등 6 영역). 실제 이중언어 상황(독일어 고객 · 영어 상담원)을 상정. **CC BY-SA 4.0** |
| 한국어판 | KO | ✕ | 영어 원문을 다중 LLM 합의 번역(§7.1) → silver |
| 끼어들기·Barge-in·해소 턴(새로 생성) | EN → DE·KO | ✕ | 영어로 생성한 뒤 다중 LLM 합의 번역 → silver |

- AI Hub 병렬 말뭉치는 재배포가 불가해서 쓰지 않는다.
- **공개 라이선스 결론**: §12 참고. XDailyDialog(BY-NC-SA)와 BConTrasT(BY-SA)는 서로 호환되지 않으므로 **부분집합 두 개로 나눠 각자의 라이선스로 공개**한다.
- **오염 위험과 대응**: DailyDialog·Taskmaster 텍스트는 LLM 학습에 들어갔을 가능성이 있다.
  - 텍스트 기억이 줄 수 있는 이득은 번역 품질 쪽에만 생긴다. 이 벤치마크의 핵심(겹친 음성·타이밍·방향)과는 관계가 적다.
  - 공개 SimulST 기준선은 대화 텍스트로 학습되지 않았다.
  - 그래도 오염 탐침(대화 앞부분을 주고 뒤 턴을 이어 쓰게 해 일치율을 봄)을 보고하고, 새로 생성한 사건 발화만 따로 집계한 점수를 함께 낸다.

### 3.2 같은 문맥의 두 화자 만들기
1. 3 언어로 정렬된 같은 대화를 쓴다(XDailyDialog EN·DE + KO silver, BConTrasT EN·DE + KO silver).
2. A 턴은 언어 X 판에서, B 턴은 언어 Y 판에서 가져온다. 두 화자가 같은 문맥을 공유하고, **상대 언어 판이 곧 참조 번역**이 된다.
3. 같은 대화를 모든 언어쌍·배정 방향(A=EN/B=DE, A=DE/B=EN, A=EN/B=KO, A=KO/B=EN)에 쓴다. 쌍 사이 차이는 언어 때문에만 생긴다.
4. 스크립트 길이·턴 수·영역 분포를 언어쌍마다 같게 맞춘다.

## 4. 정보 인과적 상호작용 생성 (조건 6)

**규칙**:
- 상대의 말에 반응하는 발화는 그 발화 시작 시점까지 상대가 실제로 말한 내용에만 의존할 수 있다.
- 겹쳐서 계속 말하는 화자는 상대의 끼어든 내용을 아직 처리하지 못한 것으로 본다. 그래서 원래 계획한 말을 이어 간다.

### 4.1 사건별 생성 방식

| 사건 | 생성 방식 | 인과 보장 |
|---|---|---|
| Normal | 원 대화 순서, 자연 간격(약 200 ms 중심 분포) | 내용이 원래 대화와 같다 |
| Early Turn | B 의 원래 응답을 A 의 턴 끝 직전에 시작 | B 응답이 의존하는 A 의 마지막 정보 위치(의존점)를 찾고, **의존점 + 반응 여유(≥ 200 ms) 이후**로만 시작 시각을 둔다 |
| Backchannel | 듣는 쪽 언어의 맞장구(KO 네/응/맞아요, EN yeah/right/uh-huh, DE ja/genau/mhm)를 A 발화 중 의미 단위 경계에 삽입 | 내용이 없으므로 인과 문제가 없다. 번역·방향 전환을 하면 안 되는 사건으로 라벨한다 |
| Interruption — A 멈춤 | A 의 턴을 중간 의미 단위 경계 c 이후 반응 지연(200–600 ms)에서 자른다. B 의 끼어들기 발화는 **A[:c] 와 이전 이력만 보고 새로 생성** | 생성 입력에 A 의 뒷부분을 넣지 않는다. 사후에 누설 검사를 한다(§4.2) |
| Interruption — A 계속(동시 발화, **novelty**) | B 는 위처럼 생성. A 는 원래 계획한 둘째 문장을 계속 말하고 B 와 겹친다 | A 의 계속 발화는 원래 스크립트라 B 내용에 의존하지 않는다. 두 화자의 겹친 발화를 모두 번역해야 한다 |
| Explicit Barge-in | “잠시만요 / Wait / Moment mal” + A[:c] 에 대한 질문을 생성 | 위와 같은 인과 생성과 누설 검사 |

### 4.2 끼어들기 이후의 진행과 누설 검사

- **에피소드 단위**: [앞 이력 몇 턴] + [사건] + [해소 1–2 턴].
  - 끼어들기 뒤 대화는 원래 대본에서 갈라지므로, 해소 턴도 같은 인과 규칙으로 생성한다.
  - 그 뒤는 잘라서 짧은 에피소드로 만든다. 그래야 검수 가능한 크기가 된다.
  - 원래 순서를 유지한 긴 세션(Normal·Early Turn·Backchannel 만)은 따로 둔다.
- **자르는 지점 c 는 번역 의미 단위(MU) 경계로 한정한다**([[output-simulst-translation-unit-survey-20261002]]).
  - 그러면 A 가 실제로 말한 부분의 참조 번역이 전체 참조의 접두로 성립한다(부분 참조의 정합성).
  - 난이도는 명세를 따른다: Easy = 턴 끝 직전, Medium = 의미 단위 경계, Hard = 핵심 의미 완성 전.
- **누설 검사(자동 + 팀 표본 확인)**:
  - 자동: B 발화의 고유명사·숫자·사실이 A[c:] 에만 있고 A[:c]·이력에는 없으면 탈락.
  - 판정 LLM 두 개(EXAONE-4.0·Qwen3.8-27B)가 각각 “B 의 말이 A 의 앞부분만으로 가능한가”를 판정한다. **둘 다 통과한 것만** 남긴다. 판정자 간 일치도를 보고한다.
  - 팀 표본 확인: 영어로 생성한 원문 기준으로 표본 5 % 를 팀이 직접 본다. 번역 검수가 아니라 생성 논리 확인이다.
- **의존점 계산**: B 응답의 각 정보 단위가 A 의 어느 단어에서 처음 나오는지 정렬한다. 문맥 정렬 도구를 재사용한다. 가장 늦은 위치가 의존점이다.

## 5. 겹침 설계 (조건 4)

### 5.1 자연 대화 기준
- 대화 시간 중 겹침은 약 4–7 % 이고, 화자 교대의 30–40 % 가 겹침을 포함한다. 겹침 길이 중앙값은 약 0.5 s 다(Heldner & Edlund 2010; Fisher 전화 대화는 8–15 %).
- 이 수준만으로는 “겹친 두 언어를 모두 번역”하는 능력을 측정하기 어렵다. 그래서 층을 나눈다.

### 5.2 겹침 층과 정의

| 층 | 대화 시간 대비 겹침 | 구성 | 용도 |
|---|---|---|---|
| L0 | 0 % | Normal | 기존 SimulST 와 같은 조건, 기준선 비교 |
| L1 | 약 5 % | 자연 분포(Early Turn·맞장구 위주) | 현실 수준 |
| L2 | ≥ 25 % | 동시 발화형 끼어들기(A 계속)·긴 겹침 | **novelty 평가의 중심** |

- **겹침 비율 정의**: 두 화자가 동시에 유성 발화하는 시간 / 대화 전체 발화 시간. VAD 가 아니라 TTS 강제 정렬 단어 구간으로 계산한다.
- **겹침 단어 비율**: 상대가 말하는 동안 발화된 원천 단어 비율. 이 비율도 함께 보고한다.
- **겹침 안의 번역 지표**: 겹친 구간 원천 단어에 대응하는 번역 단어의 재현(놓친 번역 비율), 그 구간만의 품질(COMET/chrF), 화자별 방향 정확도.
- **신호 조건**:
  - 두 화자의 레벨 차(SIR) −5 … +5 dB
  - 실내 잔향(RIR)·잡음(SNR 5–30 dB): 기존 잡음 뱅크 재사용
  - 전화 대역 조건(8 kHz 코덱): TAXI 대응

### 5.3 출력 형식(평가 전제)
- 시스템은 두 방향(A→B, B→A) 번역을 **동시에** 낼 수 있어야 한다.
- 출력 단위: (시각, 방향, 텍스트, 확정 여부) 열.
- 기존 단일 방향 SimulST 기준선은 방향마다 하나씩 붙인다(B1–B4).

## 6. TTS (조건 3)

### 6.1 선정 기준과 엔진 (2026-10-03 조사 반영)

기준:
1. EN·DE·KO 를 모두 지원해야 한다.
2. 리뷰어가 아는 검증된 품질이어야 한다.
3. 재현 가능하고 합성음 재배포가 가능해야 한다.
4. 화자 다양성이 있어야 한다.
5. 단어 시각을 얻을 수 있어야 한다.

| 엔진 | 언어·검증 | 라이선스·약관 | 판단 |
|---|---|---|---|
| **CosyVoice 3**(Fun-CosyVoice3) | 9 개(EN·DE·KO 포함), 다언어·교차언어 제로샷 복제. 보고 성능 test-en WER 1.68 %, 화자 유사도 69.5 % | 코드 Apache-2.0(가중치 라이선스는 저장소에 명시가 약해 확인 필요) | **공개 판 1 순위** |
| **Qwen3-TTS**(0.6B·1.7B Base·CustomVoice·VoiceDesign) | 10 개(EN·DE·KO 포함), 3 초 참조 복제. **독일어 WER 1.09–1.24, 한국어 1.74–1.76, 화자 유사도 DE 0.775 · KO 0.799**(1.7B) | Apache-2.0 | 공개 판 대안. DE·KO 수치를 공식 보고해 리뷰어 설득에 유리하다 |
| **Azure Neural TTS** | en-US·de-DE·ko-KR 원어민 음성 다수, SSML 로 쉼·속도 조절. **WordBoundary 이벤트로 단어 시각**(100 ns 단위)을 준다 | 마이크로소프트 공식 답변: 유료 리소스로 자기 텍스트를 합성하면 **상업·개인 용도로 사용·배포 가능, 별도 허락 불필요**. 합성음임을 공개하라는 책임 있는 사용 지침이 있다 | **상용 판 1 순위**: 재배포 근거가 가장 명확하고 단어 시각을 바로 얻는다 |
| Google Cloud TTS | 다수 | TTS 전용 약관 미확인. 경쟁 TTS 학습 금지로 알려짐(2 차 출처) | 예비 |
| ElevenLabs | 다언어 | 유료 플랜은 출력 소유. **출력으로 경쟁 AI 모델 학습 금지** | 예비. 평가 전용이면 가능하지만 데이터 라이선스에 학습 금지 조항을 넣어야 한다 |
| XTTS-v2 | 17 개 | CPML(비상업) | 제외 |

- **권장 구성: 공개 판 CosyVoice 3(또는 Qwen3-TTS) + 상용 판 Azure.** 두 판으로 평가셋 전체를 합성한다.
  - 주 결과는 한 판으로 보고하고, 다른 판에서 **시스템 순위가 유지됨**(순위 상관)을 보인다.
  - 공개 판 엔진은 S0 파일럿(언어별 50 문장, 명료도·화자 지표)에서 CosyVoice 3 와 Qwen3-TTS 중 고른다.
- 공개 판에는 화자 참조 음성이 필요하다. 복제 허용 라이선스(CC0·CC BY)의 원어민 녹음에서 고르거나, Qwen3-TTS VoiceDesign 으로 실존 인물이 아닌 음성을 만든다.
- 상용 판은 Azure 프리셋 원어민 음성을 쓴다. 언어별 8–12 명, 성별 균형.
- 교차 언어 복제(외국어 억양 음색)는 쓰지 않는다.

### 6.2 합성 품질 보증(논문에 수치로 보고)

| 검사 | 방법 | 기준(안) |
|---|---|---|
| 명료도 | 서로 다른 ASR 두 개(Whisper-large-v3, Qwen3-ASR)로 재인식 | EN/DE WER ≤ 5 %, KO CER ≤ 5 %. 넘으면 재합성 |
| 화자 일관성 | WavLM 계열 화자 임베딩 유사도 | 화자 내 ≥ 기준, 화자 간 분리 |
| 자연성 | UTMOS(EN) · DNSMOS(전 언어) · TTSDS(실제 음성 분포와의 거리) | 두 엔진과 실제 대화 녹음(공개 대화 코퍼스 표본)을 같은 지표로 비교. 사람 MOS 는 하지 않는다(한국어만 팀이 소규모 확인 가능) |
| 단어 시각 | 강제 정렬(MFA 독·한 모델 또는 Qwen3-ForcedAligner) | 표본 수동 확인 |
| 겹침 그럴듯함 | 겹침 길이·시작 위치 분포를 자연 대화 통계와 비교 | 고겹침 층(L2)은 의도적 스트레스 조건이라고 명시 |

## 7. 참조와 라벨

### 7.1 참조 번역 (사람 검수 없이)
- **gold**: XDailyDialog·BConTrasT 의 EN↔DE 사람 번역을 그대로 쓴다.
- **silver**: 한국어 전체, 그리고 새로 생성한 끼어들기·해소 발화. 만드는 절차:
  1. 서로 다른 번역기 **세 개**가 독립 번역한다: EXAONE-4.0, Qwen3.8-27B, **MADLAD-400-10B-MT**(Google, Apache-2.0, 450+ 언어, Hibiki 가 정렬에 쓴 모델). 세 계열이 달라 한 모델 편향을 줄인다.
     - **DeepL 은 제외**: 약관 8.1.1(g) 가 출력으로 “기계 번역 알고리즘을 개발·학습”하는 것을 금지하고, (e) 는 벤치마크 시험을 금지한다. 번역 시스템 평가용 참조로 공개하는 것은 위험하다.
  2. 번역끼리 의미 일치를 검사한다:
     - 참조 없는 품질 추정: **MetricX-24 하이브리드**(Apache-2.0, 참조 유무 모두 채점, XXL·XL·Large)를 주로 쓰고, CometKiwi(비상업 라이선스, 허깅페이스 동의 필요)를 보조로 쓴다
     - 역번역 의미 유사도
     - 숫자·고유명사 일치
  3. 셋 중 둘 이상이 일치 기준을 넘으면 통과한다. 판정 LLM 이 그중 하나를 주 참조로 고르고, 나머지는 **다중 참조**로 함께 둔다.
  4. 기준을 못 넘긴 발화는 평가셋에서 뺀다. 버린 비율을 보고한다.
- **채점**:
  - 다중 참조 BLEU·chrF
  - 참조마다 COMET 를 낸 평균
  - 참조 없는 QE 점수
  - gold 와 silver 구간을 나눠 집계해서, silver 참조가 결론을 바꾸지 않는지 보인다(EN↔DE 는 gold 와 silver 를 모두 만들어 두 점수의 시스템 순위 상관을 보고)
- **부분 참조**: 끊긴 발화(A 멈춤)는 의미 단위 경계 c 에서만 자른다. 주 참조 번역에서 c 까지 정렬된 앞부분을 부분 참조로 쓴다. 정렬은 문맥 정렬 도구를 쓴다.
- 필요 모델(MetricX-24·CometKiwi·LaBSE·MADLAD-400)은 T5 에 받고, 서버 업로드는 사용자가 한다.

### 7.2 라벨
- **사건 라벨**: 유형·시작 시각·난이도·관련 화자, 그리고 기대 행동.
  - 맞장구 → 전환 없음
  - 끼어들기 → 이전 방향 마무리 또는 중단 + 새 방향 시작
  - 동시 발화 → 두 방향 모두 번역
  - 이 라벨로 [[task-cst-bench-v0-metrics]] 의 Stop/Switch Latency·놓친 끼어들기·잘못된 전환을 계산한다.
- **단어 시각**: TTS 출력 강제 정렬. 상용 TTS 가 단어 경계 이벤트를 주면 그것으로 대조한다.

## 8. 리뷰어 설득 장치

1. **합성→실제 일반화 검증(sim2real)**: TAXI(DE↔EN) 확보 뒤 같은 시스템들의 순위를 실제 대화와 합성음에서 비교한다(순위 상관). 겹침이 적은 L0/L1 조건끼리 대조한다.
2. **TTS 교차 검증**: 두 엔진 사이의 시스템 순위 안정성(§6.1).
3. **자동 검증의 투명성**: 합성 품질 지표, 판정자 간 일치도(누설·참조 선택), 참조 합의 탈락률, gold/silver 순위 상관을 보고한다.
4. **통계 검정력**: 사건 유형 × 난이도 × 언어쌍마다 사건 ≥ 300 개를 둔다. 짝 bootstrap 신뢰구간을 보고한다.
5. **모든 것을 공개**: 스크립트·생성 프롬프트·합성 설정·시드·혼합 코드·라벨.
6. **한계 명시**: 합성 운율, 비유창성 부족, 스크립트 대화의 계획성. 실제 세트(TAXI·Verbmobil·Phase 5 한국어 수집)로 보완한다고 적는다.

## 9. 추가로 고려한 조건

- **오염**: 공개 대화(DailyDialog 계열)는 LLM 학습에 들어갔을 가능성이 높다 → 시험셋은 신규 작성.
- **라이선스**:
  - 스크립트: 신규 작성은 CC BY 로 공개, BConTrasT·Taskmaster 는 CC BY 4.0(확인 필요)
  - TTS: Apache-2.0
  - 참조 음성: CC0/CC BY
  - AI Hub 는 재배포가 불가해서 쓰지 않는다
- **윤리**: 실존 인물 음색을 동의 없이 복제하지 않는다. 합성음임을 데이터 카드에 표기한다.
- **언어별 형평성**: 언어마다 화자 수·발화 속도·사건 분포를 맞춘다. 맞장구·끼어들기 표현 목록은 대화 코퍼스 빈도로 정한다. 한국어는 팀이 직접 확인한다.
- **턴 길이 분포**: 짧은 턴이 많으면 의미 단위 방식의 이점이 작아진다. 실제 대화 턴 길이 분포(Switchboard·TAXI 확보 후)에 맞추고 보고한다.
- **비유창성**: 일부 스크립트에 간투사·자기 정정(“오전에… 아니 오후”)을 넣는다. semantic commitment 평가의 핵심 사례다.
- **기준선 실행 가능성**: 단일 방향 시스템도 화자별 채널 조건에서는 바로 돌 수 있게 2 채널 판을 함께 제공한다. 단일 채널 혼합 조건은 우리 joint 모델과 cascade 기준선만 처리할 수 있다는 점을 명시한다.

## 10. 규모와 일정(안) — 평가셋만

| 단계 | 내용 | 산출물 |
|---|---|---|
| S0 (1 주) | TTS 파일럿(두 엔진, 언어별 50 문장): 명료도·화자·자연성 지표. 상용 약관 확인. 원천 대화 선별 | 품질 수치 초안, 엔진 확정 |
| S1 (1–2 주) | 원천 대화 선별(XDailyDialog·BConTrasT), KO silver 번역(3 번역기 합의) | 3 언어 병렬 스크립트 |
| S2 (2 주) | 정보 인과 사건 생성기, 누설 검사, 의미 단위 경계, 생성 발화 번역 | 사건 스크립트 + 참조 |
| S3 (1–2 주) | 두 엔진 합성·혼합(층·채널·잡음 조건)·라벨, 품질 보증 | **CST-Bench-Syn v0 평가셋**(두 판) |
| S4 | 기준선 B0–B4 실행, kill criterion | [[task-cst-kill-criterion-check]] |

- 규모: 언어쌍마다 평가셋 약 10 h(층·사건 유형·난이도마다 사건 ≥ 300). 학습·개발 세트는 만들지 않는다.
- 다만 시스템 설정(지연 단계 등)을 고르는 데 쓸 **작은 보정 분할(약 5 %)**을 평가셋과 겹치지 않게 떼어 두는 것을 권한다. 모든 기준선에 똑같이 적용한다.

## 11. 결정 사항 (2026-10-03 조사 후 권고)

| 항목 | 권고 | 근거 |
|---|---|---|
| 상용 TTS | **Azure Neural TTS** | 출력 사용·배포에 대한 공식 허용 답변. de-DE·ko-KR 원어민 음성. 단어 시각 이벤트 |
| 공개 TTS | CosyVoice 3 또는 Qwen3-TTS(파일럿으로 결정) | 둘 다 Apache-2.0·DE·KO 지원. Qwen3-TTS 는 DE·KO 품질 수치를 공식 보고 |
| 세 번째 번역기 | **MADLAD-400-10B-MT**(DeepL 제외) | DeepL 약관이 MT 개발·벤치마크 사용을 금지. MADLAD 는 Apache-2.0 |
| 품질 추정 | MetricX-24(주) + CometKiwi(보조) | MetricX-24 는 Apache-2.0 이고 참조 없이도 채점 |
| 공개 라이선스 | 부분집합별로 CC BY-NC-SA 4.0(XDailyDialog 계열)·CC BY-SA 4.0(BConTrasT 계열) | 두 ShareAlike 라이선스가 서로 호환되지 않음(§12) |
| 입력 조건 | 단일 채널 혼합(주) + 화자별 2 채널(보조) 모두 제공 | novelty 는 혼합 조건, 단일 방향 기준선 실행은 2 채널 조건. [[decision-mono-input]] 과 일치 |
| 보정 분할 | 평가셋의 약 5 % 를 떼어 둔다 | 지연 단계 등 설정을 시험 분할로 고르면 결과가 부풀려진다. 모든 시스템에 같은 규칙을 적용 |

남은 확인 사항:
- CosyVoice 3 가중치 라이선스
- Taskmaster-1 원 라이선스(BConTrasT 는 CC BY-SA 4.0 으로 공개됨)
- Azure 요금(평가셋 약 100 만 자 기준)

## 12. 재배포 약관과 비용 (2026-10-03 조사)

### 12.1 구성 요소별 재배포 가능 여부

| 구성 요소 | 라이선스·약관 | 공개본에 넣을 수 있나 | 조건·조치 |
|---|---|---|---|
| XDailyDialog 텍스트 | CC BY-NC-SA 4.0(DailyDialog 파생) | ○ | 출처 표시, 비상업, **같은 라이선스로** 공개 |
| BConTrasT 텍스트 | CC BY-SA 4.0. 원본 Taskmaster-1 은 CC BY 4.0 | ○ | 출처 표시, **CC BY-SA 4.0 으로** 공개 |
| ↳ 두 텍스트를 한 저작물로 합치기 | BY-SA 와 BY-NC-SA 는 서로 다른 ShareAlike 라 **합친 2 차 저작물은 불가** | – | **부분집합 두 개로 나눠** 각자 라이선스로 배포(묶음 배포는 가능) |
| Azure TTS 합성음 | 마이크로소프트 공식 답변: 유료 리소스로 자기(또는 적법하게 라이선스받은) 텍스트를 합성하면 상업·개인 용도로 사용·배포 가능, 허락·로열티 불필요. 무료(F0) 등급은 답변이 엇갈림 | ○ | **유료(S0, 종량제) 리소스로 생성**. 입력 텍스트 권리 확보(위 두 라이선스로 충족). 합성음임을 데이터 카드에 표시(행동 강령) |
| CosyVoice 3 / Qwen3-TTS 합성음 | 모델 Apache-2.0(Fun-CosyVoice3-0.5B 가중치도 Apache-2.0), 출력 제한 없음 | ○ | 복제용 참조 음성은 CC0 로 하거나 CC BY 이면 출처 표시. 실존 인물 무단 복제 금지 |
| Qwen3.8-27B 번역 | Apache-2.0 | ○ | – |
| MADLAD-400 번역 | Apache-2.0 | ○ | – |
| **EXAONE-4.0 번역** | EXAONE AI Model License(NC): 출력은 비상업 연구용, **라이선서 모델과 경쟁하는 모델을 개발·개선하는 데 출력 사용 금지** | △ **위험** | 공개 참조 텍스트에는 넣지 않는다. EXAONE 은 합의 판정·누설 검사 같은 **내부 판정에만** 쓰고, 공개 참조는 Qwen3.8·MADLAD 번역에서 고른다 |
| DeepL 번역 | 약관이 MT 개발·학습·벤치마크 사용 금지 | ✕ | 쓰지 않는다 |
| 채점 모델(MetricX-24 Apache, CometKiwi NC) | 평가에만 사용, 공개본에 포함하지 않음 | – | 논문에 버전만 명시 |
| 잡음·잔향(혼합에 쓸 경우) | RIRS_NOISES 는 Apache-2.0, MUSAN·DEMAND 는 원천마다 다르고 일부 SA, ETSI 는 재배포 제한 가능 | △ | **깨끗한 화자별 음원 + 혼합 스크립트·잡음 목록·시드**를 공개하고 사용자가 재현하게 한다. 혼합본은 재배포 가능한 잡음만으로 만든다 |

### 12.2 Azure TTS 비용

| 항목 | 값(2026 기준, 2 차 출처) |
|---|---|
| Neural(표준) 음성 | **$16 / 100 만 자** |
| Neural HD 음성 | $22 / 100 만 자(2026-03 인하, 이전 $30) |
| 무료(F0) | 월 50 만 자. 단, 재배포 근거를 확실히 하려면 유료 리소스 권장 |
| 약정 최저 단계 | 월 $960 / 8,000 만 자(필요 없음) |
| 과금 문자 | 글자·숫자·공백·문장부호 모두. `<speak>`·`<voice>` 외 SSML 태그 안 글자도 과금. 중국어 한자는 2 자로 계산(한글이 2 자인지는 문서에 명시 없음 — 보수적으로 2 배 가정) |

추정(평가셋 전체, 두 언어쌍 × 화자 배정 양방향, 약 20 h 음성):
- 영어·독일어는 초당 약 14 자, 한국어는 초당 약 8 자이고 2 배 과금을 가정했다.
- 재합성 20 %, 보정 분할과 파일럿, SSML 태그까지 더하면 **약 100–150 만 자 → 표준 $16–24, HD $22–33**.
- 화자별 다른 음성을 쓰거나 판을 하나 더 만들어도 **$100 이하**다.
- 공식 가격 페이지는 로그인 후 계산기로만 보여서, 위 단가는 2 차 출처(2026)다. 결제 전 가격 계산기로 확인한다.

## 출처 (관측일 2026-10-03)
- Azure TTS 출력 사용·배포(마이크로소프트 Q&A 공식 답변): https://learn.microsoft.com/en-us/answers/questions/652774/usage-license-of-wav-mp3-generated-by-azure-text-2
- Azure SSML·WordBoundary: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-synthesis-markup-structure
- DeepL Pro 약관(8.1.1 e·f·g): https://www.deepl.com/en/pro-license
- ElevenLabs 출력 권리(2 차 정리): https://terms.law/ai-output-rights/elevenlabs/
- CosyVoice 공식 저장소: https://github.com/FunAudioLLM/CosyVoice
- Qwen3-TTS 공식 저장소: https://github.com/QwenLM/Qwen3-TTS
- MetricX-24: https://huggingface.co/google/metricx-24-hybrid-xxl-v2p6
- MADLAD-400 MT: https://huggingface.co/google/madlad400-3b-mt
- BConTrasT(CC BY-SA 4.0): https://github.com/Unbabel/BConTrasT
- XDailyDialog(CC BY-NC-SA 4.0, 전문 번역): https://github.com/liuzeming01/XDailyDialog
- Taskmaster-1(CC BY 4.0): https://github.com/google-research-datasets/Taskmaster/blob/master/TM-1-2019/README.md
- EXAONE AI Model License 1.2 NC: https://scancode-licensedb.aboutcode.org/exaone-ai-model-1.2-nc.html
- Qwen3.8-27B Apache-2.0: https://www.eweek.com/news/alibaba-qwen3-8-27b-license-apac-china/
- Fun-CosyVoice3-0.5B 가중치 Apache-2.0: https://huggingface.co/agiws/Fun-CosyVoice3-0.5B
- Azure TTS 가격(2 차): https://texttolab.com/blog/azure-text-to-speech-pricing , 공식 페이지 https://azure.microsoft.com/en-us/pricing/details/speech/
- Azure 과금 문자(SSML 태그): https://learn.microsoft.com/en-us/answers/questions/584662/what-characters-are-billabling-in-text-to-speech
- Azure 상업 사용·무료 등급 답변: https://learn.microsoft.com/en-us/answers/questions/2124037/can-i-use-azure-text-to-speech-to-generate-mp3-for
- CosyVoice 3 언어·라이선스: https://tts.ai/voices/cosyvoice3/ (2 차 정리 사이트, 공식 저장소 확인 필요)
- Qwen3-TTS: https://rits.shanghai.nyu.edu/ai/%F0%9F%93%A2-qwen3%E2%80%91tts-open%E2%80%91source-text%E2%80%91to%E2%80%91speech-tts-family/ (2 차 정리, 확인 필요)
- XDailyDialog(ACL 2023): https://aclanthology.org/2023.acl-long.684
- WMT20 Chat Translation(BConTrasT): https://www.statmt.org/wmt20/chat-task.html , https://aclanthology.org/2020.wmt-1.3
- 겹침·간격 통계: Heldner & Edlund 2010 https://staff.fnwi.uva.nl/r.fernandezrovira/teaching/cosp/cosp2016/docs/HeldnerEdlund2010.pdf ; Talking Turns https://arxiv.org/pdf/2503.01174 ; 전화 대화 분리 연구 https://arxiv.org/pdf/2205.15700
- TTS 평가 관행: https://arxiv.org/pdf/2606.21343 , TTSDS https://arxiv.org/html/2407.12707v3

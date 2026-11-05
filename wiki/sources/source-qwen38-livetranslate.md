---
type: source
status: active
created: 2026-11-05
updated: 2026-11-05
summary: Qwen3.8-LiveTranslate(2026-09) — 오디오·글자 한 인과 스트림(Interleave)에 동시통역·원문·화자 표시, LAAL 2.3 s·60 언어; 우리 CST 와 겹치는 점·다른 점
sources:
  - https://qwen.ai/blog?id=qwen3.8-livetranslate
  - https://www.alibabacloud.com/blog/qwen3-8-livetranslate-names-the-speaker--carries-the-meaning-_603581
---

# Qwen3.8-LiveTranslate (Alibaba Qwen, 2026-09-19 발표)

- 관찰일 2026-11-05. 사용자 질문 "우리의 프로젝트 목표가 Qwen3.8-LiveTranslate 과 매우 유사하지 않아?"
- 출처
  - 공식 글(본문을 읽은 판): https://www.alibabacloud.com/blog/qwen3-8-livetranslate-names-the-speaker--carries-the-meaning-_603581 — qwen.ai 블로그 판은 본문이 읽히지 않았다.
  - 발표: https://x.com/Alibaba_Qwen/status/2101206705111757253
  - 써 본 글(정성적): https://www.mindstudio.ai/blog/qwen3-8-livetranslate-hands-on
  - 비교 글(DER 9.7 %·Seed LiveInterpret 2.0 30.6 % 인용 — 회사 자체 셋): https://www.orcarouter.ai/blog/qwen-3-8-live-translate-vs-gemini-3-5-live-translate
- 수치는 모두 회사가 낸 값이다. 다른 곳에서 검증한 값은 아직 없다.

## 무엇인가
- 실시간 동시통역 모델(API 전용 `qwen3.8-livetranslate-flash-realtime`, 가중치·크기 비공개, Hybrid-MoE).
- Interleave 구조: Thinker 가 영상·오디오·원문·번역을 시간 순서대로 한 인과 시퀀스에 엮는다. 들은 오디오와 낸 번역을 캐시로 다시 쓴다. Talker 가 원 화자 음색으로 말소리를 낸다.
- 평균 지연(LAAL) 2.8 → 2.3 s. 60 언어(말소리 출력은 29 언어). 청크 크기는 밝히지 않았다.
- 새 기능
  - 화자 구분: 여러 사람이 차례로 말하면 번역·원문과 함께 화자 표시를 돌려준다. 평가는 DER(자체 다화자 긴 오디오 셋 Omnilingua-MSpeaker, 14 방향 — 공개 셋은 찾지 못했다).
  - 원문·번역 동시 표시, 긴 문맥으로 고유명사·지시어를 일관되게.
- 밝히지 않은 것: 화자 구분이 모델 안(종단간)인지, 겹친 말 처리, 한 세션에서 두 언어 화자를 서로의 언어로 옮기는지(예제 클라이언트는 목표 언어 하나).
- 써 본 글: 짧은 문장 번역은 좋고, 말이 끝났는지 판단(턴)과 지연이 약점이다.

## 우리 프로젝트(CST-S2TT)와
- 겹치는 점: 한 인과 스트림 LLM 이 오디오를 받으며 원문 받아쓰기·번역·화자 표시를 함께 낸다. 우리 슬롯 블록(오디오 → 슬롯 태그 → 받아쓰기 → `<TRANS>` 번역 칸)과 같은 계열이다.
- 다른 점(우리 쪽에서 내세울 수 있는 것 — 그쪽이 밝히지 않은 부분이 많아 확인 전까지는 주장 수준)
  - 두 언어 대화: 한 모노 스트림의 두 화자를 각자 상대 언어로(화자마다 목표 언어) — 그쪽은 세션 목표 언어 하나로 보인다.
  - 지연: 우리 받아쓰기 방출 δ10 평균 0.92 s(번역 LAAL 은 같은 정의로 다시 재야 한다) 대 그쪽 LAAL 2.3 s.
  - 겹친 말이 많은 회의(AMI 4 인)의 화자 귀속을 cpWER 로 잰다. 그쪽은 DER 만 낸다.
  - 크기·공개: 1.7B, 공개 가능한 구성 대 비공개 대형 MoE.
- 그쪽이 앞서는 점: 60 언어·말소리 출력·영상 입력, 대규모 학습.
- 원고 영향: [[output-nemotron35-transcribe-comparison-20261029]] 의 "어느 것도 이 과제를 하지 않는다"(화자 귀속 동시 번역)는 더는 맞지 않다. 기여를 두 언어 대화 양방향·짧은 지연·겹침 회의 귀속으로 좁혀야 한다 → [[output-paper-draft-speaker-slots-icml2027-20261103]]
- 할 수 있는 것: API 로 TAXI(실제 독–영 통화)·우리 대화 셋을 같은 채점으로 재 외부 기준으로 넣는다(외부 서비스에 오디오를 보내는 일이라 데이터 반출 정책 확인이 먼저다).

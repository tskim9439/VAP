---
type: output
status: active
created: 2026-10-02
updated: 2026-10-02
summary: 동시 번역의 단위 조사 — 현 <SEM_END> 는 문장급(EN 11 s·KO 8 s/단위)이라 WRITE 단위로 너무 크다, 번역 기준 의미 단위(MU)를 따로 정의할 것
sources:
  - '[[source-cst-s2tt-project-spec-v0-1]]'
  - '[[output-semcommit-recipe-v0.3]]'
related:
  - '[[conversational-simultaneous-s2tt]]'
  - '[[cst-bench]]'
raw_authors:
  - tskim
---

# 동시 번역은 어떤 단위로 써야 하나 — `<SEM_END>` 는 적합한가 (2026-10-02)

## 질문

- 지금 라벨링한 `<SEM_END>` 위치를 CST-S2TT 에서 번역을 쓰는(WRITE) 단위로 그대로 쓸 수 있는가?
- 동시 번역 연구는 실시간 번역을 어떤 단위로 수행하는가?

## 결론

1. **`<SEM_END>` 는 사실상 문장급 단위라 WRITE 단위로는 너무 크다.**
   - 단위 하나의 평균은 영어 25.7 단어·11.0 초, 한국어 10.6 어절·8.4 초다(아래 실측).
   - 사람 통역사의 지연은 영→한 평균 약 3 초다.
   - Hibiki 는 단어 단위 정렬로 LAAL 5.6 초인데, 문장 단위로 쓰면 21.5 초였다.
   - `<SEM_END>` 로 쓰면 동시통역이 아니라 문장 단위 순차 번역에 가깝다.
2. **정의 축이 다르다.**
   - `<SEM_END>` 는 원천 언어 혼자서 판단한 “전사가 의미상 완결되어 바뀌지 않을 자리”다.
   - 동시 번역의 단위는 **번역 결과가 더 이상 바뀌지 않는 원천 접두**로 정의된다(Meaningful Unit, 번역 접두 일관성).
   - 그래서 언어쌍과 방향(KO→EN / EN→KO)에 따라 달라진다.
3. **한국어 규칙 음성은 번역 단위와 충돌한다.**
   - 레시피 v0.3.3 은 한국어 연결어미(-고·-서·-면·-는데…)로 끝난 자리를 확정 금지(N)로 둔다.
   - 그런데 KO→EN 동시통역에서는 연결어미 절 끝이 **대표적인 청크 경계**다. 술어가 이미 나와서 “…, and”로 이어 번역할 수 있다.
4. **그래도 버릴 것은 아니다.**
   - 문장 끝 `<SEM_END>` 는 거의 확실히 번역 단위의 경계이기도 하다(정밀도는 높고 재현율이 낮을 것으로 예상, 미검증).
   - 두 가지 용도가 남는다:
     - 큰 단위의 확정·문맥 리셋 지점(revision-free 채점의 상위 경계)
     - 통제 세트의 끼어들기 난이도 “Easy(문장 끝)”
5. **권고: 번역 단위를 따로 정의해 라벨링한다.**
   - 방법: 교사 LLM 번역으로 번역 접두 일관성(MU) 또는 문맥 정렬(contextual alignment)을 계산한다.
   - 그 전에 기존 `<SEM_END>` 와 MU 의 겹침을 재는 작은 실험으로 1–4 를 확인한다(§5).

## 1. 지금 `<SEM_END>` 의 실제 단위 (실측)

- 대상: 스냅숏 20260929-1637 main 라벨에서 무작위 파트 언어별 25 개(영어 8,501 스트림·37.4 h, 한국어 4,461 스트림·14.4 h). 등급 A 만 셌다.
- 스크립트: mxc `VAPKT-data/results/sem_unit_stats.py`

| | 단어(어절)/A | 초/A | A 사이 단어 수 중앙 / p90 | 구두점 가지 A | 스트림 끝 A | A 없는 스트림 |
|---|---:|---:|---:|---:|---:|---:|
| 영어 | 25.7 | 11.0 | 16 / 36 | 57 % | 45 % | 18 % |
| 한국어 | 10.6 | 8.4 | 9 / 16 | 97 % | 66 % | 5 % |

- 레시피 v0.3.3 의 결론과 일치한다: 8B 교사에서 A 는 사실상 참조 전사의 문장 끝 구두점이다. → [[output-semcommit-recipe-v0.3]]
- 한국어는 A 의 66 % 가 스트림 끝이라, 발화(턴) 끝과 거의 같다.

## 2. 동시 번역 연구의 단위

| 계열 | 단위 | 정하는 방법 | 대표 |
|---|---|---|---|
| 고정 정책 | 단어·토큰(k 개 뒤) | 규칙 | wait-k (STACL, Ma et al. 2019) |
| 적응 정책(학습) | 단어·음성 프레임마다 READ/WRITE | 모델이 판단 | MILk·MMA, DiSeg(Zhang & Feng, ACL 2023 Findings: 음성 특징마다 분절 여부를 미분 가능하게 학습), EASiST(2025, 명시적 read/write 토큰 + 정책 헤드) |
| 의미 단위(MU) | 원천 접두 중 그 번역이 **전체 문장 번역의 접두**가 되는 최소 구간 | 번역 모델로 라벨 생성 → 분류기 | Zhang et al. EMNLP 2020: 중→영·독→영에서 wait-k·chunk 보다 품질-지연 우세. 사람 통역의 MU 묶기에서 착안 |
| 문맥 정렬 | 목표 단어마다 “예측 가능해지는” 원천 위치 | MT 로그우도가 가장 크게 오르는 원천 위치 | Hibiki(Kyutai 2025): 프→영, 문맥 정렬 LAAL 5.6 s vs 문장 정렬 21.5 s, BLEU 25.9 vs 25.6 |
| 구문 청크 | 명사구·동사-목적어·구두점 경계 | 의존 구문 분석 | SASST(AAAI 2026): 구문 인지 제거 시 BLEU 38.5 → 23.2(비슷한 지연), 청크의 82 % 가 구문 경계와 일치(고정 길이 23 %) |
| 정책 없음(재번역) | 새 입력마다 처음부터 다시 번역, 수정 허용 | — | Arivazhagan et al. 2020: 품질·지연은 좋지만 화면 깜빡임(수정)이 많다 |
| 분절 없는 스트림 | 끝없는 오디오에서 이력 선택 | 주의 기반 | StreamAtt + StreamLAAL(Papi et al., ACL 2024), InfiniSST(대화형 multi-turn 정식화) |

공통 결론:
- 문장 단위는 품질은 높지만 지연이 문장 길이만큼 붙는다.
- 너무 짧은 단위(고정 k)는 어순이 먼 쌍에서 품질이 무너진다.
- 그 사이의 **의미·구문 청크 + 적응 정책**이 표준이다.

## 3. 어순이 먼 언어쌍(한↔영)

- 한국어·일본어는 엄격한 핵 후치(head-final), 영어는 핵 선행이다.
- 문장 단위 병렬 코퍼스로 학습하면 지연이 커지거나 아직 오지 않은 말(동사)을 예측해야 한다(Han et al., WMT 2021 — 목표 문장을 청크 단위로 재배열·정제해 단조 정렬을 만든 뒤 wait-k 로 학습, BLEU·단조성 개선).
- 영→일 동시통역사는 원천을 청크로 끊어 **청크 단위 단조 번역**(chunk-wise monotonic translation, CWMT)을 한다.
  - 이를 참조로 쓰면 동시 모델이 문장 번역 참조보다 높게 평가된다(Doi et al., IWSLT 2024).
  - 시사점: 평가 참조도 단위 설계와 맞물린다.
- 방향마다 단위가 달라야 한다:
  - KO→EN: 술어(절 끝 어미)가 나와야 번역이 안정된다. 연결어미 절 경계는 좋은 단위다. 문장 끝까지 기다릴 필요는 없다.
  - EN→KO: 영어 구(명사구·전치사구) 단위로 앞당겨 쓸 수 있다. 다만 한국어 동사를 끝에 두려면 재배열·보류가 필요하다.
  - 사람 통역에서도 원천 언어에 따라 분절 방법이 달라야 한다는 보고가 있다(ETRI 계열 동시통역 연구·특허, 미검증 2차 인용).
- 사람의 지연(ear-voice span): 영→한 약 800 문장 분석에서 평균 3 초(Lee, Meta 2002). 다른 연구는 EVS 3.66 초·TTS 5.34 초를 보고했다.

## 4. 대화(CST)에서 추가로 생기는 단위 요구

- **턴 끝은 강제 확정 지점이다.** 상대가 말을 시작하면 이전 방향은 flush 하거나 멈춘다(Stop Latency).
  - 단위가 크면 끼어들기 때 버려야 할 미번역 분량이 커진다.
- **맞장구는 단위를 만들면 안 된다.** “네/yeah” 가 번역 단위로 잡히면 잘못된 방향 전환이 된다.
  - 기존 `<SEM_END>` 레시피의 대답어 가지(`punct_resp`)·스트림 머리 대답어 음성(`reply_prefix`)은 이 판단에 재사용할 수 있다.
- **자기 정정**(“오전에… 아니 오후 세 시”)은 확정 금지 구간이다. 기존 Stage C 의 REVISION 판단과 같은 문제다.

## 5. 다음 실험 제안 — `<SEM_END>` 대 번역 단위 겹침 측정

1. 라벨된 스트림 일부(언어별 수백 개)를 교사 LLM 으로 번역한다(KO→EN, EN→KO, 판정 LLM 규칙대로 Qwen3.8-27B·EXAONE-4.0).
2. 단위 계산:
   - **MU 경계**: Zhang 2020 Algorithm 1. 접두 번역(이전 확정분 강제)이 전체 번역(빔 후보 집합)의 접두가 되는 최소 위치
   - **문맥 정렬**: Hibiki 식. 목표 단어별로 원천 위치의 로그우도 증가가 최대인 곳
3. 비교 지표:
   - `<SEM_END>` A 의 MU 경계 정밀도·재현율
   - 단위 길이(초) 분포
   - 연결어미 N 자리의 MU 비율
4. 결과에 따라 번역 단위 라벨(`<MU>` 또는 WRITE 지점)을 새로 정의하고, 통제 세트의 끼어들기 난이도(Easy·Medium·Hard)를 그 단위에 맞춘다.

## 출처 (관측일 2026-10-02)

- Zhang et al., Learning Adaptive Segmentation Policy for Simultaneous Translation, EMNLP 2020 — https://aclanthology.org/2020.emnlp-main.178 (본문 확인: MU 정의·Algorithm 1)
- Zhang & Feng, End-to-End Simultaneous Speech Translation with Differentiable Segmentation, ACL 2023 Findings — https://arxiv.org/abs/2305.16093
- Labiausse et al.(Kyutai), High-Fidelity Simultaneous Speech-To-Speech Translation (Hibiki), 2025 — https://arxiv.org/html/2502.03382v1 (문맥 정렬 식·Table 4 확인)
- SASST: Leveraging Syntax-Aware Chunking and LLMs for Simultaneous Speech Translation — https://arxiv.org/abs/2508.07781 (검색 요약 기준)
- Arivazhagan et al., Re-translation versus Streaming for Simultaneous Translation, IWSLT 2020 — https://aclanthology.org/2020.iwslt-1.27
- Papi et al., StreamAtt, ACL 2024 — https://aclanthology.org/2024.acl-long.202
- Han et al., Monotonic Simultaneous Translation with Chunk-wise Reordering and Refinement, WMT 2021 — https://aclanthology.org/2021.wmt-1.119
- Doi et al., Word Order in English-Japanese Simultaneous Interpretation (CWMT), IWSLT 2024 — https://arxiv.org/abs/2406.08940
- EASiST, Efficient and Adaptive Simultaneous Speech Translation with Fully Unidirectional Architecture — https://arxiv.org/abs/2504.11809
- Lee, Ear Voice Span in English into Korean Simultaneous Interpretation, Meta 47(4) 2002 — https://www.erudit.org/fr/revue/meta/2002/v47/n4/008039ar.pdf (검색 요약 기준)
- 그 밖: InfiniSST·Seed LiveInterpret 2.0·STACL(검색 결과 요약만 확인)

불확실성:
- SASST·EASiST·EVS 수치는 검색 요약으로만 확인했다. 본문은 열지 않았다.
- 한국어 연결어미 절이 좋은 KO→EN 단위라는 판단은 언어학적 추론이다. §5 실험으로 확인해야 한다.

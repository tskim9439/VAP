---
type: output
status: active
created: 2026-10-07
updated: 2026-10-07
summary: 표준 평가셋 계획 — 스트리밍 ASR(LibriSpeech·FLEURS·Kspon)·동시 음성번역(IWSLT26 MCIF·MuST-C·FLEURS·CoVoST2·kosp2e)·2인 ASR+diarization, 지표·지연 구간·확보 방법
sources:
  - '[[2026-10-07-eval-set-survey]]'
related:
  - '[[decision-eval-train-decontamination]]'
  - '[[output-paper-draft-cst-s2tt-icml-20261005]]'
  - '[[decision-cst-paper-v1-scope]]'
  - '[[cst-bench]]'
---

# 표준 평가셋 계획: 스트리밍 ASR · 동시 음성번역 · 2인 대화 (2026-10-07)

## 질문
- 최종 모델이 나와도 **스트리밍 ASR** 과 **Simul-S2TT** 평가는 필수다(사용자, 2026-10-06). CST-Bench 외에 어떤 공개 평가셋·지표·지연 기준을 써야 기존 연구와 비교되는가?
- 근거: 관련 논문·모델 카드·공유 과제 페이지를 직접 읽은 조사 4 건(`raw/sources/research/2026-10-07-eval-set-survey/`).

## 결론
- **영어 ASR** 은 LibriSpeech test-clean/other 가 사실상 필수다(조사한 스트리밍 연구 13 개 중 11 개). **독일어·한국어 스트리밍 ASR** 의 공통 비교 기준은 **FLEURS** 뿐이다(Nemotron 3.5 가 80–1120 ms 청크별 수치 공개).
- **동시 음성번역(영→독)** 의 기준이 IWSLT 2026 에서 바뀌었다. 분할 없는 장문(MCIF dev / ACL Talks test), 주 지표 **XCOMET-XL + LongYAAL(계산 제외)**, 지연 구간 **저 0–2 s · 고 2–4 s** 이다. 기존 논문 비교용으로 MuST-C tst-COMMON 을 함께 쓴다.
- **독→영** 은 IWSLT 트랙이 없다. FLEURS(SeamlessStreaming), CoVoST 2/CVSS(StreamSpeech), Europarl-ST·Audio-NTREX-4L(Hibiki-Zero)이 기준이다.
- **한↔영** 은 공개 표준이 사실상 **FLEURS** 뿐이다(SeamlessStreaming 이 방향별 수치 공개). 한→영 **kosp2e**, 영→한 **EnKoST-C**(ETRI, TED)가 보조다.
- **두 언어 2인 대화 + 화자 귀속 번역** 의 공개 평가셋은 없다. 가장 가까운 JSTAR 의 RealConv(스페인어↔영어)는 사내 데이터다 → CST-Bench 의 필요성 근거.
- **2인 단일 채널 ASR + diarization** 표준은 CALLHOME part 2(2인 부분집합)·CH109·Fisher test(모두 LDC) 와 LibriSpeechMix 2mix(무료)다.
- **가장 가까운 선행 모델: Factorized DSM**(arXiv 2610.04333, 2026-10-03) — Nemotron-3.5 스트리밍 인코더 + Qwen3.5-0.8B + 지연 방출. 우리 백본과 거의 같은 구성이라 백본 자체는 새로움으로 내세우기 어렵다. 관련 연구에 넣고 LibriSpeech 에서 비교한다.

## 1. 스트리밍 ASR (단일 화자)

| 언어 | 필수 | 권장 | 선택 | 비교 대상(공개 수치) |
|---|---|---|---|---|
| 영어 | LibriSpeech test-clean/other(보유), FLEURS en_us | Open ASR Leaderboard 8 세트(AMI, Earnings22, GigaSpeech, LS, SPGI, TED-LIUM 3, VoxPopuli) | VoxPopuli-AA(보유, 학습과 화자 겹침 → 부록) | Nemotron-EN(8 세트, 청크 80–1120 ms), Kyutai STT(8 세트, 지연 0.5/2.5 s), F-DSM(LS, 160–800 ms), Moshi(LS), Whisper-Streaming·SeamlessStreaming(DSM 논문의 재실행 수치) |
| 독일어 | FLEURS de_de | Common Voice de(Mozilla Data Collective), MLS de | – | Nemotron 3.5 FLEURS de WER 9.81→8.31(80→1120 ms), Voxtral Realtime(240–2400 ms) |
| 한국어 | KsponSpeech eval-clean/other(보유), FLEURS ko_kr | – | Zeroth-Korean(CC BY 4.0) | Nemotron 3.5 FLEURS ko CER 7.59→7.12, Qwen3-ASR(오프라인 CER 2.57, 1.7B), Voxtral Realtime(한국어를 WER 로 채점 — 직접 비교 불가) |

- 지표: 영어·독일어 WER(Whisper 정규화기, Open ASR Leaderboard 코드와 같게). 한국어는 **공백 제외 CER 을 주 지표**로, Kspon 논문 비교용으로 공백 포함 CER 도. Kspon 참조는 철자 전사(spelling) 쪽을 쓴다(ESPnet 관례).
- 지연: 설정 지연(δ, 청크) + **강제 정렬 단어 끝 대비 단어 방출 지연**(DSM·Whisper-Streaming 방식) + RTF.
- 직접 다시 돌려야 하는 것: Whisper-Streaming(독일어), SeamlessStreaming(언어별 ASR 수치 없음), Qwen3-ASR 스트리밍 모드(독·한 수치 없음).

## 2. 동시 음성번역 (단일 화자, 고정 방향)

| 방향 | 필수 | 권장 | 선택 | 비교 대상 |
|---|---|---|---|---|
| 영→독 | **MCIF 영→독 장문**(IWSLT26 dev, CC BY 4.0), **MuST-C v1.0 tst-COMMON**(전체 강연 = 장문 / 문장 분할 두 방식) | FLEURS 영→독 | ACL 60/60 dev, IWSLT26 ACL test(참조 공개 여부 확인) | IWSLT 2025·2026 참가 시스템, InfiniSST·StreamAtt(장문), AlignAtt·EDAtt(문장), SeamlessStreaming(FLEURS: AL 1.87–2.05 s, BLEU 29.8–30.2) |
| 독→영 | FLEURS 독→영, CoVoST 2 독→영 test | Europarl-ST 독→영(단문), Audio-NTREX-4L(장문, TTS) | – | SeamlessStreaming(FLEURS: AL 1.68–1.85 s, BLEU 34.0–34.4), StreamSpeech(CVSS-C, S2TT 는 재실행 필요), Hibiki-Zero(독→영 BLEU 28.7 단문 / 29.1 장문) |
| 영→한 | FLEURS 영→한 | EnKoST-C tst-COMMON(ETRI, TED, 접근 조건 확인 필요) | – | SeamlessStreaming(AL 2.06–2.30 s, BLEU13a 8.3–8.7) |
| 한→영 | FLEURS 한→영 | kosp2e test(CC BY/BY-SA/BY-NC-SA, 학술용) | AI Hub 71693·71379 검증셋(내국인 전용, 비교 대상 없음) | SeamlessStreaming(AL 3.26–3.58 s, BLEU 20.9–21.2), Whisper(한→영 21.3) |

- 품질 지표: **XCOMET-XL(IWSLT26 주 지표)**, COMET(wmt22), BLEU·chrF. BLEU 토크나이저는 반드시 명시한다. 한국어 목표는 ko-mecab(sacreBLEU 기본)과 Seamless 비교용 13a 를 함께 쓴다.
- 지연 지표:
  - 장문: **LongYAAL**(OmniSTEval ≥ v0.1.10, 이전 버전은 과소 추정 버그) + StreamLAAL(IWSLT25·InfiniSST·StreamAtt 비교)
  - 문장 단위: AL·LAAL(+YAAL)
  - 계산 제외(CU)를 항상 보고하고, 계산 포함(CA)은 하드웨어를 밝혀 함께 보고(papi2025how 권고)
- 지연 구간: 표준 벤치마크는 **IWSLT 구간(저 ≤ 2 s, 고 2–4 s, 계산 제외)** 으로 보고한다. 시스템마다 여러 운영점으로 곡선도 그린다.
  - 참고: CST-Bench 의 동시 구간(턴 절반, TAXI 1.54 s·2.31 s)이 IWSLT 저지연 구간과 비슷해 해석이 맞물린다.
  - 한국어 목표의 지연 단위(문자/토큰)를 명시한다.

## 3. 2인 대화 ASR + diarization (보조)

| 세트 | 성격 | 접근 | 비교 대상 |
|---|---|---|---|
| CST-Bench(TAXI·합성) | 두 언어 2인, 단일 채널 | 보유·구축 중 | 본 논문 주 평가 |
| LibriSpeechMix test-clean 2mix | 낭독 2인 혼합, 무료 | LibriSpeech 로 생성 | t-SOT, Sortformer, multitalker-parakeet(cpWER) |
| CALLHOME part 2 2인 부분집합(148 녹음) | 8 kHz 전화 2인 | LDC2001S97 | Streaming Sortformer(0.32/1.04/10 s DER), pyannote |
| CH109 | 2인 전화 109 세션 | LDC97S42 | Streaming Sortformer, multitalker-parakeet(cpWER), JEDIS-LLM |
| Fisher English test(172 대화) | 2인 전화 | LDC2004S13 | DiarizationLM, JEDIS-LLM(WDER·cpWER) |

- DER 규약을 섞지 않는다: Sortformer 계열은 0.25 s collar·겹침 포함, pyannote 는 collar 0. 둘 다 보고한다.
- 대화 음성번역 공개 세트(Fisher-CALLHOME 스→영, DiariST-AliMeeting 중→영)는 우리 언어 범위 밖이라 쓰지 않는다. SA-BLEU(DiariST)는 CST-Bench 에서 지표로만 함께 보고한다.

## 4. 확보 상태와 할 일

| 구분 | 세트 | 할 일 |
|---|---|---|
| 보유(mxc) | LibriSpeech test, KsponSpeech eval, VoxPopuli-AA | 바로 평가 가능 |
| 공개·무료(사용자가 T5 에 받아 업로드) | FLEURS(en_us·de_de·ko_kr test), MCIF, MuST-C v1.0 En-De(신청서), Europarl-ST, Audio-NTREX-4L, kosp2e(한국어 스크립트는 요청), Zeroth, Open ASR 8 세트 묶음(`hf-audio/esb-datasets-test-only-sorted`, 일부 gated) | p0: FLEURS·MCIF·MuST-C, p1: 나머지 |
| 조건부 | CoVoST 2 독→영(Common Voice 독일어 음성은 Mozilla Data Collective 경유), Common Voice de, EnKoST-C(ETRI 포털), AI Hub(내국인 전용) | 접근 경로 확인 |
| 유료·라이선스 | CALLHOME·CH109·Fisher(LDC) | 회사 LDC 회원 여부 확인 |

- **확인 필요(규정): AI Hub 이용 약관에는 국외 반출 시 별도 협약이 필요하다는 조항이 있다.** KsponSpeech 가 이미 mxc 에 있으므로, mxc(Azure)의 리전이 국내인지 확인해야 한다.
- 채점기 정렬: `cstbench eval` 에 XCOMET-XL 과 LongYAAL(OmniSTEval) 을 추가해 IWSLT26 과 같은 숫자를 낸다. 장문 재분할은 IWSLT26 이 SoftSegmenter 를, 기존 연구가 mwerSegmenter 를 쓴다. 우리 채점기는 mwerSegmenter 방식이므로 두 방식을 모두 보고할 수 있게 한다.

## 4-1. 확보 결과와 겹침 (2026-10-08)
- 사용자가 Europarl-ST v1.1(영↔독)과 YODAS-Granary 독일어(2,966 파일, 약 844 GB, 원본 목록과 일치)를 mxc `VAPKT-data/data/corpora/` 에 받았다.
- Europarl-ST 영→독 test 의 영어 연설 일부가 백본 학습(VoxPopuli 영어)과 겹친다 → [[decision-eval-train-decontamination]]

## 5. 논문 반영 (안)
- 본문 6 절에 "Standard tasks" 표 하나:
  - ASR: LibriSpeech test-other, FLEURS de·ko, Kspon eval-other
  - 번역: MCIF 영→독(저지연 구간), FLEURS 독→영·영→한·한→영
  - 각 시스템의 주 운영점 값
- 부록: 세트별 품질-지연 곡선, Open ASR 8 세트, MuST-C, CoVoST 2, Europarl-ST, kosp2e, LibriSpeechMix·CALLHOME·CH109.
- 관련 연구: Factorized DSM, Voxtral Realtime, Kyutai STT, IWSLT 2026, LongYAAL/OmniSTEval 추가(참고문헌 검증 필요).

## 불확실성
- IWSLT 2025 test 와 2026 dev(MCIF)의 중복 여부, IWSLT26 ACL test 참조 공개 여부는 확인하지 못했다.
- Voxtral Realtime 은 한국어를 WER 로 채점해 다른 시스템과 직접 비교할 수 없다.
- Nemotron 3.5 카드는 MCV·MLS 도 평가셋으로 적었지만 FLEURS 수치만 공개했다.
- EnKoST-C·MLC-SLM 의 라이선스, kosp2e 일부의 비상업 조건, AI Hub 데이터를 평가에 쓰는 것이 "학습용" 조항에 해당하는지 확인이 필요하다.
- 조사는 원문을 직접 열어 확인했지만, 웹 검색 없이 진행해 최근 논문 목록은 망라적이지 않다.

# PLAN — 단계별 실행 계획

> README 의 로드맵(4 단계)을 **실행 단위**로 푼 문서다. 단계마다 목표 · 학습 DB · 모델/시퀀스 설정 · 평가 · 관문 · 이월 조건 · 연결된 태스크를 둔다.
> 상태와 수치는 각 단계의 "현재" 절에만 쓴다. 실험 일지는 `wiki/tasks/`, 결과 보고서는 `wiki/outputs/`.
> 갱신 규칙: 관문을 통과·실패하거나 DB/평가 정의가 바뀔 때 갱신한다. 일일 진행은 여기 쓰지 않는다.

**최종 갱신: 2026-09-05** · 현재 단계: **Stage 1 진입 직전**

---

## 원칙

1. **복잡도를 한 축씩 올린다.** 단일 화자 ASR → 대규모 → 두 채널·화자 태그 → 겹침 → turn-taking 헤드 → 실전 제약. 두 축을 동시에 바꾸지 않는다.
2. **각 단계는 이전 단계의 체크포인트에서 시작한다.** 새 단계에서 실패하면 그 단계에서 추가한 축만 의심한다.
3. **관문은 같은 encoder 의 자체 baseline 을 분모로 둔다.** 비인과 오프라인 모델을 인과 스트리밍 모델의 기준으로 쓰지 않는다(U0.5 에서 배운 것).
4. **공통 불변 설정** — 80 ms causal chunk · Nemotron `[56,0]` frozen · `<NEXT_AUDIO>/<EMPTY_AUDIO>` · `<DELAY_d>` 무작위화 · M=4 · KV cache · committed-prefix decode. 이 설정은 단계가 올라가도 바꾸지 않는다.

---

## Stage 0 — 준비 (완료, 2026-09-03 ~ 09-05)

| 항목 | 결과 | 기록 |
|---|---|---|
| 환경 | rack4 `tskim_env` + **mxc `sa_tskim` conda `vapasr`**, 스모크 통과 | `decision-compute-environment`, `.env` |
| Causality 감사 | Nemotron `[56,0]` ≤80 ms · Qwen AuT 비인과 → **Lookahead 관문 발동, encoder 확정** | `output-encoder-causality-audit` |
| 표현 비교 probing | 7 encoder × 205 h 캐시(447 GB) → causal probe. **H1 미지지**(프레임율 > 인코더) | `output-stage1-encoder-probing` |
| VAP baseline | TurnBench dev 0.841 @ FP 0.045 / 463 ms 재현 | `output-vap-turnbench-baseline-reproduction` |
| 데이터 검증 | AI Hub 분리 stereo · otoSpeech · TurnBench · target 파이프라인(55,139 창) | `output-vap-target-pipeline` |
| U0 정렬 | ForcedAligner 로 3 코퍼스 전량, M=4·δ=2 QC | `task-uslm-feasibility-u0` |
| U0.5 adapter bridge | Nemotron→thinker adapter, RNN-T 대비 −6 pt → **통과** | `output-uslm-u05-adapter-bridge` |
| U1 two-speaker 선행 실험 | v0 방출 붕괴 → v1 next_weight 0.3 으로 해소, 정확도 미달(EN 0.52) → **단일 화자로 회귀 결정** | `task-uslm-u1-interleaved-asr` |

Stage 0 의 두 결정이 이후를 규정한다: (a) backbone = Nemotron `[56,0]` → adapter → Qwen3-ASR thinker, (b) two-speaker 를 한 번에 풀지 않는다.

---

## Stage 1 — 단일 화자 스트리밍 ASR 파일럿 (모델 시퀀스 탐색)

**목표** — `[z_k] → text 0..M → <NEXT_AUDIO>` 라는 **시퀀스 자체**가 학습 가능한지, 방출 결정·정렬·지연 파이프라인이 설계대로 동작하는지 작은 데이터로 빠르게 확인한다. 성능이 아니라 **동작의 정상성**을 본다.

**학습 DB**

| 코퍼스 | 분량 | 경로 | 비고 |
|---|---|---|---|
| LibriSpeech `train-clean-100` | 100 h EN | `$MXC_LIBRISPEECH_DIR/train-clean-100` | 같은 speaker/chapter 안에서 20–30 s 스트림으로 묶는다 |
| KsponSpeech 부분집합 | ~100 h KO | `$MXC_KSPONSPEECH_DIR/KsponSpeech_01` 일부 | 발화 단위 배포 → 가변 앞뒤 침묵을 둔 독립 시퀀스 |

두 코퍼스 모두 mxc 공용 영역에 있다(읽기 전용). 정렬은 Qwen3-ForcedAligner(`$MXC_ALIGNER_DIR`)로 80 ms 격자에 붙인다.

**모델 / 시퀀스** — U0.5 adapter + LoRA 에서 초기화. `<SPK_A/B>`·두 채널 merge·overlap 직렬화·VAP/event 헤드는 **제거**. `next_weight` 는 v1 에서 확인된 0.3 부근에서 시작하되 sweep 한다.

**평가**

| 지표 | 데이터 | 기준 |
|---|---|---|
| WER / CER | LibriSpeech `dev-clean`·`dev-other`, KsponSpeech `eval_clean`·`eval_other` | 학습 진행에 따라 **지속 하락**(절대값 관문 없음) |
| 방출 건강도 | held-out 스트림 | tok/chunk 가 참조 토큰율(EN·KO 각각)에 수렴, `<NEXT_AUDIO>` 지배 없음, M 강제 비율 < 5 % |
| 지연 | 정렬 종료 시각 대비 | p50 ≈ δ·80 ms + 정렬 오차, **evidence-time 위반 0** |
| 속도 | 스트리밍 디코드 | p99 tick < 80 ms |

**관문** — 위 네 축이 모두 정상이면 통과. 절대 WER 은 묻지 않는다.

**실패 시** — Stage 2 로 가지 않는다. 의심 순서: (1) `<NEXT_AUDIO>` 손실 가중치·목표 시각 허용 창(±1 chunk), (2) 정렬 품질(QC 재검), (3) 방출 결정을 별도 헤드로 분리, (4) adapter 용량. 한 번에 하나만 바꾼다.

**현재** — v2(화자별 오디오 토큰) 진단 run 종료 후 착수. 필요한 코드: LibriSpeech/KsponSpeech 리더, mono 시퀀스 생성기(두 채널 코드에서 분기). 학습 서버는 **mxc** 권장(H200, 데이터 현지).

**태스크** — `task-uslm-u1-interleaved-asr`(U1a-0 절)

---

## Stage 2 — 단일 화자 대규모 ASR (성능 확보)

**목표** — 스트리밍 ASR 로서 **성립하는 성능**을 확보한다. 이 체크포인트가 Stage 3 이후의 출발점이 되므로, 여기서 밀리면 뒤가 전부 밀린다.

**학습 DB — 2 단계**

| 하위 단계 | 코퍼스 | 분량 | 비고 |
|---|---|---|---|
| 2-1 대규모 | LibriSpeech 960 h + KsponSpeech 965 h | ~1,900 h | **언어별 균형 sampling**(KO/EN batch 비율 실험). Stage 1 ckpt 에서 시작 |
| 2-2 대화 mono 적응 | otoSpeech · AI Hub TS_01_5/VS_02 의 **화자별 채널을 mono 스트림으로 분리** | ~350 h | 대화 mono 70–80 % + 대규모 DB replay 20–30 % (실험으로 조정). 자발 발화·맞장구·잡음 도메인으로 옮긴다 |

2-2 의 대화 코퍼스는 현재 rack4 에만 있다 — mxc 에서 돌리려면 `/data3/tskim/corpora` 와 정렬 결과(`manifests/align`)를 azcopy 로 옮겨야 한다.

**모델** — Stage 1 과 동일 시퀀스. encoder unfreeze 여부는 2-1 에서 ablation(기본 frozen).

**평가**

| 지표 | 데이터 | 기준 |
|---|---|---|
| WER (EN) | LibriSpeech `test-clean`·`test-other`, otoSpeech held-out | 공식 split 으로 보고 |
| CER (KO) | KsponSpeech `eval_clean`·`eval_other`, AI Hub VS_02 held-out | 〃 |
| **관문 분모** | **Nemotron RNN-T `[56,0]` 80 ms 스트리밍**, 같은 데이터 | 동일 encoder·동일 지연 예산 |
| 지연·속도 | Stage 1 과 동일 | p99 tick < 80 ms 유지 |

**관문** — 대화 mono 적응(2-2) 후 **WER/CER 상대 열화 ≤ 10 % vs Nemotron RNN-T `[56,0]`**, evidence-time 위반 0. (참고선: 오프라인 Qwen3-ASR — 분모로 쓰지 않는다.)

**실패 시** — Stage 3 진행 금지. 재설계 한 차례: loss weighting · 정렬 · emission state machine · encoder unfreeze 범위. 그래도 미달이면 IS-SLM 구조 자체를 재고한다(fallback 없음).

**태스크** — `task-uslm-u1-interleaved-asr`(U1a-1, U1a-2 절)

---

## Stage 3 — 다화자 대화: 화자 구분 + turn-taking

**목표** — Stage 2 체크포인트에 **대화의 축**을 하나씩 더한다: 두 채널 → 화자 태그(경량 diarization) → 비중첩 대화 → 실제 overlap → turn-taking 헤드. 마지막에 H2 를 판정한다.

**학습 DB** — otoSpeech(EN, 104.9 h) · AI Hub 성인 TS_01_5 실내 196.6 h + VS_02 실외 51.7 h(KO) · TurnBench dev(평가 전용). 모두 화자별 분리 채널. target(VAP256·hazard·이벤트)은 `vapasr/data/targets.py` 로 이미 생성돼 있다(55,139 창).

**하위 단계와 추가하는 축**

| 하위 | 추가하는 것 | 데이터 처리 | 관문 |
|---|---|---|---|
| 3-1 두 채널 | 화자별 오디오 토큰(chunk 당 A·B 2 개, merge 없음 — v2 진단 결과로 확정) | 비중첩 구간만 | Stage 2 관문 유지 |
| 3-2 화자 태그 | `<SPK_A/B>` 방출 (**diarization 은 별도 모듈이 아니라 이 토큰이다**) | 〃 | 화자 귀속 오류율 보고, WER 유지 |
| 3-3 overlap | 겹침 직렬화 규약, backlog 관리 | 실제 overlap 포함 | WER 유지 + backlog p99 보고 |
| 3-4 turn-taking 헤드 | VAP256 · next-onset hazard τ · VAD 헤드 + `<SPEECH_ONSET|ENDPOINT>` 토큰 | 전체 | 아래 turn 평가 |
| 3-5 하이브리드 ablation | 50 Hz CPC 사이드 브랜치 (Stage 0 의 프레임율 결과 흡수) | 〃 | 3-4 대비 EOT/INT 개선 여부 |

**평가 — TurnBench 프로토콜** (`turn-taking-evaluation-protocol`)

| 능력 | 지표 | 비교 대상 |
|---|---|---|
| EOT / INT / backchannel | recall @ FP ≤ 0.045 · 0.10, latency p50 | VAP oto fine-tune(0.841/463 ms) · **같은 encoder 의 encoder-only probe**(Stage 0, 0.868/553) · DualTurn · cascade(ASR→LLM endpoint) |
| 전사 | WER/CER | **가드레일: Stage 2 대비 열화 ≤ 5 %** |
| 화자 | 귀속 오류율, overlap 구간 별도 | — |
| 시스템 | 총 RTF, tick p99, backlog | 한 모델 vs cascade 비용 |

**H2 판정** — "IS-SLM 상태 위 헤드" 가 "같은 encoder 의 encoder-only probe" 보다 EOT/INT 에서 유의하게 나으면 지지. 아니면 통합의 가치는 시스템 이점(한 모델·공유 계산)으로 제한된다.

**실패 시** — stereo 에서만 실패하면 3-1~3-3 을 분리 ablation(어느 축이 깨뜨리는지). turn 헤드에서 실패하면 손실 가중·헤드 구조·50 Hz 하이브리드 순으로. Stage 2 결과를 건드리지 않는다.

**태스크** — `task-uslm-u1-interleaved-asr`(U1b/c) · `task-uslm-u3-multitask` · `task-add-missing-baselines` · `task-time-to-next-turn-survival-head` · `task-event-label-heuristics-validation`

---

## Stage 4 — 실전 스트리밍

**목표** — 벤치마크 모델을 **실제로 돌아가는 스트리밍 시스템**으로 만든다. 노출 편향·지연 제어·장문·한국어 평가·배포.

| 하위 | 내용 | 평가 / 관문 | 태스크 |
|---|---|---|---|
| 4-1 자기 이력 조건화 | 교사 강제 100 % → self history 혼합, corruption 학습, 20–60 s 창 carry | **gold/self WER·delay 격차** 보고·축소 | `task-uslm-u2-self-conditioned` |
| 4-2 적응 방출 | 지연 조건부 학습 → WER–delay RL(Muse 식) | Pareto 개선 (chunk 80/160/320 곡선) | `task-latency-quality-curve` |
| 4-3 장문·배포 | audio KV 요약, p99/backlog hard limit, 모델 확대(0.6B → 1.7B 검토) | **1 h 연속 RTF**, p99 tick < 80 ms 유지 | — |
| 4-4 한국어 벤치마크 | 어노테이션 프로토콜(IPU 3–5 k 지점, Fleiss κ), 배포는 어노테이션 레이어만 | IAA, 재현 가능한 채점기 | `task-korean-benchmark-design` · `decision-korean-benchmark-release-scope` |
| 4-5 강건성 | KO/EN temperature sampling, 잡음(실외 도메인), 최종 공동 미세조정 | 실외 CER, 언어별 균형 | `task-bilingual-and-qwen-port` |

4-1 은 Stage 3 결과에 따라 **앞당길 수 있다** — Stage 1–2 에서 노출 편향이 정확도 병목으로 확인되면 Stage 2 안에서 먼저 한다.

---

## 관문 요약

| 관문 | 조건 | 결과 |
|---|---|---|
| Lookahead | backbone lookahead > 320 ms | **발동** → Nemotron `[56,0]` 확정, Qwen AuT 제외 |
| H1 | probing 에서 기각 | **기각됨** → turn 기대치 하향, 50 Hz 하이브리드 필수(3-5) |
| U0.5 | adapter 가 RNN-T 보다 나쁨 | **통과** |
| Stage 1 | 방출·정렬·지연 비정상 | Stage 2 금지, 시퀀스 재설계 |
| Stage 2 | WER/CER 열화 > 10 % 또는 evidence 위반 | Stage 3 금지, 한 차례 재설계 후 구조 재고 |
| Stage 3 | stereo 에서만 실패 / turn 헤드 실패 | 축 분리 ablation / 헤드 재설계 |
| AI Hub 약관 | 어노테이션 파생물 공개 불가 | 한국어 기여를 내부 평가로 격하 (**약관 원문 확인 미완**) |
| 디스크 | rack4 /data4 < 200 G | 체크포인트 정리 → mxc Lustre 로 이전 검토 |

---

## 데이터 한눈에

| 코퍼스 | 언어 | 분량 | 형태 | 쓰는 단계 | 위치 |
|---|---|---|---|---|---|
| LibriSpeech | EN | 960 h (+dev/test) | 낭독, mono, jsonl 동봉 | 1, 2 | mxc `$MXC_LIBRISPEECH_DIR` |
| KsponSpeech | KO | 965 h (+eval) | 자유대화 **발화 단위**, mono | 1, 2 | mxc `$MXC_KSPONSPEECH_DIR` |
| otoSpeech | EN | 104.9 h | 대화, 화자별 채널 | 2-2, 3 | rack4 `/data3/tskim/corpora` |
| AI Hub 성인 TS_01_5 / VS_02 | KO | 196.6 / 51.7 h | 대화, 분리 stereo, 실내/실외 | 2-2, 3, 4-5 | rack4 (성인 전체 2,765 h 수신 가능) |
| TurnBench dev / test | EN | 7.3 h / 116 대화 | 평가 전용, gold EOT/INT | 3 | rack4 |
| 특징 캐시 · 정렬 | — | 447 GB · 3 코퍼스 | Stage 0 산출물 | 2-2 부터 재사용 | rack4 → mxc 전송 필요 |

라이선스: LibriSpeech CC BY 4.0 · KsponSpeech AI Hub 123 · AI Hub 71631/71632(재배포 제약) · otoSpeech·TurnBench(Sesame). 최종 모델은 OpenMDW-1.1(Nemotron) + Apache 2.0(Qwen3-ASR) 병기.

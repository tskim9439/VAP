# VAP-ASR — Streaming Conversational Projection

> **한 줄**: streamable speech encoder 하나에서 **실시간 전사**와 **앞으로 2 초의 대화 역학**(누가 언제 말할지, 끼어들기, 맞장구)을 동시에 예측하는 한국어·영어 모델을 만든다.
> 이 README 는 **현황판**이다 — 목표·구조·로드맵·완료된 결과·열린 TODO·데이터/인프라·문서 지도를 한 곳에 둔다. 마일스톤마다 에이전트가 갱신한다.
> 태스크 상세는 `wiki/tasks/`, 생성 대시보드는 `wiki/todo.md`, 단계별 체크리스트는 `TODO.md`, 운영 규칙 정본은 `AGENTS.md`.

**최종 갱신: 2026-09-04** · 현재 단계: **Phase 1 (Stage 1 매트릭스 완료 직전) → IS-SLM U0 착수** · 주력 모델 = **IS-SLM** (이중 프레임율+RNN-T 안은 2026-09-04 기각) · GPU: rack4 #1

---

## 1. 목표와 연구 질문

기존 endpointing(Muse Voice Transcribe 등)은 "**지금** 발화가 끝났는가"를 묻는다. 이 프로젝트는 VAP(Voice Activity Projection)처럼
"**앞으로** 누가 언제 말할 것인가"를 묻되, 그것을 streaming ASR 과 **하나의 표현/모델**에서 낸다. 사람은 상대 턴 종료 **−151 ms** 에 이미 움직이고,
최고 성능 VAP 는 **368 ms**(TurnBench test) — 이 격차가 연구 대상이다.

매 시점 `t` 의 streaming audio `x_≤t` 에서 동시에:
- `P(Y_text | x_≤t)` — 스트리밍 전사
- `P(A_{t:t+2s} | x_≤t, Y_≤t)` — 미래 2 s 화자 활동 / floor 전이 / interruption / backchannel

**가설 (각각 하나의 ablation)**
- **H1** ASR 로 사전학습된 streaming 표현이 CPC 등 SSL 표현보다 turn-taking 에 유리하다 → Stage 1 (결과: 아래 §4, **현재까지 지지되지 않음**)
- **H2** ASR 의 incremental linguistic state 를 음향 VAP 와 결합하면 mid-turn pause 와 true EOT 를 더 잘 구분한다 → Stage 3 / U3
- **H3** binary EOT 보다 future activity + time-to-next-turn(hazard) joint 예측이 더 빠르면서 FP 가 낮다 → Phase 3

경쟁 가설: **DualTurn**(dual-channel 생성형 사전학습, VAP 대비 F1 0.633 vs 0.389)은 "ASR 사전학습이 답"이라는 H1 과 정면 경쟁 — 필수 baseline.

---

## 2. 목표 모델 구조 — IS-SLM 단일 주력

**Interleaved Streaming SLM (IS-SLM)** — `wiki/outputs/output-interleaved-streaming-slm-architecture.md`, 결정 `decision-target-architecture` (2026-09-04 확정)

```text
 audio 80 ms → Nemotron FastConformer [56,0] (causal ≤80 ms, 12.5 Hz) → new adapter (→13 Hz, thinker 임베딩 공간) → soft token z_k (joint chunk token)
   → Qwen3-ASR-0.6B-hf thinker LM (LoRA, KV cache) → text / <SPK_A|B> / <SPEECH_ONSET|ENDPOINT> 토큰 0..M 개 → <NEXT_AUDIO> 로 다음 chunk 대기
   → 같은 audio-position hidden 위 병렬 헤드: VAP256 (미래 2 s) · next-onset hazard (τ, censored) · VAD
 융합: 명시적 합산 없음(기본) / zero-init gated contextual residual (ablation)
 볼트 보완: 50 Hz 음향 사이드 브랜치 하이브리드(U3 ablation) · joint chunk token · 겹침 텍스트 직렬화 규약 · U1 WER 관문
```

- 두 종류의 출력 clock: sparse(전사·화자·onset/endpoint·제어)는 **토큰**, dense(VAP·hazard)는 **audio-clock 병렬 헤드**.
- Muse 는 closed weights — 설계 참조 + API black-box 비교 대상. 차별점은 **미래 투사 헤드**와 **두 화자 스트림**.
- **기각된 안**: 이중 프레임율(50 Hz CPC + 12.5 Hz FastConformer) + RNN-T 융합 모델 (`output-model-architecture-proposal`, superseded).
  그 실측 근거(50 Hz 가 turn 타이밍에 유리)는 IS-SLM 의 **50 Hz 사이드 브랜치**로 흡수한다.
- **최종 backbone (2026-09-04 확정)**: **Nemotron 3.5 FastConformer `[56,0]` → new adapter → Qwen3-ASR-0.6B-hf thinker LM** (LoRA). U0.5 adapter bridge test로 연결 품질을 조기 검증하되, AuT로의 계획된 교체는 없다 → `decision-asr-backbone`.
- 대조군은 외부 것: VAP(oto fine-tune) · TurnBench 동봉 baseline · Nemotron RNN-T(전사) · Stage 1 frozen probe(표현별 turn 상한).
  "통합의 가치" = **같은 encoder 의 encoder-only probe 대 IS-SLM 상태 위 헤드**.

---

## 3. 로드맵

### Paper 1 — 표현 비교(Stage 1) + IS-SLM U0–U3 (Phase 0–4)

| Phase | 내용 | 상태 |
|---|---|---|
| **0** 검증·환경 | 컨테이너 학습 스택, causality 감사, AI Hub 실물 검증, VAP baseline 재현, target 파이프라인 | **완료** (5/5) |
| **1** Stage 1 | frozen encoder 7종 특징 캐시 → 고정 용량 causal probe 매트릭스 (고유율 / 12.5 Hz) → H1 판정 | 캐시 완료, **매트릭스 13/14 + 재채점 진행 중** |
| 2 IS-SLM 본체 | **U1** interleaved ASR(WER 관문) → **U2** self-conditioned → **U3** 멀티태스크 헤드 + 50 Hz 하이브리드 + H2 판정 | **U1 착수(09-04)** |
| 3 Objective | τ hazard head (censoring), 이벤트 라벨 검증(TurnBench dev gold 로 즉시 가능) | 대기 |
| 4 한국어 | 어노테이션 프로토콜 설계(IPU 3–5 k 지점, Fleiss κ), 배포는 어노테이션 레이어만(AI Hub 재배포 제약) | 대기 |
| 5 Bilingual/robustness | KO/EN temperature sampling, 잡음 강건성, 최종 backbone 공동 미세조정 | 대기 |

### USLM 트랙 U0–U5 — U0–U3 은 Paper 1(Phase 2), U4–U5 는 Paper 2

| 단계 | 내용 | 관문 | 상태 |
|---|---|---|---|
| **U0** streamability + 타당성 | encoder truncation audit(완료) · Qwen3 tokenizer 토큰율 → chunk 당 M 예산 · ForcedAligner 정렬·QC · interleaved target 생성기 | — | **진행 중(09-04)** — 토큰율(KO p99 0.78 tok/80 ms, 폭주 없음)·**M=4 확정**(aihub 전체 정렬 기준 이월 2.1 %)·생성기 완료; 정렬 aihub-ts01-5 완료, otoSpeech 334/420 |
| **U0.5** adapter bridge test | 캐시 특징 증류로 Nemotron→thinker adapter 초기화 → 오프라인 ASR 미세조정 → WER | **≤ +10–15 % vs AuT+thinker / RNN-T** | **실험 완료(09-04)** — 최선 oto 18.2–18.8 / 실내 17.5 / 실외 14.5–15.1 %. 관문 원안 부분 미달이나 동일 인코더 RNN-T(25.4 %) 대비 −6 pt → adapter 병목 아님. **관문 재정의 후 통과(09-04)** — RNN-T `[56,0]` 보다 우수 + 오프라인 대비 ≤ +50 %. 보고서 `output-uslm-u05-adapter-bridge` |
| **U1** aligned interleaved ASR | Nemotron frozen + Qwen3-0.6B(LoRA) + adapter, 텍스트 스트림만, δ 무작위화(`<DELAY_d>`), M=4 | **WER/CER 상대 열화 ≤ 10 % vs RNN-T `[56,0]`** | **착수(09-04)** — 데이터·모델·학습/스트리밍 평가 코드 완성, 스모크 후 v0 run |
| U2 self-conditioned streaming | self history 혼합, gold/self 격차, corruption 학습, 20–60 s 창 carry | 격차 보고 | 대기 (p1, ~10-30) |
| **U3** conversational multi-task | audio-clock 헤드 + onset/endpoint/speaker 토큰 + **50 Hz 사이드 브랜치 하이브리드** ablation, encoder-only probe 와 비교(H2) | WER 가드레일 ≤ 5 %, turn·총 RTF | 대기 (p0, ~11-27) |
| U4 adaptive emission | 지연 조건부 학습 → WER–delay RL (Muse 식) | Pareto 개선 | Paper 2 |
| U5 long-context·배포 | audio KV 요약, p99/backlog hard limit, 모델 확대 | 1 h RTF | Paper 2 |

### 결정 관문

| 관문 | 조건 | 결과 |
|---|---|---|
| Lookahead | backbone lookahead > 320 ms | **발동(Qwen AuT)** → encoder 는 Nemotron `[56,0]` 확정 |
| U0.5 | adapter bridge WER 열화 > 15 % | adapter 용량·정렬·초기화·unfreeze 범위를 재설계; backbone 자동 교체 없음 |
| H1 | Stage 1 에서 기각 | IS-SLM turn 기대치 하향; 정당성은 H2 + 시스템 이점(한 모델·공유 계산). 50 Hz 하이브리드 필수 |
| U1 | WER 열화 > 10 % | IS-SLM 구조 재고 — **fallback 없음**(A 기각). 남는 것은 RNN-T + encoder probe 스택 → 조기 판정 |
| AI Hub | 약관상 어노테이션 파생물 공개 불가 | 한국어 기여를 내부 평가로 격하 (**약관 원문 확인 미완**) |
| /data4 | 여유 < 200 G | 체크포인트 정리 |

---

## 4. 완료된 것과 핵심 결과

### Phase 0 (2026-09-03)

| 항목 | 결과 |
|---|---|
| 환경 | rack4 `tskim_env`, conda `vapasr`(py3.11), torch 2.6+cu124, NeMo git 3.1, qwen-asr 0.0.6, 원 VAP. 스모크 4종 통과 |
| **Causality 감사** | 절단 실험(fp32). CPC 0 ms · **Nemotron [56,0] ≤80 ms** · Qwen AuT **기본 sdpa 경로 비인과(마스크 미호출 버그)**, 블록 모드 0–800 ms(평균 420). 마스크 주입 패치로 chunked-causal(WER +5.9 %) / frame-causal(≤80 ms, WER +23 %) 확보 |
| AI Hub 실물 | 16 kHz 진짜 분리 stereo(누설 −64 dB), 라벨 온셋 오차 중앙값 30 ms, 성인 **2,765 h**. 서버 보유 TS_01_5 196.6 h + VS_02 51.7 h. API 키로 직접 수신 |
| VAP baseline | TurnBench dev 에서 oto fine-tune ckpt **0.841 @ FP 0.045 / 463 ms** 공식 예측과 완전 일치. 리더보드 0.845/0.055/368 은 test split |
| Target 파이프라인 | `vapasr/data/`: 코퍼스 3종(AI Hub·otoSpeech·TurnBench dev) → VAD@50 Hz → VAP256(원 VAP 코드)·τ hazard·이벤트 파생. 20 s 창 55,139개 |

### Phase 1 — 특징 캐시 (2026-09-04): 7 인코더 × 205 h = **447 GB**, 세그먼트 이어붙이기 fp32 exact (수용장 107 s·TF32·**AuT 13 Hz**·Whisper 정규화 함정 해결)

### Stage 1 매트릭스 — TurnBench dev, frozen encoder + 동일 causal head(2.83 M), 4 epoch, 고정 FP 예산 최대 recall

| encoder | Hz | causal | EOT R@fp≤0.045 / p50 | EOT R@fp≤0.10 / p50 | INT R@fp≤0.10 / p50 | 한국어 val CE 실내 / **실외** |
|---|---:|---|---|---|---|---|
| fbank (사전학습 없음 floor) | 50 | ✓ | 0.799 / 988 ms | 0.838 / 761 | 0.533 / 1462 | — |
| **cpc** | 50 | ✓ | **0.880 / 499** | 0.893 / 448 | 0.922 / 987 | 2.90 / 3.18 |
| cpc → 12.5 Hz | 12.5 | ✓ | 0.867 / 527 | 0.873 / 486 | 0.914 / 1037 | — |
| nemotron-c0 | 12.5 | ✓ | 0.868 / 553 | 0.868 / 528 | 0.916 / 1338 | 3.11 / 3.30 |
| qwen-aut-causal | 13 | ✓ | 0.862 / 444 | 0.865 / 423 | **0.945 / 909** | 2.88 / **5.81** |
| qwen-aut-cc1s (블록 끝으로 접음) | 13 | 블록(0–800) | 0.834 / 456 | 0.841 / 1517† | 0.841 / 1517 | 2.54 / **6.52** |
| wavlm-base / large | 50 | ✗ (20 s 창) | 0.80 / — | 0.81 / — | 0.90 / — | — |
| **VAP oto fine-tune (기준)** | 50 | ✓ | 0.841 / 463 | — | 0.957 / 896 | — |

† cc1s 는 TurnBench causality 규약대로 1 s 블록 끝으로 확률을 접은 뒤의 값 — 접기 전(371/243 ms)은 무효. WavLM 은 비인과 참조(latency 무의미).

**읽기 (잠정)** — 결과 페이지 `wiki/outputs/output-stage1-encoder-probing.md`
1. frozen probe 가 같은 FP 에서 fine-tune VAP 보다 EOT recall 이 높다 (latency 는 40–90 ms 느림). 학습 데이터가 더 많음(oto 105 h + AI Hub 50 h)은 명시.
2. **H1 미지지**: Nemotron(12.5 Hz) ≈ CPC@12.5 Hz < CPC@50 Hz. **프레임율 효과 > 인코더 효과** → IS-SLM U3 의 50 Hz 사이드 브랜치 하이브리드 근거.
3. INT 는 사전학습 유무에 크게 의존(fbank 0.53 vs 0.92) — qwen-causal 이 INT 최강(0.945). run 간 분산 큼 → seed 반복 필요.
4. **Qwen AuT 는 잡음에 취약**: 실내(학습 도메인) CE 는 CPC 와 같거나 낫지만 실외(SNR +5/+10 dB)에서 붕괴. 스튜디오 벤치마크만으로는 안 보이는 축.
5. INT latency 1.0–1.6 s 로 VAP(896) 보다 느림 — probe 구조/손실에서 볼 지점.
6. lookahead 를 정직하게 접으면 cc1s 의 이점은 사라진다(EOT 0.834 / 456 ms) — 블록 lookahead 는 latency 로 그대로 돌아온다.

---

## 5. TODO — 열린 태스크 (2026-09-04, 상세는 `wiki/tasks/`, 대시보드 `wiki/todo.md`)

| 우선 | 마감 | 태스크 | 비고 |
|---|---|---|---|
| **p0** | 10-22 | Stage 1 encoder probing — **doing** | 매트릭스 마무리 → seed 반복 · EN-only 학습 조건 · 최종 표·H1 판정 |
| p1 | 09-17 | 영어 코퍼스 확보 | otoSpeech·TurnBench 완료, **CANDOR·SpokenWOZ 채널 확인 남음** |
| p1 | 09-18 | **USLM U0** streamability+타당성 — **doing** | 토큰율 ✓(KO 폭주 없음, M=4), 생성기 ✓, ForcedAligner 정렬 실행 중(밤새) |
| **p0** | 09-25 | **USLM U0.5** adapter bridge test — **doing** | 기준선: 오프라인 Qwen 13.9/13.5/13.3 %, 스트리밍 RNN-T[56,0] 35.8/37.0/34.5 %; random-init 3000 step 18.7/20.6/18.3 %; 증류(cos 0.77) init 6000 step: lr 2e-4 20.0/18.3/16.9 %, **lr 1e-4 18.2/19.2/16.1 %** (수렴 전, 관문 16.0/15.5 미달); random-init 12k **20.1/17.7/16.6 %**, 증류-init 12k **19.3/17.8/15.1 %**(8000 step 18.8/17.5/14.5) — 4 run 모두 oto 18–20 / 실내 17–19 / 실외 15–17 % 수렴, 증류 이득은 잡음 범위. adapter+thinker 는 이미 같은 `[56,0]` 인코더의 자체 RNN-T(25.4/23.4/24.5 %)보다 우수 → **관문 재정의(RNN-T 우수 + 오프라인 ≤ +50 %) 후 통과, U1 착수** |
| **p0** | 10-16 | **USLM U1** interleaved ASR | Nemotron frozen + Qwen3-0.6B LoRA, **WER 관문 ≤10 %** |
| p1 | 10-30 | USLM U2 self-conditioned | gold/self 격차 |
| **p0** | 11-27 | **USLM U3** 멀티태스크 | 헤드 + 50 Hz 하이브리드 + H2 |
| p1 | 09-19 | 체크포인트 보존 정책 | /data4 여유 ~520 G, best+last 규칙 |
| p1 | 09-24 | Paper 1 범위 확정 | 주장 한 문장, 학회·마감, 필수 실험 목록 |
| p1 | 10-29 | 누락 baseline | turnbench 동봉 rms_vad·dualturn·wavlm_causal 재사용, cascade 실측 |
| p1 | 10-29 | latency-quality 곡선 | chunk 80/160/320 + lookahead 회계 + ko-KR CER |
| p1 | 12-10 | τ hazard head | H3 |
| p1 | 12-24 | 한국어 벤치마크 설계 | 어노테이션 레이어 배포 |
| ~~p1~~ | ~~01-28~~ | ~~Qwen AuT causal 적응~~ — **취소** | 최종 backbone에서 AuT를 사용하지 않으므로 주 경로에서 제거 |
| p2 | 12-10 | 이벤트 라벨 휴리스틱 검증 | TurnBench dev gold(EOT 1,904/INT 347)로 즉시 가능 |
| p2 | 01-14 | 한국어 turn 단서 문헌 | |
| p2 | 02-11 | Bilingual + robustness | 최종 backbone 유지, KO/EN temperature sampling |

**사용자 확인 필요**: AI Hub 이용약관의 어노테이션 파생물 공개 조항 · GPU 배정(날마다 `.env.local` `GPU_DEFAULT`) · `AGENTS.md` 유지보수 PR(`.env`, `sync-rack4.sh`, `TODO.md`, `experiments/`, `vapasr/` 반영).

**완료 (6)**: 환경 구축 · causality 감사 · AI Hub 실물 검증 · VAP baseline 재현 · target 파이프라인 · 특징 캐시. **폐기 (2)**: Stage 2/3 (A 안 전용).

---

## 6. 데이터·인프라 현황

| | 값 |
|---|---|
| 서버 | `rack4` (A100-PCIE-40GB ×4, 공용) · 컨테이너 `tskim_env`(root) · 오늘 GPU = `.env.local` `GPU_DEFAULT` |
| 프로젝트 | `/home/tskim/VAP` — 로컬이 정본, `scripts/sync-rack4.sh push` 로 미러 (bind mount 라 컨테이너가 즉시 봄) |
| 코퍼스 (`/data3/tskim/corpora`) | AI Hub 성인: TS_01.실내_5 **196.6 h**(757 wav), VS_02.실외 **51.7 h**(186), 라벨 11,023 JSON · otoSpeech **104.9 h**(420, 16 k 사본 `otoSpeech16k`) · TurnBench dev 38(7.3 h) / test 116 |
| Manifest (`/data3/tskim/manifests`) | aihub-ts01-5 · aihub-vs02 · otoSpeech · turnbench-dev — VAD@50 Hz npz + 이벤트 + QC PNG |
| 특징 캐시 (`/data3/tskim/features`) | fbank · cpc · nemotron-c0 · qwen-aut-causal · qwen-aut-cc1s · wavlm-base · wavlm-large × 4 manifest, 447 GB |
| 체크포인트 | `/data4/tskim/VAPASR/experiments/probe/<run>/` (probe.pt, results.json, predictions-dev.json) — **/data4 97 % 사용, 보존 정책 미정** |
| 서드파티 | `/data3/tskim/third_party/{VoiceActivityProjection,vap_turn_taking,datasets_turntaking,turnbench}` |
| 디스크 | /data3 1.8 TB 여유 · /data4 ~520 G |

---

## 7. 코드 맵

| 경로 | 역할 |
|---|---|
| `vapasr/data/` | `corpora.py`(AI Hub·otoSpeech·TurnBench 리더) · `vad.py`(에너지 VAD) · `targets.py`(VAP256·τ hazard·이벤트) · `dataset.py`(원본 오디오 20 s 창) |
| `vapasr/features/encoders.py` | frozen encoder 7종 공통 `encode()`, 세그먼트 이어붙이기(수용장·격자·트림) |
| `vapasr/probe/` | 캐시 특징 로더(frame rate 통일) · 고정 용량 causal probe head |
| `experiments/` | `build_targets.py` · `extract_features.py` · `run_feature_cache.sh` · `train_probe.py`(TurnBench 자동 채점, `--eval-only`) · `run_stage1.sh` · `show_probe_results.py` · `causality_audit.py` · `qwen_aut_mask.py` · `reproduce_vap_turnbench.sh` · `aihub_label_stats.py` · `verify_aihub_sample.py` · 진단 `diag_*.py` |
| `scripts/` | `sync-rack4.sh`(push/pull/exec/shell/**bg/jobs**) · `setup-container-env.sh` · `activate-env.sh` · `aihub-download.sh` · `aihub-upload.sh` · 볼트 운영 4종 |
| `.env` / `.env.local` | 경로·서버·캐시·데이터셋 ID(커밋) / API 키·HF 토큰·GPU_DEFAULT(비커밋) |
| `raw/sources/experiments/` | 실험 원본 기록(JSON, QC PNG) |

자주 쓰는 명령:
```bash
./scripts/sync-rack4.sh status | push | jobs                 # 상태 · 동기화 · bg 작업
./scripts/sync-rack4.sh bg <이름> '<명령>'                    # 세션과 분리된 서버 작업
./scripts/sync-rack4.sh exec 'python experiments/show_probe_results.py --task eot'
```

---

## 8. 문서 지도 (`wiki/`)

- 계획: `outputs/output-streaming-vap-research-plan` (v2) · **주력 구조 `output-interleaved-streaming-slm-architecture`** · 합산 v0+12항목 평가 `output-unified-slm-architecture-plan` · (superseded) `output-model-architecture-proposal`
- 결과: `output-encoder-causality-audit` · `output-vap-turnbench-baseline-reproduction` · `output-vap-target-pipeline` · `output-feature-cache-and-compute-budget`
- 결정: `decisions/decision-asr-backbone` · `decision-target-architecture` · `decision-korean-benchmark-release-scope` · `decision-compute-environment`
- 개념: `voice-activity-projection` · `streaming-causality-and-latency-budget` · `turn-taking-objectives` · `turn-taking-evaluation-protocol` · `acoustic-linguistic-fusion` · `korean-turn-taking-cues`
- 상태·로그: `status.md` · `log.md` · `index.md`(생성) · `todo.md`(생성)

---

## 9. 볼트 운영 (요약)

이 저장소는 Karpathy 의 LLM Wiki 패턴을 따른다: `raw/` 불변 원천, `wiki/` 에이전트 유지 합성. **운영 규칙 정본은 `AGENTS.md`**.

```bash
./scripts/init-local-user.sh   # .llm-wiki-local/user.yaml (커밋 안 됨)
./scripts/install-skills.sh    # .skills/ → 런타임 링크
```
| 하고 싶은 것 | 방법 |
|---|---|
| 자료 추가 | `raw/inbox/` 에 넣고 "ingest 해줘" |
| 질문 / 태스크 / 린트 / 병합 준비 | wiki-query · wiki-task · wiki-lint · wiki-merge 스킬 |

`wiki/index.md` · `wiki/log.md` · `wiki/todo.md` 는 생성 파일 — 직접 편집 금지. `main` 은 보호(PR 병합).
Git 원격: `origin` = github.com:tskim9439/VAP, **SSH 443 경유**(`ssh://git@ssh.github.com:443/tskim9439/VAP.git` — 회사망이 22 번 차단). 첫 커밋 2026-09-04.

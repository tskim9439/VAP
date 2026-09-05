# VAP-ASR — Streaming Conversational Projection

> **한 줄**: 80 ms 단위로 들어오는 대화 음성에서 **지금 말한 것(전사)** 과 **앞으로 2 초 안에 일어날 일(누가 말을 이어갈지·끊을지·맞장구칠지)** 을 **하나의 스트리밍 모델**이 동시에 낸다. 한국어·영어.
>
> 이 README 는 **현황판**이다 — 목표·동작 방식·4 단계 로드맵·현재 위치·인프라·문서 지도만 둔다.
> 단계별 세부 계획(학습 DB·평가·관문·이월 조건)은 **[`PLAN.md`](PLAN.md)**, 태스크 단위 기록은 `wiki/tasks/`, 운영 규칙 정본은 `AGENTS.md`.

**최종 갱신: 2026-09-05** · 현재 위치: **Stage 1 (단일 화자 파일럿) 진입 직전** — two-speaker 진단 run(v2) 마무리 중 · 주력 구조 = **IS-SLM** · 학습 서버: rack4 (진행 중 run) + **mxc 준비 완료**

---

## 1. 목표

기존 endpointing(Muse Voice Transcribe 등)은 "**지금** 발화가 끝났는가"를 묻는다. 이 프로젝트는 VAP(Voice Activity Projection)처럼 "**앞으로** 누가 언제 말할 것인가"를 묻되, 그것을 **스트리밍 ASR 과 같은 모델·같은 표현**에서 낸다.

- 사람은 상대 턴이 끝나기 **−151 ms** 전에 이미 움직인다. 최고 성능 VAP 는 **368 ms**(TurnBench test). 이 격차를 줄이는 것이 목표다.
- 전사와 대화 역학을 따로 돌리는 cascade 가 아니라, **한 모델이 한 번의 계산**으로 둘 다 내야 한다 — 지연·비용·일관성 모두에서 이유가 있다.
- 최종 산출물: 한국어·영어 대화에 대해 **80 ms 마다** (전사 토큰, 화자, 미래 2 s 활동 확률, 다음 발화 시작까지의 시간 분포)를 내는 단일 체크포인트와, 그것을 재는 평가 프로토콜.

**검증할 가설** — 각각 하나의 ablation 으로 대응한다.

| | 가설 | 어디서 |
|---|---|---|
| H1 | ASR 로 사전학습된 스트리밍 표현이 CPC 등 SSL 표현보다 turn-taking 에 유리하다 | Stage 0 probing — **현재까지 지지되지 않음**(프레임율 효과가 더 큼) |
| H2 | ASR 의 incremental linguistic state 를 음향 VAP 와 결합하면 mid-turn pause 와 true EOT 를 더 잘 가른다 | Stage 3 |
| H3 | binary EOT 보다 future activity + time-to-next-turn(hazard) joint 예측이 더 빠르면서 FP 가 낮다 | Stage 3–4 |

경쟁 가설 **DualTurn**(dual-channel 생성형 사전학습, VAP 대비 F1 0.633 vs 0.389)은 필수 baseline 이다.

---

## 2. 예상 동작 방식

### 2.1 시간축에서 본 입출력

두 화자의 오디오가 80 ms chunk 로 들어오고, 모델은 매 chunk 마다 **토큰 0~M 개**(전사·화자·제어)를 내고 `<NEXT_AUDIO>` 로 다음 chunk 를 기다린다. 같은 시점의 hidden 위에서 **audio-clock 헤드**가 미래 2 s 활동과 다음 발화 시작 시각 분포를 낸다.

```text
시간 →      0 ms      80       160      240      320      400      480      560   ...
화자 A  ▁▁▁▁▁▁▁▁▁███████████████████████████████████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
화자 B  ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁████████████
             │        │        │        │        │        │        │        │
chunk       [z₀]     [z₁]     [z₂]     [z₃]     [z₄]     [z₅]     [z₆]     [z₇]
             │        │        │        │        │        │        │        │
토큰 출력    NEXT     <SPK_A>  "오늘"   NEXT     "삼청동"  NEXT     <ENDPOINT> <SPK_B>
(sparse)              "오"              "날씨"                       NEXT     "네"
                      NEXT              NEXT                                   NEXT
             │        │        │        │        │        │        │        │
VAP 헤드    ────────────────────────────────────────────────────────────────────
(dense,     P(A 계속) 0.9  0.9  0.9  0.85  0.7  0.4  0.2  0.1     ← 턴 종료를 미리 본다
80 ms 마다)  P(B 시작) 0.05 0.05 0.05 0.1  0.25 0.5  0.7  0.85    ← B 가 곧 말한다
hazard 헤드  τ(다음 onset 까지) 분포 — 560 ms 이전에 이미 "곧" 으로 수렴
```

- **전사는 토큰으로**(필요할 때만 방출, 지연 ≈ δ·80 ms + 정렬 오차), **대화 역학은 매 chunk 확률로**. 두 출력 clock 이 다르다.
- `<NEXT_AUDIO>` 는 RNN-T 의 blank 와 같은 역할이다. 학습 라벨의 70–85 % 가 이것이라 **가중치 보정 없이는 방출이 붕괴**한다(U1 v0 에서 실측).
- 지연 δ 는 학습 시 무작위화(`<DELAY_d>`)해 추론 시 조절 가능하게 둔다.

### 2.2 모델 구조 — IS-SLM (Interleaved Streaming SLM)

```text
 audio A ─┐  80 ms chunk
 audio B ─┴─→ Nemotron 3.5 FastConformer [56,0]  ─→ adapter ─→ soft token z_k ─┐
              (causal ≤80 ms · 12.5 Hz · frozen)    (1024-d → thinker 임베딩)    │
                                                                                │
   ┌────────────────────────────────────────────────────────────────────────────┘
   │
   └─→ Qwen3-ASR-0.6B thinker LM (LoRA, KV cache)
         ├─ 토큰 출력:  text · <SPK_A|B> · <SPEECH_ONSET|ENDPOINT> · <NEXT_AUDIO> · <DELAY_d>
         └─ 같은 audio-position hidden 위 병렬 헤드:
              VAP256 (미래 2 s 두 화자 활동) · next-onset hazard τ · VAD
              (+ Stage 3 ablation: 50 Hz CPC 사이드 브랜치 하이브리드)
```

- **backbone 은 확정**(2026-09-04): Nemotron `[56,0]` encoder + 새 adapter + Qwen3-ASR thinker. AuT 로의 교체 계획 없음 → `decision-asr-backbone`.
- Muse Voice Transcribe 는 closed weights — 설계 참조 + API black-box 비교 대상. 차별점은 **미래 투사 헤드**와 **두 화자 스트림**.
- 기각된 안: 이중 프레임율(50 Hz CPC + 12.5 Hz) + RNN-T 융합. 그 근거(50 Hz 가 타이밍에 유리)는 사이드 브랜치로 흡수.
- 구조 상세: `wiki/outputs/output-interleaved-streaming-slm-architecture.md`.

---

## 3. 로드맵 — 4 단계

복잡도를 한 번에 섞지 않는다. **단일 화자 ASR 을 먼저 통과시키고**, 그 위에 화자·겹침·turn-taking·실전 제약을 순서대로 얹는다. 각 단계의 학습 DB·평가·관문·이월 조건은 **[`PLAN.md`](PLAN.md)** 에 있다.

| 단계 | 무엇을 | 핵심 관문 | 상태 |
|---|---|---|---|
| **0** 준비 (완료) | 환경 · causality 감사 · 데이터 검증 · VAP baseline 재현 · 표현 비교 probing · 정렬 · adapter bridge | — | **완료** — H1 미지지, backbone 확정, U0.5 통과 |
| **1** 단일 화자 스트리밍 ASR **파일럿** | LibriSpeech 100 h + KsponSpeech 100 h 로 `[z_k] → text 0..M → <NEXT_AUDIO>` 시퀀스 자체를 검증. 방출·정렬·지연 파이프라인 탐색 | loss·방출 정상, held-out WER/CER 지속 하락, evidence-time 위반 0 | **다음 실행** (v2 진단 run 종료 후) |
| **2** 단일 화자 **대규모** ASR | LibriSpeech 960 h + KsponSpeech 965 h → 대화 mono(otoSpeech·AI Hub 분리 채널) 적응 | **WER/CER 상대 열화 ≤ 10 % vs Nemotron RNN-T `[56,0]`** | 계획 |
| **3** 다화자 대화 — 화자 구분 + turn-taking | 두 채널 → `<SPK_A/B>` → 비중첩 → 실제 overlap 순으로 복잡도 추가. VAP·hazard·VAD 헤드 + onset/endpoint 토큰. H2 판정 | WER 가드레일 ≤ 5 % · TurnBench EOT/INT recall@FP · 화자 귀속 오류 | 계획 (v1/v2 two-speaker 실험은 진단으로 보존) |
| **4** 실전 스트리밍 | 자기 이력 조건화(노출 편향) · 지연–정확도 적응 방출 · 장문 KV 요약 · 한국어 벤치마크 · 배포 | gold/self 격차 · WER–delay Pareto · 1 h RTF · p99 tick < 80 ms | 계획 |

**결정 관문(요약)** — 상세는 `PLAN.md` §관문.

| 관문 | 조건 | 결과 |
|---|---|---|
| Lookahead | backbone lookahead > 320 ms | **발동(Qwen AuT)** → Nemotron `[56,0]` 확정 |
| H1 | probing 에서 기각 | IS-SLM 의 turn 기대치 하향. 정당성은 H2 + 한 모델·공유 계산. 50 Hz 하이브리드 필수 |
| Stage 2 | WER/CER 열화 > 10 % 또는 evidence-time 위반 | Stage 3 진행 금지. loss weighting·정렬·emission·encoder unfreeze 재설계 |
| Stage 3 | Stage 2 통과 후 stereo 에서만 실패 | channel merge·speaker token·overlap 직렬화 분리 ablation. 실패 시 구조 재고 |

---

## 4. 현재 위치와 핵심 결과 (2026-09-05)

**지금**: two-speaker U1 v1(12k) 이 관문 미달(oto WER 0.52 / AI Hub 실내 CER 0.339 · 실외 0.271). 원인 분해로 화자 배정·겹침은 배제됐고 **단독 발화에서도 top-1 54–67 %** 가 본질 → 단일 화자부터 다시 쌓는 Stage 1 로 전환. v2(화자별 오디오 토큰)는 merge 분포 이동 가설의 진단 run 으로 완주시킨다.

**Stage 0 에서 얻은 것**

| 항목 | 결과 |
|---|---|
| Causality 감사 | Nemotron `[56,0]` lookahead ≤80 ms 확인. Qwen AuT 는 기본 경로 비인과(마스크 미호출), 블록 모드 0–800 ms |
| 표현 비교 probing (TurnBench dev) | CPC@50 Hz EOT 0.880/499 ms > Nemotron@12.5 Hz 0.868/553 ≈ CPC@12.5 Hz 0.867 → **프레임율 > 인코더**. Qwen AuT 는 실외 잡음에서 붕괴(CE 5.8 vs 3.3) |
| VAP baseline 재현 | oto fine-tune ckpt TurnBench dev **0.841 @ FP 0.045 / 463 ms** 공식과 일치 |
| 데이터 | AI Hub 성인 분리 stereo 확인(누설 −64 dB) · otoSpeech 104.9 h · TurnBench dev/test · 특징 캐시 7 인코더 × 205 h = 447 GB |
| U0 정렬 | 3 코퍼스 전량(otoSpeech 420 / ts01-5 190 / vs02 186), M=4 · δ=2 에서 otoSpeech 이월 0.62 % |
| U0.5 adapter bridge | oto 18.2–18.8 / 실내 17.5 / 실외 14.5–15.1 % — 동일 encoder RNN-T(25.4 %) 보다 우수, 오프라인 Qwen 대비 ≤ +50 % → **통과** |

전체 표와 해석: `wiki/outputs/output-stage1-encoder-probing.md`, `output-uslm-u05-adapter-bridge.md`, `wiki/tasks/task-uslm-u1-interleaved-asr.md`.

---

## 5. 데이터·인프라

| | rack4 (진행 중 run) | **mxc** (2026-09-04 구축) |
|---|---|---|
| GPU | A100-40 GB × 4, 공용 | **H200 143 GB × 8**, 공용 |
| 컨테이너 / env | `tskim_env` · conda `vapasr` | `sa_tskim` · conda `vapasr` (`$MXC_CONDA_DIR`, Lustre) |
| 프로젝트 | `/home/tskim/VAP` | `/soundai/users/tskim/VAPKT` (= Blob `users/tskim/VAPKT`) |
| 모델 | HF 캐시 | `/soundai/Model/{nemotron-3.5-asr-streaming-0.6b, Qwen3-ASR-0.6B, Qwen3-ForcedAligner-0.6B}` — 스모크 4/4 통과 |
| 대화 코퍼스 | `/data3/tskim/corpora`: AI Hub TS_01_5 196.6 h · VS_02 51.7 h · otoSpeech · TurnBench | (미전송) |
| ASR 코퍼스 (Stage 1–2) | — | **LibriSpeech 960 h · KsponSpeech 965 h** — `$MXC_LIBRISPEECH_DIR`, `$MXC_KSPONSPEECH_DIR` (공용, 읽기 전용) |
| 특징 캐시 · 정렬 결과 | `/data3/tskim/{features,manifests/align}` 447 GB | (미전송) |
| 체크포인트 | `/data4/tskim/VAPASR` — **97 % 사용** | `$MXC_CKPT_ROOT` (Lustre, 51 T 여유) |
| 동기화 | `scripts/sync-rack4.sh` | `scripts/sync-mxc.sh` — 둘 다 **`--delete` 없음**(원격 삭제 금지) |

모든 경로는 `.env` 한 곳에서 관리한다(mxc 는 `MXC_*` 접두사). 비밀·GPU 배정은 `.env.local`. mxc 망은 일부 CDN 이 차단돼 있다 — 우회 방법은 `.env` 주석 참고.

---

## 6. 코드 맵

| 경로 | 역할 |
|---|---|
| `vapasr/data/` | 코퍼스 리더 · VAD@50 Hz · VAP256/hazard/이벤트 target · 20 s 창 데이터셋 |
| `vapasr/features/encoders.py` | frozen encoder 7 종 공통 `encode()`, 세그먼트 이어붙이기 |
| `vapasr/probe/` | 캐시 특징 로더 · 고정 용량 causal probe head (Stage 0) |
| `vapasr/uslm/` | adapter · interleaved target/model(`model.py`: `next_weight`, `audio_spk`, `next_bias`) · 다음 변경 = mono 경로 + LibriSpeech/KsponSpeech 리더 |
| `experiments/` | `u0_align*.py` · `u05_*.py` · `u1_train_interleaved.py` · `u1_diag_emission.py` · `u1_show_sequence.py` · Stage 0 스크립트(`extract_features.py`, `train_probe.py`, `causality_audit.py` …) |
| `scripts/` | `sync-rack4.sh` · `sync-mxc.sh` · `setup-container-env.sh` · `wiki-regen.py` · 볼트 운영 |

```bash
./scripts/sync-mxc.sh status | push | exec '<명령>'      # mxc
./scripts/sync-rack4.sh status | push | bg <이름> '<명령>' | jobs   # rack4
```

---

## 7. 문서 지도

- **계획**: [`PLAN.md`](PLAN.md)(단계별 실행 계획) · `wiki/outputs/output-interleaved-streaming-slm-architecture`(구조) · `output-streaming-vap-research-plan`(연구 계획 v2)
- **결정**: `wiki/decisions/decision-asr-backbone` · `decision-target-architecture` · `decision-compute-environment` · `decision-korean-benchmark-release-scope`
- **결과**: `output-stage1-encoder-probing` · `output-uslm-u05-adapter-bridge` · `output-encoder-causality-audit` · `output-vap-turnbench-baseline-reproduction` · `output-feature-cache-and-compute-budget`
- **개념**: `voice-activity-projection` · `streaming-causality-and-latency-budget` · `turn-taking-objectives` · `turn-taking-evaluation-protocol` · `acoustic-linguistic-fusion` · `korean-turn-taking-cues`
- **태스크·상태**: `wiki/tasks/`(진행 기록) · `wiki/todo.md`(생성) · `wiki/status.md` · `wiki/log.md`

`TODO.md` 는 이전 Phase/Paper 구조의 체크리스트다 — `PLAN.md` 로 대체 예정.

---

## 8. 볼트 운영 (요약)

Karpathy 의 LLM Wiki 패턴: `raw/` 불변 원천, `wiki/` 에이전트 유지 합성. **운영 규칙 정본은 `AGENTS.md`**.

| 하고 싶은 것 | 방법 |
|---|---|
| 자료 추가 | `raw/inbox/` 에 넣고 "ingest 해줘" |
| 질문 / 태스크 / 린트 / 병합 | wiki-query · wiki-task · wiki-lint · wiki-merge 스킬 |

`wiki/index.md` · `log.md` · `todo.md` 는 생성 파일(직접 편집 금지, `scripts/wiki-regen.py`). `main` 은 보호(PR 병합).
Git 원격: `github.com:tskim9439/VAP`, SSH 443 경유(회사망 22 번 차단).

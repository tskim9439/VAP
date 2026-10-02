# CST-S2TT — Conversational Simultaneous Speech-to-Text Translation

> **한 줄**: 서로 다른 언어를 쓰는 두 사람이 대화할 때, AI 통역기가 상대 언어의 **텍스트 번역**을 실시간으로 내면서 다음을 함께 판단한다.
> - 언제 듣고 언제 쓰기 시작할지
> - 언제 기다리고 멈출지
> - 어느 방향(A→B / B→A)으로 전환할지
>
> 그 능력을 재는 벤치마크 **CST-Bench** 를 만든다.
>
> 이 README 는 **현황판**이다 — 목표·범위·단계·현재 위치·인프라·문서 지도만 둔다.
> - 단계별 실행 계획: **[`PLAN.md`](PLAN.md)**
> - 체크리스트: [`TODO.md`](TODO.md)
> - 태스크 기록: `wiki/tasks/`
> - 운영 규칙 정본: `AGENTS.md`

**최종 갱신: 2026-10-02** · 목표 전환: 스트리밍 ASR + turn-taking 투사(VAP-ASR) → **CST-S2TT**
- 결정: `wiki/decisions/decision-project-goal-cst-s2tt.md`
- 원 명세: `raw/sources/CST_S2TT_Project_Spec_v0.1.pdf`
- 이전 README·PLAN·TODO: `plans/*-vap-asr-*.md`

---

## 1. 목표

**핵심 메시지**: 기존 동시통역(SimulST)은 *음성 스트림*을 번역한다. 우리는 *대화*를 동시에 번역한다.
**핵심 가설**: 대화형 동시통역에서 번역 정책과 turn-taking 정책은 독립이 아니다.

```text
Speaker A (언어 X)  ↕  AI Interpreter  ↕  Speaker B (언어 Y)

매 시점:
  누가 말하는가 → 번역 방향은 → 진짜 턴인가 맞장구인가
  → 의미가 충분한가 → WRITE / WAIT → CONTINUE / STOP / SWITCH
```

개념 모형: P(Y_t, A_t, S_t | X_≤t, H_t).
- Y: 번역 토큰
- A: 상호작용 행동
- S: 화자·턴 상태
- X: 지금까지의 음성
- H: 대화 이력

| | 기존 SimulST | CST-S2TT |
|---|---|---|
| 원천 화자·번역 방향 | 주어짐·고정 | 실시간 판단·전환 |
| 입력 | 단일 스트림·발화 | multi-turn 대화 |
| semantic commit | 일부 연구 | 핵심 문제(확정 번역은 수정 불가) |
| 맞장구·끼어들기 | 보통 무시 | 명시적 처리 |
| 출력 | 텍스트 또는 음성 | **텍스트**(첫 단계) |

**첫 논문 범위**
- 포함:
  - 2 인, 서로 다른 언어, 실제 multi-turn
  - 양방향 동시 S2TT, semantic commitment
  - 턴 전환·맞장구·끼어들기
- 제외: TTS/S2ST·음성 자연성·3 인 이상·화자 분리·code-switching 중심 과제

**예상 기여**
1. 과제 정의
2. CST-Bench: 실제 대화 내용 + 통제된 상호작용 타이밍
3. 상호작용 지표 프로토콜: 방향·끼어들기·맞장구·Stop/Switch Latency
4. 번역·turn-taking·commit 정책을 함께 모델링하는 스트리밍 구조

## 2. CST-Bench 와 기준선

- **Natural Conversation Set**: 실제 대화 그대로
- **Controlled Interaction Set**:
  - 실제 대화의 내용·화자·턴 의존성은 유지하고 타이밍만 합성한다
  - 합성 유형: Normal · Early Turn · Interruption · Backchannel · Explicit Barge-in
  - 끼어들기 난이도(Easy/Medium/Hard)는 semantic chunk 경계 기준이다

| 기준선 | 구성 | 목적 |
|---|---|---|
| B0 | 오프라인 ST | 품질 참조 |
| B1 | 2×SimulST | 가장 단순한 양방향 |
| B2 | VAD + 2×SimulST | 음성 활동 라우팅 |
| B3 | VAP + 2×SimulST | turn-taking 예측 결합 |
| B4 | Oracle 턴 경계 + 2×SimulST | 상한 |
| B5 | Joint Conversational SimulST | 제안 모델 |

**지표**
- 번역: COMET·BLEU·chrF, 확정 텍스트만 채점
- 지연: AL·LAAL·첫 토큰·semantic commit
- 상호작용: 방향 정확도·잘못된 중단/전환·놓친 끼어들기·Stop/Switch Latency

**Kill criterion**: joint 모델 전에 B3 가 통제 세트의 상호작용 지표를 거의 다 푸는지 확인한다. 다 풀면 방향을 재검토한다.

## 3. 단계

| 단계 | 핵심 작업 | 산출물 | 상태 |
|---|---|---|---|
| **0** Feasibility / Data Access | 교차언어 대화 코퍼스 확보(TAXI·Verbmobil·CS-Dialogue·Bangor·ETRI), 라이선스 | 공통 schema 파서 | **다음** |
| **1** CST Data Pipeline | 턴 재구성·화자/언어 매핑·번역 정규화·합성 타임라인 | 통제 상호작용 샘플 생성기 | 계획 |
| **2** CST-Bench v0 | 두 세트·지표·revision-free commit 프로토콜 | 모델 독립 평가 도구 | 계획 |
| **3** Strong Baselines | 2×SimulST·VAD·VAP·Oracle, kill criterion | 파이프라인 한계 분석 | 계획 |
| **4** Joint Model | 턴·commit·번역 joint 모델 | 제안 모델 + ablation | 계획 |
| **5** Korean Extension | ETRI 또는 KO↔EN 소규모 수집 | 한국어 벤치마크 | 계획 |
| **6** S2ST Extension | 스트리밍 TTS 연결 | S2ST 데모 | 후속 |

**현재 최우선**: 모델보다 CST-Bench v0 와 strong pipeline 기준선을 먼저 완성한다.

## 4. 현재 위치 (2026-10-02)

- 목표를 전환한 직후다. 교차언어 대화 데이터는 아직 없다.
- 이전 목표에서 만든 것 중 이어지는 자산:

| 자산 | CST-S2TT 에서의 쓰임 | 기록 |
|---|---|---|
| 스트리밍 백본(Nemotron 인과 인코더 + Qwen3-ASR thinker, δ 지연, final 모드) | SimulST 원천 이해부, B5 출발점, B0 오프라인 참조 | `wiki/outputs/output-stage1-pilot-eval-20261001.md` |
| `<SEM_END>` 의미 commit 라벨·골드·commit 지연 지표 | semantic chunk 경계, 끼어들기 난이도, commit 지연 | `output-semcommit-recipe-v0.3`, `output-semcommit-gold-v1` |
| VAP TurnBench 재현 | B3 기준선 | `output-vap-turnbench-baseline-reproduction` |
| single-turn ASR 평가(VoxPopuli-AA + Kspon, 짝 bootstrap) | cascade 기준선 ASR 품질 관리 | `experiments/single-turn-asr-evaluation.md` |

- 진행 중 job(목표 전환 전 제출): semcommit 라벨링(79205). Stage 1 본학습 결과(q17-s1m-r0·r3)는 보존해 원천 ASR 기준선으로 쓴다.

## 5. 데이터·인프라

| | **mxc** (기본 서버) |
|---|---|
| GPU | H200 × 8 (로그인 호스트 직접 실행 가능), SLURM apex·hpc (제출은 사용자만) |
| 프로젝트 | `/soundai/users/tskim/VAPKT` |
| 산출물 | `/soundai/users/tskim/VAPKT-data` (`/lustre` 는 읽기 전용) |
| 모델 | `/soundai/Model/{nemotron-3.5-asr-streaming-0.6b, Qwen3-ASR-0.6B, Qwen3-ASR-1.7B, …}`, 체크포인트 `/soundai/Model/VAPASR` |

- 운영 규칙(요약):
  - 서버에서 파일·폴더 삭제 금지
  - 외부 데이터는 로컬 T5 SSD 에만 받고 서버 업로드는 사용자가 한다
  - 파일 전송은 azcopy 로 한다
  - SLURM 환경 변수는 `sbatch --export` 로만 넘긴다
- 경로는 `.env`, 비밀은 `.env.local` 에 둔다.

## 6. 코드 맵 (이전 목표에서 이어짐)

| 경로 | 역할 |
|---|---|
| `vapasr/hf/` | 스트리밍 모델(HF), 배치·오프라인 디코드, 2-pass, 실시간 세션 |
| `vapasr/data/` | semcommit 데이터셋, final 모드 형식, 잡음·속도 증강, single-turn 평가·AA-WER |
| `vapasr/features/` | Nemotron 온라인 인코더(att_context·SpecAugment) |
| `experiments/` | 학습(`semcommit_train.py`)·평가(`eval_single_turn_asr.py`, `semcommit_twopass_eval.py`)·라벨링(`semcommit_teacher*`) |
| `slurm/` | mxc 제출 스크립트(학습·라벨링) |

CST 전용 코드는 아직 없다. Phase 0–2 에서 데이터 파서·타임라인 생성기·평가 도구를 새로 만든다.

## 7. 문서 지도

- **목표·계획**:
  - [`PLAN.md`](PLAN.md)
  - `wiki/overview.md`
  - `wiki/decisions/decision-project-goal-cst-s2tt.md`
  - `wiki/sources/source-cst-s2tt-project-spec-v0-1.md`
- **개념**: `wiki/concepts/conversational-simultaneous-s2tt.md` · `wiki/concepts/cst-bench.md`
- **태스크·상태**: `wiki/tasks/task-cst-*.md` · `wiki/status.md` · `wiki/todo.md`(생성) · `wiki/log.md`(생성)
- **이전 목표 기록**: `plans/README-vap-asr-20260905.md` · `plans/PLAN-vap-asr-20260905.md` · `plans/TODO-vap-asr-20260903.md`

## 8. 볼트 운영 (요약)

Karpathy 의 LLM Wiki 패턴을 따른다. `raw/` 는 불변 원천이고 `wiki/` 는 에이전트가 유지하는 합성이다. **운영 규칙 정본은 `AGENTS.md`**.

| 하고 싶은 것 | 방법 |
|---|---|
| 자료 추가 | `raw/inbox/` 에 넣고 "ingest 해줘" |
| 질문 / 태스크 / 린트 / 병합 | wiki-query · wiki-task · wiki-lint · wiki-merge 스킬 |

- `wiki/index.md` · `log.md` · `todo.md` 는 생성 파일이다. 직접 편집하지 않는다.
- `main` 은 보호돼 있어 PR 로만 병합한다.

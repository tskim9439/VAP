# TODO — Streaming Conversational Projection ASR

> **실행용 체크리스트.** 단계 순서와 관문을 한눈에 보기 위한 파일이다.
> 각 항목의 상세·완료 조건은 `wiki/tasks/` 의 태스크 파일이 정본이고,
> owner 별 대시보드는 `wiki/todo.md`(생성) 다. 여기서 체크한 항목은
> 해당 태스크 파일의 `status` 도 함께 바꾼다.
>
> 계획 근거: `wiki/outputs/output-streaming-vap-research-plan.md`
> 마지막 갱신: 2026-09-03 (Phase 0 환경 구축 완료)

---

## 매 세션 루틴

- [ ] 시작: `./scripts/sync-rack4.sh status` — 접속·GPU 점유·`/data4` 여유 확인
- [ ] 종료: `./scripts/sync-rack4.sh push` — 로컬이 정본, 서버는 미러
- [ ] 위키를 바꿨으면 `wiki/log/` 샤드 추가

---

## Phase 0 — 검증과 환경 (9월) ⚠ 나머지 전부를 막고 있음

### 환경
- [x] **컨테이너 학습 환경 구축** — torch/NeMo/transformers, 4 GPU 인식, `requirements.txt` 커밋 ✓ 2026-09-03
      `wiki/tasks/task-setup-training-environment.md` · p0 · ~09-12
- [ ] **체크포인트 보존 정책** — best+last 만 유지, 특징 캐시는 `/data3`, 200G 경고
      `wiki/tasks/task-checkpoint-retention-policy.md` · p1 · ~09-19
- [ ] `.llm-wiki-local/user.yaml` 의 `member_id: tskim` 확인
- [ ] Git private 원격 생성 → 초기 커밋 → `main` 보호 설정
- [ ] `AGENTS.md` 유지보수 PR — `.env` / `sync-rack4.sh` / `TODO.md` 를 Directory Contract 에 반영

### 데이터
- [x] **AI Hub 신청·승인 및 실물 검증** — 16 kHz stereo, 누설 −64 dB, 온셋 오차 30 ms ✓ 2026-09-03 (약관 원문 확인만 남음)
      `wiki/tasks/task-verify-aihub-stereo-and-access.md` · p0 · ~09-10
- [ ] **영어 코퍼스 확보** — otoSpeech(104h), CANDOR(850h), SpokenWOZ 채널 확인
      `wiki/tasks/task-secure-english-corpora.md` · p1 · ~09-17
- [ ] 코퍼스를 `/data3/tskim/corpora/<이름>/` 에 배치, manifest 를 `/data3/tskim/manifests/` 에

### 방법론 관문
- [x] **Encoder causality·lookahead 감사** — 절단 실험, Nemotron chunk 5단계 + Qwen AuT ✓ 2026-09-03 → `wiki/outputs/output-encoder-causality-audit.md`
      `wiki/tasks/task-audit-encoder-causality-lookahead.md` · p0 · ~09-17
      ➜ **관문 발동**: Qwen AuT 평균 420 ms → `decision-asr-backbone` 수정 (Paper 1 = Nemotron 단일, 80/160 ms chunk)
- [ ] **Paper 1 범위 확정** — 주장 한 문장, 목표 학회·마감, 필수 실험 목록, 관련 연구 초안
      `wiki/tasks/task-paper1-scoping.md` · p1 · ~09-24

---

## Phase 1 — Stage 1: Representation 비교 (10월) → Paper 1 핵심 결과

- [x] **VAP target 파이프라인** — `vapasr/data/`, 코퍼스 3종 156 h → 55,139 창 ✓ 2026-09-03
      `wiki/tasks/task-build-vap-target-pipeline.md` · p0 · ~10-01
- [x] **VAP baseline 재현** — oto 체크포인트 dev 0.841/0.045/463 ms 동봉 예측과 완전 일치 ✓ 2026-09-03 (리더보드 수치는 test split)
      `wiki/tasks/task-reproduce-vap-turnbench-baseline.md` · p0 · ~10-01
- [x] **컴퓨트 예산 + 특징 캐시** — 7 인코더 × 205 h = 447 GB, RTF·peak 실측 ✓ 2026-09-04
      `wiki/tasks/task-compute-budget-and-feature-cache.md` · p1 · ~10-08
- [ ] **Stage 1 encoder probing** — CPC / WavLM B·L / FastConformer / AuT / fbank floor · **학습기 완성 2026-09-04**, 캐시 완료 후 매트릭스 실행
      공통 frame rate + probe 용량 고정 + lookahead 명시. 50–100 h 로 시작
      `wiki/tasks/task-stage1-encoder-probing.md` · p0 · ~10-22
      ➜ **관문**: H1 지지 / 기각 / 불확실 을 명시적으로 결론. 기각이면 Paper 2 보류
- [ ] **누락 baseline** — VAD+threshold, cascade(실측), DualTurn, JAL-Turn
      `wiki/tasks/task-add-missing-baselines.md` · p1 · ~10-29
- [ ] **Latency-quality 곡선** — chunk 80/160/320/560/1120 ms, lookahead 합산, 한국어 CER 동반
      `wiki/tasks/task-latency-quality-curve.md` · p1 · ~10-29

---

## USLM 트랙 — 단일 주력 (2026-09-04 확정). U0 는 여기, U1–U3 은 Phase 2, U4–U5 는 Paper 2

- [ ] **U0 토큰율 + 정렬** — Qwen3 tokenizer 프레임당 토큰 분포(KO/EN), ForcedAligner 정렬·QC
      `wiki/tasks/task-uslm-feasibility-u0.md` · p1 · ~09-18
- [ ] U4 adaptive emission (WER–delay RL) · U5 long-context·배포 — Paper 2
      계획·평가: `wiki/outputs/output-unified-slm-architecture-plan.md`, 결정: `decision-target-architecture`

## Phase 2 — IS-SLM 본체 (10~11월) ← 이중 프레임율/RNN-T 안은 2026-09-04 기각

- [x] **U0.5 adapter bridge test** — 4 run 완료, 최선 18.2–18.8 / 17.5 / 14.5–15.1 %. 관문 재정의(RNN-T `[56,0]` 우수 + 오프라인 ≤ +50 %) 후 **통과** (09-04) → `output-uslm-u05-adapter-bridge`
      `wiki/tasks/task-uslm-u05-adapter-bridge.md` · p0 · ~09-25
- [~] **U1 aligned interleaved ASR** — Nemotron [56,0] → adapter → Qwen3-ASR thinker LoRA, 텍스트 스트림만. **관문 WER ≤ +10 % vs RNN-T `[56,0]`**. 09-04 착수: 데이터/모델/학습·스트리밍 평가 코드 완성, 스모크 → v0 run
      `wiki/tasks/task-uslm-u1-interleaved-asr.md` · p0 · ~10-16
- [ ] **U2 self-conditioned streaming** — gold/self 격차, corruption 학습
      `wiki/tasks/task-uslm-u2-self-conditioned.md` · p1 · ~10-30
- [ ] **U3 conversational multi-task** — 헤드 + 토큰, 50 Hz 하이브리드 ablation, encoder-only probe 대 비교(H2)
      `wiki/tasks/task-uslm-u3-multitask.md` · p0 · ~11-27
- [x] ~~Stage 2 multitask VAP + WER 가드레일~~ · ~~Stage 3 linguistic state fusion~~ — A 안 기각으로 폐기, U3 로 병합

## Phase 3 — Objective (12월)

- [ ] **Time-to-next-turn hazard head** — discrete-time hazard, censoring 처리, −600/−300/0 ms P(EOT) 곡선
      `wiki/tasks/task-time-to-next-turn-survival-head.md` · p1 · ~12-10
- [ ] **이벤트 라벨 휴리스틱 검증** — otoSpeech 사람 라벨 대비 F1 ≥ 0.8, BACKCHANNEL↔INT 혼동행렬
      `wiki/tasks/task-event-label-heuristics-validation.md` · p2 · ~12-10

---

## Phase 4 — 한국어 (12월 ~ 1월)

- [ ] **한국어 벤치마크 설계** — IPU 경계 3–5k 지점, 어미별 층화, 3인 + Fleiss κ, 오디오 없는 배포 형식
      `wiki/tasks/task-korean-benchmark-design.md` · p1 · ~12-24
      ➜ 배포 범위는 `decision-korean-benchmark-release-scope` (어노테이션 레이어만)
- [ ] **한국어 turn 단서 문헌 조사** — CA/TRP, 어미–floor 이양, clause-final prosody
      `wiki/tasks/task-korean-turn-cue-literature-review.md` · p2 · ~01-14
- [ ] 어노테이션 수행 (설계 완료 후 태스크 생성)

---

## Phase 5 — Bilingual / robustness (2월) — H1 지지 시에만

- [x] ~~Qwen AuT causal 적응 연구~~ — 최종 backbone에서 AuT를 사용하지 않아 취소
      `wiki/tasks/task-qwen-aut-causal-adaptation.md` · done · 09-04
- [ ] **Bilingual 학습 + 강건성 평가** — temperature sampling, mono vs bi 3조건, 실내/실외·SNR 평가, ForcedAligner 한국어 확인
      `wiki/tasks/task-bilingual-and-qwen-port.md` · p2 · ~02-11

---

## 논문

### Paper 1 — "Do streaming ASR representations project conversational futures?"
- [ ] Stage 1 결과표 (encoder × 조건, lookahead 명시)
- [ ] Latency-recall 곡선 (사람 −151 ms / VAP 368 ms 기준선 표기)
- [ ] Fusion 이득 + WER 회귀표
- [ ] 한국어/영어 대조 분석
- [ ] 초안 → 내부 리뷰 → 제출

### Paper 2 — Unified streaming perception model (Paper 1 결과에 조건부)
- [ ] Qwen3-ASR 통합 모델 + hazard/event head
- [ ] Muse-style adaptive emission (WAIT/EMIT), delay-aware RL
- [ ] Muse API black-box 비교

---

## 결정 관문 요약

| 관문 | 조건 | 결과 |
|------|------|------|
| Lookahead | 어느 backbone 이든 > 320 ms | **발동(Qwen)** — Paper 1 은 Nemotron 단일. Qwen 은 적응 연구 후 재판단 |
| H1 | Stage 1 에서 기각 | IS-SLM turn 기대치 하향; 정당성은 H2 + 시스템 이점에서. 50 Hz 하이브리드 필수 |
| AI Hub | ~~실물이 stereo 가 아니거나~~ **stereo 확인됨.** 약관상 어노테이션 공개 불가 시 | 한국어 기여를 내부 평가로 격하 |
| `/data4` | 여유 < 200 G | 체크포인트 정리 또는 저장 경로 재결정 |

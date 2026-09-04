---
type: status
status: active
created: 2026-09-03
updated: 2026-09-04
summary: 현재 유지보수 상태, 다음 액션, 린트 로테이션 담당
---

# 상태

마지막 갱신: 2026-09-04

## 볼트 상태

- 스키마 골격 초기화 완료.
- 첫 ingest 완료 — [[source-chatgpt-research-plan]] 및 검증으로 확보한 외부 자료 6건.
- 실험 환경 구성 완료 — rack4 / `tskim_env` 연결, `.env` 단일 관리, 첫 동기화 성공.
- **최종 backbone 확정** — Nemotron [56,0] → adapter → Qwen3-ASR-0.6B-hf thinker (사용자, 2026-09-04). 관문 U0.5 adapter bridge test 신설.
- **목표 구조 확정 — IS-SLM 단일 주력.** 이중 프레임율+RNN-T 안은 기각(사용자, 2026-09-04). Phase 2 = U1–U3. Stage 2/3 태스크 폐기, U1/U2/U3 태스크 신설, 이중 프레임율(A) 은 Paper 1 주 모델·대조군. → [[decision-target-architecture]], [[output-unified-slm-architecture-plan]]
- 특징 캐시 완료(7 인코더 × 205 h, 447 GB) → **Stage 1 매트릭스 실행 중** (2026-09-04 11:49). 첫 CPC probe: dev EOT 0.899 @ FP 0.058 / 456 ms. → [[output-feature-cache-and-compute-budget]]
- **Phase 0 완료.** target 파이프라인 — 코퍼스 3종 156 h, 20 s 창 55,139개. → [[output-vap-target-pipeline]]
- VAP baseline 재현 완료 — dev EOT 0.841/0.045/463 ms, oto 체크포인트 동봉 예측과 동일. → [[output-vap-turnbench-baseline-reproduction]]
- AI Hub 실물 검증 완료 — 16 kHz 진짜 분리 stereo, 라벨 온셋 오차 30 ms. VS_02 186 wav + 라벨 전체 서버 보관.
- causality 감사 완료 — Nemotron 80 ms chunk 통과, Qwen AuT 조건 위반 → [[decision-asr-backbone]] 수정.
- 학습 스택 구축 완료 — conda `vapasr`, torch 2.6 cu124, NeMo git, qwen-asr, 원 VAP. 스모크 4종 통과.
  → [[decision-compute-environment]]
- 위키 페이지 34개 (source 7, concept 7, question 6, decision 4, output 9, task 25).
- 파생 파일 재생성 완료.

## 다음 액션

- **U0.5 통과(관문 재정의, 09-04)** → U1 착수. 보고서 [[output-uslm-u05-adapter-bridge]]. U1 v0 run 결과로 스트리밍 방출 열화 판정.

### 연구 (Phase 0 — 나머지 전부를 막고 있음)


전체 목록 → [[todo]]

### 볼트 운영

- [ ] `.llm-wiki-local/user.yaml` 의 `member_id: tskim` 확인 —
      에이전트가 git config 기반으로 생성했으므로 사용자 확인 필요
- [x] 원격 `origin` = github.com:tskim9439/VAP — **회사망이 22 번 포트를 차단**하므로 remote URL 을 `ssh://git@ssh.github.com:443/...` 로 설정(2026-09-04). 첫 커밋 598a8bd → `main` 푸시 완료. `gh` 는 KT 엔터프라이즈 토큰만 허용되어 API 사용 불가(SSH 는 됨)
- [ ] `main` 브랜치 보호 설정(GitHub 웹에서)
- [ ] 대용량 바이너리 정책 결정 — 오디오 데이터를 볼트에 넣지 않는다는 방침 명문화 필요
- [ ] `AGENTS.md` Directory Contract 에 `.env` / `sync-rack4.sh` 반영 (유지보수 PR)

## 미해결 이슈

| 이슈 | 영향 | 추적 |
|------|------|------|
| AI Hub 재배포 제약 | 한국어 벤치마크 공개 범위 | [[question-korean-corpus-licensing]] |
| Qwen AuT 비인과·블록 lookahead 420ms | 비교 결과만 보존; 최종 backbone에서 제외 | [[task-qwen-aut-causal-adaptation]] (취소) |
| DualTurn 이 VAP 를 크게 앞섬 | H1 의 경쟁 가설 | [[question-asr-representation-vs-ssl-for-vap]] |
| 이벤트 라벨 정확도 미검증 | auxiliary head 신뢰도 | [[question-event-label-derivation-validity]] |
| 한국어 turn 단서 근거 부재 | 논문 서술 | [[question-korean-turn-cue-literature]] |
| SpokenWOZ 채널 구조 불명 | 영어 데이터 규모 | [[question-spokenwoz-channel-structure]] |
| **`/data4` 97% 사용, 575G 여유** | 체크포인트 저장 공간 | [[decision-compute-environment]] |

## 스키마 이슈 (유지보수 PR 필요)

`AGENTS.md` 에서 발견한 불일치. 스키마 변경은 리뷰가 필요해 직접 고치지 않았다.

- 280–281행: 런타임 경로가 `.codex/skills/` 로 **중복** 표기. `.claude/skills/` 와
  `.codex/skills/` 를 의도한 것으로 보고 구현했다.
- 85–89행: Agent Skills 는 정본을 `.codex/skills/` 로, Directory Contract 는
  `.skills/` 로 적었다. 후자를 따랐다.
- 12–15행: "codex Code" 가 "Claude Code" 오타로 보인다.
- 280행은 "five workflow skills" 인데 Agent Skills 절에는 4개만 나열되어 있다.
  Team Execution Layer 가 언급한 `wiki-task` 를 다섯 번째로 넣었다.
- 275–277행: `scripts/` 를 4개 스크립트 "only" 로 열거하는데,
  실험 서버 동기화용 `sync-rack4.sh` 를 추가했다. Directory Contract 에
  `.env` / `.rsyncignore` / `sync-rack4.sh` 를 반영하는 유지보수 PR 이 필요하다.

## 린트 로테이션

주 1회 Lint Workflow 를 수행한다.

| 주차 | 담당 | 수행일 | 리포트 |
|------|------|--------|--------|
| 2026-W37 | tskim | 예정 | — |

## 팀 멤버

| member_id | 역할 | 비고 |
|-----------|------|------|
| tskim | researcher | 로컬 신원 확인 필요 |

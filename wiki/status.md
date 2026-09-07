---
type: status
status: active
created: 2026-09-03
updated: 2026-09-07
summary: 현재 유지보수 상태, 다음 액션, 린트 로테이션 담당
---

# 상태

마지막 갱신: 2026-09-07

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
- **ASR 출력 표기 실측·정규화 계층 도입** — Qwen3-ASR과 Nemotron 모두 EN 숫자는
  단어, KO 숫자는 한글 읽기로 출력. `vapasr/data/textnorm.py`로 타깃·채점 규약을
  통합하고 KsponSpeech 숫자 이중표기는 발음형을 선택한다. → [[source-asr-output-style-probe]],
  [[asr-text-normalization]]
- **lexical `asr-tn-v1.0.0` 규약·구현 동결** — LibriSpeech·KsponSpeech의 EN/KO
  target·score, 지원·미지원 숫자 문법, quarantine, golden/tokenizer test, manifest
  fingerprint를 고정했다. 전체 transcript audit에서 LibriSpeech 잔존 digit 0,
  KsponSpeech train 620k 중 quarantine 68건을 확인했다. 목적 표본 사람 검토는 남았고,
  영어 display 학습은 lexical 관문 뒤로 연기했다.
  → [[output-asr-tn-v1-spec]], [[source-asr-tn-v1-audit]]
- **Stage 1 mono overfit 기능 관문 통과** — @900부터 EN·KO WER/CER 0,
  `viol80=0`, 방출률=참조율이며 @1500까지 유지. 경합 전 decoder-only tick p99는
  151–186 ms로 실시간성 관문은 별도 미통과. → [[output-stage1-mono-pilot]]
- **Stage 1 mono 개선 run A/B 완료 — B(full FT) 채택.** B는 A LoRA r16보다
  dev-clean/dev-other/kspon-dev 오류를 상대 20.3%/22.0%/7.6% 줄였다. KO 과소 방출은
  해소됐지만 `viol80=4.85%`, tick p99 105–145 ms와 RNN-T 대비 2.2–3.8배 오류가 남았다.
  EN은 @4000 이후 정체해 같은 데이터의 30 epoch 반복보다 Stage 2 데이터 확장과
  windowed alignment·latency loss가 우선이다. → [[source-stage1-mono-run-ab]],
  [[output-stage1-mono-pilot]]
- **1,930 h lexical ASR 30 epoch 학습 진행 중**(사용자, 2026-09-07). 현재 recipe와
  `asr-tn-v1.0.0`은 기준선으로 유지하고, 다음 run용 데이터 확장은 NIKL 500 h와
  Switchboard·otoSpeech·AMI IHM 묶음을 먼저 준비한다.
  → [[output-stage1-asr-data-expansion-priority]]
- 위키 페이지 수와 파생 파일은 현재 작업 브랜치의 병합 전 절차에서 다시 생성·집계한다.

## 다음 액션

- **Stage 2 scaling curve** — B(full FT)를 초기값으로 200/500/1,000/1,900 h에서
  full-dev를 평가한다. 500 h에서 RNN-T 오류비 2배 이하, 1,000–1,900 h에서 1.5배
  이하로 줄지 않으면 hard-δ 목표를 재설계한다. → [[output-stage1-mono-pilot]]
- **정렬·방출 목표 ablation** — 단일 forced-alignment 시점 대신 허용 window/soft
  target, latency loss, streaming+offline 공동 학습을 비교하고 KO next_weight는
  0.15/0.175/0.2/0.25 Pareto sweep으로 본다. → [[output-stage1-mono-pilot]]
- **decoder tick 최적화** — 현재 수치는 encoder·adapter·flush를 제외한 하한이며 한 chunk가
  최대 `M+2` thinker forward를 호출한다. `<NEXT_AUDIO>`와 다음 audio를 한 2-token forward로
  합친 뒤 end-to-end service time을 다시 잰다. → [[output-stage1-mono-pilot]]
- **다음 ASR DB 준비** — NIKL 500 h, Switchboard train, otoSpeech train, AMI IHM을
  Pack X1으로 준비한다. 모든 DB에 speaker/session split, overlap filter, audio dedup,
  corpus parser와 license metadata 관문을 적용한다. MNSC는 local 4,035 h와 official
  2,000 h 차이를 해소한 뒤 500 h cap으로 시작한다.
  → [[output-stage1-asr-data-expansion-priority]]
- **TN v1 후속** — lexical `asr-tn-v1.0.0`은 현재 1,930 h 기준선에 고정한다. 새 corpus
  parser는 `asr-tn-v1.1.0`으로 분리하고 기존 LibriSpeech·KsponSpeech target을 바꾸지
  않는다. 영어 display 계약과 학습은 lexical 관문 뒤로 연기한다.
  → [[output-asr-tn-v1-spec]], [[source-asr-tn-v1-audit]]

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
| KsponSpeech TN 목적 표본 5×40행 사람 검토 미완료 | current lexical run의 사후 감사; 오류 발견 시 기존 산출물을 덮지 않고 새 TN version으로 수정 | [[source-asr-tn-v1-audit]] |
| 영어 미지원 숫자의 문맥별 읽기 | 연도·전화·분수·단위 등을 부분 변환하면 라벨 왜곡 | [[output-asr-tn-v1-spec]] |
| B mono decoder tick p99 105–145 ms | 80 ms 실시간 deadline 미충족 | [[output-stage1-mono-pilot]] |
| B full FT도 RNN-T 대비 오류 2.2–3.8배 | 데이터 규모와 단일시점 정렬 목표의 일반화 한계를 분리해야 함 | [[source-stage1-mono-run-ab]] |
| KO `viol80=4.85%`, 지연 p99 797 ms | 과소 방출 해소와 맞바꾼 조기 방출 꼬리 | [[source-stage1-mono-run-ab]] |

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

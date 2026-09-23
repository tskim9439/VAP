---
type: status
status: active
created: 2026-09-03
updated: 2026-09-22
summary: 승인 2356시간 학습 중이며, 별도 speechlm Arrow에서 중복을 뺀 대화형 ASR P0 후보와 재선별 관문을 확정했다.
---

# 상태

마지막 갱신: 2026-09-22 (아래 기존 볼트 상태는 2026-09-07의 기록)

## 2026-09-22 speechlm Arrow ASR 확장 조사

- `/soundai/DB/speechlm/speechlm_v1_encoder_alignment_2607`와 대응 `DI_LAB_CATALOG` tar를 읽기 전용으로 조사했다. 현재 승인셋과 동일한 AI Hub 134-1/2, KsponSpeech, LibriSpeech, AMI, Switchboard는 전체 제외하고 YODAS는 `en129`만 제외한다.
- 과거 원음이 없던 것으로 기록된 AI Hub 자유대화·전화·인터뷰·상담·고객응대·회의의 8 kHz mono tar가 실제로 존재한다. People's Speech clean, Earnings22, MNSC PART6와 함께 P0 probe 후보로 정했다.
- 해당 Arrow에는 현 학습용 word/token alignment가 없고 원전사도 Qwen-verbatim 규약이 아니다. Arrow 참조 member만 resolve한 뒤 기존 Qwen–Whisper 5% 합의, PCM 중복, forced alignment QC를 다시 통과시킨다. `000035` tar 폴더의 비참조 Common Voice 조각 때문에 디렉터리 glob은 금지한다. → [[output-speechlm-arrow-asr-review]]

## 2026-09-21 데이터 선별

- v1 전량3,757,805건 완료. 유지1,260,807건·약1,637시간, 보류2,496,998건.918개 교사 작업 모두 정상 종료, 추론 오류·OOM 분할0.9월21일02:14 KST 완료, GPU 해제 확인.
- 사용자 지시로 **Qwen–Whisper 정규화 편집거리 비율 ≤5%**까지 유지한다. 원전사 일치는 필수 조건에서 제외하고 차이 수치만 보존한다. 기존 안전 보류는 유지한다.
- 별도 `pair-v0.6-full-20260921`에서 CPU worker8개·GPU0장으로 **375만 건 재선별 완료**. 유지1,770,981건·2,359.703시간, 추가510,174건·722.486시간. 추가분은 `pair_only`로 구분하고 원전사·기존 산출물은 보존한다. Qwen 원출력 타깃 후보·TN 비교 전용 규약 유지. 학습/화자/EOT 승인 아님. → [[output-data-selection-automatic-framework]].

## 2026-09-20 데이터 선별 후속

- **원출력 보존·처리량 확대:** 사용자 지시로 TN을 저장에 강제하지 않는다. 원전사/Qwen/Whisper 반환 문자열을 따로 보존하고 비교에만 TN을 적용한다. 자동 통과633건은 Qwen 원출력을 학습 타깃 후보로 제공하되 교사 pseudo-label임을 명시한다. 1792건 후처리 재생성 완료, GPU 재추론 없음. 전량 준비 경로 `auto-v0.5-full-20260920`에 Qwen512/Whisper256·decode32·prefetch·최대4 GPU 큐를 연결했다. 준비 완료 후 자동 시작하며 아직 전량 추론 완료는 아니다. → [[output-data-selection-automatic-framework]].

- **첫 자동 검증 완료:** 1792건×두 교사 정상 추론, ASR silver633 / 불일치 보류1154 / 빈 전사 보류5. 신규 테스트11개 통과. 균등 DB 표본의 유지율을 전체 모집단에 외삽하지 않는다.

- **운영 방식 전환:** 개별 사례 재청취 요청을 중단했다. 자동 유지 후보·명백한 불능 쌍 제외·원인별 HOLD 정책, 지문 고정·재개 가능한 최대 4 GPU 큐를 구현했다. HOLD 전체를 사람에게 넘기지 않는다. 전량 4,264,749건 QC 재판정: READY 3,757,805 / HOLD 505,884 / ASR 불능 쌍 1,060. 14개 DB ×128건=1,792건을 자동 교사 판정에 연결했다. 원음·전사·TN·기존 학습 입력은 유지한다. → [[output-data-selection-automatic-framework]].

- **00:37 KST 후속 청취문 수신:** 진단 8개 key 일치, 청취문 4개/빈 값 4개, resolution 전부 unreviewed. `네 네`, `great`, `And i understand that too`, `let's keep this perspective`를 원형 보존했다. AMI 청취 파일·구간과 VoxPopuli `in` 생략 의도 확인이 남아 최종 승인·타깃 변경 없이 기존 실패 11개 보류를 유지한다. → [[source-data-selection-human-diagnostic-v04]].

- 방송 `네 네 네`는 원 발음전사부터 존재했고 동일 TN으로 재현됐다. 라벨 3개·오디오 2개 사본 일치. AMI A→Headset-0와 234.102–234.834초 crop은 원주석·파형과 일치해 증폭/문맥 재청취를 준비했다. VoxPopuli 실패 6개 모두 단일 VAD 구간이며 알려진 저장소에서 원 세션 미발견. → [[output-data-selection-diagnostic-v04]].
- 사람 검수 32개를 판정기에 연결해 886개 재판정 완료: 실패 11개 전부 보류, 나머지 21개 REVIEW. 원문·타깃·파형 변경 시 검수 재사용 거부. 보호는 검수 인자를 전달한 실행에 적용되며 기존 학습 입력은 유지했다.
- 전환 전 기록: 회귀 테스트 32개 통과, 후속 청취 8개 사례·13개 WAV hash 검증. 당시 GPU 0장. **다음 단계는 추가 청취가 아니라 자동 배치의 운영 검증과 같은 recipe의 전량 처리다.** 개별 불확실성은 HOLD로 보존하고 전체 진행을 막지 않는다.

## 2026-09-19 데이터 선별 후속

- **23:28 KST 추가 검수 수신:** 32개 key·source 일치, 전사 pass/fail 21/11·경계 27/5·채널/화자 29/3, 턴 전부 미검수. VoxPopuli 길이 일치·AMI 동일 채널이 실제 전사 품질을 보장하지 않았다. 방송 `raw_text=네.` / `text=네 네 네` 불일치와 AMI 가청성, 경계 의심을 우선 진단한다. 표준어·숫자·filler 제거 선호는 기록하되 TN/전사는 자동 변경하지 않았다. 전량 screening·새 GPU 작업 미시작. → [[source-data-selection-human-review-v03]], [[output-data-selection-repair-v03]].

- **14:12 KST 후속 완료:** NIKL 두 세션 681개 검사, 기존 후보 625개를 새 WAV·포맷 예외표로 복구. Qwen/v3 각각 625개 정상, 합의 후보 247·일반 검수 345·교정 검수 32·기존 사람 쟁점 보류 1. VoxPopuli 177,422개 모두 공식 VAD 연결 길이와 0 sample 오차이며 1,054개는 비연속 연결이다. 방송·AMI 다채널 3,657개 중 동일 718·상이 2,939. GPU 작업 종료, 추가 청취 32개 준비. 전량 선별·학습은 아직 미시작. → [[output-data-selection-repair-v03]].

- 13:02 KST 사람 검수 JSON 반영: 48 kHz 진단 16개 key 일치. 전사 pass 15/fail 1, 경계·화자 pass 16, 턴 pass 5/uncertain 2/unreviewed 9. 실패 1개는 발음 충실도와 의미 보정의 기준 쟁점으로 원라벨을 유지하고 타깃 변경을 보류한다. 이 검수는 두 세션의 제한적 포맷 복구 근거이며 전체 NIKL/EOT/학습 자동 승인은 아니다. → [[source-data-selection-human-review-20260919]], [[output-data-selection-review-v02]] §4.

- CPU QC는 02:09 KST 완료: 14개 DB 4,264,749행, 음향 통과/전사 대기 3,812,311 · 검수 451,073 · 오류 1,365. MNSC 보류 유지.
- 완료 결과 SHA256·후보 행 대응을 재검증했다. NIKL 2024/25 약 +0.4초, VoxPopuli 약 −0.2초의 체계적 차이는 기존 비율 경고를 통과한 행에도 있다. 시간축 정확성과 ASR 기술 QC를 구분해야 한다.
- 886개 층화 표본의 청취 페이지·Qwen/v2/v3 대조 완료(각 정상 775 / 오디오 오류 111). 오디오 825개 링크와 기본 파형 hash 775개를 로컬에서 검증했다. NIKL 2022 두 세션 16개의 48 kHz 가설 진단에서 Qwen CER 93.59→14.74%로 내려 포맷 오해석 근거가 강해졌다. 모든 진단 GPU 작업 종료. 전량 teacher screening과 학습 승인은 아직 아니다. 현황·경로·사람 검수 관문: [[output-data-selection-review-v02]].

## 2026-09-18 데이터 선별 (당시 기록)

- [[source-data-selection-plan]]을 분석하고 [[output-data-selection-execution-plan]]으로 구체화했다. 원문은 지정된 `raw/inbox/DataSelectionPlan.md`에 보존한다.
- 기존 15개 DB의 4,634,601 발화 metadata 감사와 806개 표본의 Qwen/Whisper-v2/v3 점검을 완료했다. GPU는 mxc에서 동시에 최대 2장만 사용했고 현재 교사 GPU 작업은 종료됐다.
- MNSC는 CSV ID/전사 충돌과 대량 원음 경로 결손 때문에 source hold. AI Hub 방송의 길이 계산 가정 오류, NIKL/VoxPopuli의 crop 길이 불일치를 발견해 전수 오디오 검사를 진행 중이다. 23:27 기준 CPU **8→32 worker**로 확대했고 완료 DB 8개 재사용, 134-1의 저장된 44,282행을 복구해 56,570행까지 진행했다. 전체 품질 선별 완료는 아니다.
- 최신 GPU 허용량은 동시 **최대 4장**이며 현재 CPU QC는 GPU 0장이다. 현재 PID `4129725`, 로그 `/soundai/users/tskim/VAPKT-data/data-selection-v01-audio-w32.log`; 기존 로그는 중지됐다. 산출물 재사용 방식과 처리율은 [[output-data-selection-execution-plan]] §8.5 참조.
- 교사 합의는 검수 후보이지 GOLD가 아니다. overlap·맞장구·무음을 일괄 제거하지 않으며 전사·시각·화자·턴 품질을 분리한다.
- 다음 관문: 실제 원음/시간축 대응 복구 → 사람 청취로 규칙 보정 → 전량 teacher screening → 시각/화자 QC → Phase 1/2 공통 serializer 연결. 기존 학습 데이터는 아직 교체하지 않는다.

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

- **speechlm 2607 단일 화자 manifest** — 26개 카탈로그, 47,662,558행,
  108,512.79시간의 압축 Arrow 생성을 완료했다. 실제 Arrow 행 수 재계수도 summary와
  일치했고 tar/duration 누락은 0이다. 다음은 기존 데이터 중복 제거, source별 cap,
  Qwen·Whisper 합의 QC다. → [[output-speechlm-single-speaker-manifest]]

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

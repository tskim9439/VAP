<!-- generated: do not edit -->
# 활동 로그

마지막 생성: 2026-09-04

`wiki/log/` 의 샤드를 최신순으로 이어붙인 다이제스트다. 직접 편집하지 않는다.

## [2026-09-04] task | U0.5 adapter bridge 실험 완료(4 run), U0 토큰율·M 예산·정렬 진행

- Changed: `task-uslm-u05-adapter-bridge`(4 run 결과·종합 판정·관문 재검토 제안), `task-uslm-feasibility-u0`, `README.md`(§4 U0/U0.5 상태, p0 행),
  `wiki/status.md`(Git 원격 SSH 443). 코드: `experiments/u05_{distill_adapter,baselines,asr_finetune}.py`, `u0_{token_rate,align,interleave_stats}.py`,
  `vapasr/uslm/{data,model}.py`, `vapasr/data/interleave.py`. 서버 산출물 `/data4/tskim/VAPASR/experiments/uslm/{u05-asr-*,adapter-distill-*}`, 정렬 `/data3/tskim/manifests/align/`.
- Reason: 최종 backbone(Nemotron [56,0] → adapter → Qwen3-ASR thinker) 연결 품질 조기 검증. 기준선 오프라인 Qwen 13.9/13.5/13.3 %,
  Nemotron RNN-T [56,0] 25.4/23.4/24.5 %, [56,13] 19.0/16.9/17.0 %. 증류(block8s 교사, cos 0.777) init 과 random init, lr 2e-4/1e-4, 6k/12k step
  4 조합 → 모두 oto 18–20 / 실내 17–19 / 실외 15–17 % 수렴(최선 증류 lr 1e-4: 6k 18.2/19.2/16.1, 12k@8000 18.8/17.5/14.5). 증류 이득은 잡음 범위.
  관문 원안(오프라인 ×1.15)은 실외만 통과하나 동일 인코더의 RNN-T 대비 −6 pt → adapter 병목 아님, 격차는 인과 인코더 상한으로 해석.
  U0: KO 토큰율 p99 0.78 tok/80 ms(폭주 없음), chunk M=4(KO 이월 2.6 %), 생성기 완료, 정렬 aihub 완료·oto 진행. 운영: 한 run 25 GB → 순차 체인(pid 대기),
  GitHub 는 회사망 22 번 차단으로 SSH 443 경유, 첫 커밋 598a8bd 푸시.
- Next: (사용자 결정) U0.5 관문 재정의 vs Nemotron 상위 블록 unfreeze 1 회. 정렬 완료 후 `u0_interleave_stats` 전체 재실행·M 확정 → U1(interleaved ASR) 착수.
  Stage 1 잔여(seed 반복, EN-only, DualTurn).
- By: tskim

## [2026-09-04] task | Stage 1 probe 학습기 작성 (encoder probing 착수)

- Changed: `vapasr/probe/{__init__,data,model}.py`(신규), `experiments/{train_probe.py,run_stage1.sh,show_probe_results.py}`(신규),
  `vapasr/features/encoders.py`(fbank floor 인코더, device/`to()`, module 참조), `experiments/extract_features.py`(--seg-s, --ids, --device,
  대화별 `empty_cache`), `experiments/run_feature_cache.sh`(turnbench-dev 는 --include-flagged), `task-stage1-encoder-probing` → doing,
  `TODO.md`·`todo.md`.
- Reason: 사용자 요청. 캐시된 frozen 특징 위에 고정 용량 causal probe head(VAP 256 + VAD)를 학습하고 TurnBench dev 를 공식 sweep/scorer 로
  자동 채점하는 파이프라인. 스모크(CPC, 300 step) 통과, 1 epoch ≈ 3 분. 발견·수정: (1) TurnBench scorer 는 dev 38 대화 전부를 요구 —
  플래그(한 채널 무음) 대화 tb-172 도 캐시해야 함, (2) 캐시 작업 2개가 GPU 1 메모리 34.8 GB 를 점유(캐싱 할당기) → 대화별 empty_cache
  추가, 가벼운 인코더는 `--device cpu` 로 우회, (3) 채점 출력을 grep 으로 가리다 coverage 오류를 놓쳤음 → 파일 저장 후 요약.
  첫 CPC 1-epoch 결과(dev, fp≤0.1 sweep): EOT R 0.899 / FP 0.091 / p50 445 ms, INT 0.943 / 0.099 / 893 — VAP 와 같은 FP(0.045)에서의
  비교는 채점 재실행 후 기록.
- Next: 캐시 완료(feat-cache-A/B) → `run_stage1.sh` 매트릭스(인코더 × 고유율/12.5 Hz) → [[task-stage1-encoder-probing]] 판정.
  DualTurn encoder 확보 검토. AI Hub(한국어) 평가는 VAP 지표(val CE/acc)만 — 한국어 이벤트 gold 는 벤치마크 태스크에서.
- By: tskim

## [2026-09-04] task | IS-SLM 최종 backbone 확정: Nemotron [56,0] → adapter → Qwen3-ASR thinker

- Changed: `decision-asr-backbone` → accepted(최종 고정 조합, U0.5 관문), 신설 `task-uslm-u05-adapter-bridge`(p0, ~09-25), `task-uslm-u1-interleaved-asr` backbone 문구,
  `task-qwen-aut-causal-adaptation` 취소, bilingual 태스크에서 Qwen AuT 포팅 제거, `README.md`, `TODO.md`, `wiki/overview.md`, `wiki/status.md`, 관련 설계 outputs.
- Reason: 사용자가 "Nemotron 3.5 FastConformer [56,0] → new adapter → Qwen3-ASR-0.6B-hf LM" 조합을 최종 선택. encoder는 실측 causal ≤80 ms·
  잡음 강건·ko-KR이고 thinker는 Qwen audio embedding에서 text를 생성하도록 사전학습됐다. 분포 간극은 기존 Qwen audio tower의 thinker 입력 embedding을
  교사로 삼은 표현 증류와 짧은 ASR 미세조정으로 조기 검증한다. 실패 시 adapter 학습을 재설계하며 backbone은 자동 교체하지 않는다.
- Next: U0(토큰율·정렬)과 U0.5(adapter bridge)를 병행 착수. Stage 1 결과 페이지는 cc1s 재채점 후.
- By: tskim

## [2026-09-04] task | 특징 캐시 완료(447 GB), Stage 1 매트릭스 시작

- Changed: `task-compute-budget-and-feature-cache` → done, `output-feature-cache-and-compute-budget` 최종 표(stable), `experiments/show_cache_stats.py`(신규),
  `todo.md`·`TODO.md`·`status.md`. 서버: `/data3/tskim/features/` 7 인코더 × 816 대화 완비(fbank 추가, tb-172 보충), `stage1-matrix` bg 작업 시작.
- Reason: 캐시 작업 3건(A/B/C) 완료. 실측 RTF·peak·용량으로 표를 갱신했다. 사용자 질문("캐시를 돌리는 이유")에 답함 — Stage 1 은 인코더를
  freeze 한 공정 비교(H1)이며 인코더 출력은 고정값이라 한 번(≈17–26 GPU h)만 계산하면 probe 매트릭스(조건당 ≈3 분)를 수십 번 돌릴 수 있다.
  `run_stage1.sh fbank cpc nemotron-c0 qwen-aut-causal qwen-aut-cc1s wavlm-base wavlm-large` (각 고유율 + 공통 12.5 Hz, 4 epoch) 시작.
- Next: 매트릭스 결과 → 고정 FP 격자(0.045/0.06/0.08/0.10) recall 표 → H1 판정 초안. 학습 데이터 EN-only 조건 추가 검토.
- By: tskim

## [2026-09-04] query | 통합 SLM 구조 — 합산(v0) 대 Interleaved(IS-SLM) 검토, IS-SLM 채택

- Changed: `output-unified-slm-architecture-plan`(신규, 합산 v0 + 12항목 평가), `output-interleaved-streaming-slm-architecture`(외부 보고서, 말미에 '검토와 통합' 절 추가),
  `decision-target-architecture`(신규 → B′ 채택으로 갱신), `task-uslm-feasibility-u0`(신규, M 예산·interleaved 생성기로 조정), `index.md`·`todo.md`·`TODO.md`·`status.md`.
- Reason: 사용자가 RNN-T 독립 전사 대신 통합 SLM 을 목표로 제시(합산 융합 아이디어) → v0 계획·평가 작성. 이어 사용자가 IS-SLM 보고서를
  제시하며 더 적합하다고 판단 → 대조 검토. 동의: 합산은 KV 가 주는 정보와 중복이라 gated residual ablation 으로 격하, `<NEXT_AUDIO>` 가변
  방출이 토큰율 상한을 해소, dense 는 병렬 헤드. 보고서 보완: (1) 12.5 Hz audio-clock 위 turn 헤드는 Stage 1 실측(50 Hz 우위, INT 1.3 s)과
  충돌 → 50 Hz 사이드 브랜치 하이브리드, (2) 겹침 발화 텍스트 직렬화 규약, (3) tick 당 (2+M) forward 비용 → joint chunk token 기본,
  (4) 초기화 경로(Nemotron adapter / causal AuT + Qwen3-ASR thinker), (5) U1 WER 관문 ≤10 %.
- Next: [[task-uslm-feasibility-u0]] 착수(토큰율 → M, ForcedAligner 정렬, interleaved target 생성기). Stage 1 매트릭스 완료 후 H1 판정과 함께 하이브리드 필요성 확정.
- By: tskim

## [2026-09-04] query | 이중 프레임율+RNN-T 안 기각, IS-SLM 단일 주력 확정

- Changed: `decision-target-architecture`(확정: A 기각, B′ 단일 주력, 대조군은 외부), `output-model-architecture-proposal` → superseded,
  `output-unified-slm-architecture-plan`·`output-streaming-vap-research-plan` 갱신 표식, `task-stage2-…`·`task-stage3-…` → 폐기(done, U3 병합),
  신설 `task-uslm-u1-interleaved-asr`(p0)·`task-uslm-u2-self-conditioned`(p1)·`task-uslm-u3-multitask`(p0), `TODO.md` Phase 2 절 교체,
  `README.md` §2·§3·§5 갱신, `todo.md`·`index.md`·`status.md`.
- Reason: 사용자 결정 — "이중 프레임율 + RNN-T 는 완전 기각, IS-SLM 을 주력으로". Paper 1 = Stage 1 표현 비교 + U0–U3. fallback 이 없으므로
  U0·U1 을 가장 먼저 짧게 돌려 WER 관문(≤10 %)을 조기 판정한다. 50 Hz 이점(Stage 1 실측)은 U3 하이브리드 ablation 으로 흡수.
  "통합의 가치" 는 같은 encoder 의 encoder-only probe 대 IS-SLM 상태 위 헤드로 판정.
- Next: Stage 1 매트릭스 마무리(재채점) → 결과 페이지·H1 판정 → U0 착수.
- By: tskim

## [2026-09-04] query | README 를 종합 현황판으로 재작성, USLM 단계 U0–U5 통일

- Changed: `README.md`(전면 재작성: 목표·가설·구조 A/B′·로드맵 Phase 0–5 + U0–U5·결정 관문·완료 결과·Stage 1 표·TODO·데이터/인프라·코드 맵·문서 지도),
  `output-interleaved-streaming-slm-architecture`(단계 N → U N, U3 하이브리드 ablation), `output-unified-slm-architecture-plan`(검증 계획 U0–U5),
  `decision-target-architecture`, `task-uslm-feasibility-u0`, `TODO.md`.
- Reason: 사용자 요청 — U 단계 표기를 학습 계획 보고서와 통일하고, README 만 봐도 현황·TODO 를 파악할 수 있게. IS-SLM 보고서의 U0–U5 를
  정본 번호로 채택(토큰율·정렬은 U0, 하이브리드는 U3). Stage 1 매트릭스 잠정 표(13/14)와 5개 판독, Qwen 실외 취약성, cc1s 재채점 중 표기.
- Next: cc1s 재채점·wavlm-large 12.5 완료 후 README §4 표 확정, Stage 1 결과 페이지·H1 판정. U0 착수.
- By: tskim

## [2026-09-04] query | Interleaved Streaming SLM 구조 계획과 비판적 평가

- Changed: `wiki/outputs/output-interleaved-streaming-slm-architecture.md`, `wiki/concepts/streaming-conversational-projection-asr.md`
- Reason: 독립 RNN-T 없이 streamable speech encoder의 soft token과 LLM text state를 반복 결합해 실시간 전사와 미래 대화 역학을 함께 예측하는 통합 SLM 구조를 설계하고, summation fusion·emission policy·causality·실시간 deadline의 실패 조건을 평가했다.
- Next: interleaving-only / raw sum / gated contextual residual 세 조건의 최소 ASR 실험과 80 ms p99 deadline 측정.
- By: tskim

## [2026-09-04] query | 목표 모델 구조 제안

- Changed: `wiki/outputs/output-dual-rate-conversational-projection-architecture.md`, `wiki/concepts/streaming-conversational-projection-asr.md`
- Reason: 현재 Stage 1 중간 결과와 causality 감사를 반영해 목표 모델의 구체적인 dual-rate 구조, 학습 단계, ablation, 중단 조건을 제안했다.
- Next: Stage 1 전체 결과와 seed 반복 후 acoustic-only 대 dual-rate 최소 실험으로 구조의 전제를 검증한다.
- By: tskim

## [2026-09-04] decision | U0.5 관문 재정의·통과, U1 interleaved streaming ASR 학습 준비 착수

- Changed: `decision-asr-backbone`(관문 재정의), `task-uslm-u05-adapter-bridge` → done, 신설 `output-uslm-u05-adapter-bridge`(결과 보고서),
  `task-uslm-u1-interleaved-asr` → doing(설계·구현 기록), `README.md`, `TODO.md`, `wiki/status.md`, `wiki/index.md`.
  코드: `vapasr/uslm/interleave_data.py`(창 데이터셋), `vapasr/uslm/model.py::InterleavedASR`(joint chunk token·특수 토큰·스트리밍 디코더),
  `experiments/u1_train_interleaved.py`(학습 + 스트리밍 평가), `scripts/wiki-regen.py`(log/todo 재생성).
- Reason: 사용자 결정 — U0.5 관문을 "동일 인코더 RNN-T `[56,0]` 보다 우수 + 오프라인 Qwen 대비 ≤ +50 % 상대" 로 재정의. 원안(오프라인 ×1.15)은
  비인과 시스템 기준으로 인과 시스템을 재는 비교였다. 결과(18.2–18.8 / 17.5 / 14.5–15.1 %) 는 새 관문을 충족하므로 backbone 유지, U1 착수.
  U1 설계: 창 시작 = 양 화자 침묵, δ ∈ {2,3,4,6} 무작위 + `<DELAY_d>` 조건화, M=4, 특수 토큰은 임베딩 여유 행 + grad mask, U0.5 ckpt 초기화.
- Next: 스모크(`u1-smoke`) 통과 → v0 run(12k step) → 관문 판정(RNN-T `[56,0]` 대비 ≤ +10 %, 실질 목표 오프라인 U0.5 수준 유지) + 지연 분포·evidence 위반률 보고.
  정렬: otoSpeech 완료, vs02 진행(107/186) → 완료 후 val 창 확대. 인코더 상위 블록 unfreeze ablation.
- By: tskim

## [2026-09-03] task | AI Hub 실물 검증 완료 — 진짜 분리 stereo

- Changed: `task-verify-aihub-stereo-and-access` → done, `raw/sources/experiments/2026-09-03-aihub-71631-vs02-verify.json`(신규 raw),
  `experiments/aihub_extract_and_verify.sh`(신규), `experiments/verify_aihub_sample.py`(무작위 표본·JSON 인덱스·`_f` 파서),
  `source-conversation-corpora`(실물 검증 표), `todo.md`·`TODO.md`·`status.md`·`index.md`. 서버: VS_02.실외 186 wav 해제.
  운영: 오늘 GPU 배정 1번 → `.env.local` `GPU_DEFAULT=1`, `activate-env.sh` 가 `CUDA_VISIBLE_DEVICES` 기본값으로 사용.
  장시간 작업용 `sync-rack4.sh bg/jobs` 추가.
- Reason: VS_02.실외(6.4 GB) 수신 후 무작위 100 wav 검증. **16 kHz / 2 ch / PCM_16** (라벨의 48000 은 원본 표기),
  채널 누설 중앙값 **−64 dB**, 상관 6e-5 — 진짜 분리. 에너지 VAD overlap 비율 중앙값 7.4 % 로 실제 대화 역학 존재.
  라벨 StartTime vs VAD 온셋 오차 중앙값 **30 ms**(p90 370) — 라벨 통계의 '겹침 50 %' 는 발화 끝이 넉넉한 탓.
  VAP target 은 채널 VAD 로, 라벨은 화자·텍스트·대략적 온셋으로 쓴다. 첫 검증 실행에서 `_f` 미정의로 JSON 항목이
  null 이 나온 편집 실수를 고쳐 재실행했다.
- Next: 이용약관 원문에서 어노테이션 파생물 공개 가능 여부 확인(사용자). 2차 TS_01.실내_5(≈95 h) 수신 →
  [[task-build-vap-target-pipeline]]. VAP baseline 재현 진행 중(GPU 1).
- By: tskim

## [2026-09-03] task | 컨테이너 학습 환경 구축 완료

- Changed: `wiki/tasks/task-setup-training-environment.md` → `done`. 신규
  `scripts/setup-container-env.sh`, `scripts/activate-env.sh`, `scripts/smoke-test-models.py`,
  `env/requirements.txt`, `env/requirements-lock.txt`, `env/README.md`. `.env` 에 캐시·conda 항목 추가.
  `sync-rack4.sh exec/shell` 이 activate-env.sh 를 자동 로드. `wiki/sources/source-nemotron-3-5-asr-streaming.md`,
  `source-qwen3-asr.md`, `question-encoder-lookahead-and-causality.md` 에 실측 반영.
  `wiki/todo.md`, `wiki/log.md`, `TODO.md`, `status.md` 갱신.
- Reason: Phase 0 첫 태스크. rack4 `tskim_env` 에 conda env `vapasr`(Python 3.11) 를 만들고
  torch 2.6.0+cu124 / NeMo git main 3.1.0 / transformers 4.57.6 / qwen-asr 0.0.6 / 원 VAP 계열 3개를
  설치했다. 스모크 4종 통과: Nemotron(638M, 79.4 ms/frame, ctx [56,0]·[56,3]), Qwen3-ASR(audio_tower 186M),
  ForcedAligner(한국어 지원), 원 VAP(CPC 5.8M, 50 Hz). 발견 두 가지 — (1) Nemotron 의 언어는
  `transcribe(target_lang=)` 키워드가 dataset 까지 도달하지 않아 **manifest `"lang"` 필드**로 줘야 한다
  (PyPI 3.0.0·git main 공통). (2) `qwen-asr` 가 transformers 를 5.x→4.57.6 으로 다운그레이드하나
  NeMo 와 충돌 없음. 캐시는 `/data3/tskim/cache`, 서드파티 코드는 `/data3/tskim/third_party`.
- Next: [[task-audit-encoder-causality-lookahead]] (Nemotron 은 문서상 우측 context 확인, conv 암묵
  lookahead 절단 실험 남음; Qwen AuT 는 chunk 직접 호출 코드 필요), [[task-verify-aihub-stereo-and-access]],
  [[task-reproduce-vap-turnbench-baseline]] (missing 8 키 확인).
- By: tskim

## [2026-09-03] task | 연구 실행 체크리스트 TODO.md 와 환경 태스크 2건

- Changed: `TODO.md`(신규, 저장소 루트), `wiki/tasks/task-setup-training-environment.md`(신규),
  `wiki/tasks/task-checkpoint-retention-policy.md`(신규), `wiki/todo.md`·`wiki/index.md` 재생성,
  `wiki/status.md` 카운트 갱신.
- Reason: 사용자가 연구 수행용 ToDo 리스트 파일을 요청했다. 생성 대시보드 `wiki/todo.md` 는
  owner 별 표라 단계·관문·의존성이 보이지 않아, 루트에 Phase 0~5 + 논문 + 결정 관문을 담은
  실행 체크리스트를 별도로 두었다. 상세는 태스크 파일이 정본이며 체크리스트는 그 링크다.
  실측에서 드러난 환경 공백(torch 미설치, /data4 여유 575G)이 태스크로 빠져 있어 2건을 추가했다.
- Next: `TODO.md` 는 Directory Contract 에 없으므로 `AGENTS.md` 유지보수 PR 에 함께 반영.
  Phase 0 p0 5건 착수.
- By: tskim

## [2026-09-03] task | VAP baseline TurnBench dev 재현 완료 — 공식과 완전 일치

- Changed: `task-reproduce-vap-turnbench-baseline` → done, `wiki/outputs/output-vap-turnbench-baseline-reproduction.md`(신규),
  `raw/sources/experiments/2026-09-03-vap-turnbench-repro/`(점수 3종·예측 2종), `experiments/reproduce_vap_turnbench.sh`,
  `source-turnbench`(dev 수치), `turn-taking-evaluation-protocol`(기준선 표 split 명시), `task-stage1-encoder-probing`
  (head 학습 데이터 통일), `todo.md`·`TODO.md`·`status.md`·`index.md`·`log.md`.
- Reason: 사용자 요청. HF gated 데이터셋 3개 접근 확보 후 dev(38 대화)에서 (1) 동봉 예측 재채점, (2) 사전학습 원본
  직접 예측, (3) oto fine-tune 체크포인트 직접 예측을 수행했다. (3) 은 sweep 임계값(0.91615/0.85913)과 점수
  (EOT 0.841/0.045/463 ms, INT 0.957/0.100/896 ms)가 동봉 예측과 **완전히 일치** — 재현 성공. (2) 는 0.793/0.094/613 으로
  otoSpeech fine-tune 효과가 큼을 확인. 리더보드 수치(0.845/0.055/368)는 test split 이므로 논문에서 split 명시 필요.
  INT 는 FP(373) 가 TP(332) 수준으로 오경보가 과제. EOT p10 latency 가 음수(−34 ms) — projection 의 선점 사례.
  운영: GPU 배정 변경으로 GPU 3 → 1 로 이전 후 재실행. 심볼릭 링크 깊이 오류 1회.
- Next: [[task-build-vap-target-pipeline]](p0, 마지막 Phase 0), otoSpeech 290 GB·TS_01.실내_5 수신 완료 대기,
  [[task-add-missing-baselines]] 는 turnbench 동봉 baseline(rms_vad, dualturn, wavlm causal) 재사용.
- By: tskim

## [2026-09-03] task | Qwen AuT attention 마스크 복원·변형 실험

- Changed: `experiments/qwen_aut_mask.py`, `experiments/qwen_aut_mask_eval.py`(신규),
  `raw/sources/experiments/2026-09-03-qwen-aut-mask-eval.json`(신규 raw), `output-encoder-causality-audit` 추가 실험 절,
  `task-qwen-aut-causal-adaptation` p2→p1 및 체크 항목 갱신, `decision-asr-backbone` 5항 추가, `source-qwen3-asr`,
  `todo.md`·`TODO.md`·`index.md`. 부수: AI Hub 절차를 PC 다운로드 → `scripts/aihub-upload.sh` 전송으로 변경
  (API 키 발급 불가 확인), `source-conversation-corpora`·`task-verify-aihub-stereo-and-access` 갱신.
- Reason: 사용자 질문 "`_prepare_attention_mask` 를 직접 복원할 수 없나" 에 대한 실험. 레이어 forward 를 감싸
  `cu_seqlens` 로 마스크를 주입했다. block 1 s 가 per-block 실측과 일치해 패치 검증. chunked-causal 1 s 는
  lookahead 420 ms·WER +5.9 % 로 as-is 최선이나 관문은 여전히 초과. **프레임 causal 마스크에서 lookahead 80 ms,
  WER 23.5 %(단일 발화)** — 학습에 없던 마스크에서도 단어 대부분이 보존되어 causal fine-tune 으로 Qwen 을
  살릴 가능성이 열렸다. block 8 s(배포 의도)가 sdpa 무마스크와 11.8 % 다른 점도 확인 — transformers 백엔드
  결과의 재현성 주의.
- Next: 제대로 된 평가셋에서 마스크별 WER/CER, causal 마스크 소규모 fine-tune 회복 폭 측정
  ([[task-qwen-aut-causal-adaptation]]). AI Hub 는 사용자 다운로드 대기.
- By: tskim

## [2026-09-03] task | 특징 캐시 파이프라인 구축·검증, 추출 시작 (Phase 1)

- Changed: `vapasr/features/{__init__,encoders}.py`(신규), `experiments/{extract_features.py,run_feature_cache.sh,make_16k_copy.py,diag_length_invariance.py,diag_stitching.py,diag_qwen_determinism.py}`(신규),
  `experiments/qwen_aut_mask.py`(causal 모드 좌측 창), `wiki/outputs/output-feature-cache-and-compute-budget.md`(신규),
  `task-compute-budget-and-feature-cache` 진행, `source-qwen3-asr`(13 Hz), `index.md`·`todo.md`·`status.md`.
  서버: `otoSpeech16k` 사본(420 대화), `/data3/tskim/features/` 추출 시작(feat-cache-A: cpc·nemotron-c0, B: qwen ×2·wavlm ×2).
- Reason: Stage 1 encoder probing 을 분 단위로 돌리기 위한 frozen 특징 캐시. 인코더 7종을 공통 인터페이스로 감싸고
  긴 파일 세그먼트 처리를 fp32 무분할과 일치시키는 과정에서 네 가지 함정을 실측으로 잡았다: (1) 층 누적 수용장(Nemotron 107 s →
  겹침 120 s), (2) TF32 노이즈, (3) 출력 프레임 수 ≠ 길이×Hz — **Qwen AuT 는 1 s 당 13 프레임(13 Hz)** 이라는 사실 포함,
  (4) Whisper 프론트엔드의 utterance 정규화. 최종 검증 cpc/nemotron 0 프레임, qwen ≤5 프레임(수치 드리프트). RTF: cpc 0.0018,
  nemotron 0.0052, qwen 0.018, wavlm 0.019 → 214 h 에 ≈17 GPU 시간, 저장 ≈430 GB. 사용자 질문("encoder 도 학습시켜야
  하지 않나")에 단계 구분을 답함 — Stage 1 frozen(H1 검증) → Stage 2 unfreeze → Qwen causal fine-tune.
- Next: 캐시 완료 후 stats 로 표 갱신·task done. [[task-stage1-encoder-probing]] 의 probe head 학습 코드(캐시 로더 + VAP head + TurnBench 평가).
  [[task-event-label-heuristics-validation]] 은 dev gold 로 병행 가능.
- By: tskim

## [2026-09-03] task | VAP target 파이프라인 완료 — Phase 0 종료

- Changed: `vapasr/` 패키지 신설 (`data/conversation.py`, `vad.py`, `corpora.py`, `targets.py`, `dataset.py`), `experiments/build_targets.py`,
  `recompute_events.py`, `test_dataset.py`, `raw/sources/experiments/2026-09-03-target-pipeline/`(stats 3 + QC 7),
  `wiki/outputs/output-vap-target-pipeline.md`(신규), `task-build-vap-target-pipeline` → done, `task-event-label-heuristics-validation`
  갱신, `todo.md`·`TODO.md`·`status.md`·`index.md`·`log.md`. `.env` `AIHUB_ADULT_ROOT`, 서버 ASCII 심볼릭 링크(`adult-*`).
  서버: `/data3/tskim/manifests/{aihub-vs02,otoSpeech,turnbench-dev}/` (npz + manifest + qc).
- Reason: Phase 0 마지막 p0. 코퍼스 3종을 공통 `Conversation` 으로 읽어 채널 VAD@50 Hz 를 저장하고, VAP 256-class(원 VAP
  코드 재사용)·hazard τ(censoring)·이벤트(SHIFT/HOLD/INT/BC)를 로드 시 파생한다. 12.5 Hz 는 bins 2/5/8/10. 합성 VAD 테스트와
  QC 이미지 검수로 이벤트 규칙 결함 2건(가짜 SHIFT, terminal overlap→INT)과 판정창(1→3 s)을 고쳤고, AI Hub 화자↔채널
  매핑을 파일별 자동 판정(3/186 뒤바뀜)했다. 결과 156 h, 20 s 창 55,139개, `WindowDataset` 50/12.5 Hz 동작 확인.
  운영 교훈: bg 래퍼에 한글·괄호 경로를 넘기면 깨진다 → ASCII 링크 사용; matplotlib 누락으로 QC 1회 실패.
  추가: TS_01.실내_5 수신·해제 후 빌드 — 757 파일 **196.6 h**(zip 크기 추정 95 h 의 2×, wav 압축률 때문). 서버 총 보유 ≈ 360 h.
- Next: Phase 1. [[task-compute-budget-and-feature-cache]](otoSpeech 리샘플 300 ms/item 해소), [[task-event-label-heuristics-validation]]
  (TurnBench dev gold 로 즉시 가능), [[task-stage1-encoder-probing]]. AI Hub TS_01.실내_5 수신 대기.
- By: tskim

## [2026-09-03] task | Encoder causality·lookahead 감사 완료 — Qwen 조건 위반

- Changed: `experiments/causality_audit.py`(신규), `raw/sources/experiments/2026-09-03-causality-audit.json`(신규 raw),
  `wiki/outputs/output-encoder-causality-audit.md`(신규), `wiki/tasks/task-qwen-aut-causal-adaptation.md`(신규),
  `task-audit-encoder-causality-lookahead` → done, `decision-asr-backbone` 수정(감사 결과 절 추가, summary 변경),
  `question-encoder-lookahead-and-causality` → stable(답변), `source-qwen3-asr`·`source-nemotron`·
  `streaming-causality-and-latency-budget` 에 실측 반영, `TODO.md`·`todo.md`·`index.md`·`status.md`·`log.md` 갱신.
  부수: `scripts/aihub-download.sh`, `experiments/verify_aihub_sample.py`, `.env` AI Hub 항목 (사용자 질문 대응).
- Reason: Phase 0 p0. 특징 단위 절단 실험(fp32, rel tol 1e-3)으로 encoder 별 실효 lookahead 를 측정했다.
  CPC/VAP 0 ms(대조군). Nemotron `[56,0]` ≤80 ms, `[56,1]` ≤160, `[56,3]` ≤320, `[56,6]` ≤480, `[56,13]` ≤880 —
  문서의 chunk 크기가 최대 lookahead 임을 확인. **Qwen3 AuT 는 transformers/sdpa 경로에서
  `_prepare_attention_mask` 가 호출되지 않아 전체 발화 양방향**(n_window_infer 800 vs 100 출력 비트 동일),
  의도된 1 s 블록 모드도 lookahead 0–800 ms(평균 420) + 블록 간 좌측 context 부재. 320 ms 관문 발동 →
  Paper 1 은 Nemotron 단일 backbone(80/160 ms chunk), Qwen 은 적응 연구 결과에 종속으로 결정 수정.
  첫 실행에서 (1,128,T) 텐서의 mel 축을 잘라 Nemotron 이 0 으로 나온 버그를 잡았고,
  vap 패키지가 켜는 전역 deterministic 모드도 해제했다.
- Next: [[task-verify-aihub-stereo-and-access]](사용자 신청 대기), [[task-build-vap-target-pipeline]],
  [[task-reproduce-vap-turnbench-baseline]]. Nemotron ko-KR CER 을 80/160 ms chunk 에서 측정
  ([[task-latency-quality-curve]]). Qwen 적응 연구는 p2.
- By: tskim

## [2026-09-03] schema | 볼트 스키마 초기화

- Changed: `raw/{inbox,sources,meetings,assets}`, `wiki/` 전체 디렉토리와 시드 페이지,
  `scripts/{init-local-user,pull-safe,sync-user-branch,install-skills}.sh`,
  `.skills/{wiki-ingest,wiki-query,wiki-lint,wiki-merge,wiki-task}/SKILL.md`,
  `.gitignore`, `.gitattributes`
- Reason: `AGENTS.md` 의 Directory Contract, Agent Skills, Git Collaboration Policy 에
  따라 빈 저장소에 볼트 골격을 구성했다. 스킬은 `.skills/` 에 에이전트 중립 정본으로
  두고, 런타임 경로(`.claude/skills/`, `.codex/skills/`)는 `install-skills.sh` 가
  만드는 git-ignored 링크다.
- Next: 팀원별 `.llm-wiki-local/user.yaml` 생성, 원격 저장소 연결과 `main` 보호 설정,
  대용량 바이너리 정책 결정, 첫 원천 자료 ingest
- By: unknown

## [2026-09-03] schema | rack4 실험 환경 구성과 .env 단일 관리

- Changed: `.env`(신규, 커밋), `.rsyncignore`(신규), `.gitignore`(`.env.local` 추가),
  `scripts/sync-rack4.sh`(신규), `wiki/decisions/decision-compute-environment.md`(신규),
  `wiki/status.md` 갱신. 원격에 `/home/tskim/VAP`, `/data4/tskim/VAPASR/{experiments,exports}`,
  `/data3/tskim/{corpora,features,manifests,logs}` 생성.
- Reason: 사용자가 학습 서버(rack4), 컨테이너(tskim_env), 프로젝트 경로(/home/tskim),
  체크포인트(/data4/tskim/VAPASR), 데이터(/data3/tskim) 규약을 지정하고 하나의 파일로
  관리할 것을 요청했다. 설정 파일을 쓰기 전에 실제 접속·경로를 검증했고 네 가지를 발견했다:
  (1) `/home` 이 컨테이너에 bind mount 되어 호스트와 inode 가 동일 —
  호스트로 rsync 하면 컨테이너가 즉시 같은 파일을 보므로 별도 복사가 불필요하다.
  (2) `~/.ssh/config` 의 `rack4_tskim_env` 가 172.17.0.6 을 가리키나 실제 IP 는
  172.17.0.18 이라 접속 불가 — 도커 IP 는 재시작마다 바뀌므로 `docker exec` 경로로 고정했다.
  (3) `/data3/tskim`·`/data4/tskim` 이 root 소유라 호스트 계정으로 쓸 수 없어
  컨테이너(root)에서 하위 폴더를 만들고 1019:1019 로 chown 했다.
  (4) **`/data4` 가 97% 사용 중(575G 여유)** 이고 컨테이너에 torch 가 없다.
  macOS rsync 2.6.9 가 `--info=` 를 지원하지 않아 버전 감지 폴백을 넣었고,
  bash 3.2 의 빈 배열 확장 문제도 수정했다. 첫 push 로 112개 파일 동기화를 확인했다.
- Next: `/data4` 체크포인트 보존 정책 수립, 컨테이너 torch 등 학습 환경 구축,
  `AGENTS.md` Directory Contract 에 `.env`·`sync-rack4.sh` 반영하는 유지보수 PR,
  Git private 원격 생성 후 rsync → clone 방식 전환 검토.
- By: tskim

## [2026-09-03] ingest | ChatGPT Streaming VAP 연구 계획 초안

- Changed: `raw/inbox/ChatGPT_Research_Plan.md` → `raw/sources/` 로 분류 이동.
  신규 페이지 33개 — `wiki/sources/` 7, `wiki/concepts/` 7, `wiki/questions/` 6,
  `wiki/decisions/` 2, `wiki/outputs/` 1, `wiki/tasks/` 17.
  `wiki/overview.md`, `wiki/status.md` 갱신. 파생 파일 3종 재생성.
- Reason: 사용자가 ChatGPT 로 작성한 streaming ASR + VAP 통합 연구 초안을 제공하고
  개선·계획 수립·태스크 도출을 요청했다. 초안이 인용한 8개 자료를 웹으로 검증한 결과
  **전부 실재**했으나, 계획을 수정해야 하는 사실 4건을 발견했다:
  (1) DualTurn(arXiv 2603.08216)이 VAP 를 weighted F1 0.633 vs 0.389 로 앞섰는데
  초안에 누락 — H1 의 경쟁 가설이자 필수 baseline.
  (2) Muse Voice Transcribe 는 **closed weights, API 전용** — backbone 후보에서 제외.
  (3) AI Hub 는 내국인 한정 + 재배포 제약 — "Korean TurnBench" 공개 배포 불가.
  (4) Qwen3 AuT 의 causality/lookahead 가 미문서화 — 검증 없이는 latency 비교가 무효.
  추가로 encoder lookahead 회계, Stage 1 교란 변수 통제, τ 의 생존분석 정식화,
  손실 균형과 WER 가드레일, 누락 baseline 3종을 계획에 반영했다.
- Next: Phase 0 의 p0 태스크 4건이 나머지를 막고 있다 —
  AI Hub 실물 검증, encoder causality 감사, VAP baseline 재현, target 파이프라인.
  볼트 운영으로는 `.llm-wiki-local/user.yaml` 의 member_id 사용자 확인,
  원격 저장소 연결, 초기 커밋이 남았다.
- By: tskim

## [2026-09-03] ingest | AI Hub 라벨 전체 통계 + TurnBench 재현 환경 준비

- Changed: `raw/sources/experiments/2026-09-03-aihub-71631-label-stats.json`(신규 raw), `experiments/aihub_label_stats.py`(신규),
  `experiments/verify_aihub_sample.py`(실제 스키마·천단위 구분자 반영), `scripts/aihub-download.sh`(사용 확인),
  `source-conversation-corpora`(라벨 통계 절), `source-turnbench`(코드 저장소 절), `task-verify-aihub-stereo-and-access`,
  `task-build-vap-target-pipeline`(채널 VAD 우선), `task-add-missing-baselines`(turnbench baseline 재사용),
  `task-reproduce-vap-turnbench-baseline` → doing. 서버: `/data3/tskim/corpora/aihub/71631/` 라벨 4개 해제,
  VS_02.실외 다운로드 중; `/data3/tskim/third_party/turnbench` 클론·설치.
- Reason: 사용자가 AI Hub API 키를 확보해(.env → .env.local 로 즉시 이동, 로컬·서버 모두) 서버 직접 다운로드가
  가능해졌다. 라벨 11,023 JSON 통계: **2,765 h**(Training 2,370 / Validation 396), 파일 중앙값 15 min, 3.3 M 발화,
  1.72 M 화자 교대, 10 ms 해상도. 그러나 교대의 50 % 가 '겹침'·겹침/gap 중앙값 ~1 s 로 **라벨 시간은 전사용 발화
  구간이지 VAD 경계가 아님** → VAP target 은 채널별 에너지 VAD 로 만들기로. 라벨상 48 kHz/2 ch 는 스펙(16 kHz)과
  달라 실물 확인 대기. 사용자 요청으로 VAP baseline 재현 착수: TurnBench HF 데이터셋 3개 식별(모두 gated),
  `SesameAILabs/turnbench` 에 scorer·sweep·baseline 20종(dualturn, wavlm causal, rms_vad …)·VAP `predictions-dev.json`
  동봉 확인. 리더보드 VAP 는 **oto fine-tune 체크포인트**(θ 0.9161/0.8591).
- Next: 사용자가 HF 약관 동의 + `HF_TOKEN` 을 `.env.local` 에 → dev/test/otoSpeech 다운로드 → 동봉 predictions-dev 재채점
  → `--pretrained` / `oto` 체크포인트로 직접 예측 재현. VS_02 도착 시 `verify_aihub_sample.py` 로 실물 검증.
- By: tskim

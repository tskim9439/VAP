<!-- generated: do not edit -->
# 활동 로그

마지막 생성: 2026-09-11

`wiki/log/` 의 샤드를 최신순으로 이어붙인 다이제스트다. 직접 편집하지 않는다.

## [2026-09-11] query | turn 토큰·시퀀스 사례·데이터 생성 파이프라인을 별도 제안서로

- Changed: `wiki/outputs/output-phase2-turn-token-proposal.md` 생성. 정본 `output-phase2-streaming-asr-diarization-plan.md` 는 직전 커밋의 §4.4·§5.3 삽입을 되돌려 원상 복구. `output-phase2-final-plan.md` 는 superseded 유지, 링크 수정
- Reason: 사용자가 정본 계획을 기본으로 택하고 교차 직렬화를 선호했으며, turn 토큰 제안은 "정본에 바로 적용하지 말고 별도 보고서로" 요청. 제안서에 규약·사례 7 종·전자동 데이터 파이프라인(VAD → 채널별 정렬 → derive_events 확장 → 슬롯 → 혼합 → 직렬화 → QC)·데이터 종류별 손실 규칙·정본 반영 시 변경 지점을 정리
- Next: 사용자 검토 후 정본 §4.4·§5.3 반영 여부 결정
- By: tskim

## [2026-09-11] query | 제안서 개정 — 토큰 종류를 의미 라벨로, LLM 시드 → 증류 → 전량 → 사람 검증

- Changed: `output-phase2-turn-token-proposal.md` §3b·§3c 신설, 요약·반영 지점·불확실성 갱신
- Reason: 사용자 검토 — 라벨이 음향(타이밍)에만 의존하고 TurnBench 에 과도하게 맞춰졌다는 지적. 위치는 VAD, 종류는 의미(완결/미완결/맞장구)로 분리하고, 텍스트만으로 사후 문맥을 보는 LLM 시드 10 만 → 증류 분류기 → 전량 1 천만 경계 → 사람 κ ≥ 0.7 검증의 저비용 경로를 제안. 완결성 정확도·응답 기회 P/R 을 주 지표로, TurnBench 는 EN 외부 검증
- Next: 사용자 검토 → L1 프롬프트·층화 표본 설계
- By: tskim

## [2026-09-11] query | Phase 2 최대 2화자 스트리밍 ASR·화자 구분·turn-taking 개발 계획

- Changed: `wiki/outputs/output-phase2-streaming-asr-diarization-plan.md`, `wiki/outputs/output-phase1-report.md`의 후속 계획 링크
- Reason: 사용자가 요청한 mono 최대 2화자, 겹침 화자별 전사, 의미 기반 turn-taking 목표를 E2 기반 구조·데이터·학습·평가·실행 순서로 구체화했다. 기존 엄격 관문 미달과 신규 제안 관문을 구분했다.
- Next: Q0 고정 평가 pack·대화 스키마·합산 mixer·serializer 및 작은 overfit 구현. 현재 문서는 계획이며 학습 job은 제출하지 않았다.
- By: tskim

## [2026-09-11] query | Phase 2 실행 요약 — 일정·데이터 보유 현황·결정 사항

- Changed: `wiki/outputs/output-phase2-plan.md` 생성(정본 [[output-phase2-streaming-asr-diarization-plan]] 의 보조 페이지), `output-phase1-report.md` §7 이월 항목에 두 링크
- Reason: 사용자의 Phase 2 계획 요청에 두 세션이 동시에 계획을 썼다. 설계·관문은 정본으로 일원화하고, 이 페이지에는 정본이 "실측 전" 으로 남긴 서버 가용량(71631 248 h·otoSpeech 104.9 h·TurnBench dev 보유, CANDOR 없음)·달력 일정(Q0–Q4, 09-15 → 11-14)·혼합 비율·학습 비용·사용자 결정 4 건만 둔다
- Next: 사용자 결정(EN 대화 코퍼스, 71631 추가 반입, 단일 모델·항상 태그, turn 출력 형식) → Q0 착수. Switchboard/CallHome 시간축 복원 가능 여부 확인
- By: tskim

## [2026-09-11] query | Phase 2 정본 계획 비판과 대안

- Changed: `wiki/outputs/output-phase2-plan-critique.md` 생성, `output-phase2-plan.md` 에 링크
- Reason: 사용자가 정본 계획의 비판과 더 나은 해법 분석을 요청. 코드 주장 3 건 검증(모두 사실), 제안 9 건(채택 권고 5: 청크 내 화자별 그룹 직렬화, 텍스트 전용 완결성 사전학습+LLM 약라벨, ASR 강화 병렬 트랙, E2 레시피 초기값, MVP 경로; 조건부 4: 화자 슬롯 메모리, 슬롯 조건 오디오 토큰 2 개, 자연형 overlap 합성, turn 헤드 δ=2)
- Next: 사용자가 채택할 제안을 고르면 정본 계획에 반영
- By: tskim

## [2026-09-11] query | Phase 2 최종 계획안(권고)

- Changed: `wiki/outputs/output-phase2-final-plan.md` 생성, 비판·실행 요약 페이지에 링크
- Reason: 사용자가 비판을 반영한 단일 최종 계획안을 요청. 정본 계획의 계약·관문·평가 규율 + 비판의 제안(청크 내 화자별 그룹 직렬화, 텍스트 전용 완결성 사전학습·LLM 약라벨, 슬롯 메모리, 슬롯 조건 오디오 토큰 대안, ASR 강화 트랙 A, 자연형 overlap 합성, E2 레시피, turn 헤드 δ=2, MVP) + 실행 요약의 일정·데이터를 한 문서로
- Next: 사용자 결정 5 건(§13) → Q0 착수(09-15)
- By: tskim

## [2026-09-11] query | 최종 계획안 §4 확장 — 사례별 블록 구성과 Muse 식 turn 토큰

- Changed: `output-phase2-final-plan.md` §4 전면 개정(토큰 6 종, 블록 문법, 시각 규칙, 무음·단일 화자·교대·맞장구·끼어들기·동시 시작·flush 사례, 학습 규칙, 디코드 상태 기계), §3·§5·§7·§9·§11 정합
- Reason: 사용자 지적 — 무음·단일 화자 등 사례별 블록 구성과 학습 방법이 부족하고, 화자별 `<end_of_turn>` 류 turn 라벨(Muse 처럼)이 필요
- Next: serializer 단위 테스트를 §4.4 사례로 작성(Q0)
- By: tskim

## [2026-09-11] query | Phase 2 비판 검토와 계획 v1.1 개정

- Changed: `output-phase2-streaming-asr-diarization-plan`, `output-phase2-plan`, `output-phase2-critique-response`, 비판글 답변 링크, 동시 추가된 `output-phase2-final-plan`의 개정 정본 우선 적용 안내
- Reason: 사용자 요청으로 비판 P1–P9를 코드·1차 자료와 비교하고 채택·조건부 수정·반론을 기록했다. ASR 강화와 의미 학습 트랙을 추가하고 직렬화·slot memory·recipe·MVP를 구체화했다.
- Next: Q0 데이터·저장 parity·serializer 검증, S 라벨 pilot과 A 기준선. 이번 작업은 문서 개정이며 코드 구현·학습 제출은 하지 않았다.
- By: tskim

## [2026-09-11] query | Phase 2 무음·단독·겹침 블록 및 화자별 start/end 토큰 명세

- Changed: `output-phase2-block-and-turn-label-spec`, `output-phase2-streaming-asr-diarization-plan` v1.2, `output-phase2-plan`, `source-muse-voice-transcribe`
- Reason: 사용자 요청에 따라 18가지 블록 사례, 화자별 독립 OPEN/CLOSED 상태, start/end 공동 생성, 실제 label 생성·관측 시각·loss·EOF·부분 감독 계약을 구체화했다. Muse 공식 token 표기와 우리 확장을 구분했다.
- Next: Q0 라벨 생성기·serializer·상태 parser 및 fixture 구현, 신뢰도 높은 자연 turn 창 검수. 이번 작업은 계획·명세이며 코퍼스 전체 라벨 생성이나 모델 학습을 수행하지 않았다.
- By: tskim

## [2026-09-10] query | Stage 2 run E2(인코더 해동) 최종 평가 보고서

- Changed: `wiki/outputs/output-stage2-e2-final-eval.md` 신규(§4 스윕·§5 test 는 측정 중)
- Reason: 인코더 해동 + LR 2e-5 로 8 코퍼스를 학습한 E2 가 C2 를 모든 셋에서 앞섬(select δ=2 0.076/0.139/0.167/0.292). 기준 모델 교체
- Next: 스윕·test 수치 채움, E3(추가 epoch)·NEXT_AUDIO 가중 실험, 디코더 최적화
- By: tskim

## [2026-09-10] query | Stage 2 run D2 최종 평가 보고서

- Changed: `wiki/outputs/output-stage2-d2-final-eval.md` 신규(§4 스윕·C2 δ4 는 측정 중), `vapasr/uslm/mono_data.py` 결함 수정 기록
- Reason: 확장 DB run D2 의 최종 성능을 C2 와 같은 표본으로 비교. 평가 중 71631 오디오 조립 버그(src_offset_s 누락)를 발견해 D2 를 오염 run 으로 판정
- Next: D3(수정 로더) 학습 → 같은 보고서 형식으로 비교, E1 인코더 해동은 D3 위에서
- By: tskim

## [2026-09-10] query | Phase 1 종합 보고서

- Changed: `wiki/outputs/output-phase1-report.md` 신규
- Reason: 사용자 요청 — 지금까지의 실험·구현·데이터·모델 구조·개념을 Phase 1 로 묶어 정리(파일럿 → C2 → D/D2/D4 → E2, 실시간 데모·MLX 포함)
- Next: test 수치 반영(E2 보고서 §5 와 동기화), Phase 2(turn-taking 헤드) 계획 페이지
- By: tskim

## [2026-09-09] task | mxc 모델 로드 지연 해결 — 로컬 디스크 컨테이너·노드 스테이징·인코더 캐시

- Changed: `wiki/sources/source-mxc-model-load-latency.md`, `scripts/stage-env-local.sh`, `scripts/activate-env.sh`, `vapasr/hf/stage.py`, `vapasr/features/online.py`(인코더 캐시), `slurm/s3_train_hf.sbatch`·`s2_prep.sbatch`·`s3_eval.sbatch`, `.env`(sa_tskim_fd)
- Reason: NFS 위 conda env 때문에 추론 로드 16 분·학습 시작 25 분. 로컬 디스크로 옮겨 17 초.
- Next: 67293(D2) requeue 에서 노드 스테이징 동작 확인(`[stage]`·`로컬 env 사용` 로그)
- By: tskim

## [2026-09-09] task | 한국어 DB 확장 — AI Hub 71631 자유대화 · 031/033 방송 manifest

- Changed: `vapasr/data/aihub.py`(71631 stereo 리더 병렬화·NFC 디렉토리 매칭, 031/033 간투사 `/` 제거), `vapasr/data/textnorm.py` v1.3.0, `experiments/s1_build_manifest.py`(aihub71631-train/dev, aihub-bc-train), `wiki/outputs/output-dataset-schema-v1.md` §5b
- Reason: NIKL 추가 오디오가 없어 71631 → 031/033 → 98 순으로 한국어 데이터를 늘리기로 함. 결과 aihub71631-train 231,647 발화 159 h · dev 66,365 발화 44 h · aihub-bc-train 448,867 발화 578 h. 98 은 라벨 파일이 서버에 없어 보류.
- Next: `slurm/s2_prep.sbatch` 로 세 manifest 정렬(2 노드) → D2 TRAIN 에 추가. 98 라벨 확보 여부 확인.
- By: tskim

## [2026-09-09] query | run C2 최종 모델 WER/CER·타이밍 평가 보고서

- Changed: `wiki/outputs/output-stage2-c2-final-eval.md`, `raw/sources/experiments/2026-09-08-hf-C2-final-eval/`
- Reason: 사용자 요청 — 1,930 h full FT 최종 모델의 정확도와 방출 타이밍 평가. select 표본으로 checkpoint 선정(final), δ·next_bias 스윕으로 지연–정확도 트레이드오프, RNN-T 대조군·파일럿과 비교. δ=4 에서 dev-clean 0.056 / dev-other 0.122 / kspon-dev 0.156(대조군 0.044 / 0.082 / 0.202).
- Next: 전체 dev 수치(§4, slurm/s3_eval.sbatch), 디코더 tick 최적화, D run(66523) 결과와 비교.
- By: tskim

## [2026-09-09] query | SoulX-Duplug(arXiv 2603.14877) 와 우리 설계 비교

- Changed: `wiki/sources/source-soulx-duplug.md` 신규
- Reason: 사용자가 유사 논문으로 지목. interleaved 청크 streaming ASR + 상태 토큰 구조가 우리와 같은 계열이며, 작은 청크 스트리밍 ASR 의 어려움을 동일하게 보고하고 외부 ASR teacher-forcing 으로 우회
- Next: VAP 층 설계 시 상태 토큰 5 종·토큰별 손실 가중·LLM 라벨링 파이프라인 참고
- By: tskim

## [2026-09-08] task | 모델·학습 코드 HF 전환 (vapasr.hf + VapAsrTrainer) 구현·검증

- Changed: `vapasr/hf/{configuration_vapasr,modeling_vapasr,liger,data,trainer}.py`, `experiments/s3_train_hf.py`, `experiments/hf_parity.py`, `slurm/s3_train_hf.sbatch`, `scripts/activate-env.sh`, `wiki/sources/source-hf-trainer-migration.md`, `raw/sources/experiments/2026-09-08-hf-trainer-migration/`
- Reason: [[output-huggingface-training-framework-choice]] 의 권장(HF-native 모델 + Trainer, Liger 는 동등성 확인 후) 을 구현. 기존 ckpt 와 수치 패리티(손실·디코드 동일), Liger +24 % 처리량, 2-GPU 스모크에서 학습·분산 평가·저장·재개·선점 동작 확인.
- Next: 8-노드 SLURM 실전 run(hf-C3), select/final 평가 스크립트 HF 이관, DeepSpeed/FSDP 는 인코더 해동 시 검토.
- By: tskim

## [2026-09-08] task | 데이터 스키마 v1 (manifest·정렬 카드, 검증기, 로더 관문)

- Changed: `vapasr/data/schema.py`, `experiments/ds_cards.py`, `vapasr/uslm/mono_data.py`(카드 관문), `experiments/s1_build_manifest.py`(dataset.json), `experiments/s1_align.py --card`, `slurm/s2_prep.sbatch`, `wiki/outputs/output-dataset-schema-v1.md`
- Reason: 코퍼스·단계가 늘면서 manifest/정렬/캐시 계약이 코드에 흩어져 있어 스키마를 못 박고 카드(dataset.json/align.json)로 재현성·검증을 관리. 기존 행은 수정하지 않음.
- Next: 전 산출물 카드 생성 결과 확인, VAPASR_STRICT_SCHEMA 기본화 시점 결정, parquet 캐시.
- By: tskim

## [2026-09-08] query | 현행 모델 구조와 시퀀스 규약 보고서

- Changed: `wiki/outputs/output-vapasr-model-and-sequence.md`
- Reason: 사용자 요청 — 현재 VAP-ASR 모델(인코더·adapter·thinker·특수 토큰)과 80 ms 청크 인터리브 시퀀스·라벨·손실·디코드·학습 설정을 코드 기준으로 한 곳에 정리. 인코더/thinker 설정은 서버의 .nemo·config.json, 스트림 통계는 정렬 캐시, 예시는 s1_show_sequence 출력으로 확인.
- Next: 인코더 해동·디코드 속도 개선 시 §2·§6 갱신.
- By: tskim

## [2026-09-08] query | VAPASR Hugging Face 전환과 학습 프레임워크 선택

- Changed: `wiki/outputs/output-huggingface-training-framework-choice.md`
- Reason: 현재 VAPASR의 custom bilingual batching, weighted sparse CE, streaming sentinel, SLURM 선점 재개를 보존하면서 Hugging Face 친화적으로 전환할 주 경로를 비교했다. HF-native core와 Trainer/Accelerate를 채택 후보로 두고 DeepSpeed·Liger는 독립 검증 옵션, MS-SWIFT는 추후 편의 계층, NeMo/Megatron 전체 전환은 대규모 확장 전까지 보류했다.
- Next: 진행 중인 1,930 h run을 변경 없이 완료한 뒤 HF config/model/processor와 checkpoint round-trip 회귀 시험부터 구현한다.
- By: tskim

## [2026-09-08] query | HF Trainer 마이그레이션 구현 검토

- Changed: `wiki/outputs/output-huggingface-training-framework-choice.md`
- Reason: [[source-hf-trainer-migration]]과 실제 구현을 대조해 legacy 패리티·Liger·2-GPU 분산 학습 검증을 통과로 판정하고, 8-node 실행 전 adapter 인자 누락과 정확한 data resume·recipe override·평가 이력·부분 checkpoint 위험을 기록했다.
- Next: `slurm/s3_train_hf.sbatch`의 초기화 변수 순서를 먼저 수정하고, 무중단/재개 parameter hash 시험과 8-node 짧은 smoke를 수행한다.
- By: tskim

## [2026-09-07] task | asr-tn-v1.0.0 구현·audit·동결

- Changed: `vapasr/data/textnorm.py`, `vapasr/data/kspon.py`, `experiments/tn_audit.py`, `experiments/s1_build_manifest.py`, `experiments/s1_align.py`, `vapasr/uslm/mono_data.py`, `tests/test_textnorm.py`, `tests/golden/asr-tn-v1.0.0.json`, `wiki/decisions/decision-asr-tn-v1-freeze.md`, `wiki/sources/source-asr-tn-v1-audit.md`, `wiki/outputs/output-asr-tn-v1-spec.md`(동결 판정 절), `raw/sources/experiments/2026-09-07-asr-tn-v1-audit/`
- Reason: Stage 2 데이터 확장 전에 정규화 규약을 동결하라는 지시. spec 대로 구현하고 LibriSpeech 960 h·KsponSpeech 01–05 전체 audit 와 230 h diff 를 수행해 관문 7/8 통과, fingerprint 관문을 정렬기·학습기에 넣었다.
- Next: Kspon 01–05 review TSV 사용자 검토(관문 6) → librispeech-960·kspon-full manifest 생성(fingerprint) → align-asr-tn-v1/ 정렬·특징 추출 → scaling curve.
- By: tskim

## [2026-09-07] query | Streaming SpeechLLM 및 저지연 ASR 관련 연구 비교

- Changed: `wiki/sources/source-streaming-speech-llm-related-work.md`, `wiki/sources/source-muse-voice-transcribe.md`, `wiki/outputs/output-interleaved-streaming-slm-architecture.md`, `wiki/outputs/output-stage1-mono-pilot.md`
- Reason: Samsung Intermixed SpeechLLM, Speech ReaLLM, MoChA decoder-only streaming ASR, Moshi, Muse, Ar-RNN-T, transducer MinLT를 방출 정책·정렬·지연·공개성·VAPASR 적용 순서로 비교하기 위해 작성했다.
- Next: 1,930 h Stage 2A scaling 결과 뒤 windowed target과 expected-latency loss의 우선순위를 판정한다.
- By: tskim

## [2026-09-07] query | Stage 1 RNN-T 동급 가능성 평가

- Changed: `wiki/sources/source-stage1-mono-run-ab.md`, `wiki/sources/source-stage1-mono-pilot-6000-sentinel-partial.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/status.md`
- Reason: Stage 1 개선 run A/B 결과를 근거로 IS-SLM이 RNN-T 동급 정확도와 예측 가능한 방출 타이밍을 동시에 달성할 가능성을 엄격히 평가하고, 단순 epoch 연장보다 데이터·정렬 목표·런타임 개선을 우선하는 관문을 기록했다.
- Next: B 기반 200/500/1,000/1,900 h scaling curve, full-dev 평가, windowed alignment·latency loss·KO next_weight Pareto sweep, end-to-end tick 측정.
- By: tskim

## [2026-09-07] query | Stage 1 ASR 데이터 확장 추천 리스트

- Changed: `wiki/outputs/output-stage1-asr-data-expansion-priority.md`, `wiki/status.md`
- Reason: 1,930 h 30 epoch 학습과 병행해 준비할 EN·KO 코퍼스를 준비도, 전사 품질, 도메인 상보성, 중복, 라이선스와 평가 오염 위험으로 우선순위화했다.
- Next: Pack X1(NIKL 400–500 h, Switchboard, otoSpeech, AMI IHM)의 corpus parser·split·QC manifest를 만든다.
- By: tskim

## [2026-09-07] query | 영어 대소문자·문장부호 출력 규약

- Changed: `wiki/outputs/output-asr-tn-v1-spec.md`, `wiki/concepts/asr-text-normalization.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/sources/source-asr-output-style-probe.md`, `wiki/status.md`
- Reason: 영어 대소문자와 문장부호를 평가에서 제거하는 데 그치지 않고 최종 ASR 출력 능력으로 학습·평가해야 한다는 요구를 TN v1에 반영했다.
- Next: lexical audit 동결 후 1,930 h alignment를 시작하고, display 평가 subset과 provenance, Qwen exact-match 필터·masked loss·지표를 full-FT 전에 별도로 동결한다.
- By: tskim

## [2026-09-07] query | asr-tn-v1.0.0 동결 후보 규약

- Changed: `wiki/outputs/output-asr-tn-v1-spec.md`, `wiki/concepts/asr-text-normalization.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/status.md`
- Reason: 1,930 h manifest·alignment 생성 전에 LibriSpeech·KsponSpeech의 target/score 규약, 지원·미지원 숫자 패턴, quarantine, golden test, fingerprint와 버전 조건을 고정할 문서 정본이 필요했다.
- Next: TN v1 구현, 전체 transcript audit, 기존 230 h target/token-ID diff, 목적 표본 검토 후 frozen 판정.
- By: tskim

## [2026-09-07] ingest | mxc `/soundai/DB` 음성 데이터셋 서베이

- Changed: `raw/sources/mxc-soundai-DB-survey.md`, `raw/sources/SLM원천데이터.md`, `wiki/sources/source-mxc-soundai-dataset-survey.md`, `wiki/sources/source-conversation-corpora.md`, `wiki/status.md`
- Reason: `raw/inbox/`의 서버 DB 조사표와 SLM-SPEECH ID 목록을 연구 자료로 분류하고, 서버 실측과 공식 규모·포맷·사용 목적의 차이를 보존한 source note로 합성했다.
- Next: NIKL PCM·시간·TN 표본과 MNSC 중복·PART 구성을 감사한다.
- By: tskim

## [2026-09-06] task | Stage 1 mono 파일럿 준비 완료 — overfit 통과(옛·새 규약), 대화 코퍼스 mxc 업로드, 6,000-step 파일럿 시작

- Changed: `task-uslm-u1-interleaved-asr`(U1a-0 절 신설: 데이터·검증·코드 결함 수정·overfit 결과·업로드), `plans/stage1-mono-pilot.md`(§3.3 텍스트 규약, §4 flush 규약, §8 판정 기록),
  `.env`(MXC_OTOSPEECH_DIR·MXC_AIHUB_*·MXC_TURNBENCH_* 추가), `raw/sources/experiments/2026-09-05-asr-output-style-probe.md`(신설).
  코드: `vapasr/data/{kspon,streams,textnorm}.py`, `vapasr/uslm/{mono_data,mono_model}.py`, `experiments/s1_{verify_data,build_manifest,align,extract_features,train_mono,tok_check}.py`, `vapasr/data/interleave.py`(끝 토큰 폐기 결함 수정).
- Reason: [[decision-mono-input]] 에 따른 단일 화자 mono 경로 신설. 검증기·manifest·정렬·특징 캐시·데이터셋·모델·학습기를 새로 두고 overfit 으로 파이프라인 정상성을 확인했다.
  텍스트 규약은 Qwen3-ASR·Nemotron 출력 실측(EN 숫자 단어, KO 숫자 한글 읽기)에 맞춰 `textnorm.py` 로 통일하고 KsponSpeech 를 `align2/` 에 재정렬했다.
  overfit: 옛 규약 1,500 step·새 규약 900 step 모두 EN/KO 오류 0, tok/chunk = 참조, evidence 위반 0, 지연 p50 ≈ +200 ms(δ=2). tick p99 151–186 ms 는 80 ms 미달 → 실시간성 관문은 단독 측정 전까지 미통과.
  대화 코퍼스(otoSpeech16k 23 GB, AI Hub 71631 wav 54 GB, TurnBench 17 GB)는 rack4 → 맥 T5 → mxc `/soundai/DB/raw/` 로 업로드(rack4→mxc 직접 전송은 정책상 불가, 맥에서는 HF 차단).
- Next: 6,000-step 파일럿(`s1-mono-pilot`, sentinel 1 k 마다) → `--select` 큰 dev 표본으로 ckpt·bias 선택 → `--final` 보고 세트 전량. 대조군(Nemotron RNN-T `[56,0]`·Qwen 오프라인, eval_other 는 같은 2,687 개) 측정.
  tick p99 단독 측정 + CUDA graph/배치 디코드 검토. mxc 잡파일(`._*` 70 개, `_xfer_test` 768 MB)·rack4 azcopy 정리는 사용자 승인 대기.
- By: tskim

## [2026-09-06] ingest | Stage 1 mono 6,000-step sentinel 중간 결과

- Changed: `raw/sources/experiments/2026-09-06-stage1-mono-pilot-6000-sentinel-partial.md`, `wiki/sources/source-stage1-mono-pilot-6000-sentinel-partial.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/status.md`
- Reason: 파일럿의 EN WER 하락과 정상 방출 타이밍, KO 과소 방출, RNN-T 대비 큰 절대 격차를 기록하고, 6,000 mixed step이 언어별 0.39–0.46 epoch에 불과하며 LR이 먼저 0이 되는 과소학습 조건임을 분리해 해석했다.
- Next: @6000 나머지 sentinel 완료, 같은 큰 dev에서 ckpt·bias 선택, 교사강제 정확도와 자유실행 S/D/I 진단, 필요 시 KO next-weight sweep, Stage 2 전 13k–16k one-epoch 연장 대조.
- By: tskim

## [2026-09-06] ingest | Stage 1 mono overfit 1,500-step 완료

- Changed: `raw/sources/experiments/2026-09-06-stage1-mono-overfit-1500.md`, `wiki/sources/source-stage1-mono-overfit-1500.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/status.md`
- Reason: @900부터 EN·KO 내용·방출률·설계 지연·evidence-time이 모두 수렴해 @1500까지 유지된 결과를 근거로 overfit 기능 관문을 통과 처리하고, GPU 경합으로 오염된 tick 측정과 여전히 남은 80 ms 실시간성 관문을 분리했다.
- Next: `align2` 완료 후 `overfit-v2` 선택 ID의 KO 변경 타깃·EN 다중발화 경계 coverage를 확인하고, 부족하면 표적 회귀 표본을 추가한다. 900-step 재검증 통과 시 대조군 측정 후 6,000-step 파일럿을 실행하며 최종 tick은 한가한 GPU에서 단독 측정한다.
- By: tskim

## [2026-09-05] ingest | Stage 1 mono overfit @600 타이밍

- Changed: `raw/sources/experiments/2026-09-05-stage1-mono-overfit-600-timing.md`, `wiki/sources/source-stage1-mono-overfit-600-timing.md`, `wiki/outputs/output-stage1-mono-pilot.md`, `wiki/status.md`
- Reason: KO·EN 16개 overfit의 내용·방출 타이밍·처리시간 결과를 보존하고, EN evidence-time 위반에서 실제 선행 방출과 반복 토큰 매칭 오류를 분리하며, 라벨 이월과 런타임 처리 backlog를 구분하기 위해 합성했다.
- Next: @900–1500 위반 토큰 문맥 감사, leading-silence shift test, deferred `<NEXT_AUDIO>`+다음 audio forward 최적화, 새 KsponSpeech 발음형 `align2/` QC와 overfit 재현, end-to-end chunk service time 기반 deadline miss·누적 runtime lag 측정.
- By: tskim

## [2026-09-05] ingest | Qwen3-ASR · Nemotron 출력 표기 실측

- Changed: `wiki/sources/source-asr-output-style-probe.md`, `wiki/concepts/asr-text-normalization.md`, `wiki/sources/source-qwen3-asr.md`, `wiki/sources/source-nemotron-3-5-asr-streaming.md`, `wiki/status.md`
- Reason: 두 사전학습 ASR의 실제 영어·한국어 숫자 출력과 구두점·태그 동작을 근거로, 늘어나는 코퍼스의 학습 타깃과 채점 표기를 하나의 규약으로 관리하기 위해 합성했다. 12개 목적 표본의 일반화 한계와 구현 감사에서 드러난 `num2words` 재현성 위험도 함께 기록했다.
- Next: `num2words` 의존성과 실패 정책을 고정하고 숫자 문맥별 단위 검사를 추가한 뒤, 새 KsponSpeech manifest·`align2/`로 overfit을 재검증한다.
- By: tskim

## [2026-09-05] decision | 모델 입력을 mono 단일 채널로 확정, 채널별 입력(merge·화자별 오디오 토큰) 폐기

- Changed: 신설 `decision-mono-input`(accepted). `decision-asr-backbone` 세부 3항과 `decision-target-architecture` 결정 절의
  "joint chunk token / 화자별 인코딩 → merge" 를 superseded 표시하고 새 결정으로 링크(두 페이지의 나머지 결정은 유효).
  `README.md` §2(입출력 그림·블록 그림·Stage 3 표·관문), `PLAN.md`(원칙 4 신설, Stage 0 결정 (c), Stage 3 학습 DB·3-1~3-3·실패 조건·데이터 표), `wiki/index.md`.
- Reason: 사용자 결정 — 실제 서비스 입력은 마이크 하나의 mono 오디오다. 코퍼스가 화자별 분리 채널로 배포된다는 이유로 채널별 입력을
  전제한 설계는 배포 불가이고, 화자 구분을 모델이 배우지 않는다. 분리 채널은 라벨(VAD·정렬 → `<SPK_A/B>`·VAP256·hazard)과
  overlap 제어 혼합 합성에만 쓴다. U0.5 adapter 가 단일 채널 분포로 학습된 점과도 일치(U1 v1 merge 분포 이동 가설 원천 제거).
- Next: Stage 1 mono 리더·시퀀스 생성기 신설(두 채널 코드 `interleave_data.py`/`model.py` 는 주 경로 제외). Stage 3 은 분리 채널 합산 혼합으로
  3-1 혼합(전사만) → 3-2 화자 태그 → 3-3 overlap. mono 입력 encoder-only probe 대조군 재구성. U1 v2 run 처리(완주/중단)는 사용자 결정 대기.
- By: tskim

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

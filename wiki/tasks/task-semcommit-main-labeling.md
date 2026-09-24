---
type: task
status: open
owner: tskim
due: TBD
priority: p1
created: 2026-09-24
updated: 2026-09-24
summary: semcommit <SEM_END> 본 라벨링 실행 문서 — mxc 에서 전 코퍼스(전 언어·대화체 포함)를 레시피 v0.3.3 으로, 판정 LLM 은 EXAONE-4.0 과 Qwen3.8-27B 만 사용
sources:
  - '[[output-semcommit-recipe-v0.3]]'
  - '[[output-semcommit-gold-v1]]'
  - '[[decision-semcommit-turn-eot-scope]]'
  - '[[output-stage1-asr-data-expansion-priority]]'
---

# semcommit 본 라벨링 (대규모, mxc)

이 문서는 다른 세션이 **그대로 따라 실행하는 절차서**다. 왜 이렇게 하는지는 [[output-semcommit-recipe-v0.3]], 골드셋은 [[output-semcommit-gold-v1]], 범위 결정은 [[decision-semcommit-turn-eot-scope]] 를 본다.
원 계획서(`raw/inbox/streaming_asr_semantic_commit_plan.md`)는 설계 배경일 뿐이다 — `<TURN_END>`, N 정책, 등급 방식은 이 문서와 레시피 v0.3 이 우선한다.

## 배경
- rack4 소규모 검증(2026-09-24)에서 레시피 v0.3.3 이 정해졌다: 후보 = Stage A ∪ 마지막 단어·구간 끝·구두점 문장 끝, 등급 = 가지별 P(SAFE)·P(REVISION) 임계값(골드 dev 절반에서 조정),
  N = 사람 전사 표지 + 규칙 음성(스트림 머리 대답어·한국어 연결어미). 8B 교사로도 교사 A 정밀도 0.89–0.99 / 재현율 0.70–0.82, 모델 최고 F1 영어 0.80·한국어 0.84.
- 본 학습은 **두 언어 모두 시간 제한 없이 전 데이터, 대화체 포함**, Phase 1 은 **`<SEM_END>` 만** 가르친다(턴 종료는 Phase 2 `<EOT>`).
- **판정 LLM 은 EXAONE-4.0 과 Qwen3.8-27B 두 모델만 쓴다(사용자 결정 2026-09-24).** Stage A(후보 제안)·Stage B(판정자)·Stage C(미래 안정성) 모두 이 둘 중에서 고른다.
  gpt-oss·Qwen3-8B·EXAONE-3.5 등 다른 모델은 본 라벨링에 쓰지 않는다(rack4 결과는 기준선 비교용으로만).

## 지켜야 할 규칙 (mxc)
- 원격 서버(mxc·rack4)에서 파일·폴더를 **지우거나 옮기지 않는다**(`rm`·`rmdir`·`mv`·`find -delete`·`--delete` 동기화 금지).
- `/lustre` 는 읽기만 한다. 산출물은 `/soundai/users/tskim/VAPKT-data/`(`.env` 의 `MXC_*`) 아래에만 쓴다. Blob NFS 라 작은 파일을 많이 만들지 말고 shard 단위 파일로 쓴다.
- 파일 전송·코드 동기화는 **azcopy**. 모델·외부 데이터는 **mxc 에서 직접 받지 않는다** — 로컬 맥에서 T5 SSD(`/Volumes/Samsung_T5`)로 받아 azcopy 로 올린다.
- SLURM 제출·취소는 **사용자가** 스케줄러 셸에서 한다. 세션은 sbatch 스크립트와 제출 명령을 준비하고, 로그 파일로 진행을 본다. 선점 없는 `apex` 권장, `slurmmxch200v5-hpc-52` 는 `--exclude`.
- 컨테이너 기본 python 을 쓰지 않는다(vapasr env 는 `scripts/activate-env.sh`, 노드 로컬 복사는 `scripts/stage-env-local.sh`). 판정 모델이 더 새 transformers 를 요구하면 **별도 교사 env** 를 만들고 vapasr env 는 건드리지 않는다.
- 비밀값(SAS·HF 토큰)은 `.env.local` 에만. `.env` 는 커밋된다.
- 평가 전용 데이터는 학습 라벨에서 뺀다: Earnings-22, TurnBench dev/test, 골드 v1 스트림(LibriSpeech test-clean, GigaSpeech test, Kspon eval_clean·eval_other).

## 0. 준비
1. **코드**: 브랜치 `tskim/query/semcommit-streaming-asr`(원격 푸시됨, 커밋 8ebc155 이후)를 mxc 작업 디렉터리 `/soundai/users/tskim/VAPKT` 로 동기화.
2. **골드 묶음**: 로컬 `/Volumes/Samsung_T5/VAPKT-DB/semcommit-gold-v1`(words·gold·scores·wav, 55 MB, `SHA256SUMS`)을 `/soundai/users/tskim/VAPKT-data/data/semcommit/gold/v1` 로 올리고 `sha256sum -c SHA256SUMS`.
3. **판정 모델**(로컬에서 받아 올림, `/soundai/Model/<이름>`):
   - **EXAONE-4.0**: 판정에는 32B(`LGAI-EXAONE/EXAONE-4.0-32B`)를 쓴다(1.2B 는 판정자로 쓰지 않는다). transformers 내장 아키텍처(`Exaone4ForCausalLM`, 4.54 이상). 라이선스(비상업 조건) 확인.
   - **Qwen3.8-27B**: 사용자 지정 모델 — **정확한 HF repo id·아키텍처·필요 transformers 버전을 먼저 확인**한다(2026-09-24 이 문서 작성 시점에 rack4 에 없고 검증된 적 없음).
     참고: rack4 의 Qwen3.5 는 `Qwen3_5ForConditionalGeneration`(model_type `qwen3_5`)이었다 — 같은 계열이면 지금 로더(`AutoModelForCausalLM`)로 안 올라갈 수 있다.
   - 둘 다 bf16 로 H200 143 GB 한 장에 올라간다(32B ≈ 64 GB, 27B ≈ 54 GB). 로더는 `local_files_only=True` 라 전체 스냅샷이 로컬에 있어야 한다.
     작업 시작 때 가중치를 노드 로컬 `/tmp/sa_tskim/...` 로 복사하면 NFS 로드 시간을 줄인다(env 복사와 같은 방식).
4. **LibriSpeech-PC 구두점 전사**(`--pnc-dir`, rack4 `/data5/LibriSpeech/librispeech_pnc`)를 mxc 에 올린다 — 없으면 영어 LibriSpeech 에 `punct_final` 이 안 붙어 A 가 거의 생기지 않는다.

## 1. 교사 어댑터 추가 (코드)
지금 `vapasr/data/semcommit_llm.py` 의 `KINDS` 는 `qwen3 · exaone35 · gptoss` 뿐이고, `LLM` 은 `AutoModelForCausalLM` 으로만 올린다.
1. 두 모델용 kind 를 추가한다(예: `exaone4`, `qwen38`). 각 모델의 chat template 을 읽고 **추론(thinking) 모드를 끄는** template kwargs 를 정한다(EXAONE-4.0 은 `enable_thinking` 인자를 받는 것으로 알려져 있다 — 템플릿에서 확인).
   `trust_remote_code` 는 내장 아키텍처면 False.
2. Qwen3.8-27B 가 CausalLM 이 아니면(ConditionalGeneration 등) 텍스트 전용으로 올리는 경로를 `LLM.model` 에 추가한다.
3. 테스트(`tests/test_semcommit_llm.py`, 가짜 모델): 새 kind 의 프롬프트 조립·라벨 채점 경로. 프롬프트 판은 `semcommit-prompt-v0.2+ab90c940` 그대로(바꾸면 골드 관문부터 다시).
4. **스모크**(GPU 1 장, 골드 ks-eval·ls-test 각 5 스트림): stageA·stageB·stageC 를 두 모델로 돌려 확인 —
   Stage A JSON 파싱 실패 없음, `inference_error` 0, 라벨 채점 행의 `tok_ok`/`type_tok_ok` 가 참(거짓이면 strict 등급에서 결측), SAFE/WAIT 분포가 한쪽으로 쏠리지 않음
   (gpt-oss-20b 는 영어 판정 97 % WAIT 로 탈락했었다 — 같은 실패를 여기서 거른다).

## 2. 교사 관문: 구성 고르기와 임계값
골드 v1 네 파트(ls-test · gs-test · ks-eval · ks-long)에서 후보 구성을 라벨링하고 골드와 대조한다. 스크립트가 추가 후보 채점 → 가지별 임계값 조정 → 등급(규칙 음성 포함) → 대조를 한 번에 한다.

```bash
cd /soundai/users/tskim/VAPKT
GOLD=/soundai/users/tskim/VAPKT-data/data/semcommit/gold/v1 GPU=0 PY=<env python> \
OUT=/soundai/users/tskim/VAPKT-data/data/semcommit/gate/g1-qwen \
A=/soundai/Model/<Qwen3.8-27B>:qwen38:qwen3.8-27b \
B="/soundai/Model/EXAONE-4.0-32B:exaone4:exaone4-32b /soundai/Model/<Qwen3.8-27B>:qwen38:qwen3.8-27b" \
C=/soundai/Model/<Qwen3.8-27B>:qwen38:qwen3.8-27b \
bash experiments/semcommit_teacher_gate.sh
```
- **구성**: Stage B 는 항상 두 모델(판정자 2). Stage A·C 는 두 모델 중 하나 — `g1-qwen`(A·C = Qwen3.8-27B)과 `g2-exaone`(A·C = EXAONE-4.0)를 모두 돌려 비교한다.
  언어별로 다른 구성을 골라도 된다(예: 영어 g1, 한국어 g2 — 그러면 본 라벨링도 언어별로 그 구성).
- **출력**: `OUT/gate.json`(파트별 후보 재현·A 정밀도/재현율·N 정밀도·B 중 실제 확정 + 조정 보고), `OUT/thresholds.json`(본 라벨링 입력), `OUT/thresholds.report.json`(가지별 dev/test).
- **통과 기준**(test 절반 기준, 8B 교사 v0.3.3 이 기준선):

  | 항목 | 기준 | 8B 기준선 |
  |---|---|---|
  | A 정밀도(언어별) | ≥ 0.90 | 영어 0.955 · 한국어 0.926 |
  | A 재현율(언어별) | 기준선 이상 | 영어 0.725 · 한국어 0.814 |
  | 켜진 가지마다 test 정밀도 | ≥ 0.85(dev 는 조정에서 0.90 보장) | — |
  | N 정밀도 | ≥ 0.95 | 0.99–1.00 |
  | 후보 재현 | 기준선 이상 | 0.79–0.94 |
  | Stage A 실패·B/C 오류 | 스트림의 1 % 미만 | — |

  큰 교사로 `other` 가지(구두점 없는 자리)가 켜지면 그 가지가 재현율을 올리는지, 구두점 없는 코퍼스에 A 를 줄 수 있는지 본다(§3 표의 “구두점 없음” 코퍼스에 중요).
- 결과를 [[output-semcommit-recipe-v0.3]] 의 표와 나란히 적고, 고른 구성과 `thresholds.json` 을 원자료 폴더에 남긴다(§6).

## 3. 코퍼스별 words.jsonl
라벨은 words.jsonl 단어 번호에 붙는다. 형식(`vapasr/data/semcommit_words.py` 머리 주석):
`{id, set, lang, K, duration_s, text, tokens, segments:[{path, offset_s, silence_before_s, dur_s, utt_id, raw_text}], words:[{i, text, a, b, end_time, seg, tags}], pnc_text?}` —
단어는 Qwen3-ASR 토큰 경계로 자르고 `end_time` 은 정렬 시각, `tags` ⊂ `filler · rep · unclear · punct_final · punct_comma`. **`punct_final` 이 A 의 주 출처**다.

| 코퍼스 | 언어 | 빌더 | 구두점(`punct_final`) 출처 | 상태 |
|---|---|---|---|---|
| LibriSpeech 960 h | EN | `experiments/semcommit_build_words.py --set librispeech-960` | LibriSpeech-PC(`--pnc-dir`) | 빌더 있음 |
| KsponSpeech 965 h | KO | `semcommit_build_words.py --set kspon-full` | Kspon 원 전사 | 빌더 있음 |
| NIKL 일상대화 | KO | **없음 — 작성** | 전사 확인 | |
| AI Hub 71631 자유대화 | KO | **없음 — 작성** | 전사 확인 | |
| Switchboard | EN | **없음 — 작성** | 전사 확인 | |
| otoSpeech · AMI IHM · 그 밖 | EN | **없음 — 작성** | 전사 확인 | |

- LibriSpeech·Kspon: forced-align manifest(`<manifests>/original/<set>/streams.jsonl` + `<manifests>/<set>/aligned-manifest.jsonl.gz`)의 mxc 위치를 확인하고,
  `--ls-root $MXC_LIBRISPEECH_DIR --kspon-root $MXC_KSPONSPEECH_DIR --tokenizer $MXC_QWEN_ASR_DIR --n 0`(전부)으로 만든다. 제외 통계(`--stats`)를 남긴다.
- **새 빌더**(대화체): 한 스트림 = 한 화자의 연속 발화(다른 화자와 겹치는 구간 제외), 8–30 s, 원 발화(IPU) 경계를 `segments` 로 보존(`seg_end` 후보가 된다).
  단어 시각은 forced alignment(Qwen3-ForcedAligner, `/soundai/Model/Qwen3-ForcedAligner-0.6B`; 대량 정렬 파이프라인 `experiments/speechlm_asr_align.py` 활용 검토),
  `punct_final`·filler 등 태그는 원 전사의 구두점·표지에서. 스트림 id 는 코퍼스 접두사로 **전 코퍼스에서 유일**하게(`ls-`·`ks-`·`nikl-`·`swb-` …). 학습 분할만 쓰고 평가용 분할은 따로 둔다.
  새 빌더마다 테스트와 표본 20 스트림 눈검사(단어 분할·시각·태그)를 한다. 참고 구현: `experiments/semcommit_build_gigaspeech.py`(GigaSpeech test 용 — 정렬부터 words 까지).
- **구두점 유무를 코퍼스마다 측정**한다: 단어 중 `punct_final` 비율, 스트림 끝 단어의 `punct_final` 비율. 구두점이 없거나 드문 코퍼스는
  (a) 관문에서 `other` 가지가 켜졌으면 그대로 라벨링, (b) 아니면 SEM 학습에서 빼거나 구두점 복원 모델로 `punct_final` 을 만든 뒤 그 정밀도를 골드로 다시 잰다 — 사용자와 정한다.
- 대화체 영어의 골드는 지금 GigaSpeech(대리)뿐이다. Switchboard 를 크게 쓰면 그 평가 분할로 gold v2(같은 절차, `semcommit_gold.py packet/merge`)를 권한다.

## 4. 본 라벨링 (shard 단위)
- **shard**: 코퍼스별 words.jsonl 을 줄 단위로 나눈다(예: 2,000 스트림). `grade` 는 입력 지문의 words 해시를 확인하므로 **A/B/C/grade 를 shard 별로** 돌린다.
- **한 프로세스 = 한 모델·GPU 한 장**. shard 마다 stageA → stageB(EXAONE-4.0) → stageB(Qwen3.8-27B) → stageC 순서, 노드당 GPU 8 장에 서로 다른 shard 를 배정한다
  (`--gpu 0`…`7` — 프로세스를 `CUDA_VISIBLE_DEVICES` 로 그 GPU 에 고정한다).
  같은 모델로 여러 shard 를 이어 돌리게 묶으면 모델 적재 횟수가 준다. 멈추면 같은 명령을 다시 돌리면 끝난 행은 건너뛴다(`inference_error` 행만 재시도, 남으면 exit 2).
- **먼저 처리량 시험**: shard 1 개로 단계별 초당 행 수를 재고 전체 소요를 추정해 사용자에게 알린 뒤 전량을 제출한다(이 파이프라인은 HF transformers 기반이다 — vLLM 백엔드는 없다).
- 경로: `/soundai/users/tskim/VAPKT-data/data/semcommit/labels/v0.3.3-<구성>/<코퍼스>/shard-00000/{A,B.exaone4-32b,B.qwen3.8-27b,C,labels}.jsonl` (+ `*.fingerprint.json`, `labels.stats.json`).

```bash
W=<shard words.jsonl>; O=<shard 출력 디렉터리>; T="<env python> experiments/semcommit_teacher.py"; X="--extra-candidates last,seg_end,punct_final"
$T stageA --words $W --model $QWEN --kind qwen38 --judge-name qwen3.8-27b --out $O/A.jsonl --gpu 0 --batch-size 8          # 구성 g1 기준(A = Qwen)
$T stageB --words $W --stageA $O/A.jsonl --model $EXAONE --kind exaone4 --judge-name exaone4-32b --out $O/B.exaone4-32b.jsonl --gpu 0 --batch-size 16 $X
$T stageB --words $W --stageA $O/A.jsonl --model $QWEN --kind qwen38 --judge-name qwen3.8-27b --out $O/B.qwen3.8-27b.jsonl --gpu 0 --batch-size 16 $X
$T stageC --words $W --stageA $O/A.jsonl --model $QWEN --kind qwen38 --judge-name qwen3.8-27b --out $O/C.jsonl --gpu 0 --batch-size 16 $X
$T grade --recipe v0.3 --thresholds <관문 OUT>/thresholds.json --words $W --stageA $O/A.jsonl \
  --stageB $O/B.exaone4-32b.jsonl $O/B.qwen3.8-27b.jsonl --stageC $O/C.jsonl \
  --judges-en exaone4-32b,qwen3.8-27b --judges-ko exaone4-32b,qwen3.8-27b --out $O/labels.jsonl --stats $O/labels.stats.json
```
`grade` 기본값이 레시피 v0.3.3 이다(`--extra-candidates last,seg_end,punct_final`, `--neg-rules reply_prefix,conn_final,conn_mid`). 관문과 **같은 판정자 이름·같은 모델 경로**를 쓴다.

## 5. 점검
- **shard·코퍼스별 통계**(`labels.stats.json` 합산): 스트림 수·라벨 없는 스트림, 100 단어당 A·B·N, 스트림 끝 A 비율, 규칙 음성 수, `why` 분포, 오류·결측.
  rack4 v0.3.3 참고값: ls-train 100 단어당 A 4.3 · N 0.02, ks-train A 12.6 · N 12.6(규칙 음성이 대부분). 코퍼스 성격에 따라 달라도 되지만 0 에 가깝거나 크게 튀면 원인을 본다(구두점 없음·태그 누락·정렬 실패).
- **눈검사**: 코퍼스마다 20 스트림을 골라 A·N 자리를 읽어 본다(특히 새 빌더 코퍼스).
- 관문 파트(골드)의 점수는 §2 의 `gate.json` 이 그대로 기준이다 — 본 라벨링 뒤 다시 잴 필요는 없다(같은 모델·임계값).

## 6. 인계와 기록
- 학습: `experiments/semcommit_train.py` 에 코퍼스·shard 별 `--train-words`/`--train-labels` 쌍을 준다(`<SEM_END>` 만, `--turn-end` 끔). 학습 뒤 save/load parity 를 확인하고,
  평가는 골드 대조(`semcommit_eval.py` → `semcommit_gold.py score-eval`, 언어별 bias 운영점 — 8B 결과에서 최고 F1 은 영어 bias 0~+2, 한국어 −2).
- 기록: 위키 output 페이지(고른 구성·관문 결과·코퍼스별 통계·구두점 측정), `raw/sources/experiments/<날짜>-semcommit-main-labeling/`(gate.json·thresholds·stats·스크립트, 대용량 라벨 제외, `SHA256SUMS`), 로그 샤드, 커밋·푸시.

## 알려진 한계
- Qwen3.8-27B 의 이름·아키텍처는 검증 전이다. 두 모델 모두 이 파이프라인에서 돌려 본 적이 없다 — §1 스모크와 §2 관문이 그 검증이다.
- 규칙 음성은 한국어 형태론과 대답어 목록에 기댄다. 영어 대화체에는 대응 규칙이 없고(영어 오판은 후보 아닌 자리에서 많이 난다), 한국어 축약 -ㄴ데(`한데`·`인데`)·다시 말하기는 아직 규칙 밖이다.
  새 규칙은 먼저 골드로 재고(`raw/sources/experiments/2026-09-24-semcommit-recipe-v0.3/analysis/conn_mid.py` 방식) 넣는다.
- 골드는 Claude 가 만든 것이고 대화체 영어는 대리(GigaSpeech)다.

## 완료 조건
- [ ] 코드·골드 묶음·두 모델·LibriSpeech-PC 가 mxc 에 있고 체크섬 확인
- [ ] 두 모델용 kind 추가·테스트·스모크 통과
- [ ] 관문 g1·g2 결과와 고른 구성, `thresholds.json` 기록(통과 기준 충족)
- [ ] 코퍼스별 words.jsonl(새 빌더 포함)과 구두점 측정표, 구두점 없는 코퍼스 처리 결정
- [ ] 처리량 시험 후 전량 라벨링 완료(오류·결측 1 % 미만)
- [ ] 코퍼스별 통계·눈검사, 위키·원자료 기록, 커밋·푸시

## 진행 기록
- 2026-09-24: 생성. 판정 LLM 을 EXAONE-4.0 · Qwen3.8-27B 로 한정(사용자 결정).

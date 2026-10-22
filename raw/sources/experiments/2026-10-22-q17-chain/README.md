# Qwen3-ASR 1.7B 체인 — 0.6B 와 같은 단계·레시피 (2026-10-22, mxc)

사용자: "Qwen asr 1.7B 를 이용해서도 학습시키고 싶어 학습 스크립트를 준비해줘" → "Stage 1과 2 도 적합하게 수정되어 있는지 검토해줘"

## 배경
- 1.7B(`/soundai/Model/Qwen3-ASR-1.7B`) 계열은 Stage 1 까지 있었다.
  - `semcommit-q17-s0`: adapter 워밍업. q06-s0 와 인자가 같다(Qwen 경로만 다르다).
  - `semcommit-q17-s1m-r3`: 영·한, 인코더 [56,3], δ1–8, 3 epoch.
  - `q17-s1m3-d10`: 2026-10-08 에 r3 에서 영·한·독 δ4–10 1 epoch 로 시작했다(job 82730). checkpoint-1000(5,391 step 중)까지 저장한 뒤 21:28 에 멈췄다(사용자 취소).
- 2026-10-09 사용자 결정으로 이후 단계는 0.6B 로 갔다(실시간성 우선, 영어는 1.7B 와 같거나 낫고 한국어 0.6–1.6 %p 뒤짐 — [[2026-10-09-decision-next-stage-from-q06]]).
- 0.6B 체인과 단계별 순수 step 시간(s_per_step 중앙값 × step 수, apex 2 노드):
  - `q06-s0` → `q06-s1m3-d10` 10.6 h → `q06-s2-ast` 6.8 h → `q06-s3-ast` 6.8 h → `q06-s4-ast` 4.1 h
  - → 대화 `stage3-st3-p11-revive` 1.1 h(ck3000 채택) → `stage3-st4-m2a`(V1) / `stage3-v3-p1-tel-xlw`(R1x) 1.0 h

## 1.7B 체인 (`slurm/submit-q17-chain.sh`, 단계마다 앞 단계 성공 뒤 — sbatch afterok)

| 단계 | run | init | 레시피 | 어림 시간 |
|---|---|---|---|---|
| 1 | `q17-s1m3-d10-s0` | q17-s0/final | `submit-q17-s1m3-d10-s0.sh` = q06-s1m3-d10 판(영·한·독 δ4–10 3 epoch, worker 4) | 약 13 h(실측) |
| 2 | `q17-s2-ast` | q17-s1m3-d10-s0/final | `submit-q17-s2-ast.sh` = q06 판(번역 모드 0.5, 2 epoch) | 약 8–9 h |
| 3 | `q17-s3-ast` | q17-s2-ast/final | `submit-q17-s3-ast.sh` = q06 판(번역 모드 0.8 + 영어 음량 증강) | 약 8–9 h |
| 4 | `q17-s4-ast` | q17-s3-ast/final | `submit-q17-s4-ast.sh` = q06 판(+ CV4 영어 낭독·번역) | 약 5–9 h |
| 5 | `q17-st3-p11-revive` | q17-s4-ast/final | 대화 Stage 3 — 0.6B p11 제출 명령 그대로(4,000 step) | 약 1.3–2.5 h |
| 6 | `q17-v3-p1-tel-xlw` | q17-st3-p11-revive/checkpoint-3000 | V3 R1x — 0.6B R1x 제출 명령 그대로(2,500 step) | 약 1.2–2.2 h |

- 프로젝트 단계 번호(2026-10-14 결정): 체인 1 = Stage 1 ASR, 체인 2–4 = Stage 2 AST 1–3 차, 체인 5 = Stage 3(2 인 대화), 체인 6 = Stage 4(대화 번역, V3).
  - run 이름의 s2·s3·s4-ast 는 0.6B 때 Stage 2(번역)를 세 번 이어 학습한 차수 번호다. 프로젝트 Stage 3·4 가 아니다(사용자 질문 "Stage 4 까지 갔다가 다시 stage 3-4 를 하는 이유" — 2026-10-23 표기 정리).
- 시간(2026-10-23 다시 어림): 합 약 37–46 h 에 단계마다 시작·재시작 시간이 더해진다. 6 단계는 10-24 낮–밤에 끝날 것으로 본다.
  - 1 단계는 1.7B 실측이다. job 94809 의 step 1–3,630 중앙값이 2.86 s/step 이다.
  - 같은 step 구간의 0.6B(82738·82844)는 1.22 s/step 이었다. 1.7B 가 2.3 배 느리다.
  - 2–6 단계 0.6B step 시간(작업 로그 `s_per_step` 중앙): s2 1.66 · s3 1.58 · s4 1.03 · p11 0.96 · R1x 1.39 s. 여기에 1.7B 배율 1.2–2.3 을 곱했다.
    - 배율 1.2 는 데이터 읽기가 병목일 때다(1 단계 전체 중앙 비 2.86/2.33).
    - 배율 2.3 은 GPU 계산이 병목일 때다(같은 step 구간 비).
- 어림을 두 번 틀렸다.
  - 처음(속도비 2.4, 합 38 h): 0.6B 시간으로 `results.json` 의 학습 시간을 썼다. 이 값은 마지막으로 이어 한 구간만 센다(q06-s1m3-d10 의 "4.2 h" 는 checkpoint-10000 뒤 구간뿐, 실제는 실패를 포함해 15 h).
  - 10-22 저녁(속도 같음, 약 30 h): 10-08 의 한 구간만 보고 정했다. 그 구간에서는 1.7B(82730)와 0.6B(82738)가 둘 다 1.1–1.7 s/step 이었다.
  - 이 결론에 쓴 0.6B 단계별 중앙값도 `results.json` 의 log_history 에서 뽑았는데, 일부 구간만 남아 있었다. 지금 값은 작업 로그 전체에서 다시 뽑았다.
  - step 시간은 데이터 읽기 상태에 따라 1.2–2.9 s 로 크게 흔들린다.
- `--s1 resume`: Stage 1 을 10-08 run `q17-s1m3-d10` 을 checkpoint-1000 에서 이어 하는 판으로 바꾼다(`submit-q17-s1m3-d10.sh`, 남은 4,391 step 약 2.9 h). 2 단계 init 도 그 final 로 넘어간다. 0.6B 와 Stage 1 이 다르다는 점은 아래 검토 3 에 적었다.
- 5 단계는 평가 없이 0.6B 와 같은 위치(checkpoint-3000)를 6 단계 init 으로 쓴다. 0.6B 는 p11 ck3000 을 평가로 골랐다.
- 저장 공간: 1.7B 체크포인트 하나는 27 GB 다(0.6B 14 GB). 체인 전체로 체크포인트 약 70 개, 약 2 TB 다. `/soundai` 는 5.0 PB 라 문제없고, 아무것도 지우지 않는다.

## Stage 1·2 검토 (2026-10-22 20시)

### 1. 10-08 의 1.7B Stage 1 은 노드 메모리 부족으로 세 번 죽었다 → worker 4
- job 82730 의 `.err` 에 다음 기록이 있다(DataLoader worker 강제 종료):
  - 18:41 `Detected 2 oom_kill events`
  - 20:08 `Detected 1 oom_kill event`
- 원인은 0.6B 82738 과 같다. 독일어 1.3M 항목이 더해져 worker 마다 복사되는 데이터셋이 커졌다. 0.6B 는 `--num-workers 4` 로 고쳤지만 1.7B 스크립트에는 빠져 있었다(1.7B 는 기본 8).
- 두 Stage 1 스크립트 모두 worker 4 로 맞췄다. worker 수는 지문에 들어가지 않는다.

### 2. 1.7B 체크포인트를 이어 받지 못하는 문제 → 고침
- 1.7B 는 `model-00001-of-00002.safetensors` 와 `model-00002-of-00002.safetensors` 두 조각, 그리고 index 로 저장된다.
- `semcommit_train.py`·`stage3_train.py` 의 재개 탐색은 `model.safetensors` 한 파일만 보았다. 그래서 1.7B 체크포인트를 미완성으로 여겼다.
  - Stage 1 이어 하기는 checkpoint-1000 을 못 찾고 init 부터 다시 시작했을 것이다.
  - 모든 1.7B 단계에서 requeue(실패 재시작·선점)마다 처음부터 다시 했을 것이다.
- 공용 `vapasr.hf.stage.last_complete_checkpoint` 로 바꿨다. 한 파일이거나, index 가 가리키는 조각이 모두 있으면 완성으로 본다.
  - 테스트: `tests/test_resume_checkpoint.py`
- 다른 곳은 조각 저장을 이미 처리한다.
  - 학습 끝 final 재로드 parity 는 r3 에서 PARITY True 였다.
  - 평가 로드(`from_pretrained`)와 로컬 복사(`stage_dir` 의 `*.safetensors.index.json`)도 조각 저장을 다룬다.
  - `avg_checkpoints.py` 는 safetensors 파일마다 평균한다.

### 3. Stage 1 레시피가 0.6B 와 달랐다 → 기본을 0.6B 와 같은 run 으로
- 10-08 판(`q17-s1m3-d10`):
  - r3 에서 시작한다. r3 은 영·한 δ1–8 3 epoch 다.
  - 그 뒤 영·한·독 δ4–10 을 1 epoch 만 한다.
  - 독일어와 δ4–10 step 은 0.6B Stage 1 의 1/3 이다. epoch 당 독일어 1,241 / 5,391 step.
- 0.6B 와 크기만 다르게 비교하려고 기본 Stage 1 을 새 run `q17-s1m3-d10-s0` 로 바꿨다.
  - q17-s0 에서 q06-s1m3-d10 과 같은 인자로 3 epoch 를 돈다. 16,173 step, 약 10.6 h.
  - 이어 하기 판(약 2.9 h)보다 약 7.7 h 더 걸린다.
- 10-08 폴더(`semcommit-q17-s1m3-d10`)는 그대로 둔다.
  - 같은 run 이름으로 init 만 바꾸면 안 된다. 지문에 init 이 없어서 checkpoint-1000(r3 판)을 이어 받아 버린다. 그래서 run 이름을 새로 지었다.
- 인코더 오른쪽 문맥:
  - mirror 판은 q06 처럼 `--encoder-right-context 3` 을 준다. q17-s0 은 [56,0] 이고, q06-s0 도 같았다.
  - 이어 하기 판은 이 옵션을 주지 않는다. init r3 과 checkpoint-1000 config 가 이미 [56,3] 이다. 옵션을 주면 지문에 키가 생겨 재개가 거부된다.

### 4. Stage 2 는 0.6B 와 같다
- `submit-q17-s2-ast.sh` 는 q06-s2-ast 와 init·run 이름 외에 같다. 0.6B 의 실제 학습 인자(`results.json`)와도 같다:
  - worker 4, [56,3] 고정
  - 번역 모드 0.5, δ4–10, 2 epoch, lr 6e-5, warmup 500
  - 같은 SNAP·ASR·DE 목록과 잡음 뱅크
- 번역 타깃 묶음(`ast-mono-v1/targets-fix-gen-v3-bf16/pack/`)은 토큰 id(`ids.npy` int32)다.
  - 0.6B 와 1.7B 의 tokenizer 어휘 지문(`vocab_sha256`, 추가 토큰 포함)이 같다(q06-s1m3-d10 final 대 q17 checkpoint-1000). 그래서 그대로 쓸 수 있다.
  - words·labels 파일 지문도 같다.
- GPU 메모리: 배치는 토큰 예산(16,384)으로 묶인다. 같은 예산의 1.7B Stage 1 이 GPU 메모리 문제 없이 돌았다(실패는 모두 노드 RAM 부족).

### 5. 이어 하기 판의 지문 확인 (CPU dry-run)
- 방법: 제출 래퍼가 만드는 인자 그대로 `semcommit_train.py --dry-run` 을 돌려 checkpoint-1000 지문과 대조했다.
  - 실행 위치: mxc 호스트
  - 산출: `VAPKT-data/tmp/q17-s1-fpcheck-20261022/`
- 결과(20:59): **지문 차이 0** — 지금 코드로 이어 하기가 받아들여진다.
  - 네 셋(en·ko·short_ko·de)의 시퀀스 불변식 검사(64 항목 × δ4–10)도 불일치 0 이다.
  - 체인은 기본 mirror 로 제출돼 이 판은 쓰지 않았다. 나중에 빠른 판이 필요하면 `--s1 resume` 을 그대로 쓸 수 있다.

## 제출 기록
- 2026-10-22 20:42, 사용자: `bash slurm/submit-q17-chain.sh`(기본 mirror)
- job 번호(단계마다 앞 job 성공 뒤 시작):
  - 1 = 94809(`q17-s1m3-d10-s0`)
  - 2 = 94810, 3 = 94811, 4 = 94812
  - 5 = 94813, 6 = 94814
- Stage 1 시작 점검(94809, 20:45 시작, hpc-36·42, 21:29 확인):
  - init: `q17-s0/final`
  - 인코더 att_context [56,0] → [56,3], 학습 중 고정
  - worker 4: 명령줄의 `--num-workers 8 --num-workers 4` 중 뒤 값이 이긴다
  - 학습 파라미터 2,336M
  - epoch 당 5,391 step, world 16 → 16,173 step
  - 첫 손실은 0.6B 같은 판(82738)과 비슷하게 내려간다:
    - step 10: 10.46 대 9.78
    - step 30: 4.21 대 4.26
    - step 100: 1.09 대 1.27
  - step 시간(step 140–160): 1.49–1.55 s. 0.6B 초반과 같은 수준이다.
  - 실패·메모리 부족 없음. 알려진 71631 항목 하나만 이웃 항목으로 바뀐다(0.6B 와 같다).

## 평가 (SLURM, `slurm/submit-evals.sh`, 2026-10-23)
- 사용자: "평가 작업도 SLURM 통해 수행할 수 있도록 명령어를 알려줘" · "AMI_IHM 이랑 AMI_SDM 에 대해서도 평가를 수행하고 싶어"
- 묶음마다 작업 목록을 만들어 `slurm/v3-eval-node.sbatch`(apex 1 노드 8 GPU, 시간 한도 24 h)에 낸다.

| 묶음 | 내용 | 시작 조건 |
|---|---|---|
| `q17-s1` | Stage 1 final 단일 발화 ASR. 0.6B 10-09 평가와 같은 명령이다: 영·한 vpaa-kspon(GPU 0·1), 독일어 yodas-de(GPU 2·3), δ4·6·8·10 + final | job 94809 성공 뒤 |
| `q17-v3` | V3 R1x 의 ck1500·ck2000·final·avg3 × 9 작업(0.6B R1x 와 같은 `v3_eval_task.sh`). final·avg3 은 AMI IHM-mix·SDM 을 4 몫씩 더 돈다. 평균은 GPU 7 첫 작업이 만든다 | job 94814 성공 뒤 |
| `r1x-ami` | 0.6B R1x final·avg3 의 AMI(모델마다 4 몫) — 1.7B 와 비교할 0.6B 값 | 바로 |

- 제출(사용자, 2026-10-23 00:41):
  - `r1x-ami` = job 94839
  - `q17-s1` = job 94840(94809 뒤)
  - `q17-v3` = job 94841(94814 뒤, 45 작업)

- AMI(`v3_eval_task.sh ami <run> <대상> r/w`):
  - 명령은 V2 `v2_eval_extra.sh` 12 단계와 같다: `ami-bench-v1` 공식 dev 18·eval 16 회의, 창 전부, δ4·10, 4 슬롯 선언.
  - 창을 w 몫으로 나눠 돈다. 이어 하기가 되고, 마지막으로 끝난 몫이 합치기·오류 분해·`ami_report.py` 를 돈다.
  - 스모크(호스트 GPU, R1x final, 창 2 개 몫 0/4)에서 디코드와 보고서가 나왔다.
- V1 ck2000 은 10-21 에 AMI 를 쟀다. 공식 eval δ10 기준:
  - IHM-mix: cp 54.0 · agn 29.5 · 귀속 24.6
  - SDM: cp 60.9 · agn 36.2 · 귀속 24.7
  - 0.6B 단일 GPU 로 4.5–5.7 h 걸렸다.
- `stage_dir`(평가가 체크포인트를 노드 로컬로 복사)를 고쳤다.
  - 문제: 같은 노드의 여러 일꾼이 같은 체크포인트를 동시에 부르면 공용 임시 파일(`<파일>.tmp`)을 서로 잘라 쓰는 구조였다. 로컬 재현에서는 문제가 없었다.
  - 이제 잠금(`.stage.lock`)으로 한 번만 복사하고 임시 파일은 프로세스마다 다른 이름이다. AMI 몫처럼 같은 체크포인트를 여러 GPU 가 동시에 부를 때 필요하다.

## 제출 (사용자, 스케줄러 셸)
```
(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-chain.sh --dry-run)              # 미리 보기
(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-chain.sh)                        # 1–6 단계(Stage 1 = 0.6B 와 같은 레시피)
(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-chain.sh --s1 resume)            # Stage 1 을 10-08 checkpoint-1000 이어 하기로(빠름, 레시피 다름)
(cd /soundai/users/tskim/VAPKT && bash slurm/submit-evals.sh r1x-ami)                    # 0.6B R1x AMI — 바로
(cd /soundai/users/tskim/VAPKT && bash slurm/submit-evals.sh q17-s1)                     # 1.7B Stage 1 평가 — job 94809 성공 뒤
(cd /soundai/users/tskim/VAPKT && bash slurm/submit-evals.sh q17-v3)                     # 1.7B V3 평가 + AMI — job 94814 성공 뒤
```

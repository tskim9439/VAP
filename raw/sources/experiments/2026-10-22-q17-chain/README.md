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
| 1 | `q17-s1m3-d10-s0` | q17-s0/final | `submit-q17-s1m3-d10-s0.sh` = q06-s1m3-d10 판(영·한·독 δ4–10 3 epoch, worker 4) | 10.6 h |
| 2 | `q17-s2-ast` | q17-s1m3-d10-s0/final | `submit-q17-s2-ast.sh` = q06 판(번역 모드 0.5, 2 epoch) | 6.8 h |
| 3 | `q17-s3-ast` | q17-s2-ast/final | `submit-q17-s3-ast.sh` = q06 판(번역 모드 0.8 + 영어 음량 증강) | 6.8 h |
| 4 | `q17-s4-ast` | q17-s3-ast/final | `submit-q17-s4-ast.sh` = q06 판(+ CV4 영어 낭독·번역) | 4.1 h |
| 5 | `q17-st3-p11-revive` | q17-s4-ast/final | 대화 Stage 3 — 0.6B p11 제출 명령 그대로(4,000 step) | 1.1 h |
| 6 | `q17-v3-p1-tel-xlw` | q17-st3-p11-revive/checkpoint-3000 | V3 R1x — 0.6B R1x 제출 명령 그대로(2,500 step) | 1.0 h |

- 시간은 0.6B 같은 단계의 순수 step 시간이다. 합 약 30 h 에 단계마다 시작 약 15 분과 재시작 시간이 더해진다.
  - 1.7B 도 같은 값으로 잡은 근거: 10-08 에 같은 데이터·같은 시각으로 돈 1.7B(82730)와 0.6B(82738)의 step 시간이 같았다. step 110–1,110 에서 1.11–1.69 s 대 1.12–1.49 s 다. 데이터 읽기가 병목이다.
  - 4–6 단계는 0.6B 도 데이터 읽기에 덜 묶였다(0.96–1.40 s/step). 1.7B 는 이 단계들에서 더 걸릴 수 있다(1.7B 실측 없음).
  - 처음 어림(1.7B 속도비 2.4, 합 38 h)은 틀렸다. 0.6B 쪽 시간으로 `results.json` 의 학습 시간을 썼는데, 이 값은 마지막으로 이어 한 구간만 센다. 예를 들어 q06-s1m3-d10 의 "4.2 h" 는 checkpoint-10000 뒤 구간뿐이다. 실제로는 실패 세 번을 포함해 15 h 걸렸다(10-08 17:17 → 10-09 08:19).
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
- 결과: (진행 중 — 아래에 기록)

## 제출 기록
- 2026-10-22 20:42, 사용자: `bash slurm/submit-q17-chain.sh`(기본 mirror)
- job 번호(단계마다 앞 job 성공 뒤 시작):
  - 1 = 94809(`q17-s1m3-d10-s0`)
  - 2 = 94810, 3 = 94811, 4 = 94812
  - 5 = 94813, 6 = 94814

## 제출 (사용자, 스케줄러 셸)
```
(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-chain.sh --dry-run)              # 미리 보기
(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-chain.sh)                        # 1–6 단계(Stage 1 = 0.6B 와 같은 레시피)
(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-chain.sh --s1 resume)            # Stage 1 을 10-08 checkpoint-1000 이어 하기로(빠름, 레시피 다름)
```

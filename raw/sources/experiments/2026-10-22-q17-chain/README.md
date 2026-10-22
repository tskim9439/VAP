# Qwen3-ASR 1.7B 체인 — 0.6B 와 같은 단계·레시피 (2026-10-22, mxc)

사용자: "Qwen asr 1.7B 를 이용해서도 학습시키고 싶어 학습 스크립트를 준비해줘"

## 배경
- 1.7B(`/soundai/Model/Qwen3-ASR-1.7B`) 계열은 Stage 1 까지 있었다: `semcommit-q17-s1m-r3`(영·한, 인코더 [56,3], 3 epoch, 8.1 h). 2026-10-08 에 영·한·독 δ4–10 이어 학습 `q17-s1m3-d10` 을 시작했다가 checkpoint-1000(5,391 step 중)에서 멈췄다.
- 2026-10-09 사용자 결정으로 이후 단계는 0.6B 로 갔다(실시간성 우선, 영어는 1.7B 와 같거나 낫고 한국어 0.6–1.6 %p 뒤짐 — [[2026-10-09-decision-next-stage-from-q06]]).
- 0.6B 체인: `q06-s0` → `q06-s1m3-d10`(4.2 h) → `q06-s2-ast`(6.0 h) → `q06-s3-ast`(4.9 h) → `q06-s4-ast` → 대화 `stage3-st3-p11-revive`(1.1 h, ck3000 채택) → `stage3-st4-m2a`(V1) / `stage3-v3-p1-tel-xlw`(R1x, 0.9 h). 시간은 apex 2 노드.

## 1.7B 체인 (`slurm/submit-q17-chain.sh`, 단계마다 앞 단계 성공 뒤 — sbatch afterok)

| 단계 | run | init | 레시피 | 어림 시간 |
|---|---|---|---|---|
| 1 | `q17-s1m3-d10` | q17-s1m-r3/final(이어 하기 ← checkpoint-1000) | `slurm/submit-q17-s1m3-d10.sh`(2026-10-08 그대로, 영·한·독 δ4–10 1 epoch) | 약 3 h |
| 2 | `q17-s2-ast` | q17-s1m3-d10/final | `submit-q17-s2-ast.sh` = q06 판(번역 모드 0.5, 2 epoch) | 약 10 h |
| 3 | `q17-s3-ast` | q17-s2-ast/final | `submit-q17-s3-ast.sh` = q06 판(번역 모드 0.8 + 영어 음량 증강) | 약 10 h |
| 4 | `q17-s4-ast` | q17-s3-ast/final | `submit-q17-s4-ast.sh` = q06 판(+ CV4 영어 낭독·번역) | 약 10 h |
| 5 | `q17-st3-p11-revive` | q17-s4-ast/final | 대화 Stage 3 — 0.6B p11 제출 명령 그대로(4,000 step) | 약 2.7 h |
| 6 | `q17-v3-p1-tel-xlw` | q17-st3-p11-revive/checkpoint-3000 | V3 R1x — 0.6B R1x 제출 명령 그대로(2,500 step) | 약 2.2 h |

- 시간은 0.6B 단계 시간 × 1.7B 속도비(q17-s1m-r3 2.44 s/step 대 q06-s1m3-d10 0.93 s/step ≈ 2.4)로 어림했다. 합은 약 38 h 다.
- 0.6B 와 다른 점:
  - Stage 1 은 0.6B 가 q06-s0 에서 영·한·독 3 epoch 였고, 1.7B 는 영·한 3 epoch(q17-s1m-r3) 뒤 영·한·독 1 epoch 이다(2026-10-08 계획).
  - 5 단계는 평가 없이 0.6B 와 같은 위치(checkpoint-3000)를 6 단계 init 으로 쓴다. 0.6B 는 p11 ck3000 을 평가로 골랐다.
- 확인한 것:
  - q17 Stage 2–4 래퍼는 q06 판과 머리 설명·init·run 이름 외에 같다.
  - 체인 dry-run(mxc)에서 6 단계 sbatch 가 모두 나온다. 5·6 단계 인자는 0.6B 원래 p11·R1x 명령과 init·run·의존성·제외 노드 외에 같다.
  - checkpoint-1000 의 encoder 오른쪽 문맥 3·δ4–10·Qwen3-ASR-1.7B 는 0.6B 와 같은 설정이다.
  - 모델 코드는 헤드·어댑터 크기를 config hidden 에서 잡는다(1.7B = 2048).
- 위험: Stage 1 이어 하기는 체크포인트 지문(데이터 파일 해시·설정)이 지금과 다르면 거부된다. 지문 계산 코드는 2026-10-08 run 때와 같다(aa64f73 은 작업 트리 코드를 기록만 했다). 거부되면 1 단계 job 이 바로 실패하고, 뒤 단계는 시작하지 않는다(DependencyNeverSatisfied — 취소는 사용자).
- 코드 변경:
  - `submit-stage3-train.sh --afterok JOBID`(init 이 아직 없어도 체인 제출)
  - `submit-semcommit-train-apex.sh --exclude`(기본 제외에 더함)
  - 둘 다 안 쓰면 명령은 그대로다. 체인은 기본으로 hpc-50·51·101·102 를 더 뺀다(V3 R1·R2 와 같다).
- 6 단계 뒤 평가(같은 큐 또는 SLURM 평가 노드)와 비교는 0.6B R1x final 과 한다. 레벨 증강(C2b) 결과가 좋으면 6 단계만 `--from 6` 으로 tel-v2 레시피로 다시 낼 수 있다.

## 제출 (사용자, 스케줄러 셸)
```
(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-chain.sh --dry-run)    # 미리 보기
(cd /soundai/users/tskim/VAPKT && bash slurm/submit-q17-chain.sh)              # 1–6 단계 제출
```

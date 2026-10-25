---
type: status
status: active
created: 2026-09-03
updated: 2026-10-25
summary: 2026-10-25 1.7B 회의+4 슬롯 AMI 창 cp −5~−8, SDM 53.5 오류 세분, 화자 분리 통합 E0 39.3·E1 SDM DER 23.0·E2 학습 중·E3 준비
---

# 상태

마지막 갱신: 2026-10-25 (아래 기존 볼트 상태는 2026-09-07의 기록)

## 2026-10-21 V3 — 화자 귀속 보정

- 사용자 요청: TAXI 에서 VibeVoice 보다 앞서도록 화자 귀속을 완화하는 방법을 조사해 V3 로 적용하고, 기존 방식의 문제도 고친다 → [[output-v3-attribution-plan-20261021]]
- 진단: TAXI 의 VibeVoice 대비 차이는 전부 화자 귀속이다(V1 받아쓰기 18.4 대 VibeVoice 20.1, 귀속 몫 8.0 대 0.4). 오류의 89 % 가 턴 시작에서 놓친 교대 태그다.
- 원인: V1 은 교차 언어 TAXI 에서 <EOT> 를 하나도 내지 않는다. 학습 문맥에만 있는 감독 없는 합성 <EOT> 뒤에서 교대를 배웠다.
- V3-p1 구현 완료(기본 끔, 로컬 시험 통과): <EOT> 문맥 제거(C5), 교대·머무름 가중(C1), 2 회선 전화 교차 언어 합성(C2), dev 튜닝 디코드 bias(C3), 비-TAXI 전화 dev·턴 결정 지표(C4). TAXI 채점에 cp·agn·귀속 몫 분해를 더했다.
- V2 최종: 받아쓰기는 크게 올랐고(회의 agn −6–9, AMI 공식 eval −6–8 cp) 화자 귀속은 그대로다. 탭 뺀 대조(notap)로 TAXI 하락이 탭 탓이 아님을 확인했다 → [[2026-10-20-v2-multispeaker]]
- V3 진행: 검토 반영·mxc 본 checkout 반영(md5 대조)·mxc 시험 168 통과 → tel 셋·dev 빌드와 R2 dry-run·smoke 진행 중 → R1 `v3-p1`·R2 `v3-p1-nodata` 제출 명령(사용자 제출)
- V3 학습 결과(R1 `v3-p1` 94261, R2 `v3-p1-nodata` 94262, 둘 다 PARITY OK): TAXI 세션 통째 δ10 ck1500 에서 R2 cp 21.6 · agn 16.7 · 귀속 몫 4.9(V1 26.4 · 18.4 · 8.0). VibeVoice 스트리밍(20.5)과의 차이가 5.9 → 1.1 pt 로 줄었다. 체크포인트 선택·회귀 가드는 평가 큐가 끝난 뒤 dev 로 한다 → [[2026-10-21-v3-attribution]]
- R2 디코드 bias(C3): tune 반쪽에서 `wi=0.5` 를 골랐지만 check 반쪽에서 태그를 하나도 바꾸지 않았다(출력 동일). 사전 규칙대로 채택하지 않고 R2 는 bias 0 이다. 교대 결정의 여유가 커서 작은 bias 로는 움직이지 않는다.
- R4 구현(사용자 제안): 교차 언어 창도 먼저 말한 화자를 `<SPK_A>` 로, prefix 에는 두 언어만 적는다(`--xl-slot-order arrival`). 평가는 창마다 슬롯 ↔ 언어 짝을 cpWER 식으로 정한다. 시험 통과, mxc 반영 뒤 점검(`v3_check.sh r4`)을 마치면 제출 명령을 드린다. 온셋 분류(R2 ck1500·V1 ck2000)는 진행 중이다.
- 온셋 분류(TAXI 통째 δ10): 앞 턴이 맞았던 교대 중 놓침 V1 7.8 % → R2 4.0 %. 남은 놓침은 대부분 번역(`<TRANS>`) 뒤 같은 슬롯으로 받아쓰기를 다시 여는 RESUME 이다. 두 모델 모두 TAXI 에서 `<EOT>` 를 내지 않는다.
- **같은 언어 대화 가드 실패(R1·R2 모든 체크포인트)**: Stage 3 7 코퍼스 δ10 agn 18.6 → 21.7–26.4. 원인은 교대 태그 과잉이다(aihub71631 태그 3배, `<SPK_A> 어 <SPK_B> 어 …` 핑퐁과 강제 진행). 현재 코드로 V1 을 다시 재면 저장본과 400/400 창이 같아 디코드 코드는 원인이 아니다. 교대 가중(C1)을 같은 언어 셋에도 건 탓으로 본다.
- 사용자 결정 A: R2x `v3-p1-xlw`(교대 가중을 교차 언어 합성 셋에만, job 94525)와 R4x `v3-p1-xlw-arrival`(R2x + 교차 언어 슬롯 먼저 말한 순서, job 94526)를 제출했다. 평가 큐 8 개가 체크포인트를 기다린다(단계마다 GPU 를 새로 고르고 OOM 이면 한 번 이어 하도록 큐를 고쳤다). 공용 GPU OOM 으로 깨진 R1·R2 평가 5 단계는 재실행 중이다.
- **R2x·R4x 결과(2026-10-22)**: 같은 언어 과잉 교대는 사라졌다(Stage 3 δ10 agn R2x ck2000 18.55 · R4x final 18.37 = V1 수준, aihub71631 태그 V1 수준). 대신 TAXI 귀속 이득을 약 1 pt 돌려줬다. TAXI 통째 δ10 은 R2x 22.3–23.9, R4x 21.8–22.9 다(R2 21.6–23.3, VibeVoice 스트리밍 20.5).
  - 사전 규칙 선택은 R2x ck2000(TAXI 23.9), R4x ck1500(TAXI 22.9)이다. 둘 다 후보 기준(TAXI cp < 20.5)에 못 미친다.
  - arrival 효과(R4x − R2x)는 TAXI 에서 유의하지 않다. 합성 xl 번역은 −1–2 BLEU 다. 세부는 [[2026-10-21-v3-attribution]] 14 절에 있다.
- **TAXI 귀속 진단(2026-10-22 20시)**: R1x 의 남은 TAXI 오류는 목소리가 비슷한(F0 차 < 40 Hz) 실제 통화에서 영어 쪽 교대를 놓치는 것이다(15/199 대 0/80). 레벨·잡음·채널 대비 가설은 TAXI 없이 만든 진단 dev 로 기각했다(R1x 는 합성 조건에서 강하다). 그래서 1.7B 6 단계는 R1x 레시피 그대로 간다.
- **Qwen3-ASR 1.7B 체인 Stage 1·2 검토(2026-10-22 20시)**: 사용자 요청으로 검토해 세 가지를 고쳤다.
  - 10-08 의 1.7B Stage 1 은 worker 8 개로 노드 메모리가 부족해(oom_kill) 세 번 죽었다. worker 4 로 맞췄다.
  - 1.7B 체크포인트는 두 조각으로 저장되는데 재개 탐색이 이를 못 찾았다. 그래서 재시작이 init 부터 다시 했다. 공용 함수로 고쳤다.
  - Stage 1 레시피가 0.6B 와 달랐다(r3 뒤 1 epoch). 기본을 0.6B 와 같은 `q17-s1m3-d10-s0`(q17-s0 → 3 epoch)로 바꾸고, 이어 하기 판은 `--s1 resume` 으로 남겼다.
  - Stage 2 는 0.6B 와 같다. 번역 타깃 토큰 id 는 tokenizer 어휘 지문이 같아 그대로 쓴다.
  - 20:42 사용자 제출(기본 mirror): job 94809(Stage 1 `q17-s1m3-d10-s0`)–94814(V3 R1x), 단계마다 afterok.
  - 2026-10-23 00:21 기준 Stage 1 은 3,630/16,173 step 이다. 실패는 없고 2.86 s/step 이다.
    - 같은 step 구간의 0.6B 는 1.22 s/step 이었다. 1.7B 가 2.3 배 느리다.
    - 그래서 체인 어림을 약 37–46 h(6 단계는 10-24 낮–밤 끝)로 고쳤다. 10-22 저녁의 "속도 같음, 약 30 h" 는 10-08 한 구간만 본 틀린 어림이었다. → `raw/sources/experiments/2026-10-22-q17-chain/README.md`
- **진행(2026-10-24 23시)**:
  - 1.7B 체인: Stage 1 은 10-23 13:38, AST 1 차는 23:22, AST 2 차(94811)는 10-24 11:47 에 끝났다. AST 3 차(94812)는 23:05 에 97.6 % 다.
    - AST 3 차는 노드 메모리 부족(oom_kill)으로 두 번 죽고 이어 갔다(재시작 한도 5 중 2).
    - 다음은 대화 Stage 3(94813) → R1x(94814) → 평가(94841)이고, 대화 레시피는 r1x 그대로다.
    - 10-25 01시: 5 단계 01:07 끝, R1x 6 단계 01:09 시작. 사용자 요청으로 회의 + 4 슬롯 6 단계(`q17-v3-p1-tel-xlw-mt4`)를 R1x 6 단계(94814) 뒤에 이어 낸다(`--after-job 94814`). 사용자 제출 01:31: 학습 95326 → 평가 95327.
      - 01:39 사용자: 평가 94841(R1x)·95327(회의 + 4 슬롯)을 취소하고, 두 평가를 회의 + 4 슬롯 학습(95326) 성공 뒤 노드 하나씩으로 다시 냈다.
        - R1x 평가 95328(`q17-v3`), 회의 + 4 슬롯 평가 95329(`q17-mt4-v3`), 각 45 작업. 학습 94814 → 95326 은 그대로다.
    - **결과(10-25 11시)**: 학습 둘(02:30·03:51)과 평가 둘(07:29·07:54)이 끝났다.
      - 회의 + 4 슬롯이 AMI 창 cp −5 ~ −8(전 분할 유의, 받아쓰기 덕)과 TAXI δ10 −2.0(final, 경계선)·−2.7(avg3, 유의)로 낫다.
      - dev 귀속·같은 언어 대화·대화 BLEU 는 같다. 단일 화자 LibriSpeech other +0.55–0.88 이 가드를 넘는다.
      - 1.7B 와 0.6B 의 R1x 는 TAXI·AMI 가 같다. → `raw/sources/experiments/2026-10-22-q17-chain/README.md`
    - **AMI SDM 오류 세분(10-25 11시)**: 사용자 목표 "SDM 20 미만".
      - 1.7B 회의 + 4 슬롯 avg3 의 SDM eval δ10 은 cp 53.5 = agn 28.6 + 귀속 24.9 다.
      - 귀속은 겹치지 않은 긴 턴을 통째로 다른 슬롯에 붙이는 것(11.6 pt)이 가장 크다. 4 인 창 단어의 75 % 를 2 슬롯으로 낸다.
      - 받아쓰기 오류의 47 % 는 겹친 구간(단어 21 %, S+D 57 %)에서 난다. 겹치지 않은 원거리 음성은 17.6 %(IHM 9.9 %)다.
      - 20 미만에는 귀속, 겹침, 원거리가 함께 풀려야 한다. → `raw/sources/experiments/2026-10-25-ami-sdm-analysis/README.md`
      - 사용자 제안: 화자 분리를 모델 안에서 학습해 통합한다. 활동 헤드 → 주입, 화자 캐시, 화자 임베딩 손실 순으로 검증을 제안했다.
      - 참고 자료 검토(Streaming Sortformer·Nemotron 3 Diarization·SSA·FastMSS) → 화자 분리 가지 + SSA 주입 + 슬롯별 디코드 + AOSC 설계와 E0–E4 시험 순서. → [[output-diarization-integration-20261025]]
      - 사용자 "계획대로 진행하는 데 1.7b 모델로 진행해".
        - E0(정답 구간으로 다시 귀속, 학습 없음): SDM eval cp 53.5 → 39.3(귀속 24.9 → 10.7). 한 스트림 형식의 천장이다. 남는 몫은 겹침이다.
        - E1 화자 분리 가지(1.7B 회의 + 4 슬롯 인코더 동결 + Transformer 12 층 38.9M, 도착 순서 BCE + PIT)는 스모크(30 step·저장 대조)와 평가 경로까지 통과했다.
          제출은 `slurm/submit-diar-train.sh`(1 노드 apex, 20k step)다.
        - E1 학습 job 95482(사용자 제출 15:32, hpc-19)는 18:06 에 끝났다(2 시간 28 분).
          - AMI 창 DER(80 ms, collar 0, 최적 짝, dev 에서 고른 임계 0.45): eval IHM-mix 18.67, SDM 23.03. 도착 순서로 짝지으면 31.0, 36.6 이다.
          - miss 가 가장 크다(SDM 13.1). 그 46 % 가 겹친 구간이고, 말이 적은 화자를 통째로 놓친다. → `raw/sources/experiments/2026-10-25-ami-sdm-analysis/README.md` E1 절
          - 본 학습은 끝에 저장·재로드 대조를 하지 않았다. 다시 읽은 final 이 학습 로그 끝 값과 맞음을 확인했고, `diar_train.py` 가 끝에 늘 대조하도록 고쳤다.
        - E2 대상 화자 모드(정답 활동 주입 + 화자별 디코드)를 구현했다: 데이터셋 target_spk, 인코더 pre-encode 주입(adapter_inject), `--target-spk`, 평가 `--target-oracle`.
          1.7B 스모크·저장 대조와 평가 경로를 통과했다. 제출은 `slurm/submit-q17-e2-tgt.sh`(학습 4,000 step → AMI 정답 활동 평가 8 몫)다.
        - E2 학습 job 95486·평가 job 95487(사용자 제출 16:11, low_p_hpc 2 노드, MAX_RESTART 20 — 사용자 요청으로 제출 스크립트 기본 파티션을 low_p_hpc 로 바꿨다).
          선점이 없으면 학습 약 1.5 h(1.7B 회의 + 4 슬롯 2 노드 1.3 s/step 기준), 평가 약 2 h(어림 — 같은 모델 AMI 창 평가 4 몫 각 62 분, 화자별 디코드라 창당 약 4 배).
          - 17:02 선점이 정기 저장(checkpoint-500) 도중에 와 저장이 잘렸다. 신호 뒤 약 70 s 에 종료됐고, 1.7B 전체 저장은 약 3.5 분이다.
          - 18:17 에 step 0 부터 다시 시작했다(hpc-[32,38], 2.5–2.8 s/step). 새 checkpoint-500 은 완전하다. 선점이 더 없으면 21:30 무렵 학습이 끝난다.
          - low_p_hpc 에서 1.7B 는 선점 저장이 유예 안에 끝나지 않는다. 선점마다 마지막 500 step checkpoint 이후를 잃는다.
        - E3 준비: E1 예측 활동을 E2 화자별 디코드에 주입하는 평가를 만들었다.
          - `cst_stage3_eval --target-act`, `v3_eval_task.sh amie3`, `submit-evals.sh q17-e3`(임계 0.45).
          - 정답 활동 경로가 고치기 전 출력과 같고, 예측 경로가 도는 것을 확인했다. 제출은 사용자가 한다(95486 성공 뒤).
      - init 은 같은 5 단계 ck3000 이다. 1.7B 점검(dry-run·스모크 20 step·저장 대조)은 통과했다. 평가는 `q17-mt4-v3` 다.
  - 1.7B Stage 1 평가(94840): 같은 레시피 0.6B 대비 한국어·독일어는 모든 δ 에서 0.4–0.7 pt 유의하게 낫다. 영어는 δ4·6 에서 0.3–0.5 나쁘고 δ8 이후 같다. → `raw/sources/experiments/2026-10-22-q17-chain/README.md`
  - 0.6B 합친 Stage 2(94850·94851): 받아쓰기는 나빠진 곳이 없다.
    - 번역은 24 비교 중 FLEURS 영→한 δ4 하나만 −1.44 [−2.29, −0.56] 로 유의하게 낮아, 미리 정한 기준상 불합격이다.
    - 채택 여부는 사용자가 정한다. → `raw/sources/experiments/2026-10-23-q06-st2-merged/README.md`
  - 아직 내지 않은 것: 0.6B 회의 + 4 슬롯 시험(`submit-q06-mt4.sh`), 회의 통째 평가(`submit-evals.sh v2-amifull`).
- **AMI 회의 통째 평가(2026-10-24 01시)**: 사용자 "AMI 평가에 한번에 돌릴 수 있도록 창 길이를 늘리고 싶은데, 자르지 않고".
  - `cst_stage3_eval.py --window full`: 회의마다 창 하나로, 자르지 않고 한 스트림으로 디코드한다(세션 단위 cpWER).
    - 인코더는 겹침 조각, LLM 은 긴 문맥 eager 디코드다(`vapasr/hf/long_session.py`).
    - 대조: 300 s 인코딩 max|Δ| 5e-5, 60 s·6 분 디코드가 기존 `decode_batch` 와 토큰까지 같다.
  - V2 final(4 슬롯)을 문맥 전부로 돌리면 3.5–4 분(약 5,600 위치, 학습 창의 두 배쯤) 뒤 말을 그친다. 14 분 회의와 49.5 분 회의가 같다.
  - LLM 문맥만 최근 S 초로 두는 `--context-s` 를 넣었다(오디오는 자르지 않는다). 이렇게 하면 끝까지 받아쓴다(IS1009a δ10 agn 20–22.5, 창 평가 21.1).
    - 그러나 세션 cp 는 84–89 다(창 평가 39.9). 네 화자를 두 슬롯에 담고, 회의 중간에 짝이 바뀐다.
    - 학습이 창마다 슬롯을 새로 정하므로, 회의 수준 귀속은 배운 적이 없다.
  - 평가 묶음: `submit-evals.sh v2-amifull`(문맥 30 s, 노드 하나 약 4 h), `mt4-amifull`·`q17-mt4-amifull`. 아직 제출하지 않았다. → V3 README 28 절
- **`<EOT>` 없는 R1x 결과 — 채택 안 함(2026-10-24 00시)**:
  - dev 귀속은 같았다(−0.07 대 −0.12).
  - 한국어 대화 받아쓰기가 무너졌다: 같은 언어 agn 18.7 → 22–30, aihub71631 은 가설/참조 1.5 · 강제 진행 15/200.
  - TAXI 도 0.8–1.7 pt 나빠졌다. 독일어·영어 대화는 그대로다.
  - 회의 + 4 슬롯 시험은 `<EOT>` 있는 R1x 기반으로 되돌렸다(`submit-q06-mt4.sh`, low_p_hpc). → V3 README 27 절
- **`<EOT>` 없는 대화 학습 시험(2026-10-23 09시)**: 사용자 판단 — 턴 예측은 주 목표가 아니고 화자 귀속이 더 중요하다.
  - `--no-eot` 로 `<EOT>` 를 정답·문맥 모두에서 뺀다. 머무름 가중 기준점은 뺀 `<EOT>` 의 청크 끝이다.
  - 0.6B 대화 Stage 3·4 를 원래 명령 + `--no-eot` 로 다시 학습한다: job 94867 → 94868 → 평가 94869.
  - 판단 기준: dev 귀속이 R1x 보다 나쁘지 않고 가드를 통과하면 채택한다. 채택하면 1.7B 대화 두 단계(94813·94814)를 `--no-eot` 로 다시 낸다.
  - 근거 자료: 교대는 `<EOT>` 에 묶여 있지 않다(같은 언어 dev 70 %·AMI 85 %·TAXI 거의 전부가 `<EOT>` 없이 교대). → V3 README 22·24 절
- **0.6B 합친 Stage 2 시험 준비(2026-10-23 01시)**: 사용자 질문("화자 태그는 대화에서부터 하고, 한 단계로 통합하면 안 좋으려나?")과 apex 2 노드 추가 허용에 따라 준비했다.
  - `q06-st2-merged`: Stage 1 final 에서 s4 최종 설정으로 4 epoch(29,720 step)를 한 번에 돈다. 지금은 s2·s3·s4 세 번, 43,436 step 이다.
  - 평가는 `slurm/submit-eval-ast.sh` 로 s4 와 같은 7 셋을 잰다(학습 성공 뒤 afterok).
  - 판단 기준(미리 정함): 번역 BLEU 가 어느 방향·δ 에서도 s4 보다 유의하게 낮지 않고, 받아쓰기도 유의하게 나빠지지 않으면 채택한다.
  - 화자 태그는 Stage 2 에 그대로 둔다(대화 형식의 한 슬롯 경우이고, replay 데이터에도 필요하다).
  - 사용자 제출: 학습 job 94850(01:12 시작) → 평가 job 94851(afterok).
  - 1.7B 체인은 세 번 판 그대로 둔다. → `raw/sources/experiments/2026-10-23-q06-st2-merged/README.md`
- **평가 SLURM 묶음·AMI(2026-10-23 00시)**: 사용자 요청으로 `slurm/submit-evals.sh` 를 만들었다. 묶음마다 `v3-eval-node.sbatch` 로 낸다.
  - `r1x-ami`: 0.6B R1x final·avg3 의 AMI 공식 dev·eval(IHM-mix·SDM). 바로 시작한다.
  - `q17-s1`: 1.7B Stage 1 단일 발화 ASR. job 94809 성공 뒤.
  - `q17-v3`: 1.7B V3 평가 9 종 + 평균 + AMI. job 94814 성공 뒤.
  - 사용자 제출(00:41): r1x-ami = job 94839, q17-s1 = 94840, q17-v3 = 94841.
  - AMI 는 V2 12 단계와 같은 명령을 4 몫으로 나눠 돈다(`v3_eval_task.sh ami`). V1 ck2000 기준은 공식 eval δ10 IHM-mix cp 54.0 · SDM 60.9 다.
  - 같은 체크포인트를 여러 GPU 가 동시에 복사하던 `stage_dir` 는 잠금으로 고쳤다.
  - **R1x AMI 결과(job 94839, 01:50)**: V1 과 cp 차이 ±1.3 pt 안이고, 유의한 것은 dev SDM δ10 −1.3 하나다. 퇴행도 이득도 없다.
    - 공식 eval δ10: IHM-mix 53.96(V1 54.02) · SDM 61.26(V1 60.88).
    - AMI 오류는 받아쓰기와 3–4 인 회의 귀속 문제다. → V3 README 23 절
- **체크포인트 평균(2026-10-22 12시)**: R4x 평균 `v3-p1-xlw-arrival-avg3` 이 R4x 계열에서 dev 귀속 몫 최소(0.76)다. TAXI 통째 δ10 21.7 · 귀속 4.5 다. V1 대비 cp −4.7 [−9.0, −0.8] · 귀속 −3.5 [−6.4, −0.9] 로 처음으로 95 % 구간이 0 아래다. 같은 언어 가드는 통과하고, 번역 가드(합성 xl BLEU −0.7, TAXI 영→독 −1.3)는 조금 넘는다. R2x 평균은 오히려 나빠졌다(TAXI agn 20.3).
- 다음 세 갈래(사용자 "셋다 진행해보자"): R1x `v3-p1-tel-xlw`(R1 + 교대 가중 교차 언어 셋만, job 94613)·R1x-a `v3-p1-tel-xlw-arrival`(job 94614)·R2m `v3-p1-sl15`(같은 언어 셋 교대·머무름 1.5, job 94615) 학습 중이다. R2x·R4x 체크포인트 평균(ck1500·ck2000·final)은 평가 중이다. 세부는 [[2026-10-21-v3-attribution]] 15 절에 있다.

## 2026-10-20 V2 첫 학습 v2-p1 결과(중간)

- v2-p1(m2a 설정 + 두 스트림 입력·최대 4 화자·창 20–90 s·회의·합성 3–4 인) 3,000 step 완료, PARITY OK → [[2026-10-20-v2-multispeaker]]
  - 첫 step 의 `slot_activity` IndexError(화자 상한 줄이기 반올림)를 고쳐 restart 2/5 부터 정상이었다.
- V1(m2a ck2000) 대비, δ10:
  - 2 인 같은 언어 대화는 V2 ck2000 이 낫다. cp 25.59 → 24.46, agn 18.18 → 17.03(짝 bootstrap 구간이 0 아래), `<EOT>` 재현율 0.72 → 0.79, 대화 번역 +0.9–1.5 BLEU.
  - 2 인 창의 화자 귀속 몫은 그대로다(7.4). 단일 화자는 ±0.5 pt, CallHome 독일어 21.6 → 19.7 이다.
  - **TAXI 는 모든 체크포인트에서 5.0–8.4 pt 나쁘다**(28.1 → 33.4). 대부분 두 화자 말이 한 언어 슬롯으로 몰린 짧은 세션 탓이다.
    - C·D 슬롯 새기는 0 이다. 탭을 끄면 더 나빠진다(37.3). 남은 후보는 데이터·창 변경이고, 가르려면 학습 대조가 필요하다.
  - 영↔독 합성 δ10 영어 반복 생성이 ck2500·final 에서 늘었다(WER 50–56).
- 탭 뺀 대조 v2-p1-notap 제출(job 93980, 93778 뒤): DB·창·4 슬롯은 v2-p1 과 같다. 탭 몫과 DB 몫을 가른다(평가 큐 ck2000·final).
- 다음: 긴 창·회의·3–4 인 합성·AMI(IHM-mix·SDM)·회의 번역 평가가 진행 중이다(내일 아침 예상) → V1·notap 대비 귀속 효과를 보고 TAXI 대책 학습을 정한다(사용자).

## 2026-10-20 st4-m2(전화 대역 증강) 결과·V2-a 두 스트림 입력

- st4-m2a(전화 대역 증강)·st4-m2b(+ CallHome 독일어) 학습 완료(PARITY OK) → [[2026-10-19-stage4-dialogue-translation]]
  - TAXI 받아쓰기 cpWER 40.9 → 28.1(m2a ck2000), 독→영 빈 턴 26 → 3. m2b 는 m2a 보다 1.6–2.6 pt 나쁘다.
  - VibeVoice 스트리밍(19.2)과의 비교 → [[output-vibevoice-stream-comparison-20261020]]
  - 모든 평가가 끝났다(final 은 OOM 뒤 이어 돌리기). m2a ck2000 이 가장 고르다 → Stage 4 최종 교체 여부는 사용자 결정 대기.
- V2: M0(화자 정보는 인코더 아래층 96 %·LLM 입력 69 %) → 사용자 의문으로 VibeVoice 식 **두 스트림 입력(V2-a)** 으로 방향을 바꿨다 → [[output-v2-speaker-memory-plan-20261020]]
  - Nemotron L00 → `adapter_tap`(0 초기화) → LLM 입력에 더함. 구현·단위 시험 통과, mxc 학습 전 점검 진행 중 → [[2026-10-20-v2a-dual-stream]]
  - 파일럿 제안: st4-m2a 와 같은 설정·seed + `--enc-tap-layer 0`(m2a 를 대조군으로)

## 2026-10-20 Stage 4 최종 → st4-m1 ck1500, VibeVoice 외부 기준선

- **Stage 4 최종 = `/soundai/Model/VAPASR/stage3-st4-m1/checkpoint-1500`**(사용자 결정): TAXI 받아쓰기 cpWER 40.9 · 영→독 BLEU 27.6 · 독→영 21.3, Stage 3 회귀 없음 → [[2026-10-19-stage4-dialogue-translation]]
- VibeVoice-ASR-Streaming-7B 대비: 학습 코퍼스의 held-out 대화(한·독)는 우리가 낫고 영어는 비슷하다. 실제 독↔영 전화(TAXI)는 19.2 대 40.9 로 크게 뒤진다.
- 다음: 전화 영역 데이터(CallHome 독일어 단일 화자 셋 `rep_chde`, dry-run 통과)·전화 대역 증강, 영어 반복 생성 원인

## 2026-10-19 Stage 4 준비

- 사용자 결정 → [[decision-stage4-prep-20261019]], 계획·진행 → [[output-stage4-prep-plan-20261019]]
- S1 대화 번역 타깃: 시험 생성(코퍼스당 3 대화)의 관문 통과는 수리 뒤 96–99 % 다. 전체 생성(약 39 GPU 시간)은 사용자 제출을 기다린다.
- S2 네 흐름 시퀀스(두 화자 전사 + 번역)를 구현했다. dry-run 은 문제 0 이다 → [[2026-10-19-stage4-dialogue-translation]]
- S3 교차 언어 합성 대화와 S4 번역 평가 도구를 구현했다. 시험 합성 KO↔EN 에서 오라클은 만점이다. Stage 3 최종을 그대로 쓰면 번역 BLEU 9–17 이다.
- S4 에 TAXI(실제 독↔영 전화 대화) 어댑터를 더했다. Stage 3 최종을 그대로 쓰면 영→독 BLEU 18, 독→영 4 다.
- 파일럿 st4-p1 완료: TAXI δ10 독→영 BLEU 3.7 → 22.7, 영→독 18.0 → 26.4, 받아쓰기 cpWER 93.5 → 43.2(ck2000 최고, step 이 늘수록 내려감). Stage 3 받아쓰기·귀속·`<EOT>` 회귀는 없다.
- 다음: st4-p1 체크포인트 선택(사용자), 영↔독 합성 영어 δ10 WER 이상·FLEURS 한국어 CER +2–3 pt 확인, 대화 발화 단독 디코드 대조(제안). S7(TTS)은 엔진·원천 대화 결정을 기다린다.
- Stage 3 파일럿 p12(NSC 격리 발화 되살리기)는 독일어 귀속을 줄이지 못했다(ck3000·final 모두 구간이 0 을 포함). Stage 3 최종은 p11 ck3000 그대로다.

## 2026-10-19 Stage 3 최종 교체 → p11 ck3000

- **Stage 3 최종 = `stage3-st3-p11-revive/checkpoint-3000`**(사용자 결정): p10 레시피 + 격리 발화 되살리기 → [[2026-10-18-stage3-ko-revive]]
  - δ10 cp 25.28(p10 ck3000 25.62), 한국어 귀속 몫 6.9(8.0), 영어 4.9(6.4), EOT 재현율 0.766 · 정밀도 0.594
  - 1 분 대화로 환산하면 한국어에서 다른 사람 말로 붙는 글자가 13.0 → 11.2 다. 턴을 통째로 틀리는 경우는 7–8 턴에 1 턴꼴로 그대로다.
- p11 은 `low_p_hpc` 에서 돌렸다. 선점, 랑데부 실패 뒤 멈춤, 노드 env 복사본 누락을 고친 뒤 끝났다(sbatch 감시자·`--exclude`·복사본 파일 수 대조).

## 2026-10-18 Stage 3 최종 확정과 한국어 화자 귀속 파일럿 준비

- 10-03 – 10-17: Stage 1–2(q06 s1–s4, [[output-q06-s4-ast-eval-20261013]])에 이어 Stage 3 준비·파일럿·본 학습을 했다 → [[output-stage3-prep-plan-20261014]]
- **Stage 3 최종 = `stage3-st3-p10-sim/checkpoint-3000`**(사용자 결정): 화자 헤드 + 합성 대화, δ10 cp 25.6 · EOT 재현율 0.719
- 한국어 화자 귀속:
  - 원인 후보는 학습 대화의 마스크다(화자 교대의 69–80 %).
  - 격리 발화 되살리기를 구현했고, 파일럿 p11 은 사용자 제출을 기다린다 → [[2026-10-18-stage3-ko-revive]]
- 저장소 정리:
  - Codex 의 교사 골드 관문 미커밋 작업을 지웠다(T5 백업).
  - 커밋된 코드가 기대던 미커밋 학습 코드를 커밋했다(aa64f73).
  - speechlm approval·QC 묶음(현재 한국어 학습 데이터의 선별 근거)은 커밋해 기록으로 남겼다.

## 2026-10-02 프로젝트 목표 전환 → CST-S2TT

- 사용자가 `raw/sources/CST_S2TT_Project_Spec_v0.1.pdf` 를 제시했다. 목표를 **대화형 동시 음성→텍스트 번역(CST-S2TT)과 CST-Bench** 로 바꿨다.
  → [[decision-project-goal-cst-s2tt]], [[source-cst-s2tt-project-spec-v0-1]], [[conversational-simultaneous-s2tt]], [[cst-bench]]
- 최우선은 모델이 아니라 Phase 0–3 이다.
  - 교차언어 대화 코퍼스 확보
  - 공통 schema
  - 통제 타임라인 생성기
  - CST-Bench v0 지표
  - strong 기준선 B0–B4 와 kill criterion
  - 태스크: [[task-cst-data-access]], [[task-cst-conversation-schema-parser]], [[task-cst-timeline-generator]], [[task-cst-bench-v0-metrics]], [[task-cst-strong-baselines]], [[task-cst-kill-criterion-check]]
- `README.md`·`PLAN.md`·`TODO.md` 를 새 목표로 다시 썼다. 이전 판은 `plans/*-vap-asr-*.md` 로 보관했다.
- 이전 목표에서 이어 쓰는 자산:
  - 스트리밍 백본(인과 인코더 + Qwen3-ASR thinker, δ 지연, final 모드)
  - `<SEM_END>` commit 라벨·골드·지연 지표
  - VAP 재현(B3)
  - single-turn ASR 평가
- 전환 시점 진행 중:
  - semcommit 라벨링 job 79205(4 노드)는 완료 4,565 파트이고, 이번 순서에서 3,694 파트가 남았다
  - Stage 1 본학습 q17-s1m-r0([56,0])·r3([56,3], 3 epoch)이 끝났다:
    - 한국어는 파일럿 대비 약 1 %p 개선, 영어(VoxPopuli)는 약 1 %p 악화했다
    - 중간 checkpoint 평가로 원인을 epoch 증가(한국어 77 % 학습 구성)로 좁혔다 → [[output-stage1-pilot-eval-20261001]]
  - 라벨링 계속 여부와 Stage 2 진행 여부는 사용자 판단이 필요하다
- 재검토할 기존 결정:
  - [[decision-mono-input]]: 화자별 스트림 입력 여부
  - [[decision-target-architecture]], [[decision-semcommit-turn-eot-scope]], [[decision-multi-speaker-scope]]

## 2026-09-29 semcommit 첫 본 학습·평가

- 레시피 v0.3.5 라벨(스냅숏 20260929-0928: EN main 26.3 만 · KO main 39.5 만 · KO short 93.6 만 스트림)로 E2 에서 5 epoch 학습했다(job 76601, apex 2 노드, 약 3.6 h, δ 2·3·4·6·8). 저장·재로드 parity 는 통과했다. → [[output-semcommit-v035-d8-eval]]
- ASR 전체 셋(single-turn v1, 11,246 발화): E2 대비 12 칸 중 10 칸이 유의하게 개선됐고 악화는 없다. δ4 수치는 test-clean 5.05 · test-other 11.88 · Kspon eval_clean/other 12.26/12.26 % 다. 오프라인 수준과는 아직 차이가 커서 개선 방안을 분석 중이다.
- commit(골드 v1, δ4): F1 은 KO 짧은/긴 발화 0.85/0.79, EN 낭독/대화체 0.67/0.74 다. 한국어는 너무 자주, 영어는 너무 드물게 확정한다. 언어별 추론 bias 로 보정할 계획이다.
- 코드: 창 없는 commit 지연 지표(5103556), varlen attention(4430c94, 학습 처리량 +22–36 %), 2-pass 재디코드 계획(2c0eb99). 출력 형식 플래그는 Codex 의 미커밋 short-pool 변경과 같은 헝크에 있어 커밋을 미뤘다.
- 라벨링은 09-29 11:09 에 멈췄다. 확보량은 EN main 1,337 h · KO main 1,347 h · KO short 1,114 h 이고, 언어별 균형까지 언어당 약 800 h 가 남았다.

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

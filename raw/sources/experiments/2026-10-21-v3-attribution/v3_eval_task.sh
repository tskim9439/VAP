# V3 평가 작업 하나: bash v3_eval_task.sh <작업> <run> <대상> — GPU 는 부르는 쪽이 CUDA_VISIBLE_DEVICES 로 준다(slurm/v3-eval-node.sbatch 일꾼, GPU_IDX = 그 번호).
#   작업: dlg | xl | taxi40 | stage3 | regress | xltel | xleval | taxi90 | onset | ami | amifull | amitgt | amie3 | amiact — stage4_eval_queue.sh(1–6 단계)·v3_eval_queue.sh·v3_dev_decode.sh·onset_probe.sh 와
#   같은 명령·산출 폴더다(TRANS_LEAD_SPACE=1 과 같이 --trans-lead-space — V3 run 은 모두 그렇게 학습했다).
#   체크포인트: MODEL_ROOT(기본 /soundai/Model/VAPASR)/stage3-<run>/<대상>. 대상이 final 이면 DONE, 아니면 trainer_state.json 이 생길 때까지 기다린다.
#   건너뛰기: 끝난 작업(요약·보고 파일 있음). xltel·xleval·taxi90 은 turn-metrics-d10.json, onset 은 onset-classes-d10.json 이 있으면 건너뛴다
#   (2026-10-23 전에는 산출 폴더만 있어도 건너뛰었다 — low_p_hpc 선점 뒤 다시 돌면 덜 끝난 산출을 그대로 두게 돼 바꿨다. run 은 덮어쓰므로 처음부터 다시 한다).
#   PY 로 python 을 바꾼다(노드 로컬 env). 아무것도 지우지 않는다.
#   ami(2026-10-23, 사용자 "AMI_IHM 이랑 AMI_SDM 에 대해서도 평가를 수행하고 싶어"): AMI 공식 dev 18·eval 16 회의(ami-bench-v1 — 공식 Mix-Headset = IHM-mix,
#     Array1-01 = SDM) 창 전부, δ4·10, 4 슬롯 선언 — V2 v2_eval_extra.sh 12 단계와 같은 명령. 다섯째 인자 r/w 로 창을 w 몫으로 나눠 r 번째만 돈다
#     (cst_stage3_eval --rank·--world-size, 이어 하기 됨). 마지막으로 끝난 몫이 summarize·오류 분해·ami_report.py(공식 분할별)를 한 번 돈다.
#     산출 results/stage3-eval-ami-<run>-<태그>(+ ami-report.txt) · stage3-attr-errors-ami-<run>-<태그>.
#   amifull(2026-10-24, 사용자 "AMI 평가에 한번에 돌릴 수 있도록 창 길이를 늘리고 싶은데, 자르지 않고"): 같은 34 회의 × IHM-mix·SDM 을 창으로 자르지 않고
#     회의마다 한 스트림으로(cst_stage3_eval --window full — 세션 단위 cpWER), δ4·10, 4 슬롯 선언, LLM 문맥 창 CTX 초(기본 30 — 문맥 전부(0)면 V2 가 3.5 분 뒤
#     말을 그친다, V3 README 28 절). 몫 r/w 는 ami 와 같고 몫 0 이 --verify(앞 300 s 인코딩·앞 60 s 디코드 대조)를 한다. 귀속 오류 분해(cst_attr_errors)는
#     정렬이 O(단어²)라 돌리지 않는다. 산출 results/stage3-eval-amifull-c<CTX>-<run>-<태그>(+ ami-report.txt).
#     E7(2026-10-27): AOSC=1 이면 화자 분리 헤드(E6·E7) 예측을 회의 전체 AOSC 스트리밍으로(--aosc — 회의 내내 같은 슬롯, 없으면 조각마다 새로),
#     DIAR=<diar.pt>(DIAR_TAG=이름)이면 헤드를 그것으로 바꿔 끼운다(--diar-head-override — diar_train --aosc final). 산출 이름에 -aosc·-<DIAR_TAG> 가 붙는다.
#     GUIDE=1(2026-10-28): 화자 태그를 활동으로 이끈다(--act-tag-guide — long_session.make_retag). 산출 이름에 -guide 가 붙는다.
#   MIXOFF=1(2026-10-29, E10 슬롯 추출기 체크포인트): AMI 작업(ami·amifull)을 섞인 소리 토큰 없이(--mix-off) — 산출 이름에 -mixoff 가 붙는다.
#   N3D=<.nemo>(2026-10-29, E11): AMI 작업(ami·amifull)의 슬롯 활동을 고정 화자 분리 전문가 Nemotron 3 Diarization 으로(--n3d — 창은 창마다, 회의 통째는 회의 전체
#     스트리밍 한 번, N3D_CHUNK 기본 4). 슬롯 활동 입력 체크포인트(E6·E9·E10)에 붙이는 학습 없는 진단 — 산출 이름에 -n3d(청크가 4 가 아니면 -n3dc<청크>)가 붙는다.
#     E11 체크포인트는 config 에 전문가가 있어 N3D 없이도 그것을 쓴다(이름은 그대로).
#   GATE=BACK,FWD(2026-10-30, 활동 추종 진단 — wiki output-n3d-branch-20261029 6.5 절): AMI 작업(ami·amifull)을 활동 게이트 디코드로(--act-gate — 슬롯 첫 출력의
#     오디오 근처에 그 슬롯 활동이 없으면 그 라운드 그 슬롯을 버린다). 산출 이름에 -gate<BACK>-<FWD> 가 붙는다.
#   FOLLOW=1(2026-10-30): 같은 설정의 끝난 AMI 산출(summary.json 있음)을 디코드 없이 다시 채점해 활동 추종 진단(follow·cp_follow)을 남긴다 — 원출력에 활동(act_runs)이
#     없으면 원 평가 config 의 전문가(n3d)로 다시 계산한다(--from-predictions … --follow-n3d, GPU 가볍게). 산출은 <원 산출>-follow. GATE 와 함께 쓰지 않는다.
#   ENC_STATE=<graft.pt>(2026-10-30, U3): AMI 작업(ami·amifull)에서 모델 인코더를 그 graft.pt 의 enc(화자 분리와 함께 학습한 인코더)로 바꿔 잰다
#     (--encoder-state — ASR 지킴 확인). 산출 이름에 -enc<ENC_TAG>(기본 u3) 가 붙는다.
#   GRAFT_EXPERT=<graft.pt>(2026-11-01, G3): AMI 작업(ami·amifull)의 슬롯 활동을 고정 전문가 대신 그 graft.pt 의 이식 학생으로(--graft-expert — 창은 창마다,
#     회의 통째는 회의 전체 한 번). ENC_STATE 를 같은 graft.pt 로 주면 받아쓰기 인코더도 학생 것이다. 산출 이름에 -gx<GX_TAG>(기본 g2) 가 붙는다.
#   ENC_RC=<0|3|6|13>(2026-11-01, 지연 완화): AMI 작업(ami·amifull)에서 인코더 우측 문맥을 바꿔 잰다(--encoder-rc, 학습은 3). 산출 이름에 -rc<N>.
#   GATE_LAGS=<N> · GATE_THRESH=<0.x>(2026-11-03, E13 게이트 진단): AMI 작업(ami·amifull)에서 활동 게이트 모델(E13)의 추론 게이트만 바꾼다 — 창 N 청크(0 = 끔,
#     --gate-lags) · 게이트만 전문가 확률 > 0.x 로 연다(--gate-thresh, 태그 활동은 0.5 그대로). 산출 이름에 -gl<N> · -gt<0.x 에서 점을 뺀 것>(예 -gt03).
#   N3D_RC=<프레임>(2026-11-01): N3D 와 함께 화자 분리기 오른쪽 문맥(--n3d-rc). 산출 이름의 -n3d… 뒤에 r<N> 이 붙는다.
#   EV_TAG=<이름>(2026-10-30): AMI 작업(ami·amifull) 산출 이름에 -<이름> 을 붙인다 — 같은 체크포인트를 고친 코드로 다시 잴 때(예 E10 <BEGIN> 차단 수정 onsetfix).
#     예전 산출은 지우지 않고 그대로 둔다(이미 끝난 산출이 있으면 작업이 건너뛰므로 새 이름으로).
#   DELTAS(2026-10-28, 사용자 "AMI IHM 이랑 SDM 두개에 대해 delta 10 으로만 평가해서 비교해서 의사결정을 빠르게 하자"): AMI 작업(ami·amiact·amitgt·amie3·amifull)의 δ
#     (기본 4,10 — 예전 산출 그대로). 다르면 산출 이름에 -d<δ> 가 붙는다(예 DELTAS=10 → stage3-eval-ami-d10-<run>-<대상>).
#   amitgt(2026-10-25, E2 — wiki output-diarization-integration-20261025): 대상 화자 모드 체크포인트(adapter_inject)를 ami 와 같은 창에서 정답 화자 활동 주입·화자별
#     디코드로 잰다(cst_stage3_eval --target-oracle). 몫 r/w·합치기는 ami 와 같고, 귀속 오류 분해는 원출력 형식이 달라 돌리지 않는다. 산출 results/stage3-eval-amitgt-<run>-<태그>.
#   amiact(2026-10-26, E5): 슬롯 활동 입력 체크포인트(adapter_slot)를 ami 와 같은 창에서 정답 슬롯 활동을 오디오 토큰에 넣어 한 스트림으로 잰다(cst_stage3_eval --act-oracle).
#     출력이 한 스트림이라 귀속 오류 분해도 돈다. 산출 results/stage3-eval-amiact-<run>-<태그> · stage3-attr-errors-amiact-<run>-<태그>.
#   amie3(2026-10-25, E3): amitgt 와 같되 정답 대신 화자 분리 가지(E1)의 예측 활동(E3_PROBS, 기본 results/diar-eval-ami-q17-diar-e1-final/probs — diar_eval
#     --save-probs)을 임계 E3_THR(기본 0.45 — E1 AMI dev 최적 짝 DER 이 가장 낮은 임계, IHM-mix·SDM 같다)로 0/1 로 만들어 예측 슬롯마다 주입한다(cst_stage3_eval --target-act). 산출 results/stage3-eval-amie3-t<임계>-<run>-<태그>.
set -u
K=${1:?작업}; RUN=${2:?run}; t=${3:?대상}; SH=${4:-0/1}
SD0=${VAPASR_STAGE_DIR:-}                                                                   # 부르는 쪽(평가 노드) 값 — .env 의 /scratch 는 컨테이너 전용이라 덮이면 안 된다
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=${SD0:-/tmp/sa_tskim/vapasr-stage}   # 2026-10-23 전에는 .env 값이 이겨 "stage 실패 → 원본에서 읽음"(NFS 직접)
P=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}; R=/soundai/users/tskim/VAPKT-data/results; D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2
DL=${DELTAS:-4,10}; [[ $DL =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo "DELTAS 는 쉼표 목록: $DL"; exit 2; }; DS=""; [ "$DL" = "4,10" ] || DS="-d${DL//,/-}"   # 2026-10-28: AMI 작업 δ(산출 이름에 -d10 등)
S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16; XE=$D/sim-dialogue-xl-eval-v1; TAXI=$D/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl; DEV=$D/v3-dev
CH=$D/evaluation/single-turn-callhome-de-v1/manifest.jsonl; X=raw/sources/experiments/2026-10-21-v3-attribution
OUT=${MODEL_ROOT:-/soundai/Model/VAPASR}/stage3-$RUN; CK=$OUT/$t; tag=${t#checkpoint-}; [ "$t" != final ] && tag=ck$tag; O=$R/stage4-eval-$RUN-$tag; E3=$R/stage3-eval-$RUN-$tag
say() { echo "[$(date '+%F %T')] $K $RUN/$t(GPU ${GPU_IDX:-${CUDA_VISIBLE_DEVICES:-?}}): $*"; }
if [ "$t" = final ]; then until [ -f $OUT/DONE ]; do sleep 60; done; else until [ -f $CK/trainer_state.json ]; do sleep 60; done; fi
dev() {                                                                                     # 전화 dev·짝 비교 셋(v3_dev_decode.sh 와 같다, bias 없음)
  local o=$R/v3dev-$RUN-$tag-$1-sessions; [ -f $o/turn-metrics-d10.json ] && { say "이미 끝 — 건너뜀 $o"; return 0; }; mkdir -p $o
  $P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $DEV/$1/sessions.jsonl --out $o --deltas 10 --max-window 90 --batch-size 8 > $o/run.log 2>&1
  $P experiments/cst_stage4_taxi.py score --sessions $DEV/$1/sessions.jsonl --out $o --turn-metrics > $o/summary.log 2>&1; }
say "시작"; mkdir -p $O
case $K in
  dlg) [ -f $O/dlg/summary.json ] && { say "이미 끝"; exit 0; }
       $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --trans $S1 --deltas 4,10 --limit-windows 100 \
         --trans-lead-space --batch-size 32 --backchannel 1.0,3 --out $O/dlg >> $O/dlg.log 2>&1
       $P experiments/cst_stage4_eval.py summarize --out $O/dlg > $O/dlg-summary.log 2>&1 ;;
  xl) [ -f $O/xl/summary.json ] && { say "이미 끝"; exit 0; }
      $P experiments/cst_stage4_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $XE --trans $XE/targets-sim_en-de --corpora sim_en-de --deltas 4,10 --limit-windows 200 \
        --trans-lead-space --batch-size 32 --out $O/xl >> $O/xl.log 2>&1
      $P experiments/cst_stage4_eval.py summarize --out $O/xl > $O/xl-summary.log 2>&1 ;;
  taxi40) [ -f $O/taxi/report-d10.json ] && { say "이미 끝"; exit 0; }
          $P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O/taxi --deltas 4,10 > $O/taxi.log 2>&1
          $P experiments/cst_stage4_taxi.py score --sessions $TAXI --out $O/taxi > $O/taxi-summary.log 2>&1 ;;
  stage3) [ -f $E3/summary.json ] && { say "이미 끝"; exit 0; }; mkdir -p $E3 $R/stage3-attr-errors-$RUN-$tag
          $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $P2 --split $P2/splits/v2.json --out $E3 --deltas 4,10 --limit-windows 200 \
            --batch-size 32 --verify --backchannel 1.0,3 >> $E3/run.log 2>&1
          $P experiments/cst_stage3_eval.py summarize --out $E3 > $E3/summary.log 2>&1
          $P experiments/cst_attr_errors.py --pred $E3 --data $P2 --split $P2/splits/v2.json --out $R/stage3-attr-errors-$RUN-$tag --deltas 4,10 --workers 16 >> $R/stage3-attr-errors-$RUN-$tag/run.log 2>&1 ;;
  regress) G=$R/stage3-regress-$RUN-$tag
           if [ -f $G/libri-other/summary.json ]; then say "단일 화자 이미 끝"; else bash $R/stage3-regress/stage2_regress.sh $CK $RUN-$tag ${GPU_IDX:-0} > $R/stage3-regress/run-$RUN-$tag.log 2>&1; fi
           if [ -f $CH ] && [ ! -f $G/callhome-de/summary.json ]; then mkdir -p $G/callhome-de
             $P experiments/eval_single_turn_asr.py run --manifest $CH --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --out $G/callhome-de --deltas 4 10 --batch-size 64 > $G/callhome-de/run.log 2>&1
             $P experiments/eval_single_turn_asr.py summarize --out $G/callhome-de > $G/callhome-de/summary.log 2>&1; fi ;;
  xltel) dev xl-tel-dev ;;
  xleval) dev xl-eval-v1 ;;
  taxi90) [ -f $O/taxi90/turn-metrics-d10.json ] && { say "이미 끝 — 건너뜀"; exit 0; }
          $P experiments/cst_stage4_taxi.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $O/taxi90 --deltas 4,10 --max-window 90 --batch-size 8 > $O/taxi90.log 2>&1
          $P experiments/cst_stage4_taxi.py score --sessions $TAXI --out $O/taxi90 --turn-metrics > $O/taxi90-summary.log 2>&1 ;;
  onset) o=$R/taxi-v3onset-$RUN-$tag; [ -f $o/onset-classes-d10.json ] && { say "이미 끝 — 건너뜀"; exit 0; }; mkdir -p $o
         $P $X/taxi_v3_probe.py --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --sessions $TAXI --out $o --deltas 10 --max-window 90 --classify-onsets > $o/run.log 2>&1 ;;
  ami) r=${SH%/*}; w=${SH#*/}; [[ $r =~ ^[0-9]+$ && $w =~ ^[1-9][0-9]*$ && $r -lt $w ]] || { say "몫은 r/w(0 ≤ r < w)"; exit 2; }
       MO=""; [ "${MIXOFF:-0}" = 1 ] && MO=--mix-off                                                   # E10(2026-10-29): 섞인 소리 토큰 없이
       NO=""; NT=""; [ -n "${N3D:-}" ] && { NO="--n3d $N3D --n3d-chunk ${N3D_CHUNK:-4}"; NT=-n3d; [ "${N3D_CHUNK:-4}" = 4 ] || NT=-n3dc$N3D_CHUNK; [ -n "${N3D_RC:-}" ] && { NO="$NO --n3d-rc $N3D_RC"; NT=${NT}r$N3D_RC; }; }   # E11(2026-10-29): 고정 화자 분리 전문가
       GO=""; GT=""; [ -n "${GATE:-}" ] && { [[ $GATE =~ ^[0-9]+,[0-9]+$ ]] || { say "GATE 는 BACK,FWD: $GATE"; exit 2; }; GO="--act-gate $GATE"; GT=-gate${GATE/,/-}; }   # 2026-10-30 활동 게이트
       EO=""; ET=""; [ -n "${ENC_STATE:-}" ] && { [ -f "$ENC_STATE" ] || { say "ENC_STATE 없음: $ENC_STATE"; exit 2; }; EO="--encoder-state $ENC_STATE"; ET=-enc${ENC_TAG:-u3}; }   # U3 인코더
       XO=""; XT=""; [ -n "${GRAFT_EXPERT:-}" ] && { [ -f "$GRAFT_EXPERT" ] || { say "GRAFT_EXPERT 없음: $GRAFT_EXPERT"; exit 2; }; XO="--graft-expert $GRAFT_EXPERT"; XT=-gx${GX_TAG:-g2}; }   # G3 이식 학생 활동
       RO=""; RT=""; [ -n "${ENC_RC:-}" ] && { RO="--encoder-rc $ENC_RC"; RT=-rc$ENC_RC; }   # 지연 완화: 인코더 우측 문맥
       GLO=""; GLT=""; [ -n "${GATE_LAGS:-}" ] && { [[ $GATE_LAGS =~ ^[0-9]+$ ]] || { say "GATE_LAGS 는 0 이상 정수: $GATE_LAGS"; exit 2; }; GLO="--gate-lags $GATE_LAGS"; GLT=-gl$GATE_LAGS; }   # 2026-11-03 E13 게이트 창(0 = 끔)
       [ -n "${GATE_THRESH:-}" ] && { [[ $GATE_THRESH =~ ^0\.[0-9]+$ ]] || { say "GATE_THRESH 는 0.x: $GATE_THRESH"; exit 2; }; GLO="$GLO --gate-thresh $GATE_THRESH"; GLT=$GLT-gt${GATE_THRESH/./}; }   # 게이트만 낮은 임계
       E=$R/stage3-eval-ami$DS${MO:+-mixoff}$NT$GT$ET$XT$RT$GLT${EV_TAG:+-$EV_TAG}-$RUN-$tag; A=$R/stage3-attr-errors-ami$DS${MO:+-mixoff}$NT$GT$ET$XT$RT$GLT${EV_TAG:+-$EV_TAG}-$RUN-$tag; AB=$D/ami-bench-v1
       FO=""; if [ "${FOLLOW:-0}" = 1 ]; then                                                        # 2026-10-30 활동 추종: 끝난 원출력을 다시 채점(디코드 없음)
         [ -z "$GO$EO$XO$RO$GLO" ] && [ -f $E/summary.json ] || { say "FOLLOW: GATE·ENC_STATE·GRAFT_EXPERT·ENC_RC·GATE_LAGS·GATE_THRESH 없이, 끝난 원 산출이 있어야 한다 $E"; exit 2; }; FO="--from-predictions $E --follow-n3d"; NO=""; E=$E-follow; fi
       [ -f $E/summary.json ] && { say "이미 끝"; exit 0; }; mkdir -p $E; [ -f $E/.rank$r-of-$w.done ] && { say "몫 $r/$w 이미 끝"; exit 0; }
       $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $AB --corpora ami-ihmmix,ami-sdm --limit-windows 0 --deltas $DL --batch-size 16 \
         --backchannel 1.0,3 --max-speakers 4 --window 20,90 --window-draws 1 --declare 4 $MO $NO $GO $EO $XO $RO $GLO $FO --rank $r --world-size $w --out $E >> $E/run-rank$r.log 2>&1
       grep -q "^DONE rank=$r " $E/run-rank$r.log || { say "몫 $r/$w 실패 — $E/run-rank$r.log"; exit 1; }
       touch $E/.rank$r-of-$w.done; n=$(ls $E/.rank*-of-$w.done 2>/dev/null | wc -l)
       [ $n -lt $w ] && { say "몫 $r/$w 끝(끝난 몫 $n/$w — 합치기는 마지막 몫이)"; exit 0; }
       ( flock 9; [ -f $E/summary.json ] && exit 0
         $P experiments/cst_stage3_eval.py summarize --out $E > $E/summary.log 2>&1
         [ -n "$FO" ] || { mkdir -p $A; $P experiments/cst_attr_errors.py --pred $E --data $AB --out $A --deltas $DL --workers 16 > $A/run.log 2>&1; }
         $P raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py $E > $E/ami-report.txt 2>&1 ) 9> $E/.sum.lock
       say "AMI 합침 → $E/ami-report.txt" ;;
  amiact) r=${SH%/*}; w=${SH#*/}; [[ $r =~ ^[0-9]+$ && $w =~ ^[1-9][0-9]*$ && $r -lt $w ]] || { say "몫은 r/w(0 ≤ r < w)"; exit 2; }
       E=$R/stage3-eval-amiact$DS-$RUN-$tag; A=$R/stage3-attr-errors-amiact$DS-$RUN-$tag; AB=$D/ami-bench-v1
       [ -f $E/summary.json ] && { say "이미 끝"; exit 0; }; mkdir -p $E; [ -f $E/.rank$r-of-$w.done ] && { say "몫 $r/$w 이미 끝"; exit 0; }
       $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $AB --corpora ami-ihmmix,ami-sdm --limit-windows 0 --deltas $DL --batch-size 16 \
         --backchannel 1.0,3 --max-speakers 4 --window 20,90 --window-draws 1 --declare 4 --act-oracle --rank $r --world-size $w --out $E >> $E/run-rank$r.log 2>&1
       grep -q "^DONE rank=$r " $E/run-rank$r.log || { say "몫 $r/$w 실패 — $E/run-rank$r.log"; exit 1; }
       touch $E/.rank$r-of-$w.done; n=$(ls $E/.rank*-of-$w.done 2>/dev/null | wc -l)
       [ $n -lt $w ] && { say "몫 $r/$w 끝(끝난 몫 $n/$w — 합치기는 마지막 몫이)"; exit 0; }
       ( flock 9; [ -f $E/summary.json ] && exit 0
         $P experiments/cst_stage3_eval.py summarize --out $E > $E/summary.log 2>&1
         mkdir -p $A; $P experiments/cst_attr_errors.py --pred $E --data $AB --out $A --deltas $DL --workers 16 > $A/run.log 2>&1
         $P raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py $E > $E/ami-report.txt 2>&1 ) 9> $E/.sum.lock
       say "AMI 정답 슬롯 활동 입력(E5) 합침 → $E/ami-report.txt" ;;
  amitgt) r=${SH%/*}; w=${SH#*/}; [[ $r =~ ^[0-9]+$ && $w =~ ^[1-9][0-9]*$ && $r -lt $w ]] || { say "몫은 r/w(0 ≤ r < w)"; exit 2; }
       E=$R/stage3-eval-amitgt$DS-$RUN-$tag; AB=$D/ami-bench-v1
       [ -f $E/summary.json ] && { say "이미 끝"; exit 0; }; mkdir -p $E; [ -f $E/.rank$r-of-$w.done ] && { say "몫 $r/$w 이미 끝"; exit 0; }
       $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $AB --corpora ami-ihmmix,ami-sdm --limit-windows 0 --deltas $DL --batch-size 16 \
         --backchannel 1.0,3 --max-speakers 4 --window 20,90 --window-draws 1 --declare 4 --target-oracle --rank $r --world-size $w --out $E >> $E/run-rank$r.log 2>&1
       grep -q "^DONE rank=$r " $E/run-rank$r.log || { say "몫 $r/$w 실패 — $E/run-rank$r.log"; exit 1; }
       touch $E/.rank$r-of-$w.done; n=$(ls $E/.rank*-of-$w.done 2>/dev/null | wc -l)
       [ $n -lt $w ] && { say "몫 $r/$w 끝(끝난 몫 $n/$w — 합치기는 마지막 몫이)"; exit 0; }
       ( flock 9; [ -f $E/summary.json ] && exit 0
         $P experiments/cst_stage3_eval.py summarize --out $E > $E/summary.log 2>&1
         $P raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py $E > $E/ami-report.txt 2>&1 ) 9> $E/.sum.lock
       say "AMI 대상 화자(정답 활동) 합침 → $E/ami-report.txt" ;;
  amie3) r=${SH%/*}; w=${SH#*/}; [[ $r =~ ^[0-9]+$ && $w =~ ^[1-9][0-9]*$ && $r -lt $w ]] || { say "몫은 r/w(0 ≤ r < w)"; exit 2; }
       TH=${E3_THR:-0.45}; PD=${E3_PROBS:-$R/diar-eval-ami-q17-diar-e1-final/probs}; E=$R/stage3-eval-amie3-t$TH$DS-$RUN-$tag; AB=$D/ami-bench-v1
       [ -n "$(ls -A $PD 2>/dev/null)" ] || { say "예측 활동이 없다(diar_eval run --save-probs): $PD"; exit 2; }
       [ -f $E/summary.json ] && { say "이미 끝"; exit 0; }; mkdir -p $E; [ -f $E/.rank$r-of-$w.done ] && { say "몫 $r/$w 이미 끝"; exit 0; }
       $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $AB --corpora ami-ihmmix,ami-sdm --limit-windows 0 --deltas $DL --batch-size 16 \
         --backchannel 1.0,3 --max-speakers 4 --window 20,90 --window-draws 1 --declare 4 --target-oracle --target-act $PD --target-act-threshold $TH \
         --rank $r --world-size $w --out $E >> $E/run-rank$r.log 2>&1
       grep -q "^DONE rank=$r " $E/run-rank$r.log || { say "몫 $r/$w 실패 — $E/run-rank$r.log"; exit 1; }
       touch $E/.rank$r-of-$w.done; n=$(ls $E/.rank*-of-$w.done 2>/dev/null | wc -l)
       [ $n -lt $w ] && { say "몫 $r/$w 끝(끝난 몫 $n/$w — 합치기는 마지막 몫이)"; exit 0; }
       ( flock 9; [ -f $E/summary.json ] && exit 0
         $P experiments/cst_stage3_eval.py summarize --out $E > $E/summary.log 2>&1
         $P raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py $E > $E/ami-report.txt 2>&1 ) 9> $E/.sum.lock
       say "AMI 대상 화자(E1 예측 활동, 임계 $TH) 합침 → $E/ami-report.txt" ;;
  amifull) r=${SH%/*}; w=${SH#*/}; [[ $r =~ ^[0-9]+$ && $w =~ ^[1-9][0-9]*$ && $r -lt $w ]] || { say "몫은 r/w(0 ≤ r < w)"; exit 2; }
       AO=""; [ "${AOSC:-0}" = 1 ] && AO=--aosc; DO=""; DT=""; [ -n "${DIAR:-}" ] && { DO="--diar-head-override $DIAR"; DT="-${DIAR_TAG:-diar}"; }   # E7(2026-10-27): AOSC·헤드 바꿔 끼우기
       GD=""; [ "${GUIDE:-0}" = 1 ] && GD=--act-tag-guide                                                # E7(2026-10-28): 화자 태그를 활동으로
       MO=""; [ "${MIXOFF:-0}" = 1 ] && MO=--mix-off                                                   # E10(2026-10-29): 섞인 소리 토큰 없이
       NO=""; NT=""; [ -n "${N3D:-}" ] && { NO="--n3d $N3D --n3d-chunk ${N3D_CHUNK:-4}"; NT=-n3d; [ "${N3D_CHUNK:-4}" = 4 ] || NT=-n3dc$N3D_CHUNK; [ -n "${N3D_RC:-}" ] && { NO="$NO --n3d-rc $N3D_RC"; NT=${NT}r$N3D_RC; }; }   # E11(2026-10-29): 고정 화자 분리 전문가
       [ -n "$NO" ] && [ -n "$AO$DO$GD" ] && { say "N3D 는 AOSC·DIAR·GUIDE 와 함께 쓰지 않는다"; exit 2; }
       GO=""; GT=""; [ -n "${GATE:-}" ] && { [[ $GATE =~ ^[0-9]+,[0-9]+$ ]] || { say "GATE 는 BACK,FWD: $GATE"; exit 2; }; GO="--act-gate $GATE"; GT=-gate${GATE/,/-}; }   # 2026-10-30 활동 게이트
       EO=""; ET=""; [ -n "${ENC_STATE:-}" ] && { [ -f "$ENC_STATE" ] || { say "ENC_STATE 없음: $ENC_STATE"; exit 2; }; EO="--encoder-state $ENC_STATE"; ET=-enc${ENC_TAG:-u3}; }   # U3 인코더
       XO=""; XT=""; [ -n "${GRAFT_EXPERT:-}" ] && { [ -f "$GRAFT_EXPERT" ] || { say "GRAFT_EXPERT 없음: $GRAFT_EXPERT"; exit 2; }; XO="--graft-expert $GRAFT_EXPERT"; XT=-gx${GX_TAG:-g2}; }   # G3 이식 학생 활동
       RO=""; RT=""; [ -n "${ENC_RC:-}" ] && { RO="--encoder-rc $ENC_RC"; RT=-rc$ENC_RC; }   # 지연 완화: 인코더 우측 문맥
       GLO=""; GLT=""; [ -n "${GATE_LAGS:-}" ] && { [[ $GATE_LAGS =~ ^[0-9]+$ ]] || { say "GATE_LAGS 는 0 이상 정수: $GATE_LAGS"; exit 2; }; GLO="--gate-lags $GATE_LAGS"; GLT=-gl$GATE_LAGS; }   # 2026-11-03 E13 게이트 창(0 = 끔)
       [ -n "${GATE_THRESH:-}" ] && { [[ $GATE_THRESH =~ ^0\.[0-9]+$ ]] || { say "GATE_THRESH 는 0.x: $GATE_THRESH"; exit 2; }; GLO="$GLO --gate-thresh $GATE_THRESH"; GLT=$GLT-gt${GATE_THRESH/./}; }   # 게이트만 낮은 임계
       C=${CTX:-30}; E=$R/stage3-eval-amifull-c$C${AO:+-aosc}$DT${GD:+-guide}${MO:+-mixoff}$NT$GT$ET$XT$RT$GLT${EV_TAG:+-$EV_TAG}$DS-$RUN-$tag; AB=$D/ami-bench-v1; V=""; [ $r = 0 ] && [ -z "$AO" ] && V=--verify
       FO=""; if [ "${FOLLOW:-0}" = 1 ]; then                                                        # 2026-10-30 활동 추종: 끝난 원출력을 다시 채점(디코드 없음)
         [ -z "$GO$EO$XO$RO$GLO" ] && [ -f $E/summary.json ] || { say "FOLLOW: GATE·ENC_STATE·GRAFT_EXPERT·ENC_RC·GATE_LAGS·GATE_THRESH 없이, 끝난 원 산출이 있어야 한다 $E"; exit 2; }; FO="--from-predictions $E --follow-n3d"; NO=""; V=""; E=$E-follow; fi
       [ -f $E/summary.json ] && { say "이미 끝"; exit 0; }; mkdir -p $E; [ -f $E/.rank$r-of-$w.done ] && { say "몫 $r/$w 이미 끝"; exit 0; }
       $P experiments/cst_stage3_eval.py run --checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data $AB --corpora ami-ihmmix,ami-sdm --limit-windows 0 --deltas $DL \
         --backchannel 1.0,3 --max-speakers 4 --window full --context-s $C --declare 4 $V $AO $DO $GD $MO $NO $GO $EO $XO $RO $GLO $FO --rank $r --world-size $w --out $E >> $E/run-rank$r.log 2>&1
       grep -q "^DONE rank=$r " $E/run-rank$r.log || { say "몫 $r/$w 실패 — $E/run-rank$r.log"; exit 1; }
       touch $E/.rank$r-of-$w.done; n=$(ls $E/.rank*-of-$w.done 2>/dev/null | wc -l)
       [ $n -lt $w ] && { say "몫 $r/$w 끝(끝난 몫 $n/$w — 합치기는 마지막 몫이)"; exit 0; }
       ( flock 9; [ -f $E/summary.json ] && exit 0
         $P experiments/cst_stage3_eval.py summarize --out $E > $E/summary.log 2>&1
         $P raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py $E > $E/ami-report.txt 2>&1 ) 9> $E/.sum.lock
       say "AMI 통째 합침 → $E/ami-report.txt" ;;
  *) say "모르는 작업"; exit 2 ;;
esac
say "끝"

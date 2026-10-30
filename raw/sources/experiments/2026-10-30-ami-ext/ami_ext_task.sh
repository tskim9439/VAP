# 외부 기준 AMI 작업 하나(2026-10-30, 사용자 "이 둘을 준비해줘" — experiments/baseline_ami_ext.py): bash ami_ext_task.sh <시스템> <r/w>
#   GPU 는 부르는 쪽이 CUDA_VISIBLE_DEVICES 로 준다(slurm/v3-eval-node.sbatch 일꾼). 몫 r/w 로 회의(또는 조각)를 나눠 r 번째만 돈다(이어 하기 됨).
#   시스템: vv7b-full   VibeVoice-ASR-Streaming-7B, 회의 통째(우리 조건)
#          vv7b-seg480 같은 모델, 480 s 이하 조각마다 따로(논문 AMI 조건 — 보고값 IHM 27.48·SDM 39.01 재현 점검)
#          mtp-sortv2  Multitalker Parakeet v1 + Streaming Sortformer v2, 회의 통째(카드 조건 — 보고값 21.26·37.44 재현 점검)
#          mtp-n3d     Multitalker Parakeet v1 + Nemotron 3 Diarization(우리 E11 과 같은 화자 분리기), 회의 통째
#   참조·오디오·마스크는 우리 AMI 회의 통째 평가(v3_eval_task.sh amifull, δ10)와 같다(참조 tokenizer = E11 final). 마지막 몫이 summarize·ami_report·meeteval(whisper·ours)을 한다.
#   산출 VAPKT-data/results/ext-ami-<시스템>. Parakeet 회의 wav 는 노드 로컬 $TMPDIR(기본 /tmp/sa_tskim)/ami-ext-wav-<시스템> 에 쓴다. 아무것도 지우지 않는다.
set -u
SYS=${1:?시스템}; SH=${2:-0/1}; r=${SH%/*}; w=${SH#*/}
[[ $r =~ ^[0-9]+$ && $w =~ ^[1-9][0-9]*$ && $r -lt $w ]] || { echo "몫은 r/w(0 ≤ r < w)"; exit 2; }
SD0=${VAPASR_STAGE_DIR:-}
CODE=/soundai/users/tskim/VAPKT; cd $CODE; set -a; source .env; set +a
export PYTHONPATH=$CODE PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_STAGE_DIR=${SD0:-/tmp/sa_tskim/vapasr-stage}
P=${PY:-/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python}; D=/soundai/users/tskim/VAPKT-data; R=$D/results; AB=$D/data/ami-bench-v1
CK=/soundai/Model/VAPASR/stage3-q17-e11/final; V=$D/models/vibevoice; VB=$D/vapkt-baselines
MTP=$VB/multitalker-parakeet-v1/multitalker-parakeet-streaming-0.6b-v1.nemo   # 폴더 이름은 업로드 전 조회하지 않은 새 이름(NFS "없음" 캐시)
say() { echo "[$(date '+%F %T')] ami-ext $SYS $r/$w(GPU ${GPU_IDX:-${CUDA_VISIBLE_DEVICES:-?}}): $*"; }
case $SYS in
  vv7b-full|vv7b-seg480) SRC=$(ls -d $V/src/VibeVoice-* | head -1); export PYTHONPATH=$CODE:$V/pydeps
     X=(--system vibevoice --model $V/VibeVoice-ASR-Streaming-7B --src $SRC); [ $SYS = vv7b-seg480 ] && X+=(--segment-s 480) ;;
  mtp-sortv2) X=(--system parakeet --asr $MTP --diar $VB/diar_streaming_sortformer_4spk-v2/diar_streaming_sortformer_4spk-v2.nemo --diar-kind sortformer) ;;
  mtp-n3d) X=(--system parakeet --asr $MTP --diar $VB/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo --diar-kind n3d) ;;
  *) say "모르는 시스템"; exit 2 ;;
esac
[[ $SYS == mtp-* ]] && { for f in $MTP ${X[5]}; do [ -f $f ] || { say "없음(사용자 업로드): $f"; exit 2; }; done; X+=(--wav-dir ${TMPDIR:-/tmp/sa_tskim}/ami-ext-wav-$SYS); }
E=$R/ext-ami-$SYS; mkdir -p $E
[ -f $E/summary.json ] && { say "이미 끝"; exit 0; }; [ -f $E/.rank$r-of-$w.done ] && { say "몫 $r/$w 이미 끝"; exit 0; }
say "시작"
$P experiments/baseline_ami_ext.py run "${X[@]}" --checkpoint $CK --data $AB --corpora ami-ihmmix,ami-sdm --deltas 10 --rank $r --world-size $w --out $E >> $E/run-rank$r.log 2>&1
grep -q "^DONE rank=$r " $E/run-rank$r.log || { say "몫 $r/$w 실패 — $E/run-rank$r.log"; exit 1; }
touch $E/.rank$r-of-$w.done; n=$(ls $E/.rank*-of-$w.done 2>/dev/null | wc -l)
[ $n -lt $w ] && { say "몫 $r/$w 끝(끝난 몫 $n/$w — 합치기는 마지막 몫이)"; exit 0; }
( flock 9; [ -f $E/summary.json ] && exit 0
  $P experiments/cst_stage3_eval.py summarize --out $E > $E/summary.log 2>&1
  $P raw/sources/experiments/2026-10-20-v2-multispeaker/ami_report.py $E > $E/ami-report.txt 2>&1
  for nm in whisper ours; do R_=(); [ $nm = ours ] && R_=(--refs "")
    $P experiments/meeteval_cpwer.py --pred $E --norm $nm "${R_[@]}" --json $E/meeteval-$nm.json > $E/meeteval-$nm.txt 2>&1; done ) 9> $E/.sum.lock
say "합침 → $E/ami-report.txt · meeteval-whisper.txt"

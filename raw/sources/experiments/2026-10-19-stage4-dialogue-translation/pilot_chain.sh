# Stage 4 대화 번역 시험 생성(2026-10-19): 코퍼스당 대화 3 개를 단위로(stage4_dialogue_units.py) → Qwen3.8-27B(vLLM 0.30, 호스트 GPU 하나,
# 다른 사용자와 같이 쓰므로 메모리 0.55) 생성 v3 → 판정 v3·한 번 수리(fix) → status. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; R=$D/stage4-dlg-units-pilot
export VLLM_ENV=/tmp/sa_tskim/vllm-0.30
bash scripts/setup-vllm-env.sh stage 2>&1 | tail -2
PYTHONPATH=. HF_HUB_OFFLINE=1 /soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python experiments/stage4_dialogue_units.py --data $D/phase2 \
  --corpora aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc --root $R --limit 3 2>&1 | grep -v "regex pattern" | tail -1
GPU=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print $1, $3 - $2}' | sort -k2,2nr | head -1 | cut -d' ' -f1)
echo "[$(date +%T)] GPU $GPU ($(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i $GPU))"
export CUDA_VISIBLE_DEVICES=$GPU PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 VLLM_USE_FLASHINFER_SAMPLER=0 PATH="$VLLM_ENV/bin:$PATH"
C=/tmp/sa_tskim/cache/stage4-pilot; mkdir -p $C; export VLLM_CACHE_ROOT=$C/vllm TRITON_CACHE_DIR=$C/triton XDG_CACHE_HOME=$C TORCHINDUCTOR_CACHE_DIR=$C/inductor
$VLLM_ENV/bin/python -u experiments/ast_mono_generate.py run --root $R --prompt v3 --gen-dir gen-v3-bf16 --gpu-memory-utilization 0.55 --max-num-seqs 256 --rank 0 --world 1 2>&1 | grep -E "ITEM_|rank 0" 
$VLLM_ENV/bin/python -u experiments/ast_mono_generate.py fix --root $R --gen-dir gen-v3-bf16 --fix-dir fix-gen-v3-bf16 --gpu-memory-utilization 0.55 --max-num-seqs 256 --rank 0 --world 1 2>&1 | grep -vE "INFO|WARNING|^\s*$" | tail -8
$VLLM_ENV/bin/python experiments/ast_mono_generate.py status --root $R --gen-dir gen-v3-bf16 2>&1 | tail -8
echo PILOT_DONE

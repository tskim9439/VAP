# E3 점검(mxc 호스트, 2026-10-25): cst_stage3_eval --target-oracle 의 대상 화자 루프를 예측 활동(--target-act, E1 확률)도 받게 고친 뒤
#   (1) 정답 활동 경로가 그대로인지 — E2 점검 때(고치기 전 코드) 같은 모델·같은 창으로 낸 /tmp/sa_tskim/e2-eval-smoke 와 행을 대조하고,
#   (2) 예측 활동 경로가 도는지 — 같은 창에서 예측 슬롯·cp 를 본다. 모델은 E2 스모크(/tmp/sa_tskim/e2-smoke-train/final, 20 step — 수치는 의미 없다), AMI SDM 6 창, δ10.
#   GPU=4 bash e3_check.sh [임계(기본 0.5)]. 산출 /tmp/sa_tskim/e3-check-{oracle,pred-t<임계>} — 지우지 않는다(있으면 이어 한다).
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false VAPASR_LOCAL_CACHE=/tmp/sa_tskim
export VAPASR_STAGE_DIR=/tmp/sa_tskim/vapasr-stage CUDA_VISIBLE_DEVICES=${GPU:-0}
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; TH=${1:-0.5}; CK=/tmp/sa_tskim/e2-smoke-train/final
PD=/soundai/users/tskim/VAPKT-data/results/diar-eval-ami-q17-diar-e1-final/probs
COMMON="--checkpoint $CK --encoder $MXC_NEMOTRON_DIR --data /soundai/users/tskim/VAPKT-data/data/ami-bench-v1 --corpora ami-sdm --limit-windows 6 --deltas 10
  --batch-size 16 --backchannel 1.0,3 --max-speakers 4 --window 20,90 --window-draws 1 --declare 4 --target-oracle"
echo "[$(date "+%F %T")] e3 check: 정답 활동(회귀)"
$P experiments/cst_stage3_eval.py run $COMMON --out /tmp/sa_tskim/e3-check-oracle 2>&1 | grep -v "regex pattern" | tail -3
echo "[$(date "+%F %T")] e3 check: 예측 활동(임계 $TH)"
$P experiments/cst_stage3_eval.py run $COMMON --target-act $PD --target-act-threshold $TH --out /tmp/sa_tskim/e3-check-pred-t$TH 2>&1 | grep -v "regex pattern" | tail -3
python3 - /tmp/sa_tskim/e2-eval-smoke /tmp/sa_tskim/e3-check-oracle /tmp/sa_tskim/e3-check-pred-t$TH <<'EOF'
import json, sys
rows = lambda d: {(r["id"], r["delta"]): r for r in map(json.loads, open(f"{d}/predictions-rank0.jsonl"))}
old, new, pred = (rows(d) for d in sys.argv[1:4]); keys = ("hyp_text", "ref_text", "slot", "cp", "agn", "n_ref", "n_hyp", "spill", "perm")
same = sorted(old) == sorted(new) and all(old[k][q] == new[k][q] for k in old for q in keys)
print(f"정답 활동 회귀: 행 {len(old)} 대 {len(new)} · 같은 값 {same}")
for k in old:
    if any(old[k][q] != new[k].get(q) for q in keys): print("  다름", k, [q for q in keys if old[k][q] != new[k].get(q)])
for k, r in sorted(pred.items()):
    o = new.get(k, {})
    print(f"  {k[0]:28s} 참조 화자 {r['n_spk']} · 예측 슬롯 {r.get('hyp_slots')} · n_ref {r['n_ref']} · cp {r['cp']}(정답 활동 {o.get('cp')}) · agn {r['agn']}({o.get('agn')})")
print("E3_CHECK_DONE")
EOF

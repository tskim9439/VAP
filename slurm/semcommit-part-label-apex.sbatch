#!/usr/bin/env bash
# speechlm semcommit teacher labeling, part-unit (all 19 DBs). Submit with slurm/submit-semcommit-label-v035-apex-n2.sh.
# Every part is finished end to end before the next (DONE.json), parts alternate across DBs, so a partial run is a usable
# training set at any time (experiments/semcommit_collect_done.py). Resubmit the same script to continue; do not run two at once.
#SBATCH --job-name=SA_SFT_FullDuplex
#SBATCH --partition=apex
#SBATCH --nodes=2
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --exclude=slurmmxch200v5-hpc-52
#SBATCH --output=/soundai/users/tskim/VAPKT/logs/train/%x-%j.out
#SBATCH --error=/soundai/users/tskim/VAPKT/logs/train/%x-%j.err
set -euo pipefail

PROJECT=/soundai/users/tskim/VAPKT
PY=/soundai/users/tskim/VAPKT-data/conda/envs/semcommit-teacher/bin/python
QWEN=/soundai/Model/Qwen3.8-27B
EXAONE=/soundai/Model/EXAONE-4.0-32B
QWEN_ASR=/soundai/Model/Qwen3-ASR-0.6B
: "${RESULTS:?speechlm alignment results directory required}"
: "${QC_SPLIT:?semcommit_split_qc_pass.py output directory required}"
: "${APPROVAL_SUMMARY:?gate-passed training-eligibility summary required}"
: "${GATE_ACCEPTED:?accepted gold gate attestation required}"
: "${THRESHOLDS:?accepted gate thresholds.json required}"
: "${A_KIND:?qwen38 or exaone4 required}"
: "${C_KIND:?qwen38 or exaone4 required}"
: "${OUT:?labels root required}"
export A_KIND C_KIND
cd "$PROJECT"
test -x "$PY" && test -d "$RESULTS" && test -f "$QC_SPLIT/parts-order.tsv" && test -f "$APPROVAL_SUMMARY"
test -f "$GATE_ACCEPTED" && test -f "$THRESHOLDS" && test -d "$QWEN" && test -d "$EXAONE" && test -d "$QWEN_ASR"
test "${SLURM_JOB_NUM_NODES:?}" -ge 1 && test "${SLURM_NTASKS:?}" -eq $((SLURM_JOB_NUM_NODES * 8))
export PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PY" - "$APPROVAL_SUMMARY" "$GATE_ACCEPTED" "$THRESHOLDS" "$QC_SPLIT/summary.json" <<'PY'
import hashlib, json, os, pathlib, sys
approval, gate, thresholds, split = map(pathlib.Path, sys.argv[1:])
a, g, s = json.loads(approval.read_text()), json.loads(gate.read_text()), json.loads(split.read_text())
if not a.get("complete") or a.get("training_eligible") is not True:
    raise SystemExit("speechlm approval is not training eligible")
if g.get("accepted") is not True or g.get("thresholds_sha256") != hashlib.sha256(thresholds.read_bytes()).hexdigest():
    raise SystemExit("gold gate acceptance/threshold digest mismatch")
selected = next((r for r in g.get("results", []) if r.get("kind") == g.get("selected_kind")), None)
report = thresholds.with_name("thresholds.report.json")
if (selected is None or not report.is_file()
        or selected.get("thresholds_report_sha256") != hashlib.sha256(report.read_bytes()).hexdigest()):
    raise SystemExit("selected gate report is missing or altered")
expected_kind = {"qwen": "qwen38", "exaone": "exaone4"}.get(g.get("selected_kind"))
if not expected_kind or (expected_kind, expected_kind) != (os.environ["A_KIND"], os.environ["C_KIND"]):
    raise SystemExit("A/C model kind does not match accepted gold gate")
if (a.get("gate_acceptance_sha256") != hashlib.sha256(gate.read_bytes()).hexdigest()
        or a.get("source_approval_sha256") != g.get("source_approval_sha256") or not a.get("approval_fingerprint")):
    raise SystemExit("training eligibility is not bound to this gold gate and source approval")
if s.get("approval_fingerprint") != a["approval_fingerprint"] or s.get("decision_file_sha256") != a.get("decision_file_sha256"):
    raise SystemExit("QC split is not bound to this training eligibility")
print(f"PREFLIGHT_OK parts={s['parts']} pass_rows={s['pass_rows']}")
PY
EXTRA_CANDIDATES=$("$PY" -c "import json,pathlib,sys; print(','.join(json.loads(pathlib.Path(sys.argv[1]).with_name('thresholds.report.json').read_text())['extra']))" "$THRESHOLDS")
mkdir -p "$OUT/parts"

export PROJECT PY QWEN EXAONE QWEN_ASR RESULTS QC_SPLIT OUT THRESHOLDS A_KIND C_KIND APPROVAL_SUMMARY EXTRA_CANDIDATES
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_NTASKS" --ntasks-per-node=8 --cpus-per-task=8 \
  --gpus-per-task=1 --gpu-bind=single:1 \
  bash "$PROJECT/slurm/semcommit-part-label-worker.sh"

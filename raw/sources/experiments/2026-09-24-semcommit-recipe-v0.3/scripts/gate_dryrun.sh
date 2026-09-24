#!/usr/bin/env bash
# Dry run of experiments/semcommit_teacher_gate.sh (recipe v0.3) on the existing v0.2/v0.3 teacher rows: stage A/B/C mocked (rows pre-placed),
# tune + grade + score real, CPU only. Expect the same thresholds and gold scores as labels/v0.3.1.
set -uo pipefail
D=/data4/tskim/semcommit/gate-dryrun; mkdir -p $D/out
GD=/data4/tskim/semcommit/gold/v1; L2=/data4/tskim/semcommit/labels/v0.2; LG=/data4/tskim/semcommit/labels/v0.2-gold; L3=/data4/tskim/semcommit/labels/v0.3
declare -A BASE=([ls-test]=$L2 [ks-eval]=$L2 [gs-test]=$LG [ks-long]=$LG)
for s in ls-test gs-test ks-eval ks-long; do
  cp ${BASE[$s]}/A-$s.jsonl $D/out/A-$s.jsonl
  for j in qwen3 exaone35; do cat ${BASE[$s]}/B.$j-$s.jsonl $L3/B.$j-$s.x.jsonl > $D/out/B.$j-$s.jsonl; done
  cat ${BASE[$s]}/C-$s.jsonl $L3/C-$s.x.jsonl > $D/out/C-$s.jsonl
done
cat > $D/py.sh <<'W'
#!/usr/bin/env bash
if [[ "$1" == experiments/semcommit_teacher.py && "$2" =~ ^stage[ABC]$ ]]; then echo "(mock: $2 rows pre-placed)"; exit 0; fi
exec /opt/conda/envs/vapasr/bin/python "$@"
W
chmod +x $D/py.sh
cd /home/tskim/VAP && PY=$D/py.sh GOLD=$GD OUT=$D/out A=/x:qwen3:qwen3 B="/x:qwen3:qwen3 /x:exaone35:exaone35" C=/x:qwen3:qwen3 bash experiments/semcommit_teacher_gate.sh

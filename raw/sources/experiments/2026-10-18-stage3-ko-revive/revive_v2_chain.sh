# 되살리기 v2(2026-10-19): 정렬 대상이 아니었던 격리 발화(134-1·134-2·NSC)를 정렬 → revive-asr-v2 재료(v1 은 그대로) → 로더 점검 → p12 dry-run. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data/phase2; G=/soundai/users/tskim/VAPKT-data/results/stage3-revive; PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
$PY $G/revive_extra_input.py $D $D/revive-align-v2-input aihub134-1,aihub134-2,nsc > $G/v2-input.log 2>&1; echo "입력 끝 $(date +%T)"; grep '^{' $G/v2-input.log
for c in aihub134-1 aihub134-2 nsc; do $PY experiments/p2_align.py --dialogues $D/revive-align-v2-input/$c.dialogues.jsonl --out-root $D/align-revive-v2 > $G/v2-align-$c.log 2>&1; echo "정렬 $c 끝 $(date +%T) · EXIT $? · $(tail -1 $G/v2-align-$c.log | cut -c1-160)"; done
$PY experiments/stage3_revive_tokens.py --data $D --corpora aihub71631,aihub134-1,aihub134-2,kec,ztc,nsc --out $D/revive-asr-v2 --extra-align $D/align-revive-v2 > $G/build-v2.log 2>&1; echo "재료 끝 $(date +%T) · EXIT $?"
sed 's#revive-asr-v1#revive-asr-v2#' $G/revive_probe.py > $G/revive_probe_v2.py; $PY $G/revive_probe_v2.py $G/probe-v2.json aihub134-1,aihub134-2,nsc > $G/probe-v2.log 2> $G/probe-v2.err; echo "점검 끝 $(date +%T) · EXIT $?"
sed -e 's#revive-asr-v1#revive-asr-v2#' -e 's#stage3-dry-revive#stage3-dry-revive2#' $G/stage3_dry_revive.sh > $G/stage3_dry_revive2.sh; bash $G/stage3_dry_revive2.sh > $G/dry-v2.log 2>&1; echo "dry-run 끝 $(date +%T) · $(tail -1 $G/dry-v2.log)"
echo V2_DONE

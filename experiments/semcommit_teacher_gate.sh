#!/usr/bin/env bash
# semcommit 교사 관문 — 후보 교사 구성으로 골드 스트림에 Stage A·B·C 를 돌리고 등급(A/B/N)을 매겨 골드셋과 대조한다(GPU 1 장, 모델을 하나씩 순차로).
# 본 라벨링 전에 교사·레시피를 고르는 싼 점검: 골드 스트림만(gold v1 ≈ 650 스트림, 11 k 단어) 돈다.
#
#   GOLD=<words-<set>.jsonl·gold-<set>.jsonl 가 있는 디렉터리> OUT=<출력 디렉터리> GPU=0 \
#   A=/models/Qwen3-32B:qwen3:qwen3-32b \
#   B="/models/EXAONE-4.0-32B:exaone35:exaone-32b /models/gpt-oss-120b:gptoss:gptoss-120b" \
#   C=/models/Qwen3-32B:qwen3:qwen3-32b \
#   bash experiments/semcommit_teacher_gate.sh
#
# 모델 지정 = 경로:kind[:판정자 이름] (kind ∈ qwen3 | exaone35 | gptoss — vapasr/data/semcommit_llm.KINDS). B 는 공백으로 여러 판정자(등급의 --judges-*).
# 선택: SETS(기본 "ls-test gs-test ks-eval ks-long"), MASK(기본 CONTINUATION,QUALIFICATION), BS(배치, 기본 16), STRICT(기본 --strict).
# 출력: OUT/{A,B.<이름>,C,labels}-<set>.jsonl + OUT/gate-<set>.json(score-labels) + OUT/gate.json(모음). 재실행하면 끝난 행은 건너뛴다(teacher 이어하기).
set -uo pipefail
: "${GOLD:?GOLD 디렉터리}"; : "${OUT:?OUT 디렉터리}"; : "${A:?A=경로:kind[:이름]}"; : "${B:?B=\"경로:kind[:이름] ...\"}"; : "${C:?C=경로:kind[:이름]}"
GPU=${GPU:-0}; SETS=${SETS:-"ls-test gs-test ks-eval ks-long"}; MASK=${MASK:-CONTINUATION,QUALIFICATION}; BS=${BS:-16}; STRICT=${STRICT:---strict}
cd "$(dirname "$0")/.."; mkdir -p "$OUT"
PY=${PY:-python}; T="$PY experiments/semcommit_teacher.py"
spec(){ local p k n; IFS=: read -r p k n <<< "$1"; echo "$p" "$k" "${n:-$k}"; }
step(){ echo "[$(date "+%F %T")] >>> $*"; "$@"; rc=$?; echo "[$(date "+%F %T")] <<< rc=$rc"; [ $rc -eq 0 ] || { echo "!! FAILED rc=$rc: $*"; exit $rc; }; }
read -r AP AK AN <<< "$(spec "$A")"; read -r CP CK CN <<< "$(spec "$C")"
JUDGES=(); for b in $B; do read -r _ _ n <<< "$(spec "$b")"; JUDGES+=("$n"); done; JL=$(IFS=,; echo "${JUDGES[*]}")
for s in $SETS; do
  W=$GOLD/words-$s.jsonl; [ -f "$W" ] || { echo "!! $W 없음"; exit 1; }
  step $T stageA --words "$W" --model "$AP" --kind "$AK" --judge-name "$AN" --out "$OUT/A-$s.jsonl" --gpu "$GPU" --batch-size 8
  BF=()
  for b in $B; do read -r BP BK BN <<< "$(spec "$b")"
    step $T stageB --words "$W" --stageA "$OUT/A-$s.jsonl" --model "$BP" --kind "$BK" --judge-name "$BN" --out "$OUT/B.$BN-$s.jsonl" --gpu "$GPU" --batch-size "$BS"
    BF+=("$OUT/B.$BN-$s.jsonl"); done
  step $T stageC --words "$W" --stageA "$OUT/A-$s.jsonl" --model "$CP" --kind "$CK" --judge-name "$CN" --out "$OUT/C-$s.jsonl" --gpu "$GPU" --batch-size "$BS"
  step $T grade --words "$W" --stageA "$OUT/A-$s.jsonl" --stageB "${BF[@]}" --stageC "$OUT/C-$s.jsonl" --judges-en "$JL" --judges-ko "$JL" --c-mask-types "$MASK" $STRICT \
       --out "$OUT/labels-$s.jsonl" --stats "$OUT/labels-$s.stats.json"
  $PY experiments/semcommit_gold.py score-labels --words "$W" --labels "$OUT/labels-$s.jsonl" --gold "$GOLD/gold-$s.jsonl" > "$OUT/gate-$s.json"
  echo "[gate] $s $(cat "$OUT/gate-$s.json")"
done
$PY - "$OUT" $SETS <<'EOF'
import json, sys
out, sets = sys.argv[1], sys.argv[2:]
res = {s: json.load(open(f"{out}/gate-{s}.json")) for s in sets}
json.dump(res, open(f"{out}/gate.json", "w"), indent=1)
keys = ("gold_commit", "cand_recall", "A", "A_precision", "A_recall", "N", "N_precision", "B_commit_share")
print("set        " + "  ".join(f"{k:>14s}" for k in keys))
for s, r in res.items(): print(f"{s:10s} " + "  ".join(f"{(r.get(k) if r.get(k) is not None else float('nan')):>14.3f}" if isinstance(r.get(k), float) else f"{str(r.get(k)):>14s}" for k in keys))
EOF

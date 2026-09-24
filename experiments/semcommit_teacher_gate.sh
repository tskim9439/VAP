#!/usr/bin/env bash
# semcommit 교사 관문 — 후보 교사 구성으로 골드 스트림에 Stage A·B·C 를 돌리고 등급(A/B/N)을 매겨 골드셋과 대조한다(GPU 1 장, 모델을 하나씩 순차로).
# 본 라벨링 전에 교사·레시피를 고르는 싼 점검: 골드 스트림만(gold v1 599 스트림, 14.5 k 단어) 돈다.
# 레시피 v0.3(기본): 후보 = Stage A ∪ 마지막 단어·구간 끝·구두점 문장 끝 → 골드 dev 절반에서 가지(구두점 / 그 외)별 임계값을 맞추고
# (semcommit_gold.py tune, 가지마다 정밀도 ≥ FLOOR, 못 넘는 가지는 끔) → 그 임계값으로 등급. 본 라벨링은 OUT/thresholds.json 을 grade --thresholds 로 쓴다.
#
#   GOLD=<words-<set>.jsonl·gold-<set>.jsonl 가 있는 디렉터리> OUT=<출력 디렉터리> GPU=0 \
#   A=/models/Qwen3-32B:qwen3:qwen3-32b \
#   B="/models/EXAONE-4.0-32B:exaone35:exaone-32b /models/gpt-oss-120b:gptoss:gptoss-120b" \
#   C=/models/Qwen3-32B:qwen3:qwen3-32b \
#   bash experiments/semcommit_teacher_gate.sh
#
# 모델 지정 = 경로:kind[:판정자 이름] (kind ∈ qwen3 | exaone35 | gptoss — vapasr/data/semcommit_llm.KINDS). B 는 공백으로 여러 판정자(등급·tune 의 --judges-*).
# 선택: SETS(기본 "ls-test gs-test ks-eval ks-long"), RECIPE(기본 v0.3 | v0.2), FLOOR(기본 0.90), EXTRA(기본 last,seg_end,punct_final), BS(배치, 기본 16),
#       v0.2 전용 MASK(기본 CONTINUATION,QUALIFICATION)·STRICT(기본 --strict).
# 출력: OUT/{A,B.<이름>,C,labels}-<set>.jsonl + OUT/gate-<set>.json(score-labels) + OUT/gate.json(모음), v0.3 은 OUT/thresholds{,.report}.json.
# 재실행하면 끝난 행은 건너뛴다(teacher 이어하기).
set -uo pipefail
: "${GOLD:?GOLD 디렉터리}"; : "${OUT:?OUT 디렉터리}"; : "${A:?A=경로:kind[:이름]}"; : "${B:?B=\"경로:kind[:이름] ...\"}"; : "${C:?C=경로:kind[:이름]}"
GPU=${GPU:-0}; SETS=${SETS:-"ls-test gs-test ks-eval ks-long"}; RECIPE=${RECIPE:-v0.3}; FLOOR=${FLOOR:-0.90}; EXTRA=${EXTRA:-last,seg_end,punct_final}
MASK=${MASK:-CONTINUATION,QUALIFICATION}; BS=${BS:-16}; STRICT=${STRICT:---strict}
case "$RECIPE" in v0.2) XC=();; v0.3) XC=(--extra-candidates "$EXTRA");; *) echo "!! RECIPE=$RECIPE (v0.2 | v0.3)"; exit 1;; esac
cd "$(dirname "$0")/.."; mkdir -p "$OUT"
PY=${PY:-python}; T="$PY experiments/semcommit_teacher.py"; GT="$PY experiments/semcommit_gold.py"
spec(){ local p k n; IFS=: read -r p k n <<< "$1"; echo "$p" "$k" "${n:-$k}"; }
step(){ echo "[$(date "+%F %T")] >>> $*"; "$@"; rc=$?; echo "[$(date "+%F %T")] <<< rc=$rc"; [ $rc -eq 0 ] || { echo "!! FAILED rc=$rc: $*"; exit $rc; }; }
read -r AP AK AN <<< "$(spec "$A")"; read -r CP CK CN <<< "$(spec "$C")"
JUDGES=(); for b in $B; do read -r _ _ n <<< "$(spec "$b")"; JUDGES+=("$n"); done; JL=$(IFS=,; echo "${JUDGES[*]}")
bfiles(){ local s=$1 b; for b in $B; do read -r _ _ n <<< "$(spec "$b")"; echo "$OUT/B.$n-$s.jsonl"; done; }
for s in $SETS; do
  W=$GOLD/words-$s.jsonl; [ -f "$W" ] || { echo "!! $W 없음"; exit 1; }
  step $T stageA --words "$W" --model "$AP" --kind "$AK" --judge-name "$AN" --out "$OUT/A-$s.jsonl" --gpu "$GPU" --batch-size 8
  for b in $B; do read -r BP BK BN <<< "$(spec "$b")"
    step $T stageB --words "$W" --stageA "$OUT/A-$s.jsonl" --model "$BP" --kind "$BK" --judge-name "$BN" --out "$OUT/B.$BN-$s.jsonl" --gpu "$GPU" --batch-size "$BS" ${XC[@]+"${XC[@]}"}; done
  step $T stageC --words "$W" --stageA "$OUT/A-$s.jsonl" --model "$CP" --kind "$CK" --judge-name "$CN" --out "$OUT/C-$s.jsonl" --gpu "$GPU" --batch-size "$BS" ${XC[@]+"${XC[@]}"}
done
if [ "$RECIPE" = v0.3 ]; then
  TW=(); TG=(); TA=(); TB=(); TC=()
  for s in $SETS; do TW+=("$GOLD/words-$s.jsonl"); TG+=("$GOLD/gold-$s.jsonl"); TA+=("$OUT/A-$s.jsonl"); TC+=("$OUT/C-$s.jsonl"); mapfile -t -O "${#TB[@]}" TB < <(bfiles "$s"); done
  step $GT tune --words "${TW[@]}" --gold "${TG[@]}" --stageA "${TA[@]}" --stageB "${TB[@]}" --stageC "${TC[@]}" --judges-en "$JL" --judges-ko "$JL" \
       --extra-candidates "$EXTRA" --floor "$FLOOR" --out "$OUT/thresholds.json" --report "$OUT/thresholds.report.json"
fi
for s in $SETS; do
  W=$GOLD/words-$s.jsonl; mapfile -t BF < <(bfiles "$s")
  if [ "$RECIPE" = v0.3 ]; then G=(--recipe v0.3 --thresholds "$OUT/thresholds.json" --extra-candidates "$EXTRA"); else G=(--c-mask-types "$MASK" $STRICT); fi
  step $T grade --words "$W" --stageA "$OUT/A-$s.jsonl" --stageB "${BF[@]}" --stageC "$OUT/C-$s.jsonl" --judges-en "$JL" --judges-ko "$JL" "${G[@]}" \
       --out "$OUT/labels-$s.jsonl" --stats "$OUT/labels-$s.stats.json"
  $GT score-labels --words "$W" --labels "$OUT/labels-$s.jsonl" --gold "$GOLD/gold-$s.jsonl" > "$OUT/gate-$s.json"
  echo "[gate] $s $(cat "$OUT/gate-$s.json")"
done
$PY - "$OUT" $SETS <<'EOF'
import json, os, sys
out, sets = sys.argv[1], sys.argv[2:]
res = {s: json.load(open(f"{out}/gate-{s}.json")) for s in sets}
tr = f"{out}/thresholds.report.json"
if os.path.exists(tr): res["_tune"] = json.load(open(tr))
json.dump(res, open(f"{out}/gate.json", "w"), indent=1, ensure_ascii=False)
keys = ("gold_commit", "cand_recall", "A", "A_precision", "A_recall", "N", "N_precision", "B_commit_share")
print("set        " + "  ".join(f"{k:>14s}" for k in keys))
for s in sets:
    r = res[s]; print(f"{s:10s} " + "  ".join(f"{r.get(k):>14.3f}" if isinstance(r.get(k), float) else f"{str(r.get(k)):>14s}" for k in keys))
for lang, r in (res.get("_tune") or {}).get("languages", {}).items():
    print(f"tune {lang}: {json.dumps(r['thresholds'])} | dev P {r['dev']['P']:.3f} R {r['dev']['R']:.3f} | test P {r['test']['P']:.3f} R {r['test']['R']:.3f}")
EOF

# V3 C2 생성(mxc 호스트 CPU): 2 회선 전화 조건 교차 언어 합성 sim_en-de-tel — 학습 셋(1 만 세션) + held-out dev 셋(160 세션, 30–75 s), 이어서 dry-run·수용 점검.
# 입력은 xl_build.sh 의 sim_en-de 와 같고(words·번역 타깃·잡음 뱅크) 다른 점만: 영어 풀 가중(otoSpeech·Switchboard 위주, LibriSpeech 뺌, CV·VoxPopuli 조금),
# 엄격 교대(--p-hold 0.05 · 맞장구 끔 · 교대 간격 0.05–0.8 s · 턴 안 쉼 0.1–0.24 s), 화자별 독립 전화 채널(--phone-per-speaker 1.0, 10 % 같은 채널), 레벨 ±8 dB.
# dev 셋은 xl_eval_build.sh 처럼 held-out 대화만(--heldout, words 없음)·같은 시드(0)·같은 tel 옵션. TAXI 는 쓰지 않는다(평가 전용).
# 산출 VAPKT-data/data/sim-dialogue-xl-tel-v1 · sim-dialogue-xl-tel-eval-v1(각 sim_en-de-tel.* · targets-sim_en-de-tel/ · mono/sim_en-de-tel/), tmp/stage4-dry-xl-tel*.json.
# 산출 경로가 하나라도 있으면 시작하지 않는다(덮어쓰지 않는다). 아무것도 지우지 않는다. 학습 셋 약 15–20 분·약 29 GB(xl_build 와 같은 규모).
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; SNAP=$D/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=$D/semcommit-work/asr-words-v1/lists/asr-words-20260929.list; CV4=$D/cv4-en-v1/lists/cv4-en-words-train.list
S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16; A2=$D/ast-mono-v1/targets-fix-gen-v3-bf16; A2C=$D/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VAPASR_TAR_INDEX_ROOT=$D/speechlm-asr-tar-index-v1
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; TOKZ=/soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000
O=$D/sim-dialogue-xl-tel-v1; OE=$D/sim-dialogue-xl-tel-eval-v1; DRY=$D/../tmp/stage4-dry-xl-tel.json; DRYE=$D/../tmp/stage4-dry-xl-tel-eval.json; K=sim_en-de-tel
for x in $O $OE $DRY $DRYE; do [ -e "$x" ] && { echo "이미 있다 — 덮어쓰지 않고 멈춘다: $x"; exit 1; }; done
TEL="--p-hold 0.05 --bc-scale 0 --fto-clip 0.05,0.8 --intra-pause 0.1,0.24 --jitter-db 8 --phone-per-speaker 1.0 --phone-same-frac 0.1 --name-suffix tel"
DE="--lang2 German --phase2-corpora2 kec,ztc,nsc --pool-weights2 kec:0.35,ztc:0.35,nsc:0.3"
acc() {   # 수용: 겹침 0 · 맞장구 0 · hold ≈ 0.05 · 화자 레벨 차 |Δ| 중앙 ≈ 4–5 dB(±8 dB 독립 두 화자 → 이론 16(1 − 1/√2) ≈ 4.7) · 턴 하나 = episode 하나 · 교대 간격 ≥ 0.05 s
  $PY - "$1/$K.stats.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); t = s["tel"]; c = t["chan"] or {}; md = t["level_diff_db"]["q10_50_90"][1]
ok = dict(failed=s["failed"] == 0, overlap=s["overlap_ratio_mean"] == 0 and s["self_overlap"] == 0, bc=s["bc_per_min"] == 0, hold=abs(t["hold_ratio"] - 0.05) <= 0.02,
          gain=md is not None and 4.0 <= md <= 5.5, turn_episode=t["split_turns"] <= 0, gap=(t["change_gap_s"]["min"] or 0) >= 0.049, chan=c.get("phone") == c.get("tracks"))
print(f"ACCEPT {sys.argv[1]}: 세션 {s['sessions']} 실패 {s['failed']} · {s['hours']} h · 겹침 {s['overlap_ratio_mean']} · 맞장구/분 {s['bc_per_min']} · 교대/분 {s['changes_per_min_mean']} · "
      f"hold {t['hold_ratio']} · |Δ| q10/50/90 {t['level_diff_db']['q10_50_90']} dB · 교대 간격 q10/50/90 {t['change_gap_s']['q10_50_90']} s(최소 {t['change_gap_s']['min']}) · "
      f"갈라진 턴 {t['split_turns']} · 같은 채널 세션 {c.get('same')}/{c.get('sessions')} · 전화 트랙 {c.get('phone')}/{c.get('tracks')} · μ-law {c.get('mulaw')} · "
      + ("OK" if all(ok.values()) else f"CHECK {[k for k, v in ok.items() if not v]}"), flush=True)
EOF
}
echo "[$(date '+%F %T')] $K 학습 셋 시작"
$PY experiments/sim_build_dialogues.py --lang English $DE --out $O --sessions 10000 --workers 24 --phase2 $P2 --split $P2/splits/v2.json --noise-bank $D/noise-bank-v1 \
  --phase2-corpora otoSpeech --words @$SNAP/main-words.list --words @$ASR --words @$CV4 --words-frac 0.5 --pool-weights otoSpeech:0.45,swbd:0.45,cv:0.05,vp:0.05 \
  --trans $S1 --trans $A2/en-de --trans $A2C/en-de $TEL 2>&1 | grep -v "regex pattern" | tail -3
[ -f $O/$K.stats.json ] || { echo "학습 셋 빌드 실패(stats 없음): $O"; exit 1; }
echo "[$(date '+%F %T')] $K held-out dev 셋 시작"
$PY experiments/sim_build_dialogues.py --lang English $DE --out $OE --sessions 160 --dur 30,75 --workers 24 --phase2 $P2 --split $P2/splits/v2.json --noise-bank $D/noise-bank-v1 --heldout \
  --phase2-corpora otoSpeech --pool-weights otoSpeech:1 --trans $S1 $TEL 2>&1 | grep -v "regex pattern" | tail -3
[ -f $OE/$K.stats.json ] || { echo "dev 셋 빌드 실패(stats 없음): $OE"; exit 1; }
echo "[$(date '+%F %T')] dry-run 시작"
$PY experiments/cst_stage4_dryrun.py --data $O --tokenizer $TOKZ --corpora $K --trans $O/targets-$K --max-windows 1500 --audio-sample 6 --procs 1 --out $DRY 2>&1 | grep -E '^\{|all_ok'
$PY experiments/cst_stage4_dryrun.py --data $OE --tokenizer $TOKZ --corpora $K --trans $OE/targets-$K --audio-sample 6 --procs 1 --out $DRYE 2>&1 | grep -E '^\{|all_ok'
acc $O; acc $OE
echo "XL_TEL_DONE $(date '+%F %T')"

# V3 C2b 생성(mxc 호스트 CPU): 레벨 폭을 넓힌 2 회선 전화 교차 언어 합성 sim_en-de-tel — tel-v1(build_tel.sh)과 같은 인자·시드(0), 화자별 레벨만 ±8 → ±12 dB.
#   같은 시드·인자라 대화(발화·시각·채널)는 tel-v1 과 같고 레벨 난수만 넓게 퍼진다 → 레벨 효과만 가르는 대조. ±12 dB 는 대칭 일반값(TAXI 분포에 맞추지 않는다).
#   이론 |Δ| 중앙 2·12·(1 − 1/√2) ≈ 7.0 dB, q90 ≈ 16.4 dB. 학습 셋만(dev 는 v3-dev/xl-tel-dev·-en12·-de12 를 쓴다).
#   산출 VAPKT-data/data/sim-dialogue-xl-tel-v2(sim_en-de-tel.* · targets-sim_en-de-tel/ · mono/), tmp/stage4-dry-xl-tel-v2.json. 있으면 멈춘다. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; SNAP=$D/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637
ASR=$D/semcommit-work/asr-words-v1/lists/asr-words-20260929.list; CV4=$D/cv4-en-v1/lists/cv4-en-words-train.list
S1=$D/stage4-dlg-units-v1/targets-fix-gen-v3-bf16; A2=$D/ast-mono-v1/targets-fix-gen-v3-bf16; A2C=$D/ast-mono-cv4en-v1/targets-fix-gen-v3-bf16
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VAPASR_TAR_INDEX_ROOT=$D/speechlm-asr-tar-index-v1
PY=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; TOKZ=/soundai/Model/VAPASR/stage3-st3-p11-revive/checkpoint-3000
O=$D/sim-dialogue-xl-tel-v2; DRY=$D/../tmp/stage4-dry-xl-tel-v2.json; K=sim_en-de-tel
for x in $O $DRY; do [ -e "$x" ] && { echo "이미 있다 — 덮어쓰지 않고 멈춘다: $x"; exit 1; }; done
TEL="--p-hold 0.05 --bc-scale 0 --fto-clip 0.05,0.8 --intra-pause 0.1,0.24 --jitter-db 12 --phone-per-speaker 1.0 --phone-same-frac 0.1 --name-suffix tel"
DE="--lang2 German --phase2-corpora2 kec,ztc,nsc --pool-weights2 kec:0.35,ztc:0.35,nsc:0.3"
acc() {   # 수용: 겹침 0 · 맞장구 0 · hold ≈ 0.05 · 화자 레벨 차 |Δ| 중앙 ≈ 6–8 dB(±12 dB 독립 두 화자 → 이론 24(1 − 1/√2) ≈ 7.0) · 턴 하나 = episode 하나 · 교대 간격 ≥ 0.05 s
  $PY - "$1/$K.stats.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); t = s["tel"]; c = t["chan"] or {}; md = t["level_diff_db"]["q10_50_90"][1]
ok = dict(failed=s["failed"] == 0, overlap=s["overlap_ratio_mean"] == 0 and s["self_overlap"] == 0, bc=s["bc_per_min"] == 0, hold=abs(t["hold_ratio"] - 0.05) <= 0.02,
          gain=md is not None and 6.0 <= md <= 8.0, turn_episode=t["split_turns"] <= 0, gap=(t["change_gap_s"]["min"] or 0) >= 0.049, chan=c.get("phone") == c.get("tracks"))
print(f"ACCEPT {sys.argv[1]}: 세션 {s['sessions']} 실패 {s['failed']} · {s['hours']} h · 겹침 {s['overlap_ratio_mean']} · 맞장구/분 {s['bc_per_min']} · 교대/분 {s['changes_per_min_mean']} · "
      f"hold {t['hold_ratio']} · |Δ| q10/50/90 {t['level_diff_db']['q10_50_90']} dB · 교대 간격 q10/50/90 {t['change_gap_s']['q10_50_90']} s(최소 {t['change_gap_s']['min']}) · "
      f"갈라진 턴 {t['split_turns']} · 같은 채널 세션 {c.get('same')}/{c.get('sessions')} · 전화 트랙 {c.get('phone')}/{c.get('tracks')} · μ-law {c.get('mulaw')} · "
      + ("OK" if all(ok.values()) else f"CHECK {[k for k, v in ok.items() if not v]}"), flush=True)
EOF
}
echo "[$(date '+%F %T')] $K 학습 셋 시작"
$PY experiments/sim_build_dialogues.py --lang English $DE --out $O --sessions 10000 --workers 24 --phase2 $P2 --split $P2/splits/v2.json --noise-bank $D/noise-bank-v1 --noise-p 0.3 \
  --phase2-corpora otoSpeech --words @$SNAP/main-words.list --words @$ASR --words @$CV4 --words-frac 0.5 --pool-weights otoSpeech:0.45,swbd:0.45,cv:0.05,vp:0.05 \
  --trans $S1 --trans $A2/en-de --trans $A2C/en-de $TEL 2>&1 | grep -v "regex pattern" | tail -3
[ -f $O/$K.stats.json ] || { echo "학습 셋 빌드 실패(stats 없음): $O"; exit 1; }
$PY - $D/sim-dialogue-xl-tel-v1/$K.refined.dialogues.jsonl $O/$K.refined.dialogues.jsonl <<'EOF2'
import json, sys
strip = lambda x: {k: v for k, v in x.items() if k != "meta"}
A = [strip(json.loads(l)) for l in open(sys.argv[1], encoding="utf-8")]; B = [strip(json.loads(l)) for l in open(sys.argv[2], encoding="utf-8")]
same = len(A) == len(B) and all(a == b for a, b in zip(A, B)); print(f"  tel-v1 과 같은 대화(meta 밖 전부): {same} · 세션 {len(A)}/{len(B)}")
if not same: sys.exit(1)
EOF2
[ $? = 0 ] || { echo "tel-v1 과 대화가 다르다 — 멈춘다"; exit 1; }
echo "[$(date '+%F %T')] dry-run 시작"
$PY experiments/cst_stage4_dryrun.py --data $O --tokenizer $TOKZ --corpora $K --trans $O/targets-$K --max-windows 1500 --audio-sample 6 --procs 1 --out $DRY 2>&1 | grep -E '^\{|all_ok'
acc $O
echo "XL_TEL_V2_DONE $(date '+%F %T')"

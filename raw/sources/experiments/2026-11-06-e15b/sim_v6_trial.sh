#!/usr/bin/env bash
# E15b 겹침 많은 합성 영어 회의(v6-ovl) 설정 고르기(2026-11-06, 사용자 "2번부터 진행해." — mxc 호스트 CPU, 제출 없음):
#   설정 A·B·C 를 300 세션씩 VAPKT-data/tmp/sim-v6-trial/<판>/ 에 만들고(새 폴더), AMI(ami-bench-v1 공식 dev·eval, phase2 ami 학습 분할)와 같은 지표로 견준다
#   — 겹침 몫 = 말소리 프레임(80 ms) 중 둘 이상이 말하는 몫, 둘 · 셋+ 몫, 시간 대비 겹침(overlap_stats). v4-meet(지금 sim_en-meet)도 같이 잰다.
#   재료는 v4 와 같은 split v4 학습 분할 + 영어 words, 풀 가중은 큰 단일 화자 셋(swbd·cv·vp·ls) 쪽으로 옮긴다(AMI·ICSI 발화 재사용을 줄인다).
#   bash sim_v6_trial.sh   → tmp/sim-v6-trial/report.txt. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; T=/soundai/users/tskim/VAPKT-data/tmp/sim-v6-trial
W=/soundai/users/tskim/VAPKT-data/data/semcommit-work; mkdir -p $T
POOL=otoSpeech:0.15,ami:0.08,icsi:0.08,notsofar:0.04,swbd:0.2,cv:0.15,vp:0.15,ls:0.15
BASE="--lang English --sessions 300 --dur 60,120 --phase2 $P2 --split $P2/splits/v4.json --phase2-corpora otoSpeech,ami,icsi,notsofar --stats-corpora ami,icsi,notsofar
  --words @$W/labels/speechlm-all19-v035/snapshots/20260929-1637/main-words.list --words @$W/asr-words-v1/lists/asr-words-20260929.list
  --words @$D/cv4-en-v1/lists/cv4-en-words-train.list --words-frac 0.5 --pool-weights $POOL --p-back 0.2
  --noise-bank $D/noise-bank-v1 --noise-p 0.7 --noise-snr 5,25 --rir-p 0.3 --far-per-speaker 0.7 --name-suffix ovl --workers 24 --seed 0"
declare -A V=(
  [A]="--speakers 3:0.4,4:0.6 --p-interrupt 0.35 --interrupt-fto=-2.5,-0.3 --bc-scale 3.0"
  [B]="--speakers 3:0.3,4:0.7 --p-interrupt 0.5 --interrupt-fto=-3.0,-0.3 --bc-scale 3.0"
  [C]="--speakers 3:0.3,4:0.7 --p-interrupt 0.6 --interrupt-fto=-3.5,-0.5 --bc-scale 3.5 --p-hold 0.2"
)
for k in A B C; do
  [ -e $T/$k ] && { echo "[$(date "+%F %T")] $k 이미 있다 — 건너뜀"; continue; }
  echo "[$(date "+%F %T")] 판 $k: ${V[$k]}"
  $P experiments/sim_build_dialogues.py $BASE --out $T/$k ${V[$k]} > $T/$k.log 2>&1; echo "[$(date "+%F %T")] SIM_EXIT $k $?"
done
$P - > $T/report.txt 2>&1 <<'EOF'
import json, glob, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.dialogue import Dialogue, build_episodes
from vapasr.data.sim_dialogue import overlap_stats
import numpy as np
D = "/soundai/users/tskim/VAPKT-data/data"; T = "/soundai/users/tskim/VAPKT-data/tmp/sim-v6-trial"
man = json.load(open(f"{D}/ami-bench-v1/manifest.json")); keep = set(man["dev"]) | set(man["eval"])
def frames(d):
    eps = build_episodes(d); K = int(np.ceil(d.duration_s / 0.08)) + 1; spk = {s: i for i, s in enumerate(sorted({e.speaker for e in eps}))}
    A = np.zeros((K, max(1, len(spk))), bool)
    for e in eps: A[int(e.start / 0.08): int(np.ceil(e.end / 0.08)), spk[e.speaker]] = True
    n = A.sum(1); return int((n >= 1).sum()), int((n == 2).sum()), int((n >= 3).sum())
def report(name, path, filt=None, limit=None):
    sp = o2 = o3 = 0; ovr = []; dur = 0.0; m = 0
    for l in open(path, encoding="utf-8"):
        d = Dialogue.from_json(l)
        if filt and not filt(d): continue
        a, b, c = frames(d); sp += a; o2 += b; o3 += c; st = overlap_stats(d); ovr.append(st["overlap_ratio"]); dur += d.duration_s; m += 1
        if limit and m >= limit: break
    print(f"{name:34s} 대화 {m:5d} · {dur / 3600:6.1f} h · 말소리 프레임 중 겹침 {100 * (o2 + o3) / max(1, sp):5.1f} %(둘 {100 * o2 / max(1, sp):4.1f} · 셋+ {100 * o3 / max(1, sp):4.1f}) · 시간 대비 겹침 평균 {100 * np.mean(ovr):5.1f} %", flush=True)
report("AMI 공식 dev·eval(ami-bench-v1 sdm)", f"{D}/ami-bench-v1/ami-sdm.refined.dialogues.jsonl", lambda d: d.conv_id.split(":")[-1] in keep)
report("phase2 ami(전체)", f"{D}/phase2/ami.refined.dialogues.jsonl")
report("v4-meet sim_en-meet(지금, 앞 300)", f"{D}/sim-dialogue-v4-meet/sim_en-meet.refined.dialogues.jsonl", limit=300)
for k in "ABC":
    for p in glob.glob(f"{T}/{k}/sim_en-ovl.refined.dialogues.jsonl"): report(f"v6 판 {k}", p)
EOF
echo "[$(date "+%F %T")] TRIAL_DONE"

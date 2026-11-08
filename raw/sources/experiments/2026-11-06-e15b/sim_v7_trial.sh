#!/usr/bin/env bash
# 겹침 데이터 v7 설정 고르기(2026-11-08, 사용자 "화자 겹침 데이터를 더 만들어서 학습되도록 하는 게 낫지 않을까?" → "겹침 길이를 더 늘려도 될거 같아. 완전 겹치는 케이스도 추가하고"):
#   AMI 에서 놓친 단어의 절반이 보통 발화끼리 겹친 구간이라, v6(sim_en-ovl, 말소리 중 겹침 22.8 %)보다 내용 있는 말끼리 길게·완전히 겹치는 쪽으로 판 A·B·C 를 300 세션씩 만든다.
#   v6 와 같은 재료(split v4 학습 분할 + 영어 words, 큰 단일 화자 셋 쪽 풀)·원거리·잡음. 맞장구는 줄이고(bc 1.0) 화자 수는 2 인 쪽으로 옮긴다(AMI 오류 대부분이 둘 겹침).
#   지표: 말소리 프레임(80 ms) 중 둘·셋+ 겹침 몫, 겹침 구간 길이(이어진 겹침 프레임, 중앙값·90 %), 90 % 이상 겹친 발화 몫 — AMI 공식 dev·eval, v6 판 C 와 함께.
#   bash sim_v7_trial.sh   → VAPKT-data/tmp/sim-v7-trial/report.txt(새 폴더). mxc 호스트 CPU, 제출 없음. 아무것도 지우지 않는다.
set -u
cd /soundai/users/tskim/VAPKT; set -a; source .env; set +a
export PYTHONPATH=/soundai/users/tskim/VAPKT PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1
P=/soundai/users/tskim/VAPKT-data/conda/envs/vapasr/bin/python; D=/soundai/users/tskim/VAPKT-data/data; P2=$D/phase2; T=/soundai/users/tskim/VAPKT-data/tmp/sim-v7-trial
W=/soundai/users/tskim/VAPKT-data/data/semcommit-work; mkdir -p $T
POOL=otoSpeech:0.15,ami:0.08,icsi:0.08,notsofar:0.04,swbd:0.2,cv:0.15,vp:0.15,ls:0.15
BASE="--lang English --sessions 300 --dur 60,120 --phase2 $P2 --split $P2/splits/v4.json --phase2-corpora otoSpeech,ami,icsi,notsofar --stats-corpora ami,icsi,notsofar
  --words @$W/labels/speechlm-all19-v035/snapshots/20260929-1637/main-words.list --words @$W/asr-words-v1/lists/asr-words-20260929.list
  --words @$D/cv4-en-v1/lists/cv4-en-words-train.list --words-frac 0.5 --pool-weights $POOL --p-back 0.2
  --noise-bank $D/noise-bank-v1 --noise-p 0.7 --noise-snr 5,25 --rir-p 0.3 --far-per-speaker 0.7 --name-suffix ovl2 --workers 24 --seed 0"
declare -A V=(
  [A]="--speakers 2:0.4,3:0.4,4:0.2 --p-interrupt 0.8 --interrupt-fto=-8,-1 --bc-scale 1.0 --p-hold 0.2 --p-full-overlap 0.15"
  [B]="--speakers 2:0.4,3:0.4,4:0.2 --p-interrupt 0.8 --interrupt-fto=-8,-1 --bc-scale 1.0 --p-hold 0.2 --p-full-overlap 0.3"
  [C]="--speakers 2:0.4,3:0.4,4:0.2 --p-interrupt 0.9 --interrupt-fto=-10,-2 --bc-scale 1.0 --p-hold 0.2 --p-full-overlap 0.3"
  [D]="--speakers 2:0.4,3:0.4,4:0.2 --p-interrupt 0.8 --interrupt-fto=-8,-1 --bc-scale 1.0 --p-hold 0.2 --p-full-overlap 0.3 --split-s 8"   # 2026-11-08 사용자 "판 B 로 split-s 8 도 같이 시험해줘" — 긴 발화를 덜 쪼개 겹침 구간을 길게
)
for k in A B C D; do
  [ -e $T/$k ] && { echo "[$(date "+%F %T")] $k 이미 있다 — 건너뜀"; continue; }
  echo "[$(date "+%F %T")] 판 $k: ${V[$k]}"
  $P experiments/sim_build_dialogues.py $BASE --out $T/$k ${V[$k]} > $T/$k.log 2>&1; echo "[$(date "+%F %T")] SIM_EXIT $k $?"
done
$P - > $T/report.txt 2>&1 <<'EOF'
import json, glob, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.dialogue import Dialogue, build_episodes
import numpy as np
D = "/soundai/users/tskim/VAPKT-data/data"; T = "/soundai/users/tskim/VAPKT-data/tmp/sim-v7-trial"
man = json.load(open(f"{D}/ami-bench-v1/manifest.json")); keep = set(man["dev"]) | set(man["eval"])
def frames(d):
    eps = build_episodes(d); K = int(np.ceil(d.duration_s / 0.08)) + 1; spk = {s: i for i, s in enumerate(sorted({e.speaker for e in eps}))}
    A = np.zeros((K, max(1, len(spk))), bool)
    for e in eps: A[int(e.start / 0.08): int(np.ceil(e.end / 0.08)), spk[e.speaker]] = True
    n = A.sum(1); ov = n >= 2
    segs, run = [], 0                                                                      # 이어진 겹침 구간 길이(s)
    for x in ov:
        if x: run += 1
        elif run: segs.append(run * 0.08); run = 0
    if run: segs.append(run * 0.08)
    full = sum(1 for e in eps if e.end - e.start >= 1.0 and ov[int(e.start / 0.08): int(np.ceil(e.end / 0.08))].mean() >= 0.9)   # 1 s 이상 발화 중 90 % 이상 겹친 것
    return int((n >= 1).sum()), int((n == 2).sum()), int((n >= 3).sum()), segs, full, sum(1 for e in eps if e.end - e.start >= 1.0)
def report(name, path, filt=None, limit=None):
    sp = o2 = o3 = 0; segs = []; full = n1 = 0; dur = 0.0; m = 0
    for l in open(path, encoding="utf-8"):
        d = Dialogue.from_json(l)
        if filt and not filt(d): continue
        a, b, c, s, f, k = frames(d); sp += a; o2 += b; o3 += c; segs += s; full += f; n1 += k; dur += d.duration_s; m += 1
        if limit and m >= limit: break
    q = np.quantile(segs, [0.5, 0.9]) if segs else [0, 0]
    print(f"{name:30s} 대화 {m:5d} · {dur / 3600:5.1f} h · 말소리 중 겹침 {100 * (o2 + o3) / max(1, sp):5.1f} %(둘 {100 * o2 / max(1, sp):4.1f} · 셋+ {100 * o3 / max(1, sp):4.1f})"
          f" · 겹침 구간 중앙값 {q[0]:4.2f} s · 90 % {q[1]:4.2f} s · 90 % 이상 겹친 발화(1 s+) {100 * full / max(1, n1):4.1f} %", flush=True)
report("AMI 공식 dev·eval(sdm)", f"{D}/ami-bench-v1/ami-sdm.refined.dialogues.jsonl", lambda d: d.conv_id.split(":")[-1] in keep)
for p in glob.glob("/soundai/users/tskim/VAPKT-data/tmp/sim-v6-trial/C/sim_en-ovl.refined.dialogues.jsonl"): report("v6 판 C(sim_en-ovl 설정)", p)
for k in "ABCD":
    for p in glob.glob(f"{T}/{k}/sim_en-ovl2.refined.dialogues.jsonl"): report(f"v7 판 {k}", p)
EOF
echo "[$(date "+%F %T")] TRIAL_DONE"

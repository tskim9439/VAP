# 평가 창(δ10, 한국어)마다 정답(전사 감독 발화)에서 라벨↔0.6B 불일치 발화가 차지하는 시간 몫으로 묶어 agn·cp·귀속 몫을 본다. 읽기만 한다.
import json, glob, collections
D = "/soundai/users/tskim/VAPKT-data/data/phase2"; R = "/soundai/users/tskim/VAPKT-data/results"
KO = ("aihub71631", "aihub134-1", "aihub134-2")
err = {}
for c in KO:
    for l in open(f"{D}/asrcheck/{c}.utts.jsonl", encoding="utf-8"): r = json.loads(l); err[(r["conv"], r["utt"])] = r["err"]
win = {}
for f in glob.glob(f"{R}/stage3-eval-st3-p9-spkhead-ck3000/predictions-rank*.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["delta"] == 10 and r["corpus"] in KO: win[r["id"]] = r["L"]
need = {i.rsplit("@", 1)[0] for i in win}; spans = collections.defaultdict(list)
for c in KO:
    for l in open(f"{D}/{c}.refined.dialogues.jsonl", encoding="utf-8"):
        cid = l[13:400].split('"', 1)[0]
        if cid not in need: continue
        d = json.loads(l)
        for u in d["utterances"]:
            if u.get("tokens"): spans[cid].append((u["start"], u["end"], err.get((cid, u["utt_id"].split("#")[0]))))
def share(i):
    cid, t0 = i.rsplit("@", 1); t0 = float(t0); t1 = t0 + win[i]; tot = mid = hi_ = 0.0
    for s, e, x in spans[cid]:
        o = max(0.0, min(e, t1) - max(s, t0))
        if o <= 0: continue
        tot += o
        if x is None: continue
        if x >= 0.1: hi_ += o
        elif x >= 0.05: mid += o
    return (hi_ / tot if tot else 0.0), ((hi_ + mid) / tot if tot else 0.0)
for run in ("q06-st3-r1-ck2000-v3", "st3-p9-spkhead-ck3000"):
    W = {w["id"]: w for w in map(json.loads, open(f"{R}/stage3-attr-errors-{run}/windows.jsonl")) if w["delta"] == 10 and w["id"] in win}
    for name, k in (("라벨↔0.6B ≥ 0.1 발화 시간 몫", 0), ("라벨↔0.6B ≥ 0.05 발화 시간 몫", 1)):
        B = collections.defaultdict(lambda: [0, 0, 0, 0])
        for i, w in W.items():
            x = share(i)[k]; b = "0 %" if x == 0 else "0–20 %" if x < .2 else "20–40 %" if x < .4 else "≥ 40 %"
            v = B[b]; v[0] += w["cp"]; v[1] += w["agn"]; v[2] += w["n_ref"]; v[3] += 1
        print(f"[{run} · {name}]")
        for b in ("0 %", "0–20 %", "20–40 %", "≥ 40 %"):
            if b in B: cp, agn, n, m = B[b]; print(f"   {b:8s} 창 {m:3d} · 참조 {n:6d} 글자 · cp {100*cp/n:5.1f} · agn {100*agn/n:5.1f} · 귀속 몫 {100*(cp-agn)/n:5.1f}")
    tot = [sum(w[k] for w in W.values()) for k in ("cp", "agn", "n_ref")]; print(f"   전체 cp {100*tot[0]/tot[2]:.1f} · agn {100*tot[1]/tot[2]:.1f}")

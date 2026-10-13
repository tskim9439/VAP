import json, sys, random, numpy as np, soundfile as sf, collections
sys.path.insert(0, "/soundai/users/tskim/VAPKT"); sys.path.insert(0, "/soundai/users/tskim/VAPKT/experiments")
from vapasr.data.dialogue import Dialogue
from vapasr.data.dialogue_refine import channel_vad
import nsc_pseudo_label as N
D = "/soundai/users/tskim/VAPKT-data/data/phase2"
lines = open(f"{D}/nsc.dialogues.jsonl", encoding="utf-8").readlines(); random.Random(0).shuffle(lines)
agg = collections.Counter(); rows = []
for l in lines[:30]:
    d = Dialogue.from_json(l); vad = channel_vad(d)
    e = {}
    for r in ("speaker", "interlocutor"):
        x, sr = sf.read(d.channels[r].path, dtype="float32"); e[r] = N.frame_db(x, sr)
    m = min(len(v) for v in e.values())
    for u in d.utterances:
        o = "interlocutor" if u.speaker == "speaker" else "speaker"; a, b = int(u.start * 50), min(m, int(u.end * 50))
        if b <= a: continue
        es, eo = e[u.speaker][a:b], e[o][a:b]
        valid = e[u.speaker][:m] > -95; floor = np.percentile(e[u.speaker][:m][valid], 10)
        nov = not any(s < u.end and t > u.start for s, t in vad.get(u.speaker, []))
        ozero = float((eo < -95).mean()); lvl = float(np.median(es) - floor); sup = bool(u.text)
        k = ("novad" if nov else "vad", "ozero>0.5" if ozero > 0.5 else "ozero<=0.5", "sup" if sup else "masked"); agg[k] += 1
        rows.append((nov, ozero, lvl, sup, u.raw[:60], d.conv_id, u.speaker, u.start, u.end))
for k in sorted(agg): print(k, agg[k])
nv = [r for r in rows if r[0]]; v = [r for r in rows if not r[0]]
print("level above floor (dB) median: novad", np.median([r[2] for r in nv]).round(1), "vad", np.median([r[2] for r in v]).round(1))
print("examples novad & other-channel zeroed:")
for r in [r for r in nv if r[1] > 0.5][:8]: print("  ", r[5], r[6], f"{r[7]:.1f}-{r[8]:.1f}", f"ozero {r[1]:.2f} lvl {r[2]:.1f} sup {r[3]}", r[4])
print("examples novad & other present:")
for r in [r for r in nv if r[1] <= 0.5][:8]: print("  ", r[5], r[6], f"{r[7]:.1f}-{r[8]:.1f}", f"ozero {r[1]:.2f} lvl {r[2]:.1f} sup {r[3]}", r[4])

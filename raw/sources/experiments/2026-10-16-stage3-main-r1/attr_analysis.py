# checkpoint-2000 δ10 창별: 화자 귀속 몫 = (cp − agn)/n_ref 를 겹침 비율·화자 교대 수·한 화자 몫·창 길이로 나눠 본다. 읽기만 한다.
import json, glob, collections, sys
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from vapasr.data.dialogue import Dialogue, build_episodes
R = "/soundai/users/tskim/VAPKT-data/results/stage3-eval-q06-st3-r1-ck2000"; D = "/soundai/users/tskim/VAPKT-data/data/phase2"
rows = [json.loads(l) for f in glob.glob(f"{R}/predictions-rank*.jsonl") for l in open(f)]; rows = [r for r in rows if r["delta"] == 10]
need = {r["id"].rsplit("@", 1)[0] for r in rows}; eps = {}
for c in ("aihub71631", "aihub134-1", "aihub134-2", "otoSpeech", "kec", "ztc", "nsc"):
    for l in open(f"{D}/{c}.refined.dialogues.jsonl", encoding="utf-8"):
        cid = l[13:300].split('"', 1)[0] if l.startswith('{"conv_id": "') else json.loads(l)["conv_id"]
        if cid in need: d = Dialogue.from_json(l); eps[d.conv_id] = build_episodes(d)
def feats(cid, t0, L):
    t1 = t0 + L; E = [e for e in eps[cid] if e.end > t0 and e.start < t1]
    act = collections.defaultdict(float); ov = 0.0; step = 0.05; t = t0
    while t < t1:
        on = {e.speaker for e in E if e.start <= t < e.end}
        if len(on) >= 2: ov += step
        for s in on: act[s] += step
        t += step
    E.sort(key=lambda e: e.start); changes = sum(1 for a, b in zip(E, E[1:]) if a.speaker != b.speaker)
    tot = sum(act.values()) or 1e-9; dom = max(act.values()) / tot if act else 1.0
    return ov / L, changes / (L / 60), dom
B = collections.defaultdict(lambda: [0, 0, 0])            # bucket → [attr err, agn err, n_ref]
for r in rows:
    cid, t0 = r["id"].rsplit("@", 1); ov, chg, dom = feats(cid, float(t0), r["L"]); a = r["cp"] - r["agn"]
    for k, v in (("겹침", "0 %" if ov == 0 else "<5 %" if ov < .05 else "5–10 %" if ov < .10 else "≥10 %"),
                 ("교대/분", "<5" if chg < 5 else "5–10" if chg < 10 else "10–20" if chg < 20 else "≥20"),
                 ("한 화자 몫", "≥90 %" if dom >= .9 else "70–90 %" if dom >= .7 else "50–70 %"),
                 ("화자 수", str(r["n_spk"]))):
        b = B[(k, v)]; b[0] += max(0, a); b[1] += r["agn"]; b[2] += r["n_ref"]
tot = [sum(r[x] for r in rows) for x in ("cp", "agn", "n_ref")]
print(f"전체 δ10: cp {100 * tot[0] / tot[2]:.1f} · agn {100 * tot[1] / tot[2]:.1f} · 화자 귀속 몫(cp−agn) {100 * (tot[0] - tot[1]) / tot[2]:.1f} pt · 창 {len(rows)}")
for k in ("겹침", "교대/분", "한 화자 몫", "화자 수"):
    for (kk, v), (a, g, n) in sorted(B.items()):
        if kk == k: print(f"  {k:6s} {v:8s} 화자 귀속 몫 {100 * a / n:5.1f} pt · agn {100 * g / n:5.1f} · 참조 단위 {n}")

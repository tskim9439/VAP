# TAXI speaker-attribution dissection (read-only, aggregate output only; no transcript text or session ids are printed).
import collections, itertools, json, re, statistics as st, sys, zlib
from cstbench.evaluate import load_jsonl
from vapasr.data.cst_eval import align, ed
from baseline_vibevoice import hyp_words

R = "/soundai/users/tskim/VAPKT-data/results"
S = load_jsonl("/soundai/users/tskim/VAPKT-data/data/cst/taxi/cstbench-v0.1/taxi-natural/sessions.jsonl")
simple = lambda s: re.sub(r"[^\w' ]+", " ", s.lower()).split()
ONLY = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] else None
W = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0


def ours(f):
    H = collections.defaultdict(list)
    for x in load_jsonl(f): H[x["session"]].append((x["lang"], x["text"], x["t"]))
    return H


def vv(d):
    out = {}
    for l in open(f"{d}/taxi-pred.jsonl", encoding="utf-8"):
        r = json.loads(l); out[r["session"]] = hyp_words(dict(r, lang=None))
    return out


SYS = [("V1d10", lambda: ours(f"{R}/stage4-eval-st4-m2a-ck2000/taxi90/asr-d10.jsonl"), "lang"),
       ("V1d4", lambda: ours(f"{R}/stage4-eval-st4-m2a-ck2000/taxi90/asr-d4.jsonl"), "lang"),
       ("V1d10w40", lambda: ours(f"{R}/stage4-eval-st4-m2a-ck2000/taxi/asr-d10.jsonl"), "lang"),
       ("V2d10", lambda: ours(f"{R}/stage4-eval-v2-p1-ck2000/taxi90/asr-d10.jsonl"), "lang"),
       ("V2notap", lambda: ours(f"{R}/taxi-probe-v2-p1-ck2000-notap/asr-d10.jsonl"), "lang"),
       ("VVstream", lambda: vv(f"{R}/baseline-vibevoice-stream-taxi"), "spk"),
       ("VVoffline", lambda: vv(f"{R}/baseline-vibevoice-offline-taxi"), "spk")]


def prep(s):
    turns = sorted(s["turns"], key=lambda t: (t["start_s"], t["end_s"]))
    T, words = [], []
    for ti, t in enumerate(turns):
        ws = simple(t["transcript"]); n = len(ws); a, b = t["start_s"], t["end_s"]
        prev = turns[ti - 1] if ti else None
        maxend = max((u["end_s"] for u in turns[:ti]), default=None)
        T.append(dict(spk=t["speaker"], lang=t["lang"], a=a, b=b, n=n, gap_given=t.get("gap_before_s"),
                      gap=(a - prev["end_s"]) if prev else None, gapmax=(a - maxend) if maxend is not None else None,
                      change=None if prev is None else prev["speaker"] != t["speaker"]))
        for k, w in enumerate(ws):
            words.append(dict(t=a + (b - a) * (k + 1) / n, tm=a + (b - a) * (k + 0.5) / n, ti=ti, pos=k, w=w, spk=t["speaker"]))
    words.sort(key=lambda x: (x["tm"], x["ti"], x["pos"]))
    busy = collections.defaultdict(list)
    for t in T: busy[t["spk"]].append((t["a"], t["b"]))
    for x in words: x["ov"] = any(a <= x["tm"] <= b for sp, iv in busy.items() if sp != x["spk"] for a, b in iv)
    return T, words


def walign(Rw, H, off, Wd):
    n, m = len(Rw), len(H); INF = 10 ** 9
    D = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        ri = Rw[i - 1]; rt, rw = ri["t"] + off, ri["w"]; Di, Dp = D[i], D[i - 1]
        for j in range(1, m + 1):
            h = H[j - 1]; dt = h[0] - rt
            c = Dp[j - 1] + ((0 if h[2] == rw else 1) if -Wd <= dt <= Wd else INF)
            a = Dp[j] + 1; b = Di[j - 1] + 1
            Di[j] = c if c <= a and c <= b else (a if a <= b else b)
    out = []; i, j = n, m
    while i or j:
        if i and j:
            h = H[j - 1]; ri = Rw[i - 1]; dt = h[0] - ri["t"] - off
            c = (0 if h[2] == ri["w"] else 1) if -Wd <= dt <= Wd else INF
            if D[i][j] == D[i - 1][j - 1] + c: out.append((i - 1, j - 1)); i -= 1; j -= 1; continue
        if i and D[i][j] == D[i - 1][j] + 1: out.append((i - 1, None)); i -= 1
        else: out.append((None, j - 1)); j -= 1
    return out[::-1], D[n][m]


def cp_agn(s, hyp):
    turns = sorted(s["turns"], key=lambda t: t["start_s"]); langs = sorted({t["lang"] for t in turns})
    spk_of_lang = {s["speakers"][k]["lang"]: k for k in s["speakers"]}
    ref_all = [w for t in turns for w in simple(t["transcript"])]
    refs = [[w for t in turns if t["speaker"] == sp for w in simple(t["transcript"])] for sp in sorted(s["speakers"])]
    hw = sorted(hyp, key=lambda x: x[2]); hyp_all = [w for _, wd, _ in hw for w in simple(wd)]
    by = collections.defaultdict(list)
    for sp, wd, _ in hw: by[sp] += simple(wd)
    hs = sorted(by.values(), key=len, reverse=True)
    m = max(len(refs), len(hs)); rp = refs + [[]] * (m - len(refs)); hp = hs + [[]] * (m - len(hs))
    e_cp = min(sum(ed(rp[k], hp[p[k]]) for k in range(m)) for p in itertools.permutations(range(m))) if m <= 6 else 10 ** 9
    return e_cp, ed(ref_all, hyp_all), len(ref_all)


def pct(a, b): return f"{100 * a / b:5.1f}" if b else "  n/a"


def binof(x, edges, names):
    for e, nm in zip(edges, names):
        if x <= e: return nm
    return names[-1]


PREP = {s["session_id"]: prep(s) for s in S}
VOC = {"English": set(), "German": set()}
for _s in S:
    for _t in _s["turns"]: VOC[_t["lang"]].update(simple(_t["transcript"]))
def wlang(w):
    e = w in VOC["English"]; d = w in VOC["German"]
    return "English" if e and not d else "German" if d and not e else ("both" if e and d else "oov")


import os
def load_eot(f):
    E = collections.defaultdict(list)
    if os.path.exists(f):
        for x in load_jsonl(f): E[x["session"]].append((x["t"], x["lang"]))
    return E

def eot_study(name, asr, eotf):
    H = ours(asr); EO = load_eot(eotf)
    dts = []
    for s in S:
        T, Rw = PREP[s["session_id"]]
        hw = sorted([(t, lab, u) for lab, w, t in H.get(s["session_id"], []) for u in simple(w)], key=lambda x: x[0])
        for i, j in align([x["w"] for x in Rw], [x[2] for x in hw]):
            if i is not None and j is not None and Rw[i]["w"] == hw[j][2]: dts.append(hw[j][0] - Rw[i]["t"])
    off = st.median(dts)
    cells = collections.Counter(); neot = sum(len(v) for v in EO.values()); ntot = 0
    for s in S:
        sid = s["session_id"]; T, Rw = PREP[sid]
        hw = sorted([(t, lab, u) for lab, w, t in H.get(sid, []) for u in simple(w)], key=lambda x: x[0])
        pairs, _ = walign(Rw, hw, off, W)
        lab = [None] * len(Rw)
        for i, j in pairs:
            if i is not None and j is not None: lab[i] = hw[j][1]
        tl = collections.defaultdict(collections.Counter)
        for i, x in enumerate(Rw):
            if lab[i] is not None: tl[x["ti"]][lab[i]] += 1
        maj = {ti: c.most_common(1)[0][0] for ti, c in tl.items()}
        for ti in range(1, len(T)):
            t, p = T[ti], T[ti - 1]
            if not t["change"] or ti not in maj: continue
            ntot += 1
            wrong = maj[ti] != t["lang"]
            lo, hi = p["b"] + off - 0.4, t["a"] + off + 0.6
            ev = [(tt, lg) for tt, lg in EO.get(sid, []) if lo <= tt <= hi]
            prev_slot = maj.get(ti - 1)
            k_any = "EOT있음" if ev else "EOT없음"
            k_prev = "앞슬롯EOT" if any(lg == prev_slot for _, lg in ev) else ("다른슬롯EOT" if ev else "없음")
            cells[("wrong" if wrong else "ok", k_any)] += 1; cells[("wrong" if wrong else "ok", k_prev)] += 1
    print(f"== {name}: 오프셋 {off:.2f} s · <EOT> {neot} 개 · 교대 턴(정렬됨) {ntot}")
    for w in ("ok", "wrong"):
        a, b = cells[(w, "EOT있음")], cells[(w, "EOT없음")]
        print(f"  {w:5s} 턴 {a + b:3d}: 경계에 EOT 있음 {a} ({100 * a / max(1, a + b):.1f} %) · 앞 턴 슬롯의 EOT {cells[(w, '앞슬롯EOT')]} · 다른 슬롯 EOT {cells[(w, '다른슬롯EOT')]}")

R3 = "/soundai/users/tskim/VAPKT-data/results"
for b in sys.argv[3].split(","):
    d = f"{R3}/taxi-v3probe-st4-m2a-ck2000-b{b}"
    if os.path.exists(f"{d}/asr-d10.jsonl"): eot_study(f"V1 m2a ck2000 tag_bias {b}", f"{d}/asr-d10.jsonl", f"{d}/eot-d10.jsonl")

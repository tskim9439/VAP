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

# reference-only facts (printed once)
if not ONLY or "REF" in ONLY:
    nT = sum(len(T) for T, _ in PREP.values()); nW = sum(len(w) for _, w in PREP.values())
    ch = sum(1 for T, _ in PREP.values() for t in T if t["change"]); cont = sum(1 for T, _ in PREP.values() for t in T if t["change"] is False)
    print(f"REF sessions {len(S)} turns {nT} words {nW} change-turns {ch} continuation-turns {cont} first-turns {len(S)}")
    L = [t["n"] for T, _ in PREP.values() for t in T]
    print(f"REF turn length words: median {st.median(L)} mean {st.mean(L):.1f} p10 {sorted(L)[len(L)//10]} p90 {sorted(L)[9*len(L)//10]} max {max(L)} | <=3w {sum(x<=3 for x in L)} 4-10w {sum(4<=x<=10 for x in L)} 11-20w {sum(11<=x<=20 for x in L)} >20w {sum(x>20 for x in L)}")
    G = [t["gap"] for T, _ in PREP.values() for t in T if t["gap"] is not None]
    Gg = [(t["gap"], t["gap_given"]) for T, _ in PREP.values() for t in T if t["gap"] is not None and t["gap_given"] is not None]
    print(f"REF gap(prev turn) : overlap(<0) {sum(g<0 for g in G)} 0-0.3 {sum(0<=g<0.3 for g in G)} 0.3-1 {sum(0.3<=g<1 for g in G)} >1 {sum(g>=1 for g in G)} | gap_given==computed(+-0.01) {sum(abs(a-b)<0.01 for a,b in Gg)}/{len(Gg)}")
    D = [s["duration_s"] for s in S]; OV = [s["overlap_ratio"] for s in S]
    print(f"REF duration median {st.median(D):.1f} min {min(D):.1f} max {max(D):.1f} | overlap_ratio median {st.median(OV):.3f} max {max(OV):.3f} | words in overlap {sum(x['ov'] for _, w in PREP.values() for x in w)}")
    roles = collections.Counter((v["role"], v["lang"]) for s in S for v in s["speakers"].values())
    print("REF role/lang", dict(roles))
    first = collections.Counter()
    for s in S:
        T, _ = PREP[s["session_id"]]; flip = zlib.crc32(s["session_id"].encode()) % 2; A = "German" if flip else "English"
        first[(f"A={A[:2]}", f"first={T[0]['lang'][:2]}", f"firstrole={s['speakers'][T[0]['spk']]['role']}")] += 1
    print("REF slotA/first-speaker cells", dict(first))
    nturns = [len(PREP[s['session_id']][0]) for s in S]
    print(f"REF turns/session median {st.median(nturns)} min {min(nturns)} max {max(nturns)}")


def analyze(name, H, kind):
    out = []  # per session result
    # pass 1: offset from plain alignment
    dts = []
    for s in S:
        T, Rw = PREP[s["session_id"]]
        hw = sorted([(t, lab, u) for lab, w, t in H.get(s["session_id"], []) for u in simple(w)], key=lambda x: x[0])
        for i, j in align([x["w"] for x in Rw], [x[2] for x in hw]):
            if i is not None and j is not None and Rw[i]["w"] == hw[j][2]: dts.append(hw[j][0] - Rw[i]["t"])
    off = st.median(dts); q = sorted(dts)
    stats = dict(off=off, iqr=(q[len(q) // 4], q[3 * len(q) // 4]))
    for s in S:
        sid = s["session_id"]; T, Rw = PREP[sid]
        hw = sorted([(t, lab, u) for lab, w, t in H.get(sid, []) for u in simple(w)], key=lambda x: x[0])
        pairs, _ = walign(Rw, hw, off, W)
        lab = [None] * len(Rw); exact = [False] * len(Rw); ins = 0; hword = [None] * len(Rw)
        for i, j in pairs:
            if i is not None and j is not None: lab[i] = hw[j][1]; exact[i] = hw[j][2] == Rw[i]["w"]; hword[i] = hw[j][2]
            elif i is None: ins += 1
        spks = sorted(s["speakers"])
        C = collections.Counter((lab[i], Rw[i]["spk"]) for i in range(len(Rw)) if lab[i] is not None)
        hl = sorted({l for l, _ in C} | ({l for _, l, _ in H.get(sid, [])}))
        # language-bound map (ours) and best bijection map
        best, bestmap = -1, {}
        for perm in itertools.permutations(hl + [None] * len(spks), len(spks)):
            if any(p is not None and perm.count(p) > 1 for p in perm): continue
            sc = sum(C[(p, sp)] for p, sp in zip(perm, spks) if p is not None)
            if sc > best: best, bestmap = sc, {p: sp for p, sp in zip(perm, spks) if p is not None}
        if kind == "lang":
            lmap = {s["speakers"][sp]["lang"]: sp for sp in spks}
        else:
            lmap = bestmap
        e_cp, e_agn, nref = cp_agn(s, H.get(sid, []))
        flip = zlib.crc32(sid.encode()) % 2
        hfirst = hw[0][1] if hw else None
        hchg = sum(1 for a, b in zip(hw, hw[1:]) if a[1] != b[1])
        out.append(dict(hchg=hchg, hword=hword, s=s, T=T, Rw=Rw, lab=lab, exact=exact, ins=ins, lmap=lmap, pmap=bestmap, cp=e_cp, agn=e_agn, n=nref, flip=flip, hfirst=hfirst, nhyp=len(hw)))
    report(name, kind, out, stats)
    return out


def report(name, kind, out, stats):
    P = lambda *a: print(f"[{name}]", *a)
    N = sum(len(o["Rw"]) for o in out); nal = sum(sum(l is not None for l in o["lab"]) for o in out); nex = sum(sum(o["exact"]) for o in out)
    cp = sum(o["cp"] for o in out); agn = sum(o["agn"] for o in out); nn = sum(o["n"] for o in out)
    P(f"offset median {stats['off']:.2f}s IQR [{stats['iqr'][0]:.2f},{stats['iqr'][1]:.2f}] | ref words {N} aligned {pct(nal, N)}% exact {pct(nex, N)}% | hyp words {sum(o['nhyp'] for o in out)} ins {sum(o['ins'] for o in out)} | cp {pct(cp, nn)} agn {pct(agn, nn)} attr {pct(cp - agn, nn)}")
    # wrong flags under language-bound (lmap) and perm-invariant (pmap)
    for o in out:
        o["wrongL"] = [None if l is None else o["lmap"].get(l) != x["spk"] for l, x in zip(o["lab"], o["Rw"])]
        o["wrongP"] = [None if l is None else o["pmap"].get(l) != x["spk"] for l, x in zip(o["lab"], o["Rw"])]
    wl = sum(sum(bool(w) for w in o["wrongL"]) for o in out); wp = sum(sum(bool(w) for w in o["wrongP"]) for o in out)
    wl_ex = sum(sum(bool(w) and e for w, e in zip(o["wrongL"], o["exact"])) for o in out)
    swapped = sum(1 for o in out if kind == "lang" and sum(bool(w) for w in o["wrongL"]) > sum(w is False for w in o["wrongL"]))
    P(f"misattributed aligned words: lang-bound {wl} ({pct(wl, N)}% of ref, {pct(wl, nal)}% of aligned; exact-match share {pct(wl_ex, wl)}%) | perm-invariant {wp} ({pct(wp, N)}%) | sessions majority-swapped {swapped}")
    key = "wrongL"
    TOT = wl if wl else 1
    # ---- turn-level classification
    tc = collections.Counter(); tw = collections.Counter(); sub = collections.Counter()
    for o in out:
        o["turnstate"] = []
        for ti, t in enumerate(o["T"]):
            idx = sorted([i for i, x in enumerate(o["Rw"]) if x["ti"] == ti and o[key][i] is not None], key=lambda i: o["Rw"][i]["pos"])
            fl = [o[key][i] for i in idx]; nw = sum(fl)
            role = "first" if t["change"] is None else ("change" if t["change"] else "cont")
            if not idx: cat = "noalign"
            elif nw == 0: cat = "correct"
            elif nw == len(fl): cat = "whole"
            else:
                cat = "partial"
                segs = [(k, len(list(g))) for k, g in itertools.groupby(fl)]
                wsegs = [i for i, (k, _) in enumerate(segs) if k]
                if len(wsegs) > 1: sub["multi"] += nw
                elif wsegs[0] == 0: sub["prefix(late switch)"] += nw
                elif wsegs[0] == len(segs) - 1: sub["suffix(early switch)"] += nw
                else: sub["middle"] += nw
            tc[(cat, role)] += 1; tw[(cat, role)] += nw
            o["turnstate"].append(dict(cat=cat, nw=nw, nal=len(fl), maj=(nw * 2 > len(fl)) if fl else None, first=(fl[0] if fl else None), last=(fl[-1] if fl else None),
                                       firstlab=(o["lab"][idx[0]] if idx else None), lastlab=(o["lab"][idx[-1]] if idx else None)))
    for role in ("first", "change", "cont"):
        n_t = sum(v for (c, r), v in tc.items() if r == role and c != "noalign")
        P(f"turns[{role}] n={n_t} whole-wrong {tc[('whole', role)]} ({pct(tc[('whole', role)], n_t)}%) partial {tc[('partial', role)]} correct {tc[('correct', role)]} noalign {tc[('noalign', role)]} | wrong words: whole {tw[('whole', role)]} partial {tw[('partial', role)]} (share of all wrong {pct(tw[('whole', role)] + tw[('partial', role)], TOT)}%)")
    whole_w = sum(v for (c, r), v in tw.items() if c == "whole"); part_w = sum(v for (c, r), v in tw.items() if c == "partial")
    P(f"wrong words in whole-wrong turns {whole_w} ({pct(whole_w, TOT)}%) vs partial {part_w} ({pct(part_w, TOT)}%) | partial subtype words {dict(sub)}")
    # ---- recognition quality inside wrong vs correct change turns, and language of hyp words (TAXI-reference lexicon)
    q = collections.defaultdict(collections.Counter)
    for o in out:
        for i, x in enumerate(o["Rw"]):
            t = o["T"][x["ti"]]; ts = o["turnstate"][x["ti"]]
            if not t["change"]: continue
            k = ts["cat"]; q[k]["ref"] += 1; q[k]["al"] += o["lab"][i] is not None; q[k]["ex"] += o["exact"][i]
    P("recognition in change turns by attribution cat " + " | ".join(f"{k}: ref {c['ref']} aligned {pct(c['al'], c['ref']).strip()}% exact {pct(c['ex'], c['ref']).strip()}%" for k, c in sorted(q.items())))
    lg = collections.defaultdict(collections.Counter)
    for o in out:
        for i, x in enumerate(o["Rw"]):
            if o[key][i] is None: continue
            true = o["T"][x["ti"]]["lang"]; hl = wlang(o["hword"][i]); rel = "own-lang" if hl == true else ("other-lang" if hl in VOC else hl)
            k = ("WRONG" if o[key][i] else "ok") + ("/exact" if o["exact"][i] else "/sub"); lg[k][rel] += 1
    P("hyp-word language vs speaker language (lexicon) " + " | ".join(f"{k}: " + " ".join(f"{r} {pct(c[r], sum(c.values())).strip()}%" for r in ["own-lang", "other-lang", "both", "oov"]) + f" (n{sum(c.values())})" for k, c in sorted(lg.items())))
    refchg = sum(sum(1 for a, b in zip(o["T"], o["T"][1:]) if a["spk"] != b["spk"]) for o in out)
    P(f"hyp slot changes total {sum(o['hchg'] for o in out)} vs ref speaker changes {refchg}")
    # ---- (b) position in turn
    PB = [0, 1, 2, 4, 9, 19, 10 ** 6]; PN = ["0", "1", "2", "3-4", "5-9", "10-19", "20+"]
    for role in ("change", "cont", "first"):
        c = collections.Counter(); n = collections.Counter()
        for o in out:
            for i, x in enumerate(o["Rw"]):
                t = o["T"][x["ti"]]; r = "first" if t["change"] is None else ("change" if t["change"] else "cont")
                if r != role or o[key][i] is None: continue
                b = binof(x["pos"], PB, PN); n[b] += 1; c[b] += o[key][i]
        P(f"pos-in-turn[{role}] " + " ".join(f"{b}:{pct(c[b], n[b]).strip()}%(n{n[b]})" for b in PN))
    # first-k words of change turns vs later (only partial+correct turns i.e. excluding whole-wrong turns -> late-switch signature)
    c = collections.Counter(); n = collections.Counter()
    for o in out:
        for i, x in enumerate(o["Rw"]):
            t = o["T"][x["ti"]]
            if not t["change"] or o[key][i] is None or o["turnstate"][x["ti"]]["cat"] == "whole": continue
            b = "k<3" if x["pos"] < 3 else "k>=3"; n[b] += 1; c[b] += o[key][i]
    P(f"change turns excluding whole-wrong: wrong rate first3 {pct(c['k<3'], n['k<3'])}% (n{n['k<3']}) later {pct(c['k>=3'], n['k>=3'])}% (n{n['k>=3']})")
    # ---- (c) turn length
    LB = [3, 10, 20, 10 ** 6]; LN = ["<=3", "4-10", "11-20", ">20"]
    c = collections.Counter(); n = collections.Counter(); wc = collections.Counter(); wn = collections.Counter(); mj = collections.Counter()
    for o in out:
        for ti, t in enumerate(o["T"]):
            ts = o["turnstate"][ti]
            if ts["nal"] == 0: continue
            b = binof(t["n"], LB, LN); n[b] += 1; c[b] += ts["cat"] == "whole"; mj[b] += bool(ts["maj"]); wc[b] += ts["nw"]; wn[b] += ts["nal"]
    P("turn-len " + " | ".join(f"{b}: turns {n[b]} whole-wrong {pct(c[b], n[b]).strip()}% maj-wrong {pct(mj[b], n[b]).strip()}% word-wrong {pct(wc[b], wn[b]).strip()}% share {pct(wc[b], TOT).strip()}%" for b in LN))
    # ---- (d) gap before turn (change turns)
    GB = [-1e-9, 0.3, 1.0, 1e9]; GN = ["overlap", "0-0.3", "0.3-1", ">1"]
    for role in ("change", "cont"):
        c = collections.Counter(); n = collections.Counter(); wc = collections.Counter(); wn = collections.Counter()
        for o in out:
            for ti, t in enumerate(o["T"]):
                ts = o["turnstate"][ti]
                if ts["nal"] == 0 or t["change"] is None or (role == "change") != t["change"]: continue
                b = binof(t["gap"], GB, GN) if t["gap"] >= 0 else "overlap"; n[b] += 1; c[b] += bool(ts["maj"]); wc[b] += ts["nw"]; wn[b] += ts["nal"]
        P(f"gap[{role}] " + " | ".join(f"{b}: turns {n[b]} maj-wrong {pct(c[b], n[b]).strip()}% word-wrong {pct(wc[b], wn[b]).strip()}% share {pct(wc[b], TOT).strip()}%" for b in GN))
    # ---- (e) first turn / slot initialisation (ours only)
    if kind == "lang":
        cells = collections.defaultdict(lambda: collections.Counter())
        for o in out:
            s = o["s"]; T = o["T"]; A = "German" if o["flip"] else "English"; fl = T[0]["lang"]
            k = (f"A={A[:2]}", f"first={fl[:2]}", "A==first" if A == fl else "A!=first")
            c = cells[k]; c["sess"] += 1; ts = o["turnstate"][0]
            c["first_maj_wrong"] += bool(ts["maj"]); c["first_whole"] += ts["cat"] == "whole"
            c["hyp_starts_A"] += o["hfirst"] == A; c["hyp_starts_firstlang"] += o["hfirst"] == fl
            c["wrong"] += sum(bool(w) for w in o[key]); c["al"] += sum(w is not None for w in o[key])
            c["swapped"] += sum(bool(w) for w in o[key]) > sum(w is False for w in o[key])
            c["sess_with_err"] += any(o[key])
        for k, c in sorted(cells.items()):
            P(f"init {k}: sess {c['sess']} first-turn maj-wrong {c['first_maj_wrong']} (whole {c['first_whole']}) | hyp first word in slot A {c['hyp_starts_A']} / in first-speaker lang {c['hyp_starts_firstlang']} | word-wrong {pct(c['wrong'], c['al'])}% | sess with err {c['sess_with_err']} swapped {c['swapped']}")
        agg = collections.Counter()
        for k, c in cells.items():
            agg[k[2] + "_sess"] += c["sess"]; agg[k[2] + "_wrong"] += c["wrong"]; agg[k[2] + "_al"] += c["al"]; agg[k[2] + "_f"] += c["first_maj_wrong"]; agg[k[2] + "_hA"] += c["hyp_starts_A"]
        P(f"init summary: A==first sess {agg['A==first_sess']} word-wrong {pct(agg['A==first_wrong'], agg['A==first_al'])}% first-turn wrong {agg['A==first_f']} hyp-starts-A {agg['A==first_hA']} | A!=first sess {agg['A!=first_sess']} word-wrong {pct(agg['A!=first_wrong'], agg['A!=first_al'])}% first-turn wrong {agg['A!=first_f']} hyp-starts-A {agg['A!=first_hA']}")
        byl = collections.Counter()
        for o in out:
            for i, x in enumerate(o["Rw"]):
                if o[key][i] is None: continue
                lg = o["T"][x["ti"]]["lang"][:2]; byl[lg + "_n"] += 1; byl[lg + "_w"] += o[key][i]
        P(f"wrong rate by true language: EN {pct(byl['En_w'], byl['En_n'])}% DE {pct(byl['Ge_w'], byl['Ge_n'])}%")
    # ---- (f) streaks over turns (maj-wrong), and toggles at changes
    tr = collections.Counter(); streaks = []; to_end = 0
    for o in out:
        seq = [(ts["maj"], t["change"]) for ts, t in zip(o["turnstate"], o["T"]) if ts["maj"] is not None]
        for (a, _), (b, ch) in zip(seq, seq[1:]): tr[(a, b, ch)] += 1
        run = 0
        for k, (a, _) in enumerate(seq):
            if a: run += 1
            if (not a or k == len(seq) - 1) and run:
                streaks.append(run); to_end += bool(a and k == len(seq) - 1); run = 0
    def cond(a, ch=None):
        num = sum(v for (x, y, c), v in tr.items() if x == a and y and (ch is None or c == ch)); den = sum(v for (x, y, c), v in tr.items() if x == a and (ch is None or c == ch))
        return f"{pct(num, den).strip()}% (n{den})"
    P(f"streak: P(next maj-wrong|wrong) {cond(True)} [at change {cond(True, True)}, cont {cond(True, False)}] vs P(next wrong|correct) {cond(False)} [change {cond(False, True)}, cont {cond(False, False)}]")
    sc = collections.Counter(min(x, 5) for x in streaks)
    P(f"streak lengths(turns) {dict(sorted(sc.items()))} (5=5+) | streaks {len(streaks)} reaching session end {to_end} | wrong-turn total {sum(streaks)}")
    tg = collections.Counter()
    for o in out:
        for ti in range(1, len(o["T"])):
            t = o["T"][ti]
            if not t["change"]: continue
            p, c = o["turnstate"][ti - 1], o["turnstate"][ti]
            if p["lastlab"] is None or c["firstlab"] is None: continue
            tog = p["lastlab"] != c["firstlab"]; tg[(bool(p["last"]), bool(c["first"]), tog)] += 1
    nt = sum(tg.values())
    P(f"at ref speaker changes (n{nt}): toggled {pct(sum(v for k, v in tg.items() if k[2]), nt)}% | prevOK->curOK {tg[(False, False, True)]}+{tg[(False, False, False)]}(no-toggle) prevOK->curWRONG(missed change) {tg[(False, True, False)]}+{tg[(False, True, True)]}(toggle) prevWRONG->curWRONG(toggled, inverted state kept) {tg[(True, True, True)]}+{tg[(True, True, False)]}(no-toggle) prevWRONG->curOK(re-sync) {tg[(True, False, False)]}(no-toggle)+{tg[(True, False, True)]}(toggle)")
    # within-turn hyp switches
    wsw = 0; wpairs = 0
    for o in out:
        for ti in range(len(o["T"])):
            idx = sorted([i for i, x in enumerate(o["Rw"]) if x["ti"] == ti and o["lab"][i] is not None], key=lambda i: o["Rw"][i]["pos"])
            for a, b in zip(idx, idx[1:]): wpairs += 1; wsw += o["lab"][a] != o["lab"][b]
    P(f"within-turn hyp slot switches {wsw} per {wpairs} adjacent aligned word pairs ({pct(wsw, wpairs)}%)")
    # ---- onset decomposition over time-ordered aligned words
    onset = collections.Counter(); onset_w = collections.Counter(); rec = collections.Counter(); onset_ov = collections.Counter()
    for o in out:
        seq = [(i, o[key][i]) for i in range(len(o["Rw"])) if o[key][i] is not None]
        cur = None; prev_i = None; prev_w = False
        for i, w in seq:
            x = o["Rw"][i]
            if w and not prev_w:
                if prev_i is None: typ = "session-start"
                else:
                    px = o["Rw"][prev_i]
                    if px["ti"] == x["ti"]: typ = "within-turn"
                    elif px["spk"] != x["spk"]: typ = "speaker-change(missed)"
                    else: typ = "same-speaker-next-turn"
                onset[typ] += 1; cur = typ; onset_ov[typ] += x["ov"]
            if not w and prev_w and prev_i is not None:
                px = o["Rw"][prev_i]
                rec["within-turn" if px["ti"] == x["ti"] else ("speaker-change" if px["spk"] != x["spk"] else "same-speaker-next-turn")] += 1
            if w: onset_w[cur] += 1
            prev_i, prev_w = i, w
    P(f"error onsets {dict(onset)} (onset word in overlap {dict(onset_ov)}) | wrong words by onset type " + " ".join(f"{k}:{pct(v, TOT).strip()}%" for k, v in onset_w.most_common()) + f" | recoveries {dict(rec)}")
    # ---- clues: time since session start, previous turn short, speaker balance
    TB = [5, 15, 30, 50, 1e9]; TN = ["0-5s", "5-15", "15-30", "30-50", "50+"]
    c = collections.Counter(); n = collections.Counter()
    for o in out:
        for i, x in enumerate(o["Rw"]):
            if o[key][i] is None: continue
            b = binof(x["tm"], TB, TN); n[b] += 1; c[b] += o[key][i]
    P("wrong by time-in-session " + " ".join(f"{b}:{pct(c[b], n[b]).strip()}%(n{n[b]})" for b in TN))
    c = collections.Counter(); n = collections.Counter()
    for o in out:
        for ti in range(1, len(o["T"])):
            t, p, ts = o["T"][ti], o["T"][ti - 1], o["turnstate"][ti]
            if not t["change"] or ts["maj"] is None: continue
            k = "prev<=3w" if p["n"] <= 3 else "prev>3w"; n[k] += 1; c[k] += ts["maj"]
            k2 = "prev_dur<1s" if p["b"] - p["a"] < 1 else ("prev_dur1-4s" if p["b"] - p["a"] < 4 else "prev_dur>=4s"); n[k2] += 1; c[k2] += ts["maj"]
            k3 = "cur<=3w" if t["n"] <= 3 else "cur>3w"; n[k3] += 1; c[k3] += ts["maj"]
    P("change-turn maj-wrong by context " + " ".join(f"{k}:{pct(c[k], n[k]).strip()}%(n{n[k]})" for k in ["prev<=3w", "prev>3w", "prev_dur<1s", "prev_dur1-4s", "prev_dur>=4s", "cur<=3w", "cur>3w"]))
    # turn after a backchannel-like short turn that was itself wrong/right
    c = collections.Counter(); n = collections.Counter()
    for o in out:
        for ti in range(1, len(o["T"])):
            p, ts, pts = o["T"][ti - 1], o["turnstate"][ti], o["turnstate"][ti - 1]
            if ts["maj"] is None or pts["maj"] is None: continue
            k = ("prevShort" if p["n"] <= 3 else "prevLong") + ("_prevWrong" if pts["maj"] else "_prevOK"); n[k] += 1; c[k] += ts["maj"]
    P("next-turn maj-wrong by prev(short/long, wrong/ok) " + " ".join(f"{k}:{pct(c[k], n[k]).strip()}%(n{n[k]})" for k in sorted(n)))
    # ---- session level
    per = []
    for o in out:
        wr = sum(bool(w) for w in o[key]); al = sum(w is not None for w in o[key])
        per.append((wr, o["cp"] - o["agn"], o))
    nerr = sum(1 for wr, _, _ in per if wr > 0); n5 = sum(1 for wr, _, _ in per if wr >= 5); n20 = sum(1 for wr, _, _ in per if wr >= 20)
    per.sort(key=lambda x: -x[0]); top = per[:10]
    attr_tot = sum(max(0, a) for _, a, _ in per)
    per_a = sorted(per, key=lambda x: -x[1])
    P(f"sessions with >=1 wrong word {nerr}/86, >=5 {n5}, >=20 {n20} | worst-10 share of wrong words {pct(sum(w for w, _, _ in top), TOT)}% | worst-10 share of cp-agn {pct(sum(max(0, a) for _, a, _ in per_a[:10]), attr_tot)}% | worst-20 share wrong words {pct(sum(w for w, _, _ in per[:20]), TOT)}%")
    if kind == "lang" or name.startswith("VV"):
        allL = [o["T"] for _, _, o in per]
        for r, (wr, a, o) in enumerate(top, 1):
            s, T = o["s"], o["T"]; L = [t["n"] for t in T]; ov = sum(1 for t in T if t["gap"] is not None and t["gap"] < 0)
            spw = collections.Counter()
            for t in T: spw[t["spk"]] += t["n"]
            minor = min(spw.values()) / max(1, sum(spw.values()))
            A = "German" if o["flip"] else "English"; al = sum(w is not None for w in o[key])
            stt = collections.Counter(ts["cat"] for ts in o["turnstate"])
            P(f"  worst#{r}: wrong {wr}/{al} aligned ({pct(wr, al).strip()}%) cp-agn {a} | dur {s['duration_s']:.0f}s turns {len(T)} (change {sum(1 for t in T if t['change'])}) turn-len med {st.median(L)} min {min(L)} max {max(L)} <=3w {sum(x <= 3 for x in L)} | overlapped-turn starts {ov} overlap_ratio {s['overlap_ratio']:.3f} | first {T[0]['lang'][:2]} slotA {A[:2] if kind == 'lang' else '-'} flip {o['flip']} | minority-speaker word share {minor:.2f} | turn cats {dict(stt)} | first turn {o['turnstate'][0]['cat']}")
    # correlation of session-level wrong rate with speaker balance and short-turn share
    bal = collections.defaultdict(lambda: [0, 0, 0])
    for wr, a, o in per:
        spw = collections.Counter()
        for t in o["T"]: spw[t["spk"]] += t["n"]
        minor = min(spw.values()) / max(1, sum(spw.values()))
        k = "minor<0.2" if minor < 0.2 else ("minor0.2-0.35" if minor < 0.35 else "minor>=0.35")
        bal[k][0] += 1; bal[k][1] += wr; bal[k][2] += sum(w is not None for w in o[key])
    P("session wrong rate by speaker balance " + " ".join(f"{k}: sess {v[0]} wrong {pct(v[1], v[2]).strip()}%" for k, v in sorted(bal.items())))
    sw = collections.defaultdict(lambda: [0, 0, 0])
    for wr, a, o in per:
        sh = sum(1 for t in o["T"] if t["n"] <= 3) / len(o["T"])
        k = "short-turn-share<0.2" if sh < 0.2 else ("0.2-0.4" if sh < 0.4 else ">=0.4")
        sw[k][0] += 1; sw[k][1] += wr; sw[k][2] += sum(w is not None for w in o[key])
    P("session wrong rate by short-turn share " + " ".join(f"{k}: sess {v[0]} wrong {pct(v[1], v[2]).strip()}%" for k, v in sorted(sw.items())))
    dd = collections.defaultdict(lambda: [0, 0, 0])
    for wr, a, o in per:
        d = o["s"]["duration_s"]; k = "<30s" if d < 30 else ("30-50s" if d < 50 else ">=50s")
        dd[k][0] += 1; dd[k][1] += wr; dd[k][2] += sum(w is not None for w in o[key])
    P("session wrong rate by duration " + " ".join(f"{k}: sess {v[0]} wrong {pct(v[1], v[2]).strip()}%" for k, v in sorted(dd.items())))
    if name.startswith("VV"):
        labs = collections.Counter(len({l for l, _, _ in o_h}) for o_h in [])
        nl = collections.Counter(len(set(l for l in o["lab"] if l is not None)) for _, _, o in per)
        P(f"VV distinct hyp speaker labels (aligned) per session {dict(sorted(nl.items()))}")
        short = collections.Counter()
        for _, _, o in per:
            for ti, t in enumerate(o["T"]):
                ts = o["turnstate"][ti]
                if ts["nal"] and t["n"] <= 3: short["n"] += 1; short["maj"] += bool(ts["maj"]); short["any"] += ts["nw"] > 0
        P(f"VV short turns (<=3w) aligned {short['n']} maj-wrong {short['maj']} any-wrong {short['any']}")


for name, loader, kind in SYS:
    if ONLY and name not in ONLY: continue
    H = loader()
    if kind == "lang":
        print(f"[{name}] hyp label values {dict(collections.Counter(l for v in H.values() for l, _, _ in v))}")
    analyze(name, H, kind)
    sys.stdout.flush()

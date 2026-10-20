
# ---- follow-up analyses (appended after attr_diag.py with ONLY=NONE) ----
def core(H, kind):
    dts = []
    for s in S:
        T, Rw = PREP[s["session_id"]]
        hw = sorted([(t, lab, u) for lab, w, t in H.get(s["session_id"], []) for u in simple(w)], key=lambda x: x[0])
        for i, j in align([x["w"] for x in Rw], [x[2] for x in hw]):
            if i is not None and j is not None and Rw[i]["w"] == hw[j][2]: dts.append(hw[j][0] - Rw[i]["t"])
    off = st.median(dts); out = []
    for s in S:
        sid = s["session_id"]; T, Rw = PREP[sid]
        hw = sorted([(t, lab, u) for lab, w, t in H.get(sid, []) for u in simple(w)], key=lambda x: x[0])
        pairs, _ = walign(Rw, hw, off, W)
        lab = [None] * len(Rw); hword = [None] * len(Rw); exact = [False] * len(Rw); hj = [None] * len(Rw)
        for i, j in pairs:
            if i is not None and j is not None: lab[i] = hw[j][1]; hword[i] = hw[j][2]; exact[i] = hw[j][2] == Rw[i]["w"]; hj[i] = j
        lmap = {s["speakers"][sp]["lang"]: sp for sp in s["speakers"]}
        wrong = [None if l is None else lmap.get(l) != x["spk"] for l, x in zip(lab, Rw)]
        ts = []
        for ti, t in enumerate(T):
            idx = sorted([i for i, x in enumerate(Rw) if x["ti"] == ti and wrong[i] is not None], key=lambda i: Rw[i]["pos"])
            fl = [wrong[i] for i in idx]
            cat = "noalign" if not idx else ("correct" if not any(fl) else ("whole" if all(fl) else "partial"))
            ts.append(dict(cat=cat, idx=idx, maj=(sum(fl) * 2 > len(fl)) if fl else None, firstlab=lab[idx[0]] if idx else None, lastlab=lab[idx[-1]] if idx else None))
        out.append(dict(s=s, T=T, Rw=Rw, hw=hw, pairs=pairs, lab=lab, hword=hword, exact=exact, wrong=wrong, ts=ts, flip=zlib.crc32(sid.encode()) % 2, lmap=lmap))
    return out


def follow(name, out):
    P = lambda *a: print(f"[{name}]", *a)
    # A. whole-wrong turns: translated into slot language vs kept own language
    mode = collections.Counter(); mode_w = collections.Counter(); direc = collections.Counter()
    for o in out:
        for ti, t in enumerate(o["T"]):
            tsx = o["ts"][ti]
            if tsx["cat"] != "whole": continue
            own = sum(1 for i in tsx["idx"] if wlang(o["hword"][i]) == t["lang"]); oth = sum(1 for i in tsx["idx"] if wlang(o["hword"][i]) not in (t["lang"], "both", "oov"))
            m = "translated(>=50% other-lang)" if oth * 2 >= len(tsx["idx"]) else ("kept-own-lang(other-lang<=10%)" if oth * 10 <= len(tsx["idx"]) else "mixed")
            k = f"{t['lang'][:2]}->{'De' if t['lang'] == 'English' else 'En'}slot"
            mode[(k, m)] += 1; mode_w[(k, m)] += len(tsx["idx"]); direc[k] += 1
    P("whole-wrong turns by direction/mode " + " | ".join(f"{k} {m}: turns {v} words {mode_w[(k, m)]}" for (k, m), v in sorted(mode.items())))
    # B. missed-change rate by context (change turns only, maj-wrong while previous turn correct = onset)
    def rate(keyf, label):
        c = collections.Counter(); n = collections.Counter()
        for o in out:
            nchg = 0
            for ti in range(1, len(o["T"])):
                t, p = o["T"][ti], o["T"][ti - 1]
                if not t["change"]: continue
                nchg += 1
                if o["ts"][ti]["maj"] is None or o["ts"][ti - 1]["maj"] is None or o["ts"][ti - 1]["maj"]: continue
                k = keyf(o, ti, t, p, nchg); n[k] += 1; c[k] += o["ts"][ti]["maj"]
        P(f"onset rate (new turn wrong | prev turn ok) by {label}: " + " ".join(f"{k}:{pct(c[k], n[k]).strip()}%(n{n[k]})" for k in sorted(n)))
    rate(lambda o, ti, t, p, k: "intoEN" if t["lang"] == "English" else "intoDE", "direction")
    rate(lambda o, ti, t, p, k: "newSlot=A" if ("German" if o["flip"] else "English") == t["lang"] else "newSlot=B", "slot of new speaker")
    rate(lambda o, ti, t, p, k: f"change#{min(k, 4)}", "change index (4=4+)")
    rate(lambda o, ti, t, p, k: ("EN" if t["lang"] == "English" else "DE") + ("<=3w" if t["n"] <= 3 else "4-10w" if t["n"] <= 10 else "11-20w" if t["n"] <= 20 else ">20w"), "new-turn lang x length")
    rate(lambda o, ti, t, p, k: "dur<2s" if t["b"] - t["a"] < 2 else ("2-5s" if t["b"] - t["a"] < 5 else "5s+"), "new-turn duration")
    rate(lambda o, ti, t, p, k: "t<10s" if t["a"] < 10 else ("10-25s" if t["a"] < 25 else ("25-40s" if t["a"] < 40 else "40s+")), "turn start time")
    L = collections.defaultdict(list)
    for o in out:
        for t in o["T"]: L[t["lang"]].append(t["n"])
    P("turn length by lang " + " ".join(f"{k[:2]}: n{len(v)} median {st.median(v)} mean {st.mean(v):.1f}" for k, v in L.items()))
    # D. oracle slot relabel: remaining cp
    e_cp = e_cp_or = e_agn = nn = 0
    for o in out:
        s = o["s"]; spks = sorted(s["speakers"]); turns = sorted(s["turns"], key=lambda t: t["start_s"])
        refs = {sp: [w for t in turns if t["speaker"] == sp for w in simple(t["transcript"])] for sp in spks}
        lab2 = {j: o["Rw"][i]["spk"] for i, j in o["pairs"] if i is not None and j is not None}
        hyp_by = collections.defaultdict(list); hyp_by_or = collections.defaultdict(list)
        for j, (t, l, u) in enumerate(o["hw"]):
            hyp_by[o["lmap"].get(l, "X")].append(u); hyp_by_or[lab2.get(j, o["lmap"].get(l, "X"))].append(u)
        e_cp += sum(ed(refs[sp], hyp_by.get(sp, [])) for sp in spks) + sum(len(v) for k, v in hyp_by.items() if k not in spks)
        e_cp_or += sum(ed(refs[sp], hyp_by_or.get(sp, [])) for sp in spks) + sum(len(v) for k, v in hyp_by_or.items() if k not in spks)
        e_agn += ed([w for t in turns for w in simple(t["transcript"])], [u for _, _, u in o["hw"]]); nn += sum(len(v) for v in refs.values())
    # translation leakage: substituted aligned words in WRONG pairs whose hyp word is other-lang
    leak = sum(1 for o in out for i, x in enumerate(o["Rw"]) if o["wrong"][i] and not o["exact"][i] and wlang(o["hword"][i]) not in (o["T"][x["ti"]]["lang"], "both", "oov"))
    leak_all_sub = sum(1 for o in out for i, x in enumerate(o["Rw"]) if o["wrong"][i] and not o["exact"][i])
    P(f"lang-bound cpWER {pct(e_cp, nn)} | oracle slot relabel of aligned words -> cpWER {pct(e_cp_or, nn)} (agn {pct(e_agn, nn)}) | wrong-slot substitutions {leak_all_sub} ({pct(leak_all_sub, nn)}% of ref) of which other-lang (translated) {leak} ({pct(leak, nn)}% of ref)")
    # E. worst sessions: slot collapse
    per = sorted(out, key=lambda o: -sum(bool(w) for w in o["wrong"]))[:10]
    for r, o in enumerate(per, 1):
        hs = collections.Counter(l for _, l, _ in o["hw"]); rs = collections.Counter(o["T"][x["ti"]]["lang"] for x in o["Rw"])
        dom = max(hs, key=hs.get) if hs else None
        wt = collections.Counter(f"{o['T'][ti]['lang'][:2]}" for ti, tsx in enumerate(o["ts"]) if tsx["maj"])
        P(f"  worst#{r}: hyp share of dominant slot {pct(hs[dom], sum(hs.values())).strip()}% vs ref share of that lang {pct(rs[dom], sum(rs.values())).strip()}% | wrong turns by true lang {dict(wt)}")
    # F. continuation turns wrong: reference-label sanity via lexicon of the ref words
    for o in out:
        for ti, t in enumerate(o["T"]):
            if t["change"] is False and o["ts"][ti]["cat"] == "whole":
                c = collections.Counter(wlang(x["w"]) for x in o["Rw"] if x["ti"] == ti)
                P(f"  cont-turn whole-wrong: labelled {t['lang'][:2]}, ref-word lexicon own {c[t['lang']]} other {sum(v for k, v in c.items() if k not in (t['lang'], 'both', 'oov'))} both {c['both']} (n{t['n']}), prev-turn same speaker gap {t['gap']:.2f}s")


for name, loader, kind in SYS:
    if name not in ("V1d10", "V2d10", "V2notap", "V1d4"): continue
    follow(name, core(loader(), kind)); sys.stdout.flush()

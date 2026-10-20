#!/usr/bin/env python
"""턴 결정 지표(V3 C4) — 화자 교대 결정을 잰다. 입력: cstbench 세션(sessions.jsonl — TAXI 또는 sim_to_sessions 로 바꾼 비-TAXI dev)과 우리 받아쓰기 출력
(cst_stage4_taxi run 의 asr-d{δ}.jsonl: session·lang = 슬롯 언어·t·text), 선택으로 <EOT> 방출(taxi_v3_probe 의 eot-d{δ}.jsonl: session·lang·t)과 비교 시스템의 asr 파일.
scratchpad v3diag 의 attr_diag.py·eot_study.py 를 이식했다(TAXI 경로 없이 같은 정의). 산출은 집계뿐이다(세션 id·글자 없음).

  python experiments/cst_attr_turns.py --sessions <sessions.jsonl> --asr <asr-d10.jsonl> [--eot <eot-d10.jsonl>] [--against <asr 파일 | 결과 폴더>] \\
      [--window 5] [--boot 1000] [--seed 0] [--out <turn-metrics.json>]
  cst_stage4_taxi.py score --turn-metrics 는 asr-d*.jsonl 마다 같은 지표를 turn-metrics-d{δ}.json 으로 쓴다(같은 폴더의 eot-d{δ}.jsonl 이 있으면 함께).

정렬: 참조 단어 = 턴 전사(공통 정규화 simple_words), 시각 = 턴 구간을 단어 수로 고르게 나눈 단어 끝. 가설 단어 = 방출 시각순(같은 시각은 파일 순서).
  시스템 오프셋 = 화자 무시 편집 거리 정렬(cst_eval.align)에서 정확히 일치한 쌍의 (가설 − 참조) 시각 중앙값. 그 오프셋을 빼고 ±W s 안에서만 짝을 허용하는
  편집 거리 정렬(walign)로 참조 단어마다 가설 단어의 슬롯 언어를 물려받는다. 슬롯 언어가 참조 화자 언어와 다르면 오배정(언어 고정 — 화자 언어가 서로 다른
  세션만 잰다, 같은 세션은 skipped_same_lang 으로 센다).
턴: first(세션 첫 턴)·change(앞 턴과 화자가 다름)·cont(같은 화자) × correct(정렬 단어 모두 맞음)·whole(모두 틀림)·partial·noalign.
  partial 의 오배정 단어: late(턴 첫머리만 — 교대를 늦게 냄)·early(턴 끝만 — 일찍 냄)·middle·multi.
지표(키):
  toggle_recall      참조 교대에서 앞 턴 마지막 정렬 단어와 새 턴 첫 정렬 단어의 슬롯이 다른 비율(교대를 냈다)과 앞·새 턴 맞음/틀림 전이
  changes            가설 슬롯 교대(시각순 인접 가설 단어) 대 참조 교대 수와 비
  whole_wrong_change 통째로 틀린 change 턴(정렬 단어가 있는 change 턴 대비)
  within_turn_switch 같은 턴 안 인접 정렬 단어 쌍의 슬롯이 다른 비율(턴 안 헛교대)
  onset_miss         앞 턴 다수가 맞을 때 새 턴 다수가 틀린 비율, 새 화자 언어별(방향)
  short_turns        change 턴 ≤3 단어 대 >3 단어의 통째·다수 틀림, 턴 길이 구간별
  wrong_slot_lang    오배정 치환 단어 중 다른 언어 단어 — 언어 판정은 세션 참조 전사로 만든 언어별 어휘(번역 누출)
  slot_collapse      한 슬롯이 가설 단어의 ≥90 % 인데 그 언어의 참조 몫은 ≤70 % 인 세션
  first_turn         첫 턴(명시적 첫 선택) 분류
  error_onsets       오배정 구간이 시작한 곳(교대 놓침·턴 안·같은 화자 다음 턴)과 그 뒤 오배정 단어 몫
  split              cp·agn·attr(cst_stage4_taxi.attr_split 과 같은 정의, %)
  eot (--eot)        change 턴마다 경계 구간(앞 턴 끝 + 오프셋 − 0.4 s … 새 턴 시작 + 오프셋 + 0.6 s)의 <EOT>: 앞 턴(다수) 슬롯·다른 슬롯·없음.
                     majority = eot_study 와 같은 hit/miss(새 턴 다수 슬롯이 맞음/틀림), toggle = 교대를 냈음/놓침. f_hit·f_miss = 앞 슬롯 <EOT> 비율
  bootstrap          세션 재표집(고정 seed) 95 % CI(백분위). --against 가 있으면 같은 재표집에서 (이 시스템 − 비교) 차의 CI(paired)
온셋 분류(taxi_v3_probe --classify-onsets, onset_classes): 창별 원출력 [(k, tid)] 로 같은 가설 단어(asr 행과 같은 단어·시각·순서)를 만들고 같은 정렬을 한 뒤,
  교대마다 앞 턴 마지막 정렬 가설 단어와 새 턴 첫 정렬 가설 단어 사이 원출력 구간을 parse_emitted4 상태기계(cur, kind)로 다시 읽는다(onset_class 설명).
  × 이전 슬롯 <EOT> 있음/없음 × 새 화자 언어(방향)로 센다."""
import argparse, collections, itertools, json, os, random, re, statistics as st, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cst-bench", "src"))
from cst_stage4_taxi import attr_split, simple_words as simple  # noqa: E402

CHUNK_S = 0.08
ROLES, CATS, SUBS = ("first", "change", "cont"), ("correct", "whole", "partial", "noalign"), ("late", "early", "middle", "multi")
LEN_BINS = ((3, "<=3"), (10, "4-10"), (20, "11-20"), (10 ** 6, ">20"))
COLLAPSE = (0.9, 0.7)                                   # (가설 지배 슬롯 몫 ≥, 그 언어 참조 몫 ≤) — slot_collapse 판정
ONSET = ("OK", "RESUME", "REASSERT", "SILENT", "ABSORBED")


def prep(s):
    """세션 → (턴 [dict(spk, lang, a, b, n, gap, change)] 시각순, 참조 단어 [dict(t, tm, ti, pos, w, spk, ov)] 단어 가운데 시각순) — attr_diag.prep 과 같다."""
    turns = sorted(s["turns"], key=lambda t: (t["start_s"], t["end_s"])); T, words = [], []
    for ti, t in enumerate(turns):
        ws = simple(t["transcript"]); n = len(ws); a, b = t["start_s"], t["end_s"]; prev = turns[ti - 1] if ti else None
        T.append(dict(spk=t["speaker"], lang=t["lang"], a=a, b=b, n=n, gap=(a - prev["end_s"]) if prev else None, change=None if prev is None else prev["speaker"] != t["speaker"]))
        for k, w in enumerate(ws): words.append(dict(t=a + (b - a) * (k + 1) / n, tm=a + (b - a) * (k + 0.5) / n, ti=ti, pos=k, w=w, spk=t["speaker"]))
    words.sort(key=lambda x: (x["tm"], x["ti"], x["pos"])); busy = collections.defaultdict(list)
    for t in T: busy[t["spk"]].append((t["a"], t["b"]))
    for x in words: x["ov"] = any(a <= x["tm"] <= b for sp, iv in busy.items() if sp != x["spk"] for a, b in iv)
    return T, words


def walign(Rw, H, off, Wd):
    """참조 단어 Rw·가설 단어 H [(시각, 슬롯 언어, 단어, …)] → ([(i, j)] 정렬, 비용). 짝(일치 0·치환 1)은 |가설 시각 − 참조 시각 − off| ≤ Wd 일 때만 — attr_diag.walign 과 같다."""
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


def hyp_words(recs):
    """asr 행 → {세션: [(시각, 슬롯 언어, 정규화 단어)] 시각순(같은 시각은 파일 순서)} — attr_diag 의 ours + 정렬 입력."""
    H = collections.defaultdict(list)
    for x in recs: H[x["session"]].append((x["lang"], x["text"], x["t"]))
    return {k: sorted([(t, lab, u) for lab, w, t in v for u in simple(w)], key=lambda x: x[0]) for k, v in H.items()}


def lang_bound(s) -> bool:
    """화자마다 언어가 달라 슬롯 언어로 화자를 가를 수 있는 세션인가(교차 언어)."""
    return len({v["lang"] for v in s["speakers"].values()}) == len(s["speakers"])


def offset_of(sessions, P, HW):
    """시스템 오프셋(s): 화자 무시 정렬에서 정확히 일치한 단어의 (가설 − 참조) 시각 중앙값과 사분위. 일치가 없으면 0."""
    from vapasr.data.cst_eval import align
    dts = []
    for s in sessions:
        T, Rw = P[s["session_id"]]; hw = HW.get(s["session_id"], [])
        for i, j in align([x["w"] for x in Rw], [x[2] for x in hw]):
            if i is not None and j is not None and Rw[i]["w"] == hw[j][2]: dts.append(hw[j][0] - Rw[i]["t"])
    if not dts: return 0.0, None
    q = sorted(dts); return st.median(dts), (q[len(q) // 4], q[3 * len(q) // 4])


def lexicon(sessions):
    """언어별 어휘(세션 참조 전사) → wlang(단어): 한 언어 어휘에만 있으면 그 언어, 둘 이상 'both', 없으면 'oov'."""
    voc = collections.defaultdict(set)
    for s in sessions:
        for t in s["turns"]: voc[t["lang"]].update(simple(t["transcript"]))
    def wlang(w):
        hit = [lg for lg, v in voc.items() if w in v]
        return hit[0] if len(hit) == 1 else "both" if hit else "oov"
    return wlang


def binof(n):
    return next(nm for e, nm in LEN_BINS if n <= e)


def turn_states(T, Rw, lab, wrong):
    """턴마다 분류(attr_diag.report 의 turnstate): role·cat·sub(partial 유형)·nw·nal·idx(정렬된 참조 단어, 턴 안 순서)·maj·first·last·firstlab·lastlab."""
    out = []
    for ti, t in enumerate(T):
        idx = sorted([i for i, x in enumerate(Rw) if x["ti"] == ti and wrong[i] is not None], key=lambda i: Rw[i]["pos"])
        fl = [wrong[i] for i in idx]; nw = sum(fl); sub = None
        role = "first" if t["change"] is None else ("change" if t["change"] else "cont")
        if not idx: cat = "noalign"
        elif nw == 0: cat = "correct"
        elif nw == len(fl): cat = "whole"
        else:
            cat = "partial"; segs = [(k, len(list(g))) for k, g in itertools.groupby(fl)]; ws = [i for i, (k, _) in enumerate(segs) if k]
            sub = "multi" if len(ws) > 1 else "late" if ws[0] == 0 else "early" if ws[0] == len(segs) - 1 else "middle"
        out.append(dict(role=role, cat=cat, sub=sub, nw=nw, nal=len(fl), idx=idx, maj=(nw * 2 > len(fl)) if fl else None, first=fl[0] if fl else None,
                        last=fl[-1] if fl else None, firstlab=lab[idx[0]] if idx else None, lastlab=lab[idx[-1]] if idx else None))
    return out


def session_counts(s, T, Rw, hw, off, W, wlang, asr=None, eots=None):
    """세션 하나 → 집계용 Counter(문자열 키 — 합치면 전체, 재표집하면 bootstrap). asr: 그 세션의 asr 행(split), eots: [(시각, 슬롯 언어)]."""
    C = collections.Counter(sessions=1); pairs, _ = walign(Rw, hw, off, W)
    lab = [None] * len(Rw); exact = [False] * len(Rw); hword = [None] * len(Rw); ht = [None] * len(Rw)
    for i, j in pairs:
        if i is not None and j is not None: lab[i] = hw[j][1]; exact[i] = hw[j][2] == Rw[i]["w"]; hword[i] = hw[j][2]; ht[i] = hw[j][0]
        elif i is None: C["ins"] += 1
    lmap = {s["speakers"][sp]["lang"]: sp for sp in s["speakers"]}
    wrong = [None if l is None else lmap.get(l) != x["spk"] for l, x in zip(lab, Rw)]
    C.update(ref_words=len(Rw), hyp_words=len(hw), aligned=sum(l is not None for l in lab), exact=sum(exact), wrong=sum(bool(w) for w in wrong),
             wrong_exact=sum(bool(w) and e for w, e in zip(wrong, exact)))
    ts = turn_states(T, Rw, lab, wrong)
    for t, x in zip(T, ts):
        C[f"turn/{x['role']}/{x['cat']}"] += 1; C[f"turnw/{x['role']}/{x['cat']}"] += x["nw"]
        if x["sub"]: C[f"partial/{x['sub']}"] += x["nw"]
        if x["nal"]:
            b = binof(t["n"]); C[f"len/{b}/turns"] += 1; C[f"len/{b}/whole"] += x["cat"] == "whole"; C[f"len/{b}/maj"] += bool(x["maj"])
            C[f"len/{b}/wrong_words"] += x["nw"]; C[f"len/{b}/aligned_words"] += x["nal"]
        for a, b in zip(x["idx"], x["idx"][1:]): C["wpairs"] += 1; C["wsw"] += lab[a] != lab[b]
    for ti in range(1, len(T)):
        t, p, c, pc = T[ti], T[ti - 1], ts[ti], ts[ti - 1]
        if not t["change"]: continue
        if pc["lastlab"] is not None and c["firstlab"] is not None:
            tog = pc["lastlab"] != c["firstlab"]; C["tog_n"] += 1; C["tog_yes"] += tog
            C[f"tog/{'prevWRONG' if pc['last'] else 'prevOK'}>{'curWRONG' if c['first'] else 'curOK'}/{'toggle' if tog else 'no-toggle'}"] += 1
        if c["maj"] is not None:
            k = "le3" if t["n"] <= 3 else "gt3"; C[f"short/{k}/n"] += 1; C[f"short/{k}/maj"] += c["maj"]; C[f"short/{k}/whole"] += c["cat"] == "whole"
            if pc["maj"] is not None and not pc["maj"]: C[f"onset_n/{t['lang']}"] += 1; C[f"onset_miss/{t['lang']}"] += c["maj"]
    C["hchg"] = sum(1 for a, b in zip(hw, hw[1:]) if a[1] != b[1]); C["rchg"] = sum(1 for a, b in zip(T, T[1:]) if a["spk"] != b["spk"])
    for i, x in enumerate(Rw):
        if wrong[i] and not exact[i]:
            C["wrong_sub"] += 1; C["wrong_sub_other"] += wlang(hword[i]) not in (T[x["ti"]]["lang"], "both", "oov")
    hs = collections.Counter(l for _, l, *_ in hw); rs = collections.Counter(T[x["ti"]]["lang"] for x in Rw)
    if hs and rs:
        dom = max(hs, key=hs.get); C["collapse_scored"] += 1; C["collapse"] += hs[dom] / sum(hs.values()) >= COLLAPSE[0] and rs[dom] / sum(rs.values()) <= COLLAPSE[1]
    if ts and ts[0]["nal"]: C["first/n"] += 1; C[f"first/{ts[0]['cat']}"] += 1; C["first/maj"] += bool(ts[0]["maj"])
    elif ts: C["first/noalign"] += 1
    cur, prev_i, prev_w = None, None, False                                                       # 오배정 구간의 시작(attr_diag 의 error onsets)
    for i in (i for i in range(len(Rw)) if wrong[i] is not None):
        w, x = wrong[i], Rw[i]
        if w and not prev_w:
            if prev_i is None: cur = "session-start"
            else:
                px = Rw[prev_i]; cur = "within-turn" if px["ti"] == x["ti"] else ("speaker-change" if px["spk"] != x["spk"] else "same-speaker-next-turn")
            C[f"err_onset/{cur}"] += 1
        if not w and prev_w and prev_i is not None:
            px = Rw[prev_i]; C[f"recover/{'within-turn' if px['ti'] == x['ti'] else 'speaker-change' if px['spk'] != x['spk'] else 'same-speaker-next-turn'}"] += 1
        if w: C[f"err_onset_words/{cur}"] += 1
        prev_i, prev_w = i, w
    if asr is not None:
        c, g, n = attr_split([s], asr); C.update(cp_e=c, agn_e=g, n_split=n)
    if eots is not None:                                                                          # eot_study: 교대 경계의 <EOT>
        tl = collections.defaultdict(collections.Counter)
        for i, x in enumerate(Rw):
            if lab[i] is not None: tl[x["ti"]][lab[i]] += 1
        maj = {ti: c.most_common(1)[0][0] for ti, c in tl.items()}; C["eot_total"] += len(eots)
        for ti in range(1, len(T)):
            t, p = T[ti], T[ti - 1]
            pi, ci = ts[ti - 1]["idx"], ts[ti]["idx"]
            if t["change"] and pi and ci and ts[ti - 1]["lastlab"] == p["lang"]:                       # gate(E+/E− 정의): 앞 턴 마지막이 맞고 새 턴 첫 단어가 틀림/맞음, 두 가설 단어 방출 시각 사이 <EOT>
                evg = [lg for tt, lg in eots if ht[pi[-1]] <= tt <= ht[ci[0]]]; q = f"eot/gate/{'miss' if ts[ti]['firstlab'] != t['lang'] else 'hit'}"
                C[f"{q}/n"] += 1; C[f"{q}/any"] += bool(evg); C[f"{q}/{'prev' if any(lg == p['lang'] for lg in evg) else ('other' if evg else 'none')}"] += 1
            if not t["change"] or ti not in maj: continue
            lo, hi = p["b"] + off - 0.4, t["a"] + off + 0.6; ev = [lg for tt, lg in eots if lo <= tt <= hi]; prev_slot = maj.get(ti - 1)
            k = "prev" if any(lg == prev_slot for lg in ev) else ("other" if ev else "none")
            keys = [f"eot/majority/{'miss' if maj[ti] != t['lang'] else 'hit'}"]
            if ts[ti - 1]["lastlab"] is not None and ts[ti]["firstlab"] is not None: keys.append(f"eot/toggle/{'hit' if ts[ti - 1]['lastlab'] != ts[ti]['firstlab'] else 'miss'}")
            for q in keys: C[f"{q}/n"] += 1; C[f"{q}/any"] += bool(ev); C[f"{q}/{k}"] += 1
    return C


def pct(a, b, nd=2):
    return round(100 * a / b, nd) if b else None


def scored_change(C):
    return sum(C[f"turn/change/{c}"] for c in CATS if c != "noalign")


BOOT = {"toggle_recall": lambda C: pct(C["tog_yes"], C["tog_n"], 3), "change_ratio": lambda C: round(C["hchg"] / C["rchg"], 4) if C["rchg"] else None,
        "whole_wrong_change": lambda C: pct(C["turn/change/whole"], scored_change(C), 3), "within_turn_switch": lambda C: pct(C["wsw"], C["wpairs"], 3),
        "misattributed": lambda C: pct(C["wrong"], C["ref_words"], 3), "early_words": lambda C: pct(C["partial/early"], C["ref_words"], 3),
        "cp": lambda C: pct(C["cp_e"], C["n_split"], 3), "agn": lambda C: pct(C["agn_e"], C["n_split"], 3), "attr": lambda C: pct(C["cp_e"] - C["agn_e"], C["n_split"], 3)}


BOOT_KEYS = ("tog_yes", "tog_n", "hchg", "rchg", "turn/change/correct", "turn/change/whole", "turn/change/partial", "wsw", "wpairs", "wrong", "ref_words", "partial/early",
             "cp_e", "agn_e", "n_split")


def bootstrap(per, against=None, B=1000, seed=0):
    """세션 재표집 B 번(random.Random(seed), 세션 index 를 n 번 복원 추출) → 지표마다 값과 95 % CI(2.5·97.5 백분위).
    against(같은 세션 순서의 비교 시스템 Counter 목록)가 있으면 같은 재표집에서 차(이 − 비교)의 CI 도."""
    rng = random.Random(seed); n = len(per); vals = collections.defaultdict(list); diffs = collections.defaultdict(list)
    def q(v): v = sorted(v); return [round(v[int(round(0.025 * (len(v) - 1)))], 4), round(v[int(round(0.975 * (len(v) - 1)))], 4)] if v else None
    vec = lambda P: [[C[k] for k in BOOT_KEYS] for C in P]; pa = vec(per); pz = vec(against) if against is not None else None
    tot = lambda V, idx: collections.Counter(dict(zip(BOOT_KEYS, (sum(c) for c in zip(*(V[i] for i in idx))))))
    for _ in range(B if n else 0):
        idx = [rng.randrange(n) for _ in range(n)]; A = tot(pa, idx); Z = tot(pz, idx) if pz is not None else None
        for k, f in BOOT.items():
            a = f(A)
            if a is not None: vals[k].append(a)
            if Z is not None:
                z = f(Z)
                if a is not None and z is not None: diffs[k].append(a - z)
    tot = sum(per, collections.Counter()); out = dict(n=B, seed=seed, sessions=n, metrics={k: dict(value=f(tot), ci95=q(vals[k])) for k, f in BOOT.items()})
    if against is not None:
        zt = sum(against, collections.Counter())
        out["paired_diff"] = {k: dict(value=(None if f(tot) is None or f(zt) is None else round(f(tot) - f(zt), 4)), ci95=q(diffs[k])) for k, f in BOOT.items()}
    return out


def summarize(C, eot=False):
    """합친 Counter → 지표 dict(집계만)."""
    langs = sorted({k.split("/", 1)[1] for k in C if k.startswith("onset_n/")})
    res = dict(sessions=C["sessions"], ref_words=C["ref_words"], aligned_pct=pct(C["aligned"], C["ref_words"]), exact_pct=pct(C["exact"], C["ref_words"]),
               hyp_words=C["hyp_words"], insertions=C["ins"],
               misattributed=dict(words=C["wrong"], pct_ref=pct(C["wrong"], C["ref_words"]), pct_aligned=pct(C["wrong"], C["aligned"]), exact_share_pct=pct(C["wrong_exact"], C["wrong"])),
               toggle_recall=dict(pct=pct(C["tog_yes"], C["tog_n"]), n=C["tog_n"], toggled=C["tog_yes"],
                                  transitions={k.split("/", 1)[1]: v for k, v in sorted(C.items()) if k.startswith("tog/")}),
               changes=dict(hyp=C["hchg"], ref=C["rchg"], ratio=round(C["hchg"] / C["rchg"], 4) if C["rchg"] else None),
               turns={r: {c: C[f"turn/{r}/{c}"] for c in CATS} for r in ROLES}, turn_wrong_words={r: {c: C[f"turnw/{r}/{c}"] for c in CATS if c != "noalign"} for r in ROLES},
               whole_wrong_change=dict(turns=C["turn/change/whole"], scored=scored_change(C), pct=pct(C["turn/change/whole"], scored_change(C))),
               within_turn_switch=dict(switches=C["wsw"], pairs=C["wpairs"], pct=pct(C["wsw"], C["wpairs"])),
               partial_words={k: C[f"partial/{k}"] for k in SUBS},
               onset_miss={lg: dict(n=C[f"onset_n/{lg}"], miss=C[f"onset_miss/{lg}"], pct=pct(C[f"onset_miss/{lg}"], C[f"onset_n/{lg}"])) for lg in langs},
               short_turns={f"change_{k}": dict(n=C[f"short/{k}/n"], whole=C[f"short/{k}/whole"], maj=C[f"short/{k}/maj"], whole_pct=pct(C[f"short/{k}/whole"], C[f"short/{k}/n"]),
                                                maj_pct=pct(C[f"short/{k}/maj"], C[f"short/{k}/n"])) for k in ("le3", "gt3")},
               turn_len={b: dict(turns=C[f"len/{b}/turns"], whole_pct=pct(C[f"len/{b}/whole"], C[f"len/{b}/turns"]), maj_pct=pct(C[f"len/{b}/maj"], C[f"len/{b}/turns"]),
                                 word_wrong_pct=pct(C[f"len/{b}/wrong_words"], C[f"len/{b}/aligned_words"])) for _, b in LEN_BINS},
               wrong_slot_lang=dict(wrong_sub=C["wrong_sub"], other_lang=C["wrong_sub_other"], sub_pct_ref=pct(C["wrong_sub"], C["ref_words"]),
                                    other_lang_pct_ref=pct(C["wrong_sub_other"], C["ref_words"])),
               slot_collapse=dict(sessions=C["collapse"], scored=C["collapse_scored"], rule=dict(hyp_share_ge=COLLAPSE[0], ref_share_le=COLLAPSE[1])),
               first_turn=dict(n=C["first/n"], correct=C["first/correct"], whole=C["first/whole"], partial=C["first/partial"], noalign=C["first/noalign"], maj_wrong=C["first/maj"],
                               acc_pct=pct(C["first/n"] - C["first/maj"], C["first/n"])),
               error_onsets=dict(onsets={k.split("/", 1)[1]: v for k, v in sorted(C.items()) if k.startswith("err_onset/")},
                                 words_pct={k.split("/", 1)[1]: pct(v, C["wrong"]) for k, v in sorted(C.items()) if k.startswith("err_onset_words/")},
                                 recoveries={k.split("/", 1)[1]: v for k, v in sorted(C.items()) if k.startswith("recover/")}),
               split=dict(cp=pct(C["cp_e"], C["n_split"]), agn=pct(C["agn_e"], C["n_split"]), attr=pct(C["cp_e"] - C["agn_e"], C["n_split"]), ref_words=C["n_split"]))
    if eot:
        res["eot"] = dict(emitted=C["eot_total"])
        for by in ("majority", "toggle", "gate"):
            d = {h: {k: C[f"eot/{by}/{h}/{k}"] for k in ("n", "any", "prev", "other", "none")} for h in ("hit", "miss")}
            res["eot"][by] = dict(d, f_hit=pct(d["hit"]["prev"], d["hit"]["n"]), f_miss=pct(d["miss"]["prev"], d["miss"]["n"]))
    return res


def per_session(sessions, asr, eot=None, W=5.0):
    """→ (언어 고정 세션 순서의 Counter 목록, 오프셋 (중앙값, 사분위), 같은 언어로 건너뛴 세션 수)."""
    S = [s for s in sessions if lang_bound(s)]; P = {s["session_id"]: prep(s) for s in S}; HW = hyp_words(asr); wlang = lexicon(sessions)
    by = collections.defaultdict(list)
    for x in asr: by[x["session"]].append(x)
    E = None
    if eot is not None:
        E = collections.defaultdict(list)
        for x in eot: E[x["session"]].append((x["t"], x["lang"]))
    off, iqr = offset_of(S, P, HW)
    per = [session_counts(s, *P[s["session_id"]], HW.get(s["session_id"], []), off, W, wlang, by.get(s["session_id"], []), None if E is None else E.get(s["session_id"], []))
           for s in S]
    return per, (off, iqr), len(sessions) - len(S)


def turn_metrics(sessions, asr, eot=None, against=None, W=5.0, B=1000, seed=0):
    """cst_stage4_taxi score --turn-metrics 와 CLI 의 본체 → 지표 dict(집계만)."""
    per, (off, iqr), skipped = per_session(sessions, asr, eot, W)
    res = dict(summarize(sum(per, collections.Counter()), eot is not None), skipped_same_lang=skipped, window_s=W, offset_s=round(off, 3), offset_iqr=[round(x, 3) for x in iqr] if iqr else None)
    zp = per_session(sessions, against, None, W)[0] if against is not None else None
    res["bootstrap"] = bootstrap(per, zp, B, seed)
    return res


def brief(r) -> str:
    """지표 dict → 한 줄 요약."""
    om = " · ".join(f"{lg} {v['pct']} %(n {v['n']})" for lg, v in r["onset_miss"].items())
    s = (f"toggle recall {r['toggle_recall']['pct']} %(n {r['toggle_recall']['n']}) · 가설/참조 교대 {r['changes']['hyp']}/{r['changes']['ref']}({r['changes']['ratio']}) · "
         f"통째 틀린 교대 턴 {r['whole_wrong_change']['turns']}/{r['whole_wrong_change']['scored']} · 턴 안 헛교대 {r['within_turn_switch']['pct']} % · 새 화자 언어별 놓침 {om} · "
         f"첫 턴 {r['first_turn']['acc_pct']} % · attr {r['split']['attr']}")
    if "eot" in r: s += f" · 앞 슬롯 <EOT> f_hit {r['eot']['majority']['f_hit']} % f_miss {r['eot']['majority']['f_miss']} %"
    return s


# ── 온셋 분류(taxi_v3_probe --classify-onsets)
def streams_at(emitted, spk_ids, trans_id, eot_id):
    """원출력 [(k, tid)] → 슬롯별 전사·번역 [(방출 시각 s, tid, 원출력 위치)] — parse_emitted4 와 같은 상태기계(drop 없음)."""
    spk = list(spk_ids); src = [[] for _ in spk]; tr = [[] for _ in spk]; cur, kind = 0, 0
    for p, (k, t) in enumerate(emitted):
        t = int(t)
        if t in spk: cur, kind = spk.index(t), 0; continue
        if t == trans_id: kind = 1; continue
        if t == eot_id: continue
        (src if kind == 0 else tr)[cur].append(((int(k) + 1) * CHUNK_S, t, p))
    return src, tr


def words_at(tok, toks):
    """[(시각, tid, 위치)] → [(단어, 시각, 첫 토큰 위치, 끝 토큰 위치)] — 단어·시각은 cst_stage3_eval.words_of 와 같다(바이트 공백 분리, 시각 = 단어 끝 바이트 토큰)."""
    import bisect
    from vapasr.data.ast_eval import token_bytes
    if not toks: return []
    bs = token_bytes(tok, [t for _, t, _ in toks]); raw = b"".join(bs); cum, c = [], 0
    for b in bs: c += len(b); cum.append(c)
    out = []
    for m in re.finditer(rb"\S+", raw):
        j = min(bisect.bisect_left(cum, m.end()), len(toks) - 1); i = min(bisect.bisect_right(cum, m.start()), j)
        out.append((m.group().decode("utf-8", "replace"), toks[j][0], toks[i][2], toks[j][2]))
    return out


def emitted_words(tok, windows, spk_ids, trans_id, eot_id, langs):
    """세션의 창별 원출력 [(창 시작 s, emitted)] → (가설 단어 [(시각, 슬롯 언어, 정규화 단어, (창, 첫 위치, 끝 위치, 슬롯))] 시각순 — cst_stage4_taxi run 이 asr 에
    쓰는 단어·시각(창 시작 + , 소수 3 자리)·순서(창마다 슬롯 0, 1)와 같다, 번역 단어 {(창, 슬롯): [(정규화 단어, 첫 위치, 끝 위치)]})."""
    H, TRW = [], {}
    for wi, (w0, em) in enumerate(windows):
        src, tr = streams_at(em, spk_ids, trans_id, eot_id)
        for k in range(len(langs)):
            H += [(round(w0 + t, 3), langs[k], wd, (wi, a, b, k)) for wd, t, a, b in words_at(tok, src[k])]
            TRW[(wi, k)] = [(u, a, b) for wd, _, a, b in words_at(tok, tr[k]) for u in simple(wd)]
    return sorted([(t, lab, u, ref) for t, lab, wd, ref in H for u in simple(wd)], key=lambda x: x[0]), TRW


def absorbed(lost, trans_words) -> bool:
    """새 턴 앞머리 참조 단어(lost — 첫 정렬 단어 앞)의 절반 이상(최소 1)이 그 사이 이전 슬롯 번역 단어와 같은 순서로 일치하는가."""
    if not lost or not trans_words: return False
    from vapasr.data.cst_eval import align
    hit = sum(1 for i, j in align(list(lost), list(trans_words)) if i is not None and j is not None and lost[i] == trans_words[j])
    return hit >= max(1, (len(lost) + 1) // 2)


def onset_class(seg, ka, kb, spk_ids, trans_id, eot_id, lost=(), trans_words=()):
    """교대 하나의 온셋 분류. seg = 앞 턴 마지막 정렬 가설 단어(슬롯 ka 전사)의 끝 토큰과 새 턴 첫 정렬 가설 단어(슬롯 kb 전사)의 첫 토큰 사이 원출력 tid 들.
    parse_emitted4 상태기계로 (cur = ka, kind = 전사)에서 읽는다. 이전 슬롯 <EOT> = cur 가 ka 일 때 나온 <EOT>.
      OK        kb ≠ ka — 다른 슬롯으로 바꿨다
      ABSORBED  같은 슬롯이고, 새 턴 앞머리 참조 단어(lost)가 그 사이 이전 슬롯 번역 단어로 나왔다 — 태그 없이 번역 kind 로 이어 씀(absorbed)
      SILENT    같은 슬롯이고 사이에 태그가 없다 — 전사 kind 로 이어 씀(번역 뒤 전사는 태그로만 열리므로 사이에 <TRANS> 도 없다)
      REASSERT  마지막 태그가 같은 슬롯이고 그 앞에 이전 슬롯 <EOT> 가 있다 — <EOT> 뒤 같은 슬롯을 다시 고름
      RESUME    마지막 태그가 같은 슬롯이고 그 앞에 이전 슬롯 <EOT> 가 없다 — 보통 <TRANS> 구간 뒤 같은 슬롯 재개
    → (클래스, 이전 슬롯 <EOT> 있음, 결정 태그 앞 <TRANS> 있음 — RESUME·REASSERT 만 의미)."""
    spk = list(spk_ids); cur, eot, trans, last = ka, False, False, None
    for t in seg:
        t = int(t)
        if t in spk: last = (spk.index(t), eot, trans); cur = spk.index(t); trans = False; continue
        if t == trans_id: trans = True; continue
        if t == eot_id and cur == ka: eot = True
    if kb != ka: return "OK", eot, False
    if absorbed(lost, trans_words): return "ABSORBED", eot, False
    if last is None: return "SILENT", eot, False
    return ("REASSERT" if last[1] else "RESUME"), eot, last[2]


def onset_classes(sessions, runs, tok, spk_ids, trans_id, eot_id, W=5.0):
    """runs = {세션: [(창 시작 s, emitted [(k, tid)])] 창 순서} → 교대마다 onset_class 집계(세션 id·글자 없음).
    정렬은 turn_metrics 와 같다(emitted_words 의 가설 단어 = 그 run 의 asr 행). 두 정렬 단어가 다른 창이면 cross_window, 한쪽 턴에 정렬 단어가 없으면 noalign 으로 센다.
    교대마다 앞 턴 마지막 정렬 단어가 맞는 슬롯이었나(prev_ok)로도 나눈다 — 놓친 온셋 판단은 classes_prev_ok 로 한다(prev_wrong 은 다시 맞추기·역전 교대).
    ABSORBED 의 번역 단어는 새 턴 시작 + 오프셋 − 0.3 s 이후에 방출된 것만 본다(앞 턴 자기 번역과 섞이지 않게)."""
    from cst_stage4_taxi import slots_of
    S = [s for s in sessions if lang_bound(s)]; P = {s["session_id"]: prep(s) for s in S}; HW, TRW = {}, {}
    for s in S: HW[s["session_id"]], TRW[s["session_id"]] = emitted_words(tok, runs.get(s["session_id"], []), spk_ids, trans_id, eot_id, slots_of(s)[1])
    off, _ = offset_of(S, P, HW); C = collections.Counter(); D = collections.Counter(); G = collections.Counter()
    for s in S:
        sid = s["session_id"]; T, Rw = P[sid]; hw = HW[sid]; pairs, _ = walign(Rw, hw, off, W); hj = [None] * len(Rw); al = collections.defaultdict(list)
        for i, j in pairs:
            if i is not None and j is not None: hj[i] = j
        for i in sorted((i for i in range(len(Rw)) if hj[i] is not None), key=lambda i: Rw[i]["pos"]): al[Rw[i]["ti"]].append(i)
        for ti in range(1, len(T)):
            if not T[ti]["change"]: continue
            C["changes"] += 1
            if not al.get(ti - 1) or not al.get(ti): C["noalign"] += 1; continue
            ia, ib = al[ti - 1][-1], al[ti][0]; wa, _, la, ka = hw[hj[ia]][3]; wb, fb, _, kb = hw[hj[ib]][3]
            if wa != wb: C["cross_window"] += 1; continue
            w0, em = runs[sid][wb]; seg = [t for _, t in em[la + 1: fb]] if fb > la else []
            lost = [x["w"] for x in sorted((x for x in Rw if x["ti"] == ti and x["pos"] < Rw[ib]["pos"]), key=lambda x: x["pos"])]
            t_new = T[ti]["a"] + off - 0.3
            trw = [u for u, p0, p1 in TRW[sid].get((wb, ka), []) if p0 > la and p1 < fb and w0 + (em[p0][0] + 1) * 0.08 >= t_new]
            cls, eot, trans = onset_class(seg, ka, kb, spk_ids, trans_id, eot_id, lost, trw); e = "eot" if eot else "no_eot"
            pv = "prev_ok" if hw[hj[ia]][1] == T[ti - 1]["lang"] else "prev_wrong"
            C[f"{cls}/{e}"] += 1; D[(T[ti]["lang"], cls, e)] += 1; G[(pv, cls, e)] += 1
            if cls == "RESUME" and not trans: C["resume_without_trans"] += 1
    cell = lambda get: {c: dict(eot=get(c, "eot"), no_eot=get(c, "no_eot"), all=get(c, "eot") + get(c, "no_eot")) for c in ONSET}
    return dict(sessions=len(S), skipped_same_lang=len(sessions) - len(S), changes=C["changes"], classified=sum(C[f"{c}/{e}"] for c in ONSET for e in ("eot", "no_eot")),
                noalign=C["noalign"], cross_window=C["cross_window"], resume_without_trans=C["resume_without_trans"], window_s=W, offset_s=round(off, 3),
                classes=cell(lambda c, e: C[f"{c}/{e}"]), by_new_lang={lg: cell(lambda c, e, lg=lg: D[(lg, c, e)]) for lg in sorted({k[0] for k in D})},
                classes_prev_ok=cell(lambda c, e: G[("prev_ok", c, e)]), classes_prev_wrong=cell(lambda c, e: G[("prev_wrong", c, e)]))


def main():
    from cstbench.evaluate import load_jsonl
    ap = argparse.ArgumentParser(); ap.add_argument("--sessions", required=True); ap.add_argument("--asr", required=True); ap.add_argument("--eot", default=None)
    ap.add_argument("--against", default=None, help="비교 시스템의 asr 파일, 또는 결과 폴더(--asr 와 같은 파일 이름) — 세션 paired bootstrap")
    ap.add_argument("--window", type=float, default=5.0, help="정렬 시각 허용 W(s, 오프셋 뺀 뒤 ±W)"); ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", default=None, help="지표 json(없으면 표준 출력)"); a = ap.parse_args()
    zf = os.path.join(a.against, os.path.basename(a.asr)) if a.against and os.path.isdir(a.against) else a.against
    res = turn_metrics(load_jsonl(a.sessions), load_jsonl(a.asr), load_jsonl(a.eot) if a.eot else None, load_jsonl(zf) if zf else None, a.window, a.boot, a.seed)
    res.update(sessions_file=a.sessions, asr=a.asr, eot_file=a.eot, against_file=zf)
    print(brief(res), flush=True)
    if "paired_diff" in res["bootstrap"]:
        print("paired(이 − 비교) " + " · ".join(f"{k} {v['value']} {v['ci95']}" for k, v in res["bootstrap"]["paired_diff"].items()), flush=True)
    if a.out: json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    else: print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

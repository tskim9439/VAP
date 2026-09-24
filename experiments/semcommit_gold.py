#!/usr/bin/env python3
"""SEM_END 골드셋 — 전수 주석(모든 단어 경계를 COMMIT / AMBIG / NO 로 판정)의 패킷·병합·판정·채점.

주석 절차(gold v1): 패킷(번호 붙은 단어열 + 참조 전사) → 독립 주석자 2 명(관점이 다른 두 lens) → 불일치 경계만 판정자(adjudicator)가 결정 → 골드.
  packet       : words.jsonl → 주석 패킷 텍스트(+ id 목록)
  merge        : 주석 2 개(+ 판정) → gold.jsonl + 일치도 통계. 판정이 없거나 모자라면 --adj-packet 에 불일치 경계 목록을 쓴다.
  score-labels : 교사 라벨(labels.jsonl, A/B/N) vs 골드 — 후보 재현·A 정밀도/재현율·N 정밀도(commit_metrics.gold_label_counts)
  score-eval   : semcommit_eval 스트림 jsonl(모델 방출) vs 골드 — 설정별 P/R/F1·PCR_gold·지연(commit_metrics.gold_commit_counts)
골드 행: {id, set, lang, n_words, commit:[i], ambig:[i], how:{i: 'agree'|'adjudicated'|'unresolved'}}. 주석 형식은 GUIDELINE.md 출력 형식({id: {commit, ambig, why}}).

  python experiments/semcommit_gold.py packet --words words-ks-long.jsonl --domain "Korean|spontaneous conversation" --name ks-long --out packets/
  python experiments/semcommit_gold.py merge --words words-ks-long.jsonl --ann ann/ks-long.rules.json ann/ks-long.consumer.json --adj adj/ks-long.json --out gold/gold-ks-long.jsonl
  python experiments/semcommit_gold.py score-labels --words words-ks-long.jsonl --labels labels-ks-long.jsonl --gold gold/gold-ks-long.jsonl
  python experiments/semcommit_gold.py score-eval --words words-ks-long.jsonl --gold gold/gold-ks-long.jsonl --streams eval/r1-d4-ks-long.streams.jsonl
"""
import argparse, collections, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.hf.commit_metrics import (SEM_END_ID, finalize_gold_commits, finalize_gold_labels, gold_commit_counts, gold_label_counts, merge as merge_counts, ref_chunk)

LABELS = ("NO", "AMBIG", "COMMIT")


def read_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def words_of(paths): return {r["id"]: r for p in paths for r in read_jsonl(p)}


def render(r: dict, desc: str) -> str:
    """주석·판정 패킷의 스트림 블록: 머리줄 + 참조 전사(EN 구두점 전사 / KO Kspon 원 전사) + 번호 붙은 단어(표지 태그)."""
    lang = r["lang"]; lines = [f"### STREAM {r['id']} | {lang} | {desc} | {r['duration_s']:.1f} s | {len(r['words'])} words"]
    if lang == "English": lines.append("REFERENCE PUNCTUATED TRANSCRIPT (human-written, for parsing only): " + (r.get("pnc_text") or "(not available)"))
    else: lines.append("ORIGINAL TRANSCRIPT with Kspon markers: " + " / ".join(s.get("raw_text") or "" for s in r["segments"]))
    ws = [f"{w['i']}:{w['text']}" + (f"[{','.join(t)}]" if (t := [x for x in (w.get('tags') or []) if x in ('filler', 'rep', 'unclear')]) else "") for w in r["words"]]
    return "\n".join(lines + ["WORDS: " + " ".join(ws)])


def ann_labels(ann: dict, sid: str, n: int) -> dict:
    """주석 파일의 한 스트림 → {i: 'COMMIT'|'AMBIG'} (범위 밖·겹침은 ValueError)."""
    a = ann.get(sid)
    if a is None: raise ValueError(f"주석에 스트림 없음: {sid}")
    c, m = {int(i) for i in a.get("commit", [])}, {int(i) for i in a.get("ambig", [])}
    if c & m or any(not 0 <= i < n for i in c | m): raise ValueError(f"{sid}: 겹침 {sorted(c & m)} 또는 범위 밖(n={n})")
    return {**{i: "AMBIG" for i in m}, **{i: "COMMIT" for i in c}}


def kappa(pairs) -> float:
    """Cohen κ (라벨 쌍 목록)."""
    n = len(pairs)
    if not n: return float("nan")
    po = sum(a == b for a, b in pairs) / n; ca = collections.Counter(a for a, _ in pairs); cb = collections.Counter(b for _, b in pairs)
    pe = sum(ca[k] * cb[k] for k in LABELS) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_packet(a):
    W = [r for p in a.words for r in read_jsonl(p)]; os.makedirs(a.out, exist_ok=True)
    txt = f"PACKET {a.name} — {len(W)} streams. Annotate every stream per GUIDELINE.md.\n\n" + "\n\n".join(render(r, a.domain) for r in W) + "\n"
    open(os.path.join(a.out, f"{a.name}.txt"), "w", encoding="utf-8").write(txt); json.dump([r["id"] for r in W], open(os.path.join(a.out, f"{a.name}.ids.json"), "w"))
    print(f"{a.name}: {len(W)} streams, {sum(len(r['words']) for r in W)} words → {a.out}")


def cmd_merge(a):
    W = words_of(a.words); A = [json.load(open(p)) for p in a.ann]; names = [Path(p).stem.split(".")[-1] for p in a.ann]
    adj = json.load(open(a.adj)) if a.adj else {}
    gold, pairs, dis_blocks, st = [], [], [], collections.Counter()
    for sid, r in W.items():
        n = len(r["words"]); la, lb = (ann_labels(x, sid, n) for x in A); d = (adj.get(sid) or {})
        commit, ambig, how, disputes = [], [], {}, []
        for i in range(n):
            x, y = la.get(i, "NO"), lb.get(i, "NO"); pairs.append((x, y))
            if x == y: lab = x; h = "agree"
            elif str(i) in d:
                lab = str(d[str(i)]).upper(); h = "adjudicated"; assert lab in LABELS, f"{sid}:{i} 판정 라벨 {lab}"
            else: lab = "AMBIG"; h = "unresolved"; disputes.append(i)                    # 판정 전에는 보수적으로 AMBIG
            if lab == "COMMIT": commit.append(i)
            elif lab == "AMBIG": ambig.append(i)
            if lab != "NO" or h != "agree": how[str(i)] = h
            st[f"{h}_{lab}"] += 1
        gold.append(dict(id=sid, set=r.get("set"), lang=r["lang"], n_words=n, commit=commit, ambig=ambig, how=how))
        if disputes:
            why = [A[k].get(sid, {}).get("why", {}) for k in range(2)]
            items = [f"  i={i} ({r['words'][i]['text']}) | {names[0]}: {la.get(i, 'NO')}" + (f" — {why[0].get(str(i))}" if why[0].get(str(i)) else "")
                     + f" | {names[1]}: {lb.get(i, 'NO')}" + (f" — {why[1].get(str(i))}" if why[1].get(str(i)) else "") for i in disputes]
            dis_blocks.append(render(r, a.domain) + "\nDISPUTED BOUNDARIES (decide COMMIT / AMBIG / NO for each):\n" + "\n".join(items))
    ca = {i for i, (x, _) in enumerate(pairs) if x == "COMMIT"}; cb = {i for i, (_, y) in enumerate(pairs) if y == "COMMIT"}
    f1 = 2 * len(ca & cb) / (len(ca) + len(cb)) if ca or cb else float("nan")
    stats = dict(streams=len(gold), boundaries=len(pairs), annotators=names, kappa_3class=round(kappa(pairs), 4), commit_f1_between=round(f1, 4),
                 annotator_counts={nm: dict(collections.Counter(p[k] for p in pairs)) for k, nm in enumerate(names)}, decisions=dict(st),
                 gold_commit=sum(len(g["commit"]) for g in gold), gold_ambig=sum(len(g["ambig"]) for g in gold), unresolved=sum(1 for g in gold for h in g["how"].values() if h == "unresolved"))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f: f.writelines(json.dumps(g, ensure_ascii=False) + "\n" for g in gold)
    json.dump(stats, open(a.stats or a.out.replace(".jsonl", ".stats.json"), "w"), indent=1, ensure_ascii=False)
    if dis_blocks and a.adj_packet:
        open(a.adj_packet, "w", encoding="utf-8").write(f"ADJUDICATION — {len(dis_blocks)} streams, {stats['unresolved']} disputed boundaries.\n\n" + "\n\n".join(dis_blocks) + "\n")
    print(json.dumps(stats, ensure_ascii=False))


def _load_gold(p): return {g["id"]: g for g in read_jsonl(p)}


def cmd_score_labels(a):
    W = words_of(a.words); G = _load_gold(a.gold); L = {r["id"]: r for r in read_jsonl(a.labels)}; tot, miss = None, 0
    for sid, g in G.items():
        if sid not in W: continue
        if sid not in L: miss += 1                                                          # 라벨 없는 스트림(Stage A 실패) = 후보 0 개
        tot = merge_counts(tot, gold_label_counts(g, (L.get(sid) or {}).get("candidates", [])))
    res = dict(finalize_gold_labels(tot), streams=len(G), streams_without_labels=miss)
    print(json.dumps(res, ensure_ascii=False)); return res


def cmd_score_eval(a):
    W = words_of(a.words); G = _load_gold(a.gold); out = {}
    for sp in a.streams:
        agg = {}
        for r in read_jsonl(sp):
            sid = r["id"]
            if sid not in G or sid not in W: continue
            ws = sorted(W[sid]["words"], key=lambda w: int(w["i"])); K = int(r["K"]); d = int(r["delta"])
            sem = (r.get("event_ids") or [SEM_END_ID])[0]
            ev = [e for e in r["hyp"]["events"] if int(e["id"]) == int(sem)]
            c = gold_commit_counts(ws, G[sid], r["hyp"]["words"], ev, [ref_chunk(float(w["end_time"]), d, K) for w in ws], K)
            name = r["config"]["name"] if isinstance(r["config"], dict) else str(r["config"]); agg[name] = merge_counts(agg.get(name), c)
        out[Path(sp).name.replace(".streams.jsonl", "")] = {k: finalize_gold_commits(v) for k, v in sorted(agg.items())}
    fm = lambda x: "    -" if x is None else f"{x:5.2f}"
    for f, cfgs in out.items():
        print(f"== {f}")
        for k, v in cfgs.items():
            print(f"  {k:11s} commits {v['n_hyp']:4d}  hit {v['hit']:4d}  ambig {v['at_ambig']:3d}  no {v['at_no']:3d}  P {fm(v['P'])}  R {fm(v['R'])}  F1 {fm(v['F1'])}  "
                  f"PCR {fm(v['PCR_gold'])}  lat50 {fm((v['latency_s'] or {}).get('p50'))}  (gold commit {v['gold_commit']})")
    if a.out: json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("packet"); p.add_argument("--words", nargs="+", required=True); p.add_argument("--domain", required=True); p.add_argument("--name", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("merge"); p.add_argument("--words", nargs="+", required=True); p.add_argument("--ann", nargs=2, required=True); p.add_argument("--adj", default=None)
    p.add_argument("--domain", default="speech"); p.add_argument("--out", required=True); p.add_argument("--stats", default=None); p.add_argument("--adj-packet", default=None)
    p = sub.add_parser("score-labels"); p.add_argument("--words", nargs="+", required=True); p.add_argument("--labels", required=True); p.add_argument("--gold", required=True)
    p = sub.add_parser("score-eval"); p.add_argument("--words", nargs="+", required=True); p.add_argument("--gold", required=True); p.add_argument("--streams", nargs="+", required=True)
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    return dict(packet=cmd_packet, merge=cmd_merge, **{"score-labels": cmd_score_labels, "score-eval": cmd_score_eval})[a.cmd](a)


if __name__ == "__main__":
    main()

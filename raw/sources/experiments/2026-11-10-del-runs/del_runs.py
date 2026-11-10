#!/usr/bin/env python
"""삭제가 삭제를 부르는가(2026-11-10, 사용자 "가설 4 진단은 진행해도 돼" — wiki output-cpwer-hypotheses-20261110 가설 4).
gap_diag.py --dump 의 참조 단어 줄(sys·corpus·cid·s·utt·t·ov·cp·orc·w)을 읽어 시스템마다 삭제의 이어짐을 잰다. 디코드 없음, 읽기만.
  python del_runs.py <dump.jsonl> [--gap 2.0] [--json out]
상태는 orc(받아쓰기 — 화자 귀속과 무관): ok · sub · del. 참조 화자의 단어를 시각순으로 놓고 본다.
  A 조건부 삭제율: 같은 발화 안 바로 앞 단어가 삭제였을 때 / 맞았을 때 이 단어의 삭제율(전체, 혼자→혼자, 겹침 직후 혼자 말).
  B 삭제 묶음 길이: 같은 화자에서 gap 초 넘게 끊기지 않고 이어진 삭제 단어 묶음 — 길이별 삭제 단어 몫.
  C 겹침에서 번진 삭제: 겹친 단어를 포함한 삭제 묶음에서 마지막 겹친 단어 뒤로 이어진 혼자 말 삭제 수.
  D 통째로 빠진 발화: 발화 길이별로 모든 단어가 삭제인 발화 몫과, 그 발화가 전체 삭제에서 차지하는 몫.
"""
import argparse, collections, json


def pct(a, b): return round(100.0 * a / b, 2) if b else None


def analyse(words, gap):
    seq = collections.defaultdict(list)
    for q in words: seq[(q["cid"], q["s"])].append(q)
    T = collections.Counter(); runs = []; utts = collections.defaultdict(list)
    n = len(words); n_del = sum(q["orc"] == "del" for q in words)
    solo = [q for q in words if q["ov"] <= 1]; ovl = [q for q in words if q["ov"] >= 2]
    for key, qs in seq.items():
        qs.sort(key=lambda q: q["t"]); cur = None
        for i, q in enumerate(qs):
            utts[(key, q["utt"])].append(q)
            p = qs[i - 1] if i and qs[i - 1]["utt"] == q["utt"] else None
            if p is not None:
                d = q["orc"] == "del"
                for tag, ok in (("all", True), ("solo", p["ov"] <= 1 and q["ov"] <= 1), ("after_ovl", p["ov"] >= 2 and q["ov"] <= 1), ("in_ovl", q["ov"] >= 2)):
                    if ok: T[f"{tag}|{p['orc']}|n"] += 1; T[f"{tag}|{p['orc']}|del"] += d
            if q["orc"] == "del":
                if cur and i and qs[i - 1]["orc"] == "del" and q["t"] - qs[i - 1]["t"] <= gap: cur.append(q)
                else: cur = [q]; runs.append(cur)
            else: cur = None
    A = {}
    for tag in ("all", "solo", "after_ovl", "in_ovl"):
        A[tag] = {f"prev_{s}": dict(n=T[f"{tag}|{s}|n"], del_pct=pct(T[f"{tag}|{s}|del"], T[f"{tag}|{s}|n"])) for s in ("del", "ok", "sub")}
    bins = (("1", 1, 1), ("2-4", 2, 4), ("5-9", 5, 9), ("10+", 10, 10 ** 9))
    B = {b: dict(runs=sum(lo <= len(r) <= hi for r in runs), word_share=pct(sum(len(r) for r in runs if lo <= len(r) <= hi), n_del)) for b, lo, hi in bins}
    tails = []
    for r in runs:
        ix = [k for k, q in enumerate(r) if q["ov"] >= 2]
        if ix: tails.append(sum(q["ov"] <= 1 for q in r[ix[-1] + 1:]))
    C = dict(runs_with_overlap=len(tails), solo_tail_words=sum(tails), tail_share_of_solo_del=pct(sum(tails), sum(q["orc"] == "del" for q in solo)),
             runs_tail_ge3=sum(t >= 3 for t in tails))
    D = {}; whole_del = 0
    for b, lo, hi in (("1-2", 1, 2), ("3-5", 3, 5), ("6-15", 6, 15), ("16+", 16, 10 ** 9)):
        us = [u for u in utts.values() if lo <= len(u) <= hi]; gone = [u for u in us if all(q["orc"] == "del" for q in u)]
        whole_del += sum(len(u) for u in gone); D[b] = dict(utts=len(us), whole_deleted_pct=pct(len(gone), len(us)))
    D["share_of_all_del"] = pct(whole_del, n_del)
    return dict(n=n, del_pct=pct(n_del, n), sub_pct=pct(sum(q["orc"] == "sub" for q in words), n), solo_del_pct=pct(sum(q["orc"] == "del" for q in solo), len(solo)),
                ovl_del_pct=pct(sum(q["orc"] == "del" for q in ovl), len(ovl)), A=A, B=B, C=C, D=D)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dump"); ap.add_argument("--gap", type=float, default=2.0); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    by = collections.defaultdict(list)
    for l in open(a.dump, encoding="utf-8"):
        q = json.loads(l); by[(q["corpus"], q["sys"])].append(q)
    out = {}
    for (c, s), words in sorted(by.items()):
        r = analyse(words, a.gap); out[f"{c}|{s}"] = r
        print(f"== {c} · {s} · 참조 단어 {r['n']} · 삭제 {r['del_pct']} % 대치 {r['sub_pct']} % · 혼자 말 삭제 {r['solo_del_pct']} % · 겹친 말 삭제 {r['ovl_del_pct']} %")
        for tag, name in (("all", "전체"), ("solo", "혼자→혼자"), ("after_ovl", "겹침 직후 혼자 말"), ("in_ovl", "겹친 단어")):
            x = r["A"][tag]
            print(f"  A {name:10s} 앞 단어 삭제일 때 {x['prev_del']['del_pct']} %(n {x['prev_del']['n']}) · 맞았을 때 {x['prev_ok']['del_pct']} %(n {x['prev_ok']['n']}) · 대치일 때 {x['prev_sub']['del_pct']} %(n {x['prev_sub']['n']})")
        print("  B 삭제 묶음 길이별 삭제 단어 몫: " + " · ".join(f"{b} {v['word_share']} %({v['runs']} 묶음)" for b, v in r["B"].items()))
        cc = r["C"]; print(f"  C 겹침 포함 삭제 묶음 {cc['runs_with_overlap']} · 겹침 뒤로 번진 혼자 말 삭제 {cc['solo_tail_words']} 단어(혼자 말 삭제의 {cc['tail_share_of_solo_del']} %) · 3 단어 이상 번진 묶음 {cc['runs_tail_ge3']}")
        d = r["D"]; print("  D 통째로 빠진 발화: " + " · ".join(f"{b} 단어 {d[b]['whole_deleted_pct']} %({d[b]['utts']} 발화)" for b in ("1-2", "3-5", "6-15", "16+")) + f" · 전체 삭제 중 몫 {d['share_of_all_del']} %")
    if a.json: json.dump(out, open(a.json, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()

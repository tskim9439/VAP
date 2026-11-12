#!/usr/bin/env python
"""AMI 에서 우리가 놓친 말 — 상황별 집계와 샘플 자료(2026-11-12, 사용자 "AMI 에서 우리가 놓친 케이스에 대해 샘플과 놓친 구간에 대한 시각화를 수행해서 그 오류 유형을 보고 싶어").
입력(맥, 읽기만): gap_diag.py --dump 의 참조 단어 줄(E13 · N3D + 출력 재귀속, Parakeet + N3D — 같은 참조·같은 셈)과 .extra.jsonl(삽입·화자 짝),
  저장한 N3D 확률(<probs>/<corpus>:<회의>@full.npz), ami-bench-v1 대화 JSON(발화 구간)·mono 캐시(오디오 조각).
출력(<out> 새 폴더): data.json(상황별 집계 + 샘플), clips/<id>.mp3(샘플 구간 12 s, ffmpeg).
  python miss_viz_build.py --dump words.jsonl --probs <폴더> --bench <ami-bench-v1> --out <폴더> [--seed 20261112] [--win 12]
상황(참조 단어 전체를 나눈다, 위에서부터 먼저 해당하는 것): N3D 가 그 화자를 켜지 않음(단어 끝 시각 −0.32 ~ +0.16 s) → 겹침(그 시각 말하는 참조 화자 2 명 이상)·혼자 ×
  짧은 발화(정규화 뒤 3 단어 이하) / 긴 발화의 첫 두 단어 / 긴 발화의 나머지. 놓침 = 받아쓰기 놓침(화자 귀속과 무관한 orc 삭제·대치). 귀속 오류 = 맞게 썼지만 다른 화자 칸."""
import argparse, collections, json, os, random, subprocess, wave
import numpy as np
from scipy.optimize import linear_sum_assignment

FR = 0.08
CTX = [("n3d_off", "N3D 가 그 화자를 켜지 않음"), ("ov_short", "겹침 · 짧은 발화(3 단어 이하)"), ("ov_head", "겹침 · 긴 발화의 첫 두 단어"), ("ov_rest", "겹침 · 긴 발화의 나머지"),
       ("solo_short", "혼자 · 짧은 발화(3 단어 이하)"), ("solo_head", "혼자 · 긴 발화의 첫 두 단어"), ("solo_rest", "혼자 · 긴 발화의 나머지")]


def runs_of(b):
    """bool 배열 → [(시작 프레임, 끝 프레임)]"""
    d = np.diff(np.concatenate([[0], b.astype(np.int8), [0]])); return list(zip(np.where(d == 1)[0].tolist(), np.where(d == -1)[0].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True); ap.add_argument("--probs", required=True); ap.add_argument("--bench", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261112); ap.add_argument("--win", type=float, default=12.0); ap.add_argument("--no-audio", action="store_true")
    a = ap.parse_args(); os.makedirs(os.path.join(a.out, "clips"), exist_ok=True); rng = random.Random(a.seed)
    W = collections.defaultdict(lambda: collections.defaultdict(list))                     # sys → (corpus, cid) → 참조 단어 줄
    for l in open(a.dump, encoding="utf-8"):
        q = json.loads(l); W[q["sys"]][(q["corpus"], q["cid"])].append(q)
    INS = collections.defaultdict(list); MAP = {}
    for l in open(a.dump + ".extra.jsonl", encoding="utf-8"):
        q = json.loads(l)
        if q["sys"] != "E13": continue
        if "map" in q: MAP[(q["corpus"], q["cid"])] = q["map"]
        else: INS[(q["corpus"], q["cid"])].append(q)
    pk_ins = collections.Counter()
    for l in open(a.dump + ".extra.jsonl", encoding="utf-8"):
        q = json.loads(l)
        if q["sys"] != "E13" and "ins" in q: pk_ins[q["corpus"]] += 1
    DLG = {}
    for c in ("ami-ihmmix", "ami-sdm"):
        for l in open(os.path.join(a.bench, f"{c}.refined.dialogues.jsonl"), encoding="utf-8"):
            d = json.loads(l); DLG[(c, d["conv_id"])] = d
    M = {}                                                                                  # (corpus, cid) → 회의 자료
    for key, ws in W["E13"].items():
        c, cid = key; pk = W["Parakeet+N3D"][key]
        assert len(pk) == len(ws) and all(x["w"] == y["w"] and x["s"] == y["s"] for x, y in zip(ws, pk)), key
        z = np.load(os.path.join(a.probs, f"{cid if ':' in cid else c + ':' + cid}@full.npz")); pr = z["prob"].astype(np.float32) > 0.5; T = len(pr); ns = max(q["s"] for q in ws) + 1
        C = np.zeros((ns, pr.shape[1]))
        for q in ws:
            f = min(T - 1, int(q["t"] / FR)); C[q["s"]] += pr[max(0, f - 2): f + 1].any(0)
        r, cc = linear_sum_assignment(-C); slot = dict(zip(r.tolist(), cc.tolist()))
        d = DLG.get((c, cid)) or DLG.get((c, cid.split(":")[-1])); assert d is not None, key
        names = sorted({u["speaker"] for u in d["utterances"]}); spans = {n: sorted((u["start"], u["end"]) for u in d["utterances"] if u["speaker"] == n) for n in names}
        C2 = np.zeros((ns, len(names)))                                                    # 참조 화자 색인 → 대화 JSON 의 화자(단어 시각이 그 화자 발화 구간 안인 수)
        for q in ws:
            for j, n in enumerate(names):
                C2[q["s"], j] += any(s0 - 0.3 <= q["t"] <= e0 + 0.3 for s0, e0 in spans[n] if s0 - 1 < q["t"] < e0 + 1)
        r2, c2 = linear_sum_assignment(-C2); nm = {int(s): names[j] for s, j in zip(r2, c2)}
        groups = collections.defaultdict(list)
        for i, q in enumerate(ws): groups[(q["s"], q["utt"])].append(i)
        for idx in groups.values():
            idx.sort(key=lambda i: ws[i]["t"])
            for p, i in enumerate(idx): ws[i]["n"] = len(idx); ws[i]["pos"] = p
        for i, q in enumerate(ws):
            k = slot.get(q["s"]); f0 = max(0, int((q["t"] - 0.32) / FR)); f1 = min(T, int((q["t"] + 0.16) / FR) + 1)
            q["on"] = bool(k is not None and pr[f0:f1, k].any()); q["pk"] = pk[i]["orc"]; q["pkcp"] = pk[i]["cp"]
            q["miss"] = q["orc"] != "ok"; q["attr"] = q["orc"] == "ok" and q["cp"] != "ok"
            ov = q["ov"] >= 2
            q["ctx"] = "n3d_off" if not q["on"] else (("ov_" if ov else "solo_") + ("short" if q["n"] <= 3 else "head" if q["pos"] < 2 else "rest"))
        M[key] = dict(ws=ws, pr=pr, slot=slot, spans={s: spans[n] for s, n in nm.items()}, ins=INS.get(key, []), map=MAP.get(key, {}))
    # ── 상황별 집계
    summary = {}
    for c in ("ami-ihmmix", "ami-sdm"):
        allw = [q for (cc_, _), m in M.items() if cc_ == c for q in m["ws"]]; N = len(allw); rows = []
        for k, name in CTX:
            qs = [q for q in allw if q["ctx"] == k]; n = len(qs)
            if not n: continue
            od = sum(q["orc"] == "del" for q in qs); os_ = sum(q["orc"] == "sub" for q in qs); pd_ = sum(q["pk"] == "del" for q in qs); ps = sum(q["pk"] == "sub" for q in qs)
            top = collections.Counter(q["w"] for q in qs if q["miss"]).most_common(6)
            both = sum(q["miss"] and q["pk"] != "ok" for q in qs); only = sum(q["miss"] and q["pk"] == "ok" for q in qs)
            rows.append(dict(key=k, name=name, n=n, share=round(100 * n / N, 2), our_del=round(100 * od / n, 1), our_sub=round(100 * os_ / n, 1), pk_del=round(100 * pd_ / n, 1), pk_sub=round(100 * ps / n, 1),
                             gap=round(100 * ((od + os_) - (pd_ + ps)) / N, 2), our_pts=round(100 * (od + os_) / N, 2), only_ours=round(100 * only / max(1, od + os_), 1), top=[[w_, n_] for w_, n_ in top]))
        miss = sum(q["miss"] for q in allw); pkm = sum(q["pk"] != "ok" for q in allw)
        summary[c] = dict(N=N, rows=rows, our_miss=round(100 * miss / N, 2), pk_miss=round(100 * pkm / N, 2), our_attr=round(100 * sum(q["attr"] for q in allw) / N, 2),
                          pk_attr=round(100 * sum(q["pk"] == "ok" and q["pkcp"] != "ok" for q in allw) / N, 2),
                          our_ins=round(100 * sum(len(m["ins"]) for (cc_, _), m in M.items() if cc_ == c) / N, 2), pk_ins=round(100 * pk_ins[c] / N, 2))
    # ── 샘플
    samples = []; used = collections.defaultdict(list)
    def free(key, t0, t1): return all(t1 < a_ - 5 or t0 > b_ + 5 for a_, b_ in used[key])
    def build(kind, key, t0):
        c, cid = key; m = M[key]; t1 = t0 + a.win; sid = f"{kind}-{c[4:]}-{cid.split(':')[-1]}-{int(t0)}"
        spk = []
        for s in sorted({q["s"] for q in m["ws"]}):
            ws = [q for q in m["ws"] if q["s"] == s and t0 <= q["t"] <= t1]
            ut = [[round(max(t0, s0) - t0, 2), round(min(t1, e0) - t0, 2)] for s0, e0 in m["spans"].get(s, []) if e0 > t0 and s0 < t1]
            k = m["slot"].get(s); n3 = []
            if k is not None:
                f0, f1 = int(t0 / FR), min(len(m["pr"]), int(t1 / FR) + 1)
                n3 = [[round(a_ * FR, 2), round(b_ * FR, 2)] for a_, b_ in runs_of(m["pr"][f0:f1, k])]
            if not ws and not ut: continue
            words = []
            for q in sorted(ws, key=lambda q: q["t"]):
                st = "del" if q["orc"] == "del" else "sub" if q["orc"] == "sub" else "attr" if q["attr"] else "ok"
                to = m["map"].get(q.get("hs")) if st == "attr" else None
                words.append(dict(t=round(q["t"] - t0, 2), w=q["w"], st=st, hw=q.get("hw") if st == "sub" else None, to=to, ov=q["ov"], pk=q["pk"], on=q["on"], ctx=q["ctx"]))
            spk.append(dict(s=s, utts=ut, n3d=n3, words=words))
        ins = [dict(t=round(q["t"] - t0, 2), w=q["w"], s=m["map"].get(q["hs"])) for q in m["ins"] if t0 <= q["t"] <= t1]
        ev = sorted([(u[0], 1) for sp in spk for u in sp["utts"]] + [(u[1], -1) for sp in spk for u in sp["utts"]]); ovl = []; n_ = 0; st_ = None
        for t_, d_ in ev:
            n_ += d_
            if n_ >= 2 and st_ is None: st_ = t_
            if n_ < 2 and st_ is not None:
                if t_ - st_ > 0.05: ovl.append([st_, t_])
                st_ = None
        aw = [w for sp in spk for w in sp["words"]]; mis = [w for w in aw if w["st"] in ("del", "sub")]
        clip = None
        if not a.no_audio:
            x = np.load(os.path.join(a.bench, "mono", c, f"{c}_{cid.split(':')[-1]}.npy"), mmap_mode="r"); y = np.asarray(x[int(t0 * 16000): int(t1 * 16000)], dtype=np.float32)
            y = y / max(1e-4, float(np.abs(y).max())) * 0.9; wav = os.path.join(a.out, "clips", sid + ".wav"); mp3 = wav[:-4] + ".mp3"
            with wave.open(wav, "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(16000); f.writeframes((y * 32767).astype(np.int16).tobytes())
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ac", "1", "-ar", "16000", "-b:a", "40k", mp3], check=True); os.remove(wav); clip = f"clips/{sid}.mp3"
        used[key].append((t0, t1))
        return dict(id=sid, kind=kind, corpus=c, meeting=cid.split(":")[-1], t0=round(t0, 1), dur=a.win, audio=clip, speakers=spk, ins=ins, ovl=ovl,
                    stat=dict(n=len(aw), dele=sum(w["st"] == "del" for w in aw), sub=sum(w["st"] == "sub" for w in aw), attr=sum(w["st"] == "attr" for w in aw), ins=len(ins),
                              pk_hit=sum(w["pk"] == "ok" for w in mis), miss=len(mis)))
    def pick(kind, c, anchors_of, need):
        keys = [k for k in M if k[0] == c]; rng.shuffle(keys)
        for _ in range(400):
            key = rng.choice(keys); an = anchors_of(M[key])
            if not an: continue
            t = rng.choice(an); t0 = max(0.0, round(t - a.win * 0.55, 1)); t1 = t0 + a.win
            if sum(t0 <= x <= t1 for x in an) < need or not free(key, t0, t1): continue
            if any(sm["corpus"] == c and sm["meeting"] == key[1].split(":")[-1] and sm["kind"] == kind for sm in samples): continue
            return build(kind, key, t0)
        return None
    for k, _ in CTX:
        need = 1 if k.endswith("short") or k == "n3d_off" else 2
        for c in ("ami-ihmmix", "ami-ihmmix", "ami-sdm"):
            sm = pick(k, c, lambda m, k=k: [q["t"] for q in m["ws"] if q["ctx"] == k and q["miss"]], need)
            if sm: samples.append(sm)
    for c in ("ami-ihmmix", "ami-sdm"):
        sm = pick("attr", c, lambda m: [q["t"] for q in m["ws"] if q["attr"]], 3)
        if sm: samples.append(sm)
        sm = pick("ins", c, lambda m: [q["t"] for q in m["ins"]], 3)
        if sm: samples.append(sm)
    json.dump(dict(summary=summary, ctx=[dict(key=k, name=n) for k, n in CTX], samples=samples, seed=a.seed), open(os.path.join(a.out, "data.json"), "w"), ensure_ascii=False)
    print("샘플", len(samples), collections.Counter(s["kind"] for s in samples))
    for c, sm in summary.items():
        print(c, "참조", sm["N"], "놓침 우리", sm["our_miss"], "Parakeet", sm["pk_miss"], "귀속", sm["our_attr"], sm["pk_attr"], "삽입", sm["our_ins"], sm["pk_ins"])
        for r in sm["rows"]: print("  ", r["name"], "몫", r["share"], "우리 삭제·대치", r["our_del"], r["our_sub"], "Parakeet", r["pk_del"], r["pk_sub"], "격차", r["gap"], "우리만 놓친 몫", r["only_ours"], r["top"][:4])


if __name__ == "__main__":
    main()

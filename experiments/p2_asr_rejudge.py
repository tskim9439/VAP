#!/usr/bin/env python
"""라벨 ↔ Qwen3-ASR-0.6B 대조(p2_asr_check)에서 애매했던 발화를 큰 심판(Qwen3-ASR-1.7B)으로 다시 인식해 누가 틀렸는지 가른다(표본) + 청취 페이지.

  python experiments/p2_asr_rejudge.py --data <phase2 dir> --out <dir> [--corpora aihub71631,aihub134-1,aihub134-2] [--band 0.1,0.2] [--n 150]
      [--control 40] [--unchecked otoSpeech:150] [--judge <Qwen3-ASR-1.7B>] [--base <Qwen3-ASR-0.6B>] [--tau 0.1] [--page-per-cat 4] [--gpu 0]

표본(코퍼스마다, seed 0): 전사 감독에 쓰인 발화(정제 파일에 tokens 가 있음) 가운데 asrcheck(0.6B) err 가 band 안인 n 개 + err < 0.05 인 control 개.
  --unchecked 코퍼스:n — asrcheck 가 없는 코퍼스(otoSpeech)는 감독 발화 무작위 n 개에 0.6B·1.7B 를 둘 다 돌린다.
오디오·라벨·err 는 p2_asr_check 와 같다(원래 <corpus>.dialogues.jsonl 의 깨끗한 화자 채널 구간, 라벨 = u.text, 한국어 CER 공백 제외·영어/독일어 WER).
판정(τ): L = err(라벨 → 1.7B), J = err(1.7B → 0.6B)
  label_wrong = J < τ ≤ L(두 심판은 맞고 라벨만 다름) · base_wrong = L < τ ≤ J(라벨·1.7B 가 맞고 0.6B 만 다름)
  all_agree = L < τ, J < τ · all_differ = L ≥ τ, J ≥ τ(셋이 서로 다름 — 어려운 음성·잡음·겹침)
출력: <out>/rejudge.utts.jsonl(발화별), <out>/summary.json, <out>/index.html + clips/(로컬 열람용 — 코퍼스 음성이라 외부에 올리지 않는다)."""
import argparse, collections, difflib, glob, html, json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CATS = (("label_wrong", "라벨 틀림 — 두 심판(0.6B·1.7B)은 맞고 라벨만 다름"), ("base_wrong", "0.6B 틀림 — 라벨·1.7B 가 맞음"),
        ("all_differ", "셋 다 다름 — 어려운 음성"), ("all_agree", "셋 다 맞음(τ 안)"))


def err(lang, ref, hyp):
    from vapasr.data.cst_eval import ed
    from vapasr.data.textnorm import score_ko, score_en
    from vapasr.data.textnorm_de import score_de
    if lang == "Korean": r, h = score_ko(ref, False), score_ko(hyp, False); return ed(list(r), list(h)) / max(1, len(r))
    if lang == "German": r, h = score_de(ref).split(), score_de(hyp).split(); return ed(r, h) / max(1, len(r))
    r, h = score_en(ref).split(), score_en(hyp).split(); return ed(r, h) / max(1, len(r))


def utt_audio(d, u):
    from vapasr.data.streams import load_utt_audio
    ref = d.channels[u.speaker]
    if ref.pieces is not None:
        p = min(ref.pieces, key=lambda x: abs(x[1] - u.start)); return load_utt_audio(p[0])
    return load_utt_audio(ref.path, u.start, u.end - u.start)


def category(L, J, tau):
    return "all_agree" if L < tau and J < tau else "label_wrong" if J < tau else "base_wrong" if L < tau else "all_differ"


def mark_diff(a, b, lang):
    """b 를 a 와 견줘 다른 부분을 <mark> — 한국어는 글자, 그 밖은 단어 단위(공백 차이는 표시하지 않는다)."""
    if lang == "Korean":
        A, Bc = a.replace(" ", ""), list(b); keep = [c for c in Bc if c != " "]; sm = difflib.SequenceMatcher(None, list(A), keep, autojunk=False)
        bad = set()
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op != "equal": bad.update(range(j1, j2))
        out, k = [], 0
        for c in Bc:
            if c == " ": out.append(" "); continue
            out.append(f"<mark>{html.escape(c)}</mark>" if k in bad else html.escape(c)); k += 1
        return "".join(out).replace("</mark><mark>", "")
    A, Bw = a.lower().split(), b.split(); sm = difflib.SequenceMatcher(None, A, [w.lower() for w in Bw], autojunk=False); bad = set()
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal": bad.update(range(j1, j2))
    return " ".join(f"<mark>{html.escape(w)}</mark>" if j in bad else html.escape(w) for j, w in enumerate(Bw))


def supervised(path):
    """정제 파일에서 전사 감독 발화 {(conv, 원래 utt id)}."""
    s = set()
    for l in open(path, encoding="utf-8"):
        d = json.loads(l)
        for u in d["utterances"]:
            if u.get("tokens"): s.add((d["conv_id"], u["utt_id"].split("#")[0]))
    return s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--corpora", default="aihub71631,aihub134-1,aihub134-2"); ap.add_argument("--band", default="0.1,0.2"); ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--control", type=int, default=40); ap.add_argument("--unchecked", default="otoSpeech:150"); ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--judge", default="/soundai/Model/Qwen3-ASR-1.7B"); ap.add_argument("--base", default="/soundai/Model/Qwen3-ASR-0.6B")
    ap.add_argument("--page-per-cat", type=int, default=4); ap.add_argument("--min-dur", type=float, default=0.3); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--gpu", default="0")
    a = ap.parse_args(); os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.gpu)
    import numpy as np, soundfile as sf, torch
    from qwen_asr import Qwen3ASRModel
    from vapasr.data.dialogue import Dialogue
    lo, hi = (float(x) for x in a.band.split(",")); rng = random.Random(0); picks = []      # (corpus, conv, utt, 묶음, 0.6B 결과 | None)
    for c in [x for x in a.corpora.split(",") if x]:
        merged = os.path.join(a.data, "asrcheck", f"{c}.utts.jsonl")
        files = [merged] if os.path.exists(merged) else sorted(glob.glob(os.path.join(a.data, "asrcheck", f"{c}.shard*.utts.jsonl")))
        rows = {(r["conv"], r["utt"]): r for f in files for r in map(json.loads, open(f, encoding="utf-8"))}
        sup = supervised(os.path.join(a.data, f"{c}.refined.dialogues.jsonl"))
        ok = [r for k, r in rows.items() if k in sup and r["dur"] >= a.min_dur]
        band = [r for r in ok if lo <= r["err"] < hi]; ctrl = [r for r in ok if r["err"] < 0.05]
        for grp, pool, n in (("band", band, a.n), ("control", ctrl, a.control)):
            for r in rng.sample(pool, min(n, len(pool))): picks.append((c, r["conv"], r["utt"], grp, r))
        print(f"  {c}: 감독·대조 발화 {len(ok)} · band [{lo}, {hi}) {len(band)} ({100 * len(band) / max(1, len(ok)):.0f} %) · control {len(ctrl)}", flush=True)
    for spec in [x for x in a.unchecked.split(",") if x]:
        c, n = spec.split(":"); sup = sorted(supervised(os.path.join(a.data, f"{c}.refined.dialogues.jsonl")))
        for conv, utt in rng.sample(sup, min(int(n) * 2, len(sup))): picks.append((c, conv, utt, "unchecked", None))
    need = collections.defaultdict(set)
    for c, conv, utt, _, _ in picks: need[c].add(conv)
    utts = {}
    for c, convs in need.items():
        for l in open(os.path.join(a.data, f"{c}.dialogues.jsonl"), encoding="utf-8"):
            cid = l[13:400].split('"', 1)[0] if l.startswith('{"conv_id": "') else json.loads(l)["conv_id"]
            if cid not in convs: continue
            d = Dialogue.from_json(l)
            for u in d.utterances: utts[(c, d.conv_id, u.utt_id)] = (d, u)
    items = []
    for c, conv, utt, grp, r in picks:
        if (c, conv, utt) not in utts: continue
        d, u = utts[(c, conv, utt)]
        if not u.text or u.end - u.start < a.min_dur: continue
        items.append(dict(corpus=c, conv=conv, utt=utt, group=grp, lang=d.lang, dur=round(u.end - u.start, 2), label=u.text, base=r["hyp"] if r else None, base_err=r["err"] if r else None, _du=(d, u)))
    keep, cnt = [], collections.Counter()                                                   # unchecked 는 2 배 뽑아 길이 조건 뒤 n 개로 자른다
    lim = {spec.split(":")[0]: int(spec.split(":")[1]) for spec in [x for x in a.unchecked.split(",") if x]}
    for it in items:
        if it["group"] == "unchecked":
            if cnt[it["corpus"]] >= lim[it["corpus"]]: continue
            cnt[it["corpus"]] += 1
        keep.append(it)
    items = keep
    for i, it in enumerate(items): it["_i"] = i
    print(f"  표본 {len(items)} 발화({dict(collections.Counter((i['corpus'], i['group']) for i in items))})", flush=True)
    auds = []
    for it in items:
        try: auds.append(np.asarray(utt_audio(*it["_du"]), dtype=np.float32))
        except Exception as e: auds.append(None); it["audio_error"] = f"{type(e).__name__}: {str(e)[:100]}"
    def run(path, idx):
        m = Qwen3ASRModel.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda"); t0 = time.time(); out = {}; sec = 0.0
        for b in range(0, len(idx), a.batch):
            bi = idx[b: b + a.batch]; res = m.transcribe(audio=[(auds[i], 16000) for i in bi], language=[items[i]["lang"] for i in bi])
            for i, r_ in zip(bi, res): out[i] = r_.text if hasattr(r_, "text") else str(r_); sec += len(auds[i]) / 16000
        dt = time.time() - t0; del m; torch.cuda.empty_cache(); return out, dict(utts=len(idx), audio_s=round(sec, 1), wall_s=round(dt, 1), rtf=round(dt / max(1e-6, sec), 4))
    idx = [i for i, x in enumerate(auds) if x is not None]
    j17, sp17 = run(a.judge, idx); print(f"  1.7B: {sp17}", flush=True)
    un = [i for i in idx if items[i]["base"] is None]
    if un:
        j06, sp06 = run(a.base, un); print(f"  0.6B(대조 없던 코퍼스): {sp06}", flush=True)
        for i in un: items[i]["base"] = j06[i]; items[i]["base_err"] = round(err(items[i]["lang"], items[i]["label"], j06[i]), 3)
    else: sp06 = None
    for i in idx:
        it = items[i]; it["judge"] = j17[i]; it["L"] = round(err(it["lang"], it["label"], it["judge"]), 3); it["J"] = round(err(it["lang"], it["judge"], it["base"]), 3)
        it["cat"] = category(it["L"], it["J"], a.tau)
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True); res = [it for it in items if "cat" in it]
    with open(os.path.join(a.out, "rejudge.utts.jsonl"), "w", encoding="utf-8") as f:
        for it in res: f.write(json.dumps({k: v for k, v in it.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    summ = {}
    for key in sorted({(it["corpus"], it["group"]) for it in res}):
        g = [it for it in res if (it["corpus"], it["group"]) == key]; cc = collections.Counter(it["cat"] for it in g)
        summ[f"{key[0]}/{key[1]}"] = dict(n=len(g), **{k: round(cc[k] / len(g), 3) for k, _ in CATS}, label_err_med=float(np.median([it["L"] for it in g])),
                                          base_err_med=float(np.median([it["base_err"] for it in g])), judges_err_med=float(np.median([it["J"] for it in g])))
        print(f"  {key[0]:11s} {key[1]:9s} n {len(g):3d} · " + " · ".join(f"{k} {100 * summ[f'{key[0]}/{key[1]}'][k]:.0f} %" for k, _ in CATS) +
              f" · 중앙값 L {summ[f'{key[0]}/{key[1]}']['label_err_med']:.3f} 0.6B {summ[f'{key[0]}/{key[1]}']['base_err_med']:.3f} J {summ[f'{key[0]}/{key[1]}']['judges_err_med']:.3f}", flush=True)
    json.dump(dict(args=vars(a), speed_judge=sp17, speed_base=sp06, groups=summ), open(os.path.join(a.out, "summary.json"), "w"), ensure_ascii=False, indent=1)
    cards = collections.defaultdict(list); prng = random.Random(1)
    for c in sorted({it["corpus"] for it in res}):
        for cat, _ in CATS:
            pool = [it for it in res if it["corpus"] == c and it["cat"] == cat and it["group"] != "control"]
            for k, it in enumerate(prng.sample(pool, min(a.page_per_cat, len(pool)))):
                stem = f"{c}-{cat}-{k + 1}"; sf.write(os.path.join(a.out, "clips", f"{stem}.wav"), auds[it["_i"]], 16000, subtype="PCM_16")
                cards[cat].append(f'''<section class="case"><h3>{html.escape(c)} · {html.escape(it["utt"])} · {it["dur"]} s · {html.escape(it["group"])}</h3>
<audio controls preload="none" src="clips/{stem}.wav"></audio>
<table><tr><th>라벨</th><td>{mark_diff(it["judge"], it["label"], it["lang"])}</td><td class="e">1.7B 대비 {it["L"]:.2f}</td></tr>
<tr><th>Qwen3-ASR-0.6B</th><td>{mark_diff(it["label"], it["base"], it["lang"])}</td><td class="e">라벨 대비 {it["base_err"]:.2f}</td></tr>
<tr><th>Qwen3-ASR-1.7B</th><td>{mark_diff(it["label"], it["judge"], it["lang"])}</td><td class="e">0.6B ↔ 1.7B {it["J"]:.2f}</td></tr></table></section>''')
    rows = "".join(f'<tr><th>{html.escape(k)}</th><td>{v["n"]}</td>' + "".join(f'<td>{100 * v[c]:.0f} %</td>' for c, _ in CATS) + f'<td>{v["label_err_med"]:.3f}</td><td>{v["base_err_med"]:.3f}</td></tr>' for k, v in summ.items())
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>전사 라벨 재판정</title>
<style>:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2dfd8;--card:#fff;--mk:#ffe08a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#e8e6e1;--mut:#9a978f;--line:#34322e;--card:#1f1e1c;--mk:#6b5400}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.6 -apple-system,"Apple SD Gothic Neo",system-ui,sans-serif;max-width:1100px;margin:0 auto;padding:16px}}
.case{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}} h3{{font-size:14px;margin:0 0 6px;color:var(--mut)}}
table{{border-collapse:collapse;width:100%}} th,td{{border-top:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}} th{{width:140px;font-weight:600}} td.e{{width:130px;color:var(--mut);font-size:13px}}
mark{{background:var(--mk);color:inherit;border-radius:2px}} audio{{width:100%;max-width:520px;height:32px}} .sum td,.sum th{{text-align:right;width:auto}} .sum th:first-child{{text-align:left}}</style></head><body>
<h1>전사 라벨 재판정 — 라벨 · Qwen3-ASR-0.6B · Qwen3-ASR-1.7B</h1>
<p>표본: 전사 감독에 쓰인 발화 중 라벨↔0.6B 오류율 [{lo}, {hi}) 구간(band), 기준 비교용 &lt; 0.05(control), 대조가 없던 코퍼스(unchecked). 판정 τ = {a.tau}.
노란 표시는 견준 상대와 다른 부분이다(라벨 줄은 1.7B 와, 두 ASR 줄은 라벨과 비교). 한국어는 글자, 영어·독일어는 단어 단위이고 공백 차이는 표시하지 않는다.</p>
<table class="sum"><tr><th>코퍼스/묶음</th><th>n</th>{"".join(f"<th>{html.escape(t.split(' — ')[0])}</th>" for _, t in CATS)}<th>라벨↔1.7B 중앙값</th><th>라벨↔0.6B 중앙값</th></tr>{rows}</table>
{"".join(f"<h2>{html.escape(t)}</h2>" + "".join(cards[k]) for k, t in CATS)}</body></html>'''
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(page); print(f"→ {a.out}/index.html", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Stage 3 청취·열람 묶음 — 두 run 의 평가 원출력에서 사례(개선·악화·둘 다 귀속 실패·둘 다 잘 됨)를 골라 오디오·정답·두 출력을 한 HTML 로 묶는다.

  python experiments/cst_listen_bundle.py --base r1@2000=<eval dir>:<attr dir> --cand p9@3000=<eval dir>:<attr dir> \\
      --data <phase2 dir> --split <v2.json> --out <dir> [--delta 10] [--per-cat 6]

창·참조는 평가와 같다(cst_stage3_eval.window_ref — 전사 없는 음성 청크의 말은 참조에서 빠진다). 출력 슬롯은 창마다 cp 순열로 참조 화자에 맞춰 칠한다.
사례 고르기(δ 하나, 화자 2 명 창, 참조 단위 KO ≥ 80 글자·EN/DE ≥ 25 단어, 대화당 한 창, 언어 돌아가며):
  개선 = 후보 cp − 기준 cp 가 가장 작은 창(agn 차이 ≤ 5 pt — 귀속이 고쳐진 창) · 악화 = 그 차이가 가장 큰 창
  둘 다 귀속 실패 = 두 run 모두 (cp − agn)/n_ref ≥ 15 % 인 창 중 후보 귀속 몫이 큰 순 · 둘 다 잘 됨 = 두 run 모두 cp/n_ref ≤ 10 %, 화자 교대 많은 순
출력: <out>/index.html(로컬에서 여는 페이지), <out>/clips/<사례>_mono.wav(모델 입력과 같은 혼합), _stereo.wav(왼쪽 = 화자 A, 오른쪽 = 화자 B 채널, 있으면),
  <out>/cases.json. 출력 단어 표시: 화자 틀림(정렬된 참조 단위의 화자와 다름) · 전사 틀림(치환·삽입). 출력 시각은 방출 시각 − δ·80 ms."""
import argparse, collections, glob, html, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.cst_stage3_eval import windows_of, window_ref, words_of

LANG = {"Korean": "KO", "English": "EN", "German": "DE"}
MIN_UNITS = {"KO": 80, "EN": 25, "DE": 25}
CATS = (("better", "개선 — 후보가 귀속을 고친 창"), ("worse", "악화 — 후보가 더 나쁜 창"), ("both_bad", "둘 다 귀속 실패 — 남은 오류"), ("both_good", "둘 다 잘 됨 — 교대가 많은데 맞춘 창"))


def run_arg(s):
    name, rest = s.split("=", 1); ev, at = rest.split(":", 1); return name, ev, at


def load_run(ev, at, delta):
    rows = {r["id"]: r for f in glob.glob(os.path.join(ev, "predictions-rank*.jsonl")) for r in map(json.loads, open(f, encoding="utf-8")) if r["delta"] == delta}
    att = {w["id"]: w for w in map(json.loads, open(os.path.join(at, "windows.jsonl"), encoding="utf-8")) if w["delta"] == delta}
    return rows, att


def pick(base, cand, per_cat):
    """→ {범주: [창 id]} — 언어를 돌아가며, 대화당 한 창."""
    ids = [i for i in base[1] if i in cand[1] and base[1][i]["n_spk"] == 2 and base[1][i]["n_ref"] >= MIN_UNITS[base[1][i]["lang"]]]
    r = lambda run, i, k: 100 * run[1][i][k] / run[1][i]["n_ref"]
    att = lambda run, i: r(run, i, "cp") - r(run, i, "agn")
    keyed = {
        "better": sorted((i for i in ids if abs(r(cand, i, "agn") - r(base, i, "agn")) <= 5), key=lambda i: r(cand, i, "cp") - r(base, i, "cp")),
        "worse": sorted(ids, key=lambda i: -(r(cand, i, "cp") - r(base, i, "cp"))),
        "both_bad": sorted((i for i in ids if att(base, i) >= 15 and att(cand, i) >= 15), key=lambda i: -att(cand, i)),
        "both_good": sorted((i for i in ids if r(base, i, "cp") <= 10 and r(cand, i, "cp") <= 10), key=lambda i: -base[1][i]["changes_per_min"]),
    }
    out = {}
    for cat, order in keyed.items():
        if cat == "worse": order = [i for i in order if r(cand, i, "cp") > r(base, i, "cp")]
        if cat == "better": order = [i for i in order if r(cand, i, "cp") < r(base, i, "cp")]
        by = collections.defaultdict(list)
        for i in order: by[base[1][i]["lang"]].append(i)
        got, convs = [], set()
        while len(got) < per_cat and any(by.values()):
            for lg in ("KO", "EN", "DE"):
                while by[lg] and by[lg][0].rsplit("@", 1)[0] in convs: by[lg].pop(0)
                if by[lg] and len(got) < per_cat: i = by[lg].pop(0); got.append(i); convs.add(i.rsplit("@", 1)[0])
        out[cat] = got
    return out


def word_flags(ref_w, hyp_w, lang, perm):
    """→ (화자 틀림, 전사 틀림) 출력 단어 집합 {(출력 슬롯, 단어 index)} — attr_errors 와 같은 화자 무시 정렬."""
    from vapasr.data.cst_eval import units, align
    rw = sorted([(t, s, w) for s in (0, 1) for w, t in ref_w[s]], key=lambda x: x[0])
    R = [(u, s) for _, s, w in rw for u in units(w, lang)]
    hw = sorted([(t, perm[s], w, s, k) for s in (0, 1) for k, (w, t) in enumerate(hyp_w[s])], key=lambda x: x[0])
    H = [(u, ps, (s, k)) for _, ps, w, s, k in hw for u in units(w, lang)]
    bad, err = set(), set()
    for i, j in align([x[0] for x in R], [x[0] for x in H]):
        if j is None: continue
        if i is None or R[i][0] != H[j][0]: err.add(H[j][2])
        if i is not None and R[i][1] != H[j][1]: bad.add(H[j][2])
    return bad, err


def lines_of(items):
    """items [(시각, 슬롯, 텍스트, 표시)] → 시각순으로 같은 슬롯이 이어진 줄 [(슬롯, 시작, 끝, [(텍스트, 표시)])]."""
    out = []
    for t, s, w, f in sorted(items, key=lambda x: x[0]):
        if out and out[-1][0] == s: out[-1][2] = t; out[-1][3].append((w, f))
        else: out.append([s, t, t, [(w, f)]])
    return out


def render_lines(lines):
    rows = []
    for s, a, b, ws in lines:
        txt = " ".join(f'<span class="{f}">{html.escape(w)}</span>' if f else html.escape(w) for w, f in ws)
        rows.append(f'<div class="ln s{s}"><span class="tag">{"AB"[s]}</span><span class="tm">{a:5.1f}–{b:5.1f} s</span><span class="tx">{txt}</span></div>')
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True, type=run_arg); ap.add_argument("--cand", required=True, type=run_arg)
    ap.add_argument("--data", required=True); ap.add_argument("--split", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--delta", type=int, default=10); ap.add_argument("--per-cat", type=int, default=6); a = ap.parse_args()
    import numpy as np, soundfile as sf
    from transformers import AutoTokenizer
    from vapasr.data.cst_eval import parse_emitted
    from vapasr.data.dialogue_mix import load_channel, crop_audio
    runs = {n: load_run(ev, at, a.delta) for n, ev, at in (a.base, a.cand)}; bn, cn = a.base[0], a.cand[0]
    conf = json.load(open(os.path.join(a.base[1], "config-rank0.json")))
    tok = AutoTokenizer.from_pretrained(conf["checkpoint"]); sp = json.load(open(os.path.join(conf["checkpoint"], "config.json")))["sp_ids"]
    EOT, SPK = sp["<EOT>"], (sp["<SPK_A>"], sp["<SPK_B>"])
    sets, items = windows_of(argparse.Namespace(corpora=conf["corpora"], data=a.data, split=a.split, deltas=[a.delta], limit_windows=conf["limit_windows"],
                                                backchannel=tuple(conf["backchannel"]) if conf.get("backchannel") else None), tok)
    index = {f"{sets[c].items[i][0]}@{sets[c].items[i][1]}": (c, i) for c, i in items}
    chosen = pick(runs[bn], runs[cn], a.per_cat); os.makedirs(os.path.join(a.out, "clips"), exist_ok=True); cases = []; cards = collections.defaultdict(list)
    for cat, ids in chosen.items():
        for n, wid in enumerate(ids):
            c, i = index[wid]; ds = sets[c]; cid, t0, L = ds.items[i]; d = ds.dlgs[cid]; lang = LANG[d.lang]
            ref_w, ref_eot, mask, spk = window_ref(ds, i, a.delta, EOT, tok)
            ref_items = [(t, s, w, "") for s in (0, 1) for w, t in ref_w[s]] + [(t, s, "⏎", "eot") for s, t, p in ref_eot if p is not None and p >= 0.5]
            case = dict(cat=cat, id=wid, corpus=d.corpus, lang=lang, t0=t0, L=L, delta=a.delta, speakers=spk, runs={})
            panes = [("정답", render_lines(lines_of(ref_items)), "")]
            for name in (bn, cn):
                r, w = runs[name][0][wid], runs[name][1][wid]
                hs, he = parse_emitted([tuple(e) for e in r["emitted"]], SPK, EOT, drop_chunks=mask); hyp_w = [words_of(tok, hs[0]), words_of(tok, hs[1])]
                bad, err = word_flags(ref_w, hyp_w, d.lang, r["perm"]); sh = a.delta * 0.08
                hyp_items = [(t - sh, r["perm"][s], wd, "spk" if (s, k) in bad else ("err" if (s, k) in err else "")) for s in (0, 1) for k, (wd, t) in enumerate(hyp_w[s])]
                hyp_items += [(t - sh, r["perm"][s], "⏎", "eot") for s, t in he]
                rate = lambda k: 100 * w[k] / w["n_ref"]
                m = dict(cp=round(rate("cp"), 1), agn=round(rate("agn"), 1), attr=round(rate("cp") - rate("agn"), 1), mislabel=round(100 * w["n_mis"] / w["n_ref"], 1),
                         streak=round(100 * w["whole_streak_units"] / w["n_ref"], 1), swap=r["perm"] == [1, 0], hyp_changes=w["hyp_changes"], ref_changes=w["ref_changes"])
                case["runs"][name] = dict(m, text=r["hyp_text"]); panes.append((name, render_lines(lines_of(hyp_items)), m))
            stem = f"{cat}-{n + 1:02d}-{lang}"
            sf.write(os.path.join(a.out, "clips", f"{stem}_mono.wav"), np.asarray(crop_audio(ds.mono(cid), t0, t0 + L), dtype=np.float32), 16000, subtype="PCM_16")
            st = None
            try:
                ch = []
                for s in spk:
                    x = crop_audio(load_channel(d.channels[s], d.duration_s), t0, t0 + L).astype(np.float32)
                    g = 10 ** (-23 / 20) / max(float(np.sqrt(np.mean(x ** 2))), 1e-6); x = x * g; pk = float(np.max(np.abs(x))) if len(x) else 0.0
                    ch.append(x / pk * 0.98 if pk > 0.98 else x)
                if len(ch) == 2:
                    T = min(len(ch[0]), len(ch[1])); sf.write(os.path.join(a.out, "clips", f"{stem}_stereo.wav"), np.stack([ch[0][:T], ch[1][:T]], 1), 16000, subtype="PCM_16"); st = f"clips/{stem}_stereo.wav"
            except Exception as e: case["stereo_error"] = f"{type(e).__name__}: {str(e)[:120]}"
            w0 = runs[bn][1][wid]; case.update(stem=stem, overlap=w0["overlap"], changes_per_min=w0["changes_per_min"], n_ref=w0["n_ref"], unit="글자" if lang == "KO" else "단어")
            cases.append(case)
            mt = "".join(f'<tr><th>{html.escape(nm)}</th><td>{m["cp"]}</td><td>{m["agn"]}</td><td>{m["attr"]}</td><td>{m["streak"]}</td><td>{m["hyp_changes"]}/{m["ref_changes"]}</td><td>{"예" if m["swap"] else ""}</td></tr>'
                         for nm, _, m in panes[1:])
            audio = f'<div class="au"><span>모델 입력(mono)</span><audio controls preload="none" src="clips/{stem}_mono.wav"></audio></div>'
            if st: audio += f'<div class="au"><span>스테레오(왼쪽 A · 오른쪽 B)</span><audio controls preload="none" src="{st}"></audio></div>'
            cards[cat].append(f'''<section class="case" id="{stem}"><h3>{stem} · {html.escape(d.corpus)} · {html.escape(cid)} @ {t0:.1f} s · {L:.1f} s</h3>
<p class="meta">겹침 {100 * case["overlap"]:.0f} % · 교대 {case["changes_per_min"]:.0f} /분 · 참조 {case["n_ref"]} {case["unit"]} · δ{a.delta} · 화자 A = {html.escape(spk[0])}, B = {html.escape(spk[1] if len(spk) > 1 else "-")}</p>
{audio}
<table class="m"><tr><th></th><th>cp %</th><th>agn %</th><th>귀속 몫</th><th>이어짐</th><th>교대 출력/정답</th><th>슬롯 뒤집음</th></tr>{mt}</table>
<div class="panes">{"".join(f'<div class="pane"><h4>{html.escape(nm)}</h4>{body}</div>' for nm, body, _ in panes)}</div></section>''')
            print(f"  {stem} {wid}", flush=True)
    toc = "".join(f'<li><a href="#cat-{k}">{html.escape(t)}</a> ({len(cards[k])})</li>' for k, t in CATS)
    body = "".join(f'<h2 id="cat-{k}">{html.escape(t)}</h2>' + "".join(cards[k]) for k, t in CATS)
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Stage 3 청취 묶음</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2dfd8;--a:#1f5fa8;--b:#b5521b;--abg:#eaf1fa;--bbg:#fbeee6;--bad:#c62828;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#e8e6e1;--mut:#9a978f;--line:#34322e;--a:#7fb0ea;--b:#f0a070;--abg:#1d2a3a;--bbg:#3a271c;--bad:#ff7b7b;--card:#1f1e1c}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,"Apple SD Gothic Neo",system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px}}
h1{{font-size:22px}} h2{{margin-top:36px;border-bottom:2px solid var(--line);padding-bottom:4px}} h3{{font-size:15px;margin:0 0 4px}} h4{{margin:0 0 6px;font-size:14px}}
.case{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin:14px 0}} .meta{{color:var(--mut);margin:0 0 8px;font-size:13px}}
.au{{display:flex;gap:10px;align-items:center;margin:4px 0;font-size:13px;color:var(--mut)}} .au span{{width:170px}} audio{{height:32px;flex:1;max-width:520px}}
table.m{{border-collapse:collapse;margin:8px 0;font-size:13px}} table.m th,table.m td{{border:1px solid var(--line);padding:2px 8px;text-align:right}} table.m th:first-child{{text-align:left}}
.panes{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}} @media (max-width:900px){{.panes{{grid-template-columns:1fr}}}}
.pane{{border:1px solid var(--line);border-radius:6px;padding:8px;overflow-wrap:anywhere}}
.ln{{display:grid;grid-template-columns:22px 92px 1fr;gap:4px;padding:2px 4px;border-radius:4px;margin:2px 0;font-size:14px}} .ln.s0{{background:var(--abg)}} .ln.s1{{background:var(--bbg)}}
.tag{{font-weight:700}} .s0 .tag{{color:var(--a)}} .s1 .tag{{color:var(--b)}} .tm{{color:var(--mut);font-variant-numeric:tabular-nums;font-size:12px}}
.spk{{color:var(--bad);font-weight:600;text-decoration:underline wavy}} .err{{text-decoration:underline dotted var(--mut)}} .eot{{color:var(--mut)}}
.legend span{{margin-right:14px}} ul{{margin:4px 0}}
</style></head><body>
<h1>Stage 3 청취 묶음 — {html.escape(bn)}(기준) vs {html.escape(cn)}(후보), δ{a.delta}</h1>
<p class="legend"><span class="ln s0" style="display:inline"><b>A</b> 창에서 먼저 말한 화자</span><span class="ln s1" style="display:inline"><b>B</b> 다른 화자</span>
<span class="spk">화자 틀림</span><span class="err">전사 틀림</span><span class="eot">⏎ 턴 끝(EOT)</span></p>
<p class="meta">출력 슬롯은 창마다 cp 순열로 정답 화자에 맞춰 칠했다("슬롯 뒤집음" = A/B 를 바꿔 낸 창). 출력 시각은 방출 시각 − δ·80 ms. 정답은 평가와 같다 — 전사 없는 음성 구간의 말은 빠진다.
cp·agn 은 이 창의 오류율(한국어 글자, 영어·독일어 단어), 귀속 몫 = cp − agn, 이어짐 = 연속 턴이 통째로 뒤바뀐 단위 비율.</p>
<ul>{toc}</ul>{body}</body></html>'''
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(page)
    json.dump(dict(base=bn, cand=cn, delta=a.delta, cases=cases), open(os.path.join(a.out, "cases.json"), "w"), ensure_ascii=False, indent=1)
    print(f"사례 {len(cases)} → {a.out}/index.html", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""합성 대화 미리보기 — sim_build_dialogues 산출에서 세션 몇 개를 골라 mono 캐시 오디오(앞 --clip-s 초)와 화자 A/B 시간 줄·전사를 한 HTML 로.

  python experiments/sim_preview.py --sim <sim dir> --out <dir> [--langs en,ko,de] [--per-lang 4] [--clip-s 45]
로컬에서 여는 페이지다(재료가 코퍼스 음성이라 외부에 올리지 않는다)."""
import argparse, html, json, os, random, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sim", required=True); ap.add_argument("--out", required=True); ap.add_argument("--langs", default="en,ko,de")
    ap.add_argument("--per-lang", type=int, default=4); ap.add_argument("--clip-s", type=float, default=45.0); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    import numpy as np, soundfile as sf
    from vapasr.data.dialogue import Dialogue
    from vapasr.data.dialogue_dataset import mono_cache_path
    from vapasr.data.sim_dialogue import overlap_stats
    os.makedirs(os.path.join(a.out, "clips"), exist_ok=True); cards = []
    for xx in [x for x in a.langs.split(",") if x]:
        p = os.path.join(a.sim, f"sim_{xx}.refined.dialogues.jsonl")
        if not os.path.exists(p): continue
        ds = [Dialogue.from_json(l) for l in open(p, encoding="utf-8")]; pools = sorted({d.meta.get("pool") for d in ds}); rng = random.Random(a.seed); pick = []
        for pl in pools:                                                                 # 풀마다 하나씩 먼저
            c = [d for d in ds if d.meta.get("pool") == pl]
            if c: pick.append(rng.choice(c))
        pick = (pick + rng.sample([d for d in ds if d not in pick], max(0, min(len(ds) - len(pick), a.per_lang - len(pick)))))[: max(a.per_lang, len(pools))]
        for d in pick:
            x = np.load(mono_cache_path(os.path.join(a.sim, "mono"), d)).astype(np.float32)[: int(a.clip_s * 16000)]
            stem = d.conv_id; sf.write(os.path.join(a.out, "clips", f"{stem}.wav"), x, 16000, subtype="PCM_16"); o = overlap_stats(d)
            rows = []
            for u in sorted(d.utterances, key=lambda u: u.start):
                if u.start > a.clip_s: break
                s = 0 if u.speaker == "a" else 1
                rows.append(f'<div class="ln s{s}"><span class="tag">{"AB"[s]}</span><span class="tm">{u.start:5.1f}–{u.end:5.1f} s</span><span class="tx">{html.escape(u.text or "(조각)")}</span></div>')
            cards.append(f'''<section class="case"><h3>{html.escape(stem)} · {html.escape(d.lang)} · 풀 {html.escape(str(d.meta.get("pool")))} · {d.duration_s:.0f} s(앞 {a.clip_s:.0f} s)</h3>
<p class="meta">화자 A = {html.escape(d.meta["keys"][0])} · B = {html.escape(d.meta["keys"][1])} · 턴 {d.meta["turns"]} · 맞장구 {d.meta["bc"]} · 나눈 긴 발화 {d.meta.get("split", 0)} ·
분당 교대 {o["changes_per_min"]} · 겹침 {100 * o["overlap_ratio"]:.1f} % · 이득 {html.escape(json.dumps(d.meta.get("gains")))}</p>
<audio controls preload="none" src="clips/{stem}.wav"></audio>{"".join(rows)}</section>''')
    page = f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>합성 대화 미리보기</title>
<style>:root{{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2dfd8;--abg:#eaf1fa;--bbg:#fbeee6;--a:#1f5fa8;--b:#b5521b;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#e8e6e1;--mut:#9a978f;--line:#34322e;--abg:#1d2a3a;--bbg:#3a271c;--a:#7fb0ea;--b:#f0a070;--card:#1f1e1c}}}}
body{{background:var(--bg);color:var(--fg);font:15px/1.55 -apple-system,"Apple SD Gothic Neo",system-ui,sans-serif;max-width:1000px;margin:0 auto;padding:16px}}
.case{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}} h3{{font-size:15px;margin:0 0 4px}} .meta{{color:var(--mut);font-size:13px;margin:0 0 6px}}
audio{{width:100%;max-width:560px;height:32px;margin:4px 0 8px}} .ln{{display:grid;grid-template-columns:22px 100px 1fr;gap:4px;padding:2px 4px;border-radius:4px;margin:2px 0;font-size:14px}}
.ln.s0{{background:var(--abg)}} .ln.s1{{background:var(--bbg)}} .tag{{font-weight:700}} .s0 .tag{{color:var(--a)}} .s1 .tag{{color:var(--b)}} .tm{{color:var(--mut);font-size:12px;font-variant-numeric:tabular-nums}}</style></head><body>
<h1>합성 2 인 대화 미리보기</h1><p class="meta">서로 다른 화자의 실제 발화를 턴으로 번갈아 놓은 합성 대화다. 오디오는 학습이 읽는 mono 캐시(화자별 크기 맞춤 + 섞은 뒤 잡음·잔향)의 앞부분이다.
"(조각)"은 긴 발화를 쉼에서 나눈 뒤쪽 조각(전사는 토큰으로만 있다).</p>{"".join(cards)}</body></html>'''
    open(os.path.join(a.out, "index.html"), "w", encoding="utf-8").write(page); print(f"→ {a.out}/index.html · 사례 {len(cards)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""코퍼스 정합성 프로브: 학습 로더가 만드는 스트림 오디오(assemble_stream)를 기준 모델(C2 final)로 전사해 manifest 타깃(lexical_text)과 WER/CER 을 코퍼스별로 비교.
새 코퍼스의 오디오–텍스트가 어긋나면(채널·오프셋·라벨 불일치) 그 코퍼스만 오류율이 튄다. 2026-09-09: D2 step 2000 평가 급락(dev-clean 0.34)의 원인 추적.
  CUDA_VISIBLE_DEVICES=1 python experiments/probe_corpus_sanity.py --manifests librispeech-960,voxpopuli-train,yodas-en129,kspon-full,nikl-1000,aihub71631-train,aihub-bc-train --n 12 --delay 4"""
import os, sys, json, random, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--manifests", required=True); ap.add_argument("--n", type=int, default=12); ap.add_argument("--delay", type=int, default=4)
ap.add_argument("--model", default="/soundai/Model/VAPASR/hf-C2/final"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--max-s", type=float, default=25.0); ap.add_argument("--show", type=int, default=2)
a = ap.parse_args()
MAN = os.environ.get("MXC_DATA_MANIFEST_DIR", "/soundai/users/tskim/VAPKT-data/data/manifests")
from vapasr.data.streams import assemble_stream
from vapasr.uslm.mono_data import lang_of
from vapasr.hf.infer import load_model, transcribe

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
def err(hyp, ref, lang): 
    if lang == "Korean": h, r = list(hyp.replace(" ", "")), list(ref.replace(" ", ""))
    else: h, r = hyp.split(), ref.split()
    return lev(h, r) / max(1, len(r))

model, tok = load_model(a.model, device="cuda"); print("model loaded", flush=True)
for m in a.manifests.split(","):
    lang = lang_of(m); rows = []
    with open(os.path.join(MAN, m, "streams.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r.get("mode", "stream") == "stream" and r["duration_s"] <= a.max_s: rows.append(r)
            if len(rows) >= 4000: break
    random.Random(a.seed).shuffle(rows); rows = rows[:a.n]; errs = []; t0 = time.time(); shown = 0
    for r in rows:
        wav = assemble_stream(r); ref = " ".join(s.get("lexical_text", s["text"]) for s in r["segments"])
        try: res = transcribe(model, tok, wav, lang=lang, delay=a.delay); hyp = res.text(tok)
        except Exception as e: print(f"  ! {r['id']}: {type(e).__name__}: {e}", flush=True); continue
        e = err(hyp, ref, lang); errs.append(e)
        if shown < a.show or e > 0.6:
            shown += 1; print(f"  [{r['id'][:48]}] {r['duration_s']:.1f}s err={e:.2f}\n     ref: {ref[:120]}\n     hyp: {hyp[:120]}", flush=True)
    if errs: print(f"== {m} ({lang}) n={len(errs)} mean err={sum(errs)/len(errs):.3f} median={sorted(errs)[len(errs)//2]:.3f} max={max(errs):.2f} ({time.time()-t0:.0f}s)", flush=True)

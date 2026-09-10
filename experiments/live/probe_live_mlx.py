#!/usr/bin/env python
"""MLX thinker 동일성·속도: 같은 prefix+오디오 청크 임베딩에 대해 torch(MPS) vs MLX logits 비교, 두 샘플의 스트리밍 전사와 청크당 시간.
  source scripts/local-env.sh; python experiments/live/probe_live_mlx.py [--mlx ~/Desktop/VAPKT-models/hf-E2-final-thinker-mlx]"""
import os, sys, time, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ap = argparse.ArgumentParser(); M = os.environ.get("VAPASR_LOCAL_MODELS", os.path.expanduser("~/Desktop/VAPKT-models"))
ap.add_argument("--ckpt", default=os.path.join(M, "hf-E2-final")); ap.add_argument("--mlx", default=os.path.join(M, "hf-E2-final-thinker-mlx")); a = ap.parse_args()
from vapasr.hf.infer import load_model, load_audio, prefix_ids
from vapasr.hf.live import LiveSession
from vapasr.hf.live_mlx import MlxThinker
import mlx.core as mx
model, tok = load_model(a.ckpt, device="mps", dtype=torch.float16); th = MlxThinker(a.mlx); print("loaded", a.mlx, flush=True)
# 1) logits 동일성: prefix + 임의 임베딩 3 개
pre = prefix_ids(model, tok, "Korean", 4); emb = model.get_input_embeddings()
with torch.inference_mode():
    from transformers import DynamicCache
    cache = DynamicCache(); e = emb(torch.tensor(pre, device="mps")); lt = model.thinker(inputs_embeds=e[None], past_key_values=cache, use_cache=True).logits[0, -1].float().cpu()
    lm = th.step(th.embed_ids(pre)); lm = torch.from_numpy(np.array(lm))
    print(f"prefix logits: max|diff| {float((lt-lm).abs().max()):.3f} · argmax torch {int(lt.argmax())} mlx {int(lm.argmax())} · top5 same {set(lt.topk(5).indices.tolist())==set(lm.topk(5).indices.tolist())}")
    torch.manual_seed(0); x = (torch.randn(3, 1024) * 0.5).to("mps", torch.float16)
    for i in range(3):
        lt = model.thinker(inputs_embeds=x[i][None, None], past_key_values=cache, use_cache=True).logits[0, -1].float().cpu(); lm = torch.from_numpy(np.array(th.step(mx.array(x[i].float().cpu().numpy()))))
        print(f"  step {i}: max|diff| {float((lt-lm).abs().max()):.3f} · argmax {int(lt.argmax())}/{int(lm.argmax())} · top5 same {set(lt.topk(5).indices.tolist())==set(lm.topk(5).indices.tolist())}")
# 2) 스트리밍 전사·속도
for name, lang in (("samples/en_1272-128104-0000.flac", "English"), ("samples/ko_KsponSpeech_620001.pcm", "Korean")):
    wav = load_audio(os.path.join(M, name)); out = {}
    for be in ("torch", "mlx"):
        s = LiveSession(model, tok, lang=lang, delay=4, mlx_thinker=th if be == "mlx" else None); ev = []; t = time.time()
        for i in range(0, len(wav), 1280): ev += s.feed(wav[i: i + 1280])
        ev += s.finish(); tk = [e.tick_ms for e in ev]; en = [e.enc_ms for e in ev]; de = [e.dec_ms for e in ev]; out[be] = s.text()
        print(f"[{lang}] {be:5s} {time.time()-t:.1f}s RTF {(time.time()-t)/(len(wav)/16000):.2f} · tick p50 {np.percentile(tk,50):.0f} ms (enc {np.percentile(en,50):.0f} / dec {np.percentile(de,50):.0f}) p90 {np.percentile(tk,90):.0f} · {s.text()}", flush=True)
    print(f"   same text: {out['torch'] == out['mlx']}")

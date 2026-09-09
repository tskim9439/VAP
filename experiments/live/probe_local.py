#!/usr/bin/env python
"""로컬(Mac MPS/CPU) 실시간 파이프라인 점검: 로컬 체크포인트 + 인코더 캐시로 모델을 올리고, 샘플 파일을 오프라인 transcribe 와 80 ms 스트리밍(LiveSession)으로 전사해 텍스트·청크당 시간을 본다.
  source scripts/local-env.sh; python experiments/live/probe_local.py [--device mps] [--dtype fp16]"""
import os, sys, time, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
ap = argparse.ArgumentParser(); ap.add_argument("--device", default="mps"); ap.add_argument("--dtype", default="fp16"); ap.add_argument("--models", default=os.environ.get("VAPASR_LOCAL_MODELS", os.path.expanduser("~/Desktop/VAPKT-models"))); a = ap.parse_args()
from vapasr.hf.infer import load_model, load_audio, transcribe
from vapasr.hf.live import LiveSession
DT = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}[a.dtype]
t0 = time.time(); model, tok = load_model(os.path.join(a.models, "hf-C2-final"), device=a.device, dtype=DT); print(f"model loaded {time.time()-t0:.0f}s device={a.device} thinker={DT or 'fp32'}", flush=True)
for name, lang in (("samples/en_1272-128104-0000.flac", "English"), ("samples/ko_KsponSpeech_620001.pcm", "Korean")):
    wav = load_audio(os.path.join(a.models, name)); t = time.time(); ref = transcribe(model, tok, wav, lang=lang, delay=4); t_off = time.time() - t
    print(f"[{lang}] {len(wav)/16000:.2f}s 오디오 · offline {t_off:.1f}s: {ref.text(tok)}", flush=True)
    s = LiveSession(model, tok, lang=lang, delay=4); ev = []; t = time.time()
    for i in range(0, len(wav), 1280): ev += s.feed(wav[i: i + 1280])
    ev += s.finish(); tk = [e.tick_ms for e in ev]; en = [e.enc_ms for e in ev]; de = [e.dec_ms for e in ev]
    print(f"[{lang}] live {time.time()-t:.1f}s (RTF {(time.time()-t)/(len(wav)/16000):.2f}): {s.text()}\n   tick p50 {np.percentile(tk,50):.0f} ms (enc {np.percentile(en,50):.0f} / dec {np.percentile(de,50):.0f}) p90 {np.percentile(tk,90):.0f} max {max(tk):.0f} · same={ref.text(tok)==s.text()}", flush=True)

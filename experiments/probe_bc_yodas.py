#!/usr/bin/env python
"""031/033(bc)·YODAS 스트림 오디오 통계(원본 sr·채널·길이, 로드 길이, RMS, 무음 비율)와 C2 전사 — 빈 출력의 원인(무음/다른 멤버/포맷) 판별."""
import os, sys, json, random, argparse, io
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, soundfile as sf
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); ap.add_argument("--model", default="/soundai/Model/VAPASR/hf-C2/final"); a = ap.parse_args()
MAN = os.environ.get("MXC_DATA_MANIFEST_DIR", "/soundai/users/tskim/VAPKT-data/data/manifests")
from vapasr.data.streams import load_seg_audio, SR
from vapasr.data.archive import read_member
from vapasr.hf.infer import load_model, transcribe
def rows(m, n, seed=0):
    out = []
    with open(os.path.join(MAN, m, "streams.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if 4 <= r["segments"][0]["dur_s"] <= 12: out.append(r)
            if len(out) >= 3000: break
    random.Random(seed).shuffle(out); return out[:n]
model, tok = load_model(a.model, device="cuda")
for m, lang in (("aihub-bc-train", "Korean"), ("yodas-en129", "English")):
    print(f"===== {m}", flush=True)
    for r in rows(m, a.n):
        s = r["segments"][0]; p = s["path"]; ref = s["lexical_text"]
        try:
            if "::" in p: arc, mem = p.split("::", 1); b = read_member(arc, mem); info = sf.info(io.BytesIO(b))
            else: arc, mem = os.path.dirname(p), p; info = sf.info(p.split("#ch")[0])
            w = load_seg_audio(s)
            frames = w[: len(w) // 400 * 400].reshape(-1, 400); fr = np.sqrt((frames ** 2).mean(1)); sil = float((fr < 1e-3).mean())
            h = transcribe(model, tok, w, lang=lang, delay=4).text(tok)
            print(f"[{s['utt_id']}] {os.path.basename(arc)}::{os.path.basename(mem)} src sr={info.samplerate} ch={info.channels} {info.duration:.2f}s {info.subtype} | manifest {s['dur_s']}s | loaded {len(w)/SR:.2f}s rms={float(np.sqrt(np.mean(w**2))):.4f} peak={float(np.abs(w).max()):.3f} silent_frames={sil:.2f}\n   ref: {ref[:100]}\n   hyp: {h[:100]}", flush=True)
        except Exception as e: print(f"[{s['utt_id']}] {p}: ERR {type(e).__name__}: {e}", flush=True)

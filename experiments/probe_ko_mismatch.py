#!/usr/bin/env python
"""71631·031/033 오디오–텍스트 불일치 추적(2026-09-09, D2 step 2000 급락). 71631: 같은 발화를 반대 채널·오프셋 이동으로 전사해 어느 쪽이 라벨과 맞는지.
bc: zip 멤버 중복(같은 stem 이 여러 zip 에)·오디오 통계(길이·RMS·sr)·전사."""
import os, sys, json, random, argparse, zipfile, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=8); ap.add_argument("--model", default="/soundai/Model/VAPASR/hf-C2/final"); a = ap.parse_args()
MAN = os.environ.get("MXC_DATA_MANIFEST_DIR", "/soundai/users/tskim/VAPKT-data/data/manifests")
from vapasr.data.streams import load_utt_audio, load_seg_audio, SR
from vapasr.hf.infer import load_model, transcribe
def rows(m, n, seed=0, max_s=12):
    out = []
    with open(os.path.join(MAN, m, "streams.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["duration_s"] <= max_s and r["segments"][0]["dur_s"] >= 2.0: out.append(r)
            if len(out) >= 3000: break
    random.Random(seed).shuffle(out); return out[:n]
def cer(h, r):
    h, r = list(h.replace(" ", "")), list(r.replace(" ", "")); prev = list(range(len(r) + 1))
    for i, x in enumerate(h, 1):
        cur = [i]
        for j, y in enumerate(r, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1] / max(1, len(r))
model, tok = load_model(a.model, device="cuda")
def tr(wav): return transcribe(model, tok, wav, lang="Korean", delay=4).text(tok)
print("===== aihub71631-train: 채널·오프셋 변형 전사", flush=True)
for r in rows("aihub71631-train", a.n):
    s = r["segments"][0]; path, ch = s["path"].rsplit("#ch", 1); ch = int(ch); off, dur = s["src_offset_s"], s["dur_s"]; ref = s["lexical_text"]
    variants = {"as-is": (ch, off), "other-ch": (1 - ch, off), "off-1s": (ch, max(0, off - 1)), "off+1s": (ch, off + 1)}
    res = {}
    for k, (c, o) in variants.items():
        try: w = load_utt_audio(f"{path}#ch{c}", o, dur + (2 if k.startswith("off") else 0)); res[k] = (tr(w), float(np.sqrt(np.mean(w ** 2))))
        except Exception as e: res[k] = (f"ERR {type(e).__name__}", 0.0)
    print(f"[{s['utt_id']}] ch{ch} off={off} dur={dur} ref: {ref}", flush=True)
    for k, (h, rms) in res.items(): print(f"   {k:9s} rms={rms:.3f} cer={cer(h, ref):.2f}  {h[:80]}", flush=True)
print("===== aihub-bc-train: zip 멤버·오디오 통계·전사", flush=True)
zips = sorted(glob.glob("/soundai/DB/raw/aihub/**/*.zip", recursive=True))
for r in rows("aihub-bc-train", a.n):
    s = r["segments"][0]; ref = s["lexical_text"]; p = s["path"]; arc, mem = p.split("::", 1)
    try:
        w = load_seg_audio(s); import soundfile as sf; from vapasr.data.archive import read_member; import io
        b = read_member(arc, mem); info = sf.info(io.BytesIO(b)); h = tr(w)
        print(f"[{s['utt_id']}] {os.path.basename(arc)}::{mem} src sr={info.samplerate} ch={info.channels} dur={info.duration:.2f}s | manifest dur={s['dur_s']} | loaded {len(w)/SR:.2f}s rms={float(np.sqrt(np.mean(w**2))):.3f}\n   ref: {ref[:90]}\n   hyp: {h[:90]} (cer {cer(h, ref):.2f})", flush=True)
    except Exception as e: print(f"[{s['utt_id']}] {p}: ERR {type(e).__name__}: {e}", flush=True)
    stem = os.path.splitext(os.path.basename(mem))[0]; hits = []
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf: hits += [f"{os.path.basename(z)}::{n}" for n in zf.namelist() if os.path.splitext(os.path.basename(n))[0] == stem]
        except Exception: pass
        if len(hits) > 4: break
    print(f"   같은 stem 멤버 {len(hits)}: {hits[:4]}", flush=True)

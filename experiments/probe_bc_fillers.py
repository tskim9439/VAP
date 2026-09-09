#!/usr/bin/env python
"""031/033 방송 라벨의 간투사·반복 생략률: Nemotron RNN-T(오프라인, 견고)로 발화를 전사해 라벨(ref)에 없는 간투사가 음성에 얼마나 있는지 센다. kspon 을 대조군으로."""
import os, sys, json, random, argparse, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); a = ap.parse_args()
MAN = os.environ.get("MXC_DATA_MANIFEST_DIR", "/soundai/users/tskim/VAPKT-data/data/manifests")
from vapasr.data.streams import load_seg_audio
import nemo.collections.asr as nemo_asr, glob
m = nemo_asr.models.ASRModel.restore_from(sorted(glob.glob(os.environ["MXC_NEMOTRON_DIR"] + "/*.nemo"))[0], map_location="cuda"); m.eval()
FILL = ["이제", "인제", "그", "막", "좀", "진짜", "어", "음", "아", "뭐", "약간", "그냥", "이렇게", "저기"]
def rows(mn, n):
    out = []
    with open(os.path.join(MAN, mn, "streams.jsonl")) as f:
        for line in f:
            r = json.loads(line); s = r["segments"][0]
            if 3 <= s["dur_s"] <= 12: out.append(s)
            if len(out) >= 3000: break
    random.Random(1).shuffle(out); return out[:n]
for mn in ("kspon-full", "aihub-bc-train", "aihub71631-train"):
    segs = rows(mn, a.n); wavs = [load_seg_audio(s) for s in segs]
    with torch.no_grad(): hyps = m.transcribe(wavs, batch_size=8, verbose=False)
    hyps = [h.text if hasattr(h, "text") else h for h in hyps]
    miss = collections.Counter(); extra_words = 0; ref_words = 0; hyp_words = 0
    for s, h in zip(segs, hyps):
        ref = s["lexical_text"].split(); hw = h.split(); ref_words += len(ref); hyp_words += len(hw)
        rc = collections.Counter(ref); hc = collections.Counter(hw)
        for w in FILL:
            if hc[w] > rc[w]: miss[w] += hc[w] - rc[w]
        extra_words += max(0, len(hw) - len(ref))
    print(f"== {mn}: n={len(segs)} ref 단어 {ref_words} · RNN-T 단어 {hyp_words} · 초과 {extra_words} ({extra_words/max(1,ref_words):.1%}) · 라벨에 없는 간투사: {dict(miss.most_common(8))}", flush=True)
    for s, h in list(zip(segs, hyps))[:4]: print(f"   ref: {s['lexical_text'][:90]}\n   rnnt: {h[:90]}", flush=True)

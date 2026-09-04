#!/usr/bin/env python
"""AuT 마스크 복원/변형별 (1) 실효 lookahead, (2) ASR 열화 (패치 전 전사 대비 WER) 측정.
실행: source scripts/activate-env.sh && CUDA_VISIBLE_DEVICES=3 python experiments/qwen_aut_mask_eval.py
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.qwen_aut_mask import patch_aut, unpatch_aut
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(False)
from qwen_asr import Qwen3ASRModel

AUDIO = os.path.join(os.environ.get("DATA_ROOT", "/data3/tskim"), "smoke", "libri1.wav")
CUTS = [3.5, 6.25, 9.75, 12.0]; REL_TOL = 1e-3

def wer(ref, hyp):
    r, h = ref.lower().split(), hyp.lower().split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=int); d[:, 0] = range(len(r) + 1); d[0, :] = range(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            d[i, j] = min(d[i-1, j] + 1, d[i, j-1] + 1, d[i-1, j-1] + (r[i-1] != h[j-1]))
    return d[-1, -1] / max(1, len(r))

m = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.float32, device_map="cuda", max_new_tokens=256)
root = next(v for v in vars(m).values() if isinstance(v, nn.Module)); enc = root.thinker.audio_tower.float().eval()
fe = m.processor.feature_extractor
y, sr = sf.read(AUDIO); y = y.astype(np.float32)
o = fe(y, sampling_rate=sr, return_tensors="pt", padding="longest", truncation=False, return_attention_mask=True)
feats = o.input_features[0].cuda(); T = int(o.attention_mask[0].sum()); feats = feats[:, :T]
mod = sys.modules[type(enc).__module__]; gl = getattr(enc, "_get_feat_extract_output_lengths", None) or mod._get_feat_extract_output_lengths

def encode(f):
    L = torch.tensor([f.shape[-1]], device="cuda")
    with torch.inference_mode(): r = enc(f.contiguous(), feature_lens=L, aftercnn_lens=gl(L))
    h = r.last_hidden_state if hasattr(r, "last_hidden_state") else r[0]
    return h.reshape(-1, h.shape[-1]).float()

def lookahead(encode_fn):
    h_full = encode_fn(feats); rows = []
    for c in CUTS:
        h_cut = encode_fn(feats[..., :int(c * 100)]); n = min(len(h_cut), len(h_full))
        rel = (h_cut[:n] - h_full[:n]).abs().amax(1) / (h_full.std() + 1e-8)
        ch = (rel > REL_TOL).nonzero().flatten(); first = int(ch[0]) if len(ch) else n
        rows.append((c, (n - first) * 80))
    return rows

def transcribe():
    with torch.inference_mode(): return m.transcribe(audio=AUDIO)[0].text

results = []
unpatch_aut(enc); base_txt = transcribe(); print("BASE (sdpa, 마스크 없음):", base_txt[:140])
configs = [
    ("none", dict()),
    ("block", dict(block_fbank_frames=100)),                     # 1 s 블록 — per-block 결과와 일치해야 패치 검증
    ("block", dict(block_fbank_frames=800)),                     # 8 s 블록 = 학습/추론 기본 의도
    ("chunked-causal", dict(block_fbank_frames=100)),            # 1 s chunk + 전체 좌측 context
    ("chunked-causal", dict(block_fbank_frames=100, left_ctx_blocks=7)),   # 좌측 7 s 한정 (학습 블록 8 s 유사)
    ("chunked-causal", dict(block_fbank_frames=200)),            # 2 s chunk
    ("causal", dict(block_fbank_frames=100)),                    # 프레임 causal (OOD)
]
for mode, kw in configs:
    patch_aut(enc, mode=mode, **kw)
    name = f"{mode}" + (f"[{kw.get('block_fbank_frames',0)*10} ms" + (f", left {kw['left_ctx_blocks']} blk" if 'left_ctx_blocks' in kw else "") + "]" if kw else "")
    t = time.time(); rows = lookahead(encode); txt = transcribe(); w = wer(base_txt, txt)
    la = [r[1] for r in rows]
    print(f"\n[{name}]  lookahead ms @cuts {[(c, int(l)) for c, l in rows]}  → min {min(la):.0f} / mean {np.mean(la):.0f} / max {max(la):.0f}")
    print(f"  WER vs base = {w*100:.1f}%   ({time.time()-t:.1f}s)\n  → {txt[:140]}")
    results.append(dict(mode=name, lookahead_ms=la, lookahead_mean=float(np.mean(la)), lookahead_max=float(max(la)), wer_vs_base=w, text=txt))
unpatch_aut(enc)
print("\n" + "=" * 90); print(f"{'mode':42s} {'lookahead mean/max':>22s} {'WER vs base':>12s}")
for r in results: print(f"{r['mode']:42s} {r['lookahead_mean']:9.0f} / {r['lookahead_max']:<9.0f} {r['wer_vs_base']*100:10.1f}%")
out = os.path.join(os.environ.get("DATA_LOG_DIR", "/tmp"), f"qwen-aut-mask-{time.strftime('%Y%m%d-%H%M')}.json")
json.dump(dict(audio=AUDIO, base_text=base_txt, cuts=CUTS, results=results), open(out, "w"), indent=1, ensure_ascii=False); print("saved", out)

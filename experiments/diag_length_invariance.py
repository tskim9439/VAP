#!/usr/bin/env python
"""인코더 출력이 입력 총 길이에 따라 바뀌는지(길이 불변성) fn 수준에서 진단. 세그먼트 캐시 설계 근거."""
import sys, os, numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.features import load_encoder
w = sorted(os.listdir("/data3/tskim/corpora/aihub/adult-vs02-wav"))[0]
x, sr = sf.read("/data3/tskim/corpora/aihub/adult-vs02-wav/" + w, dtype="float32", always_2d=True); x = torch.from_numpy(x.T).cuda(); T = x.shape[1] / sr
print(f"len: 파일 {T:.1f} s", flush=True)
for name in sys.argv[1:] or ["cpc", "nemotron-c0"]:
    enc = load_encoder(name); fn = enc._fn; hz = enc.frame_hz
    with torch.inference_mode():
        full = fn(x).float().cpu().numpy(); std = np.abs(full).std()
        for L in (60, 120, 150, 200, 250):
            if L >= T: continue
            h = fn(x[:, : sr * L]).float().cpu().numpy(); n = h.shape[1] - 2; r = np.abs(full[:, :n] - h[:, :n]).max(-1) / std
            bad = np.flatnonzero((r > 1e-3).any(0)); print(f"{name}: fn(x[:{L}s]) vs full({T:.0f}s): rel>1e-3 {len(bad)}/{n}  max {r.max():.2e}  첫 불일치 {round(bad[0]/hz,1) if len(bad) else None}s", flush=True)
        a = fn(x[:, : sr * 150]).float().cpu().numpy(); b = fn(x[:, : sr * 200]).float().cpu().numpy(); n = a.shape[1] - 2
        r = np.abs(a[:, :n] - b[:, :n]).max(-1) / std; print(f"{name}: 150s vs 200s: rel>1e-3 {int((r > 1e-3).any(0).sum())}/{n} max {r.max():.2e}", flush=True)
    if name.startswith("nemotron"):
        import nemo.collections.asr as nemo_asr
        m = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu"); c = m.cfg.encoder; e = m.encoder
        for k in ("att_context_style", "att_context_size", "att_context_probs", "pos_emb_max_len", "xscaling", "conv_context_size", "conv_kernel_size", "subsampling", "causal_downsampling", "use_pytorch_sdpa", "n_layers", "d_model"):
            print("cfg", k, "=", c.get(k, "?"))
        print("cfg max_audio_length:", getattr(e, "max_audio_length", "?"), "| pos_enc:", type(getattr(e, "pos_enc", None)).__name__)
    del enc; torch.cuda.empty_cache()

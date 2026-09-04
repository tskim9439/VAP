#!/usr/bin/env python
"""두 backbone 을 실제로 로드해 추론하고, 인코더 frame rate·lookahead 설정을 보고한다.
컨테이너 안에서:  source scripts/activate-env.sh && python scripts/smoke-test-models.py [nemotron|qwen|aligner ...]
"""
import os, sys, time, traceback
import numpy as np

steps = sys.argv[1:] or ["audio", "nemotron", "qwen", "aligner", "vap"]
audio_path = os.path.join(os.environ.get("DATA_ROOT", "/tmp"), "smoke", "libri1.wav")

def hdr(t): print(f"\n=== {t} ===", flush=True)

def step_audio():
    hdr("샘플 오디오 준비")
    import librosa, soundfile as sf
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)
    if not os.path.exists(audio_path):
        y, sr = librosa.load(librosa.example("libri1"), sr=16000, mono=True)
        sf.write(audio_path, y, 16000)
    y, sr = sf.read(audio_path)
    print(f"{audio_path}  {len(y)/sr:.2f}s @ {sr} Hz")

def step_nemotron():
    hdr("Nemotron 3.5 ASR streaming 0.6B (NeMo)")
    import json, torch, nemo, nemo.collections.asr as nemo_asr, soundfile as sf
    print("nemo", nemo.__version__)
    y, sr = sf.read(audio_path)
    # 언어 전달: transcribe 의 target_lang 키워드는 dataset 까지 도달하지 않는다 (NeMo 3.0/3.1 공통).
    # lhotse NeMo 어댑터가 manifest 의 "lang" 필드를 supervision.language 로 매핑하므로 manifest 로 준다.
    mf = audio_path.replace(".wav", "_manifest.json")
    with open(mf, "w") as f:
        f.write(json.dumps({"audio_filepath": audio_path, "duration": len(y)/sr, "text": "", "lang": "en-US"}) + "\n")
    t = time.time()
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b").cuda().eval()
    print(f"로드 {time.time()-t:.1f}s  params {sum(p.numel() for p in m.parameters())/1e6:.0f}M")
    enc = m.encoder
    print("encoder:", type(enc).__name__, "| 기본 att_context_size:", getattr(enc, "att_context_size", "?"),
          "| subsampling_factor:", getattr(enc, "subsampling_factor", "?"), "| d_model:", getattr(enc, "d_model", "?"))
    for ctx in ([56, 0], [56, 3]):          # 80 ms(우측 0) vs 320 ms(우측 3)
        enc.set_default_att_context_size(ctx)
        t = time.time()
        with torch.inference_mode():
            out = m.transcribe(mf, batch_size=1, verbose=False)
        txt = out[0].text if hasattr(out[0], "text") else out[0]
        print(f"ctx={ctx} transcribe {time.time()-t:.1f}s →", str(txt)[:110])
    x = torch.tensor(y, dtype=torch.float32)[None].cuda()
    with torch.inference_mode():
        feats, flen = m.preprocessor(input_signal=x, length=torch.tensor([x.shape[1]]).cuda())
        h, hlen = enc(audio_signal=feats, length=flen)
    ms = len(y)/sr/h.shape[-1]*1000
    print(f"encoder out {tuple(h.shape)} for {len(y)/sr:.2f}s → {ms:.1f} ms/frame")
    print("cache-aware API:", [n for n in dir(enc) if "cache" in n.lower() and not n.startswith("_")][:8])

def step_qwen():
    hdr("Qwen3-ASR-0.6B (qwen-asr, transformers backend)")
    import torch
    from qwen_asr import Qwen3ASRModel
    t = time.time()
    m = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.bfloat16, device_map="cuda:0", max_new_tokens=128)
    print(f"로드 {time.time()-t:.1f}s")
    t = time.time(); r = m.transcribe(audio=audio_path)
    print(f"transcribe {time.time()-t:.1f}s →", r[0].text[:120])
    import torch.nn as nn
    root = next((v for v in vars(m).values() if isinstance(v, nn.Module)), None) or m
    cands = [(n, mod) for n, mod in root.named_modules() if n.count(".") <= 2 and any(k in n.lower() for k in ("audio", "encoder", "aut"))]
    for n, mod in cands[:6]:
        print(f"  module {n or '<root>'}: {type(mod).__name__}  {sum(p.numel() for p in mod.parameters())/1e6:.0f}M")
    top = [n for n, _ in root.named_children()]; print("  top-level children:", top)

def step_aligner():
    hdr("Qwen3-ForcedAligner-0.6B (한국어 포함 11개 언어)")
    import torch
    from qwen_asr import Qwen3ForcedAligner
    a = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda:0")
    r = a.align(audio=audio_path, text="he hoped there would be stew for dinner turnips and carrots and bruised potatoes", language="English")
    items = r[0].timestamps if hasattr(r[0], "timestamps") else r
    print("align items:", len(items)); print("first 3:", [str(i) for i in list(items)[:3]])

def step_vap():
    hdr("원 VAP (CPC, 50 Hz) — /data3/tskim/third_party/VoiceActivityProjection")
    import torch, soundfile as sf, glob
    repo = os.path.join(os.environ.get("DATA_ROOT", "/data3/tskim"), "third_party", "VoiceActivityProjection")
    ckpt = sorted(glob.glob(os.path.join(repo, "example", "VAP_*.ckpt")))[-1]
    os.chdir(repo)  # 체크포인트 내부의 상대 경로(cpc 가중치 등) 해석용
    from vap.model import VapConfig, VapGPT, load_older_state_dict
    m = VapGPT(VapConfig())
    missing, unexpected = m.load_state_dict(load_older_state_dict(ckpt), strict=False)
    m = m.cuda().eval()
    print("ckpt:", os.path.basename(ckpt), f"| params {sum(p.numel() for p in m.parameters())/1e6:.1f}M",
          f"| missing {len(missing)} unexpected {len(unexpected)}")
    print("encoder:", type(m.encoder).__name__ if hasattr(m, "encoder") else "?",
          "| frame_hz:", getattr(m, "frame_hz", getattr(VapConfig(), "frame_hz", "?")))
    y, sr = sf.read(audio_path); y = torch.tensor(y, dtype=torch.float32)
    wav = torch.stack([y, torch.zeros_like(y)])[None].cuda()   # (1, 2ch, T): 화자 A 만 발화
    with torch.inference_mode():
        out = m.probs(wav) if hasattr(m, "probs") else m(waveform=wav)
    keys = list(out.keys()) if isinstance(out, dict) else type(out)
    print("output keys:", keys)
    for k in ("p_now", "p_future", "p_all", "probs", "logits", "vad"):
        if isinstance(out, dict) and k in out:
            t = out[k]; print(f"  {k}: {tuple(t.shape)} → {y.shape[0]/sr/t.shape[1]*1000:.1f} ms/frame")

for s in steps:
    try: globals()[f"step_{s}"]()
    except Exception: print(f"!! {s} 실패"); traceback.print_exc()
print("\n완료")

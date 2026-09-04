#!/usr/bin/env python
"""U0.5 기준선 WER/CER — 같은 val 발화(대화 단위 분할)에서 (a) 원본 Qwen3-ASR(AuT+thinker, 오프라인) (b) Nemotron RNN-T.
python experiments/u05_baselines.py [--max-utts 400] → $CKPT_EXP_DIR/uslm/baselines.json
"""
import os, sys, json, time, argparse, tempfile
import numpy as np, soundfile as sf, torch, jiwer
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.uslm import UttFeatureDataset, normalize_text
from vapasr.uslm.data import AUDIO_ROOT
ap = argparse.ArgumentParser(); ap.add_argument("--manifests", default="otoSpeech,aihub-ts01-5,aihub-vs02"); ap.add_argument("--max-utts", type=int, default=400); ap.add_argument("--which", default="qwen,nemotron"); ap.add_argument("--ctx", type=int, default=0, help="Nemotron att_context 우측 프레임 (0=80 ms, 13=1120 ms)"); a = ap.parse_args()
out = os.path.join(os.environ.get("CKPT_EXP_DIR", "/tmp"), "uslm"); os.makedirs(out, exist_ok=True)
sets = {m: UttFeatureDataset([m], split="val", max_utts=a.max_utts) for m in a.manifests.split(",")}
_audio = {}
def audio(u):
    name = u["manifest"]; key = (name, u["conv"], u["speaker"])
    if key not in _audio:
        if name.startswith("aihub"): x, sr = sf.read(os.path.join(AUDIO_ROOT[name], u["conv"] + ".wav"), dtype="float32", always_2d=True); x = x[:, u["speaker"]]
        else: x, sr = sf.read(os.path.join(AUDIO_ROOT[name], u["conv"].split("oto-")[1], f"speaker_{u['speaker']+1}_audio.wav"), dtype="float32")
        _audio.clear(); _audio[key] = (x, sr)
    x, sr = _audio[key]; seg = x[int(u["start"] * sr): int(u["end"] * sr)]
    return (seg if len(seg) >= int(0.1 * sr) else None), sr
def score(refs, hyps, lang):
    R = [normalize_text(r, lang) for r in refs]; H = [normalize_text(h, lang) for h in hyps]
    if lang == "Korean": return dict(cer=jiwer.cer(R, H), n=len(R))
    return dict(wer=jiwer.wer(R, H), n=len(R))
res = {}
if "qwen" in a.which:
    from qwen_asr import Qwen3ASRModel
    m = Qwen3ASRModel.from_pretrained("Qwen/Qwen3-ASR-0.6B", dtype=torch.bfloat16, device_map="cuda", max_new_tokens=256)
    for name, ds in sets.items():
        refs, hyps, t0 = [], [], time.time()
        for i in range(len(ds)):
            it = ds.items[i][3] | {"manifest": name}; x, sr = audio(it)
            if x is None: continue
            try: r = m.transcribe(audio=(x, sr), language=("Korean" if name.startswith("aihub") else "English")); hyps.append(r[0].text); refs.append(it["text"])
            except Exception as ex: print("skip", it["conv"], it["start"], type(ex).__name__, flush=True)
        res[f"qwen3-asr/{name}"] = score(refs, hyps, "Korean" if name.startswith("aihub") else "English") | dict(sec=time.time() - t0, example=[refs[0][:40], hyps[0][:40]])
        print(name, "qwen3-asr", res[f"qwen3-asr/{name}"], flush=True)
    del m; torch.cuda.empty_cache()
if "nemotron" in a.which:
    import nemo.collections.asr as nemo_asr
    m = nemo_asr.models.ASRModel.from_pretrained("nvidia/nemotron-3.5-asr-streaming-0.6b", map_location="cpu").cuda().eval()
    m.encoder.set_default_att_context_size([56, a.ctx])   # 0 = 80 ms 스트리밍, 13 = 1120 ms
    for name, ds in sets.items():
        refs, hyps, t0 = [], [], time.time(); tmpd = tempfile.mkdtemp(); mf = os.path.join(tmpd, "m.json"); lang = "ko-KR" if name.startswith("aihub") else "en-US"
        with open(mf, "w") as f:
            for i in range(len(ds)):
                it = ds.items[i][3] | {"manifest": name}; x, sr = audio(it)
                if x is None: continue
                p = os.path.join(tmpd, f"{i}.wav"); sf.write(p, x, sr)
                f.write(json.dumps(dict(audio_filepath=p, duration=len(x) / sr, text="", lang=lang)) + "\n"); refs.append(it["text"])
        with torch.inference_mode(): outs = m.transcribe(mf, batch_size=16, verbose=False)
        hyps = [o.text if hasattr(o, "text") else o for o in outs]
        res[f"nemotron-rnnt[56,{a.ctx}]/{name}"] = score(refs, hyps, "Korean" if name.startswith("aihub") else "English") | dict(sec=time.time() - t0, example=[refs[0][:40], hyps[0][:40]])
        print(name, "nemotron", res[f"nemotron-rnnt[56,{a.ctx}]/{name}"], flush=True)
p = os.path.join(out, "baselines.json"); prev = json.load(open(p)) if os.path.exists(p) else {}; prev.update(res); json.dump(prev, open(p, "w"), indent=1, ensure_ascii=False); print("saved", p)

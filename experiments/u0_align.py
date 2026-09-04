#!/usr/bin/env python
"""U0 — Qwen3-ForcedAligner 로 발화 단위 토큰 시각(80 ms 격자) 생성 → BPE 토큰별 종료 시각 manifest.
python experiments/u0_align.py --corpus otoSpeech --root /data3/tskim/corpora/turnbench/otoSpeech16k --out $DATA_MANIFEST_DIR/align/otoSpeech [--limit-utts 200]
python experiments/u0_align.py --corpus aihub --root /data3/tskim/corpora/aihub/adult-vs02-wav --label-root /data3/tskim/corpora/aihub/adult-val-labels --out ... 
출력: <out>/<conv id>.jsonl (한 줄 = 발화: speaker, start, end, text, tokens[{id, text, end_time}]) + stats.json (aligner 시각 vs 라벨 경계 오차, 속도)
"""
import os, sys, json, glob, time, argparse
import numpy as np, soundfile as sf, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.corpora import parse_srt, SPEECH_LABELS
ap = argparse.ArgumentParser(); ap.add_argument("--corpus", required=True); ap.add_argument("--root", required=True); ap.add_argument("--label-root", default=None)
ap.add_argument("--out", required=True); ap.add_argument("--limit-utts", type=int, default=None); ap.add_argument("--limit-convs", type=int, default=None); ap.add_argument("--min-dur", type=float, default=0.3)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
from qwen_asr import Qwen3ForcedAligner
from transformers import AutoTokenizer
aligner = Qwen3ForcedAligner.from_pretrained("Qwen/Qwen3-ForcedAligner-0.6B", dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B")
LANG = {"aihub": "Korean", "otoSpeech": "English"}[a.corpus]
def _f(x): return float(str(x).replace(",", ""))

def utterances(conv_id, wav_paths):
    """yield (speaker, start, end, text) — aihub: JSON, otoSpeech: SRT"""
    if a.corpus == "aihub":
        js = glob.glob(os.path.join(a.label_root, "**", conv_id + ".json"), recursive=True)
        if not js: return
        d = json.load(open(js[0], encoding="utf-8")); swap = False   # 화자↔채널 판정은 manifest meta 를 따르는 게 정확하나 여기선 라벨 기준(SpeakerNo)
        for u in d.get("Conversation", []):
            s, e, t = _f(u["StartTime"]), _f(u["EndTime"]), u.get("Text", "").strip()
            if e - s >= a.min_dur and t: yield (0 if str(u["SpeakerNo"]).endswith("1") else 1, s, e, t)
    else:
        for c in (0, 1):
            for s, e, lab, t in parse_srt(os.path.join(wav_paths[c].rsplit("/", 1)[0], f"speaker_{c+1}_annotation_a.srt")):
                if lab in SPEECH_LABELS and e - s >= a.min_dur and t.strip(): yield (c, s, e, t.strip())

def align_tokens(audio, sr, text):
    """aligner 항목(단어/문자 시각) → BPE 토큰 종료 시각. offsets 로 문자 구간을 잇는다."""
    r = aligner.align(audio=(audio, sr) if False else _tmpwav(audio, sr), text=text, language=LANG)
    items = r[0].items if hasattr(r[0], "items") else r
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); ids, offs = enc["input_ids"], enc["offset_mapping"]
    # 각 aligner 항목의 문자 구간 찾기 (순차 탐색)
    spans = []; cur = 0
    for it in items:
        s = text.find(it.text, cur)
        if s < 0: s = cur
        spans.append((s, s + len(it.text), float(it.end_time), float(it.start_time))); cur = s + len(it.text)
    out = []
    for tid, (o0, o1) in zip(ids, offs):
        cover = [sp for sp in spans if sp[0] < o1 and sp[1] > o0]
        t_end = max(c[2] for c in cover) if cover else (out[-1]["end_time"] if out else 0.0)
        out.append(dict(id=int(tid), text=text[o0:o1], end_time=t_end))
    return out

import tempfile
_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
def _tmpwav(audio, sr):
    sf.write(_tmp, audio, sr); return _tmp

# 대화 목록
if a.corpus == "aihub": convs = [(os.path.splitext(os.path.basename(w))[0], [w, w]) for w in sorted(glob.glob(os.path.join(a.root, "*.wav")))]
else: convs = [(f"oto-{os.path.basename(d.rstrip('/'))}", [os.path.join(d, "speaker_1_audio.wav"), os.path.join(d, "speaker_2_audio.wav")]) for d in sorted(glob.glob(os.path.join(a.root, "*/")))]
convs = convs[: a.limit_convs] if a.limit_convs else convs
st = dict(convs=0, utts=0, tokens=0, sec=0.0, onset_err_ms=[], offset_err_ms=[], fail=0); t0 = time.time(); n_utt = 0
for cid, wavs in convs:
    outp = os.path.join(a.out, cid + ".jsonl")
    if os.path.exists(outp): continue
    audio = {}
    with open(outp + ".tmp", "w") as f:
        for spk, s, e, text in utterances(cid, wavs):
            if spk not in audio:
                x, sr = sf.read(wavs[spk] if a.corpus != "aihub" else wavs[0], dtype="float32", always_2d=True); audio[spk] = (x[:, spk if a.corpus == "aihub" else 0], sr)
            x, sr = audio[spk]; seg = x[int(s * sr): int(e * sr)]
            try:
                toks = align_tokens(seg, sr, text)
            except Exception as ex:
                st["fail"] += 1; continue
            for t in toks: t["end_time"] = round(s + t["end_time"], 3)      # 대화 절대 시각
            f.write(json.dumps(dict(speaker=spk, start=s, end=e, text=text, tokens=toks), ensure_ascii=False) + "\n")
            st["utts"] += 1; st["tokens"] += len(toks); n_utt += 1
            if toks: st["offset_err_ms"].append((e - toks[-1]["end_time"]) * 1000)       # 라벨 끝 − 마지막 토큰 끝
            if a.limit_utts and n_utt >= a.limit_utts: break
    os.replace(outp + ".tmp", outp); st["convs"] += 1
    print(f"  {cid}: utts {st['utts']} tokens {st['tokens']} {time.time()-t0:.0f}s", flush=True)
    if a.limit_utts and n_utt >= a.limit_utts: break
st["sec"] = time.time() - t0; st["utts_per_sec"] = st["utts"] / max(1e-9, st["sec"])
for k in ("onset_err_ms", "offset_err_ms"):
    v = np.array(st[k]); st[k] = dict(n=len(v), median=float(np.median(v)) if len(v) else None, p90=float(np.percentile(v, 90)) if len(v) else None)
json.dump(st, open(os.path.join(a.out, "stats.json"), "w"), indent=1); print(json.dumps(st, ensure_ascii=False))

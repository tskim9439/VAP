#!/usr/bin/env python
"""Stage 1 정렬 — streams.jsonl 의 발화마다 Qwen3-ForcedAligner 로 BPE 토큰 종료 시각(80 ms 격자)을 만든다 (plans/stage1-mono-pilot.md §3.4-(2)).

python experiments/s1_align.py --manifest librispeech-100 [--mode stream|utt|all] [--limit N] [--ids a,b] [--gpu K]
출력: $MXC_DATA_MANIFEST_DIR/align/<manifest>/<stream id>.jsonl  — 한 줄 = 발화 {speaker:0, start, end, text, tokens[{id,text,end_time}]} (스트림 절대 시각)
      + stats.json.  u0_align.py 와 같은 스키마라 u0_align_qc.py 와 interleave 빌더가 그대로 읽는다. 재개 가능(파일 존재 시 건너뜀), 동시 실행 안전(pid tmp).
모델은 로컬 디렉토리($MXC_ALIGNER_DIR, 토크나이저는 $MXC_QWEN_ASR_DIR)에서 읽는다. 발화 오디오는 원본 파일에서 직접(조립 스트림 아님) 읽고 offset_s 를 더한다.
"""
import os, sys, json, time, argparse, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--mode", default="all"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--ids", default=None)
ap.add_argument("--gpu", default=None); ap.add_argument("--min-dur", type=float, default=0.3)
a = ap.parse_args()
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if a.gpu is None:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        rows = [[int(v) for v in l.split(",")] for l in q.stdout.strip().splitlines()]; a.gpu = str(max(rows, key=lambda r: r[2] - r[1])[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import numpy as np, torch, soundfile as sf
from qwen_asr import Qwen3ForcedAligner
from transformers import AutoTokenizer
from vapasr.data.streams import read_streams, load_utt_audio, iter_utterances
from vapasr.data.kspon import SR

MAN = os.environ.get("MXC_DATA_MANIFEST_DIR", os.environ.get("DATA_MANIFEST_DIR", "/tmp"))
ALIGNER = os.environ.get("MXC_ALIGNER_DIR", "Qwen/Qwen3-ForcedAligner-0.6B"); QWEN = os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B")
mdir = a.manifest if os.path.isdir(a.manifest) else os.path.join(MAN, a.manifest); mname = os.path.basename(mdir.rstrip("/"))
out = os.path.join(MAN, "align", mname); os.makedirs(out, exist_ok=True)
rows = read_streams(mdir, mode=None if a.mode == "all" else a.mode)
if a.ids: keep = set(a.ids.split(",")); rows = [r for r in rows if r["id"] in keep]
rows = rows[: a.limit] if a.limit else rows
print(f"align ← {mname} ({a.mode}): {len(rows)} 스트림 → {out}  [GPU {os.environ['CUDA_VISIBLE_DEVICES']}]", flush=True)

aligner = Qwen3ForcedAligner.from_pretrained(ALIGNER, dtype=torch.bfloat16, device_map="cuda"); tok = AutoTokenizer.from_pretrained(QWEN)
_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

def align_tokens(audio, text, lang):
    """aligner 항목(단어/문자 시각) → BPE 토큰 종료 시각. offsets 로 문자 구간을 잇는다 (u0_align.py 와 동일 로직)."""
    sf.write(_tmp, audio, SR); r = aligner.align(audio=_tmp, text=text, language=lang)
    items = r[0].items if hasattr(r[0], "items") else r
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); ids, offs = enc["input_ids"], enc["offset_mapping"]
    spans, cur = [], 0
    for it in items:
        s = text.find(it.text, cur)
        if s < 0: s = cur
        spans.append((s, s + len(it.text), float(it.end_time))); cur = s + len(it.text)
    outp = []
    for tid, (o0, o1) in zip(ids, offs):
        cover = [sp for sp in spans if sp[0] < o1 and sp[1] > o0]
        t_end = max(c[2] for c in cover) if cover else (outp[-1]["end_time"] if outp else 0.0)
        outp.append(dict(id=int(tid), text=text[o0:o1], end_time=t_end))
    return outp

st = dict(streams=0, utts=0, tokens=0, fail=0, offset_err_ms=[], sec=0.0); T0 = time.time(); audio_cache = {}
for r in rows:
    outp = os.path.join(out, r["id"] + ".jsonl")
    if os.path.exists(outp): continue
    tmpp = outp + f".{os.getpid()}.tmp"
    with open(tmpp, "w", encoding="utf-8") as f:
        for u in iter_utterances(r):
            if u["end"] - u["start"] < a.min_dur or not u["text"]: continue
            x = audio_cache.get(u["path"]);
            if x is None:
                x = load_utt_audio(u["path"]); audio_cache[u["path"]] = x
                if len(audio_cache) > 256: audio_cache.clear(); audio_cache[u["path"]] = x
            try: toks = align_tokens(x, u["text"], r["lang"])
            except Exception as ex: st["fail"] += 1; print(f"  ! {r['id']} {u['utt_id']}: {type(ex).__name__}: {str(ex)[:80]}", flush=True); continue
            for t in toks: t["end_time"] = round(u["start"] + t["end_time"], 3)      # 스트림 절대 시각
            f.write(json.dumps(dict(speaker=0, start=u["start"], end=u["end"], text=u["text"], tokens=toks), ensure_ascii=False) + "\n")
            st["utts"] += 1; st["tokens"] += len(toks)
            if toks: st["offset_err_ms"].append((u["end"] - toks[-1]["end_time"]) * 1000)
    if os.path.exists(outp): os.remove(tmpp); continue
    os.replace(tmpp, outp); st["streams"] += 1
    if st["streams"] % 200 == 0: print(f"  {st['streams']} 스트림 · {st['utts']} 발화 · {st['tokens']} 토큰 · 실패 {st['fail']} · {time.time()-T0:.0f}s", flush=True)
st["sec"] = time.time() - T0; v = np.array(st["offset_err_ms"]); st["offset_err_ms"] = dict(n=len(v), median=float(np.median(v)) if len(v) else None, p90=float(np.percentile(v, 90)) if len(v) else None)
json.dump(st, open(os.path.join(out, f"stats-{a.mode}.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(st, ensure_ascii=False))

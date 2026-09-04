#!/usr/bin/env python
"""frozen encoder 특징을 대화별 fp16 npy 로 캐시한다 (재개 가능). RTF·GPU 메모리 기록.
python experiments/extract_features.py --encoder cpc --manifest $DATA_MANIFEST_DIR/otoSpeech --audio-root /data3/tskim/corpora/turnbench/otoSpeech16k [--limit-hours 50] [--limit N]
출력: $DATA_FEATURE_CACHE_DIR/<encoder>/<manifest name>/<id>.npy  (2, T', D) fp16 + index.jsonl + stats.json
"""
import os, sys, json, time, argparse
import numpy as np, torch, soundfile as sf
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.features import load_encoder

ap = argparse.ArgumentParser(); ap.add_argument("--encoder", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--audio-root", required=True)
ap.add_argument("--limit-hours", type=float, default=None); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--include-flagged", action="store_true")
ap.add_argument("--seg-s", type=float, default=None, help="세그먼트 길이 오버라이드 (GPU 메모리 부족 시 줄인다)"); ap.add_argument("--ids", default=None, help="쉼표 구분 id 만")
ap.add_argument("--device", default="cuda", help="cuda | cpu (cpc/fbank 처럼 가벼운 인코더는 cpu 가능)")
a = ap.parse_args()
mname = os.path.basename(a.manifest.rstrip("/")); out = os.path.join(os.environ.get("DATA_FEATURE_CACHE_DIR", "/data3/tskim/features"), a.encoder, mname)
os.makedirs(out, exist_ok=True)
st = json.load(open(os.path.join(a.manifest, "stats.json"))) if os.path.exists(os.path.join(a.manifest, "stats.json")) else {}
flagged = set() if a.include_flagged else set(st.get("flagged", {}).keys())
rows = [json.loads(l) for l in open(os.path.join(a.manifest, "manifest.jsonl"))]
rows = [r for r in rows if r["id"] not in flagged]
if a.limit_hours:   # 결정적 서브셋: id 정렬 후 누적 시간 컷
    rows.sort(key=lambda r: r["id"]); acc, keep = 0.0, []
    for r in rows:
        if acc >= a.limit_hours * 3600: break
        keep.append(r); acc += r["duration"]
    rows = keep
rows = rows[: a.limit] if a.limit else rows
if a.ids: keep = set(a.ids.split(",")); rows = [r for r in rows if r["id"] in keep]
print(f"{a.encoder} ← {mname}: {len(rows)} conv, {sum(r['duration'] for r in rows)/3600:.1f} h → {out}", flush=True)

_TB = {}
def _tb_audio(cid):
    """TurnBench dev parquet 에서 conversation_id 의 두 채널 FLAC 을 디코드 (첫 호출 시 shard 인덱스 구축)."""
    import io, glob, pyarrow.parquet as pq
    if not _TB:
        snaps = sorted(glob.glob(os.path.join(a.audio_root, "*", "data"))); files = sorted(glob.glob(os.path.join(snaps[-1] if snaps else a.audio_root, "**", "*.parquet"), recursive=True))
        for f in files:
            t = pq.read_table(f, columns=["conversation_id"]);
            for i, c in enumerate(t.column("conversation_id").to_pylist()): _TB[c] = (f, i)
    f, i = _TB[cid]; row = pq.read_table(f).slice(i, 1).to_pylist()[0]; chans = []
    for c in (1, 2):
        y, sr = sf.read(io.BytesIO(row[f"speaker_{c}_audio"]["bytes"]), dtype="float32"); chans.append(y)
    n = min(len(c) for c in chans); return np.stack([c[:n] for c in chans]), sr

def load_audio(r):
    if r["source"] == "aihub":
        x, sr = sf.read(os.path.join(a.audio_root, r["id"] + ".wav"), dtype="float32", always_2d=True); x = x.T
    elif r["source"] == "otoSpeech":
        d = os.path.join(a.audio_root, r["id"].split("oto-")[1]); chans = []
        for c in (1, 2):
            y, sr = sf.read(os.path.join(d, f"speaker_{c}_audio.wav"), dtype="float32"); chans.append(y)
        n = min(len(c) for c in chans); x = np.stack([c[:n] for c in chans])
    elif r["source"] == "turnbench-dev":
        x, sr = _tb_audio(r["id"].split("tb-")[1])
    else: raise ValueError(r["source"])
    if sr != 16000:
        import soxr; x = np.stack([soxr.resample(c, sr, 16000) for c in x])
    return x

enc = load_encoder(a.encoder)
if a.seg_s: enc.seg_s = a.seg_s
if a.device != "cuda": enc.to(a.device)
print(f"encoder {enc.name}: {enc.frame_hz} Hz, D={enc.dim}, lookahead {enc.lookahead_ms} ms, causal={enc.causal}", flush=True)
idx = open(os.path.join(out, "index.jsonl"), "a"); done = {json.loads(l)["id"] for l in open(os.path.join(out, "index.jsonl"))} if os.path.getsize(os.path.join(out, "index.jsonl")) else set()
t_audio, t_enc, t_io, hours, n = 0.0, 0.0, 0.0, 0.0, 0; torch.cuda.reset_peak_memory_stats(); T0 = time.time()
for r in rows:
    if r["id"] in done: continue
    p = os.path.join(out, r["id"] + ".npy")
    t = time.time(); x = load_audio(r); t_audio += time.time() - t
    t = time.time(); h = enc.encode(x)
    if a.device == "cuda": torch.cuda.synchronize(); torch.cuda.empty_cache()   # 긴 파일 뒤 캐시 반납 — 공유 GPU 에서 다른 작업을 굶기지 않게
    t_enc += time.time() - t
    t = time.time(); np.save(p + ".tmp.npy", h); os.replace(p + ".tmp.npy", p); t_io += time.time() - t
    idx.write(json.dumps(dict(id=r["id"], npy=p, frames=int(h.shape[1]), dim=int(h.shape[2]), frame_hz=enc.frame_hz, duration=r["duration"])) + "\n"); idx.flush()
    hours += r["duration"] / 3600; n += 1
    if n % 20 == 0: print(f"  {n} conv {hours:.1f} h | enc RTF {t_enc/(hours*3600):.4f} | {time.time()-T0:.0f}s", flush=True)
peak = torch.cuda.max_memory_allocated() / 2**30 if a.device == "cuda" else 0.0
stats = dict(encoder=enc.name, manifest=mname, n=n, hours=hours, frame_hz=enc.frame_hz, dim=enc.dim, lookahead_ms=enc.lookahead_ms, causal=enc.causal,
             rtf_encode=(t_enc / (hours * 3600)) if hours else None, sec_audio=t_audio, sec_encode=t_enc, sec_io=t_io, peak_gpu_gb=peak,
             bytes_per_hour_gb=(2 * enc.frame_hz * 3600 * enc.dim * 2) / 1e9, total_gb=sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith(".npy")) / 1e9)
json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(stats, ensure_ascii=False))

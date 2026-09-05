#!/usr/bin/env python
"""Stage 1 특징 캐시 — streams.jsonl 의 스트림을 조립해 Nemotron [56,0] mono 특징을 저장한다 (plans/stage1-mono-pilot.md §3.4-(3)).

python experiments/s1_extract_features.py --manifest kspon-100 [--encoder nemotron-c0] [--mode stream|utt|all] [--limit N] [--ids a,b]
출력: $MXC_DATA_FEATURE_CACHE_DIR/<encoder>/<manifest>/<stream id>.npy  (1, T', D) fp16  + index.jsonl + stats.json   (재개 가능)
GPU 는 여유 메모리가 가장 큰 장치를 고른다(공용 서버). TF32 off(Stage 0 교훈: 배치 길이에 따라 값이 흔들린다).
"""
import os, sys, json, time, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True, help="이름(예: kspon-100) 또는 디렉토리 경로"); ap.add_argument("--encoder", default="nemotron-c0")
ap.add_argument("--mode", default="all", help="stream | utt | all"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--ids", default=None)
ap.add_argument("--gpu", default=None, help="CUDA 장치 번호. 없으면 여유 최대 자동"); ap.add_argument("--shard", default=None, help="k/n: 병렬 워커 k 가 rows[k::n] 만 처리")
a = ap.parse_args()
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if a.gpu is None:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        rows = [[int(v) for v in l.split(",")] for l in q.stdout.strip().splitlines()]; a.gpu = str(max(rows, key=lambda r: r[2] - r[1])[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import numpy as np, torch
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
from vapasr.data.streams import read_streams, assemble_stream
from vapasr.features.encoders import load_encoder

MAN = os.environ.get("MXC_DATA_MANIFEST_DIR", os.environ.get("DATA_MANIFEST_DIR", "/tmp")); FEAT = os.environ.get("MXC_DATA_FEATURE_CACHE_DIR", os.environ.get("DATA_FEATURE_CACHE_DIR", "/tmp"))
mdir = a.manifest if os.path.isdir(a.manifest) else os.path.join(MAN, a.manifest); mname = os.path.basename(mdir.rstrip("/"))
out = os.path.join(FEAT, a.encoder, mname); os.makedirs(out, exist_ok=True)
rows = read_streams(mdir, mode=None if a.mode == "all" else a.mode)
if a.ids: keep = set(a.ids.split(",")); rows = [r for r in rows if r["id"] in keep]
rows = rows[: a.limit] if a.limit else rows
if a.shard:
    k, n = (int(x) for x in a.shard.split("/")); rows = rows[k::n]
print(f"{a.encoder} ← {mname} ({a.mode}): {len(rows)} 스트림{' shard ' + a.shard if a.shard else ''}, {sum(r['duration_s'] for r in rows)/3600:.1f} h → {out}  [GPU {os.environ['CUDA_VISIBLE_DEVICES']}]", flush=True)

enc = load_encoder(a.encoder); print(f"encoder {enc.name}: {enc.frame_hz} Hz, D={enc.dim}, lookahead {enc.lookahead_ms} ms, causal={enc.causal}", flush=True)
ip = os.path.join(out, "index.jsonl"); done = {json.loads(l)["id"] for l in open(ip)} if os.path.exists(ip) and os.path.getsize(ip) else set(); idx = open(ip, "a")
t_audio = t_enc = t_io = hours = 0.0; n = 0; torch.cuda.reset_peak_memory_stats(); T0 = time.time(); cache = {}
for r in rows:
    if r["id"] in done: continue
    p = os.path.join(out, r["id"] + ".npy")
    t = time.time(); x = assemble_stream(r, cache); t_audio += time.time() - t
    if len(cache) > 512: cache.clear()
    t = time.time(); h = enc.encode(x[None]); torch.cuda.synchronize(); t_enc += time.time() - t       # (1, T', D)
    t = time.time(); np.save(p + ".tmp.npy", h); os.replace(p + ".tmp.npy", p); t_io += time.time() - t
    idx.write(json.dumps(dict(id=r["id"], npy=p, frames=int(h.shape[1]), dim=int(h.shape[2]), frame_hz=enc.frame_hz, duration=r["duration_s"], mode=r["mode"], subset=r["subset"])) + "\n"); idx.flush()
    hours += r["duration_s"] / 3600; n += 1
    if n % 200 == 0: torch.cuda.empty_cache(); print(f"  {n} 스트림 {hours:.1f} h | enc RTF {t_enc/(hours*3600):.4f} | {time.time()-T0:.0f}s", flush=True)
stats = dict(encoder=enc.name, manifest=mname, mode=a.mode, n=n, hours=round(hours, 2), frame_hz=enc.frame_hz, dim=enc.dim, channels=1,
             rtf_encode=(t_enc / (hours * 3600)) if hours else None, sec_audio=t_audio, sec_encode=t_enc, sec_io=t_io,
             peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30, total_gb=sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if f.endswith(".npy")) / 1e9)
json.dump(stats, open(os.path.join(out, f"stats-{a.mode}.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(stats, ensure_ascii=False))

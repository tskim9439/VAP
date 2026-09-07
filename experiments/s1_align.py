#!/usr/bin/env python
"""Stage 1 정렬 — streams.jsonl 의 발화마다 Qwen3-ForcedAligner 로 BPE 토큰 종료 시각(80 ms 격자)을 만든다 (plans/stage1-mono-pilot.md §3.4-(2)).

python experiments/s1_align.py --manifest librispeech-100 [--mode stream|utt|all] [--limit N] [--ids a,b] [--gpu K]
출력: $MXC_DATA_MANIFEST_DIR/align/<manifest>/<stream id>.jsonl  — 한 줄 = 발화 {speaker:0, start, end, text, tokens[{id,text,end_time}]} (스트림 절대 시각)
      + stats.json.  u0_align.py 와 같은 스키마라 u0_align_qc.py 와 interleave 빌더가 그대로 읽는다. 재개 가능(파일 존재 시 건너뜀), 동시 실행 안전(pid tmp).
모델은 로컬 디렉토리($MXC_ALIGNER_DIR, 토크나이저는 $MXC_QWEN_ASR_DIR)에서 읽는다. 발화 오디오는 원본 파일에서 직접(조립 스트림 아님) 읽고 offset_s 를 더한다.
"""
import os, sys, json, time, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True); ap.add_argument("--mode", default="all"); ap.add_argument("--limit", type=int, default=None); ap.add_argument("--ids", default=None)
ap.add_argument("--gpu", default=None); ap.add_argument("--min-dur", type=float, default=0.3)
ap.add_argument("--out-root", default=None, help="출력 루트(기본 $MXC_DATA_MANIFEST_DIR/align-asr-tn-v1 = 규약 ID). 규약이 바뀌면 새 루트로 — 기존 산출물은 지우지 않는다")
ap.add_argument("--shard", default=None, help="k/n: 병렬 워커 k 가 rows[k::n] 만 처리 (같은 manifest 를 여러 프로세스로)")
ap.add_argument("--batch", type=int, default=64, help="배치 최대 발화 수"); ap.add_argument("--batch-sec", type=float, default=480.0, help="배치 오디오 합계 상한(초) — padding·메모리 제어")
ap.add_argument("--chunk", type=int, default=128, help="한 번에 읽어 두는 스트림 수(오디오 prefetch 단위)"); ap.add_argument("--io-threads", type=int, default=16); ap.add_argument("--reverse", action="store_true", help="스트림을 역순으로(다른 run 과 양끝에서 분담)")
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
from vapasr.data.textnorm import check_fingerprint, TEXTNORM_ID_SHORT
_mst = json.load(open(os.path.join(mdir, "stats.json"))) if os.path.exists(os.path.join(mdir, "stats.json")) else {}
check_fingerprint(_mst.get("fingerprint"), f"align {mname}")                       # asr-tn-v1.0.0: manifest fingerprint 없거나 코드와 다르면 시작 전 실패
out = os.path.join(a.out_root or os.path.join(MAN, f"align-{TEXTNORM_ID_SHORT}"), mname); os.makedirs(out, exist_ok=True)
if _mst.get("fingerprint"): json.dump(_mst["fingerprint"], open(os.path.join(out, "fingerprint.json"), "w"), indent=1)
rows = read_streams(mdir, mode=None if a.mode == "all" else a.mode)
if a.ids: keep = set(a.ids.split(",")); rows = [r for r in rows if r["id"] in keep]
rows = rows[: a.limit] if a.limit else rows
if a.shard:   # 병렬 워커: k/n → rows[k::n]. 각 워커가 다른 행을 맡아 중복 없이 나눠 처리(기존 파일은 건너뜀)
    k, n = (int(x) for x in a.shard.split("/")); rows = rows[k::n]
print(f"align ← {mname} ({a.mode}): {len(rows)} 스트림{' shard ' + a.shard if a.shard else ''} → {out}  [GPU {os.environ['CUDA_VISIBLE_DEVICES']}]", flush=True)

aligner = Qwen3ForcedAligner.from_pretrained(ALIGNER, dtype=torch.bfloat16, device_map="cuda"); tok = AutoTokenizer.from_pretrained(QWEN)
from concurrent.futures import ThreadPoolExecutor

def tokens_from_items(items, text):
    """aligner 항목(단어/문자 시각) → BPE 토큰 종료 시각. offsets 로 문자 구간을 잇는다 (u0_align.py 와 동일 로직)."""
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); ids, offs = enc["input_ids"], enc["offset_mapping"]
    spans, cur = [], 0
    for it in items:
        s0 = text.find(it.text, cur)
        if s0 < 0: s0 = cur
        spans.append((s0, s0 + len(it.text), float(it.end_time))); cur = s0 + len(it.text)
    outp = []
    for tid, (o0, o1) in zip(ids, offs):
        cover = [sp for sp in spans if sp[0] < o1 and sp[1] > o0]
        t_end = max(c[2] for c in cover) if cover else (outp[-1]["end_time"] if outp else 0.0)
        outp.append(dict(id=int(tid), text=text[o0:o1], end_time=t_end))
    return outp

def run_batch(items):
    """배치 정렬(한 번의 padded forward). 실패하면 이분해서 문제 항목만 격리한다. → 항목별 aligner items 또는 예외."""
    try:
        res = aligner.align(audio=[(it["audio"], SR) for it in items], text=[it["atext"] for it in items], language=[it["lang"] for it in items])
        return [(r.items if hasattr(r, "items") else r) for r in res]
    except Exception as ex:
        if len(items) == 1: return [ex]
        torch.cuda.empty_cache(); h = len(items) // 2; return run_batch(items[:h]) + run_batch(items[h:])

def make_batches(utts):
    """길이 순 정렬 후 (합계 초 ≤ batch_sec, 개수 ≤ batch) 로 묶는다 — padding 을 줄이고 메모리를 일정하게."""
    utts = sorted(utts, key=lambda u: u["dur"]); bs, cur, sec = [], [], 0.0
    for u in utts:
        if cur and (sec + u["dur"] > a.batch_sec or len(cur) >= a.batch): bs.append(cur); cur, sec = [], 0.0
        cur.append(u); sec += u["dur"]
    if cur: bs.append(cur)
    return bs

def load_chunk(chunk):
    """스트림 묶음의 발화 오디오를 스레드로 미리 읽는다 → [(row, [utt dict…])]."""
    def one(r):
        us = []
        if os.path.exists(os.path.join(out, r["id"] + ".jsonl")): return r, us          # 다른 run(SLURM/로그인)이 그사이 끝낸 스트림은 건너뜀
        for u in iter_utterances(r):
            if u["end"] - u["start"] < a.min_dur or not u["text"]: continue
            # 스트림 안에서 두 번째 발화부터는 선행 공백을 붙여 토큰화한다(없으면 'lost'+'i' → 'losti'). 정렬기에도 공백 포함 텍스트를 준다.
            us.append(dict(u, audio=load_utt_audio(u["path"]), dur=u["end"] - u["start"], atext=(" " + u["text"]) if u.get("idx", 0) > 0 else u["text"], lang=r["lang"], sid=r["id"]))
        return r, us
    return list(pool.map(one, chunk))

st = dict(streams=0, utts=0, tokens=0, fail=0, offset_err_ms=[], sec=0.0); T0 = time.time()
rows = [r for r in rows if not os.path.exists(os.path.join(out, r["id"] + ".jsonl"))]
if a.reverse: rows = rows[::-1]                                        # 두 run 이 같은 manifest 를 양끝에서 처리해 중간에서 만나도록
print(f"  남은 스트림 {len(rows)} (기존 파일 건너뜀{', 역순' if a.reverse else ''})", flush=True)
pool = ThreadPoolExecutor(a.io_threads); chunks = [rows[i: i + a.chunk] for i in range(0, len(rows), a.chunk)]
fut = pool.submit(load_chunk, chunks[0]) if chunks else None
for ci in range(len(chunks)):
    loaded = fut.result(); fut = pool.submit(load_chunk, chunks[ci + 1]) if ci + 1 < len(chunks) else None   # 다음 묶음 오디오를 미리 읽는다
    allu = [u for _, us in loaded for u in us]; results = {}
    for b in make_batches(allu):
        for u, res in zip(b, run_batch(b)): results[id(u)] = res
    for r, us in loaded:
        if not us: continue
        outp = os.path.join(out, r["id"] + ".jsonl"); tmpp = outp + f".{os.getpid()}.tmp"
        with open(tmpp, "w", encoding="utf-8") as f:
            for u in us:
                res = results[id(u)]
                if isinstance(res, Exception): st["fail"] += 1; print(f"  ! {r['id']} {u['utt_id']}: {type(res).__name__}: {str(res)[:80]}", flush=True); continue
                try: toks = tokens_from_items(res, u["atext"])
                except Exception as ex: st["fail"] += 1; print(f"  ! {r['id']} {u['utt_id']}: {type(ex).__name__}: {str(ex)[:80]}", flush=True); continue
                for t in toks: t["end_time"] = round(u["start"] + t["end_time"], 3)      # 스트림 절대 시각
                f.write(json.dumps(dict(speaker=0, start=u["start"], end=u["end"], text=u["text"], tokens=toks), ensure_ascii=False) + "\n")
                st["utts"] += 1; st["tokens"] += len(toks)
                if toks: st["offset_err_ms"].append((u["end"] - toks[-1]["end_time"]) * 1000)
        if os.path.exists(outp): os.remove(tmpp); continue
        os.replace(tmpp, outp); st["streams"] += 1
    if st["streams"] % 500 < a.chunk: el = time.time() - T0; print(f"  {st['streams']} 스트림 · {st['utts']} 발화 ({st['utts']/max(1,el):.1f}/s) · {st['tokens']} 토큰 · 실패 {st['fail']} · {el:.0f}s", flush=True)
pool.shutdown()
st["sec"] = time.time() - T0; v = np.array(st["offset_err_ms"]); st["offset_err_ms"] = dict(n=len(v), median=float(np.median(v)) if len(v) else None, p90=float(np.percentile(v, 90)) if len(v) else None)
json.dump(st, open(os.path.join(out, f"stats-{a.mode}.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(st, ensure_ascii=False))

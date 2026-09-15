#!/usr/bin/env python
"""Phase 2 Q0 — dialogues.jsonl 의 발화마다 화자별 채널에서 Qwen3-ForcedAligner 로 BPE 토큰 종료 시각을 만든다 (task-phase2-data-prep 순서 4).

  python experiments/p2_align.py --dialogues <dir>/<corpus>.dialogues.jsonl [--shard k/n] [--limit N] [--gpu K]
출력: <dir>/align-<tn>/<corpus>/parts/<host>-<pid>-<chunk>.jsonl — 한 줄 = {conv_id, utts:{utt_id: [[token_id, end_time_s(대화 절대 시각)], …]}, fail:[utt_id…]}.
재개 가능(완료 conv_id 건너뜀), 동시 실행 안전. 오디오는 화자 채널(path#chN 의 [start, end] 구간 또는 조각 파일)에서 읽는다 — mono 혼합이 아니라 깨끗한 채널.
s1_align.py 와 같은 정렬기·토큰 매핑(tokens_from_items). GPU 잡은 사용자가 SLURM 으로 제출한다."""
import os, sys, json, time, argparse, subprocess, socket, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--dialogues", required=True); ap.add_argument("--out-root", default=None); ap.add_argument("--shard", default=None); ap.add_argument("--limit", type=int)
ap.add_argument("--gpu", default=None); ap.add_argument("--min-dur", type=float, default=0.3); ap.add_argument("--max-dur", type=float, default=60.0)
ap.add_argument("--batch", type=int, default=64); ap.add_argument("--batch-sec", type=float, default=480.0); ap.add_argument("--chunk", type=int, default=32); ap.add_argument("--io-threads", type=int, default=8)
a = ap.parse_args()
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if a.gpu is None:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        rows = [[int(v) for v in l.split(",")] for l in q.stdout.strip().splitlines()]; a.gpu = str(max(rows, key=lambda r: r[2] - r[1])[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import numpy as np, torch
from qwen_asr import Qwen3ForcedAligner
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor
from vapasr.data.dialogue import Dialogue
from vapasr.data.streams import load_utt_audio, SR
from vapasr.data.textnorm import TEXTNORM_ID_SHORT

ALIGNER = os.environ.get("MXC_ALIGNER_DIR", "Qwen/Qwen3-ForcedAligner-0.6B"); QWEN = os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B")
corpus = os.path.basename(a.dialogues).split(".")[0]
out = os.path.join(a.out_root or os.path.join(os.path.dirname(a.dialogues), f"align-{TEXTNORM_ID_SHORT}"), corpus); PARTS = os.path.join(out, "parts"); os.makedirs(PARTS, exist_ok=True)
dlgs = [Dialogue.from_json(l) for l in open(a.dialogues, encoding="utf-8")]
if a.shard: k, n = (int(x) for x in a.shard.split("/")); dlgs = dlgs[k::n]
done = set()
for pf in os.listdir(PARTS):
    if pf.endswith(".jsonl"):
        for line in open(os.path.join(PARTS, pf), encoding="utf-8"):
            try: done.add(json.loads(line)["conv_id"])
            except Exception: pass
dlgs = [d for d in dlgs if d.conv_id not in done][: a.limit or None]
print(f"align ← {corpus}: {len(dlgs)} dialogues (기존 {len(done)} 건너뜀) → {out} [GPU {os.environ['CUDA_VISIBLE_DEVICES']}]", flush=True)
aligner = Qwen3ForcedAligner.from_pretrained(ALIGNER, dtype=torch.bfloat16, device_map="cuda"); tok = AutoTokenizer.from_pretrained(QWEN)

def tokens_from_items(items, text):
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False); ids, offs = enc["input_ids"], enc["offset_mapping"]
    spans, cur = [], 0
    for it in items:
        s0 = text.find(it.text, cur)
        if s0 < 0: s0 = cur
        spans.append((s0, s0 + len(it.text), float(it.end_time))); cur = s0 + len(it.text)
    outp = []
    for tid, (o0, o1) in zip(ids, offs):
        cover = [sp for sp in spans if sp[0] < o1 and sp[1] > o0]
        outp.append((int(tid), max(c[2] for c in cover) if cover else (outp[-1][1] if outp else 0.0)))
    return outp

def utt_audio(d: Dialogue, u):
    ref = d.channels[u.speaker]
    if ref.pieces is not None:
        p = min(ref.pieces, key=lambda x: abs(x[1] - u.start))
        if abs(p[1] - u.start) > 0.05: raise FileNotFoundError(f"no piece at {u.start}")
        return load_utt_audio(p[0])
    return load_utt_audio(ref.path, u.start, u.end - u.start)

def run_batch(items):
    try:
        res = aligner.align(audio=[(it["audio"], SR) for it in items], text=[it["text"] for it in items], language=[it["lang"] for it in items])
        return [(r.items if hasattr(r, "items") else r) for r in res]
    except Exception as ex:
        if len(items) == 1: return [ex]
        torch.cuda.empty_cache(); h = len(items) // 2; return run_batch(items[:h]) + run_batch(items[h:])

def make_batches(utts):
    utts = sorted(utts, key=lambda u: u["dur"]); bs, cur, sec = [], [], 0.0
    for u in utts:
        if cur and (sec + u["dur"] > a.batch_sec or len(cur) >= a.batch): bs.append(cur); cur, sec = [], 0.0
        cur.append(u); sec += u["dur"]
    if cur: bs.append(cur)
    return bs

st = collections.Counter(); T0 = time.time(); pool = ThreadPoolExecutor(a.io_threads); TAG = f"{socket.gethostname()}-{os.getpid()}"
def load_chunk(chunk):
    def one(d):
        us = []
        for u in d.utterances:
            dur = u.end - u.start
            if not u.text or dur < a.min_dur or dur > a.max_dur: st["skipped"] += 1; continue
            try: audio = utt_audio(d, u)
            except Exception as ex: st["audio_missing"] += 1; continue
            us.append(dict(d=d, u=u, audio=audio, dur=dur, text=" " + u.text, lang=d.lang))      # 대화 안 발화는 모두 앞 공백 포함 토큰화(s1_align 과 같이: "tv"+"and" 가 "tvand" 로 붙지 않게)
        return d, us
    return list(pool.map(one, chunk))
chunks = [dlgs[i: i + a.chunk] for i in range(0, len(dlgs), a.chunk)]; fut = pool.submit(load_chunk, chunks[0]) if chunks else None
for ci in range(len(chunks)):
    loaded = fut.result(); fut = pool.submit(load_chunk, chunks[ci + 1]) if ci + 1 < len(chunks) else None
    allu = [x for _, us in loaded for x in us]; results = {}
    for b in make_batches(allu):
        for x, res in zip(b, run_batch(b)): results[id(x)] = res
    partp = os.path.join(PARTS, f"{TAG}-{ci:06d}.jsonl"); tmpp = partp + ".tmp"
    with open(tmpp, "w", encoding="utf-8") as f:
        for d, us in loaded:
            rec = dict(conv_id=d.conv_id, utts={}, fail=[])
            for x in us:
                res = results[id(x)]
                if isinstance(res, Exception): st["fail"] += 1; rec["fail"].append(x["u"].utt_id); continue
                try: toks = tokens_from_items(res, x["text"])
                except Exception: st["fail"] += 1; rec["fail"].append(x["u"].utt_id); continue
                rec["utts"][x["u"].utt_id] = [[tid, round(x["u"].start + t, 3)] for tid, t in toks]; st["utts"] += 1; st["tokens"] += len(toks)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); st["dialogues"] += 1
    os.replace(tmpp, partp); el = time.time() - T0
    print(f"  {st['dialogues']} dialogues · {st['utts']} utts ({st['utts']/max(1,el):.1f}/s) · fail {st['fail']} · missing {st['audio_missing']} · {el:.0f}s", flush=True)
pool.shutdown(); st["sec"] = round(time.time() - T0, 1)
json.dump(dict(st), open(os.path.join(out, f"stats-{TAG}.json"), "w"), indent=1); print(json.dumps(dict(st)))

#!/usr/bin/env python
"""Stage 1 manifest 빌더 — plans/stage1-mono-pilot.md §3.1·3.2·3.4-(1).

스트림 = 학습 단위. 오디오는 저장하지 않고 로드 시 조립한다(무음 길이는 여기서 뽑아 기록 → 로더는 읽기만).
  LibriSpeech : 같은 화자·챕터 안에서 발화를 원래 순서로 이어 붙여 20–30 s 스트림(목표 25 s). 발화 사이 무음 U(0.3,1.5) s, 앞 0.3–1.0 s.
  KsponSpeech : 파일 하나 = 스트림 하나(가변 길이). 앞뒤 무음 U(0.3,1.0) s 만. 다른 파일과 절대 연결하지 않는다(폴더 순서 ≠ 화자·세션).
dev/test 는 스트림 행(mode=stream) 과 발화 단위 행(mode=utt) 을 둘 다 넣는다 — 평가기가 고른다.

컨테이너에서:  python experiments/s1_build_manifest.py [--only librispeech-100,kspon-100,...] [--workers 16]
출력: $MXC_DATA_MANIFEST_DIR/<name>/streams.jsonl + stats.json
  행: {id, corpus, split, subset, mode, lang, speaker, chapter, duration_s, n_utts,
       segments:[{utt_id, path, silence_before_s, offset_s, dur_s, text}], silence_after_s}
"""
import os, sys, json, glob, random, argparse, collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.kspon import read_trn, normalize_kspon, resolve_path, pcm_duration

ap = argparse.ArgumentParser()
ap.add_argument("--kspon-root", default=os.environ.get("MXC_KSPONSPEECH_DIR", os.environ.get("KSPONSPEECH_DIR")))
ap.add_argument("--libri-root", default=os.environ.get("MXC_LIBRISPEECH_DIR", os.environ.get("LIBRISPEECH_DIR")))
ap.add_argument("--out", default=os.environ.get("MXC_DATA_MANIFEST_DIR", os.environ.get("DATA_MANIFEST_DIR", "/tmp")))
ap.add_argument("--only", default=None, help="쉼표 구분 manifest 이름"); ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--kspon-folders", default="1-62", help="KsponSpeech_01 하위 폴더 범위(파일럿 0001~0062)")
ap.add_argument("--target-s", type=float, default=25.0); ap.add_argument("--min-s", type=float, default=20.0); ap.add_argument("--max-s", type=float, default=30.0)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args(); assert a.kspon_root and a.libri_root, ".env 의 MXC_KSPONSPEECH_DIR / MXC_LIBRISPEECH_DIR 필요"

MANIFESTS = {  # name: (corpus, [(split_dir, subset)])
    "librispeech-100":  ("librispeech", [("train-clean-100", "train-clean-100")]),
    "librispeech-dev":  ("librispeech", [("dev-clean", "dev-clean"), ("dev-other", "dev-other")]),
    "librispeech-test": ("librispeech", [("test-clean", "test-clean"), ("test-other", "test-other")]),
    "kspon-100":        ("kspon", [("train.trn", "train-01")]),
    "kspon-dev":        ("kspon", [("dev.trn", "dev")]),
    "kspon-eval":       ("kspon", [("eval_clean.trn", "eval_clean"), ("eval_other.trn", "eval_other")]),
}
names = a.only.split(",") if a.only else list(MANIFESTS)
def rng_for(sid): return random.Random(f"{a.seed}:{sid}")   # 스트림 id 로 결정적
def pct(xs): xs = np.array(xs); return {p: round(float(np.percentile(xs, p)), 2) for p in (5, 50, 95)} if len(xs) else {}

# ───────────────────────────── LibriSpeech ─────────────────────────────
def libri_utts(split):
    root = os.path.join(a.libri_root, split); utts = []
    for tp in sorted(glob.glob(os.path.join(root, "*", "*", "*.trans.txt"))):
        spk, chap = tp.split(os.sep)[-3:-1]
        for line in open(tp):
            uid, txt = line.rstrip("\n").split(" ", 1)
            utts.append(dict(utt_id=uid, speaker=spk, chapter=chap, path=os.path.join(root, spk, chap, uid + ".flac"), text=txt.lower().strip()))
    import soundfile as sf
    with ThreadPoolExecutor(a.workers) as ex:
        for u, d in zip(utts, ex.map(lambda u: sf.info(u["path"]).duration, utts)): u["dur_s"] = round(float(d), 3)
    return utts

def libri_streams(utts, subset, split):
    """챕터 안에서 발화를 순서대로 묶는다. 목표 target_s, 상한 max_s. 챕터 끝의 짧은 자투리(< min_s)는 직전 스트림에 붙이되 max_s 를 넘기지 않는다."""
    by = collections.OrderedDict()
    for u in utts: by.setdefault((u["speaker"], u["chapter"]), []).append(u)
    rows, n = [], 0
    for (spk, chap), us in by.items():
        us.sort(key=lambda u: u["utt_id"]); groups, cur, cur_s = [], [], 0.0
        for u in us:
            if cur and cur_s + u["dur_s"] > a.max_s and cur_s >= a.min_s: groups.append(cur); cur, cur_s = [], 0.0
            cur.append(u); cur_s += u["dur_s"]
            if cur_s >= a.target_s: groups.append(cur); cur, cur_s = [], 0.0
        if cur:
            if groups and cur_s < a.min_s and sum(x["dur_s"] for x in groups[-1]) + cur_s <= a.max_s: groups[-1] += cur
            else: groups.append(cur)
        for g in groups:
            sid = f"ls-{subset}-{spk}-{chap}-{n:05d}"; r = rng_for(sid); t = round(r.uniform(0.3, 1.0), 3); segs = []
            for i, u in enumerate(g):
                gap = t if i == 0 else round(r.uniform(0.3, 1.5), 3); t = t if i == 0 else t + gap
                segs.append(dict(utt_id=u["utt_id"], path=u["path"], silence_before_s=gap, offset_s=round(t, 3), dur_s=u["dur_s"], text=u["text"])); t += u["dur_s"]
            rows.append(dict(id=sid, corpus="librispeech", split=split, subset=subset, mode="stream", lang="English", speaker=spk, chapter=chap,
                             duration_s=round(t + 0.3, 3), n_utts=len(g), segments=segs, silence_after_s=0.3)); n += 1
    return rows

def utt_rows(utts, corpus, subset, split, lang, prefix, key_speaker="speaker"):
    """발화 단위 행(mode=utt): 발화 하나 + 앞뒤 무음."""
    rows = []
    for u in utts:
        sid = f"{prefix}-{subset}-utt-{u['utt_id']}"; r = rng_for(sid); lead, trail = round(r.uniform(0.3, 1.0), 3), round(r.uniform(0.3, 1.0), 3)
        rows.append(dict(id=sid, corpus=corpus, split=split, subset=subset, mode="utt", lang=lang, speaker=u.get(key_speaker), chapter=u.get("chapter"),
                         duration_s=round(lead + u["dur_s"] + trail, 3), n_utts=1, silence_after_s=trail,
                         segments=[dict(utt_id=u["utt_id"], path=u["path"], silence_before_s=lead, offset_s=lead, dur_s=u["dur_s"], text=u["text"])]))
    return rows

# ───────────────────────────── KsponSpeech ─────────────────────────────
def kspon_utts(trn_name, subset):
    lo, hi = (int(x) for x in a.kspon_folders.split("-")); rows = read_trn(os.path.join(a.kspon_root, trn_name)); out, st = [], collections.Counter()
    for rel, raw in rows:
        if subset == "train-01":
            parts = rel.split("/")
            if parts[0] != "KsponSpeech_01" or not (lo <= int(parts[1].split("_")[1]) <= hi): continue
        p = resolve_path(a.kspon_root, rel)
        if p is None: st["missing"] += 1; continue
        txt = normalize_kspon(raw)
        if not txt: st["empty_text"] += 1; continue
        out.append(dict(utt_id=os.path.splitext(os.path.basename(rel))[0], rel=rel, path=p, raw=raw, text=txt, speaker=None, chapter=None)); st["kept"] += 1
    with ThreadPoolExecutor(a.workers) as ex:
        for u, (d, odd) in zip(out, ex.map(lambda u: pcm_duration(u["path"]), out)): u["dur_s"] = round(d, 3); st["odd_byte"] += int(odd)
    return out, st

# ───────────────────────────── 실행 ─────────────────────────────
for name in names:
    corpus, parts = MANIFESTS[name]; od = os.path.join(a.out, name); os.makedirs(od, exist_ok=True); rows, stats = [], {}
    print(f"\n=== {name} ===", flush=True)
    for src, subset in parts:
        split = "train" if name.endswith("-100") else ("dev" if "dev" in name else "test")
        if corpus == "librispeech":
            utts = libri_utts(src); rs = libri_streams(utts, subset, split)
            if split != "train": rs += utt_rows(utts, "librispeech", subset, split, "English", "ls")
            sub_st = dict(utts=len(utts), speakers=len({u["speaker"] for u in utts}), chapters=len({(u["speaker"], u["chapter"]) for u in utts}),
                          utt_hours=round(sum(u["dur_s"] for u in utts) / 3600, 2))
        else:
            utts, st = kspon_utts(src, subset)
            rs = utt_rows(utts, "kspon", subset, split, "Korean", "ks")   # 파일 = 스트림. 학습도 발화 행이지만 mode 는 stream 으로 표기
            if split == "train":
                for r in rs: r["mode"] = "stream"; r["id"] = r["id"].replace("-utt-", "-")
            sub_st = dict(utts=len(utts), utt_hours=round(sum(u["dur_s"] for u in utts) / 3600, 2), **st)
        streams = [r for r in rs if r["mode"] == "stream"]
        sub_st.update(streams=len(streams), stream_hours=round(sum(r["duration_s"] for r in streams) / 3600, 2), stream_len_pct=pct([r["duration_s"] for r in streams]),
                      utts_per_stream_pct=pct([r["n_utts"] for r in streams]))
        stats[subset] = sub_st; rows += rs
        print(f"  {subset:16s} " + " · ".join(f"{k}={v}" for k, v in sub_st.items() if not isinstance(v, dict)) + f" · 길이 p50 {sub_st['stream_len_pct'].get(50)} s", flush=True)
    with open(os.path.join(od, "streams.jsonl"), "w", encoding="utf-8") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(dict(name=name, corpus=corpus, rows=len(rows), seed=a.seed, target_s=a.target_s, subsets=stats), open(os.path.join(od, "stats.json"), "w"), ensure_ascii=False, indent=1)
    print(f"  → {od}/streams.jsonl ({len(rows)} 행)")

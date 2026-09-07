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
from vapasr.data.kspon import read_trn, resolve_path, pcm_duration
from vapasr.data.textnorm import target_en, target_ko, target_flags, TEXTNORM_VERSION, fingerprint   # asr-tn-v1.0.0: 모든 타깃은 여기서, quarantine·fingerprint 기록

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
    "librispeech-960":  ("librispeech", [("train-clean-100", "train-clean-100"), ("train-clean-360", "train-clean-360"), ("train-other-500", "train-other-500")]),
    "kspon-full":       ("kspon", [("train.trn", "train-all")]),           # KsponSpeech_01–05 전체(≈970 h)
    "kspon-dev":        ("kspon", [("dev.trn", "dev")]),
    "kspon-eval":       ("kspon", [("eval_clean.trn", "eval_clean"), ("eval_other.trn", "eval_other")]),
}
names = a.only.split(",") if a.only else list(MANIFESTS)
def rng_for(sid): return random.Random(f"{a.seed}:{sid}")   # 스트림 id 로 결정적
def pct(xs): xs = np.array(xs); return {p: round(float(np.percentile(xs, p)), 2) for p in (5, 50, 95)} if len(xs) else {}

# ───────────────────────────── LibriSpeech ─────────────────────────────
QUAR = []                     # quarantine 행: 학습 manifest 에서 제외하되 ID·원문·사유를 보존(asr-tn-v1.0.0)
SRC_FILES = []                # source_transcript_sha256 용
def _quarantine(uid, subset, raw, text, reasons): QUAR.append(dict(id=uid, subset=subset, raw=raw, target=text, reasons=sorted(reasons)))

def libri_utts(split):
    root = os.path.join(a.libri_root, split); utts = []; st = collections.Counter()
    for tp in sorted(glob.glob(os.path.join(root, "*", "*", "*.trans.txt"))):
        spk, chap = tp.split(os.sep)[-3:-1]; SRC_FILES.append(tp)
        for line in open(tp):
            uid, txt = line.rstrip("\n").split(" ", 1); text = target_en(txt, "librispeech"); fl = target_flags(text, "English")
            if fl: st["quarantined"] += 1; _quarantine(uid, split, txt, text, fl); continue          # LibriSpeech 에 digit 등이 남으면 추측 변환 없이 제외
            utts.append(dict(utt_id=uid, speaker=spk, chapter=chap, path=os.path.join(root, spk, chap, uid + ".flac"), text=text, raw=txt))
    print(f"    {split}: {len(utts)} 발화, quarantine {st['quarantined']}", flush=True)
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
                segs.append(dict(utt_id=u["utt_id"], path=u["path"], silence_before_s=gap, offset_s=round(t, 3), dur_s=u["dur_s"], text=u["text"], lexical_text=u["text"], raw_text=u["raw"], display_source="none")); t += u["dur_s"]
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
                         segments=[dict(utt_id=u["utt_id"], path=u["path"], silence_before_s=lead, offset_s=lead, dur_s=u["dur_s"], text=u["text"], lexical_text=u["text"], raw_text=u["raw"], display_source="none")]))
    return rows

# ───────────────────────────── KsponSpeech ─────────────────────────────
def kspon_utts(trn_name, subset):
    lo, hi = (int(x) for x in a.kspon_folders.split("-")); tp = os.path.join(a.kspon_root, trn_name); SRC_FILES.append(tp); rows = read_trn(tp); out, st = [], collections.Counter()
    for rel, raw in rows:
        if subset == "train-01":
            parts = rel.split("/")
            if parts[0] != "KsponSpeech_01" or not (lo <= int(parts[1].split("_")[1]) <= hi): continue
        p = resolve_path(a.kspon_root, rel)
        if p is None: st["missing"] += 1; continue
        uid = os.path.splitext(os.path.basename(rel))[0]; txt = target_ko(raw, "kspon"); fl = target_flags(txt, "Korean", raw, "kspon")
        for k in fl: st[f"flag_{k}"] += 1
        if fl and subset.startswith("train"): st["quarantined"] += 1; _quarantine(uid, subset, raw, txt, fl); continue   # 학습만 제외. dev/eval 은 공개 세트 그대로(사유는 통계로)
        if not txt: st["empty_text"] += 1; continue
        out.append(dict(utt_id=uid, rel=rel, path=p, raw=raw, text=txt, speaker=None, chapter=None)); st["kept"] += 1
    with ThreadPoolExecutor(a.workers) as ex:
        for u, (d, odd) in zip(out, ex.map(lambda u: pcm_duration(u["path"]), out)): u["dur_s"] = round(d, 3); st["odd_byte"] += int(odd)
    return out, st

# ───────────────────────────── 실행 ─────────────────────────────
for name in names:
    corpus, parts = MANIFESTS[name]; od = os.path.join(a.out, name); os.makedirs(od, exist_ok=True); rows, stats = [], {}
    print(f"\n=== {name} ===", flush=True)
    for src, subset in parts:
        split = "dev" if "dev" in name else ("test" if ("test" in name or "eval" in name) else "train")
        if corpus == "librispeech":
            utts = libri_utts(src); rs = libri_streams(utts, subset, split)
            if split != "train": rs += utt_rows(utts, "librispeech", subset, split, "English", "ls")
            sub_st = dict(utts=len(utts), speakers=len({u["speaker"] for u in utts}), chapters=len({(u["speaker"], u["chapter"]) for u in utts}),
                          utt_hours=round(sum(u["dur_s"] for u in utts) / 3600, 2))
        else:
            utts, st = kspon_utts(src, subset)
            if subset == "train-all":                                   # KsponSpeech_0X → subset train-0X (id 가 kspon-100 과 호환)
                rs = []
                for part in sorted({u["rel"].split("/")[0] for u in utts}):
                    rs += utt_rows([u for u in utts if u["rel"].split("/")[0] == part], "kspon", "train-" + part[-2:], split, "Korean", "ks")
            else: rs = utt_rows(utts, "kspon", subset, split, "Korean", "ks")   # 파일 = 스트림. 학습도 발화 행이지만 mode 는 stream 으로 표기
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
    import hashlib, subprocess
    def sha_files(fs):
        h = hashlib.sha256()
        for f in sorted(fs): h.update(open(f, "rb").read())
        return h.hexdigest()
    with open(os.path.join(od, "quarantine.jsonl"), "w", encoding="utf-8") as f:
        for q in QUAR: f.write(json.dumps(q, ensure_ascii=False) + "\n")
    try: commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).decode().strip()
    except Exception: commit = None
    fp = fingerprint(os.environ.get("MXC_QWEN_ASR_DIR", os.environ.get("QWEN_ASR_DIR")))
    fp.update(source_transcript_sha256=sha_files(set(SRC_FILES)), manifest_sha256=sha_files([os.path.join(od, "streams.jsonl")]),
              quarantined_ids_sha256=hashlib.sha256("\n".join(sorted(q["id"] for q in QUAR)).encode()).hexdigest(), created_from_git_commit=commit)
    disp = collections.Counter(seg.get("display_source", "none") for r in rows for seg in r["segments"])
    json.dump(dict(name=name, corpus=corpus, rows=len(rows), seed=a.seed, target_s=a.target_s, textnorm_version=TEXTNORM_VERSION, fingerprint=fp,
                   quarantined=len(QUAR), display_source_counts=dict(disp), text_fields=dict(text="lexical_text 와 동일(하위 호환)", lexical_text="asr-tn-v1.0.0 학습·정렬 타깃", raw_text="원문 보존", display_source="none = display label 없음"), subsets=stats), open(os.path.join(od, "stats.json"), "w"), ensure_ascii=False, indent=1)
    QUAR.clear(); SRC_FILES.clear()
    print(f"  → {od}/streams.jsonl ({len(rows)} 행)")

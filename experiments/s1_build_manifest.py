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
from vapasr.data.streams import flac_duration
from vapasr.data.kspon import read_trn, resolve_path, pcm_duration
from vapasr.data.textnorm import target_en, target_ko, target_flags, TEXTNORM_VERSION, fingerprint   # asr-tn-v1.0.0: 모든 타깃은 여기서, quarantine·fingerprint 기록

ap = argparse.ArgumentParser()
ap.add_argument("--kspon-root", default=os.environ.get("MXC_KSPONSPEECH_DIR", os.environ.get("KSPONSPEECH_DIR")))
ap.add_argument("--libri-root", default=os.environ.get("MXC_LIBRISPEECH_DIR", os.environ.get("LIBRISPEECH_DIR")))
ap.add_argument("--out", default=os.environ.get("MXC_DATA_MANIFEST_DIR", os.environ.get("DATA_MANIFEST_DIR", "/tmp")))
ap.add_argument("--only", default=None, help="쉼표 구분 manifest 이름"); ap.add_argument("--workers", type=int, default=128)
ap.add_argument("--kspon-folders", default="1-62", help="KsponSpeech_01 하위 폴더 범위(파일럿 0001~0062)")
ap.add_argument("--en-open-root", default=os.environ.get("MXC_EN_OPEN_DIR", "/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/EN/TRAIN/OPEN"), help="Switchboard·MNSC CSV/wav 루트")
ap.add_argument("--nikl-root", default=os.environ.get("MXC_NIKL_DIR", "/soundai/DB/raw/nikl")); ap.add_argument("--nikl-years", default="2021,2022,2023,2024,2025")
ap.add_argument("--sample-hours", type=float, default=None, help="이름이 *-<N> 인 manifest 는 N h 표본(seed 결정적). 명시하면 그 값")
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
    # ── asr-tn-v1.1.0 확장(2026-09-07): 발화 = 스트림. *-1000 은 1,000 h 표본(seed 결정적)
    "swbd-train":       ("switchboard", [("switchboard/switchboard_train_tn.csv", "train")]),                          # 230 h, 16 kHz 전화 대화
    "mnsc-1000":        ("mnsc", [("Multitask-National-Speech-Corpus-v1/Multitask-National-Speech-Corpus-v1_train_tn.csv", "part1")]),   # PART1 낭독 3,338 h 중 1,000 h
    "nikl-1000":        ("nikl", [("years", "train")]),                                                                # NIKL 2021–2025 일상대화 중 1,000 h(대화 단위 표본)
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
    # 길이: FLAC 헤더 42 바이트만 읽는다(sf.info 대비 수십 배). split 별 캐시(_dur-cache)로 재실행은 즉시.
    cdir = os.path.join(a.out, "_dur-cache"); os.makedirs(cdir, exist_ok=True); cp = os.path.join(cdir, f"librispeech-{split}.json")
    cache = json.load(open(cp)) if os.path.exists(cp) else {}
    todo = [u for u in utts if u["utt_id"] not in cache]
    with ThreadPoolExecutor(a.workers) as ex:
        for u, d in zip(todo, ex.map(lambda u: flac_duration(u["path"]), todo)): cache[u["utt_id"]] = round(float(d), 3)
    if todo: json.dump(cache, open(cp + f".tmp{os.getpid()}", "w")); os.replace(cp + f".tmp{os.getpid()}", cp)
    for u in utts: u["dur_s"] = cache[u["utt_id"]]
    print(f"    {split}: 길이 {len(todo)} 개 읽음 (캐시 {len(utts) - len(todo)})", flush=True)
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

# ───────────────────────────── EN open CSV (Switchboard · MNSC) ─────────────────────────────
def csv_utts(corpus, rel_csv, subset, sample_hours=None):
    """`audio_path|script|sampling_rate|duration|script_tn` → 발화 목록. 텍스트는 script_tn 을 asr-tn 규약으로 다시 정규화(멱등), 빈 것·digit 은 quarantine.
    길이는 CSV 의 duration(파일을 열지 않는다). mnsc 는 PART1(낭독) 만. sample_hours 가 있으면 seed 셔플로 그만큼."""
    csvp = os.path.join(a.en_open_root, rel_csv); SRC_FILES.append(csvp); st = collections.Counter(); utts = []
    with open(csvp, encoding="utf-8", errors="replace") as f:
        hdr = f.readline().rstrip("\n").split("|")
        for line in f:
            r = dict(zip(hdr, line.rstrip("\n").split("|")))
            if len(r) != len(hdr): st["bad_row"] += 1; continue
            if corpus == "mnsc" and "ASR-PART1" not in r["audio_path"]: continue
            uid = os.path.splitext(os.path.basename(r["audio_path"]))[0]; text = target_en(r["script_tn"], corpus); fl = target_flags(text, "English")
            if int(r["sampling_rate"]) != 16000: st["not_16k"] += 1; _quarantine(uid, subset, r["script"], text, {"not_16k"}); continue
            if fl: st["quarantined"] += 1; _quarantine(uid, subset, r["script"], text, fl); continue
            utts.append(dict(utt_id=uid, speaker=uid.split("_")[0] if corpus == "switchboard" else None, chapter=None, path=os.path.join(a.en_open_root, r["audio_path"]), text=text, raw=r["script"], dur_s=round(float(r["duration"]), 3))); st["kept"] += 1
    if sample_hours:
        random.Random(a.seed).shuffle(utts); acc, keep = 0.0, []
        for u in utts:
            if acc >= sample_hours * 3600: break
            keep.append(u); acc += u["dur_s"]
        st["sampled_from"] = len(utts); utts = keep
    print(f"    {corpus}/{subset}: {len(utts)} 발화 {sum(u['dur_s'] for u in utts)/3600:.1f} h · " + " · ".join(f"{k}={v}" for k, v in st.items()), flush=True)
    return utts, st

# ───────────────────────────── NIKL 일상대화 ─────────────────────────────
def nikl_utts(subset, sample_hours=None):
    """연도별 JSON 을 훑어 발화(= PCM 파일 하나)를 모은다. original_form → asr-tn nikl 파서. 발화겹침·익명화·불명확·시각 이상은 quarantine. 표본은 대화 단위."""
    from vapasr.data.nikl import index_year, read_dialogue, pcm_path
    st = collections.Counter(); by_dlg = collections.defaultdict(list); jsons = []
    for y in a.nikl_years.split(","):
        idx = index_year(a.nikl_root, y, cache_dir=os.path.join(a.out, "_index"))
        for k, jp in idx.items():
            if not k.startswith("json:"): continue
            jsons.append(jp)
            for u in read_dialogue(jp):
                st["utts"] += 1; p = pcm_path(idx, u["id"])
                if p is None: st["no_pcm_dir"] += 1; continue
                text = target_ko(u["raw"], "nikl"); fl = set(target_flags(text, "Korean", u["raw"], "nikl"))
                if "발화겹침" in u["note"]: fl.add("overlap")
                dur = (u["end"] - u["start"]) if isinstance(u["start"], (int, float)) and isinstance(u["end"], (int, float)) else None
                if dur is None or not (0.1 <= dur <= 60): fl.add("bad_time")
                for k2 in fl: st[f"flag_{k2}"] += 1
                if fl: st["quarantined"] += 1; _quarantine(u["id"], f"{subset}-{y}", u["raw"], text, fl); continue
                by_dlg[u["dialogue"]].append(dict(utt_id=u["id"], speaker=u["speaker"], chapter=u["dialogue"], path=p, text=text, raw=u["raw"], dur_s=round(dur, 3), year=y))
    dlgs = sorted(by_dlg); random.Random(a.seed).shuffle(dlgs); utts = []; acc = 0.0
    for d in dlgs:
        if sample_hours and acc >= sample_hours * 3600: break
        utts += by_dlg[d]; acc += sum(u["dur_s"] for u in by_dlg[d])
    st["dialogues_total"] = len(dlgs); st["dialogues_kept"] = len({u["chapter"] for u in utts}); SRC_FILES.extend(sorted(jsons)[:2000])   # fingerprint: 원문 JSON 2,000 개까지
    print(f"    nikl/{subset}: {len(utts)} 발화 {acc/3600:.1f} h · " + " · ".join(f"{k}={v}" for k, v in st.items()), flush=True)
    return utts, st

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
        elif corpus in ("switchboard", "mnsc", "nikl"):
            hours = a.sample_hours if a.sample_hours else (float(name.split("-")[-1]) if name.split("-")[-1].isdigit() else None)
            utts, st = (nikl_utts(subset, hours) if corpus == "nikl" else csv_utts(corpus, src, subset, hours))
            pre = {"switchboard": "swbd", "mnsc": "mnsc", "nikl": "nikl"}[corpus]; lang = "Korean" if corpus == "nikl" else "English"
            rs = utt_rows(utts, corpus, subset, split, lang, pre)
            if split == "train":
                for r in rs: r["mode"] = "stream"; r["id"] = r["id"].replace("-utt-", "-")       # 발화 = 스트림(kspon 과 동일 규약)
            sub_st = dict(utts=len(utts), utt_hours=round(sum(u["dur_s"] for u in utts) / 3600, 2), **st)
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

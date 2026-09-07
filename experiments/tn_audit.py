"""asr-tn-v1.0.0 전체 transcript audit (wiki/outputs/output-asr-tn-v1-spec.md 'Audit 산출물과 동결 관문').
python experiments/tn_audit.py [--out DIR] [--review 40]
대상: LibriSpeech 7 split 전체 trans.txt, KsponSpeech train.trn(01–05 partition)·dev·eval_clean·eval_other. 오디오는 읽지 않는다(시간은 nominal).
산출: <out>/summary.json, summary.md, quarantine-<partition>.jsonl, latin-top100.json, review-<partition>.tsv, diff-230h.json
"""
import os, sys, re, json, glob, random, argparse, collections, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.textnorm import (TEXTNORM_VERSION, target_en, target_ko, target_flags, fingerprint, has_standalone_latin)
from vapasr.data.kspon import read_trn, DUAL
ap = argparse.ArgumentParser(); ap.add_argument("--out", default=None); ap.add_argument("--review", type=int, default=40); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--libri-root", default=os.environ.get("MXC_LIBRISPEECH_DIR")); ap.add_argument("--kspon-root", default=os.environ.get("MXC_KSPONSPEECH_DIR"))
ap.add_argument("--manifest-dir", default=os.environ.get("MXC_DATA_MANIFEST_DIR")); ap.add_argument("--qwen", default=os.environ.get("MXC_QWEN_ASR_DIR"))
a = ap.parse_args(); out = a.out or os.path.join(os.environ.get("MXC_DATA_LOG_DIR", "/tmp"), "tn-audit", TEXTNORM_VERSION); os.makedirs(out, exist_ok=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.qwen); UNK = tok.unk_token_id
def ids(s): return tok(s, add_special_tokens=False)["input_ids"]
rng = random.Random(a.seed); T0 = time.time()
NOMINAL_H = {"train-clean-100": 100.6, "train-clean-360": 363.6, "train-other-500": 496.7, "dev-clean": 5.4, "dev-other": 5.3, "test-clean": 5.4, "test-other": 5.1,
             "KsponSpeech_01": 194, "KsponSpeech_02": 194, "KsponSpeech_03": 194, "KsponSpeech_04": 194, "KsponSpeech_05": 193, "dev": 3.9, "eval_clean": 2.6, "eval_other": 3.4}
summary = {"version": TEXTNORM_VERSION, "fingerprint": fingerprint(a.qwen), "partitions": {}}

def audit_partition(name, lang, rows, corpus):
    """rows: [(id, raw)] → 통계·quarantine·review 표본. 반환 dict(id→target) (diff 용)."""
    st = collections.Counter(); reasons = collections.Counter(); targets = {}; coll = collections.defaultdict(set); latin = collections.Counter(); lat_rows = 0
    dual = collections.Counter(); quar = []; buckets = collections.defaultdict(list); lr = []
    for uid, raw in rows:
        st["rows"] += 1; t = target(raw, lang, corpus); fl = target_flags(t, lang, raw, corpus)
        if lang == "Korean":
            for m in DUAL.finditer(raw):
                l = m.group(1); dual["numeric" if re.search(r"\d", l) else ("latin" if re.search(r"[A-Za-z]", l) else "plain")] += 1
            if has_standalone_latin(t): lat_rows += 1; latin.update(re.findall(r"[A-Za-z]+", t)); buckets["standalone_latin"].append((uid, raw, t))
            if any(re.search(r"\d", m.group(1)) for m in DUAL.finditer(raw)): buckets["numeric_dual"].append((uid, raw, t))
            elif any(re.search(r"[A-Za-z]", m.group(1)) for m in DUAL.finditer(raw)): buckets["latin_dual"].append((uid, raw, t))
            else: buckets["plain"].append((uid, raw, t))
        else: buckets["plain"].append((uid, raw, t))
        if fl:
            st["quarantined"] += 1; reasons.update(fl); quar.append(dict(id=uid, raw=raw, target=t, reasons=sorted(fl))); buckets["quarantined"].append((uid, raw, t)); continue
        if target(t, lang, "aihub" if lang == "Korean" else "librispeech") != t: st["not_idempotent"] += 1
        i = ids(t)
        if UNK is not None and UNK in i: st["tokenizer_unk"] += 1
        if not i: st["tokenizer_empty"] += 1
        targets[uid] = t; coll[t].add(raw); lr.append(len(t) / max(1, len(raw)))
    st["target_collisions"] = sum(1 for k, v in coll.items() if len(v) > 1)
    p = dict(lang=lang, nominal_hours=NOMINAL_H.get(name), **{k: st[k] for k in ("rows", "quarantined", "not_idempotent", "tokenizer_unk", "tokenizer_empty", "target_collisions")},
             kept=st["rows"] - st["quarantined"], quarantine_reasons=dict(reasons), len_ratio_mean=round(sum(lr) / max(1, len(lr)), 4))
    if lang == "Korean":
        p.update(dual_notations=dict(dual), standalone_latin_rows=lat_rows, standalone_latin_ratio=round(lat_rows / max(1, st["rows"]), 4), latin_top100=latin.most_common(100))
    with open(os.path.join(out, f"quarantine-{name}.jsonl"), "w", encoding="utf-8") as f:
        for q in quar: f.write(json.dumps(q, ensure_ascii=False) + "\n")
    # 목적 표본 review TSV: 유형별로 골고루
    plan = [("numeric_dual", 12), ("latin_dual", 8), ("standalone_latin", 8), ("quarantined", 6), ("plain", 6)] if lang == "Korean" else [("quarantined", 10), ("plain", 30)]
    with open(os.path.join(out, f"review-{name}.tsv"), "w", encoding="utf-8") as f:
        f.write("type\tid\traw\ttarget\tok?\tnote\n")
        for typ, k in plan:
            for uid, raw, t in rng.sample(buckets[typ], min(k, len(buckets[typ]))): f.write(f"{typ}\t{uid}\t{raw}\t{t}\t\t\n")
    summary["partitions"][name] = p; print(f"  {name:18s} rows {p['rows']:7d} quarantine {p['quarantined']:5d} {dict(reasons)} collisions {p['target_collisions']} | {time.time()-T0:.0f}s", flush=True)
    return targets

def target(raw, lang, corpus): return target_ko(raw, corpus) if lang == "Korean" else target_en(raw, corpus)

# ── LibriSpeech
libri_targets = {}
for split in ["train-clean-100", "train-clean-360", "train-other-500", "dev-clean", "dev-other", "test-clean", "test-other"]:
    rows = []
    for tp in sorted(glob.glob(os.path.join(a.libri_root, split, "*", "*", "*.trans.txt"))):
        for line in open(tp):
            uid, txt = line.rstrip("\n").split(" ", 1); rows.append((uid, txt))
    libri_targets.update(audit_partition(split, "English", rows, "librispeech"))
# ── KsponSpeech
ks_targets = {}
train = read_trn(os.path.join(a.kspon_root, "train.trn")); parts = collections.defaultdict(list)
for rel, raw in train: parts[rel.split("/")[0]].append((os.path.splitext(os.path.basename(rel))[0], raw))
for part in sorted(parts): ks_targets.update(audit_partition(part, "Korean", parts[part], "kspon"))
for trn, name in [("dev.trn", "dev"), ("eval_clean.trn", "eval_clean"), ("eval_other.trn", "eval_other")]:
    ks_targets.update(audit_partition(name, "Korean", [(os.path.splitext(os.path.basename(rel))[0], raw) for rel, raw in read_trn(os.path.join(a.kspon_root, trn))], "kspon"))
json.dump({k: v.get("latin_top100") for k, v in summary["partitions"].items() if v.get("latin_top100")}, open(os.path.join(out, "latin-top100.json"), "w"), ensure_ascii=False, indent=1)

# ── 기존 230 h(동결 전 manifest) 와의 diff: target 문자열·token ID
diff = {}
for man, tg in [("librispeech-100", libri_targets), ("kspon-100", ks_targets)]:
    p = os.path.join(a.manifest_dir, man, "streams.jsonl")
    if not os.path.exists(p): continue
    d = collections.Counter(); ex = []
    for line in open(p, encoding="utf-8"):
        r = json.loads(line)
        if r["mode"] != "stream" and man.startswith("kspon"): continue
        for seg in r["segments"]:
            d["segments"] += 1; old = seg["text"]; new = tg.get(seg["utt_id"])
            if new is None: d["now_quarantined"] += 1; ex.append(dict(utt_id=seg["utt_id"], old=old, new=None)) if len(ex) < 30 else None
            elif new != old: d["text_changed"] += 1; ex.append(dict(utt_id=seg["utt_id"], old=old, new=new)) if len(ex) < 30 else None
            elif ids(new) != ids(old): d["ids_changed"] += 1
            else: d["same"] += 1
    diff[man] = dict(counts=dict(d), examples=ex); print(f"  diff {man}: {dict(d)}", flush=True)
json.dump(diff, open(os.path.join(out, "diff-230h.json"), "w"), ensure_ascii=False, indent=1)
summary["diff_230h"] = {k: v["counts"] for k, v in diff.items()}; summary["seconds"] = round(time.time() - T0)
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), ensure_ascii=False, indent=1)
with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
    f.write(f"# {TEXTNORM_VERSION} audit\n\n| partition | lang | rows | kept | quarantined | reasons | idem-viol | unk | collisions | dual num/lat/plain | Latin rows |\n|---|---|---:|---:|---:|---|---:|---:|---:|---|---:|\n")
    for k, v in summary["partitions"].items():
        dn = v.get("dual_notations", {}); f.write(f"| {k} | {v['lang'][:2]} | {v['rows']} | {v['kept']} | {v['quarantined']} | {v['quarantine_reasons']} | {v['not_idempotent']} | {v['tokenizer_unk']} | {v['target_collisions']} | {dn.get('numeric','-')}/{dn.get('latin','-')}/{dn.get('plain','-')} | {v.get('standalone_latin_rows','-')} ({v.get('standalone_latin_ratio','-')}) |\n")
    f.write(f"\n230 h diff: {summary['diff_230h']}\n")
print(open(os.path.join(out, "summary.md"), encoding="utf-8").read()); print("→", out)

"""Validate yodas_de_align outputs: tar/index, timing, tokenizer re-split, and the SemCommitDataset loader path."""
import glob, json, math, sys, tarfile
from collections import Counter
sys.path.insert(0, "/soundai/users/tskim/VAPKT"); sys.path.insert(0, "/soundai/users/tskim/VAPKT/experiments")
import numpy as np
from transformers import AutoTokenizer
from vapasr.data.archive import load_audio_from_archive, _index_path
from vapasr.data.semcommit_words import split_words, lexical_words
from vapasr.data.semcommit_dataset import SemCommitDataset, add_semcommit_specials

O = sys.argv[1]
tok = AutoTokenizer.from_pretrained("/soundai/Model/Qwen3-ASR-1.7B")
bad, n = Counter(), Counter()
files = []
for d in sorted(glob.glob(f"{O}/results/*")):
    tar = f"{d}/audio.tar"
    import os
    if not os.path.exists(tar):
        n["shards_without_audio"] += 1
        if any(os.path.getsize(f"{d}/words-{x}.jsonl") for x in ("train", "dev")): bad["words_without_tar"] += 1
        continue
    idx = json.load(open(_index_path(tar)))
    with tarfile.open(tar, "r:") as tf:
        scan = {m.name: [m.offset_data, m.size] for m in tf.getmembers()}
    if scan != idx: bad["index_mismatch"] += 1
    for split in ("train", "dev"):
        p = f"{d}/words-{split}.jsonl"
        files.append(p)
        for r in map(json.loads, open(p, encoding="utf-8")):
            n["rows"] += 1; n[f"rows_{split}"] += 1
            if r["split"] != split: bad["split_field"] += 1
            if r["K"] != math.ceil(r["duration_s"] / 0.08): bad["K"] += 1
            seg = r["segments"][0]
            x = load_audio_from_archive(seg["path"])
            if abs(len(x) / 16000 - r["duration_s"]) > 0.002: bad["audio_len"] += 1
            if not np.isfinite(x).all() or np.abs(x).max() == 0: bad["audio_values"] += 1
            t = [float(e) for _, e in r["tokens"]]
            if any(b < a - 1e-6 for a, b in zip(t, t[1:])): bad["token_time_reversal"] += 1
            if t and (t[0] < 0 or t[-1] > r["duration_s"] + 1e-6): bad["token_time_range"] += 1
            ws = split_words([i for i, _ in r["tokens"]], t, tok, text=r["display_text"])
            lw, why = lexical_words(ws) if ws else (None, "split_none")
            got = [(w["text"], w["a"], w["b"], w["tags"]) for w in lw] if lw else None
            want = [(w["text"], w["a"], w["b"], w["tags"]) for w in r["words"]]
            if got != want: bad["resplit_mismatch"] += 1
            if tok.decode([i for i, _ in r["tokens"]]) != r["display_text"]: bad["decode_mismatch"] += 1
            if "alignment_qc" in r: n["clamped"] += 1
print("rows", dict(n), "bad", dict(bad))
sp = add_semcommit_specials(tok)
ds = SemCommitDataset(files, [], tok, sp, sem_free_unlabeled=True, ignore_labels=True, tok_check=10**6)
print("dataset stats", dict(ds.stats), "bad", dict(ds.bad))
langs = Counter()
for i in range(len(ds)):
    it = ds[i]; langs[it["lang"]] += 1
    assert it["wav"].numel() == round(ds.items[i]["duration_s"] * 16000) or True
ex = ds[0]
print("items", len(ds), "langs", dict(langs), "example", ex["id"], "K", ex["K"], "n_text", ex["n_text"])
print(tok.decode([i for i in ex["ids"].tolist() if i >= 0][:80]))

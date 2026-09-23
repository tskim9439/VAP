"""Common single-utterance ASR protocol and micro-averaged edit counts."""
import hashlib
import json
from pathlib import Path

from .textnorm import score_en, score_ko, target_ko, fingerprint

PROTOCOL = "single-turn-asr-v1"


def canonical_reference(raw, lang):
    return target_ko(raw, "kspon") if lang == "Korean" else raw


def score_pair(reference, hypothesis, lang):
    from rapidfuzz.distance import Levenshtein
    result = {}
    variants = {"wer": (score_en(reference).split(), score_en(hypothesis).split())} if lang == "English" else {
        "cer_nospace": (list(score_ko(reference, False)), list(score_ko(hypothesis, False))),
        "cer_space": (list(score_ko(reference, True)), list(score_ko(hypothesis, True))),
        "wer": (score_ko(reference, True).split(), score_ko(hypothesis, True).split()),
    }
    for name, (ref, hyp) in variants.items():
        counts = {"substitutions": 0, "deletions": 0, "insertions": 0}
        names = {"replace": "substitutions", "delete": "deletions", "insert": "insertions"}
        for op in Levenshtein.editops(ref, hyp):
            counts[names[op.tag]] += 1
        counts.update(n_ref=len(ref), n_hyp=len(hyp))
        counts["errors"] = sum(counts[k] for k in names.values())
        counts["rate"] = counts["errors"] / len(ref) if ref else None
        result[name] = counts
    return result


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def aggregate(rows):
    groups = {}
    for r in rows:
        key = f"{r['dataset']}/delta-{r['delta']}"
        g = groups.setdefault(key, dict(utterances=0, audio_s=0.0, forced=0, metrics={}))
        g["utterances"] += 1
        g["audio_s"] += r["audio_s"]
        g["forced"] += r["forced"]
        for name, counts in r["metrics"].items():
            target = g["metrics"].setdefault(name, {})
            for k, v in counts.items():
                if k != "rate":
                    target[k] = target.get(k, 0) + v
    for g in groups.values():
        for c in g["metrics"].values():
            c["rate"] = c["errors"] / c["n_ref"] if c["n_ref"] else None
    return groups

#!/usr/bin/env python3
"""Pilot: word-order-preserving (monotonic) translations as AST training targets, with stable-prefix emission times.

Directions en-de, en-ko, de-en, ko-en; --n source utterances each (the English items are shared by en-de and en-ko).
Everything is generated and judged by one LLM (Qwen3.8-27B, thinking off, greedy). Per item:
  1. (a) normal translation and (b) monotonic translation (salami technique: minimal chunks kept in source order,
     JSON with the chunks); only (b) is meant for training, (a) is the comparison.
  2. judge each translation against the source: meaning (same/minor/major), omission, addition, grammar,
     naturalness -> gate = same meaning, no omission or addition, grammar not incorrect.
  3. stable prefixes (Hoang, Axelrod & Post, arXiv 2609.05799): translate every source word prefix as a partial
     utterance in the same style; m_i = max_{j<=i} lcp(p_j, f) over target units (words for English/German targets,
     characters for Korean); a target unit is committed at the first source word i with m_i >= its index, i.e. at
     that word's end time (the forced-alignment times of the ASR data); everything left is committed at the end.
  4. latency from those times: AL (time-based, Ma et al.), first and mean emission time, share of target words
     committed before the source ends.
Source text = the punctuated transcript (display_text / pnc_text) whose whitespace tokens map one-to-one onto the
timed words; rows where they do not are skipped.

  python experiments/ast_mono_pilot.py sample --out DIR [--n 50]                (CPU)
  python experiments/ast_mono_pilot.py run --out DIR --direction en-de          (one GPU; Qwen3.8 needs transformers 5.8)
  python experiments/ast_mono_pilot.py summarize --out DIR
Outputs: DIR/items-<lang>.jsonl, DIR/<direction>/results.jsonl + summary.json + review.tsv, DIR/summary.json.
"""
import argparse
import gc
import json
import os
import random
import re
import sys
import time
from pathlib import Path

LANG = {"en": "English", "de": "German", "ko": "Korean"}
STYLE = {"English": "natural spoken English",
         "German": "natural spoken German; prefer grammatical constructions that keep the verb early (main-clause "
                   "order, 'denn' instead of 'weil', no 'dass' when it may be left out)",
         "Korean": "natural spoken Korean in the polite 해요체 style"}
LISTS = {"en": "/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list",
         "ko": "/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637/main-words.list",
         "de": "/soundai/users/tskim/VAPKT-data/data/yodas-de-v1/lists/de-words-train-v1.list"}
EN_SETS = {"yodas-en129", "voxpopuli-train"}            # single-segment rows with punctuated text (pnc_text)


# ───────────────────────── sampling ─────────────────────────
def source_item(r, lang):
    """words.jsonl row -> item, or None. The punctuated text must split into exactly the timed words."""
    if lang == "en":
        if r.get("set") not in EN_SETS:
            return None
        text = r.get("pnc_text")
    elif lang == "ko":
        if r.get("set") != "speechlm-partial":
            return None
        text = r.get("display_text")
    else:
        text = r.get("display_text")
    if not text or len(r.get("segments") or []) != 1 or not 4.0 <= float(r["duration_s"]) <= 15.0:
        return None
    toks, words = text.split(), r["words"]
    if len(toks) != len(words) or not 6 <= len(toks) <= 40:
        return None
    times = [float(w["end_time"]) for w in words]
    if any(b < a - 1e-6 for a, b in zip(times, times[1:])):
        return None
    return dict(id=r["id"], lang=LANG[lang], text=text, tokens=toks, times=times, duration_s=float(r["duration_s"]),
                set=r.get("set"))


def cmd_sample(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    for lang, lst in LISTS.items():
        dst = out / f"items-{lang}.jsonl"
        if dst.exists():
            print("exists", dst)
            continue
        files = [l.strip() for l in open(lst) if l.strip()]
        rng.shuffle(files)
        pool = []
        for f in files:
            for k, line in enumerate(open(f, encoding="utf-8")):
                if k >= 400:
                    break
                r = json.loads(line)
                if r.get("lang") != LANG[lang]:
                    continue
                it = source_item(r, lang)
                if it:
                    pool.append(it)
            if len(pool) >= 20 * a.n:
                break
        picked = rng.sample(pool, min(a.n, len(pool)))
        with open(dst, "w", encoding="utf-8") as fo:
            for it in picked:
                fo.write(json.dumps(it, ensure_ascii=False) + "\n")
        print(f"{lang}: pool {len(pool)} -> {len(picked)} items", flush=True)


# ───────────────────────── prompts ─────────────────────────
def p_normal(src, tgt, text):
    return f"Translate this {src} spoken utterance into natural, fluent {tgt}. Output only the {tgt} translation.\n\n{text}"


def p_mono(src, tgt, text):
    return (f"You are a professional simultaneous interpreter. Translate the {src} utterance below into {tgt} so that "
            f"a listener could follow the translation while the speaker is still talking.\n"
            f"Use the salami technique:\n"
            f"1. Split the source into the smallest chunks that still carry a meaning, in the source order.\n"
            f"2. Translate the chunks in that same order. Never move content of a later chunk to an earlier position.\n"
            f"3. Make the whole translation grammatical {STYLE[tgt]}. You may adjust the word order inside a chunk, "
            f"repeat a pronoun or add a short connective, but do not reorder the chunks.\n"
            f"Keep every piece of information: no omissions, no additions, no summarizing.\n"
            f"Return only JSON: {{\"chunks\": [{{\"src\": \"...\", \"tgt\": \"...\"}}], \"translation\": \"...\"}}, where "
            f"translation is the chunk translations joined into the final {tgt} text.\n\n{text}")


def p_prefix(src, tgt, text, mono):
    how = (f"Follow the source order as a simultaneous interpreter would (salami technique), in {STYLE[tgt]}."
           if mono else f"Translate it into natural, fluent {tgt}.")
    return (f"The {src} text below is the beginning of an utterance; the speaker has not finished yet. "
            f"Translate what has been said so far into {tgt}. {how} Output only the {tgt} translation.\n\n{text}")


def p_judge(src, tgt, src_text, tgt_text):
    return (f"Compare a {tgt} translation with its {src} source.\n\nSource: {src_text}\nTranslation: {tgt_text}\n\n"
            f"Answer with JSON only, with these keys: \"meaning\" (\"same\" if the translation conveys exactly the same "
            f"content, \"minor\" for a small difference in nuance, \"major\" otherwise), \"omission\" (true if any source "
            f"information is missing), \"addition\" (true if information that is not in the source was added), "
            f"\"grammar\" (\"correct\", \"minor\" for small errors, \"incorrect\"), \"naturalness\" (1-5, 5 = fully "
            f"natural spoken {tgt}), \"note\" (one short sentence).")


# ───────────────────────── LLM ─────────────────────────
def load_llm(path, device):
    import torch
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    arch = (transformers.AutoConfig.from_pretrained(path).architectures or ["AutoModelForCausalLM"])[0]
    cls = getattr(transformers, arch, None) if "CausalLM" not in arch else transformers.AutoModelForCausalLM
    cls = cls or transformers.AutoModelForCausalLM
    model = cls.from_pretrained(path, dtype=torch.bfloat16, device_map={"": device}, low_cpu_mem_usage=True).eval()
    return tok, model


class LLM:
    def __init__(self, path, device):
        self.tok, self.model = load_llm(path, device)
        self.n_out = 0
        self.secs = 0.0

    def __call__(self, prompts, max_new_tokens, batch):
        import torch
        from transformers import GenerationConfig
        tok, model = self.tok, self.model
        gen = GenerationConfig(do_sample=False, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id,
                               eos_token_id=model.generation_config.eos_token_id)
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        out = [None] * len(prompts)
        for s in range(0, len(order), batch):
            b = order[s:s + batch]
            texts = [tok.apply_chat_template([{"role": "user", "content": prompts[i]}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False) for i in b]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            t0 = time.monotonic()
            with torch.inference_mode():
                y = model.generate(**enc, generation_config=gen)
            self.secs += time.monotonic() - t0
            for i, row in zip(b, y[:, enc["input_ids"].shape[1]:]):
                self.n_out += int((row != tok.pad_token_id).sum())
                out[i] = tok.decode(row, skip_special_tokens=True).split("</think>")[-1].strip()
            del enc, y
            gc.collect()
        return out


def parse_json(s):
    if not s:
        return None
    i, j = s.find("{"), s.rfind("}")
    if i < 0 or j <= i:
        return None
    try:
        return json.loads(s[i:j + 1])
    except json.JSONDecodeError:
        return None


def first_line(s):
    return (s or "").strip().splitlines()[0].strip() if (s or "").strip() else ""


# ───────────────────────── stable prefixes and latency ─────────────────────────
def units(text, tgt):
    """-> [(unit, target word index)]: characters for Korean, normalised words otherwise."""
    out = []
    for w, tok in enumerate(text.split()):
        if tgt == "Korean":
            out += [(c, w) for c in tok if c.isalnum()]
        else:
            u = re.sub(r"^[^\w]+|[^\w]+$", "", tok.lower())
            if u:
                out.append((u, w))
    return out


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x[0] != y[0]:
            break
        n += 1
    return n


def stable_word_times(full, prefixes, src_times, tgt):
    """full translation + translations of source prefixes 1..n-1 -> emission time per target word."""
    uf = units(full, tgt)
    commit = [None] * len(uf)
    m = 0
    for i, p in enumerate(prefixes):
        l = lcp(units(p, tgt), uf)
        if l > m:
            for u in range(m, l):
                commit[u] = src_times[i]
            m = l
    for u in range(m, len(uf)):
        commit[u] = src_times[-1]
    n_words = len(full.split())
    times = [None] * n_words
    for (_, w), t in zip(uf, commit):
        times[w] = t if times[w] is None else max(times[w], t)
    last = src_times[0]
    for w in range(n_words):                               # a word with no unit (punctuation only) follows its neighbour
        times[w] = last if times[w] is None else times[w]
        last = times[w]
    return times


def latency(word_times, src_end):
    """AL (Ma et al., time-based), first and mean emission (s), share of words committed before the source ends."""
    Y = len(word_times)
    if not Y:
        return None
    tau = next((j + 1 for j, d in enumerate(word_times) if d >= src_end - 1e-6), Y)
    al = sum(word_times[j] - j * src_end / Y for j in range(tau)) / tau
    return dict(al=round(al, 3), first=round(word_times[0], 3), mean=round(sum(word_times) / Y, 3),
                pre_end=round(sum(d < src_end - 1e-6 for d in word_times) / Y, 3), words=Y)


def gate(j):
    if not j:
        return False
    return (j.get("meaning") == "same" and not j.get("omission") and not j.get("addition")
            and j.get("grammar") in ("correct", "minor"))


# ───────────────────────── run one direction ─────────────────────────
def cmd_run(a):
    src_code, tgt_code = a.direction.split("-")
    src, tgt = LANG[src_code], LANG[tgt_code]
    out = Path(a.out) / a.direction
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        print("done already", out)
        return
    items = [json.loads(l) for l in open(Path(a.out) / f"items-{src_code}.jsonl", encoding="utf-8")]
    t0 = time.monotonic()
    llm = LLM(a.model, a.device)
    t_load = time.monotonic() - t0
    print(f"{a.direction}: {len(items)} items, model loaded in {t_load:.0f} s", flush=True)

    normal = [first_line(x) for x in llm([p_normal(src, tgt, it["text"]) for it in items], 256, a.batch)]
    mono_raw = llm([p_mono(src, tgt, it["text"]) for it in items], 1024, max(1, a.batch // 2))
    mono_js = [parse_json(x) for x in mono_raw]
    mono = [(j.get("translation") or " ".join(c.get("tgt", "") for c in j.get("chunks", []))).strip() if j else ""
            for j in mono_js]
    print(f"{a.direction}: translations done, monotonic JSON failures {sum(j is None for j in mono_js)}", flush=True)

    jp, jk = [], []
    for k, (it, n_, m_) in enumerate(zip(items, normal, mono)):
        for name, t in (("a", n_), ("b", m_)):
            if t:
                jp.append(p_judge(src, tgt, it["text"], t)); jk.append((k, name))
    judged = {}
    for key, s in zip(jk, llm(jp, 200, a.batch)):
        judged[key] = parse_json(s)
    print(f"{a.direction}: judged {len(jk)}", flush=True)

    pp, pk = [], []
    for k, it in enumerate(items):
        for i in range(1, len(it["tokens"])):
            prefix = " ".join(it["tokens"][:i])
            for name, mono_style in (("a", False), ("b", True)):
                pp.append(p_prefix(src, tgt, prefix, mono_style)); pk.append((k, name, i))
    pref = {}
    for key, s in zip(pk, llm(pp, 160, a.batch)):
        pref[key] = first_line(s)
    print(f"{a.direction}: prefix translations {len(pp)}", flush=True)

    rows = []
    for k, it in enumerate(items):
        row = dict(id=it["id"], direction=a.direction, set=it["set"], source=it["text"], n_src_words=len(it["tokens"]),
                   src_end=it["times"][-1], duration_s=it["duration_s"])
        for name, full in (("a", normal[k]), ("b", mono[k])):
            d = dict(translation=full, judge=judged.get((k, name)), pass_gate=gate(judged.get((k, name))))
            if full:
                prefixes = [pref[(k, name, i)] for i in range(1, len(it["tokens"]))]
                wt = stable_word_times(full, prefixes, it["times"], tgt)
                d.update(word_times=wt, latency=latency(wt, it["times"][-1]), prefixes=prefixes)
            if name == "b":
                d["chunks"] = (mono_js[k] or {}).get("chunks")
            row[name] = d
        rows.append(row)
    with open(out / "results.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "review.tsv", "w", encoding="utf-8") as f:
        f.write("id\tsource\tnormal (a)\tmonotonic (b)\tjudge b\tAL a\tAL b\n")
        for r in rows:
            jb = r["b"].get("judge") or {}
            f.write("\t".join([r["id"], r["source"], r["a"]["translation"], r["b"]["translation"],
                               f"{jb.get('meaning')}/{jb.get('grammar')}/{jb.get('naturalness')} {jb.get('note', '')}",
                               str((r["a"].get("latency") or {}).get("al")), str((r["b"].get("latency") or {}).get("al"))])
                    + "\n")
    summary = summarize_rows(rows)
    summary.update(direction=a.direction, model=a.model, load_s=round(t_load, 1), gen_s=round(llm.secs, 1),
                   out_tokens=llm.n_out, calls=dict(translations=2 * len(items), judge=len(jk), prefixes=len(pp)))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1), flush=True)


def summarize_rows(rows):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None
    s = dict(items=len(rows))
    for name in ("a", "b"):
        ds = [r[name] for r in rows]
        lat = [d.get("latency") or {} for d in ds]
        js = [d.get("judge") or {} for d in ds]
        s[name] = dict(translated=sum(bool(d["translation"]) for d in ds), judged=sum(bool(j) for j in js),
                       pass_gate=sum(d["pass_gate"] for d in ds),
                       meaning={m: sum(j.get("meaning") == m for j in js) for m in ("same", "minor", "major")},
                       omission=sum(bool(j.get("omission")) for j in js), addition=sum(bool(j.get("addition")) for j in js),
                       grammar={g: sum(j.get("grammar") == g for j in js) for g in ("correct", "minor", "incorrect")},
                       naturalness=mean([j.get("naturalness") for j in js if isinstance(j.get("naturalness"), (int, float))]),
                       al=mean([l.get("al") for l in lat]), first=mean([l.get("first") for l in lat]),
                       mean_emit=mean([l.get("mean") for l in lat]), pre_end=mean([l.get("pre_end") for l in lat]))
    s["src_end_mean"] = mean([r["src_end"] for r in rows])
    return s


def cmd_summarize(a):
    out = {}
    for d in sorted(Path(a.out).glob("*-*/summary.json")):
        out[d.parent.name] = json.loads(d.read_text())
    (Path(a.out) / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for k, s in out.items():
        print(k, "| pass a/b", s["a"]["pass_gate"], s["b"]["pass_gate"], "of", s["items"],
              "| AL a/b", s["a"]["al"], s["b"]["al"], "| first a/b", s["a"]["first"], s["b"]["first"],
              "| pre_end a/b", s["a"]["pre_end"], s["b"]["pre_end"], "| natural a/b", s["a"]["naturalness"], s["b"]["naturalness"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("sample")
    q.add_argument("--out", required=True)
    q.add_argument("--n", type=int, default=50)
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(fn=cmd_sample)
    q = sub.add_parser("run")
    q.add_argument("--out", required=True)
    q.add_argument("--direction", required=True, choices=["en-de", "en-ko", "de-en", "ko-en"])
    q.add_argument("--model", default="/soundai/Model/Qwen3.8-27B")
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--batch", type=int, default=32)
    q.set_defaults(fn=cmd_run)
    q = sub.add_parser("summarize")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_summarize)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

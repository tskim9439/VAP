#!/usr/bin/env python3
"""Pilot: word-order-preserving (monotonic) translations as AST training targets, with stable-prefix emission times.

Directions en-de, en-ko, de-en, ko-en; --n source utterances each (the English items are shared by en-de and en-ko).
Everything is generated and judged by one LLM (Qwen3.8-27B, thinking off, greedy). Per item:
  1. (a) normal translation and (b) monotonic translation (salami technique: minimal chunks kept in source order,
     JSON with the chunks); only (b) is meant for training, (a) is the comparison.
  2. judge each translation against the source: content (same/minor/major), omission, addition, grammar,
     naturalness -> gate = content same or minor, no omission or addition, grammar not incorrect.
  3. stable prefixes (Hoang, Axelrod & Post, arXiv 2609.05799): translate every source word prefix as a partial
     utterance in the same style; m_i = max_{j<=i} lcp(p_j, f) over target units (words for English/German targets,
     characters for Korean); a target unit is committed at the first source word i with m_i >= its index, i.e. at
     that word's end time (the forced-alignment times of the ASR data); everything left is committed at the end.
  4. latency from those times: AL (time-based, Ma et al.), first and mean emission time, share of target words
     committed before the source ends.
Source text = the punctuated transcript (display_text / pnc_text) whose whitespace tokens map one-to-one onto the
timed words; rows where they do not are skipped.
Prompt v2 (after the v1 pilot, 2026-10-08): the full translation and the prefix translations use ONE template (plain text,
"complete" vs "not finished yet"), as in the paper — v1 made the monotonic full translation with a JSON chunk prompt, so
its wording differed from the prefix translations and the stable prefix committed words too late. The monotonic
instruction now demands complete grammatical sentences (split a sentence or reorder inside a clause rather than break
grammar) and shows one example per direction. The judge separates content errors from style (gate: no major content
error, no omission or addition, grammar not incorrect).

  python experiments/ast_mono_pilot.py sample --out DIR [--n 50]                (CPU)
  python experiments/ast_mono_pilot.py run --out DIR --direction en-de          (one GPU; Qwen3.8 needs transformers 5.8)
      --engine vllm: run with the vLLM env python (scripts/setup-vllm-env.sh) and CUDA_VISIBLE_DEVICES=<one GPU>
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


# ───────────────────────── prompts (v2) ─────────────────────────
PROMPT_VERSION = "v2"
EXAMPLES = {   # one illustrative monotonic translation per direction (not taken from the data)
    ("English", "German"): ("I think that we should postpone the meeting because the client has not sent the data yet.",
                            "Ich denke, wir sollten das Treffen verschieben, denn der Kunde hat die Daten noch nicht geschickt."),
    ("English", "Korean"): ("I will send you the report after the meeting because the deadline is tomorrow.",
                            "보고서는 제가 보내 드릴게요, 회의 끝나고요. 마감이 내일이거든요."),
    ("German", "English"): ("Weil der Kunde die Daten noch nicht geschickt hat, verschieben wir das Treffen.",
                            "Because the client hasn't sent the data yet, we are postponing the meeting."),
    ("Korean", "English"): ("회의가 끝난 뒤에 보고서를 보내 드릴게요.",
                            "After the meeting, I'll send you the report."),
}


def p_translate(src, tgt, text, mono, complete=True):
    """ONE template for the full translation (complete=True) and every prefix translation (complete=False)."""
    state = (f"The {src} utterance below is complete." if complete else
             f"The {src} text below is the beginning of an utterance; the speaker has not finished yet. Translate only "
             f"what has been said so far and do not guess what comes next.")
    if mono:
        ex_src, ex_tgt = EXAMPLES[(src, tgt)]
        how = (f"Translate like a simultaneous interpreter: follow the source order chunk by chunk (salami technique), "
               f"so that the listener hears each piece of information in the same order as the speaker said it. "
               f"A chunk is a clause or phrase that carries a meaning. Every sentence you write must be a complete, "
               f"grammatical sentence of {STYLE[tgt]}; when the source order would break the grammar, split it into two "
               f"sentences or reorder words inside that clause, never produce broken word order. Keep every piece of "
               f"information: no omissions, no additions, no summarizing.\n\nExample ({src} -> {tgt}):\n{ex_src}\n-> {ex_tgt}")
    else:
        how = f"Translate it into natural, fluent {STYLE[tgt]}."
    return f"{state} {how}\n\nOutput only the {tgt} translation.\n\n{text}"


def p_judge(src, tgt, src_text, tgt_text):
    return (f"Compare a {tgt} translation with its {src} source.\n\nSource: {src_text}\nTranslation: {tgt_text}\n\n"
            f"Judge the CONTENT and the GRAMMAR separately; word choice and style are not content errors. "
            f"Answer with JSON only, with these keys: \"content\" (\"same\" if every fact and intention is conveyed, "
            f"\"minor\" if a nuance differs but no fact is wrong or missing, \"major\" if a fact is wrong, changed or "
            f"missing), \"omission\" (true if any source information is missing), \"addition\" (true if information "
            f"that is not in the source was added), \"grammar\" (\"correct\", \"minor\" for a small slip, "
            f"\"incorrect\" for broken sentences or word order), \"naturalness\" (1-5, 5 = fully natural spoken {tgt}), "
            f"\"note\" (one short sentence).")


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


class VLLMEngine:
    """Same interface as LLM, on vLLM (env from scripts/setup-vllm-env.sh); one GPU = CUDA_VISIBLE_DEVICES.
    vLLM schedules all prompts itself, so `batch` is ignored; greedy, thinking off, text only."""

    def __init__(self, path, max_num_seqs=256):
        from vllm import LLM as _LLM
        self.llm = _LLM(model=path, dtype="bfloat16", max_model_len=4096, gpu_memory_utilization=0.90,
                        max_num_seqs=max_num_seqs, limit_mm_per_prompt={"image": 0, "video": 0}, seed=0)
        self.tok = self.llm.get_tokenizer()
        self.n_out = 0
        self.secs = 0.0

    def __call__(self, prompts, max_new_tokens, batch):
        from vllm import SamplingParams
        texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False) for p in prompts]
        t0 = time.monotonic()
        res = self.llm.generate(texts, SamplingParams(temperature=0.0, max_tokens=max_new_tokens), use_tqdm=False)
        self.secs += time.monotonic() - t0
        self.n_out += sum(len(r.outputs[0].token_ids) for r in res)
        return [r.outputs[0].text.split("</think>")[-1].strip() for r in res]


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


def clean(s):
    """Model output -> the translation: first non-empty line, without a leading label or surrounding quotes."""
    lines = [l.strip() for l in (s or "").splitlines() if l.strip()]
    t = lines[0] if lines else ""
    t = re.sub(r"^(translation|übersetzung|번역)\s*:\s*", "", t, flags=re.I)
    return t.strip().strip('"“”„').strip()


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
    return (j.get("content") in ("same", "minor") and not j.get("omission") and not j.get("addition")
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
    items = [json.loads(l) for l in open(Path(a.items or a.out) / f"items-{src_code}.jsonl", encoding="utf-8")]
    t0 = time.monotonic()
    llm = VLLMEngine(a.model, a.max_num_seqs) if a.engine == "vllm" else LLM(a.model, a.device)
    t_load = time.monotonic() - t0
    print(f"{a.direction}: {len(items)} items, {a.engine} model loaded in {t_load:.0f} s", flush=True)

    normal = [clean(x) for x in llm([p_translate(src, tgt, it["text"], False) for it in items], 320, a.batch)]
    mono = [clean(x) for x in llm([p_translate(src, tgt, it["text"], True) for it in items], 320, a.batch)]
    print(f"{a.direction}: translations done, empty a/b {sum(not x for x in normal)}/{sum(not x for x in mono)}", flush=True)

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
                pp.append(p_translate(src, tgt, prefix, mono_style, complete=False)); pk.append((k, name, i))
    pref = {}
    for key, s in zip(pk, llm(pp, 160, a.batch)):
        pref[key] = clean(s)
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
                               f"{jb.get('content')}/{jb.get('grammar')}/{jb.get('naturalness')} {jb.get('note', '')}",
                               str((r["a"].get("latency") or {}).get("al")), str((r["b"].get("latency") or {}).get("al"))])
                    + "\n")
    summary = summarize_rows(rows)
    summary.update(direction=a.direction, prompt_version=PROMPT_VERSION, model=a.model, engine=a.engine,
                   load_s=round(t_load, 1), gen_s=round(llm.secs, 1),
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
                       content={m: sum(j.get("content") == m for j in js) for m in ("same", "minor", "major")},
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
        print(k, s.get("prompt_version"), "| pass a/b", s["a"]["pass_gate"], s["b"]["pass_gate"], "of", s["items"],
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
    q.add_argument("--engine", default="hf", choices=["hf", "vllm"],
                   help="hf = transformers generate (fast GDN path only with fla + causal-conv1d); vllm = one GPU via CUDA_VISIBLE_DEVICES")
    q.add_argument("--max-num-seqs", type=int, default=256, help="vllm: concurrent sequences")
    q.add_argument("--items", default=None, help="folder with items-<lang>.jsonl (default: --out)")
    q.set_defaults(fn=cmd_run)
    q = sub.add_parser("summarize")
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_summarize)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

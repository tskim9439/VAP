#!/usr/bin/env python3
"""Full-scale monotonic AST targets: every Stage 1 ASR training row, four directions (en-de, en-ko, de-en, ko-en).

Same recipe as the v2 pilot (experiments/ast_mono_pilot.py, decision-ast-monotonic-data): Qwen3.8-27B on vLLM, greedy,
thinking off; per source unit one monotonic translation (salami prompt v2), a judge verdict (gate) and a translation of
EVERY source word prefix; the stable prefix (Hoang, Axelrod & Post, arXiv 2609.05799) gives each target word the end
time of the source word at which it became stable. Only monotonic translations are made (no normal translation).
--prompt v3 (chosen 2026-10-09) puts the instruction and examples in a fixed system message per direction that is longer
than one vLLM block (784 tokens for Qwen3.8), so ~93 % of prompt tokens come from the prefix cache: 2.98x faster than v2
on a real shard (with --gpu-memory-utilization 0.95 --max-num-seqs 1024) and +2 points gate pass; fp8 adds 1.2x but
loses ~1.3 points (wiki output-ast-mono-generation-20261009).

Source units. Rows come from the lists of the 0.6B Stage 1 run (q06-s1m3-d10): speechlm main + ASR words (English,
Korean), speechlm short (Korean), YODAS-de tn2 (German); deduplicated by row id. Per row segment the source text is
the first usable of pnc_text, display_text, raw_text (annotation tags removed) and, as the fallback, the timed words
themselves. Text tokens get the forced-alignment end times of the timed words: one to one when the counts match,
otherwise through a character alignment (difflib) of the letters and digits, so "1901" takes the time of
"... nine hundred one". Tokens without letters or digits are glued to their neighbour. A segment is split into
sentences (. ? ! ... followed by a capital or a caseless script; not after abbreviations or German ordinals); a unit
longer than --max-words is split at the comma nearest its middle, else at the longest pause. Units are what is
translated; their times are row (stream) times.

  python experiments/ast_mono_generate.py prepare --root DIR [--procs 48]        (CPU, mxc host)
  python experiments/ast_mono_generate.py run --root DIR                         (one GPU per process; vLLM env;
        rank/world from SLURM_PROCID/SLURM_NTASKS — slurm/ast-mono-generate.sbatch)
  python experiments/ast_mono_generate.py status --root DIR
Layout: DIR/units/<lang>/u-NNNNN.jsonl.gz + DIR/units/manifest.json; DIR/gen/<direction>/u-NNNNN.jsonl.gz (+ .summary.json);
DIR/errors/. A finished work item is its gen file (written as .partial, then renamed); reruns skip finished items.
"""
import argparse
import difflib
import gzip
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ast_mono_pilot as P  # noqa: E402  (prompts v2, judge, gate, stable prefix, latency, VLLMEngine)

SNAP = "/soundai/users/tskim/VAPKT-data/data/semcommit-work/labels/speechlm-all19-v035/snapshots/20260929-1637"
LISTS = {  # the Stage 1 (q06-s1m3-d10) training inputs
    "main": f"{SNAP}/main-words.list",
    "short": f"{SNAP}/short-words.list",
    "asr": "/soundai/users/tskim/VAPKT-data/data/semcommit-work/asr-words-v1/lists/asr-words-20260929.list",
    "de": "/soundai/users/tskim/VAPKT-data/data/yodas-de-v1/lists/de-words-train-tn2.list",
}
CODE = {"English": "en", "Korean": "ko", "German": "de"}
DIRECTIONS = {"en": ["en-de", "en-ko"], "ko": ["ko-en"], "de": ["de-en"]}
TEXT_KEYS = ("pnc_text", "display_text", "raw_text")


# ───────────────────────── source text ─────────────────────────
def clean_text(t):
    """Remove transcription annotations: <LAUGH>, {laughing}, [noise], KsponSpeech o/ b/ l/ n/ u/ tags and the
    (written)/(spoken) pairs (written kept), '+' '*' marks, wiki '==' headings. All-caps text is lowercased."""
    t = re.sub(r"\(([^()]*)\)/\(([^()]*)\)", r"\1", t)
    t = re.sub(r"<[^<>]*>|\{[^{}]*\}|\[[^\[\]]*\]", " ", t)
    t = re.sub(r"(?<!\S)[a-z]/(?!\S)", " ", t)
    t = re.sub(r"[+*]", " ", t)
    t = re.sub(r"={2,}", " ", t)
    t = " ".join(t.split())
    if t and not any(c.islower() for c in t) and any(c.isupper() for c in t):
        t = t.lower()
    return t


def alnum(s):
    return [c for c in s.lower() if c.isalnum()]


def glue(tokens):
    """Tokens without letters or digits join the previous token (the next one at the start)."""
    out = []
    for t in tokens:
        if alnum(t) or not out:
            out.append(t)
        else:
            out[-1] += " " + t
    if len(out) > 1 and not alnum(out[0]):
        out[1] = out[0] + " " + out[1]
        out = out[1:]
    return out


def align_times(tokens, words):
    """tokens (text, glued), words [(text, end_time)] -> (end time per token, share of token letters matched (digits
    left out: they are spelled out in the words), share of word chars matched). One to one when the counts match,
    else a character alignment."""
    if len(tokens) == len(words):
        return [w[1] for w in words], 1.0, 1.0
    A, a_own, B, b_own = [], [], [], []
    for i, t in enumerate(tokens):
        for c in alnum(t):
            A.append(c); a_own.append(i)
    for j, (w, _) in enumerate(words):
        for c in alnum(w):
            B.append(c); b_own.append(j)
    if not A or not B:
        return None, 0.0, 0.0
    mapped, matched, matched_letters = [0] * len(A), 0, 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, A, B, autojunk=False).get_opcodes():
        for k in range(i1, i2):
            if tag == "equal":
                mapped[k] = j1 + k - i1
            elif tag == "replace":                    # proportional; the last char of the block -> the block end
                mapped[k] = j1 + ((k - i1 + 1) * (j2 - j1) - 1) // (i2 - i1)
            else:                                     # delete: nothing spoken for it -> the previous word char
                mapped[k] = max(j1 - 1, 0)
        if tag == "equal":
            matched += i2 - i1
            matched_letters += sum(not c.isdigit() for c in A[i1:i2])
    last = {}
    for k, i in enumerate(a_own):
        last[i] = mapped[k]
    times, t_prev = [], 0.0
    for i in range(len(tokens)):
        t = words[b_own[last[i]]][1] if i in last else t_prev
        t_prev = max(t_prev, t)
        times.append(t_prev)
    letters = sum(not c.isdigit() for c in A)
    return times, matched_letters / letters if letters else 0.0, matched / len(B)


def segment_source(r, si, n_segs):
    """-> (source key, tokens, end times) for one row segment."""
    g = (r.get("segments") or [{}])[si]
    words = [(w["text"], float(w["end_time"])) for w in r["words"] if w.get("seg", 0) == si]
    if not words:
        return None
    for key in TEXT_KEYS:
        t = g.get(key) or (r.get(key) if n_segs == 1 else None)
        if not t:
            continue
        toks = glue(clean_text(t).split())
        if not toks:
            continue
        times, a_share, b_share = align_times(toks, words)
        if times and a_share >= 0.85 and b_share >= 0.6:
            return key, toks, times
    return "words", [w for w, _ in words], [t for _, t in words]


# ───────────────────────── units ─────────────────────────
END = re.compile(r"[.?!…。？！]+[\"'”’»)\]]*$")
ABBR = {"mr.", "mrs.", "ms.", "dr.", "st.", "prof.", "vs.", "etc.", "e.g.", "i.e.", "jr.", "sr.", "no.", "inc.", "ltd.",
        "co.", "z.b.", "bzw.", "ca.", "nr.", "usw.", "d.h.", "u.a.", "vgl.", "evtl.", "ggf.", "bspw.", "sog.", "dt."}


def is_sentence_end(tok, nxt):
    if not END.search(tok):
        return False
    low = tok.lower().rstrip("\"'”’»)]")
    if low in ABBR or re.fullmatch(r"\d+\.", low) or re.fullmatch(r"(?:[^\W\d_]\.){1,3}", low):
        return False
    if nxt is None:
        return True
    first = nxt.lstrip("\"'“„‚‘«([")[:1]
    return not (first.isalpha() and first.islower())    # a lowercase Latin start continues the sentence


def split_long(lo, hi, tokens, times, max_words):
    """[lo, hi) -> list of ranges of at most max_words: cut after the comma nearest the middle, else at the longest
    pause (largest end-time step) inside the middle half."""
    if hi - lo <= max_words:
        return [(lo, hi)]
    n, mid = hi - lo, (lo + hi) / 2
    cand = [c for c in range(lo + max(1, n // 4), hi - max(1, n // 4) + 1) if re.search(r"[,;:–—]$", tokens[c - 1])]
    if cand:
        c = min(cand, key=lambda c: abs(c - mid))
    else:
        rng_ = range(lo + max(1, n // 4), hi - max(1, n // 4) + 1)
        c = max(rng_, key=lambda c: (times[c] - times[c - 1], -abs(c - mid)))
    return split_long(lo, c, tokens, times, max_words) + split_long(c, hi, tokens, times, max_words)


def split_units(tokens, times, max_words):
    out, start = [], 0
    for i, tok in enumerate(tokens):
        if is_sentence_end(tok, tokens[i + 1] if i + 1 < len(tokens) else None) or i == len(tokens) - 1:
            out += split_long(start, i + 1, tokens, times, max_words)
            start = i + 1
    return out


def row_units(r, max_words):
    segs = r.get("segments") or [{}]
    units, srcs = [], Counter()
    for si in range(len(segs)):
        s = segment_source(r, si, len(segs))
        if not s:
            continue
        key, toks, times = s
        srcs[key] += 1
        for lo, hi in split_units(toks, times, max_words):
            units.append(dict(seg=si, tokens=toks[lo:hi], times=[round(t, 3) for t in times[lo:hi]], src=key))
    return units, srcs


def _read_file(job):
    path, langs, max_words = job
    rows = []
    try:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            lang = r.get("lang")
            if lang not in langs:
                continue
            us, srcs = row_units(r, max_words)
            rows.append(dict(id=r["id"], set=r.get("set"), lang=lang, duration_s=float(r["duration_s"]), units=us, srcs=srcs))
        return path, rows, None
    except Exception as exc:  # noqa: BLE001
        return path, rows, f"{type(exc).__name__}: {exc}"


# ───────────────────────── prepare ─────────────────────────
def cmd_prepare(a):
    root = Path(a.root)
    man_path = root / "units" / "manifest.json"
    if man_path.exists():
        sys.exit(f"already prepared: {man_path}")
    jobs = []
    for name, lst in LISTS.items():
        langs = {"German"} if name == "de" else {"English", "Korean"}
        files = [l.strip() for l in open(lst) if l.strip()]
        jobs += [(f, langs, a.max_words) for f in files]
    random.Random(a.seed).shuffle(jobs)                      # shards mix sets
    writers, seen, stats = {}, set(), {}
    shards = {c: [] for c in DIRECTIONS}
    errors = []

    def flush(code, w):
        if not w["units"]:
            return
        d = root / "units" / code
        d.mkdir(parents=True, exist_ok=True)
        name = f"u-{len(shards[code]):05d}"
        tmp = d / f"{name}.jsonl.gz.partial"
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for u in w["units"]:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
        os.replace(tmp, d / f"{name}.jsonl.gz")
        shards[code].append(dict(name=name, units=len(w["units"]), words=w["words"], rows=w["rows"]))
        w.update(units=[], words=0, rows=0)

    t0 = time.monotonic()
    with Pool(a.procs) as pool:
        for n, (path, rows, err) in enumerate(pool.imap(_read_file, jobs, chunksize=1)):
            if err:
                errors.append(dict(file=path, error=err))
            for r in rows:
                if r["id"] in seen:
                    stats.setdefault("_dup", Counter())[r["set"]] += 1
                    continue
                seen.add(r["id"])
                code = CODE[r["lang"]]
                s = stats.setdefault(f"{code}/{r['set']}", Counter())
                s["rows"] += 1; s["hours"] += r["duration_s"] / 3600; s["units"] += len(r["units"])
                s["words"] += sum(len(u["tokens"]) for u in r["units"]); s.update({f"src_{k}": v for k, v in r["srcs"].items()})
                if not r["units"]:
                    s["rows_no_units"] += 1
                    continue
                w = writers.setdefault(code, dict(units=[], words=0, rows=0))
                for k, u in enumerate(r["units"]):
                    w["units"].append(dict(uid=f"{r['id']}#{k}", row=r["id"], set=r["set"], k=k, n=len(r["units"]),
                                           seg=u["seg"], src=u["src"], tokens=u["tokens"], times=u["times"]))
                    w["words"] += len(u["tokens"])
                w["rows"] += 1
                if w["words"] >= a.shard_words:
                    flush(code, w)
            if n % 200 == 0:
                print(f"[{time.monotonic() - t0:.0f} s] files {n + 1}/{len(jobs)} rows {len(seen)}", flush=True)
    for code, w in writers.items():
        flush(code, w)
    stats = {k: {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in sorted(stats.items())}
    man = dict(created=time.strftime("%F %T"), lists=LISTS, max_words=a.max_words, shard_words=a.shard_words,
               seed=a.seed, code_sha256=code_sha(), shards=shards, stats=stats, errors=errors,
               totals={c: dict(shards=len(v), units=sum(x["units"] for x in v), words=sum(x["words"] for x in v),
                               rows=sum(x["rows"] for x in v)) for c, v in shards.items()})
    man_path.write_text(json.dumps(man, ensure_ascii=False, indent=1))
    print(json.dumps(dict(totals=man["totals"], errors=len(errors)), indent=1))


def code_sha():
    h = hashlib.sha256()
    for f in (Path(__file__), Path(P.__file__)):
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


# ───────────────────────── run (GPU) ─────────────────────────
def work_items(man):
    """(direction, shard name) interleaved over shard index so early results cover every direction."""
    items, n = [], max(len(v) for v in man["shards"].values())
    for i in range(n):
        for code, dirs in DIRECTIONS.items():
            if i < len(man["shards"][code]):
                items += [(d, man["shards"][code][i]["name"]) for d in dirs]
    return items


def process(eng, root, direction, name, out_path, prompt="v2"):
    src_code, tgt_code = direction.split("-")
    src, tgt = P.LANG[src_code], P.LANG[tgt_code]
    tr = ((lambda x, c=True: P.p_translate_v3(src, tgt, x, c)) if prompt == "v3"
          else (lambda x, c=True: P.p_translate(src, tgt, x, True, c)))
    units = [json.loads(l) for l in gzip.open(root / "units" / src_code / f"{name}.jsonl.gz", "rt", encoding="utf-8")]
    t0, s0, n0, p0, c0, x0 = time.monotonic(), eng.secs, eng.n_out, eng.n_prompt, eng.n_cached, eng.n_truncated
    prompts, keys = [], []
    for k, u in enumerate(units):
        prompts.append(tr(" ".join(u["tokens"]))); keys.append((k, 0))
        for i in range(1, len(u["tokens"])):
            prompts.append(tr(" ".join(u["tokens"][:i]), False)); keys.append((k, i))
    full, pref = [""] * len(units), [[None] * (len(u["tokens"]) - 1) for u in units]
    # full translations (320 new tokens) and prefixes (160) in one engine call per length budget
    for budget, sel in ((320, [j for j, (_, i) in enumerate(keys) if i == 0]), (160, [j for j, (_, i) in enumerate(keys) if i > 0])):
        if not sel:
            continue
        for j, out in zip(sel, eng([prompts[j] for j in sel], budget, None)):
            k, i = keys[j]
            if i == 0:
                full[k] = P.clean(out)
            else:
                pref[k][i - 1] = P.clean(out)
    jk = [k for k in range(len(units)) if full[k]]
    judged = dict(zip(jk, (P.parse_json(s) for s in eng([P.p_judge(src, tgt, " ".join(units[k]["tokens"]), full[k]) for k in jk], 200, None))))
    rows, n_pass = [], 0
    for k, u in enumerate(units):
        j = judged.get(k)
        r = dict(uid=u["uid"], row=u["row"], set=u["set"], k=u["k"], n=u["n"], seg=u["seg"], src=u["src"],
                 direction=direction, source=" ".join(u["tokens"]), times=u["times"], translation=full[k], judge=j,
                 pass_gate=P.gate(j), prefixes=pref[k])
        if full[k]:
            wt = P.stable_word_times(full[k], pref[k], u["times"], tgt)
            r.update(word_times=wt, latency=P.latency(wt, u["times"][-1]))
        n_pass += r["pass_gate"]
        rows.append(r)
    tmp = out_path.with_name(out_path.name + f".partial-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lat = [r["latency"]["al"] for r in rows if r.get("latency")]
    summ = dict(direction=direction, shard=name, units=len(units), words=sum(len(u["tokens"]) for u in units),
                calls=len(prompts) + len(jk), translated=len(jk), judged=sum(v is not None for v in judged.values()),
                pass_gate=n_pass, al_mean=round(sum(lat) / len(lat), 3) if lat else None,
                out_tokens=eng.n_out - n0, prompt_tokens=eng.n_prompt - p0, cached_tokens=eng.n_cached - c0,
                truncated=eng.n_truncated - x0, gen_s=round(eng.secs - s0, 1), elapsed_s=round(time.monotonic() - t0, 1),
                prompt_version=prompt, engine=eng.config, code_sha256=code_sha(), job=os.environ.get("SLURM_JOB_ID"),
                host=os.uname().nodename, finished=time.strftime("%F %T"))
    out_path.with_name(name + ".summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    os.replace(tmp, out_path)                                  # the gen file appears last = item done
    return summ


def cmd_run(a):
    root = Path(a.root)
    man = json.loads((root / "units" / "manifest.json").read_text())
    rank = a.rank if a.rank is not None else int(os.environ.get("SLURM_PROCID", 0))
    world = a.world if a.world is not None else int(os.environ.get("SLURM_NTASKS", 1))
    mine = work_items(man)[rank::world]
    gen = root / a.gen_dir
    todo = [(d, n) for d, n in mine if not (gen / d / f"{n}.jsonl.gz").exists()]
    print(f"rank {rank}/{world}: {len(mine)} items, {len(todo)} to do ({a.gen_dir}, prompt {a.prompt}, "
          f"quantization {a.quantization})", flush=True)
    if not todo:
        return
    eng = P.VLLMEngine(a.model, a.max_num_seqs, a.quantization, a.gpu_memory_utilization)
    eng.config = dict(quantization=a.quantization, max_num_seqs=a.max_num_seqs, gpu_memory_utilization=a.gpu_memory_utilization)
    for d, n in todo:
        out = gen / d / f"{n}.jsonl.gz"
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            s = process(eng, root, d, n, out, a.prompt)
        except Exception as exc:  # noqa: BLE001  (one bad item must not stop the other ranks)
            err = root / "errors"
            err.mkdir(exist_ok=True)
            (err / f"{d}.{n}.{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}.json").write_text(
                json.dumps(dict(direction=d, shard=n, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()),
                           ensure_ascii=False, indent=1))
            print(f"ITEM_ERROR rank={rank} {d}/{n} {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            continue
        print(f"ITEM_DONE rank={rank} {d}/{n} units={s['units']} pass={s['pass_gate']} calls={s['calls']} "
              f"tok={s['out_tokens']} {s['elapsed_s']} s ({s['out_tokens'] / max(1e-9, s['gen_s']):.0f} tok/s)", flush=True)


# ───────────────────────── fix: judge v3 + repair (GPU) ─────────────────────────
def process_fix(eng, root, gen_dir, direction, name, out_path, validate=False):
    """Units that failed the v2 gate: judge again with judge v3 (fragments and fillers stated, labelled examples); those still
    failing get ONE repair translation (the judge note as feedback) that is judged with v3 again. Emission times of a repaired
    translation come from the original prefix translations (stable prefix against the new full translation).
    A unit the two judges disagree on is accepted as is only when judge v3 says content "same" (plus no omission/addition and
    grammar not incorrect); a v3 "minor" goes to repair instead (2026-10-09 validation: v3 alone also lets some real omissions
    through as "minor"). The repaired translation passes with the normal gate.
    validate=True also re-judges the units that passed v2 (agreement check with the normal gate; nothing is repaired for them).
    final per unit: v2 (passed before) | v3 (passes judge v3, content same) | repair (repaired translation passes) | fail."""
    src_code, tgt_code = direction.split("-")
    src, tgt = P.LANG[src_code], P.LANG[tgt_code]
    rows = [json.loads(l) for l in gzip.open(root / gen_dir / direction / f"{name}.jsonl.gz", "rt", encoding="utf-8")]
    t0, s0, n0 = time.monotonic(), eng.secs, eng.n_out
    pick = rows if validate else [r for r in rows if not r["pass_gate"]]
    jt = [r for r in pick if r["translation"]]
    j1 = dict(zip([r["uid"] for r in jt], (P.parse_json(x) for x in
                                            eng([P.p_judge_v3(src, tgt, r["source"], r["translation"]) for r in jt], 200, None))))
    ok3 = lambda r: P.gate(j1.get(r["uid"])) and (r["pass_gate"] or (j1.get(r["uid"]) or {}).get("content") == "same")
    rep = [r for r in pick if not r["pass_gate"] and not ok3(r)]
    note = lambda r: (j1.get(r["uid"]) or {}).get("note") or (r.get("judge") or {}).get("note")
    new = dict(zip([r["uid"] for r in rep], (P.clean(x) for x in
                                             eng([P.p_repair_v3(src, tgt, r["source"], r["translation"], note(r)) for r in rep], 320, None))))
    rj = [r for r in rep if new[r["uid"]]]
    j2 = dict(zip([r["uid"] for r in rj], (P.parse_json(x) for x in
                                            eng([P.p_judge_v3(src, tgt, r["source"], new[r["uid"]]) for r in rj], 200, None))))
    out, n = [], Counter()
    for r in pick:
        u = r["uid"]
        d = dict(uid=u, row=r["row"], k=r["k"], n=r["n"], set=r["set"], direction=direction, pass_v2=r["pass_gate"],
                 judge_v3=j1.get(u), pass_v3=ok3(r))
        if u in new:
            d.update(repair_translation=new[u], repair_judge=j2.get(u), pass_repair=P.gate(j2.get(u)))
            if new[u]:
                wt = P.stable_word_times(new[u], r["prefixes"], r["times"], tgt)
                d.update(repair_word_times=wt, repair_latency=P.latency(wt, r["times"][-1]))
        d["final"] = "v2" if r["pass_gate"] else "v3" if d["pass_v3"] else "repair" if d.get("pass_repair") else "fail"
        n[d["final"]] += 1
        if validate: n[f"v2{'pass' if r['pass_gate'] else 'fail'}_v3{'pass' if d['pass_v3'] else 'fail'}"] += 1
        out.append(d)
    tmp = out_path.with_name(out_path.name + f".partial-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    summ = dict(direction=direction, shard=name, units=len(rows), picked=len(pick), repaired=len(rep), validate=validate,
                counts=dict(n), out_tokens=eng.n_out - n0, gen_s=round(eng.secs - s0, 1), elapsed_s=round(time.monotonic() - t0, 1),
                prompt_tokens=None, judge="v3", code_sha256=code_sha(), job=os.environ.get("SLURM_JOB_ID"), host=os.uname().nodename,
                finished=time.strftime("%F %T"))
    out_path.with_name(name + ".summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    os.replace(tmp, out_path)
    return summ


def cmd_fix(a):
    root = Path(a.root)
    man = json.loads((root / "units" / "manifest.json").read_text())
    rank = a.rank if a.rank is not None else int(os.environ.get("SLURM_PROCID", 0))
    world = a.world if a.world is not None else int(os.environ.get("SLURM_NTASKS", 1))
    fix = root / a.fix_dir
    items = [(d, n) for d, n in work_items(man) if (root / a.gen_dir / d / f"{n}.jsonl.gz").exists()]
    if a.limit:
        per = Counter(); keep = []
        for d, n in items:
            if per[d] < a.limit:
                keep.append((d, n)); per[d] += 1
        items = keep
    mine = items[rank::world]
    todo = [(d, n) for d, n in mine if not (fix / d / f"{n}.jsonl.gz").exists()]
    print(f"rank {rank}/{world}: {len(mine)} generated items, {len(todo)} to fix ({a.fix_dir}, validate {a.validate})", flush=True)
    if not todo:
        return
    eng = P.VLLMEngine(a.model, a.max_num_seqs, a.quantization, a.gpu_memory_utilization)
    for d, n in todo:
        out = fix / d / f"{n}.jsonl.gz"
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            s = process_fix(eng, root, a.gen_dir, d, n, out, a.validate)
        except Exception as exc:  # noqa: BLE001
            err = root / "errors"
            err.mkdir(exist_ok=True)
            (err / f"fix.{d}.{n}.{os.environ.get('SLURM_JOB_ID', 'local')}.{os.getpid()}.json").write_text(
                json.dumps(dict(direction=d, shard=n, error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()),
                           ensure_ascii=False, indent=1))
            print(f"FIX_ERROR rank={rank} {d}/{n} {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            continue
        print(f"FIX_DONE rank={rank} {d}/{n} picked={s['picked']} repaired={s['repaired']} {json.dumps(s['counts'])} "
              f"{s['elapsed_s']} s", flush=True)


# ───────────────────────── status ─────────────────────────
def cmd_status(a):
    root = Path(a.root)
    man = json.loads((root / "units" / "manifest.json").read_text())
    items = work_items(man)
    words = {(c, s["name"]): s["words"] for c, v in man["shards"].items() for s in v}
    per = {}
    for d, n in items:
        p = per.setdefault(d, Counter())
        p["items"] += 1; p["words"] += words[(d.split("-")[0], n)]
        sp = root / a.gen_dir / d / f"{n}.summary.json"
        if (root / a.gen_dir / d / f"{n}.jsonl.gz").exists() and sp.exists():
            s = json.loads(sp.read_text())
            p["done"] += 1; p["done_words"] += s["words"]; p["units"] += s["units"]; p["pass"] += s["pass_gate"]
            p["gpu_s"] += s["elapsed_s"]; p["tok"] += s["out_tokens"]
    tot_s = tot_w = tot_dw = 0
    for d, p in per.items():
        rate = p["gpu_s"] / p["done_words"] if p["done_words"] else None
        print(f"{d}: items {p['done']}/{p['items']} · units {p['units']} · gate {p['pass'] / max(1, p['units']):.1%} · "
              f"GPU {p['gpu_s'] / 3600:.1f} h" + (f" · est. remaining {(p['words'] - p['done_words']) * rate / 3600:.0f} GPU h" if rate else ""))
        tot_s += p["gpu_s"]; tot_w += p["words"]; tot_dw += p["done_words"]
    if tot_dw:
        print(f"all: {tot_dw / tot_w:.1%} of source words · GPU {tot_s / 3600:.1f} h · est. remaining "
              f"{(tot_w - tot_dw) * tot_s / tot_dw / 3600:.0f} GPU h")
    errs = list((root / "errors").glob("*.json")) if (root / "errors").exists() else []
    print(f"errors: {len(errs)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--root", required=True)
    q.add_argument("--procs", type=int, default=48)
    q.add_argument("--max-words", type=int, default=40)
    q.add_argument("--shard-words", type=int, default=25000, help="source words per shard (~8 GPU min per direction)")
    q.add_argument("--seed", type=int, default=0)
    q.set_defaults(fn=cmd_prepare)
    q = sub.add_parser("run")
    q.add_argument("--root", required=True)
    q.add_argument("--model", default="/soundai/Model/Qwen3.8-27B")
    q.add_argument("--max-num-seqs", type=int, default=512)
    q.add_argument("--quantization", default=None, help="e.g. fp8 (online weight quantization)")
    q.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    q.add_argument("--prompt", default="v2", choices=["v2", "v3"])
    q.add_argument("--gen-dir", default="gen", help="output folder under --root (one per prompt/engine setting)")
    q.add_argument("--rank", type=int, default=None)
    q.add_argument("--world", type=int, default=None)
    q.set_defaults(fn=cmd_run)
    q = sub.add_parser("fix")
    q.add_argument("--root", required=True)
    q.add_argument("--gen-dir", default="gen-v3-bf16")
    q.add_argument("--fix-dir", default="fix-gen-v3-bf16", help="output folder under --root")
    q.add_argument("--validate", action="store_true", help="also re-judge units that passed v2 (agreement check)")
    q.add_argument("--limit", type=int, default=0, help="at most this many generated shards per direction (validation)")
    q.add_argument("--model", default="/soundai/Model/Qwen3.8-27B")
    q.add_argument("--max-num-seqs", type=int, default=1024)
    q.add_argument("--quantization", default=None)
    q.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    q.add_argument("--rank", type=int, default=None)
    q.add_argument("--world", type=int, default=None)
    q.set_defaults(fn=cmd_fix)
    q = sub.add_parser("status")
    q.add_argument("--root", required=True)
    q.add_argument("--gen-dir", default="gen")
    q.set_defaults(fn=cmd_status)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

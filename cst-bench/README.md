# CST-Bench build tools

Build tools for **CST-Bench**, a benchmark for *Conversational Simultaneous Speech-to-Text Translation*:
two people who speak different languages talk to each other, and a system must translate both
directions in real time while handling turn switches, backchannels, interruptions and overlapping speech.

This repository does **not** contain any corpus audio or text. Several source corpora may not be
redistributed, so the benchmark is defined by *build scripts, configs and checksums*: you obtain the
source data under its own license, run the build, and verify that you got exactly the same benchmark.

> Status: v0.1 — TAXI (real German↔English telephone dialogues) and the scorer are supported. The
> synthetic set (CST-Bench-Syn) and the turn-end overlap condition will be added.

## Install

```bash
pip install -e .            # Python >= 3.9; depends on numpy, scipy, soundfile
pip install -e ".[test]" && pytest
```

## Data access

| Corpus | Languages | How to get it | Redistribution |
|---|---|---|---|
| BAS TAXI 2.5 | German (dispatcher) ↔ English (client), telephone 8 kHz | [BAS CLARIN repository](https://clarin.phonetik.uni-muenchen.de/BASRepository/) — log in with your institution or a CLARIN account, open *TAXI*, download. Scientific licence free of charge; commercial licence available from BAS | Not allowed. Only IDs, configs and scripts are released here |

## Build the TAXI sessions

```bash
cstbench build-taxi --root /path/to/TAXI --out /path/to/cstbench-taxi
cstbench verify --out /path/to/cstbench-taxi/taxi-natural  --reference checksums/taxi-natural.json
cstbench verify --out /path/to/cstbench-taxi/taxi-mediated --reference checksums/taxi-mediated.json
```

`--root` is the directory that contains the `SESxxxx` folders. The build runs four steps that can
also be called separately:

1. `cstbench taxi-manifest`: reads the BAS Partitur files and writes the common conversation schema
   (`manifest.jsonl`, one session per line). Only usable turns are kept for rendering: the dispatcher (DP)
   and client (CL) turns that the corpus validators rated usable and that have a human translation.
   Pre-dialogue and hang-up items are dropped. Expected for TAXI 2.5: 86 sessions, 640 usable turns,
   52.6 min of speech.
2. `cstbench prepare-audio`: resamples usable turns from 8 kHz to 16 kHz (polyphase). The telephone
   band is kept; the upper band is not restored.
3. `cstbench render --config configs/taxi_natural.json`: trims the leading and trailing silence of each
   turn file, then joins the turns in dialogue order with sampled gaps, without overlap. TAXI turns
   were segmented with push-to-talk buttons, so the corpus has no inter-turn timing and no overlap.
4. `cstbench render --config configs/taxi_mediated.json`: the same with 1–3 s gaps. This models a
   listener who waits to read the translation before answering.

### Outputs

```text
<out>/manifest.jsonl                 common schema (see docs/FORMAT.md)
<out>/audio16k/SESxxxx/*.wav         resampled usable turns
<out>/<config>/SESxxxx/mix.wav       single-channel mix (main benchmark input)
<out>/<config>/SESxxxx/2ch.wav       per-speaker channels (channel 0 = DP, 1 = CL)
<out>/<config>/SESxxxx/timeline.json turn start/end (s), speaker, language, transcript, reference translation
<out>/<config>/sessions.jsonl        all timelines
<out>/<config>/build_info.json       version, config, per-session checksums
```

## Evaluate a system

A system writes one committed text piece per line (pieces are never retracted):

```jsonl
{"session": "taxi/SES0037", "lang": "English", "t": 3.42, "text": "which exit"}
```

`t` is the session time in seconds (from the start of `mix.wav`) at which the piece was emitted, `lang` the
output language, which selects the direction (English output = German->English). An optional `turn_id`
assigns the piece to a turn directly (oracle segmentation); otherwise the output is aligned to the reference
turns of its direction by minimum edit distance (as mwerSegmenter). Other fields, e.g. a computation-aware
clock, can be selected with `--time-key`.

```bash
pip install -e ".[eval]"
cstbench eval --sessions /path/to/cstbench-taxi/taxi-natural/sessions.jsonl --hyp hyp.jsonl \
    --out report.json --segments-out segments.jsonl
```

Per direction the report gives BLEU and chrF (sacrebleu, case-insensitive, punctuation removed, numbers
spelled out with num2words, matching the TAXI reference style), StreamLAAL (s), the end offset (s from
the end of a source turn to the last word of its translation: how long the listener waits after the
speaker stopped) and the number of turns that received no output. `segments.jsonl` holds the per-turn
source, hypothesis and reference (e.g. for COMET).

## Baselines

`baselines/offline_oracle.py` runs offline systems on oracle turn segments (quality upper bounds):
reference transcript -> LLM translation, Qwen3-ASR -> LLM translation, and Whisper speech translation
(into English only). It writes `hyp-<system>.jsonl` files for `cstbench eval`.

## Reproducibility

- Gaps are drawn with a per-session seed: `sha1(session_id)[:8] + config.seed`. Python's `random`
  generator is used, so the timelines are identical on every platform.
- `verify` compares the timeline digests exactly. A mismatch means different inputs or a different
  config. Audio hashes are reported separately: they can differ slightly across numpy, scipy and
  libsndfile versions. The reference hashes were produced with numpy 1.24.3, scipy 1.10.1 and
  libsndfile 1.2.2.
- Configs are versioned JSON files. Do not change a released config; add a new one with a new `name`.

## License

Code: Apache-2.0 (see `LICENSE`). Source corpora keep their own licenses. Nothing derived from a
non-redistributable corpus is included in this repository.

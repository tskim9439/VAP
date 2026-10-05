#!/usr/bin/env python3
"""Figure set of the CST-S2TT ICML paper (house style in figstyle.py). Run from any directory; writes ../<id>.pdf/.png.

    python3 make_figures.py                 # draft: planned parts dashed / tagged, '*' on unimplemented metrics
    python3 make_figures.py --final         # camera-ready look (no 'planned' marks, no '*')
    python3 make_figures.py --only fig1,fig4
    python3 make_figures.py --ql-metric COMET   # y axis of fig4 once COMET is in results_main.csv

Figure ids (= file stems):
  fig1_task, fig2_benchmark, fig3_model, fig4_quality_latency, fig5_oracle_gap,
  figA1_taxi_stats, figA2_taxi_session, figA3_gap_sweep, figA4_sim2real, figA5_turn_length,
  figA6_backbone_lookahead, figA7_delta_delay, figA8_unit_quality_latency, figA9_unit_granularity
  Sizes at full width: fig1 about 2.0 in and fig3 about 2.5 in tall (limits 2.2 and 2.8 in).

Inputs (data/):
  measured   taxi_turns.csv, taxi_gaps.csv, taxi_sessions.csv, fig_taxi_session_example.csv (aggregate TAXI statistics
             of the cstbench v0.1 build; no transcript or translation text, because TAXI may not be redistributed),
             backbone_lookahead_pilot.csv (pilot evaluation of the single-speaker backbone),
             unit_quality_latency.csv, unit_granularity.csv (text-level commit-unit pilot)
  templates  systems.csv, results_main.csv, results_conditions.csv, results_sim2real.csv, results_turnlen.csv
             created on first run if missing and never overwritten; blank cell = pending. The only filled result rows
             are the preliminary offline TAXI L0-natural runs (oracle input; to be re-run under the final protocol).
Schematics (fig1, fig2, fig3) use invented example utterances; the only numbers in them are config parameters.
Fig. 5 shows, per row, the main pipeline with the highest realistic-wrapper COMET (row_pipeline()) under both
wrappers; Fig. A3 shows every main pipeline (MAIN_PIPES). No single reference pipeline is selected.
"""
import argparse
import csv
import importlib.util
import math
from collections import Counter
from pathlib import Path

import numpy as np

import figstyle as fs
from figstyle import (CONFIG_COLOR, INK, INK2, LANG_COLOR, LANG_TEXT_ON, LANG_TINT, MUTED, OURS, PLANNED_LS, RULE,
                      WASH)

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE.parent / "data", HERE.parent
DRAFT = True
QL_METRIC = "chrF"


# ---------------------------------------------------------------------------------------------------------------
# data helpers
# ---------------------------------------------------------------------------------------------------------------
def rows(name):
    with open(DATA / name, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def ensure_csv(name, header, body):
    p = DATA / name
    if not p.exists():
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(body)
        print("created template", p)
    return rows(name)


SYSTEMS = [
    # system_id, label, short (= the name in Table 3; used in figure legends), kind, marker, directions, note
    ("gold_mt", "Gold transcript -> Qwen3.8-27B (offline, oracle turns)", "Gold->LLM", "offline", "o",
     "EN->DE;DE->EN;EN->KO;KO->EN", "oracle input; output time-stamped at the reference turn end"),
    ("asr_mt", "Qwen3-ASR-1.7B -> Qwen3.8-27B (offline, oracle turns)", "ASR->LLM", "offline", "s",
     "EN->DE;DE->EN;EN->KO;KO->EN",
     "oracle segments and source language; the consecutive cascade under the oracle wrapper"),
    ("whisper_st", "Whisper-large-v3 ST (offline, oracle turns)", "Whisper ST", "offline", "D", "DE->EN;KO->EN",
     "translates into English only"),
    ("gold_mu2", "Gold transcript -> Qwen3.8-27B SimulMT with MU2 commits", "Gold->LLM (MU2)", "bound", "h",
     "EN->DE;DE->EN", "streaming with oracle input; unit boundaries chosen with the whole turn in view (non-causal)"),
    ("consecutive", "VAD endpoint + LID -> Qwen3-ASR-1.7B -> Qwen3.8-27B", "Consecutive cascade", "consecutive", "X",
     "EN->DE;DE->EN;EN->KO;KO->EN", "realistic consecutive baseline (translate after the detected endpoint)"),
    ("seamless_streaming", "SeamlessStreaming", "SeamlessStreaming", "stream", "^", "EN->DE;DE->EN;EN->KO;KO->EN",
     "EMMA policy; one instance per target language behind the wrapper"),
    ("seamless_x2", "2 x SeamlessStreaming on the whole mix, one instance per target language", "2×SeamlessStreaming",
     "nowrap", "^", "EN->DE;DE->EN;EN->KO;KO->EN", "naive bidirectional baseline; no wrapper, no routing"),
    ("cascade_la", "Streaming ASR + Qwen3.8-27B SimulMT (LocalAgreement-2)", "Streaming cascade", "stream", "v",
     "EN->DE;DE->EN;EN->KO;KO->EN", "Nemotron 3.5 streaming RNN-T front end"),
    ("m4t_alignatt", "SeamlessM4T v2 + AlignAtt (simulstream)", "M4T v2 + AlignAtt", "stream", "P",
     "EN->DE;DE->EN;EN->KO;KO->EN", "offline ST model with attention policy; run on TAXI only"),
    ("canary_alignatt", "Canary-1B-v2 + AlignAtt (simulstream)", "Canary-1B-v2 + AlignAtt", "stream", "p",
     "EN->DE;DE->EN", "optional"),
    ("streamspeech", "StreamSpeech", "StreamSpeech", "stream", "<", "DE->EN", "single direction"),
    ("hibiki_zero", "Hibiki-Zero (text output)", "Hibiki-Zero", "stream", "*", "DE->EN",
     "optional; single direction; public weights (CC BY-NC-SA 4.0)"),
    ("infinisst", "InfiniSST", "InfiniSST", "stream", ">", "EN->DE", "only if weights are available"),
    ("ours", "Unified streaming model (ours)", "Ours", "ours", "o", "EN->DE;DE->EN;EN->KO;KO->EN",
     "planned; one curve over operating points"),
]
PRELIM = "preliminary offline run; to be re-run under the final protocol"
DERIVED = "EndOffset = 0 on the ideal clock by construction (pieces emitted at the reference turn end)"
# Under the oracle wrapper the consecutive cascade is ASR->LLM on gold turns (Sec. 6.1), so that row is its oracle side.
ORACLE_TWIN = {"consecutive": "asr_mt"}
MAIN_PIPES = ["consecutive", "seamless_streaming", "cascade_la"]   # decomposed on every set (Table 4)
STREAM_BASELINES = ("seamless_streaming", "cascade_la", "m4t_alignatt", "canary_alignatt", "streamspeech",
                    "hibiki_zero", "infinisst")
SINGLE_DIR = {"streamspeech": ["DE->EN"], "hibiki_zero": ["DE->EN"], "infinisst": ["EN->DE"]}
# offline TAXI L0-natural runs: (system, direction) -> (BLEU, chrF, StreamLAAL); EndOffset is 0 by construction
PRELIM_ROWS = {("gold_mt", "EN->DE"): ("48.31", "71.72", "4.62"), ("gold_mt", "DE->EN"): ("30.60", "58.05", "3.09"),
               ("asr_mt", "EN->DE"): ("42.66", "67.97", "4.62"), ("asr_mt", "DE->EN"): ("28.11", "56.61", "3.09"),
               ("whisper_st", "DE->EN"): ("30.69", "56.67", "3.09")}


def registry():
    reg = ensure_csv("systems.csv", ["system_id", "label", "short", "kind", "marker", "directions", "note"],
                     [list(s) for s in SYSTEMS])
    return {r["system_id"]: r for r in reg}


def results_main():
    hdr = ["system_id", "wrapper", "condition", "direction", "op", "BLEU", "chrF", "COMET", "StreamLAAL_mean_s",
           "EndOffset_p50_s", "empty_turn_pct", "wrong_dir_word_pct", "switch_latency_p50_s", "status", "source"]
    body = [[sid, "oracle", "L0-natural", d, "-", b, c, "", l, "0", "", "", "", "preliminary", PRELIM + "; " + DERIVED]
            for (sid, d), (b, c, l) in PRELIM_ROWS.items()]
    blank = [""] * 8
    for d in ("EN->DE", "DE->EN"):
        body.append(["gold_mu2", "oracle", "L0-natural", d, "tau50"] + blank +
                     ["pending", "planned: streaming run with oracle input"])
        body.append(["consecutive", "realistic", "L0-natural", d, "default"] + blank + ["pending", "planned: baseline run"])
    for sid in STREAM_BASELINES:
        for d in SINGLE_DIR.get(sid, ["EN->DE", "DE->EN"]):
            for wr in ("oracle", "realistic"):
                for op in ("low", "high"):
                    body.append([sid, wr, "L0-natural", d, op] + blank + ["pending", "planned: baseline run"])
        if sid == "seamless_streaming":
            for d in ("EN->DE", "DE->EN"):
                for op in ("low", "high"):
                    body.append(["seamless_x2", "none", "L0-natural", d, op] + blank + ["pending", "planned: baseline run"])
    for d in ("EN->DE", "DE->EN"):
        for op in ("M2-turnfinal", "M3-dt-low", "M3-dt-high"):
            body.append(["ours", "none", "L0-natural", d, op] + blank + ["pending", "planned: model not yet built"])
    return ensure_csv("results_main.csv", hdr, body)


COND_SETS = [("taxi", "L0-mediated", ""), ("taxi", "L0-natural", ""), ("taxi", "L1", ""),
             ("syn-en-de", "L0-natural", ""), ("syn-en-de", "L1", ""), ("syn-ko-en", "L0-natural", ""),
             ("syn-ko-en", "L1", "")] + [("taxi", "fixed-gap", g) for g in ("-0.6", "-0.4", "-0.2", "0.05", "0.2", "0.5",
                                                                          "1.0", "2.0")]


def results_conditions():
    """Row sets follow Table A5: offline systems, the consecutive cascade and the streaming pipelines on TAXI;
    M4T v2 + AlignAtt on TAXI only; Whisper ST (into English) on TAXI only. Fixed-gap renders: every main pipeline
    under both wrappers (ASR->LLM is the consecutive cascade's oracle side) and the model."""
    hdr = ["system_id", "wrapper", "corpus", "condition", "gap_s", "direction", "op", "COMET", "chrF",
           "StreamLAAL_mean_s", "EndOffset_p50_s", "switch_latency_p50_s", "wrong_dir_word_pct", "empty_turn_pct",
           "status"]
    runs = [("gold_mt", "oracle"), ("asr_mt", "oracle"), ("whisper_st", "oracle"), ("consecutive", "realistic"),
            ("seamless_streaming", "oracle"), ("seamless_streaming", "realistic"), ("cascade_la", "oracle"),
            ("cascade_la", "realistic"), ("m4t_alignatt", "oracle"), ("m4t_alignatt", "realistic"), ("ours", "none")]
    body = []
    for sid, wr in runs:
        for corpus, cond, gap in COND_SETS:
            if corpus != "taxi" and sid in ("whisper_st", "m4t_alignatt"):
                continue
            if cond == "fixed-gap" and sid in ("gold_mt", "whisper_st"):
                continue
            pair = ("EN->KO", "KO->EN") if corpus == "syn-ko-en" else ("EN->DE", "DE->EN")
            for d in pair:
                if sid == "whisper_st" and not d.endswith("->EN"):
                    continue
                pre = PRELIM_ROWS.get((sid, d)) if (corpus, cond) == ("taxi", "L0-natural") else None
                vals = ["", pre[1], pre[2], "0"] if pre else [""] * 4
                body.append([sid, wr, corpus, cond, gap, d, "main"] + vals + [""] * 4 +
                            ["preliminary" if pre else "pending"])
    return ensure_csv("results_conditions.csv", hdr, body)


def results_sim2real():
    hdr = ["system_id", "wrapper", "op", "direction", "chrF_taxi_L0natural", "chrF_syn_L0natural_edition1",
           "chrF_syn_L0natural_edition2", "status"]
    body = []
    for sid, wrs in (("gold_mt", ("oracle",)), ("asr_mt", ("oracle",)), ("consecutive", ("realistic",)),
                     ("seamless_streaming", ("oracle", "realistic")), ("cascade_la", ("oracle", "realistic")),
                     ("ours", ("none",))):
        for wr in wrs:
            for d in ("EN->DE", "DE->EN"):
                body.append([sid, wr, "main", d, "", "", "", "pending"])
    return ensure_csv("results_sim2real.csv", hdr, body)


def results_turnlen():
    hdr = ["system_class", "wrapper", "condition", "dur_bin", "EndOffset_p50_s", "StreamLAAL_mean_s", "status"]
    body = []
    for cls, wr in (("consecutive", "realistic"), ("best streaming pipeline", "realistic"), ("ours", "none")):
        for b in ("0-2", "2-4", "4-6", "6-10", "10+"):
            body.append([cls, wr, "L0-natural", b, "", "", "pending"])
    return ensure_csv("results_turnlen.csv", hdr, body)


def tag_planned(ax, x, y, ha="right", va="top", fsz=5.4):
    if DRAFT:
        ax.text(x, y, "planned", ha=ha, va=va, fontsize=fsz, color=MUTED, style="italic", zorder=6)


def pls():
    return PLANNED_LS if DRAFT else "-"


# ---------------------------------------------------------------------------------------------------------------
# fig1_task: the task on a single mono stream (illustrative; invented utterances)
# ---------------------------------------------------------------------------------------------------------------
def fig1_task():
    """Lanes in inch units (y) on a seconds axis (x). The speaker colours are keyed by the coloured turn bars next to
    the lane labels, so the figure needs no separate legend; the metric brackets carry side labels."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    fs.setup(7)
    W, H = fs.PAGE_W, 1.98
    fig = plt.figure(figsize=(W, H))
    lane_lines = ["Speaker A: English", "Speaker B: German", "(hidden reference)", "System input:", "one mono mix",
                  "Output 1: speaker-", "tagged transcript", "Output 2: translation", "into German (for B)",
                  "Output 3: translation", "into English (for A)"]
    LEFT = max(fs.text_width(fig, s_, 6.8) for s_ in lane_lines) + 3 / 72 + 0.01   # label column + tick pad
    BOTTOM, RIGHT, TOP = 0.16, 0.03, 0.02                          # inches around the lane axes
    aw, ah = W - LEFT - RIGHT, H - BOTTOM - TOP
    ax = fig.add_axes([LEFT / W, BOTTOM / H, aw / W, ah / H])
    x0, x1 = -0.05, 8.05
    sx = aw / (x1 - x0)                                           # inches per second
    yEN, yDE, yTB, yTA, yMix, yB, yA = 0.115, 0.400, 0.695, 0.855, 1.115, 1.380, 1.640
    lanes = [(yA, "Speaker A: English\n(hidden reference)"), (yB, "Speaker B: German\n(hidden reference)"),
             (yMix, "System input:\none mono mix"), ((yTA + yTB) / 2, "Output 1: speaker-\ntagged transcript"),
             (yDE, "Output 2: translation\ninto German (for B)"), (yEN, "Output 3: translation\ninto English (for A)")]
    spk_lang = {"A": "English", "B": "German"}
    A1, B1, A2 = (0.30, 2.90), (2.45, 3.95), (4.25, 7.10)
    BAR, TICK, TXT = 0.150, 0.110, 6.6                             # bar and tick heights (in), text size (pt)

    def turn(seg, y, spk, label=None):
        lang = spk_lang[spk]
        ax.add_patch(FancyBboxPatch((seg[0], y - BAR / 2), seg[1] - seg[0], BAR,
                                    boxstyle="round,pad=0,rounding_size=0.05", mutation_aspect=sx,
                                    fc=LANG_COLOR[lang], ec="none", zorder=2))
        if label:
            ax.text(seg[0] + 0.07, y, label, ha="left", va="center", fontsize=TXT, color=LANG_TEXT_ON[lang], zorder=3)

    def piece(t, y, text, spk):
        col = LANG_COLOR[spk_lang[spk]]
        ax.plot([t, t], [y - TICK / 2, y + TICK / 2], color=col, lw=1.1, solid_capstyle="round", zorder=3)
        ax.plot([t], [y + TICK / 2], "o", ms=2.4, color=col, mec="white", mew=0.4, zorder=4)
        ax.text(t - 0.05, y, text, ha="right", va="center", fontsize=TXT, color=INK, zorder=3)

    def bracket(xa, xb, y, text, side):
        ax.annotate("", xy=(xa, y), xytext=(xb, y),
                    arrowprops=dict(arrowstyle="|-|,widthA=0.22,widthB=0.22", lw=0.6, color=INK2, shrinkA=0, shrinkB=0),
                    zorder=4)
        if side == "right":
            ax.text(xb + 0.06, y, text, ha="left", va="center", fontsize=6.2, color=INK2, zorder=4)
        else:
            ax.text(xa - 0.06, y, text, ha="right", va="center", fontsize=6.2, color=INK2, zorder=4)

    def guide(x, y0, y1, skip=()):
        """Dashed guide from y0 up to y1 that leaves out the text bands listed in skip (centre, half-height)."""
        cuts = sorted((c - h, c + h) for c, h in skip)
        a = y0
        for lo, hi in cuts:
            if lo > a:
                ax.plot([x, x], [a, min(lo, y1)], color=RULE, lw=0.5, ls=(0, (2, 2)), zorder=1)
            a = max(a, hi)
        if a < y1:
            ax.plot([x, x], [a, y1], color=RULE, lw=0.5, ls=(0, (2, 2)), zorder=1)

    # turn-end overlap: grey band over both reference lanes, labelled at its left in the empty part of lane B
    ax.add_patch(plt.Rectangle((B1[0], yB - 0.105), A1[1] - B1[0], yA - yB + 0.21, color=WASH, lw=0, zorder=0))
    ax.text(B1[0] - 0.07, yB, "turn-end overlap (L1)", ha="right", va="center", fontsize=6.2, color=INK2, zorder=4)
    turn(A1, yA, "A", "I need a taxi to the station.")
    turn(B1, yB, "B", "Welcher Eingang?")
    turn(A2, yA, "A", "The north entrance, please.")
    segs = [(A1, 1.0), (B1, 0.9), (A2, 1.0)]

    rng = np.random.default_rng(0)
    t = np.linspace(0, 8, 24000)

    def envelope(xa, xb, amp):
        e = np.zeros_like(t)
        c = xa + rng.uniform(0.03, 0.12)
        while c < xb - 0.03:
            e += rng.uniform(0.45, 1.0) * np.exp(-0.5 * ((t - c) / rng.uniform(0.035, 0.07)) ** 2)
            c += rng.uniform(0.12, 0.30)
        e[(t < xa) | (t > xb)] = 0.0
        return amp * e

    env = sum(envelope(*seg, a) for seg, a in segs)
    sig = env * rng.standard_normal(t.size)
    ax.plot(t, yMix + 0.105 * sig / np.abs(sig).max(), color=MUTED, lw=0.25, zorder=2)

    # committed pieces at their emission time (never retracted), coloured by the source speaker
    piece(1.70, yTA, "A: I need a taxi", "A")
    piece(3.05, yTA, "to the station.", "A")
    piece(5.65, yTA, "A: The north entrance", "A")
    piece(7.25, yTA, "please.", "A")
    piece(3.30, yTB, "B: Welcher", "B")
    piece(4.10, yTB, "Eingang?", "B")
    piece(2.00, yDE, "Ich brauche ein Taxi", "A")
    piece(3.20, yDE, "zum Bahnhof.", "A")
    piece(5.95, yDE, "Der Nordeingang,", "A")
    piece(7.40, yDE, "bitte.", "A")
    piece(3.45, yEN, "Which", "B")
    piece(4.25, yEN, "entrance?", "B")

    # metrics: EndOffset (end of A's turn -> last German word), switch latency (start of B's turn -> first English word)
    yEO, ySW, band = (yTB + yDE) / 2, (yDE + yEN) / 2, 0.052
    bracket(A1[1], 3.20, yEO, "EndOffset", side="right")
    guide(A1[1], yEO, yA - BAR / 2, skip=[(yTB, band), (yTA, band)])
    bracket(B1[0], 3.45, ySW, "switch latency", side="left")
    guide(B1[0], ySW, yB - BAR / 2, skip=[(yDE, band), (yTA, band)])

    ax.set_xlim(x0, x1)
    ax.set_ylim(0, ah)
    ax.set_yticks([y for y, _ in lanes])
    ax.set_yticklabels([s for _, s in lanes], fontsize=6.8, color=INK, linespacing=1.1)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=2, labelsize=6.6, pad=1.5, color=RULE, labelcolor=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    # axis title on the tick-label line, left of the axis, so it costs no extra height
    ax.text(x0 - 0.04, -0.049, "session time (s)", ha="right", va="top", fontsize=6.6, color=INK2, clip_on=False)
    fs.save(fig, "fig1_task", OUT)


# ---------------------------------------------------------------------------------------------------------------
# fig2_benchmark: sources -> common schema -> deterministic renderer (timing conditions) -> system -> scorer
# ---------------------------------------------------------------------------------------------------------------
def fig2_benchmark():
    fs.setup(7)
    W, H = fs.PAGE_W, 2.51
    fig, ax = fs.canvas(W, H)
    star = "*" if DRAFT else ""

    def txt(x, y, s, size=6.5, color=INK, ha="left", va="top", weight="normal", style="normal", ls=1.25):
        ax.text(x, y, s, fontsize=size, color=color, ha=ha, va=va, weight=weight, style=style, linespacing=ls,
                zorder=6)

    def bar(x0, x1, y, lang, h=0.075):
        fs.rbox(ax, x0, y - h / 2, x1 - x0, h, fc=LANG_COLOR[lang], ec="none", lw=0, r=0.02, z=3)

    top, bot, hdr = 2.30, 0.30, 2.45
    cols = [(0.02, 1.50), (1.70, 1.28), (3.16, 1.76), (5.10, 1.63)]
    heads = ["1  Sources", "2  Common schema + audio", "3  Deterministic session renderer",
             "4  System under test + scorer"]
    for (x, w), h in zip(cols, heads):
        txt(x + 0.02, hdr, h, size=7.0, weight="bold")

    # 1 sources
    x, w = cols[0]
    ymid = (top + bot) / 2
    fs.rbox(ax, x, ymid + 0.05, w, top - ymid - 0.05, ec=RULE, r=0.05)
    txt(x + 0.07, top - 0.07, "BAS TAXI 2.5 (real speech)", size=6.9, weight="bold")
    txt(x + 0.07, top - 0.24, "German dispatcher ↔ English client\n86 sessions, 640 usable turns\n"
                              "human translations, 8 kHz phone\npush-to-talk: no timing, no overlap", color=INK2)
    fs.rbox(ax, x, bot, w, ymid - bot - 0.05, ec=MUTED if DRAFT else RULE, ls=pls(), r=0.05)
    txt(x + 0.07, ymid - 0.12, "CST-Bench-Syn (evaluation only)", size=6.9, weight="bold")
    txt(x + 0.07, ymid - 0.29, "XDailyDialog, BConTrasT dialogues\nEn↔De: human or post-edited MT\n"
                               "Ko↔En: Korean side machine-\ntranslated, consensus-checked\nTTS speech", color=INK2)
    tag_planned(ax, x + w - 0.05, bot + 0.04, va="bottom")

    # 2 schema + audio
    x2, w2 = cols[1]
    sy0, sy1 = 0.82, 1.80
    fs.rbox(ax, x2, sy0, w2, sy1 - sy0, ec=RULE, r=0.05)
    txt(x2 + 0.07, sy1 - 0.07, "manifest.jsonl", size=6.9, weight="bold")
    txt(x2 + 0.07, sy1 - 0.24, "per turn: speaker, language,\ntranscript, reference translation\n8 → 16 kHz (TAXI)\n"
                               "energy-based silence trim,\n0.1 s margin kept", color=INK2)
    fs.arrow(ax, x + w, (top + ymid) / 2, x2, sy1 - 0.25)
    fs.arrow(ax, x + w, (bot + ymid) / 2, x2, sy0 + 0.25, color=MUTED)

    # 3 renderer with timing strips (TAXI: the German dispatcher speaks first)
    x3, w3 = cols[2]
    fs.rbox(ax, x3, bot, w3, top - bot, ec=RULE, r=0.05)
    txt(x3 + 0.07, top - 0.07, "turns joined in real dialogue order;", color=INK2)
    txt(x3 + 0.07, top - 0.20, "seed = SHA-1(session id) + config seed", color=INK2)
    fs.arrow(ax, x2 + w2, (sy0 + sy1) / 2, x3, (sy0 + sy1) / 2)
    lab_x, bx0, scale = x3 + 0.07, x3 + 0.66, 0.30
    strips = [("L0 natural", [("German", 0.0, 1.6), ("English", 1.75, 3.3)],
               "gap ~ N(0.2, 0.25²) s, clipped to [0.05, 1.0]", False),
              ("L0 mediated", [("German", 0.0, 1.6), ("English", 2.6, 3.6)], "gap ~ U[1, 3] s", False),
              ("L1 (planned)" if DRAFT else "L1", [("German", 0.0, 1.6), ("English", 1.15, 2.7)],
               "next turn starts 0.2–0.8 s before the end", True)]
    y = 1.84
    for lab, bars, note, planned in strips:
        txt(lab_x, y + 0.035, lab, color=MUTED if (planned and DRAFT) else INK, va="center",
            style="italic" if (planned and DRAFT) else "normal")
        for k, (lang, a, b) in enumerate(bars):
            bar(bx0 + a * scale, bx0 + b * scale, y + (0.07 if k == 0 else -0.01), lang)
        if bars[1][1] < bars[0][2]:      # overlap region (L1)
            import matplotlib.pyplot as plt
            ax.add_patch(plt.Rectangle((bx0 + bars[1][1] * scale, y - 0.06), (bars[0][2] - bars[1][1]) * scale, 0.17,
                                       fc=CONFIG_COLOR["L1"], alpha=0.18, lw=0, zorder=2))
        txt(lab_x, y - 0.10, note, size=6.1, color=MUTED, va="center")
        y -= 0.37
    oy = 0.68
    txt(x3 + 0.07, oy + 0.17, "outputs per session", color=INK2)
    for i, (lang, ab_) in enumerate((("German", "DE"), ("English", "EN"))):
        kx = x3 + 1.10 + i * 0.32
        fs.rbox(ax, kx, oy + 0.135, 0.11, 0.065, fc=LANG_COLOR[lang], ec="none", lw=0, r=0.015)
        txt(kx + 0.14, oy + 0.17, ab_, size=6.2, color=INK2, va="center")
    for k, (name, desc) in enumerate((("mix.wav", "single-channel input"), ("2ch.wav", "per speaker (analysis)"),
                                      ("timeline.json", "turn times, references"))):
        txt(x3 + 0.07, oy - 0.02 - k * 0.13, name, weight="bold")
        txt(x3 + 0.80, oy - 0.02 - k * 0.13, desc, size=6.4, color=INK2)

    # 4 system + scorer
    x4, w4 = cols[3]
    py0, py1 = 1.92, top
    fs.rbox(ax, x4, py0, w4, py1 - py0, fc=WASH, ec=WASH, r=0.05)
    txt(x4 + 0.07, py1 - 0.06, "system hears mix.wav only and emits")
    txt(x4 + 0.07, py1 - 0.20, "committed pieces (time, language, text)")
    fs.arrow(ax, x3 + w3, 2.11, x4, 2.11)
    fs.rbox(ax, x4, bot, w4, 1.47, ec=RULE, r=0.05)
    fs.arrow(ax, x4 + w4 / 2, py0, x4 + w4 / 2, bot + 1.47)
    txt(x4 + 0.07, bot + 1.40, "cstbench eval", size=6.9, weight="bold")
    txt(x4 + 0.07, bot + 1.24, "per direction: min-edit-distance\nresegmentation onto reference turns", size=6.4,
        color=INK2)
    mrows = [("quality", f"BLEU, chrF, COMET{star}"), ("lag", f"StreamLAAL, EndOffset,\nswitch latency{star}"),
             ("errors", f"empty turns,\nwrong direction{star}"), ("aux.", f"WER{star}, cpWER{star}, DER{star}")]
    yy = bot + 0.96
    for a, b in mrows:
        txt(x4 + 0.07, yy, a, color=MUTED)
        txt(x4 + 0.47, yy, b)
        yy -= 0.14 + 0.10 * b.count("\n")
    if DRAFT:
        txt(x4 + 0.07, bot + 0.05, "* not yet in cstbench v0.1", size=6.0, color=MUTED, va="bottom")

    # No footer: the release statement is in the caption and Sec. 4.1, the rebuild check in Sec. 4.2. Crop the strip
    # that the footer used (the canvas axes counts in full towards the tight bounding box).
    crop = 0.24
    fig.set_size_inches(W, H - crop)
    ax.set_ylim(crop, H)
    fs.save(fig, "fig2_benchmark", OUT)


# ---------------------------------------------------------------------------------------------------------------
# fig3_model: conversational delayed streams (backbone built; speaker / translation outputs planned)
# ---------------------------------------------------------------------------------------------------------------
CHUNK = 0.08
A_WORDS = [("I", 0.22, 0.36), ("need", 0.40, 0.66), ("a", 0.68, 0.74), ("taxi", 0.76, 1.22),
           ("to", 1.36, 1.48), ("the", 1.50, 1.62), ("station", 1.66, 2.30)]           # speaker A, English
B_WORDS = [("Welcher", 2.02, 2.50), ("Eingang?", 2.58, 3.30)]                        # speaker B, German
A_TURN, B_TURN = (0.18, 2.40), (1.98, 3.42)                                          # 0.42 s turn-end overlap (L1)
A_MU = [(0, 3, "Ich brauche ein Taxi"), (4, 6, "zum Bahnhof.")]                      # >= 3 words and >= 1.0 s each
B_MU = [(0, 1, "Which entrance?")]                                                   # whole turn (turn end forces commit)
DS, DT = 2, 3                                                                         # delta_s, delta_t in chunks
SEQ_CHUNKS = [30, 31, 32, 33, None, 44]                                               # chunks shown in (c); None = gap


def emit(t_end, d):
    k = int(math.floor(t_end / CHUNK + 1e-9)) + d
    return k, (k + 1) * CHUNK


def fig3_model():
    """Three panels on one inch grid: (a) architecture, (b) targets on the 80 ms clock, (c) one row of the
    serialized token sequence (computed from the same timings as (b)) with a one-line key. Commit units are labelled
    'unit 1', 'unit 2' (not 'MU 1/2', which would collide with the policy name MU2)."""
    import matplotlib.pyplot as plt
    fs.setup(7)
    W, H = fs.PAGE_W, 2.42
    fig = plt.figure(figsize=(W, H))

    def axes_in(l, b, w, h):
        a = fig.add_axes([l / W, b / H, w / W, h / H])
        a.set_xlim(0, w)
        a.set_ylim(0, h)
        a.axis("off")
        return a

    def panel_label(x, y, s):
        fig.text(x / W, y / H, s, ha="left", va="top", fontsize=7.5, color=INK, weight="bold")

    # ---------------- (a) architecture (inch units) ----------------
    YA0, HA = 0.65, H - 0.65                       # (a) occupies x 0..1.74, y YA0..H
    aa = axes_in(0.0, YA0, 1.74, HA)
    panel_label(0.0, H, "(a)")
    cx = 0.71                                      # centre of the stack
    rng = np.random.default_rng(3)
    tt = np.linspace(0.10, 1.32, 420)
    env = 0.9 * (tt < 0.84) + 0.8 * (tt > 0.68)
    sig = env * rng.standard_normal(tt.size) * np.abs(np.sin(tt * 23)) ** 0.5
    aa.plot(tt, 0.155 + 0.05 * sig / np.abs(sig).max(), color=MUTED, lw=0.3)
    aa.text(cx, 0.0, "one mono mix of both speakers", ha="center", va="bottom", fontsize=5.8, color=INK2)
    blocks = [(0.27, 0.335, "Causal FastConformer encoder",
               "Nemotron 3.5, 0.6B, cache-aware\nmeasured look-ahead ≤ 80 ms"),
              (0.68, 0.245, "MLP adapter", "one embedding per 80 ms chunk"),
              (1.000, 0.335, "Qwen3-ASR thinker LM (1.7B)", "decoder-only, KV cache\nover the whole session")]
    for y, h, title, sub in blocks:
        fs.rbox(aa, 0.06, y, 1.30, h, ec=INK2, r=0.04)
        aa.text(cx, y + h - 0.035, title, ha="center", va="top", fontsize=6.2, color=INK, weight="bold")
        aa.text(cx, y + h - 0.135, sub, ha="center", va="top", fontsize=5.7, color=INK2, linespacing=1.1)
    for y0, y1 in ((0.20, 0.27), (0.605, 0.68), (0.925, 1.000)):
        fs.arrow(aa, cx, y0, cx, y1)
    fs.arrow(aa, cx, 1.335, cx, 1.42)
    aa.text(cx, 1.435, "one token stream, see (c)", ha="center", va="bottom", fontsize=5.9, color=INK)
    fs.rbox(aa, 1.42, 1.000, 0.31, 0.335, ec=INK2, ls=pls(), r=0.03)
    aa.text(1.575, 1.1675, "spk.\nactivity\nhead", ha="center", va="center", fontsize=5.2, color=INK2, linespacing=1.0)
    fs.arrow(aa, 1.36, 1.1675, 1.42, 1.1675)

    # ---------------- (b) delayed streams on the 80 ms clock (x: seconds, y: inches) ----------------
    BX, BY, BW = 2.72, 0.78, 4.0
    BH = H - BY - 0.02
    ab = fig.add_axes([BX / W, BY / H, BW / W, BH / H])
    xmin, xmax = -0.05, 3.84
    sx = BW / (xmax - xmin)                        # inches per second
    yEN, yDE, yTR, yIN, yB, yA = 0.085, 0.335, 0.625, 0.905, 1.155, 1.425
    BAR, TK = 0.13, 0.115                          # speaker-bar and tick heights (in)
    ylab = [(yA, "speaker A, English\n(hidden reference)"), (yB, "speaker B, German\n(hidden reference)"),
            (yIN, "input: one audio\nembedding / 80 ms"), (yTR, "transcript stream\n(word end + $\\delta_s$)"),
            (yDE, "translation into German\n(unit end + $\\delta_t$)"),
            (yEN, "translation into English\n(unit end + $\\delta_t$)")]
    panel_label(1.84, H, "(b)")
    ab.add_patch(plt.Rectangle((B_TURN[0], yB - 0.095), A_TURN[1] - B_TURN[0], yA - yB + 0.19, color=WASH, lw=0,
                               zorder=0))
    ab.text(A_TURN[1] + 0.05, yA, "turn-end overlap", ha="left", va="center", fontsize=5.6, color=INK2, zorder=3)

    def turn(words, span, y, lang):
        fs.rbox(ab, span[0], y - BAR / 2, span[1] - span[0], BAR, fc=LANG_COLOR[lang], ec="none", lw=0, r=0.05,
                aspect=sx)
        for w_, s, e in words:
            ab.text((s + e) / 2, y, w_, ha="center", va="center", fontsize=5.8, color=LANG_TEXT_ON[lang], zorder=3)
        for w_, s, e in words[:-1]:
            ab.plot([e + 0.006] * 2, [y - BAR / 2, y - BAR / 2 + 0.03], color="white", lw=0.6, zorder=3)

    def unit_bracket(words, i0, i1, y, lang, label):
        x0, x1 = words[i0][1], words[i1][2]
        y0, y1 = y + BAR / 2 + 0.012, y + BAR / 2 + 0.032
        ab.plot([x0, x0, x1, x1], [y0, y1, y1, y0], color=LANG_COLOR[lang], lw=0.6)
        ab.text((x0 + x1) / 2, y1 + 0.006, label, ha="center", va="bottom", fontsize=5.2, color=INK2)

    turn(A_WORDS, A_TURN, yA, "English")
    turn(B_WORDS, B_TURN, yB, "German")
    for j, (i0, i1, _) in enumerate(A_MU):
        unit_bracket(A_WORDS, i0, i1, yA, "English", f"unit {j + 1}")
    for j, (i0, i1, _) in enumerate(B_MU):
        unit_bracket(B_WORDS, i0, i1, yB, "German", "unit 1 (= turn)")
    for k in range(int(round(xmax / CHUNK))):
        ab.add_patch(plt.Rectangle((k * CHUNK + 0.006, yIN - 0.055), CHUNK - 0.012, 0.11, color="#e6e6e6", lw=0,
                                   zorder=1))

    def tick(tx, y, lang):
        ab.plot([tx, tx], [y - TK / 2, y + TK / 2], color=LANG_COLOR[lang], lw=0.9, solid_capstyle="round", zorder=3)
        ab.plot([tx], [y + TK / 2], "o", ms=2.2, color=LANG_COLOR[lang], mec="white", mew=0.4, zorder=4)

    for words, lang in ((A_WORDS, "English"), (B_WORDS, "German")):
        for _, _, e in words:
            tick(emit(e, DS)[1], yTR, lang)
    ab.text(0.47, yTR - TK / 2 - 0.025, "A: i need a taxi to the station", ha="left", va="top", fontsize=5.8, color=INK)
    ab.text(2.66, yTR - TK / 2 - 0.025, "B: welcher eingang", ha="left", va="top", fontsize=5.8, color=INK)
    e_taxi = A_WORDS[3][2]
    t_taxi = emit(e_taxi, DS)[1]
    ybr = yTR + TK / 2 + 0.05
    ab.plot([e_taxi, e_taxi], [ybr, yA - BAR / 2], color=RULE, lw=0.5, ls=(0, (2, 2)), zorder=0.5)
    ab.annotate("", xy=(e_taxi, ybr), xytext=(t_taxi, ybr),
                arrowprops=dict(arrowstyle="|-|,widthA=0.18,widthB=0.18", lw=0.55, color=INK2, shrinkA=0, shrinkB=0))
    ab.text((e_taxi + t_taxi) / 2, ybr + 0.012, "+$\\delta_s$", ha="center", va="bottom", fontsize=5.6, color=INK2)

    def unit_emit(words, mus, y, lang, bracket_for=None):
        for j, (i0, i1, text) in enumerate(mus):
            t_mu = words[i1][2]
            te = emit(t_mu, DT)[1]
            tick(te, y, lang)
            ab.text(te - 0.035, y, text, ha="right", va="center", fontsize=6.0, color=INK, zorder=3)
            if j == bracket_for:
                yb_ = y + TK / 2 + 0.05
                ab.annotate("", xy=(t_mu, yb_), xytext=(te, yb_),
                            arrowprops=dict(arrowstyle="|-|,widthA=0.18,widthB=0.18", lw=0.55, color=INK2, shrinkA=0,
                                            shrinkB=0))
                ab.text((t_mu + te) / 2, yb_ + 0.012, "+$\\delta_t$", ha="center", va="bottom", fontsize=5.6,
                        color=INK2)
                yy = yA if lang == "English" else yB
                ab.plot([t_mu, t_mu], [yb_, yy - BAR / 2], color=RULE, lw=0.5, ls=(0, (2, 2)), zorder=0.5)
        return emit(words[mus[-1][1]][2], DT)[1]

    t_flush = unit_emit(A_WORDS, A_MU, yDE, "English", bracket_for=1)
    unit_emit(B_WORDS, B_MU, yEN, "German")
    ab.text(t_flush + 0.06, yDE, "turn end: flush", ha="left", va="center", fontsize=5.4, color=MUTED, style="italic")
    ab.set_xlim(xmin, xmax)
    ab.set_ylim(0, BH)
    ab.set_yticks([y for y, _ in ylab])
    ab.set_yticklabels([s for _, s in ylab], fontsize=5.9, color=INK, linespacing=1.05)
    ab.tick_params(axis="y", length=0, pad=3)
    ab.tick_params(axis="x", length=2, width=0.6, color=RULE, labelsize=5.9, labelcolor=INK2, pad=1.5)
    for s in ("top", "right", "left"):
        ab.spines[s].set_visible(False)
    ab.spines["bottom"].set_color(RULE)
    ab.set_xticks([0, 0.8, 1.6, 2.4, 3.2])
    ab.text(xmin - 0.03, -0.049, "session time (s)", ha="right", va="top", fontsize=5.9, color=INK2, clip_on=False)

    # ---------------- (c) the serialized token sequence (computed from the same timings) ----------------
    HC = 0.60
    ac = axes_in(0.0, 0.0, W, HC)
    panel_label(0.0, HC + 0.005, "(c)")
    events = {}
    for words, lang, spk in ((A_WORDS, "English", "A"), (B_WORDS, "German", "B")):
        for w_, _, e in words:
            events.setdefault(emit(e, DS)[0], []).append(("src", w_.lower().strip("?.,!"), lang, spk))
    for words, mus, lang, tgt in ((A_WORDS, A_MU, "English", "DE"), (B_WORDS, B_MU, "German", "EN")):
        for i0, i1, text in mus:
            events.setdefault(emit(words[i1][2], DT)[0], []).append(("tgt", text, lang, tgt))

    def chunk_tokens(k):
        toks = [("audio", f"$a_{{{k}}}$", None)]
        ev = events.get(k, [])
        cur = None
        for _, w_, lang, spk in [e for e in ev if e[0] == "src"]:
            if spk != cur:
                toks.append(("tag", f"⟨{spk}⟩", lang))
                cur = spk
            toks.append(("src", w_, lang))
        for _, text, lang, t in [e for e in ev if e[0] == "tgt"]:
            toks.append(("tag", f"⟨→{t.lower()}⟩", lang))
            toks.append(("tgt", text, lang))
        toks.append(("next", "⟨next⟩", None))
        return toks

    TSZ, ph, padx, gapx = 5.9, 0.135, 0.032, 0.035

    def pill(x, y, kind, text, lang, size=TSZ, h=ph, underline=True):
        w = fs.text_width(fig, text, size, style="italic" if kind == "tgt" else "normal") + 2 * padx
        planned = kind == "tag" or kind == "tgt" or (kind == "src" and lang != "English")
        if kind == "audio":
            fc, ec, ls = WASH, RULE, "-"
        elif kind == "next":
            fc, ec, ls = "#e3e3e3", "none", "-"
        elif kind == "src":
            fc, ec, ls = LANG_TINT[lang], (MUTED if planned and DRAFT else "none"), (pls() if planned else "-")
        elif kind == "tag":
            fc, ec, ls = "white", INK2, pls()
        else:
            fc, ec, ls = LANG_TINT[lang], LANG_COLOR[lang], pls()
        fs.rbox(ac, x, y - h / 2, w, h, fc=fc, ec=ec, lw=0.6, ls=ls, r=0.025)
        ac.text(x + w / 2, y - 0.003, text, ha="center", va="center", fontsize=size, color=INK, zorder=4,
                style="italic" if kind == "tgt" else "normal", weight="bold" if kind == "tag" else "normal")
        if underline and kind != "audio":
            ac.plot([x + 0.015, x + w - 0.015], [y - h / 2 - 0.025] * 2, color=INK2, lw=0.7, solid_capstyle="butt")
        return w

    ac.text(0.30, HC - 0.07, "prefix (input only):  … language pair English–German  ⟨DELAY_2⟩  ⟨TDELAY_3⟩;   "
            "then, per 80 ms chunk k:  audio embedding, speaker-tagged transcript, target-tagged translation, ⟨next⟩",
            ha="left", va="center", fontsize=5.8, color=INK2)
    y, x = 0.335, 0.0
    ac.text(x, y, "…", ha="left", va="center", fontsize=6, color=MUTED)
    x += 0.11
    for k in SEQ_CHUNKS:
        if k is None:
            ac.text(x - 0.01, y, "…", ha="left", va="center", fontsize=6, color=MUTED)
            x += 0.12
            continue
        for kind, text, lang in chunk_tokens(k):
            x += pill(x, y, kind, text, lang) + gapx
        x += 0.035
    ac.text(x - 0.01, y, "…", ha="left", va="center", fontsize=6, color=MUTED)
    seq_right = x + 0.1
    # one-line key; the caption names every token type as well
    key = [("audio", "$a_k$", None, "audio embedding (input)"), ("next", "⟨next⟩", None, "end of chunk"),
           ("tag", "⟨A⟩", "English", "speaker tag"), ("src", "word", "English", "transcript"),
           ("tag", "⟨→de⟩", "English", "target tag"), ("tgt", "Taxi", "English", "translation")]
    KSZ, ly, x = 5.5, 0.10, 0.0
    for kind, text, lang, desc in key:
        x += pill(x, ly, kind, text, lang, size=KSZ, h=0.12, underline=False) + 0.04
        ac.text(x, ly, desc, ha="left", va="center", fontsize=KSZ, color=INK2)
        x += fs.text_width(fig, desc, KSZ) + 0.11
    ac.text(x, ly, "tint: speaker's language", ha="left", va="center", fontsize=KSZ, color=INK2)
    x += fs.text_width(fig, "tint: speaker's language", KSZ) + 0.11
    ac.plot([x, x + 0.14], [ly - 0.035] * 2, color=INK2, lw=0.7)
    ac.text(x + 0.17, ly, "training target", ha="left", va="center", fontsize=KSZ, color=INK2)
    x += 0.17 + fs.text_width(fig, "training target", KSZ) + 0.11
    if DRAFT:
        ac.text(x, ly, "dashed: planned", ha="left", va="center", fontsize=KSZ, color=MUTED, style="italic")
        x += fs.text_width(fig, "dashed: planned", KSZ, style="italic")
    fig.savefig(OUT / "fig3_model.pdf")
    fig.savefig(OUT / "fig3_model.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / "fig3_model.pdf", f"| (c) row {seq_right:.2f} in, key {x:.2f} in of {W:.2f} |",
          "events", {k: [e[1] for e in v] for k, v in sorted(events.items())})


# ---------------------------------------------------------------------------------------------------------------
# fig4_quality_latency: TAXI L0-natural, quality vs StreamLAAL (top) and vs EndOffset (bottom), per direction
# ---------------------------------------------------------------------------------------------------------------
def sys_style(reg, sid):
    """Marker style per system. Oracle input (offline systems, Gold->LLM (MU2)): filled black with a white edge, so
    coincident anchors stay distinguishable. Baselines: filled grey under the oracle wrapper (callers draw the
    realistic wrapper hollow). A baseline without wrapper (2xSeamlessStreaming): half-filled. Ours: vermillion."""
    r = reg[sid]
    if r["kind"] == "ours":
        return dict(marker="o", color=OURS, ms=3.8, mfc=OURS, mec=OURS, fill="full")
    if r["kind"] in ("offline", "bound"):
        return dict(marker=r["marker"], color=INK, ms=4.2, mfc=INK, mec="white", fill="full")
    if r["kind"] == "nowrap":
        return dict(marker=r["marker"], color=INK2, ms=3.8, mfc=INK2, mec=INK2, fill="left")
    return dict(marker=r["marker"], color=INK2, ms=3.8, mfc=INK2, mec=INK2, fill="full")


def legend_label(reg, sid):
    r = reg[sid]
    return r["short"].replace("->", "→") + (" (optional)" if r["note"].startswith("optional") else "")


def fig4_quality_latency():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fs.setup(7)
    reg = registry()
    res = [r for r in results_main() if r["condition"] == "L0-natural"]
    metric = QL_METRIC
    fig, axes = plt.subplots(2, 2, figsize=(fs.COL_W, 2.3),
                             gridspec_kw=dict(wspace=0.36, hspace=0.6, top=0.95, bottom=0.13))
    xs_def = [("StreamLAAL_mean_s", "StreamLAAL (s)", 6.6), ("EndOffset_p50_s", "EndOffset, median (s)", 4.0)]
    for c, (d, dname) in enumerate((("EN->DE", "En→De"), ("DE->EN", "De→En"))):
        ys_all = [num(r[metric]) for r in res if r["direction"] == d and num(r[metric]) is not None]
        lo = (min(ys_all) - 22) if ys_all else 30
        hi = (max(ys_all) + 6) if ys_all else 80
        for rr, (xkey, xlab, xmax) in enumerate(xs_def):
            ax = axes[rr][c]
            fs.style_axes(ax)
            pts = [r for r in res if r["direction"] == d and num(r[metric]) is not None and num(r[xkey]) is not None]
            off = [r for r in pts if reg[r["system_id"]]["kind"] == "offline"]
            if off and xkey == "StreamLAAL_mean_s":
                xo = num(off[0][xkey])
                ax.axvline(xo, color=RULE, lw=0.6, zorder=0)
                ax.text(xo - 0.12, 0.985, "turn length", transform=ax.get_xaxis_transform(), ha="right", va="top",
                        fontsize=5.2, color=MUTED)
                # latency budgets: simultaneous regime rho <= 0.5 (main operating point), consecutive rho <= 1
                ax.axvline(xo / 2, color=RULE, lw=0.6, ls=(0, (2, 2)), zorder=0)
                ax.text(xo / 2 + 0.1, 0.03, "half turn", transform=ax.get_xaxis_transform(), ha="left",
                        va="bottom", fontsize=5.2, color=MUTED)
            if off and xkey == "EndOffset_p50_s":
                ax.text(0.05, 0.985, "oracle endpoint", transform=ax.get_xaxis_transform(), ha="left", va="top",
                        fontsize=5.2, color=MUTED)
            for sid in sorted({r["system_id"] for r in pts if reg[r["system_id"]]["kind"] in ("stream", "consecutive")}):
                for op in sorted({r["op"] for r in pts if r["system_id"] == sid}):
                    o = [r for r in pts if r["system_id"] == sid and r["op"] == op and r["wrapper"] == "oracle"]
                    q = [r for r in pts if r["system_id"] == sid and r["op"] == op and r["wrapper"] == "realistic"]
                    if o and q:
                        ax.annotate("", xy=(num(q[0][xkey]), num(q[0][metric])),
                                    xytext=(num(o[0][xkey]), num(o[0][metric])),
                                    arrowprops=dict(arrowstyle="-|>", lw=0.5, color=MUTED, mutation_scale=5), zorder=1)
            ours = sorted([r for r in pts if r["system_id"] == "ours"], key=lambda r: num(r[xkey]))
            if ours:
                ax.plot([num(r[xkey]) for r in ours], [num(r[metric]) for r in ours], "-o", color=OURS, ms=3.2,
                        lw=1.1, mec="white", mew=0.4, zorder=4)
            for r in pts:
                if r["system_id"] == "ours":
                    continue
                st = sys_style(reg, r["system_id"])
                hollow = r["wrapper"] == "realistic"
                ax.plot(num(r[xkey]), num(r[metric]), ls="none", marker=st["marker"], ms=st["ms"], color=st["color"],
                        mfc="white" if hollow else st["mfc"], mec=st["color"] if hollow else st["mec"],
                        fillstyle=st["fill"], markerfacecoloralt="white", mew=0.8 if hollow else 0.45, zorder=3)
            # direct labels for the offline anchors, staggered (leader line when moved)
            sep = 0.085 * (hi - lo)
            last = None
            for r in sorted(off, key=lambda r: -num(r[metric])):
                y = num(r[metric])
                yl = y if last is None or last - y >= sep else last - sep
                lab = reg[r["system_id"]]["short"].replace("->", "→") + ("†" if r["status"] == "preliminary" else "")
                x = num(r[xkey])
                dx = 0.30 if xkey == "StreamLAAL_mean_s" else 0.22
                if abs(yl - y) > 1e-9:
                    ax.plot([x + 0.12, x + dx - 0.04], [y, yl], color=RULE, lw=0.5)
                ax.text(x + dx, yl, lab, ha="left", va="center", fontsize=5.3, color=INK2)
                last = yl
            if not any(reg[r["system_id"]]["kind"] not in ("offline",) for r in pts):
                ax.text(0.04 if xkey == "StreamLAAL_mean_s" else 0.96, 0.30,
                        "streaming, consecutive\nand ours: pending", transform=ax.transAxes,
                        ha="left" if xkey == "StreamLAAL_mean_s" else "right", va="center", fontsize=5.4,
                        color=MUTED, style="italic")
            ax.set_xlim(-0.1 if xkey == "EndOffset_p50_s" else 0, xmax)
            ax.set_ylim(lo, hi)
            ax.set_xlabel(xlab, fontsize=6.3, labelpad=1)
            ax.set_title(f"({'abcd'[rr * 2 + c]}) {dname}", loc="left", fontsize=6.8, pad=3)
            ax.tick_params(labelsize=5.9, pad=1.5)
        axes[0][0].set_ylabel(metric, fontsize=6.3, labelpad=2)
        axes[1][0].set_ylabel(metric, fontsize=6.3, labelpad=2)
    order = ["consecutive", "seamless_streaming", "cascade_la", "m4t_alignatt", "streamspeech", "infinisst",
             "hibiki_zero", "gold_mu2", "ours"]
    # a system outside the fixed list (e.g. 2xSeamlessStreaming) enters the legend once it has a plotted point
    plotted = {r["system_id"] for r in res if num(r[metric]) is not None}
    order += [sid for sid in reg if sid in plotted and sid not in order and reg[sid]["kind"] not in ("offline",)]
    hs = []
    for sid in order:
        st = sys_style(reg, sid)
        hs.append(Line2D([], [], ls="-" if sid == "ours" else "none", lw=1.1, marker=st["marker"], ms=st["ms"],
                         color=st["color"], mfc=st["mfc"], mec=st["mec"], fillstyle=st["fill"],
                         markerfacecoloralt="white", mew=0.45, label=legend_label(reg, sid)))
    hs.append(Line2D([], [], ls="none", marker="o", ms=3.8, color=INK2, mfc="white", mew=0.8,
                     label="hollow = realistic wrapper\n(VAD + language ID)"))
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, -0.005), ncol=3, fontsize=5.3,
               handlelength=1.0, handletextpad=0.4, columnspacing=0.9, labelspacing=0.3, borderaxespad=0.0)
    fs.save(fig, "fig4_quality_latency", OUT)


# ---------------------------------------------------------------------------------------------------------------
# fig5_oracle_gap: one reference pipeline under the oracle and the realistic wrapper vs ours, per condition and pair
# ---------------------------------------------------------------------------------------------------------------
GAP_SETS = [("TAXI (real speech)", "taxi", ["L0-mediated", "L0-natural", "L1"]),
            ("Syn En↔De (TTS)", "syn-en-de", ["L0-natural", "L1"]),
            ("Syn Ko↔En (TTS)", "syn-ko-en", ["L0-natural", "L1"])]
GAP_METRICS = [("COMET", "COMET (mean of the two directions)"), ("EndOffset_p50_s", "EndOffset, median (s)"),
               ("wrong_dir_word_pct", "wrong-direction words (%)")]
COND_LABEL = {"L0-mediated": "L0 mediated", "L0-natural": "L0 natural", "L1": "L1 turn-end overlap"}


def _cond_means(res, corpus, cond, gap=None):
    """Mean over the two directions for every (system, wrapper) with both directions; op == main."""
    out = {}
    sel = [r for r in res if r["corpus"] == corpus and r["condition"] == cond and r["op"] == "main"
           and (gap is None or r["gap_s"] == gap)]
    for sid, wr in {(r["system_id"], r["wrapper"]) for r in sel}:
        rr = [r for r in sel if r["system_id"] == sid and r["wrapper"] == wr]
        vals = {}
        for m, _ in GAP_METRICS:
            v = [num(r[m]) for r in rr]
            vals[m] = float(np.mean(v)) if len(v) == 2 and all(x is not None for x in v) else None
        out[(sid, wr)] = vals
    return out


def row_pipeline(means):
    """The main pipeline with the highest realistic-wrapper COMET (mean of the two directions) in one row of Fig. 5;
    None while pending. Returns (oracle key, realistic key, display name)."""
    cands = [(sid, v["COMET"]) for (sid, wr), v in means.items()
             if wr == "realistic" and sid in MAIN_PIPES and v["COMET"] is not None]
    if not cands:
        return None
    sid = max(cands, key=lambda kv: kv[1])[0]
    return (ORACLE_TWIN.get(sid, sid), "oracle"), (sid, "realistic"), registry()[sid]["short"]


def fig5_oracle_gap():
    import matplotlib.pyplot as plt
    fs.setup(7)
    res = results_conditions()
    order, ypos, y = [], {}, 0.0
    for name, corpus, conds in GAP_SETS:
        for cond in conds:
            ypos[(corpus, cond)] = y
            order.append((name, corpus, cond))
            y -= 1.0
        y -= 0.8
    fig, axes = plt.subplots(1, 3, figsize=(fs.PAGE_W, 2.35), sharey=True)
    any_pending = False
    for ax, (m, mlab) in zip(axes, GAP_METRICS):
        fs.style_axes(ax, grid_axis="x")
        vals = []
        for name, corpus, cond in order:
            means = _cond_means(res, corpus, cond)
            ref = row_pipeline(means)
            yy = ypos[(corpus, cond)]
            o = means.get(ref[0], {}).get(m) if ref else None
            rl = means.get(ref[1], {}).get(m) if ref else None
            ou = means.get(("ours", "none"), {}).get(m)
            got = [v for v in (o, rl, ou) if v is not None]
            vals += got
            if o is not None and rl is not None:
                ax.plot([o, rl], [yy, yy], color=RULE, lw=1.6, solid_capstyle="round", zorder=2)
            if o is not None:
                ax.plot([o], [yy], "o", ms=4.6, color=INK2, mec="white", mew=0.9, zorder=3)
            if rl is not None:
                ax.plot([rl], [yy], "o", ms=4.6, mfc="white", mec=MUTED, mew=1.1, zorder=3)
            if ou is not None:
                ax.plot([ou], [yy], "D", ms=4.2, color=OURS, mec="white", mew=0.8, zorder=4)
            if not got:
                any_pending = True
                ax.text(0.5, yy, "pending", transform=ax.get_yaxis_transform(),
                        fontsize=6.0, color=MUTED, style="italic", ha="center", va="center")
            elif ref and m == GAP_METRICS[0][0]:
                ax.text(0.02, yy, ref[2].replace("->", "→"), transform=ax.get_yaxis_transform(), fontsize=5.6,
                        color=MUTED, ha="left", va="center")
        if vals:
            lo, hi = min(vals), max(vals)
            pad = 0.08 * (hi - lo or 1.0)
            ax.set_xlim(lo - pad, hi + pad)
        else:
            ax.set_xticks([])
        ax.set_xlabel(mlab, fontsize=6.6, color=INK2, labelpad=2)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([ypos[(c, k)] for _, c, k in order])
    axes[0].set_yticklabels([COND_LABEL[k] for _, _, k in order], fontsize=6.4, color=INK)
    for name, corpus, conds in GAP_SETS:
        axes[0].text(-0.62, ypos[(corpus, conds[0])] + 0.72, name, transform=axes[0].get_yaxis_transform(),
                     fontsize=6.6, color=INK, weight="bold", ha="left", va="center")
    axes[0].set_ylim(min(ypos.values()) - 0.6, 1.1)
    pipe = "best realistic pipeline of the row"
    h = [plt.Line2D([], [], marker="o", ls="", ms=4.6, color=INK2, mec="white", label=f"{pipe}, oracle segmentation"),
         plt.Line2D([], [], marker="o", ls="", ms=4.6, mfc="white", mec=MUTED, mew=1.1,
                    label=f"{pipe}, VAD + language-ID routing"),
         plt.Line2D([], [], marker="D", ls="", ms=4.2, color=OURS, mec="white", label="Ours (no oracle input)")]
    fig.legend(handles=h, loc="upper center", ncol=3, fontsize=6.4, bbox_to_anchor=(0.58, 1.02), handletextpad=0.3,
               columnspacing=1.2)
    if any_pending and DRAFT:
        fig.text(0.005, 0.995, "placeholder: values pending", fontsize=5.8, color=MUTED, ha="left", va="top")
    fig.tight_layout(pad=0.4, w_pad=0.8, rect=(0.11, 0, 1, 0.9))
    fs.save(fig, "fig5_oracle_gap", OUT)


# ---------------------------------------------------------------------------------------------------------------
# appendix figures
# ---------------------------------------------------------------------------------------------------------------
def figA1_taxi_stats():
    import matplotlib.pyplot as plt
    fs.setup(7)
    turns, gaps = rows("taxi_turns.csv"), rows("taxi_gaps.csv")
    dur = {l: [float(r["dur_trimmed_s"]) for r in turns if r["lang"] == l] for l in ("German", "English")}
    words = {l: [int(r["src_words"]) for r in turns if r["lang"] == l] for l in ("German", "English")}
    g = {c: [float(r["gap_s"]) for r in gaps if r["config"] == c and r["type"] == "speaker_change"]
         for c in ("natural", "mediated")}
    sw = Counter(r["session"] for r in gaps if r["config"] == "natural" and r["type"] == "speaker_change")
    switches = [sw.get(r["session"], 0) for r in rows("taxi_sessions.csv")]

    def key(a, x, y, color, text, dx=0.07):
        a.plot([x, x + dx], [y, y], transform=a.transAxes, color=color, lw=1.3, clip_on=False, solid_capstyle="butt")
        a.text(x + dx + 0.03, y, text, transform=a.transAxes, ha="left", va="center", fontsize=5.8, color=INK2)

    fig, ax = plt.subplots(1, 4, figsize=(fs.PAGE_W, 1.62), gridspec_kw=dict(wspace=0.55))
    bins = np.arange(0, 18.5, 0.5)
    for lang in ("German", "English"):
        ax[0].hist(dur[lang], bins=bins, histtype="step", lw=1.0, color=LANG_COLOR[lang])
    key(ax[0], 0.40, 0.90, LANG_COLOR["German"], f"German, n = {len(dur['German'])}\nmedian {np.median(dur['German']):.2f} s")
    key(ax[0], 0.40, 0.62, LANG_COLOR["English"],
        f"English, n = {len(dur['English'])}\nmedian {np.median(dur['English']):.2f} s")
    ax[0].set_xlabel("turn duration after trimming (s)")
    ax[0].set_ylabel("turns")
    ax[0].set_title("(a) turn length", loc="left", fontsize=7)
    wb = np.arange(0, 40, 2)
    for lang in ("German", "English"):
        ax[1].hist(words[lang], bins=wb, histtype="step", lw=1.0, color=LANG_COLOR[lang])
    key(ax[1], 0.30, 0.90, LANG_COLOR["German"], f"German: median {np.median(words['German']):.0f}")
    key(ax[1], 0.30, 0.76, LANG_COLOR["English"], f"English: median {np.median(words['English']):.0f}")
    ax[1].set_xlabel("source words per turn")
    ax[1].set_title("(b) words", loc="left", fontsize=7)
    gb = np.arange(-1.0, 3.1, 0.1)
    ax[2].axvspan(-0.8, -0.2, color=CONFIG_COLOR["L1"], alpha=0.16, lw=0)
    ax[2].text(-0.5, 0.97, "L1", transform=ax[2].get_xaxis_transform(), ha="center", va="top", fontsize=6, color=INK2)
    for c in ("natural", "mediated"):
        ax[2].hist(g[c], bins=gb, histtype="step", lw=1.0, color=CONFIG_COLOR[c])
    floor = sum(1 for x in g["natural"] if abs(x - 0.05) < 1e-9)
    ax[2].text(0.16, 0.98, f"L0 natural: median {np.median(g['natural']):.3f} s;\n{floor}/{len(g['natural'])} at the "
               "0.05 s floor", transform=ax[2].get_xaxis_transform(), ha="left", va="top", fontsize=5.6, color=INK2)
    ax[2].text(2.0, 0.27, f"L0 mediated:\nmedian {np.median(g['mediated']):.3f} s", transform=ax[2].get_xaxis_transform(),
               ha="center", va="bottom", fontsize=5.6, color=INK2)
    ax[2].set_xlim(-1.0, 3.1)
    ax[2].set_xlabel("gap at a speaker change (s)")
    ax[2].set_ylabel("speaker changes")
    ax[2].set_title("(c) inter-turn gap", loc="left", fontsize=7)
    cnt = Counter(switches)
    med = int(np.median(switches))
    xs = np.arange(0, max(switches) + 1)
    ax[3].bar(xs, [cnt.get(int(x), 0) for x in xs], width=0.72, lw=0, color=[INK if int(x) == med else MUTED for x in xs])
    ax[3].text(med + 0.8, 0.97, f"median {med}\n{sum(switches)} switches\nin {len(switches)} sessions",
               transform=ax[3].get_xaxis_transform(), ha="left", va="top", fontsize=5.8, color=INK2)
    ax[3].set_xlabel("switches per session")
    ax[3].set_ylabel("sessions")
    ax[3].set_title("(d) direction switches", loc="left", fontsize=7)
    for a in ax:
        a.tick_params(length=2)
    fs.save(fig, "figA1_taxi_stats", OUT)
    print("  check: turns", {k: len(v) for k, v in dur.items()}, "gaps", {k: len(v) for k, v in g.items()},
          "switches", sum(switches), "sessions", len(switches), "median", med, "zero-switch", cnt.get(0, 0))


def figA2_taxi_session():
    import matplotlib.pyplot as plt
    fs.setup(7)
    rs = rows("fig_taxi_session_example.csv")
    fig, axes = plt.subplots(2, 1, figsize=(fs.PAGE_W, 1.75), sharex=True, gridspec_kw=dict(hspace=0.55))
    for ax, cfg in zip(axes, ("natural", "mediated")):
        sub = [r for r in rs if r["config"] == cfg]
        lanes = {"German": 2, "English": 1}
        for r in sub:
            s0, s1 = float(r["start_s"]), float(r["end_s"])
            ax.add_patch(plt.Rectangle((s0, lanes[r["lang"]] - 0.3), s1 - s0, 0.6, color=LANG_COLOR[r["lang"]], lw=0))
            ax.add_patch(plt.Rectangle((s0, -0.3), s1 - s0, 0.6, color=INK2, lw=0))
        end = max(float(r["end_s"]) for r in sub) + 0.5
        ax.plot([0, end], [0, 0], color=RULE, lw=0.6, zorder=0)
        gp = [float(r["gap_before_s"]) for r in sub[1:]]
        ax.set_title(f"({'ab'[cfg == 'mediated']}) L0 {cfg}: speaker-change gaps {min(gp):.2f}–{max(gp):.2f} s, "
                     f"session {end:.1f} s", loc="left", fontsize=7)
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels(["dispatcher (German)", "client (English)", "mono mix (input)"], fontsize=6.2)
        ax.set_ylim(-0.6, 2.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[1].set_xlabel("session time (s)")
    fs.save(fig, "figA2_taxi_session", OUT)


def figA3_gap_sweep():
    import matplotlib.pyplot as plt
    fs.setup(7)
    reg = registry()
    gaps = rows("taxi_gaps.csv")
    nat = np.array([float(r["gap_s"]) for r in gaps if r["config"] == "natural" and r["type"] == "speaker_change"])
    med = np.array([float(r["gap_s"]) for r in gaps if r["config"] == "mediated" and r["type"] == "speaker_change"])
    allres = results_conditions()
    res = [r for r in allres if r["corpus"] == "taxi" and r["condition"] == "fixed-gap"]
    curves = [((ORACLE_TWIN.get(p, p), "oracle"), (0, (3, 2))) for p in MAIN_PIPES] + \
             [((p, "realistic"), "-") for p in MAIN_PIPES] + [(("ours", "none"), "-")]   # every main pipeline
    gaps_s = sorted({r["gap_s"] for r in res}, key=float)
    metrics = [("COMET", "COMET (mean of directions)"), ("wrong_dir_word_pct", "wrong-direction words (%)"),
               ("EndOffset_p50_s", "EndOffset, median (s)")]
    fig, axes = plt.subplots(1, 3, figsize=(fs.PAGE_W, 1.75), gridspec_kw=dict(wspace=0.38))
    bins = np.arange(-0.9, 3.05, 0.05)
    for i, (ax, (m, ylab)) in enumerate(zip(axes, metrics)):
        tr = ax.get_xaxis_transform()
        ax.axvspan(-0.8, -0.2, color=CONFIG_COLOR["L1"], alpha=0.12, lw=0)
        h_nat, _ = np.histogram(nat, bins=bins)
        h_med, _ = np.histogram(med, bins=bins)
        sc = 0.22 / max(h_nat.max(), 1)
        ax.fill_between(bins[:-1], 0, h_nat * sc, step="post", color=CONFIG_COLOR["natural"], alpha=0.35, lw=0, transform=tr)
        ax.fill_between(bins[:-1], 0, h_med * sc, step="post", color=CONFIG_COLOR["mediated"], alpha=0.35, lw=0, transform=tr)
        if i == 0:
            ax.text(-0.5, 0.97, "L1", transform=tr, ha="center", va="top", fontsize=5.8, color=INK2)
            ax.text(0.33, 0.25, "L0 natural", transform=tr, ha="left", va="bottom", fontsize=5.4, color=INK2)
            ax.text(2.0, 0.08, "L0 mediated", transform=tr, ha="center", va="bottom", fontsize=5.4, color=INK2)
        ys = []
        for (sid, wr), ls in curves:
            pts_ = [(float(g), _cond_means(res, "taxi", "fixed-gap", gap=g).get((sid, wr), {}).get(m)) for g in gaps_s]
            pts_ = [(g, v) for g, v in pts_ if v is not None]
            if not pts_:
                continue
            ys += [v for _, v in pts_]
            st = sys_style(reg, sid)
            ax.plot([g for g, _ in pts_], [v for _, v in pts_], ls=ls, lw=1.0, color=st["color"], marker=st["marker"],
                    ms=3.0, mfc="white" if wr == "realistic" else st["mfc"], mec=st["color"], mew=0.7)
        if ys:      # leave the lower quarter to the gap histograms (drawn in axes coordinates)
            span = (max(ys) - min(ys)) or 1.0
            ax.set_ylim(min(ys) - 0.45 * span, max(ys) + 0.12 * span)
        else:
            fs.pending_note(ax, "pending: TAXI re-rendered with\nfixed gaps from −0.6 to 2 s", y=0.62)
            ax.set_yticks([])
        ax.set_xlim(-0.9, 2.3)
        ax.set_xlabel("inter-turn gap (s); < 0 = overlap")
        ax.set_ylabel(ylab)
        ax.set_title(f"({'abc'[i]})", loc="left", fontsize=7)
        ax.tick_params(length=2)
    fs.save(fig, "figA3_gap_sweep", OUT)


def figA4_sim2real():
    import matplotlib.pyplot as plt
    fs.setup(7)
    reg = registry()
    res = results_sim2real()
    pts = [r for r in res if num(r["chrF_taxi_L0natural"]) is not None and num(r["chrF_syn_L0natural_edition1"]) is not None]
    fig, ax = plt.subplots(figsize=(fs.COL_W * 0.62, 1.85))
    fs.style_axes(ax)
    if pts:
        from scipy.stats import kendalltau
        xs = [num(r["chrF_taxi_L0natural"]) for r in pts]
        ys = [num(r["chrF_syn_L0natural_edition1"]) for r in pts]
        lo, hi = min(xs + ys) - 3, max(xs + ys) + 3
        ax.plot([lo, hi], [lo, hi], color=RULE, lw=0.6, zorder=0)
        for r, x, y in zip(pts, xs, ys):
            st = sys_style(reg, r["system_id"])
            ax.plot(x, y, ls="none", marker=st["marker"], ms=st["ms"], color=st["color"],
                    mfc="white" if r["wrapper"] == "realistic" else st["mfc"], mew=0.8)
        tau = kendalltau(xs, ys).statistic
        ax.text(0.04, 0.96, f"Kendall $\\tau_b$ = {tau:.2f} (n = {len(pts)})", transform=ax.transAxes, ha="left", va="top",
                fontsize=6, color=INK2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    else:
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, color=RULE, lw=0.6)
        fs.pending_note(ax, "pending: one point per\nsystem × wrapper × direction\nKendall $\\tau_b$ = [TBD]", y=0.3)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("chrF on TAXI (real speech)")
    ax.set_ylabel("chrF on Syn En↔De (TTS)")
    fs.save(fig, "figA4_sim2real", OUT)


def figA5_turn_length():
    import matplotlib.pyplot as plt
    fs.setup(7)
    turns = rows("taxi_turns.csv")
    res = results_turnlen()
    edges = [0, 2, 4, 6, 10, 100]
    labels = ["0–2", "2–4", "4–6", "6–10", ">10"]
    keys = ["0-2", "2-4", "4-6", "6-10", "10+"]
    d = [float(r["dur_trimmed_s"]) for r in turns]
    counts = [sum(1 for x in d if edges[i] <= x < edges[i + 1]) for i in range(5)]
    fig, axes = plt.subplots(1, 2, figsize=(fs.COL_W, 1.7), gridspec_kw=dict(wspace=0.45))
    sty = {"consecutive": dict(color=INK2, marker="X", ls="-"),
           "best streaming pipeline": dict(color=INK2, marker="^", ls=(0, (3, 2))),
           "ours": dict(color=OURS, marker="o", ls="-")}
    for ax, (m, ylab) in zip(axes, (("EndOffset_p50_s", "EndOffset, median (s)"), ("StreamLAAL_mean_s", "StreamLAAL (s)"))):
        fs.style_axes(ax)
        tr = ax.get_xaxis_transform()
        sc = 0.28 / max(counts)
        ax.bar(range(5), [c * sc for c in counts], width=0.7, color=WASH, ec=RULE, lw=0.4, transform=tr, zorder=0)
        for i, c in enumerate(counts):
            ax.text(i, c * sc + 0.01, str(c), transform=tr, ha="center", va="bottom", fontsize=5.0, color=MUTED)
        got = False
        for cls, st in sty.items():
            rr = {r["dur_bin"]: num(r[m]) for r in res if r["system_class"] == cls}
            xs = [i for i, k in enumerate(keys) if rr.get(k) is not None]
            if xs:
                got = True
                ax.plot(xs, [rr[keys[i]] for i in xs], color=st["color"], marker=st["marker"], ls=st["ls"], ms=3.0,
                        lw=1.0, mfc="white" if cls != "ours" else st["color"], mew=0.7, label=cls)
        if not got:
            fs.pending_note(ax, "pending", y=0.62)
            ax.set_yticks([])
        ax.set_xticks(range(5))
        ax.set_xlim(-0.6, 4.6)
        ax.set_xticklabels(labels, fontsize=5.8)
        ax.set_xlabel("source turn duration (s)", fontsize=6.3)
        ax.set_ylabel(ylab, fontsize=6.3)
    axes[0].text(0.0, 1.04, "bars: TAXI turns per bin (n = 640)", transform=axes[0].transAxes, fontsize=5.4,
                 color=MUTED, ha="left", va="bottom")
    fs.save(fig, "figA5_turn_length", OUT)


def figA6_backbone_lookahead():
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fs.setup(7)
    rs = rows("backbone_lookahead_pilot.csv")
    sets = [("voxpopuli-aa-test WER", "(a) VoxPopuli-AA (En), WER %"), ("kspon-eval_clean CER", "(b) Kspon clean (Ko), CER %"),
            ("kspon-eval_other CER", "(c) Kspon other (Ko), CER %")]
    sty = {"[56,0]": dict(color=INK, marker="o", label="encoder [56,0] (≤ 80 ms look-ahead)"),
           "[56,3]": dict(color=OURS, marker="s", label="encoder [56,3] (+240 ms right context)")}
    fig, axes = plt.subplots(1, 3, figsize=(fs.PAGE_W, 1.75), gridspec_kw=dict(wspace=0.32))
    xfinal = 720
    for ax, (key, title) in zip(axes, sets):
        sub = [r for r in rs if r["set"] == key]
        for ctx, st in sty.items():
            pts = sorted([(float(r["nominal_delay_ms"]), float(r["error_rate_pct"])) for r in sub
                          if r["encoder_context"] == ctx and r["delta"] not in ("final", "-")])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=st["color"], marker=st["marker"], ms=3.2, lw=1.0,
                    mec="white", mew=0.4)
            fin = [float(r["error_rate_pct"]) for r in sub if r["encoder_context"] == ctx and r["delta"] == "final"]
            if fin:
                ax.plot(xfinal + (0 if ctx == "[56,0]" else 30), fin[0], ls="none", color=st["color"], marker=st["marker"],
                        ms=3.2, mec="white", mew=0.4)
        off = [float(r["error_rate_pct"]) for r in sub if r["encoder_context"].startswith("offline")]
        if off:
            ax.axhline(off[0], color=MUTED, lw=0.6, ls=(0, (3, 2)))
            ymin = min(float(r["error_rate_pct"]) for r in sub)
            below = off[0] - ymin > 0.5        # room under the line (left part of the panel is empty there)
            ax.text(135, off[0] + (-0.12 if below else 0.12), "offline Qwen3-ASR-1.7B", ha="left",
                    va="top" if below else "bottom", fontsize=5.2, color=MUTED)
        ax.set_xticks([160, 320, 480, 640, xfinal + 15])
        ax.set_xticklabels(["160", "320", "480", "640", "final"])
        ax.set_xlim(120, xfinal + 70)
        ax.set_title(title, loc="left", fontsize=7)
        ax.set_xlabel("mean nominal delay (ms)")
        ax.tick_params(length=2)
    hs = [Line2D([], [], color=st["color"], marker=st["marker"], ms=3.2, lw=1.0, label=st["label"]) for st in sty.values()]
    fig.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=5.8)
    fs.save(fig, "figA6_backbone_lookahead", OUT)


def figA7_delta_delay():
    """Re-styles the recorded worked example (fig_delta_delay_interleaving.py in this folder) in the house style."""
    import matplotlib.pyplot as plt
    src = HERE / "fig_delta_delay_interleaving.py"
    spec = importlib.util.spec_from_file_location("delta_delay_src", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fs.setup(7)                                      # serif + serif Hangul fallback, TrueType
    mod.INK, mod.INK2, mod.MUTED, mod.AXIS = INK, INK2, MUTED, RULE
    mod.BLUE_T, mod.BLUE_T2 = "#e6e6e6", "#a6a6a6"   # audio chunk / chunk containing a word end (neutral)
    mod.ORANGE_T = LANG_TINT["Korean"]                 # chunk that emits Korean text tokens
    mod.NEUTRAL = WASH
    fig = plt.figure(figsize=(fs.PAGE_W, 3.45))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 0.95], hspace=0.30, left=0.118, right=0.99, top=0.985, bottom=0.03)
    ax = fig.add_subplot(gs[0])
    mod.timeline(ax)
    bx = fig.add_subplot(gs[1])
    mod.sequence(bx)
    fig.text(0.005, 0.985, "(a)", ha="left", va="top", fontsize=7.5, color=INK, weight="bold")
    fig.text(0.005, 0.30, "(b)", ha="left", va="top", fontsize=7.5, color=INK, weight="bold")
    fs.save(fig, "figA7_delta_delay", OUT)


def figA8_unit_quality_latency():
    import matplotlib.pyplot as plt
    fs.setup(7)
    rs = rows("unit_quality_latency.csv")
    style = {"utt_end": dict(marker="D", color=MUTED, ms=3.6), "SEM_END2": dict(marker="s", color=INK2, ms=3.6),
             "MU2": dict(marker="o", color=OURS, ms=4.2)}
    short = {"utt_end": "utterance end", "SEM_END2": "SEM_END", "MU2": "MU2"}
    fig, axes = plt.subplots(1, 2, figsize=(fs.COL_W, 1.6), gridspec_kw=dict(wspace=0.4))
    for ax, d, title in ((axes[0], "KO→EN", "(a) Ko→En"), (axes[1], "EN→KO", "(b) En→Ko")):
        fs.style_axes(ax)
        sub = [r for r in rs if r["direction"] == d]
        for r in sub:
            st = style[r["policy"]]
            x, y = float(r["LAAL_s"]), float(r["chrF"])
            ax.plot(x, y, ls="none", marker=st["marker"], ms=st["ms"], color=st["color"], mec="white", mew=0.5)
            ax.text(x, y + 0.45, short[r["policy"]], ha="center", va="bottom", fontsize=5.4, color=INK2)
        ys = [float(r["chrF"]) for r in sub]
        ax.set_ylim(min(ys) - 2.5, max(ys) + 2.0)
        ax.set_xlim(0, 16.5)
        ax.set_xlabel("LAAL (s, gold timestamps)")
        ax.set_title(title, loc="left", fontsize=7)
    axes[0].set_ylabel("chrF (silver references)")
    fs.save(fig, "figA8_unit_quality_latency", OUT)


def figA9_unit_granularity():
    import matplotlib.pyplot as plt
    fs.setup(7)
    rs = rows("unit_granularity.csv")
    pol_style = {"MU": dict(marker="o", color=OURS, label="meaning unit (MU)"),
                 "SEM_END": dict(marker="s", color=INK2, label="SEM_END"),
                 "utt_end": dict(marker="D", color=MUTED, label="utterance end")}
    metrics = [("words_per_unit", "source words per unit", ["MU", "SEM_END"], 30),
               ("mean_wait_s", "mean wait per source word (s)", ["MU", "SEM_END", "utt_end"], 9)]
    fig, axes = plt.subplots(2, 2, figsize=(fs.PAGE_W * 0.72, 2.3), gridspec_kw=dict(wspace=0.12, hspace=0.75))
    for r_i, (m, xlab, pols, xmax) in enumerate(metrics):
        for c_i, d in enumerate(("KO→EN", "EN→KO")):
            ax = axes[r_i][c_i]
            fs.style_axes(ax, grid_axis="x")
            for j, p in enumerate(pols):
                v = [float(r["value"]) for r in rs if r["direction"] == d and r["metric"] == m and r["policy"] == p]
                y = len(pols) - 1 - j
                ax.plot([min(v), max(v)], [y, y], color=RULE, lw=2.2, solid_capstyle="round", zorder=1)
                st = pol_style[p]
                ax.plot(v, [y] * len(v), ls="none", marker=st["marker"], ms=3.4, color=st["color"], mec="white",
                        mew=0.5, zorder=2)
                ax.text(max(v) + 0.025 * xmax, y, f"{min(v):.2f}–{max(v):.2f}", ha="left", va="center", fontsize=5.4,
                        color=INK2)
            ax.set_yticks(range(len(pols)))
            ax.set_yticklabels([pol_style[p]["label"] for p in reversed(pols)] if c_i == 0 else [], fontsize=5.9)
            ax.tick_params(axis="y", length=0)
            ax.set_ylim(-0.6, len(pols) - 0.4)
            ax.set_xlim(0, xmax)
            ax.set_xlabel(xlab, fontsize=6.2)
            if r_i == 0:
                ax.set_title(d.replace("KO", "Ko").replace("EN", "En"), loc="left", fontsize=7)
    fs.save(fig, "figA9_unit_granularity", OUT)


FIGS = {"fig1": fig1_task, "fig2": fig2_benchmark, "fig3": fig3_model,
        "fig4": fig4_quality_latency, "fig5": fig5_oracle_gap, "figA1": figA1_taxi_stats, "figA2": figA2_taxi_session,
        "figA3": figA3_gap_sweep, "figA4": figA4_sim2real, "figA5": figA5_turn_length, "figA6": figA6_backbone_lookahead,
        "figA7": figA7_delta_delay, "figA8": figA8_unit_quality_latency, "figA9": figA9_unit_granularity}


def main():
    global DRAFT, QL_METRIC, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--only", default="")
    ap.add_argument("--ql-metric", default="chrF", choices=["chrF", "COMET", "BLEU"])
    ap.add_argument("--out", default=str(OUT), help="output folder (default: the figures/ directory)")
    a = ap.parse_args()
    DRAFT, QL_METRIC, OUT = not a.final, a.ql_metric, Path(a.out)
    names = [n for n in a.only.split(",") if n] or list(FIGS)
    for n in names:
        FIGS[n]()


if __name__ == "__main__":
    main()

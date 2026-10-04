"""Schematic of the delta-delay audio/text interleaving used by the streaming ASR backbone.

All timing values come from the worked example in wiki/outputs/output-vapasr-model-and-sequence.md section 4.4
(kspon-dev utterance ks-dev-utt-KsponSpeech_620663, 2.6 s, K = 32 chunks of 80 ms, delta = 2):
word-final token end times t_end (Qwen3-ForcedAligner, 80 ms resolution) and the chunk k = floor(t_end/80 ms) + delta
in which each word's tokens are emitted. Emission delay = (k+1)*80 ms - t_end = 233 ms for every word here.
Token counts per word (3/4/2/1/1/2) are from the same table. Romanisation (Revised Romanisation) is added for
non-Korean readers; it is not part of the data. Panel (b) shows the input/target alignment of the sequence
(labels only on text tokens and <NEXT_AUDIO>; prefix, [AUDIO_k], <EMPTY_AUDIO> positions carry no loss).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = os.path.dirname(os.path.abspath(__file__))

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
AXIS = "#c3c2b7"
BLUE_T, BLUE_T2 = "#cde2fb", "#86b6ef"   # audio chunk / chunk containing a word end (blue ramp 100 / 250)
ORANGE_T = "#f9d2c2"                      # text-token output (slot-2 orange, 30 % tint)
NEUTRAL = "#f0efec"                       # <NEXT_AUDIO> only

plt.rcParams.update({
    "font.family": ["Arial", "AppleGothic"],
    "font.size": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})

CHUNK_MS, K, DELTA = 80, 32, 2
# (hangul, romanisation, t_end ms, number of BPE tokens) - wiki section 4.4
WORDS = [
    ("그러고", "geureogo", 807, 3),
    ("오퍼가", "opeoga", 1127, 4),
    ("나만", "naman", 1447, 2),
    ("있는", "inneun", 1607, 1),
    ("게", "ge", 1687, 1),
    ("아니야", "aniya", 1927, 2),
]


def emit_chunk(t_end: int) -> int:
    return t_end // CHUNK_MS + DELTA


def timeline(ax):
    ax.set_xlim(-10, K * CHUNK_MS + 10)
    ax.set_ylim(0.35, 4.75)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_yticks([])
    ax.set_xticks(range(0, 2561, 320))
    ax.tick_params(axis="x", colors=AXIS, labelcolor=INK2, length=2.5, labelsize=6.5)
    ax.set_xlabel("time (ms)", color=INK2, fontsize=6.5, labelpad=1)

    y_aud, y_out, h = 2.05, 0.95, 0.42
    gap = 6  # ms of surface gap between cells
    end_chunks = {t // CHUNK_MS for _, _, t, _ in WORDS}
    out_chunks = {emit_chunk(t) for _, _, t, _ in WORDS}
    for k in range(K):
        x = k * CHUNK_MS + gap / 2
        ax.add_patch(Rectangle((x, y_aud), CHUNK_MS - gap, h, facecolor=BLUE_T2 if k in end_chunks else BLUE_T,
                               edgecolor="none"))
        ax.add_patch(Rectangle((x, y_out), CHUNK_MS - gap, h, facecolor=ORANGE_T if k in out_chunks else NEUTRAL,
                               edgecolor="none"))

    lab = dict(ha="right", va="center", fontsize=6.5, color=INK2, clip_on=False)
    ax.text(-25, y_aud + h / 2, "input: one audio\nembedding per\n80 ms chunk", **lab)
    ax.text(-25, y_out + h / 2, "model output\nafter chunk k", **lab)
    ax.text(-25, 3.12, "word end time\n(forced aligner)", **lab)

    # word-end markers + labels
    # every label ends at its word-end tick ("the word ends here")
    label_dx = {w[0]: -7 for w in WORDS}
    label_ha = {w[0]: "right" for w in WORDS}
    for hangul, roman, t_end, ntok in WORDS:
        ax.plot([t_end, t_end], [y_aud + h + 0.02, y_aud + h + 0.30], color=INK, linewidth=0.8, solid_capstyle="butt")
        dx = label_dx.get(hangul, 0)
        ha = label_ha.get(hangul, "center")
        ax.text(t_end + dx, y_aud + h + 0.64, hangul, ha=ha, va="bottom", fontsize=7, color=INK)
        ax.text(t_end + dx, y_aud + h + 0.37, roman, ha=ha, va="bottom", fontsize=5.8, color=INK2, style="italic")
        k = emit_chunk(t_end)
        t_emit = (k + 1) * CHUNK_MS
        ax.add_patch(FancyArrowPatch((t_end, y_aud - 0.03), (t_emit - 3, y_out + h + 0.03), arrowstyle="-|>",
                                     mutation_scale=6, linewidth=0.7, color=INK2, shrinkA=0, shrinkB=0))
        ax.text(k * CHUNK_MS + CHUNK_MS / 2, y_out - 0.08, f"{ntok}", ha="center", va="top", fontsize=5.8, color=INK2)
    ax.text(K * CHUNK_MS, y_out - 0.08, "BPE tokens emitted", ha="right", va="top", fontsize=5.8, color=MUTED)

    w0 = WORDS[0][2]
    k0 = emit_chunk(w0)
    ax.text((w0 + (k0 + 1) * CHUNK_MS) / 2 + 14, (y_aud + y_out + h) / 2, f"+{(k0 + 1) * CHUNK_MS - w0} ms",
            fontsize=6.3, color=INK, ha="left", va="center")

    # rule (top-left) and colour key (top-right)
    ax.text(0, 4.70,
            r"word-final token emitted in chunk  $k=\lfloor t_{end}/80\,\mathrm{ms}\rfloor+\delta$  ($\delta=2$ here)"
            "\n" r"emission delay $=(k+1)\cdot 80\,\mathrm{ms}-t_{end}$:  160-240 ms for $\delta=2$; 233 ms for each word below",
            ha="left", va="top", fontsize=6.3, color=INK, linespacing=1.55)
    keys = [(BLUE_T, "audio chunk"), (BLUE_T2, "chunk containing a word end"),
            (ORANGE_T, "chunk ends with text tokens + <NEXT_AUDIO>"), (NEUTRAL, "chunk ends with <NEXT_AUDIO> only")]
    kx, ky = 1715, 4.62
    for i, (c, t) in enumerate(keys):
        yy = ky - i * 0.27
        ax.add_patch(Rectangle((kx, yy - 0.09), 34, 0.18, facecolor=c, edgecolor="none"))
        ax.text(kx + 50, yy, t, ha="left", va="center", fontsize=5.8, color=INK2)


def sequence(bx):
    """Serialized LM sequence: one row for chunks 11-13 of the example, one row for the end of the stream.
    Tokens that are next-token-prediction targets (text, <NEXT_AUDIO>) get an underline; prefix, [AUDIO_k] and
    <EMPTY_AUDIO> positions carry no loss (label -100)."""
    bx.set_xlim(-1, 101)
    bx.set_ylim(-0.2, 3.3)
    bx.axis("off")

    width = {"audio": 9.0, "next": 11.0, "empty": 12.8, "text": 3.4, "dots": 2.2}
    fill = {"audio": BLUE_T, "next": NEUTRAL, "empty": NEUTRAL, "text": ORANGE_T}
    target = {"audio": False, "next": True, "empty": False, "text": True}

    def kind(t):
        if t == "…":
            return "dots"
        if t.startswith("<NEXT"):
            return "next"
        if t.startswith("<EMPTY"):
            return "empty"
        if t.startswith("[AUDIO"):
            return "audio"
        return "text"

    def row(tokens, y, x0, label):
        hh = 0.62
        bx.text(x0 - 0.8, y + hh / 2, label, ha="right", va="center", fontsize=6.3, color=INK2)
        x = x0
        for t in tokens:
            k = kind(t)
            w = width[k]
            if k == "dots":
                bx.text(x + w / 2, y + hh / 2, t, ha="center", va="center", fontsize=7, color=MUTED)
            else:
                bx.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0,rounding_size=0.3",
                                            facecolor=fill[k], edgecolor="none"))
                bx.text(x + w / 2, y + hh / 2, t, ha="center", va="center", fontsize=5.9, color=INK)
                if target[k]:
                    bx.plot([x + 0.4, x + w - 0.4], [y - 0.13, y - 0.13], color=INK2, linewidth=1.1,
                            solid_capstyle="round")
            x += w + 0.7
        return x

    row(["…", "[AUDIO_11]", "<NEXT_AUDIO>", "[AUDIO_12]", "그", "러", "고", "<NEXT_AUDIO>", "[AUDIO_13]",
         "<NEXT_AUDIO>", "…"], 2.05, 13.0, "chunks 11-13")
    xe = row(["…", "[AUDIO_31]", "<NEXT_AUDIO>", "<EMPTY_AUDIO>", "<NEXT_AUDIO>"], 0.95, 13.0, "stream end")
    bx.text(xe + 0.5, 0.95 + 0.31, "flush round: nothing left to emit", ha="left", va="center", fontsize=5.8,
            color=MUTED)

    bx.text(13.0, 3.15, "prefix (input only):  …assistant\\nlanguage Korean<asr_text><DELAY_2>", ha="left",
            va="center", fontsize=6.0, color=INK2)
    bx.plot([13.0, 15.6], [0.30, 0.30], color=INK2, linewidth=1.1, solid_capstyle="round")
    bx.text(16.3, 0.30, "= next-token CE target: text tokens (weight 1) and <NEXT_AUDIO> (weight 0.3 EN / 0.15 KO)",
            ha="left", va="center", fontsize=5.8, color=INK2)
    bx.text(16.3, 0.0, "no loss on prefix, [AUDIO_k] (adapter output replaces the audio placeholder embedding) "
                       "and <EMPTY_AUDIO> positions", ha="left", va="center", fontsize=5.8, color=INK2)


if __name__ == "__main__":
    fig = plt.figure(figsize=(6.75, 3.45))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 0.95], hspace=0.30, left=0.118, right=0.99, top=0.985, bottom=0.03)
    ax = fig.add_subplot(gs[0])
    timeline(ax)
    bx = fig.add_subplot(gs[1])
    sequence(bx)
    fig.text(0.005, 0.985, "(a)", ha="left", va="top", fontsize=7.5, color=INK)
    fig.text(0.005, 0.30, "(b)", ha="left", va="top", fontsize=7.5, color=INK)
    stem = os.path.join(OUT, "fig_delta_delay_interleaving")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=220)
    print("ok")

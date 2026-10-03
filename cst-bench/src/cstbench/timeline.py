"""Turn list -> continuous two-speaker session (timeline), deterministic given a config and a seed.

For corpora without inter-turn timing (e.g. TAXI) the gaps between turns are drawn from a distribution
named in the render config:
  normal   {"type": "normal", "mean": 0.2, "std": 0.25, "min": 0.05, "max": 1.0}
           turn-taking gaps of everyday conversation (approximating the near-zero gap distribution of
           Heldner & Edlund 2010 without producing overlap)
  uniform  {"type": "uniform", "min": 1.0, "max": 3.0}
           "mediated" conversation: the listener waits to read the translation before answering
Consecutive turns of the same speaker (a skipped invalid turn in between) get a within-speaker pause.
Outputs: single-channel mix, per-speaker channels (speaker order = channel order), turn start/end times.
Overlap conditions are produced on top of this timeline by event generators (separate module).
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import hashlib
import random

import numpy as np


@dataclass
class Placed:
    turn_id: str
    speaker: str
    start_s: float
    end_s: float
    gap_before_s: float


def session_seed(session_id: str, seed: int) -> int:
    """Platform-independent per-session seed."""
    return int(hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:8], 16) + int(seed)


def draw_gap(spec: dict, r: random.Random) -> float:
    if spec["type"] == "normal":
        return min(max(r.gauss(spec["mean"], spec["std"]), spec["min"]), spec["max"])
    if spec["type"] == "uniform":
        return r.uniform(spec["min"], spec["max"])
    raise ValueError(f"unknown gap type {spec['type']!r}")


def place_turns(turns: Sequence, gap: dict, same_speaker_pause: Tuple[float, float] = (0.3, 0.8),
                seed: int = 0, lead_s: float = 0.5) -> List[Placed]:
    """turns: objects with .turn_id, .speaker, .duration_s in dialogue order -> non-overlapping placement."""
    r = random.Random(seed)
    out: List[Placed] = []
    t, prev = lead_s, None
    for tu in turns:
        if prev is None:
            g = 0.0
        elif tu.speaker == prev.speaker:
            g = r.uniform(*same_speaker_pause)
        else:
            g = draw_gap(gap, r)
        start = t + g
        end = start + float(tu.duration_s)
        out.append(Placed(tu.turn_id, tu.speaker, round(start, 4), round(end, 4), round(g, 4)))
        t, prev = end, tu
    return out


def render(placed: List[Placed], audio: Dict[str, np.ndarray], speakers: List[str], sr: int,
           tail_s: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """placement + per-turn float32 mono audio at sr -> (mix (N,), per-speaker channels (N, len(speakers)))."""
    n = int(round((max(p.end_s for p in placed) + tail_s) * sr)) if placed else 0
    st = np.zeros((n, len(speakers)), dtype=np.float32)
    for p in placed:
        x = audio[p.turn_id]
        a = int(round(p.start_s * sr))
        b = min(a + len(x), n)
        st[a:b, speakers.index(p.speaker)] += x[: b - a]
    return np.clip(st.sum(1), -1.0, 1.0), np.clip(st, -1.0, 1.0)


def trim_silence(x: np.ndarray, sr: int, frame_s: float = 0.02, rel: float = 0.1, margin_s: float = 0.1) -> Tuple[int, int]:
    """Leading/trailing silence boundaries (sample indices): first/last 20 ms frame whose RMS exceeds
    rel x the 95th-percentile frame RMS, widened by margin. Push-to-talk turn files (TAXI) carry ~0.9 s
    leading and ~0.4 s trailing silence (medians) that would otherwise inflate the gaps."""
    fr = max(int(frame_s * sr), 1)
    n = len(x) // fr
    if n == 0:
        return 0, len(x)
    e = np.sqrt((x[: n * fr].reshape(n, fr) ** 2).mean(1))
    thr = max(float(np.percentile(e, 95)) * rel, 1e-4)
    on = np.nonzero(e > thr)[0]
    if len(on) == 0:
        return 0, len(x)
    m = int(margin_s * sr)
    return max(int(on[0]) * fr - m, 0), min((int(on[-1]) + 1) * fr + m, len(x))


def overlap_ratio(placed: List[Placed]) -> float:
    """Time with both speakers talking / time with at least one speaker talking (turn intervals)."""
    ev = sorted([(p.start_s, 1, p.speaker) for p in placed] + [(p.end_s, -1, p.speaker) for p in placed])
    active: Dict[str, int] = {}
    last, both, speech = None, 0.0, 0.0
    for t, d, s in ev:
        if last is not None:
            k = sum(1 for v in active.values() if v > 0)
            speech += (t - last) if k >= 1 else 0.0
            both += (t - last) if k >= 2 else 0.0
        active[s] = active.get(s, 0) + d
        last = t
    return both / speech if speech > 0 else 0.0

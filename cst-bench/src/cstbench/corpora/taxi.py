"""BAS TAXI reader (German dispatcher <-> English client, telephone dialogues).

Expected layout (BAS CLARIN repository download, corpus version 2.5):
  <root>/SESxxxx/<turn>.{wav,al,par,_annot.json}   and   <root>/SESxxxx/SESxxxx.cmdi.xml
Turn name = 4-digit session number + role + 2-digit turn number, roles:
  DP dispatcher (German), CL client (English), PD pre-dialogue, HP hang-up (PD/HP are discarded).
  .wav  16-bit PCM, 8 kHz mono (telephone); .al is the same signal A-law coded.
  .par  BAS Partitur file (UTF-8): ORT words, KAN pronunciations, NOI noise, TLN human translation
        ("DE>EN <text>", one line per turn). DP/CL turns without a .par were rated "garbage" by the
        corpus validators (overlap, meta talk, empty, noise only, wrong language) and are marked usable=False.
  Turns were segmented by DTMF push-to-talk, so the corpus has no overlap and no inter-turn timing
  (timing = "turn_order_only").
ORT notation (SpeechDat): [fil] filled pause, ** unintelligible, ~ truncated word (either side),
  * hardly recognizable word (prefix).
License: the corpus may not be redistributed, even partly, without written permission of the copyright
holders (University of Munich, DFKI). This reader therefore marks sessions redistributable=False.
"""
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
import re
import wave

from ..schema import Session, Translation, Turn

ROLES = {"DP": ("German", "dispatcher"), "CL": ("English", "client")}
LANG_CODE = {"DE": "German", "EN": "English"}
TURN_RE = re.compile(r"^(\d{4})(DP|CL|PD|HP)(\d{2})$")
EXPECTED = {"sessions": 86, "usable_turns": 640}          # TAXI version 2.5 as distributed by BAS


def parse_par(text: str) -> dict:
    """BAS Partitur text -> {header, ort[(i, word)], noise[{span, label}], tln[(span, direction, text)]}."""
    header, ort, noise, tln = {}, [], [], []
    for line in text.splitlines():
        m = re.match(r"^([A-Z]{3}):\s?(.*)$", line)
        if not m:
            continue
        tag, rest = m.group(1), m.group(2)
        if tag == "ORT":
            i, w = rest.split("\t", 1)
            ort.append((int(i), w.strip()))
        elif tag == "NOI":
            span, lab = rest.split("\t", 1)
            noise.append({"span": span, "label": lab.strip()})
        elif tag == "TLN":
            span, direction, txt = rest.split("\t", 2)
            tln.append((span, direction, txt.strip()))
        elif tag != "KAN":
            header[tag] = rest.strip()
    ort.sort()
    return dict(header=header, ort=ort, noise=noise, tln=tln)


def clean_ort(words: List[str]) -> str:
    """Evaluation transcript: drop [..] markers and '**', strip '*' and '~' marks (word fragments are kept)."""
    out = []
    for w in words:
        if (w.startswith("[") and w.endswith("]")) or w == "**":
            continue
        w = w.lstrip("*").strip("~")
        if w:
            out.append(w)
    return " ".join(out)


def _wav_info(path: Path) -> Tuple[int, int]:
    with wave.open(str(path)) as w:
        return w.getnframes(), w.getframerate()


def load_session(ses_dir: Path, root: Path) -> Session:
    """One SESxxxx directory -> Session with DP/CL turns in dialogue order (PD/HP discarded)."""
    turns: List[Turn] = []
    rates = set()
    for wav in sorted(ses_dir.glob("[0-9]*.wav")):
        m = TURN_RE.match(wav.stem)
        if not m or m.group(2) not in ROLES:
            continue
        role, idx = m.group(2), int(m.group(3))
        lang, _ = ROLES[role]
        n, sr = _wav_info(wav)
        rates.add(sr)
        t = Turn(turn_id=wav.stem, index=idx, speaker=role, lang=lang, audio=wav.relative_to(root).as_posix(),
                 num_samples=n, duration_s=round(n / sr, 4), usable=False)
        par = wav.with_suffix(".par")
        if par.exists():
            p = parse_par(par.read_text(encoding="utf-8"))
            words = [w for _, w in p["ort"]]
            t.words, t.transcript_raw, t.transcript, t.noise = words, " ".join(words), clean_ort(words), p["noise"]
            if p["tln"]:
                _, direction, txt = p["tln"][0]
                src, tgt = direction.split(">")
                if LANG_CODE.get(src) != lang:
                    raise ValueError(f"{wav}: TLN direction {direction} does not match speaker language {lang}")
                t.translation = Translation(lang=LANG_CODE[tgt], text=txt, source="human")
            t.usable = bool(t.transcript) and t.translation is not None
        turns.append(t)
    turns.sort(key=lambda t: t.index)
    if len(rates) > 1:
        raise ValueError(f"{ses_dir}: mixed sample rates {rates}")
    return Session(session_id=f"taxi/{ses_dir.name}", corpus="TAXI", redistributable=False,
                   speakers={r: {"lang": l, "role": ro} for r, (l, ro) in ROLES.items()},
                   audio={"root": ".", "sample_rate": rates.pop() if rates else 8000, "channel": "telephone"},
                   timing="turn_order_only", turns=turns,
                   meta={"license": "BAS TAXI: no redistribution without written permission", "corpus_version": "2.5"})


def iter_sessions(root: str) -> Iterator[Session]:
    root_p = Path(root)
    dirs = sorted(p for p in root_p.glob("SES[0-9]*") if p.is_dir() and not p.name.startswith("._"))
    for d in dirs:
        yield load_session(d, root_p)


def stats(sessions: List[Session]) -> Dict[str, object]:
    u = [t for s in sessions for t in s.usable_turns()]
    per = sorted(len(s.usable_turns()) for s in sessions)
    return dict(sessions=len(sessions), turns=sum(len(s.turns) for s in sessions), usable_turns=len(u),
                usable_DP=sum(t.speaker == "DP" for t in u), usable_CL=sum(t.speaker == "CL" for t in u),
                usable_minutes=round(sum(t.duration_s for t in u) / 60, 1),
                words=sum(len(t.transcript.split()) for t in u),
                sessions_without_usable=sum(1 for n in per if n == 0),
                usable_turns_per_session_median=per[len(per) // 2] if per else 0)

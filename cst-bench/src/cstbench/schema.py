"""Common conversation schema (one JSON object per session, stored as JSON Lines).

Session
  session_id        "<corpus>/<source session id>"
  corpus            corpus name (e.g. "TAXI")
  redistributable   whether the source audio/text may be redistributed; if False only IDs, specs and
                    scripts are released and users rebuild from their own licensed copy
  speakers          {speaker_id: {"lang": ..., "role": ...}}
  audio             {"root": ..., "sample_rate": ..., "channel": ...}; turn audio paths are relative to root
  timing            "turn_order_only" (no inter-turn timing in the source) | "absolute" (turn start/end given)
  turns             list of Turn in dialogue order
  meta              free-form provenance (license, corpus version, ...)

Turn
  turn_id, index, speaker, lang, audio, num_samples, duration_s, usable
  words             source word tokens, corpus notation kept
  transcript_raw    words joined by spaces
  transcript        evaluation transcript (corpus markup removed)
  noise             corpus noise annotations [{"span": ..., "label": ...}]
  translation       reference translation into the other speaker's language {"lang", "text", "source"}
  start_s, end_s    only when timing == "absolute"
"""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class Translation:
    lang: str
    text: str
    source: str = "human"


@dataclass
class Turn:
    turn_id: str
    index: int
    speaker: str
    lang: str
    audio: Optional[str]
    num_samples: int = 0
    duration_s: float = 0.0
    usable: bool = True
    words: List[str] = field(default_factory=list)
    transcript_raw: str = ""
    transcript: str = ""
    noise: List[dict] = field(default_factory=list)
    translation: Optional[Translation] = None
    start_s: Optional[float] = None
    end_s: Optional[float] = None


@dataclass
class Session:
    session_id: str
    corpus: str
    redistributable: bool
    speakers: Dict[str, dict]
    audio: dict
    timing: str = "turn_order_only"
    turns: List[Turn] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def usable_turns(self) -> List[Turn]:
        return [t for t in self.turns if t.usable]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @staticmethod
    def from_json(line: str) -> "Session":
        d = json.loads(line)
        turns = []
        for t in d.pop("turns"):
            tr = t.pop("translation")
            turns.append(Turn(**t, translation=Translation(**tr) if tr else None))
        return Session(**d, turns=turns)

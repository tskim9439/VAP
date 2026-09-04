"""모든 코퍼스가 공유하는 대화 표현. 오디오는 16 kHz float32 (2, T) — 채널 0 = speaker A, 1 = speaker B."""
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

SR = 16000
Segment = Tuple[float, float]          # (start_s, end_s)

@dataclass
class Utterance:
    speaker: int                       # 0 | 1
    start: float
    end: float
    text: str = ""
    label: str = ""                    # 코퍼스 원 라벨 (otoSpeech: "Normal Turn" 등, AI Hub: 감정 등)

@dataclass
class Conversation:
    id: str
    source: str                        # aihub | otoSpeech | turnbench-dev
    audio: Optional[np.ndarray]        # (2, T) float32 @16k, None 이면 미로드
    duration: float
    vad: List[List[Segment]]           # [[(s,e)...] for A, [(s,e)...] for B] — 학습에 쓰는 VAD (초)
    vad_source: str                    # "energy" | "label" | "energy∩label"
    utterances: List[Utterance] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

"""CST 공통 대화 스키마(wiki/tasks/task-cst-conversation-schema-parser.md).

한 세션 = 같은 문맥을 공유하는 두 화자의 대화 하나. JSONL 한 줄에 세션 하나.

  session_id        "<corpus>/<원본 세션 id>"
  corpus            코퍼스 이름(TAXI, XDailyDialog, BConTrasT, CST-Bench-Syn …)
  redistributable   원본 음성·텍스트를 공개본에 넣을 수 있는가(TAXI·VM 은 False → ID·스크립트만 공개)
  speakers          {화자 id: {lang, role}}
  audio             {root, sample_rate, channel}: turns[].audio 는 root 기준 상대 경로
  timing            "turn_order_only"(원본 턴 간 시각 없음) | "absolute"(turns[].start_s·end_s 가 대화 시각)
  turns             턴 목록(index 순). 각 턴:
    turn_id, index, speaker, lang, audio, num_samples, duration_s, usable
    words            원본 단어 토큰(코퍼스 표기 그대로)
    transcript_raw   words 를 공백으로 이은 것
    transcript       평가용 정리본(표기 기호·채운 쉼·알아들을 수 없는 말 제거)
    noise            [{span, label}] 코퍼스 잡음 표시
    translation      {lang, text, source} 상대 언어 참조 번역(source: "human" | "silver:<방법>")
    start_s, end_s   timing == "absolute" 일 때만
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
        return json.dumps(asdict(self), ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Session":
        d = json.loads(line)
        turns = []
        for t in d.pop("turns"):
            tr = t.pop("translation")
            turns.append(Turn(**t, translation=Translation(**tr) if tr else None))
        return Session(**d, turns=turns)

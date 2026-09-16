"""Phase 2 대화(dialogue) 계층 — N 화자 대화의 참조 표현.

정본: wiki/outputs/output-phase2-lane-plan.md §3–§5. 기존 `streams.jsonl`(단일 lane, 비겹침) 스키마와 별개이며 그것을 바꾸지 않는다.

  Utterance  : 라벨(또는 정렬)에서 온 화자별 발화. start/end 는 대화 시간축(초), text 는 TN target, tokens 는 정렬 결과 [(id, end_time_s)].
  Episode    : 같은 화자의 발화를 gap < GAP_S 로 병합한 음향 발화 구간(정본 "segment"). lane 배정·ONSET/EOT 후보의 단위.
  Dialogue   : conv_id, corpus, lang, split, duration_s, speakers(원 화자 ID), utterances, channels(화자 → 오디오 참조), observed_until_s.

관측 한계: `observed_until_s` 는 정답을 만들 때 볼 수 있는 미래의 끝(대화 끝 또는 결손 시작)이다. 모델의 관측과 다르다(정본 §5.2 label_observed_until).
"""
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
import json

GAP_S = 0.25          # 같은 화자 발화 병합 gap (정본 §3.2·§5: 화자별 VAD, gap < 0.25 s 병합)
CHUNK_S = 0.08        # audio clock

@dataclass
class Utterance:
    speaker: str; start: float; end: float; text: str = ""; raw: str = ""
    tokens: Optional[List[Tuple[int, float]]] = None      # 정렬 결과 [(token_id, end_time_s)] — 대화 시간축
    utt_id: str = ""
    word_timing: Optional[List[Tuple[str, float, float]]] = None   # 코퍼스 제공 단어 시각(NOTSOFAR-1) — 정렬 QC 용
    flags: Optional[List[str]] = None                     # TN quarantine 사유(있으면 text="" 로 두고 lexical 감독 제외)

@dataclass
class Episode:
    ep_id: int; speaker: str; start: float; end: float
    tokens: List[Tuple[int, float]] = field(default_factory=list)   # 병합된 발화의 토큰(시간순, end_time 은 [start,end] 로 clamp)
    utt_ids: List[str] = field(default_factory=list)
    lane: Optional[int] = None          # 1..R, None = lane_capacity_exhausted
    generation: int = 0                 # lane 소유 세대(재배정마다 +1)
    outcome: str = ""                   # eot_soft.classify 결과
    p_end: Optional[float] = None       # None = mask(미관측)

@dataclass
class ChannelRef:
    """화자별 오디오 참조. path 는 streams.py 규약(`wav#chN`, `archive::member`)과 같다. pieces 는 조각 코퍼스(134-1/134-2)용:
    [(path, offset_in_dialogue_s)] — 조각을 대화 시간축 offset 에 놓고 나머지는 무음."""
    path: str = ""; pieces: Optional[List[Tuple[str, float]]] = None; sr: int = 16000

@dataclass
class Dialogue:
    conv_id: str; corpus: str; lang: str; split: str; duration_s: float
    speakers: List[str]; utterances: List[Utterance]
    channels: Dict[str, ChannelRef] = field(default_factory=dict)
    observed_until_s: Optional[float] = None
    meta: Dict = field(default_factory=dict)

    def observed(self) -> float: return self.duration_s if self.observed_until_s is None else self.observed_until_s

    def to_json(self) -> str:
        d = asdict(self); return json.dumps(d, ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Dialogue":
        d = json.loads(s)
        d["utterances"] = [Utterance(**{**u, "tokens": [tuple(t) for t in u["tokens"]] if u.get("tokens") else None}) for u in d["utterances"]]
        d["channels"] = {k: ChannelRef(**{**v, "pieces": [tuple(p) for p in v["pieces"]] if v.get("pieces") is not None else None}) for k, v in d.get("channels", {}).items()}   # pieces=[] (조각 없는 화자)는 [] 로 보존 — None 이면 path="" 를 열려다 실패(2026-09-16)
        return Dialogue(**d)

def build_episodes(dlg: Dialogue, gap_s: float = GAP_S) -> List[Episode]:
    """화자별 발화를 시간순으로 gap < gap_s 이면 병합 → Episode 목록(시작 시각순, 동시면 화자 ID 순). ep_id 는 그 순서."""
    by: Dict[str, List[Utterance]] = {}
    for u in dlg.utterances:
        if u.end <= u.start: continue
        by.setdefault(u.speaker, []).append(u)
    eps: List[Tuple[float, str, Episode]] = []
    for spk, us in by.items():
        us.sort(key=lambda u: (u.start, u.end)); cur: Optional[Episode] = None
        for u in us:
            toks = [(tid, min(max(t, u.start), u.end)) for tid, t in (u.tokens or [])]
            if cur is not None and u.start - cur.end < gap_s:
                cur.end = max(cur.end, u.end); cur.tokens += toks; cur.utt_ids.append(u.utt_id)
            else:
                if cur is not None: eps.append((cur.start, spk, cur))
                cur = Episode(ep_id=-1, speaker=spk, start=u.start, end=u.end, tokens=list(toks), utt_ids=[u.utt_id])
        if cur is not None: eps.append((cur.start, spk, cur))
    eps.sort(key=lambda x: (x[0], x[1])); out = []
    for i, (_, _, e) in enumerate(eps):
        e.ep_id = i; e.tokens.sort(key=lambda t: t[1]); out.append(e)
    return out

def chunk_of(t_s: float, delay: int = 0, chunk_s: float = CHUNK_S) -> int:
    """청크 인덱스 = floor(t/0.08) + δ (E2 규약)."""
    return int(t_s / chunk_s + 1e-9) + delay

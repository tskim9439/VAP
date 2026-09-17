"""Phase 2 lane 상태 디코더·파서 — 정본 output-phase2-lane-plan §7(추론)·§10(평가).

decode_p2: stream_decode 와 같은 free-running 청크 디코드에 (1) audio 위치 hidden 의 lane 활동 확률(act_head), (2) 방출 토큰의 청크 번호를 함께 돌려준다.
LaneParser: 방출 토큰열(lane 태그·<ONSET>·텍스트·<EOT>) 을 lane 별 구간(segment) 으로 파싱한다 — 태그가 현재 lane 을 고르고, ONSET 이 구간을 열고, 텍스트는 열린 구간에 붙고, EOT 가 닫는다.
평가 보조: 편집거리(CER/WER), 토큰 정렬 지연(latency), 이벤트(ONSET/EOT) 매칭.
"""
import difflib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np, torch
from ..data.dialogue import CHUNK_S

@dataclass
class Segment:
    lane: int; k_on: Optional[int]; k_eot: Optional[int]
    tokens: List[Tuple[int, int]] = field(default_factory=list)     # (token id, 방출 청크 k)
    implicit: bool = False                                          # ONSET 없이 텍스트가 먼저 나와 연 구간
    @property
    def start(self) -> Optional[float]: return None if self.k_on is None else (self.k_on + 1) * CHUNK_S
    @property
    def end(self) -> Optional[float]: return None if self.k_eot is None else (self.k_eot + 1) * CHUNK_S

class LaneParser:
    """registry: {"<SPK_A>": id, …, "<ONSET>": id, "<EOT>": id}; lane_ids: lane 순서대로의 태그 id 목록(lane 1 = <SPK_A>)."""
    def __init__(self, registry: Dict[str, int], R: int = 6):
        names = ["<SPK_A>", "<SPK_B>"] + [f"<SPK_{i}>" for i in range(3, R + 1)]
        self.lane_of = {registry[n]: i + 1 for i, n in enumerate(names) if n in registry}; self.onset = registry["<ONSET>"]; self.eot = registry["<EOT>"]; self.R = R
    def parse(self, emits: List[Tuple[int, int]]) -> Tuple[List[Segment], dict]:
        """emits: [(chunk k, token id)] → (segments, stats{stray_eot, implicit, text_no_lane, unclosed})."""
        open_seg: Dict[int, Segment] = {}; segs: List[Segment] = []; cur: Optional[int] = None; st = dict(stray_eot=0, implicit=0, text_no_lane=0, unclosed=0, reopen=0)
        for k, tid in emits:
            if tid in self.lane_of: cur = self.lane_of[tid]; continue
            if tid == self.onset:
                if cur is None: st["text_no_lane"] += 1; continue
                if cur in open_seg: st["reopen"] += 1; continue                       # 이미 열린 lane 의 ONSET 은 무시(정본: lane 당 한 구간)
                open_seg[cur] = Segment(cur, k, None); continue
            if tid == self.eot:
                if cur is None or cur not in open_seg: st["stray_eot"] += 1; continue
                s = open_seg.pop(cur); s.k_eot = k; segs.append(s); continue
            if cur is None: st["text_no_lane"] += 1; continue
            if cur not in open_seg: open_seg[cur] = Segment(cur, k, None, implicit=True); st["implicit"] += 1
            open_seg[cur].tokens.append((tid, k))
        for s in open_seg.values(): st["unclosed"] += 1; segs.append(s)
        segs.sort(key=lambda s: (s.k_on if s.k_on is not None else 10 ** 9, s.lane)); return segs, st

@torch.inference_mode()
def decode_p2(model, tok, wav: np.ndarray, lang: str = "English", delay: int = 4, runaway_cap: Optional[int] = None, max_flush_rounds: int = 8, max_total_per_chunk: float = 6.0, next_bias: float = 0.0) -> dict:
    """wav (T,) float32 16 kHz → dict(emits=[(k, tid)], forced, rounds, K, act=(K,R) 활동 확률, ticks_ms).
    규약은 modeling_vapasr.stream_decode 와 같고(blocked·runaway cap·flush), 매 청크의 audio 위치 hidden 에 act_head 를 적용한다."""
    from .infer import prefix_ids
    dev = next(model.parameters()).device; cap = model.config.runaway_cap if runaway_cap is None else runaway_cap
    w = torch.from_numpy(np.asarray(wav, dtype=np.float32)).to(dev); K = int(round(w.shape[0] / 16000 / CHUNK_S))
    feats = model.encode(w[None], torch.tensor([w.shape[0]], device=dev), torch.tensor([K], device=dev))[0]
    emb = model.get_input_embeddings(); ce = model.chunk_embed(feats[None])[0].to(emb.weight.dtype)
    from transformers import DynamicCache
    cache = DynamicCache(); base = model.thinker.model; head = model.thinker.lm_head
    use_ac = dev.type == "cuda" and emb.weight.dtype == torch.float32
    def step(e):
        h = base(inputs_embeds=e.view(1, -1, e.shape[-1]), past_key_values=cache, use_cache=True).last_hidden_state[0, -1]
        return head(h).float(), h
    import time
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
        step(emb(torch.tensor(prefix_ids(model, tok, lang, delay), device=dev)))
        e_next = emb.weight[model.next_audio]; e_empty = emb.weight[model.empty_audio]; total = 0; max_total = int(max_total_per_chunk * K) + 64
        emits, forced, ticks, act = [], 0, [], []
        def emit_round(k, logits):
            nonlocal total, forced; n = 0
            while True:
                logits[model.blocked] = float("-inf"); logits[model.next_audio] -= next_bias; tid = int(logits.argmax())
                if tid == model.next_audio or n >= cap or total >= max_total:
                    forced += int(tid != model.next_audio); step(e_next); return n
                emits.append((k, tid)); n += 1; total += 1; logits, _ = step(emb.weight[tid])
        for k in range(K):
            t = time.time(); logits, h = step(ce[k])
            if model.act_head is not None: act.append(torch.sigmoid(model.act_head(h.float() if model.act_head[0].weight.dtype == torch.float32 else h)).float().cpu().numpy())
            emit_round(k, logits)
            if dev.type == "cuda": torch.cuda.synchronize()
            ticks.append((time.time() - t) * 1000)
        rounds = 0
        for r in range(max_flush_rounds):
            rounds += 1; logits, _ = step(e_empty)
            if emit_round(K + r, logits) == 0: break
    return dict(emits=emits, forced=forced, rounds=rounds, K=K, act=(np.stack(act) if act else np.zeros((K, 0))), ticks_ms=ticks)

# ── 평가 보조
def edit_distance(a: List, b: List) -> int:
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1): cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]

def cer(hyp: str, ref: str) -> Optional[float]:
    r = ref.replace(" ", ""); h = hyp.replace(" ", "")
    return None if not r else edit_distance(list(h), list(r)) / len(r)

def wer(hyp: str, ref: str) -> Optional[float]:
    r = ref.split(); h = hyp.split()
    return None if not r else edit_distance(h, r) / len(r)

def token_latency(hyp: List[Tuple[int, int]], ref: List[Tuple[int, float]]) -> Tuple[List[float], int]:
    """hyp [(tid, k)] · ref [(tid, t)] 를 토큰 id 로 정렬해 일치 토큰의 (방출 청크 끝 − 참조 시각) 목록과 일치 수."""
    h_ids = [t for t, _ in hyp]; r_ids = [t for t, _ in ref]; lat = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, h_ids, r_ids, autojunk=False).get_opcodes():
        if tag == "equal":
            for d in range(i2 - i1): lat.append((hyp[i1 + d][1] + 1) * CHUNK_S - ref[j1 + d][1])
    return lat, len(lat)

def match_events(hyp_t: List[float], ref_t: List[float], tol: float) -> Tuple[int, int, int]:
    """시각 목록끼리 |Δ| ≤ tol 로 1:1 탐욕 매칭 → (일치 수, hyp 수, ref 수)."""
    used = [False] * len(ref_t); hit = 0
    for h in sorted(hyp_t):
        best, bi = tol + 1, -1
        for i, r in enumerate(ref_t):
            if not used[i] and abs(h - r) < best: best, bi = abs(h - r), i
        if bi >= 0 and best <= tol: used[bi] = True; hit += 1
    return hit, len(hyp_t), len(ref_t)

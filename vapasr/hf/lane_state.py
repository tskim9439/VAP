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
    k_act_close: Optional[int] = None                               # 활동 헤드 비활성으로 닫힌 청크(EOT 토큰 없음)
    @property
    def start(self) -> Optional[float]: return None if self.k_on is None else (self.k_on + 1) * CHUNK_S
    @property
    def end(self) -> Optional[float]:
        k = self.k_eot if self.k_eot is not None else self.k_act_close
        return None if k is None else (k + 1) * CHUNK_S

class LaneParser:
    """registry: {"<SPK_A>": id, …, "<ONSET>": id, "<EOT>": id}; lane_ids: lane 순서대로의 태그 id 목록(lane 1 = <SPK_A>)."""
    def __init__(self, registry: Dict[str, int], R: int = 6):
        names = ["<SPK_A>", "<SPK_B>"] + [f"<SPK_{i}>" for i in range(3, R + 1)]
        self.lane_of = {registry[n]: i + 1 for i, n in enumerate(names) if n in registry}; self.onset = registry["<ONSET>"]; self.eot = registry["<EOT>"]; self.R = R
    def parse(self, emits: List[Tuple[int, int]], act_closed: Optional[List[Tuple[int, int]]] = None) -> Tuple[List[Segment], dict]:
        """emits: [(chunk k, token id)] (+ act_closed: [(lane, k)] 활동 헤드로 닫힌 사건) → (segments, stats{stray_eot, implicit, text_no_lane, unclosed, reopen, act_closed})."""
        open_seg: Dict[int, Segment] = {}; segs: List[Segment] = []; cur: Optional[int] = None; st = dict(stray_eot=0, implicit=0, text_no_lane=0, unclosed=0, reopen=0, act_closed=0)
        ev = [(k, 1, tid, None) for k, tid in emits] + [(k, 0, None, l) for l, k in (act_closed or [])]    # 같은 청크면 활동 닫힘(0)이 그 청크의 토큰(1)보다 먼저(디코더의 tick 순서와 같음)
        ev.sort(key=lambda x: (x[0], x[1]))
        for k, kind, tid, l in ev:
            if kind == 0:
                if l in open_seg: sg = open_seg.pop(l); sg.k_act_close = k; segs.append(sg); st["act_closed"] += 1
                continue
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

class LaneStates:
    """디코더의 lane 상태(정본 §3.1 lazy-free): FREE / OPEN / HELD(닫힘, 마지막 EOT 청크 보관). 새 화자는 FREE 중 최소 번호, 없으면 가장 오래전에 닫힌 HELD lane.
    닫힘은 EOT 방출 또는 활동 헤드 비활성 ≥ act_close_chunks 청크(정본: 0.25 s)."""
    def __init__(self, R: int, act_close_chunks: int = 3, act_thr: float = 0.5):
        self.R = R; self.state = {l: "FREE" for l in range(1, R + 1)}; self.closed_at = {}; self.inactive = {l: 0 for l in range(1, R + 1)}; self.n_close = act_close_chunks; self.thr = act_thr; self.act_closed = []
    def open_lanes(self): return [l for l in range(1, self.R + 1) if self.state[l] == "OPEN"]
    def next_lane(self) -> Optional[int]:
        free = [l for l in range(1, self.R + 1) if self.state[l] == "FREE"]
        if free: return free[0]
        held = [l for l in range(1, self.R + 1) if self.state[l] == "HELD"]
        return min(held, key=lambda l: self.closed_at.get(l, -1)) if held else None
    def open(self, l): self.state[l] = "OPEN"; self.inactive[l] = 0
    def close(self, l, k): self.state[l] = "HELD"; self.closed_at[l] = k
    def tick(self, k, probs):
        """청크 k 의 활동 확률(R,) 로 비활성 카운트 갱신 → 활동으로 닫힌 lane 목록."""
        out = []
        for l in self.open_lanes():
            self.inactive[l] = self.inactive[l] + 1 if float(probs[l - 1]) < self.thr else 0
            if self.inactive[l] >= self.n_close: self.close(l, k); out.append(l); self.act_closed.append((l, k))
        return out

@torch.inference_mode()
def decode_p2(model, tok, wav: np.ndarray, lang: str = "English", delay: int = 4, runaway_cap: Optional[int] = None, max_flush_rounds: int = 8, max_total_per_chunk: float = 6.0, next_bias: float = 0.0,
              constrain: bool = True, act_close_chunks: int = 6, act_thr: float = 0.3, onset_thr: float = 0.35) -> dict:
    """wav (T,) float32 16 kHz → dict(emits=[(k, tid)], probs=[p(선택 토큰)], forced, rounds, K, act=(K,R) 활동 확률, ticks_ms, act_closed=[(lane, k)]).
    규약은 modeling_vapasr.stream_decode 와 같고(blocked·runaway cap·flush), 매 청크의 audio 위치 hidden 에 act_head 를 적용한다.
    constrain=True 면 lane 규약으로 후보를 제한한다(정본 §3·§5 를 추론에 적용):
      · 태그 직후: 그 lane 이 OPEN 이면 {EOT, 텍스트}(NEXT 불가 — soft EOT 라 p(EOT)<0.5 여도 NEXT 에 지지 않게), 닫혀 있으면 {ONSET} 만.
      · 그 외: NEXT · OPEN/HELD lane 태그(자기 lane 복귀) · 새 lane 태그(FREE 중 최소 번호 하나) · 현재 lane 이 OPEN 일 때만 텍스트. ONSET/EOT 는 태그 없이 못 나온다.
      · onset_thr: 태그가 argmax 가 아니어도 허용 태그의 최대 확률이 onset_thr 이상이면 태그를 낸다(δ_onset=0 이라 시작 청크의 증거가 80 ms 뿐이어서 argmax 는 시작을 자주 놓친다).
      · 활동 헤드가 act_close_chunks 청크 연속 act_thr 미만이면 그 lane 을 닫는다(ONSET 없이는 다시 텍스트를 못 붙임)."""
    from .infer import prefix_ids
    dev = next(model.parameters()).device; cap = model.config.runaway_cap if runaway_cap is None else runaway_cap
    w = torch.from_numpy(np.asarray(wav, dtype=np.float32)).to(dev); K = int(round(w.shape[0] / 16000 / CHUNK_S))
    feats = model.encode(w[None], torch.tensor([w.shape[0]], device=dev), torch.tensor([K], device=dev))[0]
    emb = model.get_input_embeddings(); ce = model.chunk_embed(feats[None])[0].to(emb.weight.dtype)
    from transformers import DynamicCache
    cache = DynamicCache(); base = model.thinker.model; head = model.thinker.lm_head
    use_ac = dev.type == "cuda" and emb.weight.dtype == torch.float32
    reg = dict(model.config.phase2_registry); R = model.config.lanes; parser = LaneParser(reg, R=R); tag_of_lane = {l: t for t, l in parser.lane_of.items()}
    NEXT, ONSET, EOT = model.next_audio, parser.onset, parser.eot; V = emb.weight.shape[0]
    special = torch.tensor(sorted([NEXT, ONSET, EOT] + list(tag_of_lane.values())), device=dev)
    lanes = LaneStates(R, act_close_chunks, act_thr); cur: Optional[int] = None; just_tag = False
    def allowed_mask(flush: bool):
        """(V,) bool — constrain 규칙에 따른 허용 집합."""
        if just_tag:                                                     # 태그 직후: OPEN lane 이면 EOT 또는 텍스트, 닫힌 lane 이면 ONSET
            if lanes.state[cur] == "OPEN": m = torch.ones(V, dtype=torch.bool, device=dev); m[special] = False; m[EOT] = True
            else: m = torch.zeros(V, dtype=torch.bool, device=dev); m[ONSET] = True
            return m
        text_ok = cur is not None and lanes.state[cur] == "OPEN"            # 태그 없는 텍스트는 현재 lane 이 OPEN 일 때만
        m = torch.ones(V, dtype=torch.bool, device=dev) if text_ok else torch.zeros(V, dtype=torch.bool, device=dev)
        m[special] = False; m[NEXT] = True
        for l in range(1, R + 1):
            if lanes.state[l] != "FREE": m[tag_of_lane[l]] = True            # OPEN(계속/닫기)·HELD(자기 lane 복귀)
        nl = lanes.next_lane()
        if nl is not None and not flush: m[tag_of_lane[nl]] = True           # 새 화자용 lane 은 하나
        return m
    tag_ids = torch.tensor([tag_of_lane[l] for l in range(1, R + 1)], device=dev)
    def step(e):
        h = base(inputs_embeds=e.view(1, -1, e.shape[-1]), past_key_values=cache, use_cache=True).last_hidden_state[0, -1]
        return head(h).float(), h
    import time
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_ac):
        step(emb(torch.tensor(prefix_ids(model, tok, lang, delay), device=dev)))
        e_next = emb.weight[NEXT]; e_empty = emb.weight[model.empty_audio]; total = 0; max_total = int(max_total_per_chunk * K) + 64
        emits, probs, forced, ticks, act = [], [], 0, [], []
        def emit_round(k, logits, flush=False):
            nonlocal total, forced, cur, just_tag; n = 0
            while True:
                logits[model.blocked] = float("-inf"); logits[NEXT] -= next_bias
                if constrain: logits[~allowed_mask(flush)] = float("-inf")
                tid = int(logits.argmax())
                if constrain and tid == NEXT and not just_tag and onset_thr > 0:
                    pr = torch.softmax(logits, 0)[tag_ids]; j = int(pr.argmax())
                    if float(pr[j]) >= onset_thr: tid = int(tag_ids[j])
                if tid == NEXT or n >= cap or total >= max_total:
                    forced += int(tid != NEXT); just_tag = False; step(e_next); return n
                p = float(torch.softmax(logits, 0)[tid]); emits.append((k, tid)); probs.append(round(p, 3)); n += 1; total += 1
                if tid in parser.lane_of: cur = parser.lane_of[tid]; just_tag = True
                else:
                    if tid == ONSET and cur is not None: lanes.open(cur)
                    elif tid == EOT and cur is not None: lanes.close(cur, k)
                    just_tag = False
                logits, _ = step(emb.weight[tid])
        for k in range(K):
            t = time.time(); logits, h = step(ce[k])
            if model.act_head is not None:
                pa = torch.sigmoid(model.act_head(h.float() if model.act_head[0].weight.dtype == torch.float32 else h)).float().cpu().numpy(); act.append(pa)
                if constrain: lanes.tick(k, pa)
            emit_round(k, logits)
            if dev.type == "cuda": torch.cuda.synchronize()
            ticks.append((time.time() - t) * 1000)
        rounds = 0
        for r in range(max_flush_rounds):
            rounds += 1; logits, _ = step(e_empty)
            if emit_round(K + r, logits, flush=True) == 0: break
    return dict(emits=emits, probs=probs, forced=forced, rounds=rounds, K=K, act=(np.stack(act) if act else np.zeros((K, 0))), ticks_ms=ticks, act_closed=lanes.act_closed, constrain=constrain)

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

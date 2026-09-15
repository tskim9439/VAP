"""대화 결합(session stitching) 합성 — 정본 §8: 같은 언어의 2화자 대화 2–4개를 turn 블록 단위로 교대·부분 겹침시켜 N=4–8, S≤4 세션을 만든다.

  블록: 각 원 대화를 mutual-silence 시각에서 block_s 길이로 자른다(내부 발화·간격·겹침은 원본 그대로).
  배치: 대화들을 round-robin 으로 이어 붙이되 이음새에서 seam_overlap_s 만큼 겹친다(직전 블록 끝과 다음 블록 시작이 살짝 겹침).
  감독: lexical·lane·활동은 유지, **EOT 는 전부 mask**(meta.mask_eot=True, 정본 §8 "합성에 자연 floor EOT 를 붙이지 않는다").
  화자 ID 는 "<원 대화 인덱스>:<원 화자>" 로 구별되고 채널은 원본 구간(pieces 4-튜플)로 참조한다. 목적: N>R 재배정 사례·KO 다자·긴 간격 재등장.
"""
import random, bisect
from typing import List, Tuple, Optional
from .dialogue import Dialogue, Utterance, ChannelRef

def _silence_points(d: Dialogue, grid: float = 0.5) -> List[float]:
    spans = sorted((u.start, u.end) for u in d.utterances); starts = [s for s, _ in spans]; pm = []; m = 0.0
    for s, e in spans: m = max(m, e); pm.append(m)
    out = []; t = 0.0
    while t <= d.duration_s:
        i = bisect.bisect_left(starts, t)
        if i == 0 or pm[i - 1] <= t: out.append(round(t, 3))
        t += grid
    return out

def _blocks(d: Dialogue, block_s: Tuple[float, float], rng: random.Random) -> List[Tuple[float, float]]:
    pts = _silence_points(d); out = []; t = 0.0
    while t < d.duration_s - block_s[0]:
        want = t + rng.uniform(*block_s); i = bisect.bisect_left(pts, want)
        cand = [p for p in pts[max(0, i - 3): i + 3] if p > t + block_s[0] * 0.5]
        if not cand: break
        t1 = min(cand, key=lambda p: abs(p - want)); out.append((t, t1)); t = t1
    if d.duration_s - t >= block_s[0] * 0.5: out.append((t, d.duration_s))
    return out

def stitch(dlgs: List[Dialogue], seed: int = 0, block_s: Tuple[float, float] = (10.0, 30.0), seam_overlap_s: Tuple[float, float] = (0.0, 1.5), conv_id: Optional[str] = None) -> Dialogue:
    assert len({d.lang for d in dlgs}) == 1, "같은 언어만 결합"
    rng = random.Random(seed); blocks = [_blocks(d, block_s, rng) for d in dlgs]; T = 0.0; utts = []; pieces = {}; sources = []
    queues = [list(b) for b in blocks]; order = list(range(len(dlgs))); rng.shuffle(order); k = 0
    while any(queues):
        j = order[k % len(order)]; k += 1
        if not queues[j]: continue
        b0, b1 = queues[j].pop(0); d = dlgs[j]; off = max(0.0, T - (rng.uniform(*seam_overlap_s) if T > 0 else 0.0)); shift = off - b0
        for u in d.utterances:
            if u.start < b0 or u.start >= b1: continue
            e = min(u.end, b1); toks = [(tid, t + shift) for tid, t in (u.tokens or []) if t <= b1] if u.tokens is not None else None
            utts.append(Utterance(speaker=f"{j}:{u.speaker}", start=u.start + shift, end=e + shift, text=u.text, raw=u.raw, tokens=toks, utt_id=f"{j}:{u.utt_id}", flags=u.flags))
        for spk, ref in d.channels.items():
            key = f"{j}:{spk}"; pieces.setdefault(key, [])
            if ref.pieces is not None:
                for pc in ref.pieces:
                    p, po = pc[0], pc[1]; src0, dur = (pc[2], pc[3]) if len(pc) >= 4 else (None, None)
                    if b0 <= po < b1: pieces[key].append((p, po + shift) if src0 is None else (p, po + shift, src0, dur))
            else: pieces[key].append((ref.path, off, b0, b1 - b0))
        sources.append(dict(src=d.conv_id, block=[b0, b1], at=off)); T = off + (b1 - b0)
    spk = sorted({u.speaker for u in utts}) or sorted(pieces)
    return Dialogue(conv_id=conv_id or f"stitch:{'+'.join(d.conv_id for d in dlgs)}:{seed}", corpus=f"stitch-{dlgs[0].corpus}", lang=dlgs[0].lang, split=dlgs[0].split, duration_s=round(T, 3),
                    speakers=spk, utterances=sorted(utts, key=lambda u: (u.start, u.speaker)), channels={s: ChannelRef(pieces=pieces.get(s, [])) for s in spk},
                    meta=dict(synthetic="stitch", mask_eot=True, sources=sources, n_sources=len(dlgs)))

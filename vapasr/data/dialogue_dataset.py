"""Phase 2 학습 창 Dataset — dialogues.jsonl(+정렬 parts) → mono 혼합 창 + lane 태그·ONSET·EOT(soft) 시퀀스 + lane 활동 타깃 (정본 §4·§5·§6·§9).

샘플 키는 MonoStreamDataset(vapasr.uslm.mono_data) 과 호환: wav, ids, is_audio, chunk_of, labels, lang, delay, name, id, n_text, overflow, n_flush, K, rounds
  + soft_pos/soft_alt/soft_w (EOT soft target), activity (K,R) float, lanes (L,) int, t0.
창: 길이 window_s 범위, 시작은 아무도 말하지 않는 시각(부분 발화 방지), 끝은 잘릴 수 있다(잘린 episode 는 토큰을 창 안 것만 두고 EOT 는 mask).
제외 창: 전사 없는 음성(quarantine·미정렬·결손 조각)이 창 안에 있으면 건너뛴다(모델이 '음성인데 전사 없음' 을 배우지 않게) — 건수는 stats 에.
"""
import os, json, glob, math, random, bisect
from typing import List, Dict, Tuple, Optional
import numpy as np, torch
from torch.utils.data import Dataset
from .dialogue import Dialogue, Utterance, Episode, build_episodes, chunk_of, CHUNK_S
from .lane_alloc import allocate
from .eot_soft import assign_p_end
from .dialogue_interleave import serialize, flatten, lane_activity
from .dialogue_tokens import add_phase2_specials, lane_specials_of, R_LANES
from .dialogue_mix import mix_dialogue, crop_audio
from .streams import SR

def load_align_parts(align_dir: str) -> Tuple[Dict[str, Dict[str, list]], set]:
    """parts/*.jsonl → {conv_id: {utt_id: [(id, end_time)...]}}, 실패 utt_id 집합(conv_id/utt_id)."""
    out: Dict[str, Dict[str, list]] = {}; fail = set()
    for pf in sorted(glob.glob(os.path.join(align_dir, "parts", "*.jsonl"))):
        for line in open(pf, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: continue
            out.setdefault(r["conv_id"], {}).update({k: [tuple(t) for t in v] for k, v in r.get("utts", {}).items()})
            for u in r.get("fail", []): fail.add(f"{r['conv_id']}/{u}")
    return out, fail

class DialogueWindowDataset(Dataset):
    def __init__(self, dialogues: List[str], tok, align_dir: Optional[str] = None, R: int = R_LANES, window_s: Tuple[float, float] = (20.0, 40.0), hop_s: float = 10.0,
                 delays=(2, 3, 4, 6), delay_onset: int = 0, policy: str = "lazy_free", max_per_chunk: int = 0, seed: int = 0, mix_kw: Optional[dict] = None,
                 max_items: Optional[int] = None, allow_unaligned: bool = False, cache_convs: int = 4):
        self.tok = tok; self.sp_ids = add_phase2_specials(tok); self.sp = lane_specials_of(self.sp_ids, R); self.R = R; self.delays = tuple(delays); self.delay_onset = delay_onset
        self.policy = policy; self.M = max_per_chunk; self.mix_kw = mix_kw or {}; self.allow_unaligned = allow_unaligned; self.cache_convs = cache_convs
        self.audio_pad = tok.convert_tokens_to_ids("<|audio_pad|>"); self._pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        self.dlgs: Dict[str, Dialogue] = {}; self.stats = dict(dialogues=0, windows=0, skipped_untranscribed=0, skipped_unaligned=0, no_start=0)
        aligned, failed = load_align_parts(align_dir) if align_dir else ({}, set())
        for path in dialogues:
            for line in open(path, encoding="utf-8"):
                d = Dialogue.from_json(line); toks = aligned.get(d.conv_id, {})
                for u in d.utterances:
                    if u.utt_id in toks: u.tokens = toks[u.utt_id]
                self.dlgs[d.conv_id] = d; self.stats["dialogues"] += 1
        self.items: List[Tuple[str, float, float]] = []; rng = random.Random(seed)
        for cid, d in self.dlgs.items(): self.items += self._windows(d, window_s, hop_s, rng)
        rng.shuffle(self.items)
        if max_items: self.items = self.items[:max_items]
        self._mono: Dict[str, np.ndarray] = {}; self._order: List[str] = []

    # ── 창 선택
    def _windows(self, d: Dialogue, window_s, hop_s, rng) -> List[Tuple[str, float, float]]:
        spans = sorted((u.start, u.end) for u in d.utterances); starts = [s for s, _ in spans]; ends_max = []; m = 0.0
        for s, e in spans: m = max(m, e); ends_max.append(m)             # prefix max of ends → "t 이전에 시작한 발화가 t 를 넘는가"
        def silent(t):
            i = bisect.bisect_left(starts, t); return i == 0 or ends_max[i - 1] <= t
        bad = [(u.start, u.end) for u in d.utterances if (u.flags or (not u.text)) or (u.text and not u.tokens and not self.allow_unaligned)]
        missing = set(d.meta.get("missing", [])); bad += [(u.start, u.end) for i, u in enumerate(d.utterances, 1) if i in missing]
        out = []; t = 0.0; lo, hi = window_s
        while t + lo <= d.duration_s:
            if not silent(t): self.stats["no_start"] += 1; t += hop_s; continue
            L = min(rng.uniform(lo, hi), d.duration_s - t); t1 = t + L
            if any(s < t1 and e > t for s, e in bad): self.stats["skipped_untranscribed"] += 1; t += hop_s; continue
            out.append((d.conv_id, round(t, 3), round(L, 3))); self.stats["windows"] += 1; t += hop_s
        return out

    def __len__(self): return len(self.items)
    def prefix(self, lang: str, delay: int) -> List[int]:
        return self._pre + self.tok(f"language {lang}<asr_text>", add_special_tokens=False)["input_ids"] + [self.sp_ids[f"<DELAY_{delay}>"]]

    def mono(self, cid: str) -> np.ndarray:
        if cid not in self._mono:
            self._mono[cid], _ = mix_dialogue(self.dlgs[cid], **self.mix_kw); self._order.append(cid)
            while len(self._order) > self.cache_convs: self._mono.pop(self._order.pop(0), None)
        return self._mono[cid]

    def window_episodes(self, d: Dialogue, t0: float, L: float) -> Tuple[List[Episode], dict]:
        """창 [t0, t0+L) 의 episode(시각은 창 기준). 창 끝을 넘는 episode 는 토큰을 잘라내고 EOT 를 mask 한다."""
        t1 = t0 + L; utts = []
        for u in d.utterances:
            if u.end <= t0 or u.start >= t1: continue
            toks = [(tid, t - t0) for tid, t in (u.tokens or []) if t <= t1]
            utts.append(Utterance(speaker=u.speaker, start=max(u.start, t0) - t0, end=min(u.end, t1) - t0, text=u.text, raw=u.raw, tokens=toks, utt_id=u.utt_id))
        sub = Dialogue(conv_id=d.conv_id, corpus=d.corpus, lang=d.lang, split=d.split, duration_s=L, speakers=d.speakers, utterances=utts)
        eps = build_episodes(sub); st = allocate(eps, R=self.R, policy=self.policy, delay_onset=self.delay_onset)
        hist = assign_p_end(eps, d.observed() - t0)
        for ep in eps:
            if ep.end >= L - 1e-6 and any(u.end > t1 for u in d.utterances if u.speaker == ep.speaker and u.start < t1 and u.end > t0): ep.outcome = "truncated"; ep.p_end = None
            if d.meta.get("mask_eot"): ep.outcome = "synthetic"; ep.p_end = None          # 합성(stitch): EOT 감독 없음(정본 §8)
        return eps, dict(alloc=st, outcomes=hist)

    def sequence(self, i: int, delay: int):
        cid, t0, L = self.items[i]; d = self.dlgs[cid]; K = int(round(L / CHUNK_S)); eps, info = self.window_episodes(d, t0, L)
        chunks, st = serialize(eps, (K - 0.5) * CHUNK_S, self.sp, delay_text=delay, delay_onset=self.delay_onset, max_per_chunk=self.M)
        f = flatten(chunks, K, self.audio_pad, self.sp.empty_audio, self.prefix(d.lang, delay)); act = lane_activity(eps, K, self.R)
        return dict(f, K=K, activity=act, stats=st, info=info, cid=cid, t0=t0, L=L, lang=d.lang, corpus=d.corpus)

    def __getitem__(self, i):
        delay = random.choice(self.delays); s = self.sequence(i, delay)
        wav = crop_audio(self.mono(s["cid"]), s["t0"], s["t0"] + s["L"])
        return dict(wav=torch.from_numpy(np.ascontiguousarray(wav)), ids=torch.tensor(s["ids"]), is_audio=torch.tensor(s["is_audio"]), chunk_of=torch.tensor(s["chunk_of"]), labels=torch.tensor(s["labels"]),
                    soft_pos=torch.tensor(s["soft_pos"], dtype=torch.long), soft_alt=torch.tensor(s["soft_alt"], dtype=torch.long), soft_w=torch.tensor(s["soft_w"], dtype=torch.float),
                    activity=torch.tensor(s["activity"], dtype=torch.float), lanes=torch.tensor(s["lanes"]), lang=s["lang"], delay=delay, name=s["corpus"], id=f"{s['cid']}@{s['t0']}",
                    n_text=s["stats"].text, overflow=s["stats"].overflow, n_flush=sum(1 for c in s["chunk_of"] if c < 0) - len(self._pre), K=s["K"], rounds=0, t0=s["t0"])

def collate_dialogue(batch):
    """collate_streams 와 같은 패딩 + soft(배치 평탄화: soft_b, soft_pos, soft_alt, soft_w) + activity (B,K,R) 와 activity_mask (B,K)."""
    from ..uslm.mono_data import collate_streams
    out = collate_streams([{k: v for k, v in b.items() if k not in ("soft_pos", "soft_alt", "soft_w", "activity", "lanes", "t0")} for b in batch])
    out["soft_b"] = torch.cat([torch.full_like(b["soft_pos"], i) for i, b in enumerate(batch)]); out["soft_pos"] = torch.cat([b["soft_pos"] for b in batch])
    out["soft_alt"] = torch.cat([b["soft_alt"] for b in batch]); out["soft_w"] = torch.cat([b["soft_w"] for b in batch])
    B = len(batch); K = max(b["activity"].shape[0] for b in batch); R = batch[0]["activity"].shape[1]
    act = torch.zeros(B, K, R); am = torch.zeros(B, K)
    for i, b in enumerate(batch): k = b["activity"].shape[0]; act[i, :k] = b["activity"]; am[i, :k] = 1
    out["activity"] = act; out["activity_mask"] = am; return out

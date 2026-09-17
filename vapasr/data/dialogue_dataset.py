"""Phase 2 학습 창 Dataset — dialogues.jsonl(+정렬 parts) → mono 혼합 창 + lane 태그·ONSET·EOT(soft) 시퀀스 + lane 활동 타깃 (정본 §4·§5·§6·§9).

샘플 키는 MonoStreamDataset(vapasr.uslm.mono_data) 과 호환: wav, ids, is_audio, chunk_of, labels, lang, delay, name, id, n_text, overflow, n_flush, K, rounds
  + soft_pos/soft_alt/soft_w (EOT soft target), activity (K,R) float, lanes (L,) int, t0.
창: 길이 window_s 범위, 시작은 아무도 말하지 않는 시각(부분 발화 방지), 끝은 잘릴 수 있다(잘린 episode 는 토큰을 창 안 것만 두고 EOT 는 mask).
전사 없는 음성(quarantine·ASR 불일치·미정렬·결손 조각): untranscribed="mask"(기본) 이면 창은 유지하고 그 발화가 덮는 청크(+δ) 의 payload·NEXT label 을 전부 -100 으로 둔다(아무 학습 신호 없음; 활동 타깃은 유지).
  untranscribed="skip" 이면 이전처럼 그 창을 제외한다. 건수는 stats 에.
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
                 max_items: Optional[int] = None, allow_unaligned: bool = False, cache_convs: int = 4, min_text_tokens: int = 8, untranscribed: str = "mask", mono_cache_dir: Optional[str] = None,
                 start_mode: str = "grid", random_frac: float = 0.5):
        self.tok = tok; self.sp_ids = add_phase2_specials(tok); self.sp = lane_specials_of(self.sp_ids, R); self.R = R; self.delays = tuple(delays); self.delay_onset = delay_onset
        self.policy = policy; self.M = max_per_chunk; self.mix_kw = mix_kw or {}; self.allow_unaligned = allow_unaligned; self.cache_convs = cache_convs; self.min_text_tokens = min_text_tokens; self.untranscribed = untranscribed; assert untranscribed in ("mask", "skip"); self.mono_cache_dir = mono_cache_dir
        assert start_mode in ("grid", "mixed"); self.start_mode = start_mode; self.random_frac = random_frac
        self.audio_pad = tok.convert_tokens_to_ids("<|audio_pad|>"); self._pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        self.dlgs: Dict[str, Dialogue] = {}; self.stats = dict(dialogues=0, windows=0, skipped_untranscribed=0, skipped_unaligned=0, skipped_sparse=0, no_start=0, windows_with_mask=0, random_start=0)
        aligned, failed = load_align_parts(align_dir) if align_dir else ({}, set())
        for path in dialogues:
            for line in open(path, encoding="utf-8"):
                d = Dialogue.from_json(line); toks = aligned.get(d.conv_id, {})
                for u in d.utterances:
                    if u.utt_id in toks: u.tokens = toks[u.utt_id]
                self.dlgs[d.conv_id] = d; self.stats["dialogues"] += 1
        self.items: List[Tuple[str, float, float]] = []; rng = random.Random(seed); self._seed = seed
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
        # 전사 없는 음성: quarantine(flags) · 토큰도 텍스트도 없음 · 텍스트는 있는데 미정렬(allow_unaligned 가 아니면). 보정(refine)으로 분할된 조각은 text="" 이지만 tokens 가 있어 정상
        bad = self._bad_spans(d)
        out = []; t = 0.0; lo, hi = window_s
        while t + lo <= d.duration_s:
            if not silent(t): self.stats["no_start"] += 1; t += hop_s; continue
            L = min(rng.uniform(lo, hi), d.duration_s - t); t1 = t + L
            hit = any(s < t1 and e > t for s, e in bad)
            if hit and self.untranscribed == "skip": self.stats["skipped_untranscribed"] += 1; t += hop_s; continue
            if hit: self.stats["windows_with_mask"] += 1
            n_tok = sum(len([1 for _, tt in (u.tokens or []) if t <= tt <= t1]) for u in d.utterances if u.end > t and u.start < t1)
            if n_tok < self.min_text_tokens: self.stats["skipped_sparse"] += 1; t += hop_s; continue      # 거의 무음인 창 제외
            out.append((d.conv_id, round(t, 3), round(L, 3))); self.stats["windows"] += 1; t += hop_s
        if self.start_mode == "mixed" and out and d.duration_s > lo:
            # D1b: 창 시작을 무음 격자에만 두면 첫 ONSET 이 창 앞쪽에 몰려(청크 ≤5 가 1/3) 모델이 위치 prior 로 시작을 낸다. 임의 시점(발화 중간 포함) 시작 창을 random_frac 비율로 더한다.
            # 발화 중간 시작이면 그 발화는 창 0 초에서 시작하는 episode 가 되고(ONSET 청크 0 = "이미 말하고 있다"), 창 이전 토큰은 버린다(window_episodes).
            n_rand = int(round(len(out) * self.random_frac / max(1e-6, 1.0 - self.random_frac))); tries = 0
            rr = random.Random(f"{self._seed}:{d.conv_id}:random-start")          # 격자 창의 난수열을 건드리지 않도록 별도 스트림(grid 창 목록이 mode 와 무관하게 같게)
            while n_rand > 0 and tries < 20 * (n_rand + 1):
                tries += 1; t = rr.uniform(0.0, d.duration_s - lo); L = min(rr.uniform(lo, hi), d.duration_s - t); t1 = t + L
                hit = any(s < t1 and e > t for s, e in bad)
                if hit and self.untranscribed == "skip": continue
                n_tok = sum(len([1 for _, tt in (u.tokens or []) if t <= tt <= t1]) for u in d.utterances if u.end > t and u.start < t1)
                if n_tok < self.min_text_tokens: continue
                if hit: self.stats["windows_with_mask"] += 1
                out.append((d.conv_id, round(t, 3), round(L, 3))); self.stats["windows"] += 1; self.stats["random_start"] += 1; n_rand -= 1
        return out

    def _bad_spans(self, d: Dialogue):
        """전사 없는 음성 구간: quarantine(flags) · 토큰도 텍스트도 없음 · 텍스트는 있는데 미정렬(allow_unaligned 가 아니면) · 결손 조각. 보정으로 분할된 조각은 text="" 이지만 tokens 가 있어 정상."""
        bad = [(u.start, u.end) for u in d.utterances if u.flags or (not u.tokens and (not u.text or not self.allow_unaligned))]
        missing = set(d.meta.get("missing", [])); bad += [(u.start, u.end) for i, u in enumerate(d.utterances, 1) if i in missing]
        return bad
    def mask_chunks(self, d: Dialogue, t0: float, L: float, K: int, delay: int) -> set:
        """창 [t0,t0+L) 에서 전사 없는 구간이 덮는 청크 집합(lexical 이 δ 만큼 늦게 놓이므로 끝은 +(δ+1) 청크, EOT 후보 +0.24 s 도 포함)."""
        out = set(); tail = (delay + 1) * CHUNK_S + 0.24
        for s, e in self._bad_spans(d):
            a, b = s - t0, e - t0
            if b <= 0 or a >= L: continue
            for k in range(max(0, int(a / CHUNK_S)), min(K, int(math.ceil((b + tail) / CHUNK_S)))): out.add(k)
        return out

    def __len__(self): return len(self.items)
    def prefix(self, lang: str, delay: int) -> List[int]:
        return self._pre + self.tok(f"language {lang}<asr_text>", add_special_tokens=False)["input_ids"] + [self.sp_ids[f"<DELAY_{delay}>"]]

    def mono(self, cid: str) -> np.ndarray:
        """대화의 mono 혼합. mono_cache_dir 가 있으면 <dir>/<corpus>/<conv>.npy(float16) 에 한 번 저장하고 이후 mmap 으로 연다(창마다 채널 전체를 섞는 비용 제거).
        캐시는 experiments/p2_build_mono.py 로 미리 만들어 둘 수 있다(같은 build_mono_cache 사용)."""
        if cid in self._mono: return self._mono[cid]
        d = self.dlgs[cid]; x = None
        if self.mono_cache_dir:
            f = build_mono_cache(d, self.mono_cache_dir, self.mix_kw)
            try: x = np.load(f, mmap_mode="r")
            except Exception: x = None
        if x is None: x, _ = mix_dialogue(d, **self.mix_kw)
        self._mono[cid] = x; self._order.append(cid)
        while len(self._order) > self.cache_convs: self._mono.pop(self._order.pop(0), None)
        return x

    def window_episodes(self, d: Dialogue, t0: float, L: float) -> Tuple[List[Episode], dict]:
        """창 [t0, t0+L) 의 episode(시각은 창 기준). 창 끝을 넘는 episode 는 토큰을 잘라내고 EOT 를 mask 한다."""
        t1 = t0 + L; utts = []
        for u in d.utterances:
            if u.end <= t0 or u.start >= t1: continue
            toks = [(tid, t - t0) for tid, t in (u.tokens or []) if t0 <= t <= t1]          # 창 이전 토큰은 버린다(발화 중간 시작 창)
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
        mk = self.mask_chunks(d, t0, L, K, delay) if self.untranscribed == "mask" else set()
        f = flatten(chunks, K, self.audio_pad, self.sp.empty_audio, self.prefix(d.lang, delay), mask_chunks=mk); act = lane_activity(eps, K, self.R)
        return dict(f, K=K, activity=act, stats=st, info=info, cid=cid, t0=t0, L=L, lang=d.lang, corpus=d.corpus)

    def __getitem__(self, i):
        delay = random.choice(self.delays); s = self.sequence(i, delay)
        wav = np.asarray(crop_audio(self.mono(s["cid"]), s["t0"], s["t0"] + s["L"]), dtype=np.float32)
        return dict(wav=torch.from_numpy(np.ascontiguousarray(wav)), ids=torch.tensor(s["ids"]), is_audio=torch.tensor(s["is_audio"]), chunk_of=torch.tensor(s["chunk_of"]), labels=torch.tensor(s["labels"]),
                    soft_pos=torch.tensor(s["soft_pos"], dtype=torch.long), soft_alt=torch.tensor(s["soft_alt"], dtype=torch.long), soft_w=torch.tensor(s["soft_w"], dtype=torch.float),
                    activity=torch.tensor(s["activity"], dtype=torch.float), lanes=torch.tensor(s["lanes"]), lang=s["lang"], delay=delay, name=s["corpus"], id=f"{s['cid']}@{s['t0']}",
                    n_text=s["stats"].text, overflow=s["stats"].overflow, n_flush=sum(1 for c in s["chunk_of"] if c < 0) - len(self._pre), K=s["K"], rounds=0, t0=s["t0"])

    # ── 길이 추정(동적 배치용)
    def est_lens(self) -> np.ndarray:
        """창별 시퀀스 길이(토큰) 추정: prefix + 청크당 2(audio_pad·NEXT_AUDIO) + 창 안 텍스트 토큰 + 발화당 3(lane 태그·ONSET·EOT 근사). serialize 를 돌리지 않아 129 k 창도 수 초."""
        if getattr(self, "_est", None) is not None and len(self._est) == len(self.items): return self._est
        arr = {}
        for cid, d in self.dlgs.items():
            us = d.utterances; arr[cid] = (np.array([u.start for u in us]), np.array([u.end for u in us]), np.array([len(u.tokens or []) for u in us]))
        pre = len(self._pre) + 6; est = np.zeros(len(self.items), dtype=np.int64)
        for i, (cid, t0, L) in enumerate(self.items):
            st, en, nt = arr[cid]; m = (st < t0 + L) & (en > t0)
            est[i] = pre + 2 * int(round(L / CHUNK_S)) + int(nt[m].sum()) + 3 * int(m.sum())
        self._est = est; return est

def mono_cache_path(mono_dir: str, d: Dialogue) -> str:
    return os.path.join(mono_dir, d.corpus, d.conv_id.replace("/", "_").replace(":", "_") + ".npy")

def build_mono_cache(d: Dialogue, mono_dir: str, mix_kw: Optional[dict] = None) -> str:
    """대화의 mono 혼합을 <mono_dir>/<corpus>/<conv>.npy(float16) 로 저장(있으면 그대로) → 경로. 원자적 저장(tmp → replace)이라 여러 프로세스가 동시에 만들어도 안전."""
    f = mono_cache_path(mono_dir, d)
    if os.path.exists(f): return f
    y, _ = mix_dialogue(d, **(mix_kw or {})); os.makedirs(os.path.dirname(f), exist_ok=True)
    tmp = f[:-4] + f".{os.getpid()}.tmp.npy"; np.save(tmp, y.astype(np.float16)); os.replace(tmp, f); return f

class TokenBudgetSampler(torch.utils.data.Sampler):
    """동적 길이 배치: 창을 추정 길이순으로 정렬해 배치의 (창 수 × 최장 길이) ≤ max_tokens 가 되도록 연속 구간을 묶는다(패딩 후 실제 토큰 수 기준이라 GPU 메모리가 배치마다 비슷).
    짧은 창(20 s)은 많이, 긴 창(40 s)은 적게 → 고정 bs 보다 step 당 토큰이 균일하고 GPU 활용이 높다. 배치 순서는 epoch 마다 셔플(seed+epoch), DDP 는 BucketBatchSampler 와 같은 규약."""
    def __init__(self, ds: DialogueWindowDataset, max_tokens: int, max_bs: int = 64, seed: int = 0, rank: int = 0, world: int = 1, drop_last: bool = False):
        est = ds.est_lens(); order = sorted(range(len(ds)), key=lambda i: (int(est[i]), i)); self.batches = []; cur = []
        for i in order:                                          # 오름차순이라 est[i] 가 배치의 최장 길이 = 패딩 후 열 길이
            if cur and ((len(cur) + 1) * int(est[i]) > max_tokens or len(cur) >= max_bs): self.batches.append(cur); cur = []
            cur.append(i)
        if cur and not (drop_last and self.batches and len(cur) * int(est[cur[-1]]) < max_tokens // 2): self.batches.append(cur)   # 마지막 자투리는 예산 절반 미만이면(drop_last) 버림
        self.seed, self.rank, self.world, self.epoch = seed, rank, world, 0; self.n = len(self.batches) // world     # rank 당 배치 수(균등, 나머지 버림). 배치 < world 면 0 → 호출자가 그 코퍼스를 제외해야 한다
        self.max_tokens = max_tokens; self.tokens = [len(b) * int(est[b[-1]]) for b in self.batches]
    def set_epoch(self, e: int): self.epoch = e
    def __iter__(self):
        b = list(self.batches); random.Random(f"{self.seed}:{self.epoch}").shuffle(b); return iter(b[self.rank::self.world][: self.n])
    def __len__(self): return self.n
    def describe(self) -> dict:
        bs = [len(b) for b in self.batches]
        return dict(batches=len(self.batches), max_tokens=self.max_tokens, bs_min=min(bs) if bs else 0, bs_mean=round(sum(bs) / max(1, len(bs)), 1), bs_max=max(bs) if bs else 0, tokens_mean=round(sum(self.tokens) / max(1, len(self.tokens))), fill=round(sum(self.tokens) / max(1, len(self.tokens)) / self.max_tokens, 3))

def collate_dialogue(batch):
    """collate_streams 와 같은 패딩 + soft(배치 평탄화: soft_b, soft_pos, soft_alt, soft_w) + activity (B,K,R) 와 activity_mask (B,K)."""
    from ..uslm.mono_data import collate_streams
    B = len(batch); out = collate_streams([{k: v for k, v in b.items() if k not in ("soft_pos", "soft_alt", "soft_w", "activity", "lanes", "t0")} for b in batch])
    out["soft_b"] = torch.cat([torch.full_like(b["soft_pos"], i) for i, b in enumerate(batch)]); out["soft_pos"] = torch.cat([b["soft_pos"] for b in batch])
    out["soft_alt"] = torch.cat([b["soft_alt"] for b in batch]); out["soft_w"] = torch.cat([b["soft_w"] for b in batch])
    L = out["labels"].shape[1]; la = torch.full((B, L), -100, dtype=torch.long); sw = torch.ones(B, L)          # 모델 forward 용 (B,L) 형태
    for i, b in enumerate(batch):
        if b["soft_pos"].numel(): la[i, b["soft_pos"]] = b["soft_alt"]; sw[i, b["soft_pos"]] = b["soft_w"]
    out["labels_alt"] = la; out["soft_w_full"] = sw
    K = max(b["activity"].shape[0] for b in batch); R = batch[0]["activity"].shape[1]
    act = torch.zeros(B, K, R); am = torch.zeros(B, K)
    for i, b in enumerate(batch): k = b["activity"].shape[0]; act[i, :k] = b["activity"]; am[i, :k] = 1
    out["activity"] = act; out["activity_mask"] = am; return out

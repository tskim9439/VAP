"""U1 — interleaved streaming ASR 학습 데이터.

정렬(jsonl, `experiments/u0_align.py`) + 캐시 Nemotron 특징(12.5 Hz = 80 ms chunk 1 프레임)에서 고정 길이 창을 잘라
`vapasr.data.interleave.build_interleaved` 로 target 시퀀스를 만든다.

시퀀스 규약 (모든 것이 하나의 스트림):
  [prefix: <|im_start|>system\\n<|im_end|>\\n<|im_start|>assistant\\nlanguage {Lang}<asr_text><DELAY_d>]
  [AUDIO_0] (<SPK_x>) tok … <NEXT_AUDIO> [AUDIO_1] … <NEXT_AUDIO>
  AUDIO_k 위치는 <|audio_pad|> 자리표시자 → 모델이 두 화자 chunk 임베딩(merge)으로 대체. 손실은 audio 위치와 prefix 를 제외한 모든 위치.
창 시작점은 양 화자가 모두 침묵인 시각(부분 발화 방지) 중에서 고른다. 창 안에서 δ(지연 프레임)는 delays 에서 무작위로 뽑고 <DELAY_d> 로 조건화한다.
"""
import os, json, glob, math, random, bisect
from typing import List, Dict, Tuple, Optional
import numpy as np, torch
from torch.utils.data import Dataset
from ..data.interleave import build_interleaved, Specials
from ..probe.data import FeatureIndex
from .data import MAN, FEAT, LANG, split_ids

CHUNK_S = 0.08
SPECIAL_TOKENS = ["<NEXT_AUDIO>", "<EMPTY_AUDIO>", "<SPK_A>", "<SPK_B>"] + [f"<DELAY_{d}>" for d in range(1, 9)]

def add_specials(tok) -> Dict[str, int]:
    """tokenizer 에 특수 토큰 추가(이미 있으면 그대로) → {name: id}. Qwen3-ASR 임베딩 행렬(151936)에 여유 행이 있어 resize 불필요."""
    tok.add_tokens(SPECIAL_TOKENS, special_tokens=True)
    return {t: tok.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS}

def specials_of(ids: Dict[str, int]) -> Specials:
    return Specials(next_audio=ids["<NEXT_AUDIO>"], empty_audio=ids["<EMPTY_AUDIO>"], spk=(ids["<SPK_A>"], ids["<SPK_B>"]))

def _read_jsonl(path: str) -> List[dict]:
    """손상 줄(중단된 워커가 남긴 NUL 블록 등)은 건너뛴다."""
    out = []
    for l in open(path, errors="replace"):
        if not l.strip() or "\x00" in l: continue
        try: out.append(json.loads(l))
        except json.JSONDecodeError: continue
    return out

def bad_utterance(u: dict, max_same: int = 8, max_rate: float = 25.0) -> Optional[str]:
    """정렬 실패 발화 판정. aligner 가 긴 발화에서 항목을 못 만들면 BPE 토큰 수십~수백 개가 같은 종료 시각을 갖고,
    그 뭉치가 chunk 예산을 넘겨 수 초짜리 backlog 를 만든다(otoSpeech 최악 132 토큰/동일 시각 → backlog 181 chunk = 14.5 s)."""
    ts = [t["end_time"] for t in u["tokens"]]
    if not ts: return "empty"
    from collections import Counter
    if max(Counter(ts).values()) > max_same: return "same_time"
    if len(ts) / max(0.1, u["end"] - u["start"]) > max_rate: return "rate"
    return None

class AlignedConv:
    """한 대화의 정렬: 화자별 [(token_id, end_time)] + 발화 구간 + 침묵 시작점 후보."""
    def __init__(self, path: str, qc: bool = True, max_same: int = 8, max_rate: float = 25.0):
        self.utts: List[dict] = _read_jsonl(path)
        self.dropped = [u for u in self.utts if qc and bad_utterance(u, max_same, max_rate)] if qc else []
        keep = [u for u in self.utts if u not in self.dropped] if self.dropped else self.utts
        self.streams: List[List[Tuple[int, float]]] = [[], []]
        for u in keep:
            for t in u["tokens"]: self.streams[u["speaker"]].append((int(t["id"]), float(t["end_time"])))
        for s in self.streams: s.sort(key=lambda x: x[1])
        self.duration = max([u["end"] for u in self.utts], default=0.0)
        self._ends = [t for _, t in self.streams[0]], [t for _, t in self.streams[1]]

    def silent_starts(self, window_s: float, grid_s: float = 1.0, margin_s: float = 0.5) -> List[float]:
        """양 화자 모두 [t-margin, t+margin] 에 발화가 없는 격자 시각 t (창이 대화 안에 들어가는 것만)."""
        merged: List[List[float]] = []                                   # 두 화자 발화 구간의 합집합(겹침 병합)
        for s, e in sorted((u["start"] - margin_s, u["end"] + margin_s) for u in self.utts):
            if merged and s <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], e)
            else: merged.append([s, e])
        starts = [m[0] for m in merged]; out = []; t = 0.0
        while t + window_s <= self.duration:
            i = bisect.bisect_right(starts, t) - 1
            if not (i >= 0 and merged[i][1] > t): out.append(t)
            t += grid_s
        return out

    def tokens_in(self, t0: float, t1: float) -> List[List[Tuple[int, float]]]:
        """end_time ∈ [t0, t1) 인 토큰(창 상대 시각)."""
        return [[(tid, te - t0) for tid, te in s[bisect.bisect_left(e, t0): bisect.bisect_left(e, t1)]] for s, e in zip(self.streams, self._ends)]

def build_sequence(chunks, K: int, prefix: List[int], audio_pad: int, per_chunk_audio: int = 1):
    """build_interleaved 출력 → (ids, is_audio, chunk_of, audio_spk).
    per_chunk_audio=1: chunk 당 오디오 토큰 1 개(두 화자 merge). =2: 화자별 1 개씩(A, B 순) — merge 로 인한
    분포 이동·겹침 손상을 피하는 대신 오디오 위치가 2 배가 된다. audio_spk: 오디오 위치의 화자(-1=merge/비오디오)."""
    ids = list(prefix); is_audio = [False] * len(prefix); chunk_of = [-1] * len(prefix); audio_spk = [-1] * len(prefix)
    for k, emits in chunks:
        if k >= K: break
        for s in ([-1] if per_chunk_audio == 1 else [0, 1]):
            ids.append(audio_pad); is_audio.append(True); chunk_of.append(k); audio_spk.append(s)
        ids += emits; is_audio += [False] * len(emits); chunk_of += [-1] * len(emits); audio_spk += [-1] * len(emits)
    return ids, is_audio, chunk_of, audio_spk

class InterleavedWindowDataset(Dataset):
    def __init__(self, manifests: List[str], tok, encoder: str = "nemotron-c0", split: str = "train", window_s: float = 30.0, delays=(2, 3, 4, 6),
                 max_per_chunk: int = 4, val_frac: float = 0.08, windows_per_conv: Optional[int] = None, max_windows: Optional[int] = None,
                 seed: int = 0, align_root: Optional[str] = None, per_chunk_audio: int = 1):
        self.tok = tok; self.sp_ids = add_specials(tok); self.sp = specials_of(self.sp_ids); self.window_s = window_s; self.K = int(round(window_s / CHUNK_S))
        self.delays = tuple(delays); self.M = max_per_chunk; self.per_chunk_audio = per_chunk_audio; self.audio_pad = tok.convert_tokens_to_ids("<|audio_pad|>")
        self._pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        self.items: List[Tuple[str, str, float]] = []; self.convs: Dict[Tuple[str, str], AlignedConv] = {}; self.feat: Dict[str, FeatureIndex] = {}
        rng = random.Random(seed); align_root = align_root or os.path.join(MAN, "align")
        for name in manifests:
            fi = FeatureIndex(FEAT, encoder, name); self.feat[name] = fi; assert abs(fi.frame_hz - 1 / CHUNK_S) < 1e-6, f"{encoder} frame_hz {fi.frame_hz} != 12.5"
            tr, va = split_ids(name, list(fi.rows.keys()), val_frac); ids = tr if split == "train" else va
            for cid in ids:
                p = os.path.join(align_root, name, cid + ".jsonl")
                if not os.path.exists(p): continue
                ac = AlignedConv(p); starts = ac.silent_starts(window_s)
                if ac.dropped: starts = [t for t in starts if not any(u["start"] < t + window_s and u["end"] > t for u in ac.dropped)]   # 불량 발화가 걸친 창 제외
                if not starts: continue
                if windows_per_conv and len(starts) > windows_per_conv: starts = rng.sample(starts, windows_per_conv)
                self.convs[(name, cid)] = ac; self.items += [(name, cid, s) for s in starts]
        rng.shuffle(self.items)
        if max_windows: self.items = self.items[:max_windows]

    def __len__(self): return len(self.items)

    def prefix(self, lang: str, delay: int) -> List[int]:
        return self._pre + self.tok(f"language {lang}<asr_text>", add_special_tokens=False)["input_ids"] + [self.sp_ids[f"<DELAY_{delay}>"]]

    def window(self, name: str, cid: str, t0: float, delay: int):
        """(feats (2,K,D) float32, streams(창 상대), chunks) — 평가용 공개 API."""
        fi = self.feat[name]; ac = self.convs[(name, cid)]; k0 = int(round(t0 / CHUNK_S))
        F = np.load(fi.rows[cid]["npy"], mmap_mode="r"); f = np.asarray(F[:, k0: k0 + self.K]).astype(np.float32)
        if f.shape[1] < self.K: f = np.concatenate([f, np.zeros((2, self.K - f.shape[1], f.shape[2]), np.float32)], 1)
        streams = ac.tokens_in(t0, t0 + self.window_s)
        chunks, st = build_interleaved(streams, self.window_s, self.sp, chunk_s=CHUNK_S, delay_frames=delay, max_per_chunk=self.M)
        return f, streams, chunks, st

    def __getitem__(self, i):
        name, cid, t0 = self.items[i]; delay = random.choice(self.delays); lang = LANG[name]
        f, streams, chunks, st = self.window(name, cid, t0, delay)
        pre = self.prefix(lang, delay); ids, is_audio, chunk_of, audio_spk = build_sequence(chunks, self.K, pre, self.audio_pad, self.per_chunk_audio)
        lab = [(-100 if (a or j < len(pre)) else t) for j, (t, a) in enumerate(zip(ids, is_audio))]
        return dict(feats=torch.from_numpy(f), ids=torch.tensor(ids), is_audio=torch.tensor(is_audio), chunk_of=torch.tensor(chunk_of), audio_spk=torch.tensor(audio_spk), labels=torch.tensor(lab),
                    lang=lang, delay=delay, manifest=name, conv=cid, t0=t0, n_text=st.tokens, overflow=st.overflow_tokens)

def collate_windows(batch):
    B = len(batch); L = max(len(b["ids"]) for b in batch); K, D = batch[0]["feats"].shape[1:]
    feats = torch.stack([b["feats"] for b in batch]); ids = torch.zeros(B, L, dtype=torch.long); is_audio = torch.zeros(B, L, dtype=torch.bool)
    chunk_of = torch.full((B, L), -1, dtype=torch.long); audio_spk = torch.full((B, L), -1, dtype=torch.long); labels = torch.full((B, L), -100, dtype=torch.long); mask = torch.zeros(B, L, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"]); ids[i, :n] = b["ids"]; is_audio[i, :n] = b["is_audio"]; chunk_of[i, :n] = b["chunk_of"]; audio_spk[i, :n] = b["audio_spk"]; labels[i, :n] = b["labels"]; mask[i, :n] = 1
    return dict(feats=feats, ids=ids, is_audio=is_audio, chunk_of=chunk_of, audio_spk=audio_spk, labels=labels, mask=mask, meta=[{k: b[k] for k in ("lang", "delay", "manifest", "conv", "t0", "n_text", "overflow")} for b in batch])

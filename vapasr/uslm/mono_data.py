"""Stage 1 mono 데이터셋 — streams.jsonl 스트림 하나 = 학습 샘플 하나 (plans/stage1-mono-pilot.md §4).

입력: 특징 캐시 index.jsonl(experiments/s1_extract_features.py, (1,T',D) @12.5 Hz, mode·subset 포함)
    + 정렬 align/<manifest>/<id>.jsonl(experiments/s1_align.py, speaker=0)
시퀀스: [prefix … <DELAY_d>] [AUDIO_0] tok… <NEXT_AUDIO> [AUDIO_1] … (끝) tok… <EMPTY_AUDIO>.  화자 태그 없음(add_spk_tags=False).
chunk k = 특징 프레임 k (12.5 Hz = 80 ms). 스트림 길이가 제각각이라 길이 버킷 배치를 쓴다.
"""
import os, json, math, random
from typing import List, Dict, Tuple, Optional
import numpy as np, torch
from torch.utils.data import Dataset, Sampler
from ..data.interleave import build_interleaved
from ..probe.data import FeatureIndex
from .interleave_data import add_specials, specials_of, build_sequence, bad_utterance, _read_jsonl, CHUNK_S

MAN = os.environ.get("MXC_DATA_MANIFEST_DIR", os.environ.get("DATA_MANIFEST_DIR", "/tmp")); FEAT = os.environ.get("MXC_DATA_FEATURE_CACHE_DIR", os.environ.get("DATA_FEATURE_CACHE_DIR", "/tmp"))
LANG_OF = {"librispeech": "English", "ls": "English", "kspon": "Korean", "ks": "Korean"}
def lang_of(name: str) -> str: return LANG_OF[name.split("-")[0]]

class MonoStreamDataset(Dataset):
    def __init__(self, manifests: List[str], tok, encoder: str = "nemotron-c0", mode: str = "stream", subsets: Optional[List[str]] = None,
                 delays=(2, 3, 4, 6), max_per_chunk: int = 4, max_items: Optional[int] = None, seed: int = 0, qc: bool = True, align_root: Optional[str] = None):
        self.tok = tok; self.sp_ids = add_specials(tok); self.sp = specials_of(self.sp_ids); self.delays = tuple(delays); self.M = max_per_chunk
        self.audio_pad = tok.convert_tokens_to_ids("<|audio_pad|>"); self._pre = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)["input_ids"]
        align_root = align_root or os.path.join(MAN, "align"); self.items: List[dict] = []; self.dropped = 0; self.no_align = 0
        for name in manifests:
            fi = FeatureIndex(FEAT, encoder, name); assert abs(fi.frame_hz - 1 / CHUNK_S) < 1e-6, f"{encoder} frame_hz {fi.frame_hz} != 12.5"
            for sid, row in fi.rows.items():
                if row.get("mode", "stream") != mode or (subsets and row.get("subset") not in subsets): continue
                p = os.path.join(align_root, name, sid + ".jsonl")
                if not os.path.exists(p): self.no_align += 1; continue
                utts = _read_jsonl(p); bad = [u for u in utts if qc and bad_utterance(u)]
                if bad: self.dropped += 1; continue                     # 불량 발화(동일 종료시각 뭉침)가 있는 스트림은 통째로 제외
                toks = sorted(((t["id"], t["end_time"]) for u in utts for t in u["tokens"]), key=lambda x: x[1])
                text = " ".join(u["text"] for u in utts)
                self.items.append(dict(name=name, id=sid, K=int(row["frames"]), npy=row["npy"], lang=lang_of(name), subset=row.get("subset"), tokens=toks, text=text))
        random.Random(seed).shuffle(self.items)
        if max_items: self.items = self.items[:max_items]

    def __len__(self): return len(self.items)
    def prefix(self, lang: str, delay: int) -> List[int]:
        return self._pre + self.tok(f"language {lang}<asr_text>", add_special_tokens=False)["input_ids"] + [self.sp_ids[f"<DELAY_{delay}>"]]

    def stream(self, i: int, delay: int):
        """(feats (1,K,D) float32, ref tokens [(id,end)], chunks, stats, item) — 평가용 공개 API."""
        it = self.items[i]; f = np.asarray(np.load(it["npy"], mmap_mode="r")[:, : it["K"]]).astype(np.float32)   # (1,K,D)
        chunks, st = build_interleaved([it["tokens"]], (it["K"] - 0.5) * CHUNK_S, self.sp, chunk_s=CHUNK_S, delay_frames=delay, max_per_chunk=self.M, add_spk_tags=False)
        return f, it["tokens"], chunks, st, it

    def __getitem__(self, i):
        delay = random.choice(self.delays); f, toks, chunks, st, it = self.stream(i, delay); pre = self.prefix(it["lang"], delay)
        ids, is_audio, chunk_of, _ = build_sequence(chunks, it["K"], pre, self.audio_pad, 1)
        lab = [(-100 if (a or j < len(pre)) else t) for j, (t, a) in enumerate(zip(ids, is_audio))]
        return dict(feats=torch.from_numpy(f), ids=torch.tensor(ids), is_audio=torch.tensor(is_audio), chunk_of=torch.tensor(chunk_of), labels=torch.tensor(lab),
                    lang=it["lang"], delay=delay, name=it["name"], id=it["id"], n_text=st.tokens, overflow=st.overflow_tokens, K=it["K"])

class BucketBatchSampler(Sampler):
    """길이(K) 순으로 정렬해 비슷한 길이끼리 배치 → padding 낭비를 줄인다. 배치 순서는 매 epoch 셔플."""
    def __init__(self, ds: MonoStreamDataset, bs: int, seed: int = 0, drop_last: bool = True):
        order = sorted(range(len(ds)), key=lambda i: ds.items[i]["K"]); self.batches = [order[i: i + bs] for i in range(0, len(order), bs)]
        if drop_last and self.batches and len(self.batches[-1]) < bs: self.batches = self.batches[:-1]
        self.rng = random.Random(seed)
    def __iter__(self):
        b = list(self.batches); self.rng.shuffle(b); return iter(b)
    def __len__(self): return len(self.batches)

def collate_streams(batch):
    B = len(batch); K = max(b["feats"].shape[1] for b in batch); D = batch[0]["feats"].shape[2]; L = max(len(b["ids"]) for b in batch)
    feats = torch.zeros(B, 1, K, D); ids = torch.zeros(B, L, dtype=torch.long); is_audio = torch.zeros(B, L, dtype=torch.bool)
    chunk_of = torch.full((B, L), -1, dtype=torch.long); labels = torch.full((B, L), -100, dtype=torch.long); mask = torch.zeros(B, L, dtype=torch.long)
    for i, b in enumerate(batch):
        k = b["feats"].shape[1]; n = len(b["ids"]); feats[i, :, :k] = b["feats"]
        ids[i, :n] = b["ids"]; is_audio[i, :n] = b["is_audio"]; chunk_of[i, :n] = b["chunk_of"]; labels[i, :n] = b["labels"]; mask[i, :n] = 1
    return dict(feats=feats, ids=ids, is_audio=is_audio, chunk_of=chunk_of, labels=labels, mask=mask,
                lang=[b["lang"] for b in batch], delay=[b["delay"] for b in batch], n_text=sum(b["n_text"] for b in batch), overflow=sum(b["overflow"] for b in batch))

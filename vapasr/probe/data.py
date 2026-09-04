"""캐시된 frozen encoder 특징 + VAD(50 Hz npz) → probe 학습 창.

- 특징: /data3/tskim/features/<encoder>/<manifest>/<id>.npy (2, T', D) fp16, index.jsonl 에 frame_hz.
- 목표 frame_hz 가 특징 고유율과 다르면 시간축 재표본 (정수배 다운은 평균 풀링, 그 외 nearest).
- VAD 는 시간 구간 any 로 목표 율에 맞춘다. VAP 256-class 라벨은 원 VAP ObjectiveVAP (bins 는 율에 따라 자동).
"""
import os, json, math, random
from typing import List, Dict, Optional, Tuple
import numpy as np, torch
from torch.utils.data import Dataset
from ..data.targets import vap_labels

class FeatureIndex:
    def __init__(self, feature_root: str, encoder: str, manifest_name: str):
        self.dir = os.path.join(feature_root, encoder, manifest_name); self.rows: Dict[str, dict] = {}
        p = os.path.join(self.dir, "index.jsonl")
        if os.path.exists(p):
            for l in open(p):
                r = json.loads(l); self.rows[r["id"]] = r
        self.frame_hz = next(iter(self.rows.values()))["frame_hz"] if self.rows else None
        self.dim = next(iter(self.rows.values()))["dim"] if self.rows else None

def resample_vad(va50: np.ndarray, hz: float, n_out: int, src_hz: float = 50.0) -> np.ndarray:
    """(T50, 2) → (n_out, 2): 출력 프레임 i 가 덮는 [i/hz, (i+1)/hz) 구간의 any."""
    if abs(hz - src_hz) < 1e-6: out = va50[:n_out]
    else:
        edges = (np.arange(n_out + 1) / hz * src_hz).round().astype(int).clip(0, len(va50))
        out = np.stack([va50[a:b].max(0) if b > a else (va50[min(a, len(va50) - 1)] if len(va50) else np.zeros(2)) for a, b in zip(edges[:-1], edges[1:])]) if n_out else np.zeros((0, 2))
    if len(out) < n_out: out = np.concatenate([out, np.zeros((n_out - len(out), 2), dtype=out.dtype)])
    return out.astype(np.float32)

def resample_feats(f: np.ndarray, src_hz: float, dst_hz: float) -> np.ndarray:
    """(2, T, D) 시간축 재표본. 정수배 다운샘플은 평균, 그 외 nearest-time."""
    if abs(src_hz - dst_hz) < 1e-6: return f
    ratio = src_hz / dst_hz
    if abs(ratio - round(ratio)) < 1e-6 and ratio > 1:
        k = int(round(ratio)); T = (f.shape[1] // k) * k; return f[:, :T].reshape(f.shape[0], -1, k, f.shape[2]).mean(2)
    n_out = int(math.floor(f.shape[1] / ratio)); idx = np.minimum((np.arange(n_out) * ratio + ratio / 2).astype(int), f.shape[1] - 1)
    return f[:, idx]

class ProbeWindowDataset(Dataset):
    def __init__(self, feature_root: str, encoder: str, manifests: List[Tuple[str, str]], window_s: float = 20.0, hop_s: float = 10.0,
                 frame_hz: Optional[float] = None, exclude_flagged: bool = True, ids: Optional[set] = None, seed: int = 0, max_windows: Optional[int] = None):
        """manifests: [(manifest_dir, manifest_name), ...]. frame_hz None → 특징 고유율."""
        self.items = []; self.window_s, self.hop_s = window_s, hop_s; self.src_hz = None; self.dim = None
        for mdir, mname in manifests:
            fi = FeatureIndex(feature_root, encoder, mname)
            if not fi.rows: continue
            self.src_hz = self.src_hz or fi.frame_hz; self.dim = self.dim or fi.dim
            st = json.load(open(os.path.join(mdir, "stats.json"))) if os.path.exists(os.path.join(mdir, "stats.json")) else {}
            flagged = set(st.get("flagged", {}).keys()) if exclude_flagged else set()
            for l in open(os.path.join(mdir, "manifest.jsonl")):
                r = json.loads(l)
                if r["id"] not in fi.rows or r["id"] in flagged or (ids is not None and r["id"] not in ids) or r["duration"] < window_s: continue
                fr = fi.rows[r["id"]]
                for w in range(int((r["duration"] - window_s) // hop_s) + 1):
                    self.items.append((r["npz"], fr["npy"], w * hop_s))
        self.frame_hz = frame_hz or self.src_hz
        random.Random(seed).shuffle(self.items)
        if max_windows: self.items = self.items[:max_windows]

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        npz, npy, s0 = self.items[i]; F = np.load(npy, mmap_mode="r"); src = self.src_hz
        a, b = int(round(s0 * src)), int(round((s0 + self.window_s) * src)); f = np.asarray(F[:, a:b]).astype(np.float32)   # (2, T, D)
        if abs(self.frame_hz - src) > 1e-6: f = resample_feats(f, src, self.frame_hz)
        T = f.shape[1]; va50 = np.load(npz)["vad"].astype(np.float32); a50, b50 = int(round(s0 * 50)), int(round((s0 + self.window_s) * 50))
        va = resample_vad(va50[a50:b50], self.frame_hz, T); lab = vap_labels(va, self.frame_hz)
        return dict(feats=torch.from_numpy(f), vad=torch.from_numpy(va), vap_label=torch.from_numpy(lab))

def collate_probe(batch):
    T = min(b["feats"].shape[1] for b in batch)
    return dict(feats=torch.stack([b["feats"][:, :T] for b in batch]), vad=torch.stack([b["vad"][:T] for b in batch]), vap_label=torch.stack([b["vap_label"][:T] for b in batch]))

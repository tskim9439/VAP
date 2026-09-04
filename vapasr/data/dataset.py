"""학습용 윈도 데이터셋 — backbone 독립. manifest 디렉토리(npz + manifest.jsonl)와 원본 오디오에서 (audio, targets) 창을 만든다.

- 오디오는 로드 시 16 kHz (2, T) 로 읽는다 (aihub: stereo wav / otoSpeech: speaker_{1,2}_audio.wav).
- VAD 는 npz 의 50 Hz 를 기준으로, 요청 frame_hz(50 | 12.5)로 풀링해 VAP 라벨·τ bin 을 만든다.
- 이벤트는 (time, speaker, type) 원시 리스트로 넘긴다 — head 별 target 은 collate/loss 쪽에서 만든다.
"""
import os, json, random
from typing import List, Dict, Optional
import numpy as np, torch
from torch.utils.data import Dataset
from .targets import vad_frames, pool_vad, vap_labels, time_to_next_onset, EVENT_TYPES

SR = 16000

class WindowDataset(Dataset):
    def __init__(self, manifest_dirs: List[str], window_s: float = 20.0, hop_s: float = 10.0, frame_hz: float = 50.0,
                 audio_roots: Optional[Dict[str, str]] = None, exclude_flagged: bool = True, load_audio: bool = True, seed: int = 0):
        self.window_s, self.hop_s, self.frame_hz, self.load_audio = window_s, hop_s, frame_hz, load_audio
        self.audio_roots = audio_roots or {}; self.items = []
        for md in manifest_dirs:
            st = json.load(open(os.path.join(md, "stats.json"))) if os.path.exists(os.path.join(md, "stats.json")) else {}
            flagged = set(st.get("flagged", {}).keys()) if exclude_flagged else set()
            for line in open(os.path.join(md, "manifest.jsonl")):
                r = json.loads(line)
                if r["id"] in flagged or r["duration"] < window_s: continue
                n_win = int((r["duration"] - window_s) // hop_s) + 1
                for w in range(n_win):
                    self.items.append((r, w * hop_s))
        random.Random(seed).shuffle(self.items)

    def __len__(self): return len(self.items)

    def _audio_path(self, r):
        if r["source"] == "aihub":
            root = self.audio_roots.get("aihub"); return os.path.join(root, r["id"] + ".wav") if root else None
        if r["source"] == "otoSpeech":
            root = self.audio_roots.get("otoSpeech"); return os.path.join(root, r["id"].split("oto-")[1]) if root else None
        return None

    def _load_audio(self, r, s0):
        import soundfile as sf
        p = self._audio_path(r); n = int(self.window_s * SR)
        if p is None: return torch.zeros(2, n)
        if r["source"] == "aihub":
            x, sr = sf.read(p, start=int(s0 * sr_of(p)), frames=int(self.window_s * sr_of(p)), dtype="float32", always_2d=True)
            x = x.T
        else:
            chans = []
            for c in (1, 2):
                q = os.path.join(p, f"speaker_{c}_audio.wav"); sr = sr_of(q)
                y, _ = sf.read(q, start=int(s0 * sr), frames=int(self.window_s * sr), dtype="float32"); chans.append(y)
            x = np.stack(chans); sr = sr_of(os.path.join(p, "speaker_1_audio.wav"))
        if sr != SR:
            import soxr; x = np.stack([soxr.resample(ch, sr, SR) for ch in x])
        x = x[:, :n]; out = np.zeros((2, n), dtype=np.float32); out[:, :x.shape[1]] = x
        return torch.from_numpy(out)

    def __getitem__(self, i):
        r, s0 = self.items[i]; z = np.load(r["npz"]); va50 = z["vad"].astype(np.float32); fh0 = float(z["frame_hz"])
        f0, f1 = int(s0 * fh0), int((s0 + self.window_s) * fh0); va = va50[f0:f1]
        if abs(self.frame_hz - fh0) > 1e-6: va = pool_vad(va, int(round(fh0 / self.frame_hz)))
        lab = vap_labels(va, self.frame_hz); ttn = time_to_next_onset(va, self.frame_hz)
        ev = [(t - s0, int(s), int(k)) for t, s, k in z["events"].tolist() if s0 <= t < s0 + self.window_s]
        item = dict(id=r["id"], start=s0, vad=torch.from_numpy(va), vap_label=torch.from_numpy(lab),
                    tau_bin=torch.from_numpy(ttn["bin"]), censored=torch.from_numpy(ttn["censored"]), events=ev)
        if self.load_audio: item["audio"] = self._load_audio(r, s0)
        return item

_SR_CACHE = {}
def sr_of(p):
    if p not in _SR_CACHE:
        import soundfile as sf; _SR_CACHE[p] = sf.info(p).samplerate
    return _SR_CACHE[p]

def collate(batch):
    out = {k: torch.stack([b[k] for b in batch]) for k in ("vad", "vap_label", "tau_bin", "censored") if k in batch[0]}
    if "audio" in batch[0]: out["audio"] = torch.stack([b["audio"] for b in batch])
    out["events"] = [b["events"] for b in batch]; out["id"] = [b["id"] for b in batch]; out["start"] = [b["start"] for b in batch]
    return out

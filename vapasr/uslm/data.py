"""USLM 학습 데이터: 발화(화자·시각·텍스트) + 캐시된 encoder 특징 슬라이스.

- 발화 출처: aihub → 라벨 JSON(Conversation), otoSpeech → SRT(speech 범주). conv id 는 manifest id 와 동일.
- 분할: manifest 내 정렬된 conv id 의 마지막 val_frac 을 val (결정적). 기준선 평가와 미세조정이 같은 val 을 쓴다.
- 특징: features/<encoder>/<manifest>/<id>.npy (2, T, D) @src_hz → 화자 채널 [start,end] 슬라이스 → 13 Hz 리샘플.
"""
import os, re, json, glob, math, unicodedata
from typing import List, Dict, Iterator, Tuple, Optional
import numpy as np, torch
from torch.utils.data import Dataset
from ..data.corpora import parse_srt, SPEECH_LABELS
from ..probe.data import FeatureIndex, resample_feats

MAN = os.environ.get("DATA_MANIFEST_DIR", "/data3/tskim/manifests"); FEAT = os.environ.get("DATA_FEATURE_CACHE_DIR", "/data3/tskim/features")
LABEL_ROOT = {"aihub-ts01-5": "/data3/tskim/corpora/aihub/adult-train-labels", "aihub-vs02": "/data3/tskim/corpora/aihub/adult-val-labels"}
AUDIO_ROOT = {"aihub-ts01-5": "/data3/tskim/corpora/aihub/adult-ts01-5-wav", "aihub-vs02": "/data3/tskim/corpora/aihub/adult-vs02-wav", "otoSpeech": "/data3/tskim/corpora/turnbench/otoSpeech16k"}
LANG = {"aihub-ts01-5": "Korean", "aihub-vs02": "Korean", "otoSpeech": "English"}
def _f(x): return float(str(x).replace(",", ""))

def manifest_rows(name: str) -> Dict[str, dict]:
    return {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(MAN, name, "manifest.jsonl"))}

def split_ids(name: str, ids: List[str], val_frac: float = 0.08) -> Tuple[List[str], List[str]]:
    ids = sorted(ids); n = max(1, int(len(ids) * val_frac)); return ids[:-n], ids[-n:]

_JSON_IDX: Dict[str, Dict[str, str]] = {}
def iter_utterances(name: str, conv_id: str, min_dur: float = 0.3, max_dur: float = 20.0) -> Iterator[dict]:
    """yield dict(conv, speaker, start, end, text)"""
    if name.startswith("aihub"):
        if name not in _JSON_IDX: _JSON_IDX[name] = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(LABEL_ROOT[name], "**", "*.json"), recursive=True)}
        p = _JSON_IDX[name].get(conv_id)
        if not p: return
        d = json.load(open(p, encoding="utf-8")); swap = manifest_rows(name).get(conv_id, {}).get("meta", {}).get("speaker_channel_swapped", False)
        for u in d.get("Conversation", []):
            s, e, t = _f(u["StartTime"]), _f(u["EndTime"]), u.get("Text", "").strip(); spk = 0 if str(u["SpeakerNo"]).endswith("1") else 1
            if swap: spk = 1 - spk
            if min_dur <= e - s <= max_dur and t: yield dict(conv=conv_id, speaker=spk, start=s, end=e, text=t)
    else:
        d = os.path.join(AUDIO_ROOT[name], conv_id.split("oto-")[1])
        for c in (0, 1):
            for s, e, lab, t in parse_srt(os.path.join(d, f"speaker_{c+1}_annotation_a.srt")):
                if lab in SPEECH_LABELS and min_dur <= e - s <= max_dur and t.strip(): yield dict(conv=conv_id, speaker=c, start=s, end=e, text=t.strip())

_PUNCT = re.compile(r"[^\w\s]", re.U); _TAG = re.compile(r"<[^>]{1,20}>")
def normalize_text(t: str, lang: str) -> str:
    t = _TAG.sub(" ", t)                                            # Nemotron 언어 태그(<ko-KR>), 특수 토큰 제거
    t = unicodedata.normalize("NFKC", t).lower(); t = _PUNCT.sub(" ", t); t = re.sub(r"\s+", " ", t).strip()
    return t.replace(" ", "") if lang == "Korean" else t          # 한국어는 CER (공백 제거)

class UttFeatureDataset(Dataset):
    """(feats (T,D) @tgt_hz, text, lang, meta) — 캐시 encoder 특징에서 발화 구간을 자른다."""
    def __init__(self, manifests: List[str], encoder: str = "nemotron-c0", split: str = "train", val_frac: float = 0.08, tgt_hz: float = 13.0, max_utts: Optional[int] = None, seed: int = 0):
        self.items = []; self.tgt_hz = tgt_hz
        for name in manifests:
            fi = FeatureIndex(FEAT, encoder, name); tr, va = split_ids(name, list(fi.rows.keys()), val_frac); ids = tr if split == "train" else va
            for cid in ids:
                for u in iter_utterances(name, cid): self.items.append((name, fi.rows[cid]["npy"], fi.frame_hz, u))
        import random; random.Random(seed).shuffle(self.items)
        if max_utts: self.items = self.items[:max_utts]
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        name, npy, hz, u = self.items[i]; F = np.load(npy, mmap_mode="r")
        a, b = int(math.floor(u["start"] * hz)), max(int(math.floor(u["start"] * hz)) + 2, int(math.ceil(u["end"] * hz))); f = np.asarray(F[u["speaker"], a:b]).astype(np.float32)[None]
        f = resample_feats(f, hz, self.tgt_hz)[0]
        return dict(feats=torch.from_numpy(f), text=u["text"], lang=LANG[name], manifest=name, conv=u["conv"], start=u["start"], end=u["end"], speaker=u["speaker"])

def collate_utts(batch):
    T = max(b["feats"].shape[0] for b in batch); D = batch[0]["feats"].shape[1]
    feats = torch.zeros(len(batch), T, D); lens = torch.tensor([b["feats"].shape[0] for b in batch])
    for i, b in enumerate(batch): feats[i, : b["feats"].shape[0]] = b["feats"]
    return dict(feats=feats, lens=lens, text=[b["text"] for b in batch], lang=[b["lang"] for b in batch], meta=[{k: b[k] for k in ("manifest", "conv", "start", "end", "speaker")} for b in batch])

"""코퍼스 리더 → Conversation. 오디오는 16 kHz (2, T) 로 통일.

aihub          : stereo wav 16k + JSON(Conversation[{StartTime,EndTime,SpeakerNo,Text}]). VAD = 에너지 (라벨 EndTime 은 넉넉함).
otoSpeech      : speaker_{1,2}_audio.wav 48k mono + speaker_{1,2}_annotation_a.srt. VAD = 라벨(speech 범주) 기본, 에너지 병행 가능.
turnbench-dev  : HF parquet 행. speaker_{1,2}_audio(FLAC bytes) + annotation_a/b/c. VAD = annotation_a(speech 범주).
"""
import os, io, re, json, glob
from typing import Iterator, List, Optional, Tuple
import numpy as np, soundfile as sf
from .conversation import Conversation, Utterance, SR
from .vad import energy_vad, frames_to_segments

# otoSpeech / TurnBench 라벨 범주 → 음성 활동으로 볼 것인가
SPEECH_LABELS = {"Normal Turn", "Acknowledgement Backchannel", "Continuer Backchannel", "Filler", "Speech, Non-Linguistic", "Laughter"}
NONSPEECH_LABELS = {"Non-Speech Noise", "Channel Bleed"}
BACKCHANNEL_LABELS = {"Acknowledgement Backchannel", "Continuer Backchannel"}

def _resample(x: np.ndarray, sr: int, target: int = SR) -> np.ndarray:
    if sr == target: return x.astype(np.float32)
    try:
        import soxr; return soxr.resample(x, sr, target).astype(np.float32)
    except ImportError:
        import librosa; return librosa.resample(x.astype(np.float32), orig_sr=sr, target_sr=target)

def _num(x): return float(str(x).replace(",", ""))

# ───────────────────────────── AI Hub ─────────────────────────────
def load_aihub(wav: str, js: Optional[str] = None, load_audio: bool = True, vad_hop_ms: float = 20.0) -> Conversation:
    x, sr = sf.read(wav, dtype="float32", always_2d=True)          # (T, 2)
    audio = np.stack([_resample(x[:, 0], sr), _resample(x[:, 1], sr)])
    dur = audio.shape[1] / SR
    vad = [frames_to_segments(energy_vad(audio[c], SR, vad_hop_ms), vad_hop_ms / 1000) for c in (0, 1)]
    utts, meta = [], {}
    if js and os.path.exists(js):
        d = json.load(open(js, encoding="utf-8"))
        meta = dict(domain=d.get("ConversationInfo", {}).get("Domain"), noise=d.get("Noise", {}),
                    spk=[d.get("Speaker1", {}).get("ID"), d.get("Speaker2", {}).get("ID")])
        raw = [(0 if str(u.get("SpeakerNo", "")).endswith("1") else 1, _num(u["StartTime"]), _num(u["EndTime"]), u.get("Text", ""), u.get("SpeakerEmotionCategory", "")) for u in d.get("Conversation", [])]
        # 라벨 Speaker1/2 가 채널 0/1 에 대응하는지 파일별로 판정: Speaker1 구간이 어느 채널 VAD 와 더 겹치는가
        from .vad import segments_to_frames
        hop = vad_hop_ms / 1000; n = int(dur / hop) + 1
        ch = [segments_to_frames(vad[c], hop, n) for c in (0, 1)]
        s1 = segments_to_frames([(s, e) for k, s, e, _, _ in raw if k == 0], hop, n)
        ov = [float((s1 & ch[c]).sum()) for c in (0, 1)]
        swap = ov[1] > ov[0]; meta["speaker_channel_swapped"] = bool(swap); meta["label_channel_overlap"] = ov
        for k, s, e, t, l in raw:
            utts.append(Utterance((1 - k) if swap else k, s, e, t, l))
    return Conversation(os.path.splitext(os.path.basename(wav))[0], "aihub", audio if load_audio else None, dur, vad, "energy", utts, meta)

# ───────────────────────────── otoSpeech ─────────────────────────────
_SRT_T = re.compile(r"(\d+):(\d+):(\d+),(\d+)")
def _srt_time(s: str) -> float:
    h, m, sec, ms = map(int, _SRT_T.match(s.strip()).groups()); return h * 3600 + m * 60 + sec + ms / 1000

def parse_srt(path: str) -> List[Tuple[float, float, str, str]]:
    out = []; blocks = open(path, encoding="utf-8").read().strip().split("\n\n")
    for b in blocks:
        lines = [l for l in b.splitlines() if l.strip()]
        if len(lines) < 3 or "-->" not in lines[1]: continue
        a, c = lines[1].split("-->"); txt = " ".join(lines[2:])
        m = re.match(r"\[(.+?)\]\s*(.*)", txt); label, text = (m.group(1), m.group(2)) if m else ("", txt)
        out.append((_srt_time(a), _srt_time(c), label, text))
    return out

def load_otospeech(d: str, load_audio: bool = True, vad_from: str = "label") -> Conversation:
    meta = json.load(open(os.path.join(d, "metadata.json")))
    chans, utts, vad = [], [], [[], []]
    for c in (0, 1):
        if load_audio:
            x, sr = sf.read(os.path.join(d, f"speaker_{c+1}_audio.wav"), dtype="float32"); chans.append(_resample(x, sr))
        for s, e, label, text in parse_srt(os.path.join(d, f"speaker_{c+1}_annotation_a.srt")):
            utts.append(Utterance(c, s, e, text, label))
            if label in SPEECH_LABELS: vad[c].append((s, e))
    audio = np.stack(chans) if load_audio else None
    dur = audio.shape[1] / SR if load_audio else max((u.end for u in utts), default=0.0)
    if vad_from == "energy" and load_audio:
        vad = [frames_to_segments(energy_vad(audio[c], SR, 20.0), 0.02) for c in (0, 1)]
    return Conversation(f"oto-{meta.get('task_id', os.path.basename(d.rstrip('/')))}", "otoSpeech", audio, dur, vad, vad_from, utts, meta)

# ───────────────────────────── TurnBench dev (parquet) ─────────────────────────────
def load_turnbench_row(row: dict, load_audio: bool = True, annot: str = "a") -> Conversation:
    chans, utts, vad = [], [], [[], []]
    for c in (0, 1):
        if load_audio:
            x, sr = sf.read(io.BytesIO(row[f"speaker_{c+1}_audio"]["bytes"]), dtype="float32"); chans.append(_resample(x, sr))
        for u in row.get(f"speaker_{c+1}_annotation_{annot}", []) or []:
            utts.append(Utterance(c, u["start_s"], u["end_s"], u.get("text", ""), u.get("label", "")))
            if u.get("label") in SPEECH_LABELS: vad[c].append((u["start_s"], u["end_s"]))
    audio = np.stack(chans) if load_audio else None
    dur = audio.shape[1] / SR if load_audio else max((u.end for u in utts), default=0.0)
    return Conversation(f"tb-{row['conversation_id']}", "turnbench-dev", audio, dur, vad, "label", utts, row.get("metadata", {}))

# ───────────────────────────── 열거 ─────────────────────────────
def iter_corpus(corpus: str, root: str, limit: Optional[int] = None, load_audio: bool = True, **kw) -> Iterator[Conversation]:
    if corpus == "aihub":
        wavs = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
        jidx = {os.path.splitext(os.path.basename(j))[0]: j for j in glob.glob(os.path.join(kw.get("label_root", root), "**", "*.json"), recursive=True)}
        for w in wavs[:limit]:
            yield load_aihub(w, jidx.get(os.path.splitext(os.path.basename(w))[0]), load_audio)
    elif corpus == "otoSpeech":
        dirs = sorted((d for d in glob.glob(os.path.join(root, "*")) if os.path.isfile(os.path.join(d, "metadata.json"))), key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else 0)
        only = kw.get("only")
        for d in dirs[:limit]:
            if only and f"oto-{os.path.basename(d)}" not in only: continue
            yield load_otospeech(d, load_audio, kw.get("vad_from", "label"))
    elif corpus == "turnbench-dev":
        import pyarrow.parquet as pq
        n = 0; seen = set()
        snaps = sorted(glob.glob(os.path.join(root, "*", "data")))          # HF 캐시에 snapshot 이 여럿일 수 있음 → 최신 하나만
        files = sorted(glob.glob(os.path.join(snaps[-1] if snaps else root, "**", "*.parquet"), recursive=True))
        for f in files:
            for row in pq.read_table(f).to_pylist():
                if row["conversation_id"] in seen: continue
                seen.add(row["conversation_id"]); yield load_turnbench_row(row, load_audio); n += 1
                if limit and n >= limit: return
    else:
        raise ValueError(corpus)

def load_conversation(corpus: str, path: str, **kw) -> Conversation:
    return {"aihub": lambda: load_aihub(path, kw.get("json"), kw.get("load_audio", True)),
            "otoSpeech": lambda: load_otospeech(path, kw.get("load_audio", True), kw.get("vad_from", "label"))}[corpus]()

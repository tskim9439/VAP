"""Phase 2 코퍼스 → Dialogue (정본 §8). 화자별 채널(또는 채널 기원 조각)이 있는 A등급 자료만 다룬다.

  aihub71631 stereo : 대화 wav(2ch, 채널 = 화자) + 라벨 JSON. 화자↔채널은 corpora.load_aihub 의 채널 VAD 판정을 재사용.
  aihub crops       : 134-1(성인)·134-2(청소년) 화자별 조각 `<stem>_<n>.wav`(n = Conversation 순번, 1부터) + 라벨 JSON. 조각을 원 시각에 놓는다(pieces).
  otoSpeech         : speaker_{1,2}_audio.wav + annotation_a.srt (SPEECH_LABELS 만 발화).
  AMI               : <root>/<meeting>/<meeting>.Headset-<ch>.wav + ami_public_manual_1.6.2.zip (segments·words, agent→channel = corpusResources/meetings.xml).
  NOTSOFAR-1        : <meeting>/close_talk/CT_*.wav + gt_transcription.json(단어 시각) + gt_meeting_metadata.json.
  ICSI              : <root>/<meeting>/chan*.sph + ICSI_core_NXT.zip(Segments/Words, participant) + ICSI_original_transcripts.zip(.mrt: participant→channel).
텍스트는 raw 로 두고 TN 은 builder(experiments/p2_build_dialogues.py)에서 `textnorm.target` 으로 건다. 시각은 라벨값이며 정렬(p2_align)로 다시 뽑는다.
"""
import os, re, json, glob, zipfile, unicodedata
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple, Iterator
from .dialogue import Dialogue, Utterance, ChannelRef

def _nfc(s: str) -> str: return unicodedata.normalize("NFC", s)
def _num(x) -> float:
    s = str(x).replace(",", "")
    if ":" in s:
        p = s.split(":"); return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
    return float(s)

# ───────────────────────────── AI Hub 71631 stereo ─────────────────────────────
def load_aihub_stereo(wav: str, js: str, split: str = "train") -> Dialogue:
    from .corpora import load_aihub
    conv = load_aihub(wav, js, load_audio=True)                     # 채널 VAD 로 화자↔채널 판정(오디오는 버림)
    stem = os.path.splitext(os.path.basename(wav))[0]
    utts = [Utterance(speaker=str(u.speaker), start=float(u.start), end=float(u.end), raw=u.text, utt_id=f"{stem}_{i+1:06d}") for i, u in enumerate(conv.utterances)]
    chans = {str(c): ChannelRef(path=f"{wav}#ch{c}") for c in (0, 1)}
    return Dialogue(conv_id=f"71631:{stem}", corpus="aihub71631", lang="Korean", split=split, duration_s=float(conv.duration), speakers=["0", "1"], utterances=utts, channels=chans,
                    meta=dict(swapped=bool(conv.meta.get("speaker_channel_swapped")), domain=conv.meta.get("domain"), src_json=js))

# ───────────────────────────── AI Hub 134-1 / 134-2 crops ─────────────────────────────
def load_aihub_crops(js: str, crop_dir: str, corpus: str, split: str = "train") -> Dialogue:
    """라벨 JSON + 조각 디렉토리 → Dialogue. 빠진 조각은 meta.missing 에 순번을 남기고 발화는 유지한다(활동·turn 감독은 QC 에서 마스크)."""
    d = json.load(open(js, encoding="utf-8")); stem = os.path.splitext(os.path.basename(js))[0]
    conv = d.get("Conversation", []); utts = []; pieces: Dict[str, List[Tuple[str, float]]] = {}; missing = []
    for i, u in enumerate(conv, 1):
        spk = "0" if str(u.get("SpeakerNo", "")).endswith("1") else "1"; s, e = _num(u["StartTime"]), _num(u["EndTime"])
        p = os.path.join(crop_dir, f"{stem}_{i}.wav")
        if os.path.exists(p): pieces.setdefault(spk, []).append((p, s))
        else: missing.append(i)
        utts.append(Utterance(speaker=spk, start=s, end=e, raw=u.get("Text", ""), utt_id=f"{stem}_{i:06d}"))
    try: dur = _num(d["File"]["FileLength"])
    except Exception: dur = max((u.end for u in utts), default=0.0)
    chans = {spk: ChannelRef(pieces=pieces.get(spk, [])) for spk in ("0", "1")}
    return Dialogue(conv_id=f"{corpus}:{stem}", corpus=corpus, lang="Korean", split=split, duration_s=float(dur), speakers=["0", "1"], utterances=utts, channels=chans,
                    meta=dict(missing=missing, n_utts=len(conv), complete=not missing, src_json=js))

def iter_aihub_crops(label_root: str, crop_dir: str, corpus: str, split: str = "train", limit: Optional[int] = None) -> Iterator[Dialogue]:
    """label_root 아래 JSON(또는 zip 안 JSON) 마다 조각이 하나라도 있으면 Dialogue."""
    stems = {}
    for f in os.listdir(crop_dir):
        if f.endswith(".wav"): stems.setdefault(f[:-4].rsplit("_", 1)[0], 0); stems[f[:-4].rsplit("_", 1)[0]] += 1
    n = 0
    for jp in sorted(glob.glob(os.path.join(label_root, "**", "*.json"), recursive=True)):
        stem = os.path.splitext(os.path.basename(jp))[0]
        if stem not in stems: continue
        yield load_aihub_crops(jp, crop_dir, corpus, split); n += 1
        if limit and n >= limit: return

# ───────────────────────────── otoSpeech ─────────────────────────────
def load_otospeech_dialogue(d: str, split: str = "train") -> Dialogue:
    from .corpora import parse_srt, SPEECH_LABELS
    import soundfile as sf
    meta = json.load(open(os.path.join(d, "metadata.json"))); utts = []; chans = {}; dur = 0.0
    for c in (0, 1):
        wav = os.path.join(d, f"speaker_{c+1}_audio.wav"); info = sf.info(wav); dur = max(dur, info.frames / info.samplerate)
        chans[str(c)] = ChannelRef(path=wav)
        for k, (s, e, label, text) in enumerate(parse_srt(os.path.join(d, f"speaker_{c+1}_annotation_a.srt"))):
            if label in SPEECH_LABELS and text.strip():
                utts.append(Utterance(speaker=str(c), start=s, end=e, raw=text, utt_id=f"s{c+1}_{k:05d}"))
    tid = meta.get("task_id", os.path.basename(d.rstrip("/")))
    return Dialogue(conv_id=f"oto:{tid}", corpus="otoSpeech", lang="English", split=split, duration_s=float(dur), speakers=["0", "1"], utterances=utts, channels=chans,
                    meta=dict(actors=[meta.get("speaker_1_actor_id"), meta.get("speaker_2_actor_id")], conversation_type=meta.get("conversation_type"), src=d))

# ───────────────────────────── AMI ─────────────────────────────
class AmiAnnotations:
    def __init__(self, zip_path: str):
        self.z = zipfile.ZipFile(zip_path)
        self.meetings = {m.get("observation"): {s.get("nxt_agent"): int(s.get("channel")) for s in m} for m in ET.fromstring(self.z.read("corpusResources/meetings.xml"))}
    def load(self, meeting: str, root: str, split: str = "train") -> Dialogue:
        import soundfile as sf
        chans = self.meetings[meeting]; utts = []; crefs = {}; dur = 0.0
        for agent, ch in chans.items():
            wav = os.path.join(root, meeting, f"{meeting}.Headset-{ch}.wav")
            if not os.path.exists(wav): continue
            info = sf.info(wav); dur = max(dur, info.frames / info.samplerate); crefs[agent] = ChannelRef(path=wav)
            words = []
            for el in ET.fromstring(self.z.read(f"words/{meeting}.{agent}.words.xml")):
                if el.tag == "w" and el.get("punc") != "true" and el.get("starttime") and el.text: words.append((float(el.get("starttime")), float(el.get("endtime")), el.text))
            for n, el in enumerate(ET.fromstring(self.z.read(f"segments/{meeting}.{agent}.segments.xml"))):
                s, e = el.get("transcriber_start"), el.get("transcriber_end")
                if not (s and e): continue
                s, e = float(s), float(e); txt = " ".join(w for ws, we, w in words if ws >= s - 1e-3 and we <= e + 1e-3)
                if not txt.strip(): continue                                      # 단어 없는 segment(웃음·비음성) 제외
                utts.append(Utterance(speaker=agent, start=s, end=e, raw=txt, utt_id=f"{agent}_{n:05d}"))
        return Dialogue(conv_id=f"ami:{meeting}", corpus="AMI", lang="English", split=split, duration_s=dur, speakers=sorted(crefs), utterances=[u for u in utts if u.speaker in crefs], channels=crefs, meta=dict(channels=chans))

# ───────────────────────────── NOTSOFAR-1 ─────────────────────────────
def load_notsofar(meeting_dir: str, split: str = "train") -> Dialogue:
    gt = json.load(open(os.path.join(meeting_dir, "gt_transcription.json"))); md = json.load(open(os.path.join(meeting_dir, "gt_meeting_metadata.json")))
    utts = []; crefs = {}
    for i, u in enumerate(gt):
        spk = u["speaker_id"]; crefs.setdefault(spk, ChannelRef(path=os.path.join(meeting_dir, u["ct_wav_file_name"])))
        wt = [(w, float(a), float(b)) for w, a, b in u.get("word_timing", [])]
        raw = re.sub(r"<[^>]*>", " ", u.get("text", "")).strip()                 # <ST/> 등 태그 제거
        if not raw: continue
        utts.append(Utterance(speaker=spk, start=float(u["start_time"]), end=float(u["end_time"]), raw=raw, utt_id=f"u{i:05d}", word_timing=wt))
    return Dialogue(conv_id=f"notsofar:{md.get('meeting_id', os.path.basename(meeting_dir))}", corpus="NOTSOFAR", lang="English", split=split, duration_s=float(md.get("MeetingDurationSec", max((u.end for u in utts), default=0.0))),
                    speakers=sorted(crefs), utterances=utts, channels=crefs, meta=dict(hashtags=md.get("Hashtags"), room=md.get("Room"), n=md.get("NumParticipants")))

# ───────────────────────────── ICSI ─────────────────────────────
class IcsiAnnotations:
    """NXT core(Segments/Words, participant 속성) + original transcripts(.mrt 의 Participant Name/Channel) → 참가자별 채널."""
    def __init__(self, core_zip: str, transcripts_zip: str):
        self.z = zipfile.ZipFile(core_zip); self.t = zipfile.ZipFile(transcripts_zip)
        self.members = set(self.z.namelist()); self.tm = {os.path.basename(m): m for m in self.t.namelist() if m.endswith(".mrt")}
    def channel_map(self, meeting: str) -> Dict[str, str]:
        root = ET.fromstring(self.t.read(self.tm[f"{meeting}.mrt"]))
        return {p.get("Name"): p.get("Channel") for p in root.iter("Participant") if p.get("Name") and p.get("Channel")}
    def load(self, meeting: str, root: str, split: str = "train") -> Dialogue:
        import soundfile as sf
        pchan = self.channel_map(meeting); utts = []; crefs = {}; dur = 0.0; agents = sorted({m.split(".")[1] for m in self.members if m.startswith(f"ICSI/Segments/{meeting}.")})
        for agent in agents:
            words = []
            wm = f"ICSI/Words/{meeting}.{agent}.words.xml"
            if wm in self.members:
                for el in ET.fromstring(self.z.read(wm)):
                    if el.tag == "w" and el.get("starttime") and el.get("endtime") and el.text: words.append((float(el.get("starttime")), float(el.get("endtime")), el.text))
            for n, el in enumerate(ET.fromstring(self.z.read(f"ICSI/Segments/{meeting}.{agent}.segs.xml")).iter()):
                if not el.tag.endswith("segment") or not (el.get("starttime") and el.get("endtime")): continue
                part = el.get("participant"); ch = pchan.get(part)
                if not part or not ch: continue
                wav = os.path.join(root, meeting, f"{ch}.flac")                     # 원본 SPH 는 shorten 압축이라 sph2pipe→flac 변환본(VAPKT-data/data/audio/icsi)을 root 로 준다
                if not os.path.exists(wav): wav = os.path.join(root, meeting, f"{ch}.sph")
                if not os.path.exists(wav): continue
                if part not in crefs:
                    info = sf.info(wav); dur = max(dur, info.frames / info.samplerate); crefs[part] = ChannelRef(path=wav)
                s, e = float(el.get("starttime")), float(el.get("endtime")); txt = " ".join(w for ws, we, w in words if ws >= s - 1e-3 and we <= e + 1e-3)
                if not txt: continue                                             # 비음성(nonvocalsound) segment 제외
                utts.append(Utterance(speaker=part, start=s, end=e, raw=txt, utt_id=f"{agent}_{n:05d}"))
        return Dialogue(conv_id=f"icsi:{meeting}", corpus="ICSI", lang="English", split=split, duration_s=dur, speakers=sorted(crefs), utterances=utts, channels=crefs, meta=dict(channels=pchan))

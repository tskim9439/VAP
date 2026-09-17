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

# ───────────────────────────── held-out(미학습) 평가용: CHiME-6 dev/eval · NIKL 일상대화 2020 ─────────────────────────────
def _hms(x) -> float:
    """"HH:MM:SS.ss" 또는 초(숫자/문자열) → 초."""
    if isinstance(x, (int, float)): return float(x)
    s = str(x).strip()
    if ":" not in s: return float(s)
    parts = [float(p) for p in s.split(":")]; t = 0.0
    for p in parts: t = t * 60 + p
    return t

_CHIME_TAG = re.compile(r"\[[^\]]*\]")
def load_chime6(session: str, transcript_json: str, audio_dir: str, split: str = "dev") -> Dialogue:
    """CHiME-6: transcriptions/<split>/<S>.json (start_time/end_time "HH:MM:SS.ss", words, speaker P05…) + 바이노럴 <audio_dir>/<S>_<P>.wav(2ch → 왼쪽 채널).
    [laughs]/[noise] 등 태그 제거, [inaudible]/[redacted]/[unintelligible] 가 있던 발화는 flags=unintelligible(전사 없는 음성으로 취급)."""
    gt = json.load(open(transcript_json)); utts = []; crefs = {}
    for i, u in enumerate(sorted(gt, key=lambda u: _hms(u["start_time"]))):
        spk = u["speaker"]; wav = os.path.join(audio_dir, f"{session}_{spk}.wav")
        if spk not in crefs:
            if not os.path.exists(wav): continue
            crefs[spk] = ChannelRef(path=wav + "#ch0")
        words = u.get("words", ""); bad = bool(re.search(r"\[(inaudible|redacted|unintelligible)[^\]]*\]", words, re.I))
        raw = re.sub(r"\s+", " ", _CHIME_TAG.sub(" ", words)).strip()
        if not raw and not bad: continue
        s, e = _hms(u["start_time"]), _hms(u["end_time"])
        if e <= s: continue
        utts.append(Utterance(speaker=spk, start=s, end=e, raw=raw, utt_id=f"{session}_{i:05d}", flags=(["unintelligible"] if bad else [])))
    dur = max((u.end for u in utts), default=0.0) + 1.0
    return Dialogue(conv_id=f"chime6:{session}", corpus="CHiME6", lang="English", split=split, duration_s=dur, speakers=sorted(crefs), utterances=[u for u in utts if u.speaker in crefs], channels=crefs, meta=dict(binaural=True))

def load_nikl_dialogue(json_path: str, idx: Dict[str, str], split: str = "heldout", corpus: str = "nikl2020") -> Optional[Dialogue]:
    """NIKL 일상대화(연도별 JSON + 발화 단위 PCM): 발화 PCM 을 그 화자 채널의 [start, end] 자리에 놓는다(pieces). note='발화겹침' 발화는 flags=overlap(다른 화자 소리가 섞임).
    텍스트는 original_form(raw), TN 은 builder 의 'nikl' 파서."""
    from .nikl import read_dialogue, pcm_path, pcm_duration
    utts = []; pieces: Dict[str, list] = {}; dlg_id = None
    for u in read_dialogue(json_path):
        dlg_id = dlg_id or u["dialogue"]; p = pcm_path(idx, u["id"])
        if not p or not os.path.exists(p) or u["start"] is None or u["end"] is None: continue
        s, e = _hms(u["start"]), _hms(u["end"])
        if e <= s: e = s + pcm_duration(p)
        spk = str(u["speaker"]); pieces.setdefault(spk, []).append((p, s))
        utts.append(Utterance(speaker=spk, start=s, end=e, raw=u["raw"], utt_id=u["id"], flags=(["overlap"] if "겹침" in (u["note"] or "") else [])))
    if not utts: return None
    dur = max(u.end for u in utts) + 0.5; chans = {s: ChannelRef(path="", pieces=pcs) for s, pcs in pieces.items()}
    return Dialogue(conv_id=f"{corpus}:{dlg_id}", corpus=corpus, lang="Korean", split=split, duration_s=dur, speakers=sorted(chans), utterances=utts, channels=chans, meta=dict(src_json=json_path))

# ───────────────────────────── held-out 평가용: TurnBench dev (EN 2인 full-duplex, 화자별 채널 + 턴 라벨 3 트랙) ─────────────────────────────
TURNBENCH_NONSPEECH = {"Non-Speech Noise", "Speech, Non-Linguistic"}
def _iou(a0, a1, b0, b1) -> float:
    inter = max(0.0, min(a1, b1) - max(a0, b0)); union = max(a1, b1) - min(a0, b0); return inter / union if union > 0 else 0.0

def iter_turnbench(parquet_dir: str, audio_out: str, split: str = "heldout", consensus_iou: float = 0.5) -> Iterator[Dialogue]:
    """HF parquet(conversation_id, speaker_{1,2}_audio{bytes,path} FLAC, speaker_{1,2}_annotation_{a,b,c}[{start_s,end_s,label,text}], metadata) → Dialogue.
    발화 = 트랙 a 의 구간(비언어 라벨은 flags=nonspeech → 전사 없는 음성으로 마스크). 턴 채점용 합의(사용자 결정 2026-09-17: 3 트랙 합의 구간만) = 트랙 b·c 에 IoU ≥ consensus_iou 인 구간이 모두 있는 발화 → meta.consensus_utts.
    오디오는 <audio_out>/<conv>_speaker_{1,2}.flac 로 한 번 풀어 둔다."""
    import pyarrow.parquet as pq
    os.makedirs(audio_out, exist_ok=True)
    for pf in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        t = pq.ParquetFile(pf)
        for batch in t.iter_batches(batch_size=4):
            for row in batch.to_pylist():
                cid = str(row["conversation_id"]); utts = []; chans = {}; labels = {}; consensus = []; tracks = {}
                for sp in ("1", "2"):
                    au = row.get(f"speaker_{sp}_audio") or {}
                    if not au.get("bytes"): continue
                    wav = os.path.join(audio_out, f"{cid}_speaker_{sp}.flac")
                    if not os.path.exists(wav):
                        with open(wav + ".tmp", "wb") as f: f.write(au["bytes"])
                        os.replace(wav + ".tmp", wav)
                    chans[sp] = ChannelRef(path=wav)
                    tr = {k: (row.get(f"speaker_{sp}_annotation_{k}") or []) for k in ("a", "b", "c")}; tracks[sp] = {k: len(v) for k, v in tr.items()}
                    for i, seg in enumerate(sorted(tr["a"], key=lambda s: s["start_s"])):
                        s, e = float(seg["start_s"]), float(seg["end_s"]); lab = seg.get("label") or ""; txt = (seg.get("text") or "").strip()
                        if e <= s: continue
                        uid = f"{cid}_{sp}_{i:04d}"; labels[uid] = lab
                        nonspeech = lab in TURNBENCH_NONSPEECH or not txt
                        utts.append(Utterance(speaker=sp, start=s, end=e, raw=txt, utt_id=uid, flags=(["nonspeech"] if nonspeech else [])))
                        if not nonspeech and all(any(_iou(s, e, float(o["start_s"]), float(o["end_s"])) >= consensus_iou for o in tr[k]) for k in ("b", "c")): consensus.append(uid)
                if len(chans) < 2 or not utts: continue
                dur = max(u.end for u in utts) + 0.5; md = row.get("metadata") or {}
                yield Dialogue(conv_id=f"turnbench:{cid}", corpus="TurnBench", lang="English", split=split, duration_s=dur, speakers=sorted(chans), utterances=utts, channels=chans,
                               meta=dict(labels=labels, consensus_utts=consensus, tracks=tracks, conversation_type=md.get("conversation_type"), genders={"1": md.get("speaker_1_actor_gender"), "2": md.get("speaker_2_actor_gender")}))

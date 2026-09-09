"""AI Hub 한국어 코퍼스 리더 — asr-tn-v1.3.0 corpus parsers (2026-09-09).

71631 감정 태깅 자유대화(성인): 대화 wav(16 kHz stereo, 채널 = 화자) + JSON {Conversation: [{Text, TextNo, SpeakerNo, StartTime, EndTime, 감정 태그…}]}.
  서버 오디오는 TS_01.실내_5(757 wav)·VS_02.실외(186 wav) 만 있고 라벨은 전체. 화자→채널 대응은 파일마다 채널 VAD 로 판정(corpora.load_aihub, 뒤바뀜 3 %).
  텍스트 표지: `#@이름#`(익명화 자리표시자, 음성 마스킹) → quarantine(anon); 낱개 `@`·`#` 제거; 자모 웃음(ㅎ·ㅋ) 제거.
031/033 방송콘텐츠 통·번역 음성(한국어 원음): 발화 단위 wav(16 kHz mono, 3–6 s) 가 언어쌍×장르 zip 안에 있고, 전사 JSON 은 별도 zip.
  같은 한국어 원음이 언어쌍마다 반복되므로 파일 stem 으로 중복 제거(처음 본 zip 우선). 텍스트는 `발음전사`(숫자 발음형) 를 타깃, `철자전사` 를 원문으로 보존.
  전사작업본의 KsponSpeech 식 이중표기 `(발음)/(표기)`·`$$`·`#`·`&` 는 발음전사에는 없다(있으면 quarantine)."""
import os, re, json, glob, zipfile, unicodedata
from typing import Dict, Iterator, List, Optional, Tuple

def _nfc(s: str) -> str: return unicodedata.normalize("NFC", s)
def _find_dirs(root: str, name: str, depth: int = 3) -> List[str]:
    """root 아래 depth 단계까지 내려가며 이름이 name 과 같은(유니코드 NFC 정규화 비교) 디렉토리. macOS 에서 입력한 한글(NFD)과 서버 파일명(NFC)이 달라 glob 이 실패하는 문제."""
    out = []; want = _nfc(name)
    def walk(d, k):
        try: ents = os.listdir(d)
        except OSError: return
        for e in ents:
            p = os.path.join(d, e)
            if not os.path.isdir(p): continue
            if _nfc(e) == want: out.append(p)
            elif k > 0: walk(p, k - 1)
    walk(root, depth); return out

# ───────────────────────────── 71631 ─────────────────────────────
_ANON = re.compile(r"#@[^#]*#"); _JAMO_LAUGH = re.compile(r"(?<![가-힣])[ㅋㅎㅠㅜ]+(?![가-힣])")
def normalize_71631(raw: str) -> Tuple[str, List[str]]:
    bad = []
    if _ANON.search(raw): bad.append("anon")
    s = _ANON.sub(" ", raw); s = s.replace("@", " ").replace("#", " "); s = _JAMO_LAUGH.sub(" ", s)
    if re.search(r"[ㄱ-ㅎㅏ-ㅣ]", s): bad.append("jamo")
    return re.sub(r"\s+", " ", s).strip(), bad

def iter_71631(label_root: str, audio_root: str, subsets: Tuple[str, ...] = ("TL_01.실내",), audio_subsets: Tuple[str, ...] = ("TS_01.실내_5", "VS_02.실외"), vad_hop_ms: float = 20.0) -> Iterator[dict]:
    """JSON 마다 wav 를 찾고(없으면 건너뜀) 화자→채널을 판정해 발화 dict 를 낸다: conv, utt_id, speaker(0/1), channel, start, end, raw, emotion."""
    from .corpora import load_aihub
    wavs: Dict[str, str] = {}
    for sub in audio_subsets:
        for d in _find_dirs(audio_root, sub):
            for p in glob.glob(os.path.join(d, "*.wav")): wavs[os.path.splitext(os.path.basename(p))[0]] = p
    for sub in subsets:
        jps = [jp for d in _find_dirs(label_root, sub, depth=1) for jp in glob.glob(os.path.join(d, "**", "*.json"), recursive=True)]
        for jp in sorted(jps):
            stem = os.path.splitext(os.path.basename(jp))[0]; wav = wavs.get(stem)
            if wav is None: continue
            conv = load_aihub(wav, jp, load_audio=True, vad_hop_ms=vad_hop_ms)     # 채널 VAD 로 화자↔채널 판정
            for i, u in enumerate(conv.utterances):
                yield dict(conv=stem, wav=wav, utt_id=f"{stem}_{i + 1:06d}", speaker=u.speaker, channel=u.speaker, start=float(u.start), end=float(u.end), raw=u.text, dur_wav=conv.duration,
                           swapped=bool(conv.meta.get("speaker_channel_swapped")), domain=conv.meta.get("domain"))

# ───────────────────────────── 031/033 방송 ─────────────────────────────
_BC_MARK = re.compile(r"[\$#&\*%~/()]")
def normalize_bc(pron: str) -> Tuple[str, List[str]]:
    """발음전사 → (텍스트, 사유). 표지가 남아 있으면 quarantine."""
    bad = []
    if _BC_MARK.search(pron): bad.append("markup")
    if re.search(r"[0-9]", pron): bad.append("digit")
    s = re.sub(r"\s+", " ", pron).strip()
    return s, bad

def bc_index(roots: List[str], split: str = "Training") -> Tuple[Dict[str, Tuple[str, str, int]], Dict[str, str]]:
    """(stem → (audio zip, member, bytes)) 와 (stem → 전사 zip) — 언어쌍 간 같은 stem 은 처음 본 zip 우선."""
    audio: Dict[str, Tuple[str, str, int]] = {}; trans: Dict[str, str] = {}
    for root in roots:
        for z in sorted(glob.glob(os.path.join(root, "*", split, "*", "TS_*한국어음성_*.zip"))):
            for info in zipfile.ZipFile(z).infolist():
                if not info.filename.endswith(".wav"): continue
                stem = os.path.splitext(os.path.basename(info.filename))[0]; audio.setdefault(stem, (z, info.filename.lstrip("/"), info.file_size))
        for z in sorted(glob.glob(os.path.join(root, "*", split, "*", "TL_*음성전사_*.zip"))):
            for n in zipfile.ZipFile(z).namelist():
                if n.endswith(".json"): trans.setdefault(os.path.splitext(os.path.basename(n))[0], z)
    return audio, trans

def iter_bc(roots: List[str], split: str = "Training") -> Iterator[dict]:
    """발화 dict: utt_id(stem), zip, member, dur_s(파일 크기), genre, pair, pron(발음전사), spell(철자전사), work(전사작업본), gender, age."""
    audio, trans = bc_index(roots, split); zf: Dict[str, zipfile.ZipFile] = {}
    for stem, (az, member, nbytes) in audio.items():
        tz = trans.get(stem)
        if tz is None: continue
        if tz not in zf: zf[tz] = zipfile.ZipFile(tz)
        try: d = json.loads(zf[tz].read(next(n for n in zf[tz].namelist() if n.endswith(stem + ".json"))).decode("utf-8"))
        except Exception: continue
        genre = _nfc(re.search(r"_([^_]+)\.zip$", os.path.basename(az)).group(1)); pair = _nfc(os.path.basename(az).split("_")[1])
        yield dict(utt_id=stem, zip=az, member=member, dur_s=round(max(0, nbytes - 44) / 32000, 3), genre=genre, pair=pair, pron=d.get("발음전사", ""), spell=d.get("철자전사", ""), work=d.get("전사작업본", ""),
                   gender=d.get("발화자성별"), age=d.get("발화자연령대"))

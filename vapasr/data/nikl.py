"""NIKL 일상대화 음성 말뭉치(2020–2025) 리더·정규화 — asr-tn-v1.1.0 corpus parser.

배치(mxc /soundai/DB/raw/nikl): 연도별로 JSON 디렉토리와 PCM 디렉토리가 따로 있고 깊이가 제각각이라 재귀 glob 으로 찾는다.
  JSON: {id, metadata{speaker[…]}, document[{id, utterance[{id, form, original_form, speaker_id, start, end, note}]}]}
  PCM : <pcm_root>/<dialogue id>/<utterance id>.pcm  — 16 kHz·16-bit·mono headerless(KsponSpeech 와 동일), 발화 단위로 이미 잘려 있다.
텍스트는 **original_form**(발음대로 전사, 숫자도 한글)을 쓴다. form 은 정서법 표기(숫자 '2주')라 lexical 타깃과 맞지 않는다.
original_form 표지(2021–2025 전체 209만 발화 감사, 2026-09-07):
  ~        장음/머뭇거림 ('어~')                          → 제거
  -단어-   말더듬·중단 ('-그- 그')                         → 대시만 제거, 단어 유지(실제 발화됨)
  {laughing} 등 비언어 이벤트                             → 제거
  &name1& &company-name& 익명화 자리표시자(음성은 마스킹)   → quarantine (anon)
  (( )) 불명확 / ((추정)) 전사자 추정                        → quarantine (unintelligible)
  . ? ' 구두점                                            → textnorm 이 제거
  note == '발화겹침'                                       → quarantine (overlap; 두 화자 소리가 섞임)
"""
import os, re, glob, json
from typing import Dict, List, Tuple, Optional, Iterator

YEAR_DIRS = {  # 연도 → (JSON 루트, PCM 루트) ; 디렉토리명이 해마다 다르다(오타 DIALOGE 포함)
    "2020": ("NIKL_DIALOGUE_2020_v1.4_PCM", "NIKL_DIALOGUE_2020_v1.4_PCM"),
    "2021": ("NIKL_DIALOGUE_2021_v1.1_JSON", "NIKL_DIALOGUE_2021_v1.1_PCM"),
    "2022": ("NIKL_DIALOGUE_2022_v1.0_JSON", "NIKL_DIALOGE_2022_PCM"),
    "2023": ("NIKL_DIALOGUE_2023_v1.1", "NIKL_DIALOGUE_2023_PCM"),
    "2024": ("NIKL_DIALOGUE_2024_JSON_v1.0", "NIKL_DIALOGUE_2024_PCM_v1.0"),
    "2025": ("NIKL_DIALOGUE_2025_v1.0_JSON", "NIKL_DIALOGUE_2025_v1.0_PCM"),
}
SR, BPS = 16000, 2
_TILDE = re.compile(r"~+"); _DASH = re.compile(r"(?<!\S)-(\S+?)-(?!\S)"); _EVENT = re.compile(r"\{[^{}]*\}")
_ANON = re.compile(r"&[^&\s]*&"); _UNINTEL = re.compile(r"\(\([^()]*\)\)")

def normalize_nikl(raw: str) -> Tuple[str, List[str]]:
    """original_form → (텍스트, quarantine 사유 목록). 구두점은 남겨 두고 textnorm.target_ko 가 지운다."""
    bad = []
    if _ANON.search(raw): bad.append("anon")
    if _UNINTEL.search(raw): bad.append("unintelligible")
    s = _ANON.sub(" ", raw); s = _UNINTEL.sub(" ", s); s = _EVENT.sub(" ", s); s = _TILDE.sub("", s); s = _DASH.sub(r"\1", s)
    return re.sub(r"\s+", " ", s).strip(), bad

def find_year_dirs(root: str, year: str) -> Tuple[Optional[str], Optional[str]]:
    """(json 파일들이 있는 디렉토리 목록의 공통 루트, pcm 루트). 재귀 glob 은 비싸므로 결과를 root/_index/<year>.json 에 캐시한다."""
    jd, pd = YEAR_DIRS[year]; return os.path.join(root, jd), os.path.join(root, pd)

def index_year(root: str, year: str, cache_dir: Optional[str] = None) -> Dict[str, str]:
    """dialogue id → pcm 디렉토리, 'json:<dialogue id>' → json 경로. 한 번 훑어 캐시."""
    cp = os.path.join(cache_dir or os.path.join(root, "_index"), f"nikl-{year}.json")
    if os.path.exists(cp): return json.load(open(cp))
    jroot, proot = find_year_dirs(root, year); idx: Dict[str, str] = {}
    for p in glob.glob(os.path.join(glob.escape(jroot), "**", "*.json"), recursive=True):
        idx["json:" + os.path.splitext(os.path.basename(p))[0]] = p
    dlg = re.compile(r"^S[A-Z]RW\d+$")                          # 대화 디렉토리(SDRW2300000001 등)에서 멈춘다 — PCM 파일 500만 개를 훑지 않도록
    for cur, dirs, _files in os.walk(proot):
        keep = []
        for d in dirs:
            if dlg.match(d): idx.setdefault(d, os.path.join(cur, d))
            else: keep.append(d)
        dirs[:] = keep
    os.makedirs(os.path.dirname(cp), exist_ok=True); json.dump(idx, open(cp + ".tmp", "w")); os.replace(cp + ".tmp", cp); return idx

def read_dialogue(json_path: str) -> Iterator[dict]:
    """발화 dict: id, dialogue, speaker, raw(original_form), form, start, end, note"""
    j = json.load(open(json_path, encoding="utf-8"))
    for doc in (j["document"] if isinstance(j.get("document"), list) else [j]):
        for u in doc.get("utterance", []):
            yield dict(id=u["id"], dialogue=u["id"].split(".")[0], speaker=u.get("speaker_id"), raw=u.get("original_form", ""), form=u.get("form", ""),
                       start=u.get("start"), end=u.get("end"), note=u.get("note", "") or "")

def pcm_path(idx: Dict[str, str], utt_id: str) -> Optional[str]:
    d = idx.get(utt_id.split(".")[0]); return os.path.join(d, utt_id + ".pcm") if d else None

def pcm_duration(path: str) -> float: return (os.path.getsize(path) // BPS) / SR

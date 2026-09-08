"""VAP-ASR 데이터 스키마 v1 (2026-09-08) — manifest(streams.jsonl) · 데이터 카드(dataset.json) · 정렬 레코드(parts/*.jsonl) · 정렬 카드(align.json).

계층: manifest(정본, 스트림 = 학습 샘플 1 개) ─ id ─ 정렬(별도 산출물: textnorm 규약 + tokenizer + 정렬 모델에 종속) ─ 파생 캐시(_items-*.json.gz, 스키마 밖).
버전 규칙: 카드의 schema_version(major.minor). 행에는 버전을 두지 않는다(카드가 대표). major 가 다르면 로더가 거부, minor 는 경고.
검증기는 외부 의존성 없이 필요한 것만 검사한다(타입·필수·열거·범위·단조성). JSON Schema 문서는 docs 용으로 `export_json_schema()` 가 생성한다."""
import os, json, time, hashlib, subprocess
from typing import Dict, List, Optional, Iterable, Tuple

DS_SCHEMA = "vapasr-ds-1.0"; ALIGN_SCHEMA = "vapasr-align-1.0"
MODES = ("stream", "utt"); LANGS = ("English", "Korean"); SPLITS = ("train", "dev", "test", "eval")
CARD, ALIGN_CARD = "dataset.json", "align.json"

def _major(v: Optional[str]) -> Optional[str]: return None if not v else v.rsplit("-", 1)[-1].split(".")[0]

# ───────────────────────────── 행 검증 ─────────────────────────────
def _num(x): return isinstance(x, (int, float)) and not isinstance(x, bool)
def validate_row(r: dict) -> List[str]:
    """streams.jsonl 한 행 → 오류 목록(빈 목록 = 통과)."""
    e = []
    for k, t in (("id", str), ("corpus", str), ("split", str), ("subset", str), ("mode", str), ("lang", str), ("segments", list)):
        if k not in r: e.append(f"missing {k}")
        elif not isinstance(r[k], t): e.append(f"{k}: {type(r[k]).__name__} != {t.__name__}")
    if e: return e
    if r["mode"] not in MODES: e.append(f"mode {r['mode']!r} ∉ {MODES}")
    if r["lang"] not in LANGS: e.append(f"lang {r['lang']!r} ∉ {LANGS}")
    if r["split"] not in SPLITS: e.append(f"split {r['split']!r} ∉ {SPLITS}")
    if not _num(r.get("duration_s")) or r["duration_s"] <= 0: e.append("duration_s must be > 0")
    if "n_utts" in r and (not isinstance(r["n_utts"], int) or r["n_utts"] < 1): e.append("n_utts must be ≥ 1")
    if not r["segments"]: e.append("segments empty")
    t_prev = 0.0
    for i, s in enumerate(r["segments"]):
        for k in ("utt_id", "path", "offset_s", "dur_s", "silence_before_s", "lexical_text"):
            if k not in s: e.append(f"segments[{i}] missing {k}")
        if e: break
        if not isinstance(s["path"], str) or not s["path"]: e.append(f"segments[{i}] path empty")
        if not _num(s["offset_s"]) or s["offset_s"] < -1e-6: e.append(f"segments[{i}] offset_s < 0")
        if not _num(s["dur_s"]) or s["dur_s"] <= 0: e.append(f"segments[{i}] dur_s ≤ 0")
        if not _num(s["silence_before_s"]) or s["silence_before_s"] < -1e-6: e.append(f"segments[{i}] silence_before_s < 0")
        if _num(s["offset_s"]) and s["offset_s"] + 1e-6 < t_prev: e.append(f"segments[{i}] offset_s {s['offset_s']} < previous end {t_prev:.3f} (겹침)")
        if _num(s["offset_s"]) and _num(s["dur_s"]): t_prev = s["offset_s"] + s["dur_s"]
        if not isinstance(s["lexical_text"], str): e.append(f"segments[{i}] lexical_text not str")
    if not e and _num(r.get("duration_s")) and t_prev > r["duration_s"] + 0.05: e.append(f"segments end {t_prev:.3f} > duration_s {r['duration_s']}")
    return e

def validate_align_record(r: dict) -> List[str]:
    """정렬 레코드(parts/*.jsonl 한 줄). utts 가 빈 목록이면 '정렬할 발화 없음/오디오 누락' 레코드로 허용. 토큰 종료 시각은 발화 안에서 단조 비감소."""
    e = []
    if not isinstance(r.get("id"), str): return ["missing id"]
    if not isinstance(r.get("utts"), list): return ["missing utts"]
    for j, u in enumerate(r["utts"]):
        for k in ("text", "tokens"):
            if k not in u: e.append(f"utts[{j}] missing {k}")
        if e: break
        prev = -1.0
        for i, t in enumerate(u["tokens"]):
            if not isinstance(t, dict) or "id" not in t or "end_time" not in t: e.append(f"utts[{j}].tokens[{i}] needs id,end_time"); break
            if not isinstance(t["id"], int) or t["id"] < 0: e.append(f"utts[{j}].tokens[{i}] id"); break
            if not _num(t["end_time"]) or t["end_time"] + 1e-6 < prev: e.append(f"utts[{j}].tokens[{i}] end_time {t['end_time']} < prev {prev} (비단조)"); break
            prev = t["end_time"]
    return e

# ───────────────────────────── 카드 ─────────────────────────────
def _git_commit() -> Optional[str]:
    try: return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))).stdout.strip() or None
    except Exception: return None
def _write_json_atomic(path: str, obj: dict):
    tmp = f"{path}.{os.getpid()}.tmp"; json.dump(obj, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    try: os.replace(tmp, path)
    except OSError: json.dump(obj, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False); os.remove(tmp)

def build_dataset_card(manifest_dir: str, sample: Optional[int] = None, extra: Optional[dict] = None) -> Tuple[dict, List[str]]:
    """streams.jsonl 을 훑어 dataset.json 내용과 검증 오류(최대 20 건)를 만든다. stats.json 이 있으면 fingerprint·textnorm 을 가져온다."""
    p = os.path.join(manifest_dir, "streams.jsonl"); st = json.load(open(os.path.join(manifest_dir, "stats.json"))) if os.path.exists(os.path.join(manifest_dir, "stats.json")) else {}
    n = 0; hours = 0.0; subsets: Dict[str, int] = {}; modes: Dict[str, int] = {}; langs: Dict[str, int] = {}; corpora: Dict[str, int] = {}; splits: Dict[str, int] = {}; errs: List[str] = []; n_seg = 0
    for line in open(p, encoding="utf-8"):
        r = json.loads(line); n += 1
        if sample is None or n <= sample:
            for msg in validate_row(r):
                if len(errs) < 20: errs.append(f"{r.get('id')}: {msg}")
        hours += float(r.get("duration_s", 0)) / 3600; n_seg += len(r.get("segments", []))
        for d, k in ((subsets, "subset"), (modes, "mode"), (langs, "lang"), (corpora, "corpus"), (splits, "split")): d[r.get(k)] = d.get(r.get(k), 0) + 1
    nq = sum(1 for _ in open(os.path.join(manifest_dir, "quarantine.jsonl"), encoding="utf-8")) if os.path.exists(os.path.join(manifest_dir, "quarantine.jsonl")) else None
    card = dict(schema_version=DS_SCHEMA, name=os.path.basename(os.path.normpath(manifest_dir)), corpus=(list(corpora)[0] if len(corpora) == 1 else sorted(corpora)), splits=splits, rows=n, segments=n_seg, hours=round(hours, 2),
                subsets=subsets, modes=modes, langs=langs, textnorm_version=st.get("textnorm_version"), fingerprint=st.get("fingerprint"), quarantined=nq if nq is not None else st.get("quarantined"),
                text_fields=dict(lexical_text="학습·정렬 타깃(asr-tn 규약)", raw_text="원문(있으면)", text="lexical_text 와 동일(하위 호환)", display_source="display 텍스트 출처(none|raw|…)"),
                time_base="stream (segments[].offset_s 는 스트림 시작 기준, silence_before_s 는 삽입 무음)", audio=dict(sample_rate=16000, channels=1, path_forms=["file path", "archive.tar::member"]),
                built=dict(time=time.strftime("%F %T"), commit=_git_commit(), validated_rows=(min(n, sample) if sample else n), errors=len(errs)), **(extra or {}))
    return card, errs

def build_align_card(align_dir: str, manifest_name: Optional[str] = None, sample: Optional[int] = None, aligner: Optional[dict] = None) -> Tuple[dict, List[str]]:
    """parts/*.jsonl(+ 레거시 <id>.jsonl) 을 훑어 align.json 과 오류를 만든다. fingerprint.json(textnorm + tokenizer sha) 을 그대로 품는다."""
    fp = json.load(open(os.path.join(align_dir, "fingerprint.json"))) if os.path.exists(os.path.join(align_dir, "fingerprint.json")) else None
    n = nu = nt = n_empty = 0; errs: List[str] = []; ids = set(); dup = 0
    pdir = os.path.join(align_dir, "parts"); files = [os.path.join(pdir, f) for f in sorted(os.listdir(pdir)) if f.endswith(".jsonl")] if os.path.isdir(pdir) else []
    for f in files:
        for line in open(f, encoding="utf-8"):
            try: r = json.loads(line)
            except Exception: errs.append(f"{os.path.basename(f)}: JSON 깨짐") if len(errs) < 20 else None; continue
            n += 1
            if r["id"] in ids: dup += 1
            ids.add(r["id"])
            if not r.get("utts"): n_empty += 1
            nu += len(r.get("utts", [])); nt += sum(len(u.get("tokens", [])) for u in r.get("utts", []))
            if sample is None or n <= sample:
                for msg in validate_align_record(r):
                    if len(errs) < 20: errs.append(f"{r.get('id')}: {msg}")
    legacy = sum(1 for x in os.listdir(align_dir) if x.endswith(".jsonl")) if os.path.isdir(align_dir) else 0
    card = dict(schema_version=ALIGN_SCHEMA, manifest=manifest_name or os.path.basename(os.path.normpath(align_dir)), align_root=os.path.basename(os.path.dirname(os.path.normpath(align_dir))),
                textnorm_version=(fp or {}).get("textnorm_version"), fingerprint=fp, tokenizer=dict(json_sha256=(fp or {}).get("tokenizer_json_sha256")),
                aligner=aligner or dict(model="Qwen/Qwen3-ForcedAligner-0.6B", note="token end_time 만 사용(초, 스트림 기준)"),
                records=dict(streams=len(ids), duplicate_lines=dup, empty=n_empty, utts=nu, tokens=nt, part_files=len(files), legacy_files=legacy),
                record_format="{id, utts:[{speaker,start,end,text,tokens:[{id,text,end_time}]}]} · utts=[] 는 정렬 대상 없음/오디오 누락",
                built=dict(time=time.strftime("%F %T"), commit=_git_commit(), validated_records=(min(n, sample) if sample else n), errors=len(errs)))
    return card, errs

def write_card(dir_: str, card: dict, name: str): _write_json_atomic(os.path.join(dir_, name), card)
def read_card(dir_: str, name: str) -> Optional[dict]:
    p = os.path.join(dir_, name); return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

# ───────────────────────────── 로더 관문 ─────────────────────────────
def check_cards(manifest_dir: str, align_dir: Optional[str], where: str, tokenizer_json_sha: Optional[str] = None, strict_env: str = "VAPASR_STRICT_SCHEMA") -> None:
    """로더 시작 시 카드 검사. 카드가 없으면 경고만(이관 전 산출물). major 불일치·tokenizer sha 불일치는 strict 면 실패, 아니면 경고."""
    strict = os.environ.get(strict_env) == "1"; rank0 = os.environ.get("RANK", "0") == "0"
    def bad(msg):
        if strict: raise RuntimeError(f"{where}: {msg}")
        if rank0: print(f"  !! {where}: {msg}", flush=True)
    c = read_card(manifest_dir, CARD)
    if c is None: bad(f"dataset.json 없음(스키마 이전 산출물, experiments/ds_cards.py 로 생성)")
    elif _major(c.get("schema_version")) != _major(DS_SCHEMA): bad(f"dataset schema {c.get('schema_version')} ≠ {DS_SCHEMA}")
    if align_dir:
        a = read_card(align_dir, ALIGN_CARD)
        if a is None: bad(f"align.json 없음({align_dir})")
        else:
            if _major(a.get("schema_version")) != _major(ALIGN_SCHEMA): bad(f"align schema {a.get('schema_version')} ≠ {ALIGN_SCHEMA}")
            sha = (a.get("tokenizer") or {}).get("json_sha256")
            if tokenizer_json_sha and sha and sha != tokenizer_json_sha: bad(f"정렬의 tokenizer sha {sha[:12]} ≠ 현재 tokenizer {tokenizer_json_sha[:12]} (토큰 id 불일치 → 정렬 재생성 필요)")

def tokenizer_sha256(tok) -> Optional[str]:
    """tokenizer.json 의 sha256(정렬 fingerprint 의 tokenizer_json_sha256 과 같은 정의)."""
    try:
        p = getattr(tok, "vocab_file", None) or os.path.join(getattr(tok, "name_or_path", ""), "tokenizer.json")
        if not os.path.exists(p): p = os.path.join(tok.name_or_path, "tokenizer.json")
        return hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None
    except Exception: return None

# ───────────────────────────── JSON Schema 문서 ─────────────────────────────
def export_json_schema() -> Dict[str, dict]:
    seg = {"type": "object", "required": ["utt_id", "path", "offset_s", "dur_s", "silence_before_s", "lexical_text"],
           "properties": {"utt_id": {"type": "string"}, "path": {"type": "string", "description": "파일 경로 또는 archive.tar::member"}, "offset_s": {"type": "number", "minimum": 0}, "dur_s": {"type": "number", "exclusiveMinimum": 0},
                          "silence_before_s": {"type": "number", "minimum": 0}, "text": {"type": "string"}, "lexical_text": {"type": "string"}, "raw_text": {"type": "string"}, "display_source": {"type": "string"}, "speaker": {"type": ["string", "null"]}}}
    row = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": f"{DS_SCHEMA}/row", "type": "object",
           "required": ["id", "corpus", "split", "subset", "mode", "lang", "duration_s", "segments"],
           "properties": {"id": {"type": "string"}, "corpus": {"type": "string"}, "split": {"enum": list(SPLITS)}, "subset": {"type": "string"}, "mode": {"enum": list(MODES)}, "lang": {"enum": list(LANGS)},
                          "speaker": {"type": ["string", "null"]}, "chapter": {"type": ["string", "null"]}, "duration_s": {"type": "number", "exclusiveMinimum": 0}, "n_utts": {"type": "integer", "minimum": 1},
                          "silence_after_s": {"type": "number", "minimum": 0}, "segments": {"type": "array", "minItems": 1, "items": seg}, "meta": {"type": "object"}}}
    tok = {"type": "object", "required": ["id", "end_time"], "properties": {"id": {"type": "integer", "minimum": 0}, "text": {"type": "string"}, "end_time": {"type": "number", "minimum": 0}}}
    utt = {"type": "object", "required": ["text", "tokens"], "properties": {"speaker": {"type": "integer"}, "start": {"type": "number"}, "end": {"type": "number"}, "text": {"type": "string"}, "tokens": {"type": "array", "items": tok}}}
    arec = {"$schema": "https://json-schema.org/draft/2020-12/schema", "$id": f"{ALIGN_SCHEMA}/record", "type": "object", "required": ["id", "utts"], "properties": {"id": {"type": "string"}, "utts": {"type": "array", "items": utt}}}
    return {"row": row, "align_record": arec}

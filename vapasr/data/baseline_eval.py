"""외부 ASR 공통 평가: speaker 표기 파싱·동결 TN·MeetEval 채점. GPU 의존 없음."""
import hashlib
import json
import re
from pathlib import Path

from . import textnorm

MODEL_ID = "microsoft/VibeVoice-ASR-Streaming-1.5B"
MODEL_REVISION = "4262d23d8a539a6530cf64fbd0b1751ef9a30853"
CODE_REVISION = "1541f590c7099820f10ea012f48d2399282df69f"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tn_fingerprint():
    return {"version": textnorm.TEXTNORM_VERSION, "sha256": sha256(textnorm.__file__)}


def parse_speakers(chunks):
    """같은 화자/단어가 청크 경계를 넘어도 원문을 연결한 후 파싱한다.

    Timestamp를 발명하지 않는다. 첫 label 이전 텍스트도 unassigned로 남겨 채점.
    문장 안의 'Speaker 2:' 인용을 태그로 오인하지 않도록 줄 시작만 허용한다.
    """
    text = "".join(chunks)
    pattern = re.compile(r"(?:^|\n)[ \t]*Speaker[ \t]+(\d+)[ \t]*:")
    out, speaker, pos = [], "__unassigned__", 0
    for match in pattern.finditer(text):
        if text[pos:match.start()].strip():
            out.append({"speaker": speaker, "text": text[pos:match.start()]})
        speaker, pos = "speaker_" + str(int(match.group(1))), match.end()
    if text[pos:].strip():
        out.append({"speaker": speaker, "text": text[pos:]})
    return out


def score_text(text, lang):
    if lang in ("Korean", "ko"):
        return " ".join(textnorm.score_ko(text, spaces=False))
    if lang in ("English", "en"):
        return textnorm.score_en(text)
    raise ValueError(f"Unsupported language: {lang}")


def seglst(session, segments, lang):
    return [{"session_id": session, "speaker": x["speaker"],
             "words": score_text(x["text"], lang)} for x in segments]


def score_session(session, refs, hyps, lang):
    """정확 ORC·cp. KO는 문자 사이 공백을 넣어 같은 scorer에 전달한다."""
    from meeteval.io import SegLST
    from meeteval.wer import cpwer, orcwer
    ref, hyp = seglst(session, refs, lang), seglst(session, hyps, lang)
    nr = sum(len(x["words"].split()) for x in ref)
    nh = sum(len(x["words"].split()) for x in hyp)
    if not nr or not nh:
        errors = nh if not nr else nr
        base = dict(errors=errors, length=nr, substitutions=0,
                    deletions=nr if not nh else 0, insertions=nh if not nr else 0,
                    rate=errors / nr if nr else None)
        return {"cp": dict(base), "orc": dict(base)}
    result = {}
    for name, fn in (("cp", cpwer), ("orc", orcwer)):
        e = fn(SegLST(ref), SegLST(hyp), reference_sort=False,
               hypothesis_sort=False)[session]
        result[name] = {k: int(getattr(e, k)) for k in
                        ("errors", "length", "substitutions", "deletions", "insertions")}
        result[name]["rate"] = e.errors / e.length if e.length else None
    return result


def safe_prefix_end(utterances, duration, limit):
    """시작점 0을 유지하고 모든 화자의 발화 중간을 피하는 공통 crop 경계."""
    end = min(duration, limit)
    while True:
        crossing = [u["start"] for u in utterances if u["start"] < end < u["end"]]
        if not crossing:
            return end
        end = min(crossing)


def read_manifest(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    ids = [r["session_id"] for r in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError("Empty manifest or duplicate session_id")
    for r in rows:
        if r["purpose"] != "dev_smoke" or not 0 < r["duration_s"] <= 480:
            raise ValueError("This runner accepts dev_smoke clips up to 480 seconds only")
        if r["tn"] != tn_fingerprint():
            raise ValueError("TN fingerprint mismatch")
    return rows

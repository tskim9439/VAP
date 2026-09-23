#!/usr/bin/env python
"""기존 mono 캐시에서 공통 dev smoke WAV·참조 manifest 생성. 원본 변경 없음."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.baseline_eval import safe_prefix_end, sha256, tn_fingerprint


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--sets", default="nikl2020,chime6")
    p.add_argument("--per-set", type=int, default=2)
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--start-at", choices=("first-annotation", "session-start"), default="first-annotation",
                   help="smoke 기본: 첫 주석 발화 3초 전. locked/full-session 평가용 선택 규칙 아님")
    a = p.parse_args()
    if not 1 <= a.per_set or not 10 <= a.seconds <= 480:
        p.error("per-set >= 1 and 10 <= seconds <= 480 required")
    import numpy as np
    import soundfile as sf
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "audio").mkdir()
    rows, audit = [], []
    for corpus in a.sets.split(","):
        src = a.data / f"{corpus}.dialogues.jsonl"
        digest = sha256(src)
        dialogues = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
        selected = 0
        # 파일명순 편향을 줄인 결정적 hash순 개발 표본. 모든 선택·제외 사유 기록.
        for d in sorted(dialogues, key=lambda d: hashlib.sha256(d["conv_id"].encode()).hexdigest()):
            if selected == a.per_set:
                break
            cid = d["conv_id"]
            us = sorted(d["utterances"], key=lambda u: (u["start"], u["end"], u["speaker"]))
            start = max(0, us[0]["start"]-3) if us and a.start_at == "first-annotation" else 0
            end = safe_prefix_end(us, d["duration_s"], start+a.seconds)
            ref = [u for u in us if start <= u["start"] < end and u["end"] <= end]
            cache = a.data / "mono" / d["corpus"] / (cid.replace("/", "_").replace(":", "_") + ".npy")
            reason = None
            if end-start < 10: reason = "no_safe_clip_at_least_10s"
            elif any(u.get("flags") or not u.get("text", "").strip() for u in ref): reason = "incomplete_reference_in_prefix"
            elif not cache.is_file(): reason = "missing_mono_cache"
            audit.append(dict(corpus=corpus, session_id=cid, selected=reason is None,
                              reason=reason, duration_s=end-start, crop_start_s=start))
            if reason:
                continue
            x = np.load(cache, mmap_mode="r")
            count = round(end * 16000)
            if x.ndim != 1 or len(x) < count:
                raise ValueError(f"Malformed/short cache: {cache}")
            wav = np.asarray(x[round(start*16000):count], dtype=np.float32)
            if not np.isfinite(wav).all():
                raise ValueError(f"Non-finite audio: {cache}")
            key = hashlib.sha256(cid.encode()).hexdigest()[:16]
            audio = (a.out / "audio" / f"{key}.wav").resolve()
            sf.write(audio, wav, 16000, subtype="FLOAT")
            rows.append(dict(session_id=cid, corpus=corpus, lang=d["lang"], purpose="dev_smoke",
                             exposure="development_exposed; external_model_pretraining_unknown",
                             audio=str(audio), audio_sha256=sha256(audio), duration_s=len(wav)/16000,
                             crop_start_s=start, start_policy=a.start_at,
                             original_duration_s=d["duration_s"], input_kind="existing_project_mono_cache",
                             source=str(src), source_sha256=digest, source_cache=str(cache), tn=tn_fingerprint(),
                             reference=[dict(speaker=u["speaker"], start=u["start"]-start,
                                             end=u["end"]-start, text=u["text"]) for u in ref]))
            selected += 1
        if selected != a.per_set:
            (a.out / "selection-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
            raise RuntimeError(f"{corpus}: selected {selected}/{a.per_set}; inspect audit, no manifest published")
    (a.out / "selection-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    (a.out / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows))
    print(json.dumps(dict(manifest=str(a.out / "manifest.jsonl"), sessions=len(rows),
                          seconds=sum(r["duration_s"] for r in rows)), ensure_ascii=False))


if __name__ == "__main__":
    main()

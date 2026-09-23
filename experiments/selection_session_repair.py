#!/usr/bin/env python3
"""Evidence-scoped NIKL PCM repair. New sidecars/WAVs only; no training approval."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import digest, file_digest

SESSIONS = ("SDRW2200000598", "SDRW2200000635")
RECIPE = "nikl-explicit-session-48000-v1"


def rate_check(utt_id, nbytes, annotation_duration, header):
    """Rate is preselected for reviewed sessions; never inferred from ratios."""
    if utt_id.split(".")[0] not in SESSIONS:
        return "outside_reviewed_scope"
    if nbytes <= 0 or nbytes % 2:
        return "invalid_pcm_bytes"
    if header[:4] in (b"RIFF", b"RIFX", b"fLaC", b"OggS"):
        return "headered_audio_not_raw_pcm"
    if not math.isfinite(annotation_duration) or annotation_duration <= 0:
        return "invalid_annotation_duration"
    if abs(nbytes/96000-annotation_duration) > .03:
        return "duration_inconsistent_with_fixed_48khz"
    return None


def main():
    import numpy as np
    import soundfile as sf
    import soxr
    from vapasr.data.selection_audio import decode
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--probe", type=Path, required=True)
    ap.add_argument("--human-review", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    if not 1 <= a.workers <= 32:
        raise ValueError("workers must be 1..32")
    human = json.loads(a.human_review.read_text())
    probe = {r["key"]:r for r in map(json.loads, a.probe.open())}
    labels = {}; reviewed = Counter()
    for r in human["rows"]:
        p = probe[r["key"]]
        session = p["utt_id"].split(".")[0]
        if (session not in SESSIONS or r["source"] != "nikl-1000" or
            p.get("rate_hypothesis") != 48000 or
            any(r["labels"][axis] != "pass" for axis in ("timing", "speaker"))):
            raise ValueError("Review does not support the scoped repair")
        if p["utt_id"] in labels:
            raise ValueError("Duplicate human review")
        if file_digest(p["original_audio"]["path"]) != p["original_audio_sha256"]:
            raise ValueError("Reviewed original waveform changed")
        labels[p["utt_id"]] = r
        reviewed[session] += 1
    if set(reviewed) != set(SESSIONS) or min(reviewed.values()) < 2:
        raise ValueError("Insufficient reviewed session evidence")
    index = json.loads(a.index.read_text())
    census = json.loads((a.selection/"census.json").read_text())
    meta = census["sources"]["nikl-1000"]
    if file_digest(meta["path"]) != meta["sha256"]:
        raise ValueError("Original manifest changed")
    candidate_path = a.selection/"nikl-1000.candidates.jsonl.gz"
    candidates = {}
    with gzip.open(candidate_path,"rt") as f:
        for r in map(json.loads, f):
            if r["utt_id"].split(".")[0] in SESSIONS and r["selection_state"] == "PENDING_AUDIO_TEACHERS":
                if r["utt_id"] in candidates or r["manifest_sha256"] != meta["sha256"]:
                    raise ValueError("Candidate identity/provenance mismatch")
                candidates[r["utt_id"]] = r
    inputs=[]; source_hashes={}; extras={}
    for session in SESSIONS:
        jp=Path(index["json:"+session]); folder=Path(index[session])
        if folder.name != session:
            raise ValueError("Session directory mismatch")
        source_hashes[str(jp)]=file_digest(jp)
        us=[u for doc in json.loads(jp.read_text())["document"] for u in doc["utterance"]]
        ids=[u["id"] for u in us]
        if len(ids)!=len(set(ids)) or any(i.split('.')[0]!=session for i in ids):
            raise ValueError("Annotation IDs not unique/session-local")
        extras[session]=sorted(p.name for p in folder.glob("*.pcm") if p.stem not in ids)
        inputs.extend((u,folder/(u["id"]+".pcm")) for u in us)
    a.out.mkdir(parents=True,exist_ok=False)
    (a.out/"audio").mkdir()
    evidence=dict(recipe=RECIPE,sessions=list(SESSIONS),human_sha256=file_digest(a.human_review),
        probe_sha256=file_digest(a.probe),candidate_sha256=file_digest(candidate_path),
        annotation_hashes=source_hashes,script_sha256=file_digest(Path(__file__).resolve()),
        decode_sha256=file_digest(Path(__file__).resolve().parents[1]/"vapasr/data/selection_audio.py"),
        soxr_version=soxr.__version__,training_eligible=False)
    evidence_id=digest(evidence)

    def inspect(item):
        u,path=item;uid=u["id"]
        audit=dict(utt_id=uid,session=uid.split('.')[0],path=str(path),
                   training_eligible=False,annotation_start=u.get("start"),annotation_end=u.get("end"))
        repaired=None;override=None
        try:
            data=path.read_bytes();sha=hashlib.sha256(data).hexdigest()
            duration=float(u["end"])-float(u["start"])
            audit.update(raw_sha256=sha,nbytes=len(data),annotation_duration_s=duration,
                         duration_at_16k=len(data)/32000,duration_at_48k=len(data)/96000)
            problem=rate_check(uid,len(data),duration,data[:4])
            if problem:
                audit.update(status="HOLD",reason=problem);return audit,None,None
            x=np.frombuffer(data,dtype="<i2").astype(np.float32)/32768
            if not np.any(x):
                audit.update(status="HOLD",reason="all_zero_waveform");return audit,None,None
            audit.update(status="CONSISTENT_WITH_REVIEWED_48K_SESSION",
                         rate_evidence="session_listening_plus_file_consistency_not_header")
            if uid not in candidates:
                audit["materialization"]="not_in_existing_training_candidates"
                return audit,None,None
            r=candidates[uid]
            if Path(r["audio"]["path"]).resolve()!=path.resolve() or any(r["audio"].get(k) is not None for k in ("offset_s","duration_s")):
                raise ValueError("Candidate is not the exact standalone PCM")
            key=digest([RECIPE,r["key"],sha,evidence_id])
            target=a.out/"audio"/(key+".wav")
            wave=soxr.resample(x,48000,16000)
            sf.write(target,wave,16000,subtype="FLOAT")
            repaired=copy.deepcopy(r)
            repaired.update(key=key,source_key=r["key"],repair_recipe=RECIPE,
                repair_evidence_sha256=evidence_id,original_audio=r["audio"],
                source_pcm_format=dict(sample_rate=48000,channels=1,sample_format="s16le",sha256=sha),
                original_timeline=dict(kind=r["timeline"],start_s=r["start_s"],end_s=r["end_s"],duration_s=r["duration_s"]),
                annotation_interval=dict(start_s=u["start"],end_s=u["end"]),
                crop_origin_in_dialogue_s=None,timeline="isolated_clip",start_s=0,
                end_s=len(wave)/16000,duration_s=len(wave)/16000,
                audio=dict(path=str(target),offset_s=None,duration_s=None),
                human_review=labels.get(uid),training_eligible=False,
                invalidate=["alignment","features","teacher_outputs","sequences"],
                review_cell=["2022", "REPAIRED_SESSION_SCOPE", "explicit-48000Hz", r["stratum"]])
            if uid in labels and labels[uid]["labels"]["text"]!="pass":
                repaired["reasons"].append("human_text_review_hold")
            _,info=decode(repaired)
            repaired["audio_qc"]=dict(key=key,source=r["source"],audio=info,
                status="AUDIO_REVIEW" if info["review"] else "AUDIO_OK_TEXT_PENDING",training_eligible=False)
            override=dict(source_key=r["key"],utt_id=uid,source_path=str(path),source_sha256=sha,
                sample_rate=48000,channels=1,sample_format="s16le",evidence_sha256=evidence_id,
                repaired_key=key,repaired_path=str(target),waveform_sha256=info["sha256"],
                training_eligible=False)
            audit["materialization"]="new_16k_wav_sidecar"
        except Exception as e:
            audit.update(status="HOLD",reason=f"{type(e).__name__}: {e}")
            repaired=override=None
        return audit,repaired,override

    counts=Counter();per_session={s:Counter() for s in SESSIONS};n=0
    with ThreadPoolExecutor(max_workers=a.workers) as pool, (a.out/"session-files.jsonl").open("x") as af, (a.out/"repaired.jsonl").open("x") as rf, (a.out/"rate-overrides.jsonl").open("x") as of:
        for audit,r,override in pool.map(inspect,inputs):
            af.write(json.dumps(audit,ensure_ascii=False)+"\n")
            counts[audit["status"]]+=1;per_session[audit["session"]][audit["status"]]+=1
            if r:
                rf.write(json.dumps(r,ensure_ascii=False)+"\n");of.write(json.dumps(override)+"\n");n+=1
    summary=dict(complete=True,evidence=evidence,evidence_sha256=evidence_id,
        counts=dict(counts),per_session=per_session,unannotated_files=extras,
        existing_candidates=len(candidates),repaired_rows=n,training_eligible=0,
        repaired_sha256=file_digest(a.out/"repaired.jsonl"),
        original_manifest_unchanged=file_digest(meta["path"])==meta["sha256"])
    (a.out/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":main()

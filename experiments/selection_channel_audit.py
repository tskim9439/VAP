#!/usr/bin/env python3
"""Full audit of unresolved multichannel QC rows. Only identity supports ch0 reuse."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import io
import itertools
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import file_digest
from experiments.selection_review_audit import joined


def main():
    import numpy as np
    import soundfile as sf
    import soxr
    from vapasr.data.archive import read_member
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection",required=True,type=Path)
    ap.add_argument("--qc",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--workers",type=int,default=32)
    a=ap.parse_args()
    if not 1<=a.workers<=32:raise ValueError("workers must be 1..32")
    rows=[];fingerprints={}
    for source in ("aihub-bc-train","ami"):
        marker=json.loads((a.qc/f"{source}.done.json").read_text())
        if file_digest(marker["output"])!=marker["output_sha256"]:
            raise ValueError("QC checksum mismatch")
        cp=a.selection/f"{source}.candidates.jsonl.gz"
        fingerprints[source]=dict(candidates_sha256=file_digest(cp),qc_sha256=marker["output_sha256"])
        counts=Counter()
        with gzip.open(cp,"rt") as cf,gzip.open(marker["output"],"rt") as qf:
            for r,q in joined(map(json.loads,cf),map(json.loads,qf)):
                counts[q["status"]]+=1
                if "multichannel_without_explicit_channel_mapping" in q.get("audio",{}).get("review",[]):
                    rows.append((r,q))
        if counts!=Counter(marker["counts"]):raise ValueError("QC count mismatch")
    a.out.mkdir(parents=True,exist_ok=False)

    def inspect(pair):
        r,q=pair;ref=r["audio"]
        answer=dict(key=r["key"],source=r["source"],audio=ref,training_eligible=False)
        try:
            path=ref["path"]
            if "#ch" in path:raise ValueError("Unexpected explicit channel")
            source=io.BytesIO(read_member(*path.split("::",1))) if "::" in path else path
            with sf.SoundFile(source) as f:
                sr=f.samplerate;channels=f.channels;subtype=f.subtype;full_duration=f.frames/sr
                start=round((ref.get("offset_s") or 0)*sr)
                n=round(ref["duration_s"]*sr) if ref.get("duration_s") is not None else f.frames-start
                if start<0 or n<=0 or start+n>f.frames:raise ValueError("Crop out of bounds")
                f.seek(start);x=f.read(n,dtype="float32",always_2d=True)
            if len(x)!=n or channels<2 or not np.isfinite(x).all():raise ValueError("Invalid multichannel data")
            first=x[:,0]
            if sr!=16000:first=soxr.resample(first,sr,16000)
            sha=hashlib.sha256(np.ascontiguousarray(first,dtype='<f4').tobytes()).hexdigest()
            if sha!=q["audio"]["sha256"]:raise ValueError("QC waveform changed")
            equal=all(np.array_equal(x[:,0],x[:,c]) for c in range(1,channels))
            answer.update(status="IDENTICAL_CHANNELS" if equal else "DISTINCT_CHANNELS_REVIEW",
                source_sample_rate=sr,channels=channels,subtype=subtype,
                file_duration_s=full_duration,crop_duration_s=len(x)/sr,
                declared_duration_s=r["duration_s"],waveform_sha256=sha,
                channel_rms=np.sqrt(np.mean(x.astype(np.float64)**2,axis=0)).tolist(),
                max_abs_channel_difference=max(float(np.max(np.abs(x[:,0]-x[:,c]))) for c in range(1,channels)),
                explicit_ch0_candidate=(path+"#ch0") if equal else None,
                action="same_waveform_explicit_ch0_candidate" if equal else "human_channel_review_required")
        except Exception as e:answer.update(status="HOLD",error=f"{type(e).__name__}: {e}")
        return answer
    counts={s:Counter() for s in fingerprints}
    with ThreadPoolExecutor(max_workers=a.workers) as pool,(a.out/"channels.jsonl").open("x") as f:
        for i,r in enumerate(pool.map(inspect,rows),1):
            f.write(json.dumps(r,ensure_ascii=False)+"\n");counts[r["source"]][r["status"]]+=1
            if i%256==0:print(json.dumps(dict(processed=i,total=len(rows))),flush=True)
    summary=dict(complete=True,total=len(rows),sources=counts,fingerprints=fingerprints,
        script_sha256=file_digest(Path(__file__).resolve()),output_sha256=file_digest(a.out/"channels.jsonl"),
        training_eligible=0,applied_changes=0)
    (a.out/"summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=="__main__":main()

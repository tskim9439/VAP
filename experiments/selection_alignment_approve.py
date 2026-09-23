#!/usr/bin/env python3
"""Record human spot-check attestation and approve the automatic QC set."""
import argparse
import datetime
import html.parser
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.selection import digest, file_digest


class Cards(html.parser.HTMLParser):
    def __init__(self): super().__init__(); self.keys=[]
    def handle_starttag(self, tag, attrs):
        if tag == "article":
            value=dict(attrs).get("data-key")
            if value:self.keys.append(value)


def save(path,value):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+"\n",encoding="utf-8");tmp.replace(path)


def make_review(a):
    parser=Cards();parser.feed(a.html.read_text(encoding="utf-8"))
    if not parser.keys or len(parser.keys)!=len(set(parser.keys)):raise ValueError("invalid review HTML keys")
    review=dict(schema="qwen-verbatim-alignment-human-review-v1",
        reviewed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer=a.reviewer,
        attestation=a.attestation,source_html=str(a.html.resolve()),source_html_sha256=file_digest(a.html),
        training_eligible=False,rows=[dict(key=k,label="pass",note="") for k in sorted(parser.keys)])
    save(a.out,review);print(f"REVIEW_WRITTEN rows={len(parser.keys)} {a.out}")


def approve(a):
    qc=a.qc.resolve();review=json.loads(a.review.read_text());summary=json.loads((qc/"summary.json").read_text())
    parser=Cards();parser.feed((qc/"human-review/index.html").read_text(encoding="utf-8"))
    expected=set(parser.keys);rows=review["rows"];keys=[r["key"] for r in rows]
    if (review.get("schema")!="qwen-verbatim-alignment-human-review-v1" or set(keys)!=expected
            or len(keys)!=len(expected) or any(r.get("label")!="pass" for r in rows)
            or len(keys)!=summary["acoustic_spot_check_sample_n"]):
        raise ValueError("review does not attest PASS for the exact QC sample")
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    config=dict(schema="qwen-verbatim-training-approval-v1",qc_root=str(qc),
        qc_summary_sha256=file_digest(qc/"summary.json"),qc_fingerprint=summary["fingerprint"],
        review_sha256=file_digest(a.review),reviewer=review["reviewer"],reviewed_at=review["reviewed_at"],
        attestation=review["attestation"],policy=dict(include=["ACCEPT_ORIGINAL","ACCEPT_CLAMPED_80MS"],
            exclude=["REVIEW_80_320MS","REJECT_GT_320MS"]),code_sha256=file_digest(Path(__file__).resolve()))
    config["fingerprint"]=digest(config);save(out/"approval.json",config)
    with (out/"training-inputs.jsonl").open("w") as dst:
        for line in (qc/"accepted-inputs.jsonl").open():
            row=json.loads(line);row.update(approval_fingerprint=config["fingerprint"],training_eligible=True)
            dst.write(json.dumps(row,ensure_ascii=False)+"\n")
    result=dict(complete=True,training_eligible=True,approval_fingerprint=config["fingerprint"],
        rows=summary["automatic_accept"],excluded_review=summary["deferred_review_rows"],
        excluded_reject=summary["rejected_rows"],human_review_rows=len(keys),
        training_inputs_sha256=file_digest(out/"training-inputs.jsonl"))
    save(out/"summary.json",result);print(json.dumps(result,ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest="mode",required=True)
    m=sub.add_parser("make-review");m.add_argument("--html",type=Path,required=True);m.add_argument("--out",type=Path,required=True)
    m.add_argument("--reviewer",required=True);m.add_argument("--attestation",required=True)
    p=sub.add_parser("approve");p.add_argument("--qc",type=Path,required=True);p.add_argument("--review",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
    a=ap.parse_args();make_review(a) if a.mode=="make-review" else approve(a)


if __name__=="__main__":main()

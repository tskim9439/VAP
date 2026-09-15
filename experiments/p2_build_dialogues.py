#!/usr/bin/env python
"""Phase 2 Q0 — 코퍼스 → dialogues.jsonl (정본 output-phase2-lane-plan §8, task-phase2-data-prep 순서 2–3).

한 줄 = Dialogue(JSON): 화자별 채널 참조·발화(raw, TN text, flags)·시각(라벨값). 정렬(p2_align.py)은 이 파일을 읽어 tokens 를 채운다.
  python experiments/p2_build_dialogues.py --corpus aihub71631 --audio-root <71631_audio/Training> --label-root <labels/aihub71631> --out <dir>
  python experiments/p2_build_dialogues.py --corpus aihub134-1 --crop-dir <…/134-1…/Training/TS_02.실외> --label-root <labels/TL_02.실외> --out <dir>
  python experiments/p2_build_dialogues.py --corpus otoSpeech --root <otoSpeech16k> --out <dir>
  python experiments/p2_build_dialogues.py --corpus ami --root /soundai/DB/raw/ami --annotations …/ami_public_manual_1.6.2.zip --out <dir>
  python experiments/p2_build_dialogues.py --corpus notsofar --root /soundai/DB/raw/notsofar --out <dir>
  python experiments/p2_build_dialogues.py --corpus icsi --root /soundai/DB/raw/icsi --annotations …/ICSI_core_NXT.zip --transcripts …/ICSI_original_transcripts.zip --out <dir>
출력: <out>/<corpus>.dialogues.jsonl + <out>/<corpus>.stats.json (대화·시간·발화·quarantine·결손). 읽기 전용 입력, 서버 쓰기는 --out 아래만."""
import os, sys, json, glob, argparse, time, collections, unicodedata
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.dialogue import Dialogue
from vapasr.data.textnorm import target, target_flags, TEXTNORM_ID_SHORT as TEXTNORM_ID
from vapasr.data import dialogue_corpora as C

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True, choices=["aihub71631", "aihub134-1", "aihub134-2", "otoSpeech", "ami", "notsofar", "icsi"])
ap.add_argument("--root"); ap.add_argument("--audio-root"); ap.add_argument("--label-root"); ap.add_argument("--crop-dir")
ap.add_argument("--annotations"); ap.add_argument("--transcripts"); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int)
ap.add_argument("--split", default="train"); ap.add_argument("--workers", type=int, default=1)
a = ap.parse_args()

def nfc(s): return unicodedata.normalize("NFC", s)
TN_CORPUS = {"aihub71631": "aihub71631", "aihub134-1": "aihub71631", "aihub134-2": "aihub71631"}

def gen():
    c = a.corpus
    if c == "aihub71631":
        wavs = {os.path.splitext(os.path.basename(p))[0]: p for p in glob.glob(os.path.join(a.audio_root, "**", "*.wav"), recursive=True)}
        for jp in sorted(glob.glob(os.path.join(a.label_root, "**", "*.json"), recursive=True)):
            stem = os.path.splitext(os.path.basename(jp))[0]
            if stem in wavs: yield C.load_aihub_stereo(wavs[stem], jp, a.split)
    elif c in ("aihub134-1", "aihub134-2"):
        yield from C.iter_aihub_crops(a.label_root, a.crop_dir, c, a.split)
    elif c == "otoSpeech":
        for d in sorted(glob.glob(os.path.join(a.root, "*"))):
            if os.path.isfile(os.path.join(d, "metadata.json")): yield C.load_otospeech_dialogue(d, a.split)
    elif c == "ami":
        ann = C.AmiAnnotations(a.annotations)
        for m in sorted(ann.meetings):
            if os.path.isdir(os.path.join(a.root, m)): yield ann.load(m, a.root, a.split)
    elif c == "notsofar":
        for gt in sorted(glob.glob(os.path.join(a.root, "**", "gt_transcription.json"), recursive=True)):
            sub = "eval" if "eval" in gt else ("dev" if "dev" in gt else "train")
            yield C.load_notsofar(os.path.dirname(gt), sub)
    elif c == "icsi":
        ann = C.IcsiAnnotations(a.annotations, a.transcripts)
        for m in sorted({x.split("/")[2].split(".")[0] for x in ann.members if x.startswith("ICSI/Segments/") and x.endswith(".segs.xml")}):
            if os.path.isdir(os.path.join(a.root, m)): yield ann.load(m, a.root, a.split)

os.makedirs(a.out, exist_ok=True); outp = os.path.join(a.out, f"{a.corpus}.dialogues.jsonl"); tmp = outp + ".tmp"
st = collections.Counter(); hours = 0.0; flags = collections.Counter(); t0 = time.time(); n = 0
with open(tmp, "w", encoding="utf-8") as f:
    for d in gen():
        for u in d.utterances:
            u.text = target(u.raw, d.lang, TN_CORPUS.get(a.corpus)); fl = sorted(target_flags(u.text, d.lang, raw=u.raw, corpus=TN_CORPUS.get(a.corpus)))
            if fl: u.flags = fl; u.text = ""; st["quarantined"] += 1
            for x in fl: flags[x] += 1
            st["utts"] += 1
        st["dialogues"] += 1; hours += d.duration_s / 3600; st["speakers_total"] += len(d.speakers); st["missing_pieces"] += len(d.meta.get("missing", []))
        st[f"n_speakers={len(d.speakers)}"] += 1
        f.write(d.to_json() + "\n"); n += 1
        if a.limit and n >= a.limit: break
        if n % 200 == 0: print(f"  {n} dialogues {hours:.1f} h {time.time()-t0:.0f}s", flush=True)
os.replace(tmp, outp)
stats = dict(st, hours=round(hours, 2), textnorm=TEXTNORM_ID, flags=dict(flags), args=vars(a), sec=round(time.time() - t0, 1))
json.dump(stats, open(os.path.join(a.out, f"{a.corpus}.stats.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(stats, ensure_ascii=False))

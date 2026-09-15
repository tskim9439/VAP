#!/usr/bin/env python
"""Phase 2 Q0 — 서버에서 코퍼스 로더 스모크(각 1 건): 경로·형식·화자 수·발화 수·시간을 확인한다. GPU·torch 불필요(numpy·soundfile 만).
  python experiments/p2_smoke_loaders.py [ami notsofar icsi aihub134-1 aihub71631 otoSpeech]"""
import os, sys, json, glob, zipfile, unicodedata, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data import dialogue_corpora as C
NIA = "/soundai/databricks_build_managed/1baf7241-0193-4ef8-a79a-b892a4cc792f/asr_db/korean_16kHz/NIA24"; LAB = "/soundai/users/tskim/VAPKT-data/data/labels/aihub71631"
def nfc(s): return unicodedata.normalize("NFC", s)
def find_dir(p, n):
    for d in os.listdir(p):
        if nfc(d) == nfc(n): return os.path.join(p, d)
    raise FileNotFoundError(f"{p}/{n}")
def summary(d):
    return dict(conv_id=d.conv_id, corpus=d.corpus, lang=d.lang, duration_s=round(d.duration_s, 1), speakers=d.speakers, n_utts=len(d.utterances), channels={k: (v.path or f"pieces:{len(v.pieces or [])}") for k, v in d.channels.items()},
                first_utts=[(u.speaker, u.start, u.end, u.raw[:40]) for u in sorted(d.utterances, key=lambda u: u.start)[:3]], meta={k: v for k, v in d.meta.items() if k != "channels"})
want = sys.argv[1:] or ["ami", "notsofar", "icsi", "aihub134-1", "aihub71631", "otoSpeech"]
for w in want:
    print(f"== {w}", flush=True)
    try:
        if w == "ami":
            ann = C.AmiAnnotations("/soundai/DB/raw/ami/annotations/ami_public_manual_1.6.2.zip"); m = sorted(x for x in ann.meetings if os.path.isdir(f"/soundai/DB/raw/ami/{x}"))[0]
            print(json.dumps(summary(ann.load(m, "/soundai/DB/raw/ami")), ensure_ascii=False))
        elif w == "notsofar":
            gt = sorted(glob.glob("/soundai/DB/raw/notsofar/*/benchmark-datasets/*/*/MTG/MTG_*/gt_transcription.json"))[0]
            print(json.dumps(summary(C.load_notsofar(os.path.dirname(gt), "dev")), ensure_ascii=False))
        elif w == "icsi":
            z = zipfile.ZipFile("/soundai/DB/raw/icsi/annotations/ICSI_core_NXT.zip"); names = z.namelist()
            print("core zip top dirs:", sorted({n.split("/")[1] for n in names if n.count("/") >= 1 and n.startswith("ICSI/")})[:30])
            print("words sample:", [n for n in names if "/Words/" in n][:3]); t = zipfile.ZipFile("/soundai/DB/raw/icsi/annotations/ICSI_original_transcripts.zip")
            mrt = [n for n in t.namelist() if n.endswith(".mrt")]; print("mrt sample:", mrt[:2]); print(t.read(mrt[0])[:1200].decode("utf-8", "replace"))
            ann = C.IcsiAnnotations("/soundai/DB/raw/icsi/annotations/ICSI_core_NXT.zip", "/soundai/DB/raw/icsi/annotations/ICSI_original_transcripts.zip")
            m = sorted(x.split("/")[2].split(".")[0] for x in ann.members if x.startswith("ICSI/Segments/") and x.endswith(".segs.xml") and os.path.isdir(f"/soundai/DB/raw/icsi/{x.split('/')[2].split('.')[0]}"))[0]
            print("channel_map:", ann.channel_map(m)); print(json.dumps(summary(ann.load(m, "/soundai/DB/raw/icsi")), ensure_ascii=False))
        elif w == "aihub134-1":
            cd = find_dir(f"{NIA}/134-1_Emotion_Conv_Adult/Training", "TS_02.실외"); lr = find_dir(LAB, "TL_02.실외")
            d = next(C.iter_aihub_crops(lr, cd, "aihub134-1", limit=1)); print(json.dumps(summary(d), ensure_ascii=False))
        elif w == "aihub71631":
            wav = sorted(glob.glob("/soundai/DB/raw/aihub/71631_audio/Training/**/*.wav", recursive=True))[0]; stem = os.path.splitext(os.path.basename(wav))[0]
            js = [p for p in glob.glob(f"{LAB}/**/{stem}.json", recursive=True)][0]; print(json.dumps(summary(C.load_aihub_stereo(wav, js)), ensure_ascii=False))
        elif w == "otoSpeech":
            d = sorted(x for x in glob.glob("/soundai/DB/raw/otoSpeech16k/*") if os.path.isfile(os.path.join(x, "metadata.json")))[0]
            print(json.dumps(summary(C.load_otospeech_dialogue(d)), ensure_ascii=False))
    except Exception:
        traceback.print_exc()
print("SMOKE_DONE")

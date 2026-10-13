#!/usr/bin/env python
"""Phase 2 Q0 — 코퍼스 → dialogues.jsonl (정본 output-phase2-lane-plan §8, task-phase2-data-prep 순서 2–3).

한 줄 = Dialogue(JSON): 화자별 채널 참조·발화(raw, TN text, flags)·시각(라벨값). 정렬(p2_align.py)은 이 파일을 읽어 tokens 를 채운다.
  python experiments/p2_build_dialogues.py --corpus aihub71631 --audio-root <71631_audio/Training> --label-root <labels/aihub71631> --out <dir>
  python experiments/p2_build_dialogues.py --corpus aihub134-1 --crop-dir <…/134-1…/Training/TS_02.실외> --label-root <labels/TL_02.실외> --out <dir>
  python experiments/p2_build_dialogues.py --corpus otoSpeech --root <otoSpeech16k> --out <dir>
  python experiments/p2_build_dialogues.py --corpus ami --root /soundai/DB/raw/ami --annotations …/ami_public_manual_1.6.2.zip --out <dir>
  python experiments/p2_build_dialogues.py --corpus notsofar --root /soundai/DB/raw/notsofar --out <dir>
  python experiments/p2_build_dialogues.py --corpus icsi --root /soundai/DB/raw/icsi --annotations …/ICSI_core_NXT.zip --transcripts …/ICSI_original_transcripts.zip --out <dir>
  python experiments/p2_build_dialogues.py --corpus kec --root <VAPKT-data/data/KEC> --out <dir>      # 독일어(Stage 3, 2026-10-14)
  python experiments/p2_build_dialogues.py --corpus ztc --root <VAPKT-data/data/ZTC> --out <dir>
    독일어는 두 번 돈다: (1) 위처럼 빌드 → p2_asr_check → (2) 같은 명령에 --asr-guide <asrcheck>/<corpus>.utts.jsonl 을 붙여 다시 빌드(구어 축약 표기 → 표준 철자,
    C.canon_de). 축약 그대로의 1 차 산출은 <corpus>.verbatim.dialogues.jsonl 로 복사해 둔다.
출력: <out>/<corpus>.dialogues.jsonl + <out>/<corpus>.stats.json (대화·시간·발화·quarantine·결손). 읽기 전용 입력, 서버 쓰기는 --out 아래만."""
import os, sys, json, glob, argparse, time, collections, unicodedata, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data.dialogue import Dialogue
from vapasr.data.textnorm import target, target_flags, TEXTNORM_ID_SHORT as TEXTNORM_ID
from vapasr.data.textnorm_de import target_de, target_flags_de, TEXTNORM_DE_ID
from vapasr.data import dialogue_corpora as C

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True, choices=["aihub71631", "aihub134-1", "aihub134-2", "otoSpeech", "ami", "notsofar", "icsi", "chime6", "nikl2020", "turnbench", "kec", "ztc"])
ap.add_argument("--year", default="2020", help="nikl: 연도"); ap.add_argument("--index-cache", default=None, help="nikl: index_year 캐시 디렉토리")
ap.add_argument("--root"); ap.add_argument("--audio-root"); ap.add_argument("--label-root"); ap.add_argument("--crop-dir")
ap.add_argument("--annotations"); ap.add_argument("--transcripts"); ap.add_argument("--out", required=True); ap.add_argument("--limit", type=int)
ap.add_argument("--split", default="train"); ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--asr-guide", help="독일어: p2_asr_check 의 <corpus>.utts.jsonl — 구어 축약 표기를 같은 발화의 ASR 가설로 표준 철자로 바꾸고(C.canon_de), "
                "바뀐 전사로 다시 잰 오류율을 <같은 이름>.canon.utts.jsonl 로 쓴다(p2_refine --asr-flags 에 준다)")
a = ap.parse_args()

def nfc(s): return unicodedata.normalize("NFC", s)
TN_CORPUS = {"aihub71631": "aihub71631", "aihub134-1": "aihub71631", "aihub134-2": "aihub71631", "otoSpeech": "yodas", "ami": "yodas", "notsofar": "yodas", "icsi": "yodas", "chime6": "yodas", "nikl2020": "nikl", "turnbench": "yodas"}   # EN 대화 코퍼스는 숫자 표기가 있어 NUMERIC_CORPORA_EN 규칙("yodas")으로 숫자를 말로 푼다(textnorm 자체는 동결)

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
    elif c == "chime6":                                              # held-out 평가용: --root <audio dir with S02_P05.wav …> --transcripts <transcriptions/<split> dir> --split dev
        for jp in sorted(glob.glob(os.path.join(a.transcripts, "*.json"))):
            yield C.load_chime6(os.path.splitext(os.path.basename(jp))[0], jp, a.root, a.split)
    elif c == "turnbench":                                          # held-out 평가용: --root <parquet dir(dev/data)> --audio-root <flac 풀 곳> --split heldout
        yield from C.iter_turnbench(a.root, a.audio_root, a.split)
    elif c == "nikl2020":                                            # held-out 평가용: --root /soundai/DB/raw/nikl --year 2020 --index-cache <dir> [--limit N]
        from vapasr.data.nikl import index_year
        idx = index_year(a.root, a.year, cache_dir=a.index_cache); rng = random.Random(0)
        keys = sorted(k for k in idx if k.startswith("json:")); rng.shuffle(keys)
        for k in keys:
            d = C.load_nikl_dialogue(idx[k], idx, a.split, corpus=f"nikl{a.year}")
            if d is not None and len(d.speakers) >= 2: yield d
    elif c == "kec":
        yield from C.iter_kec(a.root, a.split)
    elif c == "ztc":
        yield from C.iter_ztc(a.root, a.split)
    elif c == "icsi":
        ann = C.IcsiAnnotations(a.annotations, a.transcripts)
        for m in sorted({x.split("/")[2].split(".")[0] for x in ann.members if x.startswith("ICSI/Segments/") and x.endswith(".segs.xml")}):
            if os.path.isdir(os.path.join(a.root, m)): yield ann.load(m, a.root, a.split)

os.makedirs(a.out, exist_ok=True); outp = os.path.join(a.out, f"{a.corpus}.dialogues.jsonl"); tmp = outp + ".tmp"
st = collections.Counter(); hours = 0.0; flags = collections.Counter(); t0 = time.time(); n = 0
guide = {}; canon = collections.Counter(); grows = []
if a.asr_guide:
    for l in open(a.asr_guide, encoding="utf-8"):
        r = json.loads(l); guide[(r["conv"], r["utt"])] = r
with open(tmp, "w", encoding="utf-8") as f:
    for d in gen():
        for u in d.utterances:
            if d.lang == "German":                                     # 로더가 붙인 사유(말 끊김·알아듣기 어려움 등)를 TN flags 와 합친다
                u.text = target_de(u.raw); fl = sorted(set(u.flags or []) | target_flags_de(u.raw))
            else:                                                      # 기존 코퍼스는 이전 동작 그대로(로더 flags 는 TN flags 가 있을 때만 덮인다)
                u.text = target(u.raw, d.lang, TN_CORPUS.get(a.corpus)); fl = sorted(target_flags(u.text, d.lang, raw=u.raw, corpus=TN_CORPUS.get(a.corpus)))
            if fl: u.flags = fl; u.text = ""; st["quarantined"] += 1
            elif a.asr_guide and d.lang == "German":                  # 축약 표기 → 표준 철자(ASR 가설로 모호한 것을 고른다)
                g = guide.get((d.conv_id, u.utt_id)); hyp = target_de(g["hyp"]) if g else None; old = u.text
                u.text, cs = C.canon_de(u.text, hyp, C.DE_CORPUS_CANON.get(a.corpus)); canon.update(cs); canon["utts_changed"] += u.text != old
                if g: grows.append(dict(conv=d.conv_id, utt=u.utt_id, dur=g["dur"], ref=u.text, hyp=g["hyp"], err=round(C.de_word_err(u.text, hyp), 3), err_verbatim=g["err"]))
            for x in fl: flags[x] += 1
            st["utts"] += 1
        st["dialogues"] += 1; hours += d.duration_s / 3600; st["speakers_total"] += len(d.speakers); st["missing_pieces"] += len(d.meta.get("missing", []))
        st[f"n_speakers={len(d.speakers)}"] += 1
        f.write(d.to_json() + "\n"); n += 1
        if a.limit and n >= a.limit: break
        if n % 200 == 0: print(f"  {n} dialogues {hours:.1f} h {time.time()-t0:.0f}s", flush=True)
os.replace(tmp, outp)
if a.asr_guide:
    gp = a.asr_guide.replace(".utts.jsonl", ".canon.utts.jsonl"); assert gp != a.asr_guide
    with open(gp + ".tmp", "w", encoding="utf-8") as f:
        for r in grows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(gp + ".tmp", gp)
    e = sorted(r["err"] for r in grows); canon.update(guided_rows=len(grows), err_median=e[len(e) // 2] if e else None, share_err_ge_0_2=round(sum(x >= 0.2 for x in e) / max(1, len(e)), 4))
stats = dict(st, hours=round(hours, 2), textnorm=(TEXTNORM_DE_ID if a.corpus in ("kec", "ztc") else TEXTNORM_ID), flags=dict(flags), canon=dict(canon), args=vars(a), sec=round(time.time() - t0, 1))
json.dump(stats, open(os.path.join(a.out, f"{a.corpus}.stats.json"), "w"), indent=1, ensure_ascii=False); print(json.dumps(stats, ensure_ascii=False))

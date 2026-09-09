#!/usr/bin/env python
"""infer_demo 의 비대화형 판: 표본 오디오(또는 --audio) 를 hf 모델로 디코드해 후처리 텍스트·원시 시퀀스·청크 표를 출력.
  CUDA_VISIBLE_DEVICES=5 python experiments/infer_run.py --ckpt /soundai/Model/VAPASR/hf-C2/final [--audio a.wav --lang Korean]"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--audio", default=None); ap.add_argument("--lang", default="Korean"); ap.add_argument("--delay", type=int, default=2); ap.add_argument("--idx", type=int, default=3)
a = ap.parse_args()
from vapasr.hf.infer import load_model, load_audio, transcribe, iter_stream
from vapasr.data.streams import read_streams
from vapasr.data.textnorm import score_en, score_ko
import jiwer, pandas as pd
pd.set_option("display.max_colwidth", 60); pd.set_option("display.width", 220)
model, tok = load_model(a.ckpt)
samples = [(a.audio, a.lang, None)] if a.audio else []
if not samples:
    M = os.environ["MXC_DATA_MANIFEST_DIR"]
    r = read_streams(os.path.join(M, "kspon-dev"), mode="utt")[a.idx]; samples.append((r["segments"][0]["path"], "Korean", " ".join(s["lexical_text"] for s in r["segments"])))
    r = read_streams(os.path.join(M, "librispeech-dev"), mode="stream", subset="dev-clean")[a.idx]; samples.append((r["segments"][0]["path"] if len(r["segments"]) == 1 else None, "English", " ".join(s["lexical_text"] for s in r["segments"]), r))
for s in samples:
    path, lang, ref = s[0], s[1], s[2]
    if path is None:                                                        # 여러 세그먼트 스트림 → manifest 규약대로 조립
        from vapasr.data.streams import assemble_stream; row = s[3]; wav = assemble_stream(dict(id=row["id"], duration_s=row["duration_s"], segments=row["segments"])); path = f"{row['id']} ({len(row['segments'])} utts, {row['duration_s']:.1f} s)"
    else: wav = load_audio(path)
    print("\n" + "=" * 100); print(f"입력: {path} · {len(wav)/16000:.2f} s · lang {lang} · δ={a.delay}")
    if ref: print("참조 :", ref)
    res = transcribe(model, tok, wav, lang=lang, delay=a.delay)
    print("후처리 텍스트 :", res.text(tok)); print("디코드 원문   :", res.text(tok, normalize=False)); print("원시 시퀀스   :", res.raw(tok)[:1500]); print("통계 :", json.dumps(res.stats(), ensure_ascii=False))
    if ref:
        rn = score_ko(ref, True) if lang == "Korean" else score_en(ref); hn = res.text(tok)
        print(("CER" if lang == "Korean" else "WER"), round(jiwer.cer(rn, hn) if lang == "Korean" else jiwer.wer(rn, hn), 4))
    print(res.table(tok).to_string(index=False, max_colwidth=60)[:6000])
    print("-- 청크 단위(방출 청크만):")
    acc = []
    for c in iter_stream(model, tok, wav, lang=lang, delay=a.delay):
        if c.ids or c.k >= res.K: acc += c.ids; print(f"  [{c.k:4d}] {c.t1*1000:6.0f} ms  +{tok.decode(c.ids)!r:26s} tick {c.tick_ms:5.1f} ms → {tok.decode(acc)[:90]}")

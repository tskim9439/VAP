#!/usr/bin/env python
"""Stage 1 대조군 — 같은 세트·같은 채점 규약으로 Nemotron RNN-T [56,0](80 ms 스트리밍, 관문 분모)와 Qwen3-ASR 오프라인(참고선)을 잰다.
plans/stage1-mono-pilot.md §5·§7.1.  eval_other 는 서버 결손분 그대로(n=2687) — 우리 모델과 같은 표본.

python experiments/s1_baselines.py [--sets dev|test|all] [--cap-stream 200 --cap-utt 1000 | --all] [--models nemotron,qwen]
출력: $MXC_CKPT_EXP_DIR/uslm/s1-baselines/baselines-<sets>.json  (세트별 WER / CER-official / CER-nospace, n)
오디오는 우리 파이프라인과 동일하게 조립한다(vapasr.data.streams.assemble_stream) — 무음 삽입까지 같은 입력.
"""
import os, sys, json, glob, time, argparse, subprocess, tempfile, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--sets", default="all"); ap.add_argument("--cap-stream", type=int, default=200); ap.add_argument("--cap-utt", type=int, default=1000); ap.add_argument("--all", action="store_true")
ap.add_argument("--models", default="nemotron,qwen"); ap.add_argument("--gpu", default=None); ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    if a.gpu is None:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        a.gpu = str(max([[int(v) for v in l.split(",")] for l in q.stdout.strip().splitlines()], key=lambda r: r[2] - r[1])[0])
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
import numpy as np, torch, soundfile as sf, jiwer
from vapasr.data.streams import read_streams, assemble_stream
from vapasr.data.textnorm import score_en, score_ko
MAN = os.environ.get("MXC_DATA_MANIFEST_DIR", os.environ.get("DATA_MANIFEST_DIR", "/tmp"))
out = os.path.join(os.environ.get("MXC_CKPT_EXP_DIR", os.environ.get("CKPT_EXP_DIR", "/tmp")), "uslm", "s1-baselines"); os.makedirs(out, exist_ok=True)
DEV = [("dev-clean", "librispeech-dev", "dev-clean", "stream"), ("dev-other", "librispeech-dev", "dev-other", "stream"), ("kspon-dev", "kspon-dev", "dev", "utt")]
TEST = [("test-clean/stream", "librispeech-test", "test-clean", "stream"), ("test-clean/utt", "librispeech-test", "test-clean", "utt"),
        ("test-other/stream", "librispeech-test", "test-other", "stream"), ("test-other/utt", "librispeech-test", "test-other", "utt"),
        ("eval_clean", "kspon-eval", "eval_clean", "utt"), ("eval_other-partial[E03314-E06000,n=2687]", "kspon-eval", "eval_other", "utt")]
spec = DEV if a.sets == "dev" else TEST if a.sets == "test" else DEV + TEST
sets = {}
for lab, m, sub, mode in spec:
    rows = read_streams(os.path.join(MAN, m), mode=mode, subset=sub); random.Random(a.seed).shuffle(rows)
    if not a.all: rows = rows[: (a.cap_stream if mode == "stream" else a.cap_utt)]
    sets[lab] = rows
print("sets: " + ", ".join(f"{k}:{len(v)}" for k, v in sets.items()), flush=True)
tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
def ref_text(r): return " ".join(s["text"] for s in r["segments"])
def score(lang, R, H):
    if lang == "Korean": return dict(cer_official=jiwer.cer([score_ko(x, True) for x in R], [score_ko(x, True) for x in H]), cer_nospace=jiwer.cer([score_ko(x, False) for x in R], [score_ko(x, False) for x in H]))
    return dict(wer=jiwer.wer([score_en(x) for x in R], [score_en(x) for x in H]))

models = a.models.split(","); res = {}
if "nemotron" in models:
    import nemo.collections.asr as nemo_asr
    nm = nemo_asr.models.ASRModel.restore_from(glob.glob(os.path.join(os.environ["MXC_NEMOTRON_DIR"], "*.nemo"))[0], map_location="cpu").cuda().eval()
    nm.encoder.set_default_att_context_size([56, 0]); print("nemotron [56,0] 로드", flush=True)
    def nemo_tx(x, lang):
        sf.write(tmp, x, 16000); mf = tmp + ".json"; open(mf, "w").write(json.dumps({"audio_filepath": tmp, "duration": len(x) / 16000, "text": "", "lang": "ko-KR" if lang == "Korean" else "en-US"}) + "\n")
        with torch.inference_mode(): o = nm.transcribe(mf, batch_size=1, verbose=False)
        return o[0].text if hasattr(o[0], "text") else str(o[0])
    for lab, rows in sets.items():
        lang = rows[0]["lang"]; R, H = [], []; t0 = time.time()
        for r in rows: R.append(ref_text(r)); H.append(nemo_tx(assemble_stream(r), lang))
        res.setdefault(lab, {})["nemotron_rnnt_56_0"] = dict(n=len(rows), **score(lang, R, H), example=(R[0][:60], H[0][:60]), sec=round(time.time() - t0))
        print(f"  [nemotron] {lab}: {res[lab]['nemotron_rnnt_56_0']}", flush=True)
    del nm; torch.cuda.empty_cache()
if "qwen" in models:
    from qwen_asr import Qwen3ASRModel
    qm = Qwen3ASRModel.from_pretrained(os.environ["MXC_QWEN_ASR_DIR"], dtype=torch.bfloat16, device_map="cuda:0", max_new_tokens=256); print("qwen3-asr 오프라인 로드", flush=True)
    def qwen_tx(x):
        sf.write(tmp, x, 16000); o = qm.transcribe(audio=tmp); return o[0].text if hasattr(o[0], "text") else str(o[0])
    for lab, rows in sets.items():
        lang = rows[0]["lang"]; R, H = [], []; t0 = time.time()
        for r in rows: R.append(ref_text(r)); H.append(qwen_tx(assemble_stream(r)))
        res.setdefault(lab, {})["qwen3_asr_offline"] = dict(n=len(rows), **score(lang, R, H), example=(R[0][:60], H[0][:60]), sec=round(time.time() - t0))
        print(f"  [qwen] {lab}: {res[lab]['qwen3_asr_offline']}", flush=True)
p = os.path.join(out, f"baselines-{a.sets}{'-all' if a.all else ''}.json")
json.dump(dict(args=vars(a), note="eval_other 는 서버 결손분(E03001–E03313 없음) 그대로 n=2687. 채점은 textnorm.score_*(태그·구두점·대문자 제거, EN 숫자→단어).", results=res), open(p, "w"), indent=1, ensure_ascii=False)
print("→", p)

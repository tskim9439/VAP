"""KsponSpeech '+'(반복)·'*'(불명확) 표기 발화에서 사전학습 ASR(Qwen3-ASR 오프라인, Nemotron RNN-T [56,0])이 반복 어절을 실제로 내는지 실측.
python experiments/tn_probe_repeats.py [--n 12] [--set eval_clean|dev]  → 원문 / v1 타깃 / Qwen / Nemotron 을 나란히 출력."""
import os, sys, json, glob, random, argparse, re, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, soundfile as sf
from vapasr.data.kspon import read_trn, resolve_path, read_pcm
from vapasr.data.textnorm import target_ko
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=12); ap.add_argument("--set", default="eval_clean"); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
root = os.environ["MXC_KSPONSPEECH_DIR"]; rows = [(rel, raw) for rel, raw in read_trn(os.path.join(root, a.set + ".trn")) if re.search(r"\S\+(\s|$)", raw)]
random.Random(a.seed).shuffle(rows); rows = rows[: a.n]; print(f"'+' 포함 발화 {len(rows)} 개 ({a.set})", flush=True)
tmp = tempfile.mktemp(suffix=".wav")
from qwen_asr import Qwen3ASRModel
qm = Qwen3ASRModel.from_pretrained(os.environ["MXC_QWEN_ASR_DIR"], dtype=torch.bfloat16, device_map="cuda:0", max_new_tokens=256)
import nemo.collections.asr as nemo_asr
nm = nemo_asr.models.ASRModel.restore_from(glob.glob(os.path.join(os.environ["MXC_NEMOTRON_DIR"], "*.nemo"))[0], map_location="cpu").cuda().eval(); nm.encoder.set_default_att_context_size([56, 0])
out = []
for rel, raw in rows:
    x, _ = read_pcm(resolve_path(root, rel)); sf.write(tmp, x, 16000)
    q = qm.transcribe(audio=tmp)[0]; q = q.text if hasattr(q, "text") else str(q)
    mf = tmp + ".json"; open(mf, "w").write(json.dumps({"audio_filepath": tmp, "duration": len(x) / 16000, "text": "", "lang": "ko-KR"}) + "\n")
    with torch.inference_mode(): o = nm.transcribe(mf, batch_size=1, verbose=False)
    n = o[0].text if hasattr(o[0], "text") else (o[0] if isinstance(o[0], str) else str(o[0]))
    r = dict(id=os.path.basename(rel), raw=raw, target_v1=target_ko(raw, "kspon"), qwen=q, nemotron=n); out.append(r)
    print(f"\n[{r['id']}]\n  raw : {raw}\n  v1  : {r['target_v1']}\n  qwen: {q}\n  nemo: {n}", flush=True)
json.dump(out, open(os.path.join(os.environ.get("MXC_DATA_LOG_DIR", "/tmp"), f"tn-probe-repeats-{a.set}.json"), "w"), ensure_ascii=False, indent=1)

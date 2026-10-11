"""Probe silent English-source translation outputs: same audio, different prefixes, δ4. Read-only (prints only)."""
import glob, json, sys, importlib.util, collections
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
import torch
spec = importlib.util.spec_from_file_location("ev", "/soundai/users/tskim/VAPKT/experiments/eval_single_turn_asr.py"); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
from vapasr.hf.infer import load_model
from vapasr.hf.batch_decode import decode_batch
from vapasr.data.ast_dataset import ast_head_text
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
CK = sys.argv[1]
model, tok = load_model(CK, encoder_path="/soundai/Model/nemotron-3.5-asr-streaming-0.6b", dtype=torch.float32)
dev = next(model.parameters()).device
R = "/soundai/users/tskim/VAPKT-data/results/ast-fleurs-q06-s2-ast"
rows = [json.loads(l) for f in sorted(glob.glob(f"{R}/predictions-rank*.jsonl")) for l in open(f)]
sil = [r for r in rows if r["dataset"] == "fleurs-en-de" and str(r["delta"]) == "4" and not r["token_ids"]][:48]
ok = [r for r in rows if r["dataset"] == "fleurs-en-de" and str(r["delta"]) == "4" and r["token_ids"] and r["audio_s"] < 10][:16]
man = {json.loads(l)["id"]: json.loads(l) for l in open("/soundai/users/tskim/VAPKT-data/data/evaluation/ast-fleurs-covost2-v1/manifest-max30s-fleurs.jsonl")}
def feats_of(rs):
    out = []
    for r in rs:
        wav = ev.read_audio(man[r["id"]], 1.0); w = torch.from_numpy(wav).to(dev)
        K = int(round(len(wav) / 16000 * model.config.frame_hz))
        out.append(model.encode(w[None], torch.tensor([len(wav)], device=dev), torch.tensor([K], device=dev))[0])
    return out
def prefix(src, spk, tgt, d):
    return list(model.config.prefix_ids) + tok(ast_head_text(src, spk, tgt), add_special_tokens=False)["input_ids"] + [model.config.sp_ids[f"<DELAY_{d}>"]]
VARIANTS = [("translate German, SPK_A, δ4", "English", "<SPK_A>", "German", 4), ("ASR only English, SPK_A, δ4", "English", "<SPK_A>", None, 4),
            ("translate German, SPK_B, δ4", "English", "<SPK_B>", "German", 4), ("translate Korean, SPK_A, δ4", "English", "<SPK_A>", "Korean", 4),
            ("translate German, SPK_A, δ10", "English", "<SPK_A>", "German", 10), ("translate English(!), SPK_A, δ4", "English", "<SPK_A>", "English", 4)]
with torch.inference_mode():
    for name, rs in (("silent-in-eval", sil), ("non-silent-in-eval(<10s)", ok)):
        F = feats_of(rs)
        for v, src, spk, tgt, d in VARIANTS:
            st = decode_batch(model, F, [prefix(src, spk, tgt, d)] * len(F), max_flush_rounds=8)
            n_sil = sum(not s.emitted for s in st)
            print(f"{name:26s} n={len(rs):2d} | {v:34s} silent={n_sil:2d}", flush=True)
        # first-chunk probabilities for one silent example
    r = sil[0]; F = feats_of([r])[0]
    print("example:", r["id"], r["audio_s"], r["reference"])
    st = decode_batch(model, [F], [prefix("English", "<SPK_A>", None, 4)], max_flush_rounds=8)
    print("ASR-only output:", tok.decode([t for _, t in st[0].emitted]))

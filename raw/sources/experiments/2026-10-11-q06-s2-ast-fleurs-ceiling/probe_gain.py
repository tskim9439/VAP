"""Silent FLEURS-English items: decode after peak-normalizing the waveform (gain only). Prints only."""
import glob, json, sys, importlib.util
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
import numpy as np, torch
spec = importlib.util.spec_from_file_location("ev", "/soundai/users/tskim/VAPKT/experiments/eval_single_turn_asr.py"); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
from vapasr.hf.infer import load_model
from vapasr.hf.batch_decode import decode_batch
from vapasr.data.ast_dataset import ast_head_text
from vapasr.data.ast_eval import split_runs
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
R = "/soundai/users/tskim/VAPKT-data/results/ast-fleurs-q06-s2-ast"
rows = [json.loads(l) for f in sorted(glob.glob(f"{R}/predictions-rank*.jsonl")) for l in open(f)]
sil = [r for r in rows if r["dataset"] == "fleurs-en-de" and str(r["delta"]) == "4" and not r["token_ids"]][:48]
man = {json.loads(l)["id"]: json.loads(l) for l in open("/soundai/users/tskim/VAPKT-data/data/evaluation/ast-fleurs-covost2-v1/manifest-max30s-fleurs.jsonl")}
model, tok = load_model("/soundai/Model/VAPASR/semcommit-q06-s2-ast/final", encoder_path="/soundai/Model/nemotron-3.5-asr-streaming-0.6b", dtype=torch.float32)
dev = next(model.parameters()).device
spk = [model.config.sp_ids["<SPK_A>"], model.config.sp_ids["<SPK_B>"]]; trans = model.config.sp_ids["<TRANS>"]
p = list(model.config.prefix_ids) + tok(ast_head_text("English", "<SPK_A>", "German"), add_special_tokens=False)["input_ids"] + [model.config.sp_ids["<DELAY_4>"]]
with torch.inference_mode():
    for target in (None, 0.1, 0.5):
        F = []
        for r in sil:
            wav = ev.read_audio(man[r["id"]], 1.0).astype(np.float32)
            if target: wav = wav * (target / max(np.abs(wav).max(), 1e-6))
            w = torch.from_numpy(wav).to(dev); K = int(round(len(wav) / 16000 * model.config.frame_hz))
            F.append(model.encode(w[None], torch.tensor([len(wav)], device=dev), torch.tensor([K], device=dev))[0])
        st = decode_batch(model, F, [p] * len(F), max_flush_rounds=8)
        n_sil = sum(not s.emitted for s in st); n_tr = sum(bool(split_runs(s.emitted, spk, trans)[1]) for s in st)
        print(f"peak→{target}: silent {n_sil}/{len(F)}, with translation {n_tr}/{len(F)}", flush=True)
        if target == 0.5:
            for r, s in list(zip(sil, st))[:3]:
                src, tr = split_runs(s.emitted, spk, trans)
                print("  REF:", r["reference"][:90], "| SRC:", tok.decode([t for _, t in src])[:90], "| TR:", tok.decode([t for _, t in tr])[:90])

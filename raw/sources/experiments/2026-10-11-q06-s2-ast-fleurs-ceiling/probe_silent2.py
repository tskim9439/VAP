"""Same silent FLEURS-English audio: Stage 1 model (plain ASR prefix) and s2 checkpoints (ASR-only ast prefix), δ4/δ6/δ10. Prints only."""
import glob, json, sys, importlib.util
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
import torch
spec = importlib.util.spec_from_file_location("ev", "/soundai/users/tskim/VAPKT/experiments/eval_single_turn_asr.py"); ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
from vapasr.hf.infer import load_model, prefix_ids
from vapasr.hf.batch_decode import decode_batch
from vapasr.data.ast_dataset import ast_head_text
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
R = "/soundai/users/tskim/VAPKT-data/results/ast-fleurs-q06-s2-ast"
rows = [json.loads(l) for f in sorted(glob.glob(f"{R}/predictions-rank*.jsonl")) for l in open(f)]
sil = [r for r in rows if r["dataset"] == "fleurs-en-de" and str(r["delta"]) == "4" and not r["token_ids"]][:48]
man = {json.loads(l)["id"]: json.loads(l) for l in open("/soundai/users/tskim/VAPKT-data/data/evaluation/ast-fleurs-covost2-v1/manifest-max30s-fleurs.jsonl")}
wavs = [ev.read_audio(man[r["id"]], 1.0) for r in sil]
for ck in sys.argv[1:]:
    model, tok = load_model(ck, encoder_path="/soundai/Model/nemotron-3.5-asr-streaming-0.6b", dtype=torch.float32)
    dev = next(model.parameters()).device
    ast = getattr(model.config, "ast_format", None) == "ast-v1"
    with torch.inference_mode():
        F = []
        for wav in wavs:
            w = torch.from_numpy(wav).to(dev); K = int(round(len(wav) / 16000 * model.config.frame_hz))
            F.append(model.encode(w[None], torch.tensor([len(wav)], device=dev), torch.tensor([K], device=dev))[0])
        for d in (4, 6, 10):
            p = (list(model.config.prefix_ids) + tok(ast_head_text("English", "<SPK_A>"), add_special_tokens=False)["input_ids"] + [model.config.sp_ids[f"<DELAY_{d}>"]]) if ast else prefix_ids(model, tok, "English", d)
            st = decode_batch(model, F, [p] * len(F), max_flush_rounds=8)
            print(f"{ck.split('/VAPASR/')[1]:40s} ast={ast} δ{d:<2d} silent={sum(not s.emitted for s in st)}/{len(F)}  ex0='{tok.decode([t for _, t in st[0].emitted])[:80]}'", flush=True)
    del model; torch.cuda.empty_cache()

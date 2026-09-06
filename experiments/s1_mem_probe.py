"""최장 배치 메모리 프로브 — 언어별로 가장 긴 bs 개 스트림으로 forward+backward 1 회 → 최대 GPU 할당량.
python experiments/s1_mem_probe.py --bs-en 12 --bs-ko 48 [--full-ft] [--no-grad-ckpt] --gpu 0
학습(s1_train_mono.py)과 같은 모델·collate 경로를 쓴다. 길이 버킷 배치라 최장 배치는 epoch 마다 반드시 한 번 나온다."""
import os, sys, argparse, time, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--bs-en", type=int, default=12); ap.add_argument("--bs-ko", type=int, default=48); ap.add_argument("--full-ft", action="store_true")
ap.add_argument("--lora-r", type=int, default=16); ap.add_argument("--no-grad-ckpt", action="store_true"); ap.add_argument("--M", type=int, default=4); ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--manifests", default="librispeech-100,kspon-100"); a = ap.parse_args()
dev = torch.device(f"cuda:{a.gpu}"); torch.backends.cuda.matmul.allow_tf32 = True
from vapasr.uslm.mono_data import MonoStreamDataset, collate_streams
from vapasr.uslm.mono_model import MonoInterleavedASR
from vapasr.uslm.model import Adapter
from qwen_asr import Qwen3ASRModel
qm = Qwen3ASRModel.from_pretrained(os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B"), dtype=torch.bfloat16, device_map=dev, max_new_tokens=8)
root = next(v for v in vars(qm).values() if isinstance(v, nn.Module)); thinker = root.thinker; tok = qm.processor.tokenizer; del root.thinker.audio_tower
names = a.manifests.split(","); ds = {n: MonoStreamDataset([n], tok, mode="stream", delays=(6,), max_per_chunk=a.M) for n in names}
model = MonoInterleavedASR(thinker, tok, Adapter(), ds[names[0]].sp_ids, lora_r=a.lora_r, full_ft=a.full_ft).to(dev); model.adapter.float()
if not a.no_grad_ckpt: model._lm().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train(); print(f"mode {'full-FT' if a.full_ft else f'LoRA r{a.lora_r}'} | grad ckpt {'off' if a.no_grad_ckpt else 'on'} | 가중치 {torch.cuda.memory_allocated(dev)/2**30:.1f} GB", flush=True)
for n in names:
    d = ds[n]; bs = a.bs_en if d.items[0]["lang"] == "English" else a.bs_ko
    idx = sorted(range(len(d)), key=lambda i: -(2 * d.items[i]["K"] + len(d.items[i]["tokens"])))[:bs]
    b = collate_streams([d[i] for i in idx]); B, L = b["ids"].shape
    torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(dev); t = time.time()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss, parts = model(b["feats"].to(dev), b["ids"].to(dev), b["is_audio"].to(dev), b["chunk_of"].to(dev), b["labels"].to(dev), b["mask"].to(dev), next_weight=0.3)
    loss.backward(); torch.cuda.synchronize(dev); model.zero_grad(set_to_none=True)
    print(f"  {n}: bs {B} × L {L} = {B*L} tok (max K {b['frames']//B if B else 0} avg) | loss {loss.item():.3f} | peak {torch.cuda.max_memory_allocated(dev)/2**30:.1f} GB | {time.time()-t:.1f}s", flush=True)

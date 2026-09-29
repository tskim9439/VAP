"""thinker 단독 fwd+bwd 마이크로벤치 — sdpa vs varlen(vapasr/hf/varlen_attention.py), 패딩·packed 배치 (2026-09-29 리뷰 후 재측정용 원본).

Qwen3-ASR-0.6B thinker text model 구조(가중치는 random — 시간은 가중치와 무관), Liger(RMSNorm·SwiGLU·RoPE, vapasr/hf/liger.py), grad ckpt(non-reentrant),
fp32 가중치 + bf16 autocast, loss = 실토큰 hidden 의 평균 제곱. 입력 길이는 설계 단계 프로토타입과 같은 난수열(random.Random(0)):
  ko54   54 × 250–320 (budget:16384 KO main 배치 모양)   mix8  8 × 200–4000   long8  8 × 3500–4000
모드: padded_sdpa (B, L)+2D mask / padded_varlen_layer_unpad (thinker.model 에 2D mask → 층마다 unpad, 분기 2) /
      padded_varlen_model_unpad (VapAsrForStreamingASR._thinker_unpadded — 모델의 실제 패딩 varlen 경로) / packed_varlen (position_ids → cu_seqlens) /
      packed_sdpa (position_ids 재시작 → sdpa 블록 대각 4D mask).
실행: 벤치 코드 복사본에서 stdin 으로
  cd <code> && CUDA_VISIBLE_DEVICES=<g> PYTHONDONTWRITEBYTECODE=1 <python> - < thinker-microbench.py
마지막 줄 'RESULT {json}' 이 결과.
"""
import os, sys, json, time, random, socket, types
sys.path.insert(0, os.getcwd())
import torch
torch.backends.cuda.matmul.allow_tf32 = True
import transformers
import qwen_asr.core.transformers_backend.modeling_qwen3_asr as M
from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRThinkerConfig
from vapasr.hf.varlen_attention import register, packed_kwargs, IMPL
from vapasr.hf.liger import apply_liger_to_thinker
from vapasr.hf.modeling_vapasr import VapAsrForStreamingASR

dev = "cuda"; register()
cfg = Qwen3ASRThinkerConfig.from_dict(json.load(open("/soundai/Model/Qwen3-ASR-0.6B/config.json"))["thinker_config"])
torch.manual_seed(0)
with torch.device(dev): tm = M.Qwen3ASRThinkerTextModel._from_config(cfg.text_config)
liger = apply_liger_to_thinker(types.SimpleNamespace(thinker=tm))
tm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); tm.train(); tm.config.use_cache = False
holder = types.SimpleNamespace(thinker=types.SimpleNamespace(model=tm)); D = tm.config.hidden_size
meta = dict(host=socket.gethostname(), date=time.strftime("%Y-%m-%d %H:%M:%S %Z"), gpu=torch.cuda.get_device_name(0), cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch=torch.__version__, transformers=transformers.__version__, liger=liger, layers=tm.config.num_hidden_layers, hidden=D,
            heads=[tm.config.num_attention_heads, tm.config.num_key_value_heads], head_dim=getattr(tm.config, "head_dim", None))
print("META", json.dumps(meta), flush=True)
R = {}


def run(tag, impl, lens, mode, iters=8, warm=3):
    tm.set_attn_implementation(impl); tm.config._attn_implementation = impl
    B, L, N = len(lens), max(lens), sum(lens); g = torch.Generator(device=dev).manual_seed(0)
    mask = torch.arange(L, device=dev)[None] < torch.tensor(lens, device=dev)[:, None]
    if mode == "packed":
        E = torch.randn(1, N, D, device=dev, generator=g, requires_grad=True); pos = torch.cat([torch.arange(n, device=dev) for n in lens])[None]
    else:
        E = torch.randn(B, L, D, device=dev, generator=g, requires_grad=True)

    def fwd():
        if mode == "packed":
            kw = packed_kwargs(position_ids=pos) if impl == IMPL else {}                     # VapAsrForStreamingASR.forward 의 packing 경로와 같게 매 forward 계산
            return tm(inputs_embeds=E, attention_mask=None, position_ids=pos, use_cache=False, **kw).last_hidden_state[0]
        if mode == "padded_model_unpad":
            return VapAsrForStreamingASR._thinker_unpadded(holder, E, mask)[mask]
        return tm(inputs_embeds=E, attention_mask=mask.long(), use_cache=False).last_hidden_state[mask]
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    for it in range(warm + iters):
        if it == warm: torch.cuda.synchronize(); t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16): h = fwd()
        h.float().pow(2).mean().backward(); E.grad = None
    torch.cuda.synchronize(); dt = (time.time() - t0) / iters
    tm.zero_grad(set_to_none=True)
    r = dict(ms=round(dt * 1000, 1), attn=impl, mode=mode, iters=iters, row_tokens=(N if mode in ("packed", "padded_model_unpad") else B * L), real_tokens=N,
             peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
    R[tag] = r; print(tag, json.dumps(r), flush=True)


rnd = random.Random(0)
ko = sorted(rnd.randint(250, 320) for _ in range(54))
mix = [rnd.randint(200, 4000) for _ in range(8)]
long_ = sorted(rnd.randint(3500, 4000) for _ in range(8))
meta["lens"] = dict(ko54=[min(ko), max(ko), sum(ko)], mix8=mix, long8=long_)
for name, lens in (("ko54", ko), ("mix8", mix)):
    run(f"{name}_padded_sdpa", "sdpa", lens, "padded")
    run(f"{name}_padded_varlen_layer_unpad", IMPL, lens, "padded")
    run(f"{name}_padded_varlen_model_unpad", IMPL, lens, "padded_model_unpad")
    run(f"{name}_packed_varlen", IMPL, lens, "packed")
    run(f"{name}_packed_sdpa", "sdpa", lens, "packed", iters=3, warm=1)
run("long8_padded_sdpa", "sdpa", long_, "padded")
run("long8_padded_varlen_model_unpad", IMPL, long_, "padded_model_unpad")
run("long8_packed_varlen", IMPL, long_, "packed")
print("RESULT " + json.dumps(dict(meta=meta, results=R)), flush=True)

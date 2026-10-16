# Thinker + adapter → MLX porting spec, and a single-stream greedy optimization plan (q06-s4-ast)

Scope: the adapter (encoder frame → thinker embedding) and the Qwen3-ASR-0.6B text decoder ("thinker") as used inside
`VapAsrForStreamingASR`. It also covers how to run them in MLX on Apple Silicon fast enough for live mic use, with greedy output
token-identical to the eval rules. The prefix, blocked ids, caps, segment/K rules and output parsing are in `spec/decode-contract.md`.
The encoder is in `spec/encoder-spec.md`. This file does not repeat them, except where the thinker engine depends on them.

Citation keys:
- `MV` = `vapasr/hf/modeling_vapasr.py`
- `CV` = `vapasr/hf/configuration_vapasr.py`
- `RT` = `vapasr/hf/realtime.py`
- `QA` = `<site-packages>/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py` (env `vapasr-local`)
- `ST` = `vapasr/data/semcommit_tokens.py`
- `MLQ` = `<site-packages>/mlx_lm/models/qwen3.py`, `MLC` = `mlx_lm/models/cache.py`, `MLB` = `mlx_lm/models/base.py` (mlx_lm 0.31.3)
- `SCR` = `scratchpad/scratch-thinker/` (the probe and benchmark scripts written for this spec; all were run under the heavy lock)

All measurements were taken on the dev box: Mac mini **M4 (base, 10-core GPU, 120 GB/s), 16 GB**, MLX 0.32.2, mlx_lm 0.31.3.
They used **random weights** of the exact 0.6B shape, because the s4 checkpoint is not on this machine. Other user processes were
resident (an idle speech-to-speech server, about 45 % memory free), so expect ±1–2 ms noise on every number.

**[critic] Review 2026-10-16.** An adversarial pass re-derived the load-bearing claims from the working tree. Every change below is
marked **[critic]**. The critic's probes are in `scratchpad/scratch-critic-thinker/`:
- `spec_vs_stream_decode.py` / `.json`: speculative batching against the repo's **real** `stream_decode`, torch fp32 CPU, under the lock.
- `spot_bench.py` / `.json`: an independent MLX fp16 timing spot check written without `mlx_lm` modules, under the lock.
- `rope_numpy_vs_torch.py` / `.json`: whether a torch-free cos/sin table reproduces the HF table.

Re-verified and found correct:
- adapter, layer math, GQA mapping, RoPE convention and MRoPE reduction (`QA` lines re-read);
- parameter counts: 596 049 920 for the thinker, and fp32 thinker + adapter + encoder = 4.84 GB;
- tokenizer sizes (62 added tokens, 151643..151704);
- the `mlx_lm` claims, the `RT` claims, the s3 emission statistics (reproduced exactly, §5.4), and the micro-benchmark magnitudes.

Corrected or added:
1. The "exact" RoPE table is only exact when it is built with **torch**. Numpy differs by 1 ulp, and MLX-op tables differ by 6e-5 (§2.4).
2. The KV capacity needs a **clamp, not an assert**. In a runaway, positions can reach 8K+100 (§4.4).
3. The converted weights must go to a `.nosync` dir, not `App/models/` (§2.6).
4. Two `mx.topk`/tie-rule pitfalls in the re-score (§4.4).
5. Final (offline) mode on this engine (§5.6, new).
6. Training also used Liger, TF32 and varlen attention (§6.1).
7. Stale `encoder-spec.md` line citations (§5.1).
8. A citation fix for the `mlx_lm` mask path (§3.2).
9. The lm_head loader fallback (§2.5).

---------------------------------------------------------------------------------------------------------------------------------

## 0. TL;DR — what to build

1. **Adapter**, in fp32 even when the thinker runs in fp16:
   `LayerNorm(1024, eps 1e-5, affine) → Linear(1024→2048)+b → GELU(exact erf) → Linear(2048→1024)+b`.
   - No other scaling or normalization. The output is cast to the thinker dtype.
     **[critic]** With the fp32-residual default, keep it in fp32 as the residual input instead (§1).
   - Cost in MLX: 0.5–0.75 ms per 4 frames.
2. **Thinker** = a plain Qwen3 decoder with these properties:
   - 28 layers, hidden 1024, 16 q heads / 8 kv heads, head_dim 128 (q_proj outputs 2048), SwiGLU 3072, RMSNorm eps 1e-6.
   - q/k RMSNorm over head_dim is applied **before** RoPE.
   - RoPE θ = 1e6, rotate-half convention. Interleaved MRoPE reduces exactly to 1-D RoPE, because all three position streams are
     identical. Position = running sequence index.
   - Scale 1/√128, no soft-cap, no sliding window.
   - Tied lm_head over **all 151 936 rows**.
3. **Engine.** Write our own ~150-line MLX module rather than calling `mlx_lm` `Model.__call__`. It loads weights straight from the
   VapAsr safetensors (`thinker.model.*`, `adapter.net.*`). It has:
   - a preallocated per-layer KV cache, (1, 8, 2048, 128) in fp16 (235 MB);
   - a single `forward(x (q,D), start, sel) → argmax ids of the selected rows` that writes KV at `start..start+q-1`;
   - the block mask (−inf) and argmax inside the graph, with fp32 logits, and one `.tolist()` sync per forward;
   - RoPE with HF-exact frequencies, i.e. `mx.fast.rope(..., base=None, freqs=1/inv_freq_HF)` or a precomputed cos/sin table. The
     `base=1e6` path is off by 5e-4 at positions ≥ 1024 (§2.4).
     **[critic]** Build the table with **numpy float32 on the CPU** (≤ 1 ulp = 6e-8 from torch) or ship a torch-built table in the
     converted weights. **Never build it with MLX ops**: `mx.power`/`mx.cos` on the GPU are off by up to 6.1e-5 (§2.4).
   - **[critic]** The preallocated KV is a capacity that the engine must **clamp** at, not assert on (§4.4). In a runaway the
     worst-case positions are 8K+100, which exceeds 2048 for K ≥ 244, i.e. segments of 19.5 s or more including the tail.
4. **Speculative audio batching** (§5) is the main speed lever.
   - All 4 frames of a 320 ms encoder step go through **one** forward: `[NEXT?, a_k, NEXT, a_k+1, NEXT, a_k+2, NEXT, a_k+3]`, with
     argmax read at the 4 audio rows.
   - Accept up to the first frame whose argmax is not `<NEXT_AUDIO>` (or none), then roll back the KV write pointer.
   - It is exactly equivalent to sequential greedy decoding.
   - Measured on M4 fp16: 18.5 ms for the batch vs 65.7 ms for 4 × q=2 at context 512. In a full decode loop it is token-identical
     to sequential and uses 33 % fewer forwards (§5.4).
5. **Precision.** Default to fp16 weights, with fp32 residual stream, fp32 norms, fp32 logits and fp32 re-scoring of the top-8
   candidates. This is the same speed as plain fp16.
   - Offer an "exact" fp32 mode. It is about 33 ms per forward on a base M4, and the projection is about 15 ms on M4 Pro and about
     7 ms on M4 Max.
   - Do not default to bf16: its logits have an ulp of 0.25–0.5 at typical magnitudes, so argmax ties are likely.
   - 8-bit and 4-bit are optional and **not exact**.
   - **[critic]** The startup heuristic in §6.4 picks fp32 whenever it fits the budget. By the §4.5 projection that is the case on an
     M4 Pro/Max (fp32 q=1 ≈ 15 ms). So on the target MacBook the effective default is usually **fp32**, and fp16 + re-score is the
     default only where fp32 does not fit (base M4, M3 Pro).
6. **Where the time goes on a base M4.** Every forward is weight-bandwidth bound: fp16 takes about 16–17 ms whatever the design
   (mlx_lm, fixed-cache eager or `mx.compile`), and almost whatever q (1–8) or context (64–1024).
   - So the only levers are fewer forwards (§5), fewer bytes (precision or quantization), and a faster machine.
   - `mx.compile` gave no measurable gain (§4.3).

---------------------------------------------------------------------------------------------------------------------------------

## 1. Adapter and `chunk_embed`

| item | value | evidence |
|---|---|---|
| module | `Adapter.net = Sequential(LayerNorm(d_in), Linear(d_in,h), GELU(), Linear(h,d_out))` | `MV:17-20` |
| dims | d_in 1024, h 2048, d_out 1024 (= thinker hidden; asserted) | `CV:13`, `MV:50`, `MV:284-287` |
| LayerNorm | eps **1e-5** (torch default), elementwise affine with bias | printed module, `SCR/keys_probe.py` |
| GELU | `nn.GELU()` ⇒ `approximate='none'` = **exact erf** GELU (MLX `nn.gelu`, not `gelu_approx`) | printed `GELU approximate: none` |
| input | encoder output (B,1,K,1024) → `feats[:,0].float()` (fp32), raw ConformerEncoder output transposed to (B,T,D) with no extra norm or scale; frames ≥ T' are zero vectors | `MV:77-79`, `vapasr/features/online.py:129-142` |
| dtype | adapter weights stay fp32 (`load_model` casts only `model.thinker`); output `.to(emb.weight.dtype)` | `vapasr/hf/infer.py:31`, `MV:227`, `RT:106` |
| keys | `adapter.net.0.weight (1024)`, `adapter.net.0.bias (1024)`, `adapter.net.1.weight (2048,1024)`, `adapter.net.1.bias (2048)`, `adapter.net.3.weight (1024,2048)`, `adapter.net.3.bias (1024)` (4 199 424 params) | `SCR/vapasr_state_dict_meta.tsv` |
| training | under bf16 autocast the two Linears ran in bf16 and LN in fp32; eval/inference is fp32 | `experiments/semcommit_train.py:397` (`bf16: True`) |

MLX implementation, verified against torch with max |Δ| = 1.3e-5 on outputs up to 10.6 (`SCR/port_check.py`):
```python
h = mx.fast.layer_norm(x_f32, W["net.0.weight"], W["net.0.bias"], 1e-5)
h = nn.gelu(h @ W["net.1.weight"].T + W["net.1.bias"])          # erf GELU
ce = (h @ W["net.3.weight"].T + W["net.3.bias"]).astype(thinker_dtype)
```
- torch `nn.Linear` and MLX `nn.Linear` both store the weight as (out, in), so no transposes are needed when loading.
- `act_head` exists only when `lanes > 0`, so it is absent in s4 (`MV:60`).
- **[critic]** Code evidence for eps and GELU: `MV:19` builds `nn.LayerNorm(d_in)` and `nn.GELU()` with no arguments, i.e. the torch
  defaults eps=1e-5 and `approximate='none'`. This holds independently of the printed module.
- **[critic]** In the fp32-residual mode (§6.2), feed the adapter output to the residual stream **in fp32**. Do not round it to fp16
  first. The reference's `inputs_embeds` are fp32 (`MV:227` with fp32 weights). Rounding ce to fp16 adds an avoidable input error
  at every audio row. Only the token rows (`E[id]`) are unavoidably fp16-rounded when E is stored in fp16.

---------------------------------------------------------------------------------------------------------------------------------

## 2. Thinker LM

### 2.1 Which class
- `VapAsrForStreamingASR.__init__` imports `qwen_asr.core.transformers_backend.{configuration,modeling}_qwen3_asr`. It builds
  `Qwen3ASRThinkerForConditionalGeneration(Qwen3ASRThinkerConfig.from_dict(config.thinker))` and **deletes `audio_tower`**
  (`MV:51-55`).
  - `from_qwen` loads `Qwen3ASRForConditionalGeneration.from_pretrained(qwen_dir, dtype=float32).thinker` (`MV:275, 280`).
  - The class comes from the pip package `qwen_asr`, not from transformers.
- The instantiated tree was verified by `SCR/keys_probe.py`:
  - `thinker.model` is `Qwen3ASRThinkerTextModel` (`QA:961-1059`).
  - Its layers are `Qwen3ASRThinkerTextDecoderLayer` (`QA:236-279`), containing `Qwen3ASRTextAttention` (`QA:143-217`),
    `Qwen3ASRTextMLP` (`QA:220-233`) and `Qwen3ASRTextRMSNorm` (`QA:50-68`).
  - The final norm is `Qwen3ASRTextRMSNorm` (`QA:979`) and the rotary embedding is `Qwen3ASRThinkerTextRotaryEmbedding`
    (`QA:781-837`).
  - The lm_head is `Linear(1024,151936,bias=False)` (`QA:1088`).
  - The attention backend is `sdpa`.

### 2.2 `text_config` of Qwen/Qwen3-ASR-0.6B
Source: `scratchpad/hf-qwen3-asr/config.json`, revision 5eb144179a02, fetched 2026-10-16. The s4 `config.json["thinker"]` embeds the
same dict (`MV:285`). Re-read it at load time and assert these values.

| field | value | | field | value |
|---|---|---|---|---|
| hidden_size | 1024 | | num_attention_heads | 16 |
| num_hidden_layers | 28 | | num_key_value_heads | 8 (GQA groups 2) |
| intermediate_size | 3072 | | head_dim | **128** (q_proj out = 2048 ≠ hidden) |
| hidden_act | silu | | rms_norm_eps | 1e-6 |
| rope_theta | 1 000 000 | | rope_scaling | `{rope_type: default, interleaved: true, mrope_interleaved: true, mrope_section: [24,20,20]}` |
| vocab_size | 151 936 | | tie_word_embeddings | true (text and thinker) |
| attention_bias | false | | max_position_embeddings | 65 536 |
| attention_dropout | 0.0 | | thinker dtype (hub) | bfloat16 (VapAsr loads fp32, `MV:280`) |

### 2.3 Layer math (exact)
For each layer ℓ, with h the residual stream of shape (q, 1024):
```
x   = RMSNorm_in(h)                                  # QA:259-260
Q   = q_norm( (x Wq^T).view(q,16,128) )              # RMSNorm over head_dim, eps 1e-6, BEFORE RoPE (QA:168-173, 188)
K   = k_norm( (x Wk^T).view(q, 8,128) )              # QA:189
V   =        (x Wv^T).view(q, 8,128)                 # QA:190
Q,K = rope(Q,pos), rope(K,pos)                       # rotate-half (QA:71-75, 116-140), cos/sin fp32 then cast to x.dtype (QA:829-837)
O   = softmax(Q K^T · 128^-0.5 + causal) V           # scale = head_dim**-0.5 = 0.08838834764831845 (QA:152); q head i ↔ kv head i//2 (repeat_kv QA:78-87)
h   = h + O.reshape(q,2048) Wo^T                     # QA:215-216, 272
h   = h + Wdown( silu(Wgate · RMSNorm_post(h)) ⊙ (Wup · RMSNorm_post(h)) )   # QA:226-233, 275-278
```
- After the last layer, `h = RMSNorm_final(h)` (`QA:1054`), then `logits = h · E^T` (tied, `QA:1241`).
- **There is no logit soft-capping, no final scaling and no bias anywhere** (`QA:1240-1241`, `attention_bias=false`).
- No sliding window: `Qwen3ASRTextAttention` passes none, and the unused thinker variant sets `sliding_window=None` (`QA:908`).
- RMSNorm (`QA:60-65`): `x32 = x.float(); x32 *= rsqrt(mean(x32²) + eps); return weight * x32.to(input_dtype)`.
  - The weight multiply happens after the cast back.
  - `mx.fast.rms_norm` multiplies in fp32 before rounding, so it differs by ≤ 1 ulp of the compute dtype.
  - The difference is invisible in fp32 and below fp16 noise.

### 2.4 RoPE / MRoPE, and positions in `stream_decode`
- **inv_freq** = `1 / θ^(arange(0,128,2)/128)` in fp32, and `attention_scaling = 1.0` for `rope_type=default`
  (`transformers/modeling_rope_utils.py:123-132`; `RT:40` asserts it).
  - Measured inv_freq[0] = 1.0 and inv_freq[63] = 1.2409e-6.
- **Interleaved MRoPE** (`QA:802-817`) works on `freqs` of shape (3, B, T, 64):
  - It starts from stream T, `freqs_t = freqs[0]`.
  - It overwrites indices `1:60:3` with stream H (`freqs[1]`) and `2:60:3` with stream W (`freqs[2]`).
  - `emb = cat(freqs_t, freqs_t)`.
  - **Proof of reduction:** if `position_ids[0] == [1] == [2]`, then `freqs[0] == freqs[1] == freqs[2]` elementwise, because
    `freqs[s] = inv_freq ⊗ pos[s]` (`QA:826-831`). So every overwrite writes the value already there and `freqs_t = inv_freq ⊗ pos`,
    which is plain 1-D RoPE. ∎
  - Numerically, `SCR/keys_probe.py` gives max|cos_MRoPE − cos_1D| = **0.0** and max|sin…| = **0.0** over positions 0..2999.
  - As a sanity check, distinct streams (offsets 0/7/11) differ by 1.99.
- **The three streams are always identical in this model:**
  - `stream_decode` and `iter_stream` call `self.thinker(inputs_embeds=…, past_key_values=cache, use_cache=True)` with **no**
    attention_mask or position_ids (`MV:228-229`, `infer.py:106-107`).
  - The thinker forward computes rope indices only when `attention_mask is not None` (`QA:1210`).
  - The text model then sets `cache_position = arange(past_seen, past_seen+L)` and
    `position_ids = cache_position.view(1,1,-1).expand(3,B,-1)` (`QA:1009-1017`).
  - **Position = absolute sequence index of every input**: prefix tokens, each audio embedding, `<NEXT_AUDIO>`, text tokens and
    `<EMPTY_AUDIO>` alike. Each takes exactly one position and there are no gaps.
  - Training used the same convention:
    - padded path: `thinker.model(inputs_embeds, attention_mask=mask)` (`MV:118`) leaves position_ids None, so the column index is
      used;
    - packed path: per-sample positions restart at 0 (`MV:101-102, 110`).
    - **[critic]** s4 actually trained on a third path. `submit-q06-s4-ast.sh:29` passes `--varlen`, which
      `submit-semcommit-train-apex.sh:77,82` turns into `--attn-impl varlen --batch-max-tokens 16384`. That is padded batches
      routed through `_thinker_unpadded` (`MV:115-116, 157-165`), whose positions are the original column index
      (`MV:162`). Rows are right-padded (`vapasr/uslm/mono_data.py:190`: `ids[i, :n] = …`), so positions are again 0..n−1 per
      sample. The conclusion is unchanged.
  - Even stock Qwen3-ASR feeds identical 3-stream positions for audio (`get_rope_index`, `QA:442-444`).
  - `RT` uses `pos = arange(start, start+q)` with the same fp32 table (`RT:41-43, 91-93`).
- **MLX RoPE numerics.** `SCR/rope_variants.py` compares each option against the HF fp32 formula on a (1,16,2048,128) unit-normal
  input:

  | MLX variant | max abs diff, all positions | max abs diff, pos ≥ 1024 |
  |---|---|---|
  | `mx.fast.rope(…, base=1e6)` (= `nn.RoPE`, which mlx_lm uses) | 5.35e-4 | 5.35e-4 |
  | `mx.fast.rope(…, base=None, freqs=mx.array(1/inv_freq_HF))` | 9.5e-7 | 9.5e-7 |
  | table `x·cos + rotate_half(x)·sin` (cos/sin from HF formula, fp32) | **0.0** | **0.0** |

  - Use the table (exact) or the `freqs=` form (fused, fp32-noise level).
  - Both cost about 0.27–0.29 ms per call standalone, and dynamic `offset`/slice inside `mx.compile` works.
  - The `scratch-critic/rope_check.py` result in `decode-contract.md §c.2(1)` agrees.
  - **[critic] The "0.0" row holds only for a table built with torch.** `SCR/rope_variants.py:3-5` built cos/sin with torch on the
    CPU. The app runtime has no torch (`audio-ui-spec.md:47`: mlx, numpy, safetensors, …). The critic measured torch-free builds
    against the torch-CPU HF table (`scratch-critic-thinker/rope_numpy_vs_torch.json`, positions 0..2047):

    | table build | inv_freq entries ≠ torch | max abs Δ cos/sin |
    |---|---|---|
    | numpy float32 (`np.float32(1e6) ** (arange(0,128,2).astype(f32)/128)`, `pos*inv`, `np.cos/np.sin`) | 1 of 64 (1 ulp) | **5.96e-8** (1 ulp; 41 456 of 262 144 entries differ) |
    | numpy angles, float64 cos/sin rounded to fp32 | — | 5.96e-8 |
    | MLX ops on the GPU (`mx.power`, `mx.cos`, `mx.sin`) | 23 of 64 | **6.1e-5** |

    - Rule: build the table **once with numpy fp32 on the CPU** (1 ulp, about 10⁴× below the `base=` error), or store a torch-built
      table in the converted weights when the converter has torch available.
    - **Never** build it with MLX array ops. The fused `freqs=` form (9.5e-7) is also acceptable.
    - "Bit-exact" is not a cross-device notion anyway. The s4 eval reference ran on H200/CUDA, whose `cos`/`sin` and K=1 matmul
      (`QA:831`) need not match torch-CPU to the ulp. What matters is removing the systematic 5e-4 error.
  - **[critic] Cross-spec:** `decode-contract.md §c.2(1)` says that a table "computed with torch/numpy" gives 0.0. That is true for
    torch only; numpy gives 1 ulp (above).

### 2.5 Vocabulary, embeddings, lm_head
- Embedding rows: **151 936** (`embed_tokens.weight` (151936,1024)).
- Tokenizer sizes:
  - The Qwen3-ASR base tokenizer has **151 705** entries: 151 643 BPE entries plus 62 added tokens, the last being
    `<asr_text>` = 151704 (`hf-qwen3-asr/tokenizer_config.json`).
  - vapasr appends the specials 151705..151726, so for s4 `len(tok)` = 151 727 (`ST:23-29, 71`;
    `vapasr/data/dialogue_tokens.py:20-21`).
  - Rows **151727..151935 are unused and not blocked**. The reference argmax runs over all 151 936 rows (`MV:235-236`; `RT:69-70`),
    so the port must too. Do not truncate to `len(tok)`.
- lm_head is **tied**:
  - VapAsr `_tied_weights_keys = ["thinker.lm_head.weight"]` (`MV:44`), and the thinker's own keys are
    `["model.embed_tokens.weight","lm_head.weight"]` (`QA:1070`).
  - Verified in `SCR/keys_probe.py`: `lm_head.weight.data_ptr() == embed_tokens.weight.data_ptr()` after construction and after a
    `save_pretrained`/`from_pretrained` round trip.
  - **The saved safetensors contains no `lm_head` key.**
  - Loader rule: take `thinker.model.embed_tokens.weight` for both input lookup and output projection. If `thinker.lm_head.weight`
    is present, assert it equals the embedding and ignore it.
  - **[critic] Fallback for the other alias.** The round trip was verified only with this Mac's transformers 4.57.6. s4 was saved on
    mxc by `trainer.save_model` (`experiments/semcommit_train.py:431`), and the transformers version there is not recorded in the repo
    (the s4 `config.json` will show `transformers_version`).
    - The tied-weight de-duplication drops one alias, normally the name in `_tied_weights_keys`.
    - If a file has **only** `thinker.lm_head.weight`, use it as E for both roles. Fail only if neither key exists.
    - Read every `*.safetensors` in the directory, as `MV:265` does, in case the file is sharded.
- Blocked ids, `<NEXT_AUDIO>` and caps: use `config.json` verbatim (`MV:58-59`). The expected s4 list is in
  `decode-contract.md §a.2`. Apply them as an additive fp32 vector (0 / −inf; plus `-next_bias` at NEXT if ever used) on fp32 logits
  inside the graph.
- `runaway_cap` is **32** for s4 (`ST:98`), so a round can emit up to 32 text tokens. This matters for the tail latency in §5.4.

### 2.6 Checkpoint keys (derived from code, dumped by instantiating on `meta`)
`SCR/vapasr_state_dict_meta.tsv` lists all 317 in-memory keys (`SCR/keys_probe.py`):
```
adapter.net.0.weight (1024,)                  adapter.net.0.bias (1024,)
adapter.net.1.weight (2048,1024)              adapter.net.1.bias (2048,)
adapter.net.3.weight (1024,2048)              adapter.net.3.bias (1024,)
thinker.model.embed_tokens.weight (151936,1024)
thinker.model.layers.{0..27}.self_attn.q_proj.weight (2048,1024)
thinker.model.layers.{i}.self_attn.k_proj.weight (1024,1024)
thinker.model.layers.{i}.self_attn.v_proj.weight (1024,1024)
thinker.model.layers.{i}.self_attn.o_proj.weight (1024,2048)
thinker.model.layers.{i}.self_attn.q_norm.weight (128,)
thinker.model.layers.{i}.self_attn.k_norm.weight (128,)
thinker.model.layers.{i}.mlp.gate_proj.weight (3072,1024)
thinker.model.layers.{i}.mlp.up_proj.weight   (3072,1024)
thinker.model.layers.{i}.mlp.down_proj.weight (1024,3072)
thinker.model.layers.{i}.input_layernorm.weight (1024,)
thinker.model.layers.{i}.post_attention_layernorm.weight (1024,)
thinker.model.norm.weight (1024,)
thinker.lm_head.weight (151936,1024)          # in memory only (tied); NOT in the saved file
```
- Thinker parameters excluding the tied head: **596 049 920**. That is 2.38 GB in fp32, 1.19 GB in fp16/bf16, about 0.63 GB in 8-bit
  and about 0.34 GB in 4-bit (group 64).
- The s4 file also holds `encoder.enc.*` (`encoder_trainable`/`encoder_saved`, `MV:249-269`; see the encoder spec). There is no
  `thinker.audio_tower.*` and no `act_head.*`.
- fp32 thinker plus encoder ≈ 4.8 GB, which matches the stated file size.
- MLX naming: strip `thinker.`, which gives exactly the `mlx_lm` qwen3 tree (`model.embed_tokens`, `model.layers.N.self_attn.q_proj`, …,
  `MLQ:32-160`). A custom module can use any names. Load with `safetensors.safe_open(path, framework="numpy")` and fetch only
  `thinker.*`/`adapter.*`, or with `mx.load`.
- **Recommended:** a one-time conversion to `App/models/<ckpt>/thinker-<dtype>.safetensors` plus `adapter-fp32.safetensors`, with a
  manifest recording the source file size, mtime and `config.json` sha256. This avoids reading 4.8 GB of fp32 at every start.
  - **[critic] Wrong location.** `App/` lives in the iCloud-synced `~/Desktop/VAPKT`. A 1.2–2.4 GB weight file there can be evicted
    to "dataless", after which reads hang. The owner's checkpoint therefore sits in a `.nosync` dir.
  - The audio-UI spec already fixes the location (`audio-ui-spec.md:47, 727-736`): `<parent>/<name without .nosync>-mlx.nosync`,
    i.e. `~/Desktop/VAPKT-models/semcommit-q06-s4-ast-mlx.nosync`, or `<parent>/<name>-mlx` on `/Volumes/*`.
  - It also requires per-tensor `safe_open(framework="numpy")` reads, an atomic rename and `vapkt-app-manifest.json`. Follow that
    spec. Never write weights under `App/`.
  - Add the numpy-built RoPE cos/sin table (§2.4) to that output, or rebuild it at load. Both are cheap: 2 × 2048 × 128 fp32 = 2 MB.
- **[critic] Re-checked sizes.** `SCR/vapasr_state_dict_meta.tsv` gives adapter + thinker = 600 249 344 params, and
  `scratch-encoder/out/state_dict.tsv` gives encoder = 636 keys and 609 141 760 params. In fp32 that is **4.84 GB**, so the stated
  4.8 GB checks out.
- **[critic]** The file also holds `encoder.pre.featurizer.{window,fb}` (638 `encoder.*` keys = 636 + 2,
  `train-85952-events.txt` `encoder_saved_keys: 638`; `decode-contract.md §g`). The thinker loader ignores every `encoder.*` key.

### 2.7 Port verified on random weights
`SCR/port_check.py` builds the real `VapAsrForStreamingASR` (torch fp32, CPU) with the downloaded config, 4 text layers and random
weights at realistic scale. It copies the weights into MLX and replays the `stream_decode` input schedule:
prefix(24) → `[a0]` → `[N,a1]` → token → `[N,a2]` `[N,a3]` `[N,a4]` `[N,a5]`.

| MLX path vs torch fp32 `thinker(inputs_embeds, DynamicCache)` | fp32 | fp16 | bf16 |
|---|---|---|---|
| fixed-KV engine, sequential (max abs Δlogit, logit scale ≈ 79) | 3.1e-4 | 0.22 | 1.9 |
| fixed-KV engine, speculative q=8 `sel=(1,3,5,7)` | 2.7e-4 | 0.17 | 1.6 |
| mlx_lm KVCache path (`live_mlx` style) | 3.1e-4 | 0.23 | 2.0 |
| argmax equal at all 8 decision points | yes | yes | yes |
| speculative q=8 vs sequential q=2, same dtype | 1.3e-4 | 0.125 | 0.5 |

The structure (norm placement, GQA mapping, rope convention, tie, adapter) is therefore correct. The last row shows that batched and
sequential runs are mathematically equal but **not bitwise** equal (§5.2).

---------------------------------------------------------------------------------------------------------------------------------

## 3. Existing MLX path and the CUDA realtime design

### 3.1 `experiments/live/export_thinker_mlx.py` (Sep 10, E2)
- It writes a HF `Qwen3ForCausalLM` config **without `rope_scaling`** (`:11-14`), so mlx_lm builds `nn.RoPE(128, traditional=False,
  base=1e6)` (`mlx_lm/models/rope_utils.py:242-251`).
  - That is correct by §2.4, apart from the 5e-4 `base=` numerics.
- It strips `thinker.` from all keys (`:16`), loading the whole `model.safetensors`, encoder included, through torch.
- It then runs `mlx_lm.convert` (`:22`), fp16 by default, with optional `q_bits` at group 64.
  - With `--q-bits`, mlx_lm also quantizes `embed_tokens`. Token, `<NEXT_AUDIO>` and `<EMPTY_AUDIO>` **input embeddings** then
    become dequantized 8-bit values, not only the lm_head. Audio embeddings are unaffected.

### 3.2 `vapasr/hf/live_mlx.py` + mlx_lm 0.31.3 internals
- **`input_embeddings` is supported.**
  - `Model.__call__(inputs, cache, input_embeddings)` (`MLQ:172-178`) passes it to `Qwen3Model`, which uses it instead of
    `embed_tokens(inputs)` (`MLQ:142-151`).
  - `MlxThinker.step` passes dummy zero ids plus `input_embeddings=e[None]` (`live_mlx.py:20-24`).
- **KV cache** (`MLC:325-355`): preallocates in steps of 256 positions, concatenating a new block when full. `update_and_fetch` writes
  `keys[..., prev:offset, :]` and returns the `[:offset]` view.
  - RoPE uses `offset=cache.offset` (`MLQ:77-80`), so position = sequence index, as in HF.
  - `trim(n)` just lowers `offset` (`MLC:375-381`). Rollback for §5 is therefore possible with this cache too, because the trimmed
    slots are overwritten by the next `update_and_fetch`.
- **Mask:** `create_attention_mask` returns `None` for N=1 and the string `"causal"` otherwise (`MLB:45-55`). This is lower-right
  aligned, which is correct because keys = offset + N.
  - **[critic] Citation fix.** With a `KVCache`, `MLB:49-50` delegates to `cache.make_mask` (`MLC:393-394`), which calls the
    *cache module's* own `create_attention_mask` (`mlx_lm/models/cache.py:114-124`). That function also returns `None` for N=1 and
    `"causal"` otherwise, so the behaviour is as stated, but the code path is a different one.
  - The trim lines are `MLC:378-381`.
- **Overheads:**
  - The model computes `as_linear` logits for **all q positions** (`MLQ:178-181`). For a ~30-token prefix that is 30 × 151 936
    logits, harmless but wasted.
  - `live.py` then masks via an additive vector and does one `.item()` per step (`live.py:79, 91`).
  - Measured mlx_lm-path latency is in §4.2 (`mlxlm_dynamic_eager`). It is **as fast as a hand-written fixed cache on a base M4**,
    because the GPU is bandwidth-bound.
- **Old `LiveSession`** (`live.py:64-142`) uses the same merged-NEXT rule as `RT` (`live.py:100-113`), with three differences:
  - it has **no `max_total` cap**;
  - it uses a **torch-MPS [56,0] encoder** (`live.py:3`) and a torch adapter;
  - it uses the legacy prefix (`infer.prefix_ids`, not ast-v1).
- **Old results (E2 checkpoint, M4):** MLX fp16 thinker at about 16 ms decode per 80 ms chunk; logits within ≤ 0.13 of torch fp16
  (MPS); identical transcripts. int8 took 12 ms (`wiki/outputs/output-phase1-report.md:119-127`).

### 3.3 `RT.GraphThinker` / `realtime_decode` (CUDA, Oct 12) — the design to mirror
- **Fixed KV** `(L,1,KV,max_len,HD)`, zero-initialized (`RT:36-38`), `max_len=2048` in `stream_rtf.py:112`.
- **RoPE table** precomputed in fp32 for every position and cast to the compute dtype (`RT:41-43`).
- **Mask** `ar ≤ pos` over the full `max_len` (`RT:56`), so the causal block within a multi-position call is implied.
- **KV write** `index_copy_` at `pos` (`RT:65`), then SDPA with `enable_gqa` and `scale=a.scaling` (`RT:66`).
- **Output:** lm_head on the last position only, `masked_fill(block, −inf).argmax()` inside the graph (`RT:69-70`).
- CUDA graphs are captured for q = 1 and 2 (`RT:72-83`). `run()` copies x/pos into static buffers and replays them; any other q
  (the prefix) runs eagerly (`RT:85-93`).
- `realtime_decode` folds the pending `<NEXT_AUDIO>` into the next audio or EMPTY forward, as q=2 (`RT:113-124`). The logits at a
  NEXT position are never used (`MV:238`).
- **Parity:** token-identical to `stream_decode` on 192 decodes / 14 441 tokens, fp32 H200, q06-s3-ast (raw
  `2026-10-12-q06-s3-ast-rtf/README.md:86`). The q=2 merge already shows that kernel-shape changes did not flip a token at fp32 on
  that set.
- **Speed:** RTF 0.14–0.15; 80 ms frame p50 7.7 ms on H200 (`wiki/outputs/output-simulst-baselines-20261012.md:152-167`).
  **[critic] Citation fix:** the RTF is at `output-simulst-baselines-20261012.md:159`. The 7.7 ms p50 frame time is not in that
  file; it is in `raw/sources/experiments/2026-10-12-q06-s3-ast-rtf/README.md:85`.

---------------------------------------------------------------------------------------------------------------------------------

## 4. Optimization plan (measured on base M4)

### 4.1 Designs measured (`SCR/bench.py`, `SCR/mlx_thinker_proto.py`)
- (i) `mlxlm_dynamic_eager`: mlx_lm `qwen3.Model` plus `KVCache`, `input_embeddings`, all-position logits and argmax on the last
  one, then `trim(q)` to keep the context constant.
- (ii) `fixed_compiled_bucket256`: our `FixedThinker`.
  - Preallocated KV (1,8,2048,128) per layer.
  - `start` passed as an int32 **array**, so shapes are static.
  - `mx.fast.rope(offset=start)` and `mx.slice_update` for the KV write.
  - `mx.fast.scaled_dot_product_attention` over `K[:, :, :Lb]`, with `Lb = ceil((start+q)/256)·256` and a bool mask
    `key_idx ≤ query_pos`.
  - lm_head only on the `sel` rows, fp32 logits + block vector, argmax in graph.
  - The whole step is wrapped in `mx.compile`, with the KV lists passed in and out.
- (ii-b) `fixed_compiled_full2048`: the same, but attention covers all 2048 slots (the `RT` style).
- (ii-c) `fixed_eager_bucket256`: (ii) without `mx.compile`.
- (iii) precision: fp16, bf16, fp32 weights, and `nn.quantize(group_size=64, bits=8|4)` applied to Linear and Embedding (q8/q4, fp16
  scales).

Each number is the wall time of one forward **including the argmax and the `.item()`/`.tolist()` sync**, taken as the median of 30
runs after 6 warm-up runs.

### 4.2 Results
**Per forward, fp16, all designs** (p50 ms, p90 in parentheses):

| design | C | q=1 | q=2 | q=4 | q=8 |
|---|---|---|---|---|---|
| mlxlm_dynamic_eager | 64 | 16.2 (19.3) | 16.3 (18.6) | 16.6 (18.2) | 17.3 (19.6) |
| mlxlm_dynamic_eager | 512 | 17.0 (18.6) | 16.5 (18.6) | 17.5 (20.3) | 18.8 (20.9) |
| mlxlm_dynamic_eager | 1024 | 16.8 (18.6) | 17.4 (20.6) | 19.1 (22.4) | 19.6 (22.9) |
| fixed_eager_bucket256 | 64 | 15.9 (18.1) | 16.2 (18.2) | 16.3 (18.3) | 17.5 (19.4) |
| fixed_eager_bucket256 | 512 | 16.6 (18.5) | 16.9 (18.6) | 17.3 (19.2) | 18.6 (21.2) |
| fixed_eager_bucket256 | 1024 | 17.4 (19.6) | 17.1 (19.0) | 19.2 (23.6) | 24.8 (26.6) |
| fixed_compiled_bucket256 | 64 | 15.7 (18.4) | 16.1 (18.3) | 16.3 (18.7) | 17.0 (20.0) |
| fixed_compiled_bucket256 | 512 | 17.0 (19.6) | 16.0 (18.5) | 17.1 (18.7) | 17.5 (19.8) |
| fixed_compiled_bucket256 | 1024 | 17.7 (20.1) | 18.7 (21.1) | 18.3 (21.1) | 20.6 (22.6) |
| fixed_compiled_full2048 | 64 | 15.9 (17.9) | 16.6 (18.2) | 17.6 (19.6) | 19.4 (21.8) |
| fixed_compiled_full2048 | 512 | 16.5 (18.4) | 16.4 (18.9) | 17.8 (19.7) | 20.0 (22.3) |
| fixed_compiled_full2048 | 1024 | 17.2 (19.0) | 16.4 (19.1) | 18.1 (20.6) | 20.6 (23.5) |

**Per forward, by precision** (p50 ms, `fixed_compiled_bucket256`; mlx_lm in parentheses):

| precision | weights | C=64 q=1 | C=64 q=8 | C=512 q=1 | C=512 q=2 | C=512 q=8 | C=1024 q=1 | C=1024 q=8 |
|---|---|---|---|---|---|---|---|---|
| fp32 | 2.38 GB | 31.3 (32.4) | 36.1 (36.5) | 33.8 (33.0) | 35.9 (35.7) | 44.0 (40.1) | 34.1 (35.2) | 42.8 (43.5) |
| fp16 | 1.19 GB | 15.7 (16.2) | 17.0 (17.3) | 17.0 (17.0) | 16.0 (16.5) | 17.5 (18.8) | 17.7 (16.8) | 20.6 (19.6) |
| bf16 | 1.19 GB | 14.8 (15.9) | 16.7 (17.0) | 16.1 (16.6) | 15.7 (16.2) | 18.2 (19.0) | 16.9 (17.5) | 20.1 (20.8) |
| q8 g64 | 0.63 GB | 8.7 (8.6) | 11.4 (12.8) | 9.1 (9.4) | 9.5 (10.3) | 13.6 (14.2) | 10.2 (10.4) | 14.9 (16.4) |
| q4 g64 | 0.34 GB | 5.3 (5.4) | 10.2 (11.8) | 5.9 (6.4) | 6.3 (6.6) | 11.6 (13.1) | 6.6 (6.8) | 13.3 (15.1) |

Full tables for every design × precision × q × C are in `SCR/bench-{fp16,bf16,fp32,q8,q4}.json`.

**Other costs** (`SCR/bench.py`, `bench2.py`, `compile_cost.py`; fp16):

| item | ms |
|---|---|
| bare `.item()` sync | 0.15–0.19 |
| CPU graph build of one forward (no eval), eager vs compiled, q=1 and q=8 | 0.60 / 0.61, 0.69 / 0.63 |
| first call of a new compiled shape (trace cost on top of the forward) | +5–7 |
| lm_head alone, fp16 / q8 / q4, q=1 (151 936×1024) | 3.9 / 1.9 / 1.2 |
| speculative q=8, lm_head on 1 row vs 4 rows | 18.0 vs 18.1 |
| 6 chained text tokens, sync per step vs `mx.async_eval` pipelining (next input = embedding of the on-device argmax) | 99.2 vs 94.5 |
| adapter, fp32, 4 frames | 0.5–0.75 |

### 4.3 Interpretation
- **Bandwidth-bound.**
  - An fp16 forward moves about 1.19 GB of weights in about 16 ms, an effective ~75 GB/s, roughly 60 % of the M4's 120 GB/s.
  - fp32 takes 2×, q8 about 0.55×, q4 about 0.35× at q=1. Time is almost flat in q for fp16/bf16/fp32 (+1–3 ms from q=1 to q=8)
    and in context up to 1024 (+1–2 ms).
  - Quantized matmuls grow with q: q8 goes 9 → 13.6 ms and q4 goes 6 → 11.6 ms from q=1 to q=8, because qmv turns into qmm.
- **Engine choice does not change speed on M4.** mlx_lm dynamic, fixed eager and fixed compiled are all within ±1 ms. CPU build is
  0.6 ms and the sync 0.18 ms per forward, about 5 % of a forward. `mx.compile` neither shrank the build time nor the GPU time.
  - Choose the engine for **correctness and control**: HF-exact RoPE, logits only on `sel` rows, trivial rollback, and a
    preallocated memory bound. Do not choose it for raw speed.
  - Attention over the full 2048 slots costs +1–2.5 ms at q=8. Slice the KV to `start+q` (eager) or to a bucket (compiled).
- **Main levers, in order:**
  1. **Fewer forwards**: speculative audio batching (§5). 4 frames take 18.5 ms instead of 65.7 ms at fp16, C=512.
  2. **Fewer bytes per forward**: precision. Only fp16/bf16/fp32 are exact candidates.
  3. **Faster machine**: §4.5.
  4. Small items: `mx.async_eval` pipelining of text-token chains saves about 0.8 ms/token, and two-stage lm_head (§4.6) could save
     about 2.5 ms/forward.
- Attention / KV is not a bottleneck at these contexts. KV fp16 at max_len 2048 is 235 MB, which equals `RT`'s 2048. Segments are
  ≤ 31 s, so positions ≈ 19 + 2K + #tokens ≤ about 1 300 (`decode-contract.md §d.7`).
  - **[critic] That is a typical value, not a bound.** Every round writes one audio or EMPTY row plus one NEXT row plus its tokens.
    The last NEXT stays pending and is never written. So positions ≤ P + 2·(K + R) + T, with R ≤ 8 flush rounds and
    T ≤ min(6K+64, 32·(K+R)) tokens (`MV:231, 237`).
  - With P ≤ 20 that gives a worst case of about **8K + 100**:
    - 2048 is exceeded for K ≥ 244, i.e. 19.5 s including the tail;
    - at the audio-UI cap K_cap = 325 the worst case is about 2 700.
  - This only happens in runaway (cap-hit) regimes, but the engine must survive it. See the clamp in §4.4.

### 4.4 Recommended engine (exact greedy)
```python
class MlxThinker:                       # App/engine/thinker.py — own module, no mlx_lm runtime dependency needed
    def __init__(self, W, cfg, blocked_ids, dtype=mx.float16, max_len=2048, resid_fp32=True, rescore_k=8):
        # W: dict from VapAsr safetensors with 'thinker.' stripped; assert shapes of §2.6 and text_config of §2.2
        # KV: [mx.zeros((1, 8, max_len, 128), dtype) for 28 layers] x2
        # block_add: fp32 (151936,) with -inf at blocked ids
        # rope: inv = 1/(1e6**(arange(0,128,2,int64).astype(f32)/128)); freqs = 1/inv  (or exact cos/sin table)
    def forward(self, x, start, sel=None) -> list[int]:
        """x (q,1024) in dtype; writes KV[start:start+q]; returns argmax ids at rows sel (default: last). One sync."""
        # per layer: hn = rms_norm(h).astype(dtype); q/k/v proj; q_norm/k_norm; rope(offset=start, freqs=…)
        #            KV[l][..., start:start+q, :] = k, v          (slice assignment, in place like mlx_lm KVCache)
        #            o = mx.fast.scaled_dot_product_attention(q, K[..., :start+q, :], V[..., :start+q, :], scale=128**-0.5,
        #                                                     mask=None if q == 1 else "causal")   # lower-right aligned
        #            h = h + o_proj(o);  h = h + mlp(rms_norm(h).astype(dtype))   (h kept fp32 if resid_fp32)
        # hs = rms_norm_final(h[sel]); logits = (hs.astype(dtype) @ E.T).astype(f32) + block_add
        # if rescore_k: cand = top-k(logits); exact = E[cand].astype(f32) @ hs_f32 + block_add[cand]; id = cand[argmax(exact)]
        # return mx.argmax(...).tolist()
    def embed(self, ids) -> mx.array: ...   # E[ids] (dtype)
```
- The caller owns `pos` (the KV write pointer), and **rollback = calling the next `forward` with a smaller `start`**. Also
  `assert start + q <= max_len`, as `RT:88` does.
  - **[critic] Replace the assert with the hard clamp of `audio-ui-spec.md` §1.6 / §5.3** (`Round.kv_full`). An assert would crash
    the live session in exactly the runaway case §4.3 shows is reachable.
    - Sequential rule: when `pos + 2 > kv_capacity`, end the round as forced, skip the remaining rounds, report `kv_full`, and let the
      app end the segment.
    - Speculative rule: only form a batch if `pos + len(xs) + 1 ≤ kv_capacity`. Otherwise decode the remaining frames with the
      sequential clamped rule. This keeps both schedules on the same (non-reference, runaway-only) behaviour.
  - **[critic] Segment reset** = set `pos ← 0`, or `pos ← P` when the prefix KV is restored. Zeroing the buffers is unnecessary,
    because stale rows are never attended (§5.3).
    - Exception: after the non-finite guard of §6.2 fires, zero or reallocate the KV buffers. In the bucket or full-length attention
      form, masked stale V rows still enter the weighted sum with weight 0, and 0·inf = NaN.
    - The `[:start+q]` slice form never reads them.
- Use `mask="causal"` only with the `[:start+q]` slice. With a bucket or full-length slice, use the explicit bool mask from
  `SCR/mlx_thinker_proto.py:_step`.
  - The bool-mask + bucket form is what `FixedThinker` and all the parity and spec-decode tests used.
  - The `"causal"`-on-slice form is what the mlx_lm path in `SCR/port_check.py` used (`MLC:355`, `MLB:45-55`). It also matched torch.
- `mx.compile`: optional, with no measured benefit. If it is used, call the 4–6 shapes once at load to warm them:
  - the prefix length P,
  - q=1 (tokens),
  - q=2 (NEXT + EMPTY flush),
  - q=7 with `sel=(0,2,4,6)` (the first group after the prefix),
  - q=8 with `sel=(1,3,5,7)`,
  - and the q=2/4/6 restarts, times the attention buckets.
- Use a single thread or worker for every MLX call (encoder and thinker), in issue order. MLX graph building is not meant to be
  interleaved from several Python threads.
- **[critic] Re-score implementation pitfalls** (pseudo-code line `cand = top-k(logits)`):
  1. In MLX 0.32.2 `mx.topk` returns **values only** (docstring checked). Use `mx.argpartition(-logits, kth=k)[..., :k]` for the
     candidate ids, as `SCR/numerics_variants.py:36` does.
  2. Tie rule. The reference picks the **lowest id** among exact maxima (torch `argmax`; `decode-contract.md §c.2(5)`), while
     `cand[argmax(exact)]` picks the earliest in argpartition order. Sort `cand` ascending first, then argmax. This is negligible but
     free.
  3. The prototype re-scores only one row (`h[0, -1:]`, `SCR/numerics_variants.py:32-39`). For speculative batches, apply it per
     `sel` row: gather k·|sel| ≤ 32 rows of E.
  4. The fp16 default (resid32 + re-score) was **not** run through the decode-loop identity test of §5.4, which used plain fp16
     (`SCR/spec_decode_sched.py` builds `FixedThinker`). §6.5 must cover it in both schedules, as its candidate list already says.

### 4.5 Projection to the target MacBook Pro (not measured)
Assume per-forward time ≈ bytes / (0.6 × peak bandwidth), with peaks per Apple spec sheets: M3 Pro 150, M3 Max 300/400, M4 Pro 273,
M4 Max 410/546 GB/s.

| machine | fp16 forward | fp32 forward |
|---|---|---|
| M4 base (measured) | 16–17 ms | 31–35 ms |
| M3 Pro | ~13 | ~26 |
| M4 Pro | ~7 | ~15 |
| M3 Max 400 / M4 Max 546 | ~5 / ~3.6 | ~10 / ~7.3 |

At under about 5 ms, the fixed CPU cost of about 0.8 ms (build + sync) becomes 15–20 %. That is where `mx.async_eval` pipelining
(§4.6) pays.

**[critic]** The Max-chip rows are probably optimistic. A q=1 forward is about 28 × 10 small kernels. On wide GPUs, single-stream
decode tends to hit a per-kernel latency floor before the bandwidth bound. Treat the M3/M4 Max fp16 numbers (3.6–5 ms) as a lower
bound. The ranking (fp32 is affordable on Pro/Max) is unaffected. This was not measured; the §6.4 startup timing decides on the real
machine.

### 4.6 Optional extras (not needed for real time on Pro/Max)
- **Async text-token chain.** Enqueue forward t+1 with input `E[argmax_t]` gathered **on device** before reading argmax_t on the host.
  This is the right next input in every case except two:
  - *Forced stop* (`n ≥ cap` or `len(out) ≥ max_total` while argmax_t is text). Sequential decoding feeds NEXT there, so discard the
    enqueued result and reuse `start`.
  - *argmax_t == NEXT*. Then the enqueued `[NEXT]` q=1 forward wrote exactly what the merged `[NEXT, a]` forward would have written
    at that slot, so keep it and start the next audio batch without a leading NEXT.

  Measured gain: about 0.8 ms per token on M4.
- **Two-stage lm_head, exact with a certificate.** Score all rows with a q4 copy of E (1.2 ms instead of 3.9 ms), take the top-k
  (k=32), re-score them with fp16/fp32 E, and accept the result if `best_exact ≥ (k+1)-th approx + ‖h‖·max_v‖E_v − Ê_v‖`. Otherwise
  fall back to the full lm_head. Expected saving is about 2.5 ms of 16 ms. It is not prototyped.
  **[critic]** The certificate must also absorb the rounding of the approximate scores themselves: a q4/fp16 matmul output in fp16
  has an ulp of 0.03–0.06 at |logit| < 128. Add `2·ulp(|approx_(k+1)|)` to the bound, or compute the approximate pass with fp32
  output.
- **Quantization (q8/q4):** a speed toggle for the base M4 only. It is not exact, and with the default mlx_lm quantization predicate
  it also changes token input embeddings. It must pass the agreement test in §6.5 before being offered.

---------------------------------------------------------------------------------------------------------------------------------

## 5. Speculative audio batching

### 5.1 Algorithm
The encoder emits frames in steps of 4. The first step also yields 4 frames for [56,3], and the final step 1–4
(`encoder-spec.md:315, 456`). For each newly available group of frames `k..g_end-1`:
```
while k < g_end:
    xs, sel = [], []
    for j, kk in enumerate(range(k, g_end)):
        if pending or j > 0: xs.append(E[NEXT])          # NEXT that closes the previous zero-token round
        xs.append(ce[kk]); sel.append(len(xs) - 1)       # audio row
    ids = thinker.forward(stack(xs), start=pos, sel=sel) # ONE forward, q ≤ 8
    acc = None
    for j, tid in enumerate(ids):                        # same stop test as MV:237 / RT:122, evaluated per audio row
        if tid == NEXT or cap <= 0 or len(out) >= max_total:
            forced += (tid != NEXT); continue            # frame k+j: zero-token round, ends with NEXT
        acc = j; break                                   # first frame that emits text
    if acc is None:
        pos += len(xs); pending = True; k = g_end        # accept everything; the trailing NEXT stays pending (merged later)
    else:
        pos += sel[acc] + 1                              # ROLLBACK: valid through the audio row of frame k+acc
        pending = False
        finish_round(k + acc, ids[acc])                  # sequential: emit tid, feed E[tid] (q=1) ... until stop → pending=True
        k = k + acc + 1                                  # re-batch the remaining frames of this group: [NEXT, a, NEXT, a, ...]
flush rounds: unchanged (sequential, RT:128-131)
```
- `finish_round` is the inner loop of `RT:121-124`, entered with the already-known `tid` and `n = 0`.
- Reference implementations are in `SCR/spec_decode_check.py` and `SCR/spec_decode_sched.py` (`decode_spec`).
- The first group after the prefix has `pending=False`, so its batch has q=7 with `sel=(0,2,4,6)`. All later full groups use q=8,
  `sel=(1,3,5,7)`.
- **[critic] Stale citations.** `encoder-spec.md:315, 456` now point at unrelated text, because that spec was edited after this one
  was written. The current locations are:
  - 4 frames per step: `encoder-spec.md:52, 353`;
  - first step = 25 mel rows → 4 frames: `:523`;
  - final step ⌊(r+7)/8⌋ = 1–4 frames, and nothing for r < 8: `:533-537`;
  - very short utterances (a single first step of T < 25 rows) → 1–4 frames: `:636`.
- **[critic] Shapes that actually occur.** Every q from 1 to 8 is possible:
  - the first group has `pending=False`: q = 2r−1 for r available frames (1, 3, 5, 7);
  - restarts and later groups have `pending=True`: q = 2r (2, 4, 6, 8);
  - the final partial step gives r = 1–4.
  - With `mx.compile`, warm all eight q values times the observed `sel` patterns, or run eager.
  - `audio-ui-spec.md` §5.3 `warmup()` lists only "q=1/q=2 decoder", so it must add these shapes.

### 5.2 Why this is exactly sequential greedy (argument)
Let S be the KV sequence before the group. In the sequential rule (`RT:117-124`, equivalently `MV:232-239`):
- the round of frame m appends `[NEXT if pending] + a_m`;
- it then takes d_m = argmax(blocked(logits at a_m));
- if `stop(d_m)` holds, meaning `d_m == NEXT or n ≥ cap or |out| ≥ max_total` with n = 0, the round ends with zero tokens. `forced`
  counts `d_m ≠ NEXT`, pending is set to True, and the **next** round appends `[NEXT, a_{m+1}]`.

1. **Same inputs, same positions.** If frames k..k+j−1 all stop, sequential decoding appends exactly
   `[NEXT?, a_k, NEXT, a_{k+1}, …, NEXT, a_{k+j}]` up to frame k+j's audio row. That is the prefix of the batch `X` up to `sel[j]`.
   Positions are absolute indices (`QA:1009-1017`; ours: `start + i`), so every row gets the same position in both schedules.
2. **Causality.** Under the causal mask, the hidden state of row i of X depends only on S and `X[:i+1]`. Rows after i (speculative
   NEXT/audio rows) cannot influence row i. So the batched logits at `sel[j]` equal the sequential logits at frame k+j's decision, as
   functions; floating point is covered in point 6.
3. **Same decisions.**
   - `cap` is never binding at an audio row, because n = 0 and cap = 32 > 0.
   - `|out|` is constant across the batch, because no token is emitted before `acc`. So `|out| ≥ max_total` has the same value for
     every row as in the sequential loop.
   - When it holds, every frame stops (forced if argmax ≠ NEXT), in both schedules.
   - The block vector and `next_bias` are the same additive vector on every row.
   - Hence d_{k+j} matches for all j ≤ acc, and the `forced` increments match.
4. **Divergence point.** At `acc`, sequential decoding emits `d_{k+acc}` and appends `E[d]` at position `start + sel[acc] + 1`,
   whereas the batch put NEXT there. Accepting exactly through `sel[acc]` and continuing sequentially from that position reproduces
   the sequential state:

   | state | value after rollback |
   |---|---|
   | KV contents | identical up to `start + sel[acc]` |
   | `pending` | False (the round is open) |
   | `n` | 0, then incremented as tokens are emitted |
   | `out`, `forced` | unchanged |
   | `k` | k + acc |

   The remaining frames of the group are re-batched after that round ends (`pending = True`), so they again see
   `[NEXT, a, NEXT, a, …]`, which is the sequential input. Induction over rounds gives identity for the whole stream. ∎
5. **No look-ahead in real time.** The batch only includes frames the encoder has already emitted (the current step's 4 frames).
   Sequential decoding of that step could not start before those frames existed either, so emission times in frame units are
   unchanged. Only the compute schedule changes.
6. **Numerics (the only caveat).** The batched q=8 forward uses different kernel shapes from q=2 and q=1: GEMM tiling, sdpa_vector
   with 8 rows, lm_head on 4 rows.
   - Results are equal mathematically but not bitwise. Measured on random weights (`SCR/port_check.py`), max abs Δlogit between
     speculative and sequential was **1.3e-4 (fp32), 0.125 (fp16), 0.5 (bf16)**.
   - This is the same class of deviation `RT` already accepts with its q=2 NEXT merge versus HF q=1. That merge was token-identical
     over 14 441 tokens at fp32.
   - An argmax can flip only at a near-tie: a top-2 margin smaller than this noise. §6.5 measures the rate on real weights.

### 5.3 What must be rolled back
**Only the KV write pointer** (`pos ← start + sel[acc] + 1`) plus the decode bookkeeping (`pending ← False`, the frame cursor).
- The stale K/V rows written at positions `> pos-1` stay in the buffer. They are harmless because:
  - every later forward writes its own K/V at its positions **before** attending;
  - every query at position p attends only to keys ≤ p (bool mask `idx ≤ pos`, or `"causal"` on the `[:start+q]` slice);
  - so a stale row is never visible before it is overwritten.
- RoPE carries no state (angles come from `start`), and no other cache exists.
- With mlx_lm's `KVCache`, the equivalent is `for c in cache: c.trim(len(xs) - sel[acc] - 1)` (`MLC:375-381`).
- **Negative control:** advancing the pointer by one extra slot (one stale entry left visible) breaks identity at emitted token #2
  (`BROKEN=stale1`, `SCR/spec_decode_sched.py`). The test is therefore sensitive to rollback errors.

### 5.4 Measurements
**Micro-benchmark** (p50 ms; one batched forward vs 4 sequential q=2 forwards with a sync each, `SCR/bench.py`):

| precision | C=64: spec q=8 / 4×q=2 | C=512 | C=1024 | text token q=1 (C=512) |
|---|---|---|---|---|
| fp16 | 16.6 / 62.7 (3.8×) | 18.5 / 65.7 (3.6×) | 19.7 / 69.1 (3.5×) | 16.3 |
| bf16 | 16.6 / 62.1 (3.7×) | 17.9 / 65.4 (3.7×) | 19.6 / 67.8 (3.5×) | 19.5 (noisy; 15.7 at C=64) |
| fp32 | 36.3 / 139.8 (3.9×) | 38.7 / 144.9 (3.7×) | 42.3 / 151.0 (3.6×) | 33.8 |
| q8 | 12.0 / 36.8 (3.1×) | 13.5 / 39.5 (2.9×) | 15.5 / 43.5 (2.8×) | 9.0 |
| q4 | 10.3 / 22.8 (2.2×) | 11.8 / 25.5 (2.2×) | 13.5 / 29.9 (2.2×) | 5.9 |

**Full decode loop, sequential (`RT` rules) vs speculative**, 28-layer random fp16/fp32 model (`SCR/spec_decode_sched.py`).
- The emission schedule is forced by a ±1e4 NEXT bias on decision rows, identical in both paths: 86 % zero-token frames, and frames
  emitting 1–8 or 40 tokens (cap-forced). The token **ids** come from the model, so they depend on the full KV context.

| run | tokens | forced | flush rounds | forwards seq → spec | identical |
|---|---|---|---|---|---|
| fp16, K=200, `max_total` 1264 | 184 | 3 / 3 | 3 / 3 | 387 → 259 | **yes** |
| fp32, K=120 | 152 | 3 / 3 | 3 / 3 | 275 → 198 | **yes** |
| fp16, K=160, `max_total` 72 (global cap path) | 72 | 9 / 9 | 1 / 1 | 233 → 134 | **yes** |
| fp16, K=200, rollback bug injected | — | — | — | — | **no** (detected) |

The unforced random-bias runs (`SCR/spec_decode_check.py`, K=40..200) were also identical. With zero emissions, a 320 ms group cost
**66 ms sequential vs 17–18 ms speculative**, i.e. 51 vs 201 forwards for K=200.

**[critic] Independent confirmation against the real reference.** Both paths above were written by the spec author, so a shared
misreading of the rules would go unnoticed. Also, the schedule test forces every NEXT decision with ±1e4, and its blocked list is
not the s4 list: it blocks `<SEM_END>` 151723 and omits 151725 (`SCR/spec_decode_sched.py:17`).
- `scratch-critic-thinker/spec_vs_stream_decode.py` compares the **repo's own** `VapAsrForStreamingASR.stream_decode`
  (`MV:218-246`, HF `DynamicCache`, step by step, separate NEXT forwards) against a torch speculative decoder that implements §5.1
  literally, with rollback = `DynamicCache.crop(start+sel[acc]+1)`.
- Setup: 2-layer random thinker, torch fp32 CPU, the s4 blocked list of `decode-contract.md §a.2`. NEXT is boosted only through the
  reference's own `next_bias` argument, identically in both paths, under the lock.
- Result: **identical `emitted`, `forced` and `flush_rounds` in 8 of 8 configurations**. These cover:
  - K ∈ {23, 37, 41, 50, 53, 61, 64}, so partial final groups;
  - cap 32 / 3 / 2, with forced cuts up to 62 per decode;
  - `max_total_per_chunk` 0.3, so the global cap is hit;
  - 1 to 8 flush rounds;
  - zero emissions (39 → 12 forwards).
- The stale-slot negative control (`crop(+1)`) diverges, as it should.
- **[critic] Timing spot check** (`scratch-critic-thinker/spot_bench.json`): an independent eager fixed-KV MLX forward (no `mlx_lm`
  modules, `freqs=` RoPE, `"causal"` on the slice), random fp16 0.6B, base M4.

  | C | q=1 | spec q=8 `sel=(1,3,5,7)` | 4 × q=2 |
  |---|---|---|---|
  | 64 | 16.1 ms | 16.3 ms | 65.9 ms |
  | 512 | 16.7 ms | 19.4 ms | 66.7 ms |
  | 1024 | 17.6 ms | 20.8 ms | 71.3 ms |

  This is within ±2 ms of §4.2/§5.4, and the 3.4–4× batching gain is confirmed.
- **[critic]** The per-group emission table below was recomputed from the raw jsonl with the stated tick model and matches to the
  digit: 14 460 estimated tokens, 86.1 % zero frames, 55.5 % zero groups, and the same mean/p50/p90/p95/p99/max rows.
  - Caveat: the raw rows contain only `ticks_ms` and `tokens`, not per-frame emissions, so the per-frame split is an estimate.
  - The totals are exact: 14 441 tokens over 30 564 frames.

**Projection with real emission statistics.**
- The s3 realtime host run (`raw/sources/experiments/2026-10-12-q06-s3-ast-rtf/host-rows-graph-stream-float32.jsonl`, 192 decodes,
  FLEURS 4 directions × δ4/δ10, +1 s tail, 14 441 tokens) gives 0.47 tokens per frame.
- Per-frame token counts were estimated from per-frame tick times as tick = 7.7 ms (q=2) + n × 5.0 ms (q=1). The calibrated total was
  14 460 vs 14 441 true, with per-utterance correlation 0.998.
- 86 % of frames emit nothing, and 55.5 % of 4-frame groups emit nothing at all.

| per 320 ms group | mean | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| tokens | 1.88 | 0 | 5 | 8 | 22 | 61 |
| forwards, sequential (4 + tokens) | 5.84 | 4 | 9 | 12 | 26 | 65 |
| forwards, speculative (1 + restarts + tokens) | 3.29 | 1 | 7 | 11 | 24 | 64 |

Estimated thinker compute per group on a **base M4**:

| | mean | p50 | p99 |
|---|---|---|---|
| fp16 sequential | ≈ 96 ms | ≈ 66 ms | ≈ 425 ms |
| **fp16 speculative** | **≈ 56 ms** | **≈ 18.5 ms** | ≈ 395 ms |
| q8 speculative | ≈ 35 ms | — | ≈ 225 ms |
| fp32 speculative | ≈ 117 ms | — | ≈ 820 ms |

- The encoder adds about 36 ms (fp32) or 17 ms (fp16) per step (`encoder-spec.md:22`).
- The p99 is **token-bound**: translation bursts of up to 32 tokens per round. Speculative batching does not help there; only a faster
  per-token forward does. Bursts create a short backlog that drains, because the mean is far below 320 ms.
- On an M4 Pro (about 0.45× the time), fp16 speculative comes to a mean of about 25 ms and a p99 of about 180 ms, and fp32 to a mean
  of about 53 ms and a p99 of about 370 ms.

### 5.5 Integration notes
- Feed only frames with index < K of the current segment. K comes from the final sample count (`decode-contract.md §c.3`).
- At segment end, the last encoder step may hold 1–4 frames. Batch whatever exists, then run the flush rounds sequentially.
- Prefix: one forward of q=P with its logits discarded (`MV:230`, `RT:110`).
- `sel` rows only. Never compute lm_head on NEXT rows or prefix rows.
- **[critic] Backlog draining (optional, still exact).** If the engine falls behind and several encoder steps' frames are already
  available, the §5.2 argument holds for any batch of already-emitted frames, not only 4. Batching up to 8 frames (q ≤ 16) drains a
  burst backlog faster. Keep q bounded and apply the KV-clamp rule of §4.4.

### 5.6 [critic] Final (offline) mode on the same engine (missing from the original spec)
The user request includes the "final" mode (`decode-contract.md §f`). Its thinker requirements differ from streaming. The source is
`vapasr/hf/offline_decode.py:25-43`, with eval call sites `EV:480-501`:
- **Input layout.**
  - The header is `pre(8) + <|audio_start|> + K × <|audio_pad|> + <|audio_end|> + mid(5) + ast head(5–11)`.
  - Audio rows sit at positions `a0 = len(pre)+1 = 9 … 9+K−1`. They are the **same causal features**, passed through the adapter
    (`offline_decode.py:32-34`).
  - The rest of the header uses token embeddings. There is no `<DELAY_d>` and there are no NEXT rows.
- **One prefill of the whole header** (q = K + 20..26, i.e. up to ~415 at 31 s), with plain running positions.
  - The **last-row logits of that prefill ARE used** for the first token (`offline_decode.py:38-40`). This differs from the streaming
    prefix, whose logits are discarded.
- **Greedy decode.**
  - q=1 steps, mask the final-mode blocked set (`decode-contract.md §f`; `EV:494-495`), then argmax.
  - Stop at `<|im_end|>` (151645, which is not emitted), or after `K+64` tokens for ast-v1 (`EV:500-501`).
  - Speculative batching does not apply.
- **KV need** ≈ 2K + 90 positions, which fits 2048 up to K ≈ 979.
  - Run it after the streaming segment has finished, reusing the same buffers with `pos ← 0`, or in a second buffer.
  - The prefill takes a different, compute-bound shape. In eager mode that is fine. With `mx.compile`, run it eagerly instead.
- **Cost (projection, not measured):**
  - The ~415-row prefill is about 2 × 0.6 G × 415 ≈ 0.5 TFLOP, i.e. on the order of 0.1–0.2 s on a base M4.
  - Then about one q=1 forward per output token (transcript + translation).

---------------------------------------------------------------------------------------------------------------------------------

## 6. Numerics

### 6.1 Facts
- **Training** used HF Trainer `bf16=True` (`experiments/semcommit_train.py:397`), i.e. autocast bf16 over fp32 master weights.
  - Inside the thinker, Linears and matmuls ran in bf16, and RMSNorm upcasts to fp32 (`QA:62`).
  - `inputs_embeds` are fp32, because embedding is not autocast and audio rows are cast to `E.dtype` (`MV:83`).
  - The residual add `fp32 + bf16 → fp32`. So **the residual stream was fp32 during training**; only matmul I/O was bf16.
- **Evaluation and the reference** ran fp32 throughout, with TF32 off (`decode-contract.md §c.2(6)`). The s3 realtime parity used fp32.
- **[critic] Training numerics also differed from HF in three more ways.** None of them changes the port target, which is the eval:
  HF sdpa, fp32, TF32 off.
  1. **Liger kernels.** They are on by default (`semcommit_train.py:84-85`) and applied at `:260` through
     `vapasr/hf/liger.py:15-21`. RMSNorm (113 modules), SwiGLU MLP (28) and `apply_rotary_pos_emb` are replaced.
     - The Stage-3 smoke that loaded `s4/final` logs `liger: {'rms_norm': 113, 'swiglu': 28, 'rope': 1}`
       (`raw/sources/experiments/2026-10-15-stage3-train-smoke/smoke-train.log:455`).
     - Liger's llama casting mode keeps the HF order (fp32 normalize, cast, then weight), so the fp32-residual conclusion above still
       holds.
  2. **TF32 matmuls on** (`semcommit_train.py:175`). The HF rotary computes its angles with an fp32 matmul with K=1 (`QA:831`).
     Whether cuBLAS rounded those to TF32 during training is **unverified**; no GPU is available here. If it did, training saw RoPE
     angles perturbed by far more than 5e-4, and the model is likely robust to that. Parity, however, is still defined against the
     TF32-off eval.
  3. **Attention.** `--attn-impl varlen` (torch FA2 varlen, `vapasr/hf/varlen_attention.py`). It is not persisted: `config.thinker`
     is the dict captured at creation (`MV:285`) and `set_attn_impl` only mutates the live config (`vapasr/speedup.py:63-71`), so
     eval loads sdpa.

### 6.2 fp16 overflow risk
- fp16 max is 65 504. Qwen-family LMs are known to develop "massive activations": a few hidden dims at the first and delimiter tokens
  reach 10³–10⁴ in the residual stream, mostly produced by an early layer's MLP down_proj.
- In a plain fp16 port, the **residual stream and the down_proj output are stored in fp16**. If any element exceeds 65 504 it becomes
  inf, then NaN through RMSNorm.
  - RMSNorm itself is safe: HF squares in fp32 (`QA:62-63`) and so does `mx.fast.rms_norm`.
  - K/V are post-RMSNorm (k_norm) or projections of normalized input, so they are bounded.
- **Evidence available here:**
  - The E2 checkpoint (same Qwen3-ASR-0.6B lineage) ran fp16 in MLX and in torch MPS with sane, identical transcripts
    (`output-phase1-report.md:123-127`), so no overflow on those samples.
  - s4 is a different full fine-tune (2 more stages, translation, longer outputs) and **was not measured**, since no weights are on
    this machine.
- **Mitigation, measured to cost nothing:**
  - Keep the residual stream `h` in **fp32** and cast only the normalized activations to fp16 for the matmuls. This mirrors the
    training autocast.
  - `SCR/numerics_variants.py`: q=1 step 15.1 ms (fp16-resid32) vs 15.2 ms (plain fp16).
  - The remaining fp16 tensors are the matmul outputs (down_proj, o_proj), the 3072-wide SwiGLU intermediate, and the logits.
  - Guard with an in-graph `isfinite` check on the selected logits rows, which comes almost free with the argmax. On failure, switch
    the session to bf16 (or fp32), rerun the last segment, and log it.
- **[critic] Provenance and a cheaper fallback.**
  - The "massive activations 10³–10⁴" statement above is general literature knowledge stated from memory. Nothing in the repo
    measures it for Qwen3-ASR-0.6B or s4. Treat it as a hypothesis for §6.5 to measure.
  - The fp32 residual does **not** protect a single fp16 matmul output (down_proj, o_proj, or the gate·up product) that exceeds
    65 504 on its own.
  - If §6.5 finds a layer with headroom below 4×, the cheaper fix is to store **that layer's MLP (or down_proj) in bf16**. The bytes
    are the same, so the speed is the same, and it has fp32 range. That is better than switching the whole session.
  - The E2 fp16 precedent is weak evidence: 2 samples (en and ko) plus 4 logit probes, torch MPS fp16 vs MLX fp16
    (`experiments/live/probe_live_mlx.py:12, 17-25`).

### 6.3 bf16 is a worse default than fp16 for greedy parity
- MLX returns logits in the weight dtype before our fp32 cast. bf16 has an 8-bit mantissa, so the logit ulp is 0.25 at |logit| 32–64
  and 0.5 at 64–128, against fp16's 0.03–0.06. Exact ties among top candidates become common, and the first index then wins.
- Random-weight Δlogit vs fp32: bf16 1.9, fp16 0.22 (§2.7). bf16 has no overflow risk and the same speed on M4 (§4.2).
- Use it only as the fallback.
- Either way, **re-score the top-k in fp32**: gather k=8 rows of E, cast to fp32, and dot with the fp32 final-norm output. This
  removes the logit-storage rounding for +0.1–0.4 ms (`SCR/numerics_variants.py`).

### 6.4 Recommendation
| mode | weights | residual / norms | logits | when |
|---|---|---|---|---|
| **default** | fp16 | fp32 | fp16 GEMM → fp32, top-8 fp32 re-score | all machines |
| exact | fp32 | fp32 | fp32 | M4 Pro / M3 Max and up (q=1 forward ≲ 15 ms), or a "parity" toggle |
| fallback | bf16 | fp32 | same as default | only if the fp16 finiteness guard fires |
| fast (opt-in) | q8 g64 | fp16/fp32 | — | base M4 only; not exact; must pass §6.5 first |

Selection heuristic at startup: time 10 forwards at q=1 and q=8. Pick fp32 if `t(q=8) + 23 × t(q=1) < ~450 ms`, which is the p99
group, else the default. Adapter is fp32 always, and the KV cache is in the thinker dtype.
**[critic]** The budget must also cover the encoder step. Both share the GPU and run on one worker (§4.4). The encoder step is
35.6 ms fp32 or 17 ms fp16 on a base M4 (`encoder-spec.md:22`). Also require that the **mean** load fits:
`enc_step + t(q=8) + 2.3 × t(q=1) ≲ 0.5 × 320 ms`, using the s3 mean of 3.29 forwards per group (§5.4). The p99 test alone tolerates
a backlog but does not bound the steady-state load.

### 6.5 Proposed agreement test (run on the MacBook with the real s4 checkpoint)
1. **Inputs.**
   - About 20 utterances per mode: en, ko, de ASR-only and en→de, en→ko, de→en, ko→en at δ4 and δ10. Use FLEURS/CoVoST samples
     with the 1 s tail, plus the 6 clips in `scratchpad/testaudio/`.
   - Compute encoder features **once** with the reference encoder path (`decode-contract.md §g`) and cache them as `.npy`, so that
     the test isolates the thinker.
2. **Reference.** torch fp32 on CPU, `VapAsrForStreamingASR.stream_decode(feats, prefix)` with the s4 `config.json` caps. This is the
   eval rule.
   **[critic]** On the Mac, `stream_decode` calls `torch.cuda.synchronize()` unconditionally (`MV:241`), so patch it to a no-op or
   use `decode_batch`. Also pass `feats` of shape (1, K, 1024). Follow the loading recipe and pitfalls in `decode-contract.md §g`:
   the encoder-cache path, `VAPASR_STAGE_DIR=""`, unset `NEMOTRON_DIR`. `scratch-critic-thinker/spec_vs_stream_decode.py` shows the
   patch and a torch speculative decoder, which can serve as the "speculative" arm of the torch side. Also record, at every decision row along the reference trajectory, the top-2 logits (margin) and the per-layer
   max |h| and max |SwiGLU intermediate|. The latter gives the fp16 headroom = 65 504 / max.
3. **Candidates.** MLX {fp32, fp16-resid32+rescore (default), plain fp16, bf16, q8, q4} × {sequential (`RT` rule), speculative}.
4. **Metrics.**
   - **Teacher-forced per-step agreement**: replay the reference input sequence and compare argmax at every decision row. This
     isolates per-step disagreement from trajectory drift. Report the disagreement rate and the reference margin at each
     disagreement.
   - **Free-running identity**: the share of utterances whose emitted `(k, tid)` lists, `forced` counts and flush rounds are
     identical; the first divergence index; token edit distance.
   - Final transcript WER/CER and translation BLEU/chrF deltas vs reference; emission-time (k) shifts of matched tokens; any
     non-finite values.
5. **Acceptance.**
   - MLX fp32 sequential and speculative: 100 % identical, or divergences only where the reference margin < 1e-3.
   - Default fp16: no non-finite values, headroom ≥ 4×, teacher-forced disagreement only at reference margin < 0.1, and metric deltas
     within the bootstrap noise of the s4 eval.
   - Speculative vs sequential in the same dtype: ≥ 99.5 % identical utterances, divergences only at near-ties.
   - q8/q4: report only.
   - Random-weight numbers here are a mechanics check, **not** a prediction: fp16 99.1 %, fp16+rescore 99.3–99.5 % per-step agreement
     over 580 teacher-forced steps vs fp32. A random model has far more near-ties than a trained one.

---------------------------------------------------------------------------------------------------------------------------------

## Unknowns / risks
1. **Real-weight numerics are unmeasured.** s4 activation magnitudes (fp16 overflow headroom) and the real near-tie rate (fp16/bf16
   flips, speculative-vs-sequential flips) need the checkpoint. §6.5 is the gate. The E2 fp16 precedent is encouraging but is a
   different fine-tune.
2. **Bitwise identity is not achievable across kernel shapes.** Speculative batching, the q=2 merge, top-k re-scoring and MLX vs
   torch kernels are all "equal up to rounding". Exact-greedy equivalence holds as math, and empirically except at near-ties.
3. **Target-machine timings are projections** from bandwidth ratios. Only the base M4 was measured, and it had resident background
   processes and ±1–2 ms noise. Verify on the MacBook at startup with the selection heuristic in §6.4.
4. **Burst latency is token-bound.** p99 of about 22 tokens per 320 ms group (up to 61, cap 32 per round) gives about 0.4 s of thinker
   time on a base M4 fp16. Speculative batching does not fix tails. The emission statistics come from s3/FLEURS read speech with a
   1 s tail; live conversational speech and s4 (more translation output) may be denser.
5. **The checkpoint layout is inferred.** No `thinker.lm_head.weight` key was verified on a `save_pretrained` round trip with this code
   and the decode-contract tiny checkpoint, but not on s4 itself. The s4 `config.json` may also carry the bf16 hub `dtype` field for
   the thinker; it is irrelevant because we cast explicitly. The loader should accept both and assert shapes.
   **[critic]** s4 was saved on mxc with an unrecorded transformers version. The loader must accept either tied alias
   (`embed_tokens` or `lm_head`) and iterate all `*.safetensors` (§2.5).
6. **mlx_lm's `nn.RoPE(base=1e6)` is not HF-exact** (5e-4 at positions ≥ 1024). Do not reuse `mlx_lm` `Model.__call__` for the parity
   engine, or swap its rope.
7. **The fixed KV caps context at 2048 positions.** Segments must stay ≤ 31 s (positions ≈ 19 + 2K + #tokens ≤ about 1 300). Assert
   it, as `RT:88` does.
   **[critic] Corrected.** The ≈ 1 300 figure is typical, not a bound. The runaway worst case is about 8K+100 (§4.3), so the engine
   needs the hard clamp of §4.4 / `audio-ui-spec.md` §1.6, not an assert.
8. **MLX/mlx_lm versions.** The MacBook env must match MLX 0.32.2 (used here) or be re-verified. That covers `mx.fast.rope(freqs=)`,
   dynamic `offset`, `slice_update`, sdpa bool and `"causal"` masks, and `argmax` first-index tie behaviour (verified in
   `decode-contract.md §c.2(5)`).
9. **GPU sharing.** Encoder and thinker run on the same GPU and memory bus. Per-group budget = encoder step (17–36 ms on M4) + thinker.
   Use one MLX worker thread with no concurrent graph building.
10. **Quantized modes change token input embeddings too** (Embedding quantization) and are not exact. Keep them opt-in until they pass
    §6.5.
11. **Benchmarks used random weights.** Bandwidth-bound timings do not depend on values, but the agreement percentages in §6.5 do.
12. **[critic] RoPE table provenance.** A torch-free runtime cannot reproduce the torch table bit for bit: numpy is within 1 ulp,
    and MLX ops are off by 6e-5 (§2.4). This is harmless for parity at the fp32 noise level, but "0.0" must not be used as a
    unit-test tolerance unless the table is built with torch.
13. **[critic] Unmeasured on the real model:** final-mode (§5.6) latency on the MacBook, and whether a per-layer bf16 fallback
    (§6.2) is ever needed.
14. **[critic] Training numerics** (Liger, possible TF32 RoPE rounding; §6.1) mean the trained model was never exposed to the exact
    eval numerics. This does not affect the parity target, but it is one more reason not to expect bitwise agreement with any
    engine.

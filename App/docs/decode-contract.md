# VapAsr s4 (q06-s4-ast) decode contract — for a token-identical MLX re-implementation

Status: written 2026-10-16 from the **working tree** of `/Users/taesookim/Desktop/VAPKT` (not git HEAD), the public Qwen3-ASR-0.6B
tokenizer/config (`scratchpad/hf-qwen3-asr/`), the public Nemotron NeMo yaml (`scratchpad/hf-nemotron/model_config.yaml`), and three
probes run on this Mac (scripts in `scratchpad/scratch-decode/`). The real s4 checkpoint was **not** available; every value that lives
in its `config.json` is marked **[config]** — the implementation must read it from `config.json` and may only *assert* the
reconstructed value given here.

Abbreviations: `SP = scratchpad = /private/tmp/claude-501/-Users-taesookim-Desktop-VAPKT/e6f0e8db-a78d-4321-bce1-f9fa25adbcf1/scratchpad`,
`MV = vapasr/hf/modeling_vapasr.py`, `EV = experiments/eval_single_turn_asr.py`, `ST = vapasr/data/semcommit_tokens.py`.

**[critic] Review 2026-10-16.** An adversarial pass re-derived every load-bearing claim from the working tree. Changes are marked
**[critic]**. The critic's probes are in `SP/scratch-critic/`:
- `probe_live_encoder.py`: yaml encoder (random, 2 layers), streaming vs offline over 52 lengths, plus a live mic simulation.
- `rope_check.py`: MLX RoPE vs HF RoPE.
- `argmax_ties.py`: `mx.argmax` tie rule.
- `tie_check.py`: lm_head tying after `from_pretrained`.
- The writer's `tok_probe.py` was re-run, and every id in §a/§b was reproduced.

What changed:
- Corrected: the provenance of the realtime-parity evidence (it was measured on s3), the RoPE advice for MLX, the qwen_asr
  citations, and the tag example in §e.
- Added: live-engine rules, cross-spec contradictions (new section at the end), and stronger evidence for §a.2.
- Unknown #2 is resolved.

---------------------------------------------------------------------------------------------------------------------------------
## 0. One-paragraph summary

Per utterance: `wav(16 kHz f32) + 16000 zero samples` → K = `int(round(n/16000*12.5))` → Nemotron [56,3] encoder features
(1,K,1024) (offline masked forward, truncated/zero-padded to K) → adapter (LN→Linear→GELU→Linear, fp32) → thinker (Qwen3 0.6B,
28 layers, plain 1-D RoPE θ=1e6). Thinker input: prefix ids (19 ids for en→de δ6, see §2.4), then for each frame k: one audio
embedding, then greedy tokens (blocked ids → −inf, argmax over all 151 936 rows) until argmax = `<NEXT_AUDIO>` (151705) or 32 tokens
in this frame or `6K+64` tokens total; `<NEXT_AUDIO>` is then *fed* (its logits discarded / merged with the next input). After K
frames: up to 8 flush rounds that input `<EMPTY_AUDIO>` (151706) with the same rule, stopping after the first round that emits 0
tokens. Output stream is split by tags: `<SPK_A>/<SPK_B>` open a transcript run, `<TRANS>` (151726) opens a translation run.
Detokenize each run with `tok.decode(ids, skip_special_tokens=True).strip()`.

---------------------------------------------------------------------------------------------------------------------------------
## (a) Special tokens and the config fields the decode needs

### a.1 Token ids (Qwen3-ASR-0.6B base vocab, len(tok)=151705 before additions)

Verified by `scratch-decode/tok_probe.py` (rebuilds the tokenizer exactly like `vapasr/hf/__init__.py:16-41 load_tokenizer`):

| token | id | source / role |
|---|---|---|
| `<\|endoftext\|>` | 151643 | Qwen special. **Not** blocked in streaming (never trained to be emitted). |
| `<\|im_start\|>` | 151644 | prefix; blocked |
| `<\|im_end\|>` | 151645 | prefix; blocked in streaming; **stop token in final mode** |
| `<\|audio_start\|>` / `<\|audio_end\|>` | 151669 / 151670 | final-mode header only; blocked |
| `<\|audio_pad\|>` | 151676 | = `config.audio_pad_id` [config]; placeholder id of audio positions (training sequence + final header); its *embedding is never used* (replaced by adapter output); blocked |
| `<asr_text>` | 151704 | end of language line; **added token with special=False** in the Qwen tokenizer (so `skip_special_tokens` would NOT drop it; it is blocked anyway) |
| `<NEXT_AUDIO>` | 151705 | "nothing more in this frame" (label, fed as input after each round). Never blocked. |
| `<EMPTY_AUDIO>` | 151706 | input for flush rounds after audio ends; blocked |
| `<SPK_A>` / `<SPK_B>` | 151707 / 151708 | in prefix (speaker of the language line) and **emitted** to open a transcript run (unblocked in ast-v1) |
| `<DELAY_1..8>` | 151709..151716 | prefix last token (δ conditioning); blocked |
| `<SPK_3..6>` | 151717..151720 | Phase-2 reserved rows; blocked (mono model) |
| `<ONSET>` | 151721 | Phase-2 reserved; blocked |
| `<EOT>` | 151722 | Phase-2 reserved; blocked (s4 lineage never used `--turn-end`) |
| `<SEM_END>` | 151723 | semantic-commit token; **NOT blocked** in streaming (s4 trained `--asr-only`, SEM never a target and its logit was masked in training) — may in principle be emitted; it is special, so it vanishes on decode |
| `<DELAY_9>` / `<DELAY_10>` | 151724 / 151725 | delay extension (rows copied from `<DELAY_8>` at creation); blocked |
| `<TRANS>` | 151726 | opens a translation run (ast-v1); unblocked |
| rows 151727..151935 | — | unused embedding rows (vocab_size 151936). **Not blocked; the lm_head must keep all 151936 rows** (logits over the full matrix, like torch) |

Code that fixes these ids: Phase 1 list `vapasr/uslm/interleave_data.py:21`; Phase 2 frozen registry `vapasr/data/dialogue_tokens.py:9-10,20`
(and `vapasr/data/schemas/phase2-registry.json`); SEM `ST:22-23`; delay extension `ST:28-29, 37-48`; `<TRANS>` `ST:70-84`.
Addition order is mandatory (ids depend on it): Phase1 → Phase2 → SEM (`ST:26,32-35`) → DELAY_9/10 (`ST:37-48`) → TRANS (`ST:74-84`);
`load_tokenizer` asserts the result equals `config.sp_ids` (`vapasr/hf/__init__.py:39-40`). The s4 checkpoint directory ships its own
tokenizer files; `load_tokenizer` uses them when `tokenizer_config.json` exists there (`__init__.py:22`).

Note — transformers 4.57.6 prints "incorrect regex pattern … set fix_mistral_regex=True" for these local Qwen tokenizer dirs. It is a
false positive (`transformers/tokenization_utils_base.py:2463-2497`: version window 4.57.2 < v < 5.0 triggers the warning; no change
is applied unless the flag is passed). **Do not pass `fix_mistral_regex=True`.**

### a.2 Expected `blocked_ids` for s4 [config]

Reconstruction of the creation chain `from_qwen` (`MV:282-283`) → `add_semantic_tokens(turn=False, lanes=0)` (`MV:193-215`,
`ST:125-131`) → `add_delay_tokens` (`ST:50-66`, adds 151724/151725) → `add_ast_tokens` (`ST:86-100`, removes SPK_A/SPK_B/TRANS):

```
[151644, 151645, 151669, 151670, 151676, 151704, 151706, 151709, 151710, 151711, 151712, 151713, 151714, 151715, 151716,
 151717, 151718, 151719, 151720, 151721, 151722, 151724, 151725]          # 23 ids
NOT blocked: <NEXT_AUDIO> 151705, <SPK_A> 151707, <SPK_B> 151708, <SEM_END> 151723, <TRANS> 151726 (+ every other vocab row)
```
**Use `config.json["blocked_ids"]` verbatim** (the model reads it at `MV:59`). The lineage started at `semcommit-q06-s0/final`
whose creation is not in this repo, so the list above is an expectation to assert/log, not a source of truth.

**[critic] More evidence for the reconstruction:** three lineage logs support it.
- **s1m3-d10 start log** (`raw/sources/experiments/2026-10-08-stage1-trilingual-d10-prep/job-82738-startup.log:2-3`):
  - `<DELAY_9>/<DELAY_10>` were added fresh at s1m3, with "이미 있던 것 없음" ("none existed before").
  - The encoder att_context changed `[56,0] → [56,3]` and was saved to config.
- **s2 ck1000 smoke** (`raw/sources/experiments/2026-10-10-ast-eval-smoke-ck1000/README.md:14`): the config has `ast_format=ast-v1`,
  `<TRANS>=151726`, `runaway_cap 32`, and the tags are unblocked.
- **Stage-3 GPU smoke on the real `s4/final`** (`raw/sources/experiments/2026-10-14-stage3-p3-serialization/README.md:94` with
  `gpu_smoke.py:11-13`): calling `add_semantic_tokens(turn=True, reinit_turn=True)` unblocked `<EOT>`, put it in `sem_registry`, removed it
  from `sem_reserved` and re-initialised its row.
  - The re-init only runs when `<EOT>` is in `sem_reserved` (`MV:209`).
  - So in s4, `<EOT>` was blocked and reserved, and the lineage used turn=False, as reconstructed.
- s4's `config.sem_reserved` is therefore expected to be `["<SPK_3>",…,"<SPK_6>","<ONSET>","<EOT>"]` (`MV:211`).
  - The tiny fixture has `[]`. This is irrelevant to decoding.

### a.3 Every `config.json` field the decode reads

| field | s4 value | read at | meaning |
|---|---|---|---|
| `sp_ids` | dict of §a.1 (22 names) [config] | `MV:58` (next/empty), `EV:366-369` (`<DELAY_d>`, `<SPK_A/B>`, `<TRANS>`), `infer.py:92`, `realtime.py:107` | special ids |
| `blocked_ids` | §a.2 [config] | `MV:59` → `model.blocked`; `realtime.GraphThinker(blocked_ids)` (`stream_rtf.py:112`) | −inf mask each step |
| `prefix_ids` | `[151644, 8948, 198, 151645, 198, 151644, 77091, 198]` = `"<\|im_start\|>system\n<\|im_end\|>\n<\|im_start\|>assistant\n"` [config] | `infer.py:92`, `EV:378`, `stream_rtf.py:125` | chat header (normalized output format) |
| `runaway_cap` | **32** (`ST:98`: `max(old or 8, 32)`; default 8 in `configuration_vapasr.py:17`) [config] | `MV:223`, `realtime.py:103`, `batch_decode.py:63` | max emitted tokens per round |
| `max_flush_rounds` | 8 (default `configuration_vapasr.py:17`; eval also passes `--max-flush 8`) [config] | `MV:223`, `realtime.py:102`, `batch_decode.py:62` | max `<EMPTY_AUDIO>` rounds |
| `frame_hz` / `chunk_s` | 12.5 / 0.08 | `EV:431`, `MV:225` | K rule |
| `encoder_left_context` / `encoder_right_context` | 56 / **3** (set by `--encoder-right-context 3`, `semcommit_train.py:241-245`, `submit-q06-s4-ast.sh:31`) [config] | `MV:73` → `online.py:101` `set_default_att_context_size([56, rc])`; `EV:360-364` | encoder att_context [56,3], chunked_limited. **[critic]** Only `encoder_right_context` is read on the load path. The left context 56 is hard-coded at `online.py:101`, and `encoder_left_context` is read only by the eval override (`EV:361`). The yaml's own default `att_context_size` is also `[56,3]`: it is the first entry of the list at `model_config.yaml:603-611`, confirmed by the critic probe. |
| `encoder_trainable` / `encoder_saved` | True / (False or True) — s4 used `--train-encoder` (`submit-q06-s4-ast.sh:29`, `semcommit_train.py:240`) [config] | `MV:261-268` | encoder weights come from the checkpoint safetensors (`encoder.enc.*`), not the .nemo |
| `ast_format` | `"ast-v1"` (`ST:99`) [config] | `EV:323` | turns on ast-v1 prefix + tag parsing |
| `adapter_d_in/d_out/hidden` | 1024/1024/2048 | `MV:50` | adapter shape |
| `thinker` (dict, incl. `text_config`) | Qwen3-ASR thinker config | `MV:52-54` | 28 layers, d 1024, 16 heads/8 kv, head_dim 128, ffn 3072, rms eps 1e-6, rope_theta 1e6, vocab 151936, tied |
| `lanes` | 0 | `EV:359` asserts 0 | mono model |
| `sem_registry` | `{"<SEM_END>": 151723}` [config] | `offline_decode.py:15` | adds to final-mode blocked set |
| `audio_pad_id` | 151676 | training only | — |
| `delays`, `special_tokens`, `sem_reserved`, `next_weight*`, `semcommit` | informational | — | not used by decode |
| `output_formats` | **absent** for s4 (ast-v1 forbids written form: `ast_dataset.py:124`; `--written-frac` default 0) | — | only normalized header exists |

Tokenizer-only ids the decode needs: `<|im_end|>` (final stop), `<|audio_start|>/<|audio_end|>/<|audio_pad|>`, `<|im_start|>`,
`<asr_text>` (final-mode blocked set, `offline_decode.py:13-22`, `offline_seq.py:16-21`), and the text tokens of the language line.

---------------------------------------------------------------------------------------------------------------------------------
## (b) Prefix construction

### b.1 Rule (ast-v1; used for ALL s4 decodes, including ASR-only)

```
prefix = config.prefix_ids
       + tok(ast_head_text(src, "<SPK_A>", tgt), add_special_tokens=False)["input_ids"]
       + [config.sp_ids[f"<DELAY_{δ}>"]]
ast_head_text(src, spk, tgt) = f"language {spk}{src}" + (f"\ntranslate {spk}{tgt}" if tgt else "") + "<asr_text>"
```
Sources: `vapasr/data/ast_dataset.py:40-42` (text), `EV:375-379` (eval assembly), training `ast_dataset.py:153-159`, `stream_rtf.py:124-126`.
**Do not** use the legacy `infer.prefix_ids` (`infer.py:91-92`, `"language English<asr_text>"` → `[11528, 6364, 151704]`): it is the
pre-AST format and differs token-wise (ast-v1 has `Ġ`(220) + `<SPK_A>` + `English`(22574, no leading space)).

- Language names exactly: `"English"`, `"Korean"`, `"German"` (`EV:93-95`; Kspon rows use `"Korean"`, YODAS `"German"`, Libri `"English"`).
- Speaker tag at eval/inference: **`<SPK_A>` on both lines** (`EV:378`, final `EV:489`, `stream_rtf.py:125`). Training drew A or B
  uniformly per item (`ast_dataset.py:233`), so `<SPK_B>` is in-distribution too (output tag then becomes `<SPK_B>`).
- Trained translation directions: **en→de, en→ko, de→en, ko→en** (`wiki/decisions/decision-ast-monotonic-data.md:31`; s4 packs
  checked at `submit-q06-s4-ast.sh:24-25`: ast-mono-v1 `en-de`, `ko-en`, …; CV4 `en-de`, `en-ko`). Evaluated: FLEURS en→de/en→ko/de→en/ko→en,
  CoVoST en→de/de→en (`EV:94-95`). de→ko / ko→de are syntactically possible but **untrained**.
- Trained delays: δ ∈ {4..10} (`submit-q06-s4-ast.sh:30`, lineage `submit-q06-s1m3-d10.sh`). δ 1–3 rows exist but were not trained in
  the q06 lineage — expose only 4..10. Eval used δ 4, 6, 8, 10 (`slurm/eval-single-turn-queue.sbatch:25`).
- Translation mode share in training: `--ast-frac 0.8` of items that have targets (`submit-q06-s4-ast.sh:31`).
- **[critic] There is no language auto-detect mode.** Every training and eval prefix names the source language explicitly:
  `head_ids(it["lang"], …)` at `ast_dataset.py:153-159`, and `r["lang"]` at `EV:378`.
  - The UI must make the user choose `src` (and `tgt` for AST).
  - A wrong `src` is out of distribution. Nothing in the repo measures its effect.

### b.2 Language-line ids (verified, `tok_probe.py`)

| mode | text after header | ids |
|---|---|---|
| ASR en | `language <SPK_A>English<asr_text>` | `11528, 220, 151707, 22574, 151704` |
| ASR ko | `language <SPK_A>Korean<asr_text>` | `11528, 220, 151707, 42, 45195, 151704` |
| ASR de | `language <SPK_A>German<asr_text>` | `11528, 220, 151707, 32079, 151704` |
| en→de | `…English\ntranslate <SPK_A>German<asr_text>` | `11528, 220, 151707, 22574, 198, 14045, 220, 151707, 32079, 151704` |
| en→ko | `…English\ntranslate <SPK_A>Korean<asr_text>` | `11528, 220, 151707, 22574, 198, 14045, 220, 151707, 42, 45195, 151704` |
| de→en | `…German\ntranslate <SPK_A>English<asr_text>` | `11528, 220, 151707, 32079, 198, 14045, 220, 151707, 22574, 151704` |
| ko→en | `…Korean\ntranslate <SPK_A>English<asr_text>` | `11528, 220, 151707, 42, 45195, 198, 14045, 220, 151707, 22574, 151704` |

Pieces: `language`=11528, `Ġ`=220, `English`=22574, `K`+`orean`=42+45195, `German`=32079, `Ċ`=198, `translate`=14045.
δ ids: 4→151712, 5→151713, 6→151714, 7→151715, 8→151716, 9→151724, 10→151725.

### b.3 Worked example: en→de, δ6 (19 ids)
```
[151644, 8948, 198, 151645, 198, 151644, 77091, 198,          # <|im_start|>system\n<|im_end|>\n<|im_start|>assistant\n
 11528, 220, 151707, 22574, 198, 14045, 220, 151707, 32079, 151704,   # language <SPK_A>English\ntranslate <SPK_A>German<asr_text>
 151714]                                                       # <DELAY_6>
```
Recommended: compute with the checkpoint tokenizer at startup **and** assert equality with this table (any tokenizer drift breaks parity).

---------------------------------------------------------------------------------------------------------------------------------
## (c) Streaming decode loop (canonical rules)

Reference implementations (all equivalent, verified): `MV:218-246 stream_decode` (serial, HF DynamicCache),
`vapasr/hf/batch_decode.py:13-86 decode_batch` (what the eval actually ran, B≤64; serial parity asserted on 2 rows per (lang, δ) — 304
checks passed for s4: `raw/sources/experiments/2026-10-13-q06-s4-ast-eval/README.md`), `vapasr/hf/realtime.py:96-133 realtime_decode`
(fixed KV + merged NEXT; token-identical to stream_decode on 192 decodes / 14 441 tokens on H200 fp32:
`wiki/outputs/output-simulst-baselines-20261012.md:159-167`).
- **[critic] Correction: the 192-decode parity was measured on `q06-s3-ast/final`, not on s4.**
  - Evidence: `"checkpoint": "/soundai/Model/VAPASR/semcommit-q06-s3-ast/final"` in
    `raw/sources/experiments/2026-10-12-q06-s3-ast-rtf/host-parity-summary-graph-stream-float32.json`, and the README of the same folder.
  - The architecture and decode code are the same for s4, so equivalence is expected there too, but it was never measured on s4.
  - The s4 evidence is the serial-vs-batch parity: 304 checks, 2 rows per (lang, δ).

### c.1 Inputs
- `feats`: (1, K, 1024) fp32 encoder output (exactly `model.encode(...)[0]`, see §d). Note `stream_decode` needs the (1,K,D) shape:
  `chunk_embed(feats[None])` = `adapter(feats[None][:, 0])` (`MV:79,227`).
- `ce = adapter(feats.float())` → (K, 1024), then cast to the thinker embedding dtype (`MV:227`, `realtime.py:106`). Adapter =
  `LayerNorm(1024, eps=1e-5) → Linear(1024,2048) → GELU(exact erf) → Linear(2048,1024)` (`MV:17-20`; keys `adapter.net.{0,1,3}.*`).
  Keep the adapter in fp32 even if the thinker is low precision (`infer.py:31` comment; old demo `live.py:95-99`).
- `E` = `thinker.model.embed_tokens.weight` (151936×1024). lm_head is **tied** to it (`MV:44`; the checkpoint stores only
  `thinker.model.embed_tokens.weight`, no `lm_head` key — verified on the tiny checkpoint).
- Constants for s4 eval: `cap = runaway_cap = 32` (`max_per_chunk=0` ⇒ `cap = runaway_cap`, `MV:231`), `max_total = int(6.0*K) + 64`
  (`MV:231`, `batch_decode.py:63`), `max_flush_rounds = 8`, `next_bias = 0` (`EV:335`).

### c.2 Pseudo-code (merged-NEXT form = realtime_decode; tokens identical to stream_decode)
```
pos = 0
forward(E[prefix])                          # logits of the last prefix position are DISCARDED (MV:230, realtime.py:110)
out = []; forced = 0; pending_next = False
def round(k, x):                            # x = ce[k] (audio) or E[EMPTY_AUDIO] (flush)
    inp = [E[NEXT_AUDIO], x] if pending_next else [x]        # one forward of 1 or 2 positions (causal within the call)
    tid = argmax_masked(forward(inp)[-1])
    pending_next = False; n = 0
    loop:
        if tid == NEXT_AUDIO or n >= cap or len(out) >= max_total:
            forced += (tid != NEXT_AUDIO)                     # the non-NEXT argmax is DISCARDED, NEXT is fed instead
            pending_next = True                               # <NEXT_AUDIO> is always fed before the next input
            return n
        out.append((k, tid)); n += 1
        tid = argmax_masked(forward([E[tid]])[-1])
for k in 0..K-1: round(k, ce[k])
for r in 0..max_flush_rounds-1:
    if round(K + r, E[EMPTY_AUDIO]) == 0: break               # stop after the first round that emitted nothing
argmax_masked(l) = argmax_{v in 0..151935}( float32(l)[v], with l[blocked_ids] = -inf, l[NEXT] -= next_bias )
```
Exact details that matter for identity:
1. Every input occupies exactly one KV position, in this order: prefix…, A_0, t…, NEXT, A_1, t…, NEXT, …, A_{K-1}, …, NEXT, EMPTY, t…, NEXT, EMPTY, …
   RoPE positions are the running index 0,1,2,… with no gaps (HF: `position_ids = cache_position` broadcast to 3 identical mRoPE
   streams, `qwen_asr/.../modeling_qwen3_asr.py:1016-1017`; interleaved mRoPE with identical t/h/w positions == plain RoPE,
   `realtime.py:5`, `experiments/live/export_thinker_mlx.py:4`). In MLX use `nn.RoPE(128, traditional=False, base=1e6)` and drop the
   mRoPE `rope_scaling` dict (attention_scaling must be 1.0, asserted `realtime.py:40`).
   **[critic] Correction: do NOT use `nn.RoPE` / `mx.fast.rope(base=1e6)` for the fp32 parity engine.** The convention
   (rotate-half, non-traditional) is right, but the angle/cos computation differs from HF.
   - Measured by `scratch-critic/rope_check.py`, on unit-normal q of shape (1,16,2048,128), max |nn.RoPE − HF| is:
     - 2.6e-5 at positions < 128
     - **5.2e-4 at positions ≥ 1024**
   - That is about 100× the fp32 matmul noise, and segments reach about 900 positions (§d.7).
   - `mlx_lm/models/qwen3.py` uses the same `nn.RoPE`, so it inherits this error.
   - Instead, precompute the table exactly like HF and `realtime.GraphThinker` (`realtime.py:41-43`, `modeling_qwen3_asr.py:821-836`):
     - `inv = 1/(1e6 ** (arange(0,128,2,int64).float()/128))` in fp32
     - `freqs = pos_f32 · inv` in fp32
     - `emb = cat(freqs, freqs)`, then `cos`/`sin` in fp32, computed with torch/numpy on the CPU once at start-up
   - Apply it as `x·cos + rotate_half(x)·sin` in MLX. The measured diff vs HF is **0.0** at all 2048 positions.
2. Feeding `<NEXT_AUDIO>` as its own forward (stream_decode `MV:238`) or merged with the next input (realtime `realtime.py:119`) gives
   the same tokens: the NEXT position's logits are never used. Causal mask inside the 2-position call.
3. The check order is `tid == NEXT` **or** `n >= cap` **or** `len(out) >= max_total`, evaluated **before** appending (`MV:237`). So a
   round emits ≤ 32 tokens; when `len(out)` reaches `max_total`, every later round (incl. flush) emits 0 → flush stops after 1 round.
4. `forced` and `ticks` are statistics only.
5. Argmax tie rule: first maximal index (torch semantics). `mx.argmax` also returned the first index in a quick tie test on this Mac.
   **[critic]** Stronger check (`scratch-critic/argmax_ties.py`): 300 vectors of length 151 936, each with 2–4 planted exact
   ties and some −inf entries. `mx.argmax` matched the first-index rule in 300/300 cases, both on `mx.gpu` and on `mx.cpu`.
6. dtype in the s4 eval: everything fp32, TF32 off (`EV:309-310, 335`; sbatch passes no `--dtype` ⇒ `float32`, `EV:580`), no autocast
   (`stream_decode`/`decode_batch` use none). Logits are cast to fp32 before masking/argmax (`MV:229`, `batch_decode.py:73`).
7. The thinker layer math (for an exact port): RMSNorm computes in fp32 then `weight * x.to(input_dtype)`; q/k RMSNorm over head_dim
   (eps 1e-6) **before** RoPE; GQA 16 q heads / 8 kv heads (q head h uses kv head h//2); SDPA scale 1/√128; SwiGLU (silu) MLP;
   final norm; logits = h·Eᵀ (`modeling_qwen3_asr.py:51-65, 877-958`; `realtime.py:51-70`). Same as `mlx_lm/models/qwen3.py`.
   **[critic] Citation fix.** The thinker's decoder layers instantiate these classes:
   - `Qwen3ASRThinkerTextDecoderLayer` (`modeling_qwen3_asr.py:236-280`) uses `Qwen3ASRTextAttention` (`:143-217`), `Qwen3ASRTextMLP` (`:220-233`)
     and `Qwen3ASRTextRMSNorm` (`:51-65`).
   - The helpers are `rotate_half`, `repeat_kv`, `eager_attention_forward` and `apply_rotary_pos_emb` (`:70-140`). The final norm is at `:979`
     and the rotary embedding at `:781-838`.
   - The `Qwen3ASRThinkerText{MLP,RMSNorm,Attention}` classes at `:840-958` are never instantiated. Their math is identical.

   "Same as mlx_lm qwen3" holds **except RoPE numerics** (item 1). If mlx_lm is reused, swap its rope for the table.

   lm_head tying was verified by `scratch-critic/tie_check.py` on the tiny fixture: after `from_pretrained`, `thinker.lm_head.weight` and
   `embed_tokens.weight` share storage.

### c.3 K, frame count and padding
- K = `int(round(n_samples / 16000 * 12.5))` with `n_samples` **including the 1.0 s tail** (`EV:299, 431`). Python `round` = round-half-even
  on the IEEE double. `infer.iter_stream` uses `n/16000/0.08` (`infer.py:103`), which differs from the eval formula at 23 of 500
  half-way lengths (e.g. n=30080: eval K=24, infer K=23) — **use the eval formula**.
- Encoder frames T' ≥ K usually (measured: en1.wav + 1 s, n=130 807 → mel 817 → T'=103 (T' = f∘f∘f(mel_len), f(L)=⌊L/2⌋+1),
  K=102); features are `out[:min(K,T')]`, rest **zero vectors**
  (pre-adapter), i.e. a padded frame is fed as `adapter(0)` = LN bias path (`online.py:139-141`, `realtime.py:182`).
- Frames are decoded only for k < K; frames ≥ K are dropped.
- **[critic] T' ≥ K always**, so the zero-pad branch is dead code for any n.
  - Offline proof: with f(L) = ⌊L/2⌋+1 ≥ L/2 + ½, we get T' = f³(T) ≥ T/8 + 0.875.
  - Since T = n//160 > n/160 − 1, T' > n/1280 + 0.75 ≥ K + 0.25, so T' ≥ K+1.
  - Streaming (realtime feeding, which drops a trailing piece shorter than 8 mel frames) yields S ≥ K. The encoder spec derives
    this, and `probe_live_encoder.py` confirmed it over 52 lengths, 10 of which had a dropped tail. Example: n = 61 000 → T 381, K 48, offline 49, stream 48.
  - Implementations may `assert T' >= K` and `S >= K`.

### c.4 Settings of the s4 evaluation (what "reference" means)
`slurm/submit-eval-q06-s4-ast.sh` → `slurm/eval-single-turn-queue.sbatch:43-45`:
`--deltas 4 6 8 10 --tail-s 1.0 --max-flush 8 --batch-size 64 --verify --final`, encoder dir `/soundai/Model/nemotron-3.5-asr-streaming-0.6b`
(but s4's own encoder weights override it, §g), dtype fp32, `next_bias=0`, no leading silence, encoder batch 1 (`EV:332-339`),
`encoder_right_context` from config (3). Sets: FLEURS (4 directions), CoVoST2 sub7 (en↔de), FLEURS-en peak-normalized, Kspon, YODAS-de,
LibriSpeech clean/other (`submit-eval-q06-s4-ast.sh:21-31`).
**[critic]** The list was missing **VoxPopuli-Cleaned-AA (English)**. `single-turn-vpaa-kspon-v1` holds VoxPopuli-AA plus Kspon
(`submit-eval-q06-s4-ast.sh:27`, `EV:25-59`). Also note that the FLEURS and CoVoST audio was re-encoded to 16-bit PCM wav at prepare time
(`EV:133,167`). The eval inputs are therefore int16-quantised, while a live mic gives float32.

---------------------------------------------------------------------------------------------------------------------------------
## (d) Encoder features: eval path vs realtime path, and what a live system must do

**Eval path (offline masked forward)** — `NemotronOnline.forward` (`vapasr/features/online.py:129-142`), called per utterance with B=1
(`EV:428-436`): preprocessor on the whole `wav+tail` (eval mode ⇒ no dither, `nemo/.../features.py:427-429`), then
`ConformerEncoder(audio_signal, length)` with default att_context [56,3] `chunked_limited` (masks emulate streaming), fp32, then
truncate/zero-pad to K.

**Realtime path** — `realtime.StreamingEncoder` (`realtime.py:136-183`): mel of the whole file once, then NeMo
`cache_aware_stream_step` per encoder chunk with `streaming_cfg` (measured on this Mac from the yaml, `probe_encoder.py`):
`chunk_size [25, 32]` mel frames, `shift_size [25, 32]`, `pre_encode_cache_size [0, 9]`, `drop_extra_pre_encoded 2` (not on the first
step), `valid_out_len 4`, `last_channel_cache_size 56`, `sampling_frames [1, 8]` (a trailing piece shorter than 8 mel frames is
dropped), `keep_all_outputs` only on the last step; outputs concatenated, truncated/zero-padded to K. Each step yields 4 encoder frames
(320 ms).
**[critic] Precision on `keep_all_outputs`.**
- The exact per-step predicate is `keep_all_outputs = (idx + shift >= T)` (`realtime.py:178`). It is not "the last step".
- When the trailing piece is dropped (`realtime.py:165-166`), **no** step has it set.
- It is numerically irrelevant for [56,3]: a step never produces more than `valid_out_len` = 4 frames (the final partial step gives
  ⌊(r+7)/8⌋ ≤ 4, per `encoder-spec.md`, end-of-utterance item 4).
- Implement the exact predicate anyway.

The critic probe re-measured `streaming_cfg` from the yaml and got the same values: chunk [25,32], shift [25,32], pre-cache [0,9],
drop 2, valid 4, channel cache 56, sampling_frames [1,8]. It also counted **636** `enc` keys (1 layer → 38, 2 layers → 64, so 24 layers → 636)
and 2 `pre` keys.

**Claimed and measured equivalence**: trained s4/s3 on H200: max |Δ| 5.7e-6 and tokens identical on 192/192 decodes
(`output-simulst-baselines-20261012.md:163,167`). **[critic] Correction:** these H200 numbers come from **q06-s3-ast only**
(`2026-10-12-q06-s3-ast-rtf/host-parity-summary-graph-stream-float32.json`, `meta.checkpoint`, `enc_stream_max_abs_diff` 5.6997e-06).
No s4 measurement exists. This Mac, random-weight encoder from the yaml, CPU fp32, en1.wav+1 s:
max |Δ| 8.9e-6, mean 1.4e-6 (26 steps). The mel has `normalize: NA` (no per-utterance statistics: `features.py:479-480` falls through
`normalize_batch` unchanged) ⇒ each mel frame depends only on its own ±256-sample window ⇒ incremental mel == full-file mel.

**What a live engine must do so that per-utterance tokens match the eval/offline reference**
1. Utterance = independent stream: fresh mel state (pre-emphasis `y[0]=x[0]`, then `x[n]-0.97x[n-1]`; STFT `center=True` zero padding of
   256 at the start), fresh encoder cache (`get_initial_cache_state`), fresh thinker KV + prefix. No carry-over between utterances.
2. Same samples: 16 kHz mono float32. Resampling/AGC differences are outside the contract (they change tokens legitimately).
3. Mel frontend = NeMo `AudioToMelSpectrogramPreprocessor` (yaml `preprocessor:`; n_fft 512, win 400, hop 160, preemph 0.97,
   power 2, log(x + 2⁻²⁴), mel_norm slaney, 128 bins, no dither at inference, pad_value 0). Valid mel length = `n // 160`; frames ≥ that
   are zeroed (`features.py:406-410, 482-486`). **Take `window` and `fb` from the checkpoint** (`encoder.pre.featurizer.window` (400,),
   `encoder.pre.featurizer.fb` (1,128,257); s4 saved 638 `encoder.*` keys = 636 encoder + these 2, `train-85952-events.txt`
   `encoder_saved_keys: 638`, matching the 636 enc keys counted here).
4. Mel frame t is final once samples up to `160·t + 256` exist ⇒ encoder step c (frames 4c..4c+3) can run once
   `5120·c + 4096` samples have arrived (first step needs mel frames 0..24). Decoder rounds for those 4 frames may run immediately — the
   decoder is causal, so scheduling never changes tokens.
5. End of utterance (endpoint): append **exactly 16 000 zero samples** (eval protocol `tail_s=1.0`, `EV:299`; training pads the tail so all
   emissions land before audio end, `semcommit_dataset.py:370`, `ast_dataset.py:10`), compute the remaining mel frames with the STFT's
   zero right padding, run the last encoder step(s) with `keep_all_outputs=True`, compute K from the final sample count, **drop encoder
   frames k ≥ K** (a complete 4-frame chunk can contain frame index K), zero-pad if T' < K, then run the flush rounds. Frames produced
   before the endpoint always have k < K (the tail adds ~12.5 frames), so they can be decoded live.

   **[critic] Additions to item 5.**
   - (a) After the endpoint, run the remaining steps with the exact realtime rules:
     - `T = n_final // 160`, with the mel truncated to T.
     - Per step, `length = min(T−idx, size) + pre_cache` and `keep_all_outputs = (idx+shift ≥ T)`.
     - **Stop and drop** a non-first chunk that has fewer than 8 mel rows (`realtime.py:160-180`).
   - (b) **Verified end to end** by `scratch-critic/probe_live_encoder.py` (random yaml encoder, 2 layers, CPU fp32). The mic was simulated
     in chunks of 160, 333, 593, 1280 and 4096 samples. The mel was recomputed on the growing buffer, keeping only frames with
     `160t+256 ≤ n`. Each encoder step ran as soon as its frames were final; at the endpoint, exactly 16 000 zeros were appended.
     - The incremental mel was **bit-identical** to the full-file mel: max |Δ| 0.0.
     - Live features vs the offline masked forward: max |Δ| ≤ 5.6e-6 in all 5 runs.
     - Every pre-endpoint frame index was < K.
   - (c) **What "reference" means for live segments.** Token identity is defined on the **exact final sample sequence** of the segment:
     the samples the engine consumed, plus whatever zero tail the segmenter appends, with K computed from that total.
     - "Exactly 16 000 zeros" is the eval/parity protocol. It is used by `segmentation:"none"` in the audio-UI spec.
     - The audio-UI spec's live policy appends `max(0, 1000 − trailing_pause_ms)` ms of zeros (`audio-ui-spec.md:94`). That segment is
       identical to the reference decode of *its own* sample sequence, not to wav + 1 s.
   - (d) **The segmenter must not cut retroactively.** Every sample already consumed by a finalised mel frame, up to `160·t+255` for the
     last computed frame t, must stay inside the segment. Otherwise the pre-endpoint frames no longer match the final sequence.
     - Simplest rule: segment end = all samples received so far, then append the zero tail.
     - Pre-emphasis restarts at the segment's first sample (`features.py:434`), including any pre-roll.
6. `max_total = 6K+64` depends on the final K and is therefore not causal. A live engine cannot reproduce it exactly; it only matters
   when the reference hit the total cap (`len(out) >= max_total`, never expected for a trained model). Use a generous causal guard
   (e.g. K_max of the max segment) and have tests skip/flag references with `total_cap_hit`.

   **[critic]** `audio-ui-spec.md:134-135` instead applies `6·K+64` with K = frames fed so far. No causal guard reproduces
   `6·K_final+64`:
   - A tighter guard (frames so far) can stop early, where the reference would continue.
   - A looser guard (K_max) can continue where the reference stopped.

   Both diverge only in cap-hit (runaway) regimes. Either is acceptable as a runaway/KV guard, provided that file-mode parity tests
   (which know the final K) use the exact `int(6.0·K)+64` and skip or flag rows with `total_cap_hit`.
   - The per-round cap must be `config.runaway_cap`, which is **32** for s4, not 8 (see the cross-spec section).
7. Keep segments ≤ 30 s (+1 s tail): eval sets are `max30s`; the CUDA realtime path sizes the KV cache at 2048 positions
   (`stream_rtf.py:112`, `realtime.py:88` asserts); positions ≈ 19 + 2K + #tokens.

---------------------------------------------------------------------------------------------------------------------------------
## (e) Output parsing, detokenization, post-processing

Training output format (ast-v1, `ast_dataset.py:7-11, 45-61`; tag insertion `semcommit_dataset.py:100,105`; sequence assembly
`vapasr/uslm/mono_data.py:29-43`): per frame `[A_k] (tag?) tok… <NEXT_AUDIO>`. Within a frame, transcript tokens precede translation
tokens of the same time. A tag is inserted **only when the stream switches** (state `last_spk` persists across frames; also before the
very first emission): `<SPK_x>` before transcript tokens, `<TRANS>` before translation tokens. Example δ4 (decision 9):
`<SPK_A> it <TRANS> Es <NEXT_AUDIO> … <SPK_A> values <TRANS> gemeinsame Werte <NEXT_AUDIO>`.
**[critic] Correction: that example is the pre-correction form.** It was copied from the stale docstring at `ast_dataset.py:9`.
- The decided and tested layout (`wiki/decisions/decision-ast-monotonic-data.md:68-69`, `tests/test_ast_dataset.py:23-30`) is, by chunk:

  | chunk | tokens |
  |---|---|
  | 77 | `<SPK_A> it <TRANS> Es <NEXT_AUDIO>` |
  | 78 | `<SPK_A> is <NEXT_AUDIO>` |
  | 81 | `about <TRANS> geht um <NEXT_AUDIO>` |
  | 86 | `<SPK_A> shared <NEXT_AUDIO>` |
  | 92 | `values <TRANS> gemeinsame Werte <NEXT_AUDIO>` |

- So **no `<SPK_A>` before "values"**: the run was already the transcript after "shared".
- A tag appears only when the run changes, and the `last_spk` state persists across chunks (`semcommit_dataset.py:93,100`).
Transcript = normalized form (EN/DE lowercase, no punctuation, spelled numbers; KO no punctuation); translation = written form
(punctuation/case), from stable-prefix teacher targets (`ast_dataset.py:7-9`).

Parser (`vapasr/data/ast_eval.py:15-24 split_runs`, used by `EV:381-385`): `run = 0`; for each emitted id: if id ∈ {SPK_A, SPK_B} →
`run = 0`, drop; elif id == TRANS → `run = 1`, drop; else append to transcript (run 0) or translation (run 1). Tokens before any
tag count as transcript. The live parser must keep `run` across frames.

Text:
- Transcript hypothesis: `tok.decode(ids, skip_special_tokens=True).strip()` (`EV:461`, final `EV:503`). This removes all
  `special=True` added tokens (incl. `<SEM_END>`); non-special added tokens like `<non_speech>`/`<think>` would survive (never seen).
- Streaming translation text: `word_emission_times` (`ast_eval.py:33-44`): concatenate byte-level BPE bytes of the translation ids
  (special → b''), `decode('utf-8', errors='replace').strip()`; word time = `(k+1)·0.08 s` of the token that completes the word's last
  byte. Final-mode translation: `tok.decode(..., skip_special_tokens=True).strip()` (`EV:393`).
- Live display: decode the **accumulated** ids of each run (Korean/German characters are split across byte tokens; per-token decode
  yields U+FFFD); hold back an incomplete trailing UTF-8 sequence. Collapse whitespace with `re.sub(r"\s+", " ", s).strip()` like
  `infer.Result.text` (`infer.py:61-66`).
- Scoring normalizers (metrics only, never for display): `single_turn_eval.py:12-26` — EN `score_en` (`textnorm.py:85-87`), KO
  `score_ko(x, False)` CER-nospace (`textnorm.py:97-100`), DE `score_de` (`textnorm_de.py:51-52`); translation BLEU/chrF via sacreBLEU,
  KO target tokenized 'char' (`ast_eval.py:59-66`).
- `output_format.py` (outfmt-v1): `normalized` = empty system slot = `prefix_ids` above; `written` = system text
  `"Output format: written (punctuation, capitalization, numerals)."` (header ids
  `[151644, 8948, 198, 5097, 3561, 25, 5326, 320, 79, 72299, 11, 6722, 2022, 11, 7857, 1127, 568, 151645, 198, 151644, 77091, 198]`).
  **s4 was never trained with `written`** (ast-v1 rejects it, `ast_dataset.py:124`) — use `normalized` only.
- ASR-only prefix (no translate line): expected output = `<SPK_A>` + transcript tokens (+ `<NEXT_AUDIO>` per frame). `<TRANS>` is not
  blocked and could appear; the eval discards any translation run in ASR mode (it only keeps run 0 as hypothesis).

---------------------------------------------------------------------------------------------------------------------------------
## (f) Final (offline) mode

Trained with `--offline-frac 0.3` (`submit-q06-s4-ast.sh:28`); sequence (`ast_dataset.py:178-191`, `offline_seq.py:1-30`):
```
pre  = tok("<|im_start|>system\n<|im_end|>\n<|im_start|>user\n") = [151644, 8948, 198, 151645, 198, 151644, 872, 198]
hdr  = pre + [151669] + [151676]*K + [151670] + mid + head
mid  = tok("<|im_end|>\n<|im_start|>assistant\n") = [151645, 198, 151644, 77091, 198]
head = tok(ast_head_text(src, "<SPK_A>", tgt))   (NO <DELAY_d>)      e.g. en→de: 11528, 220, 151707, 22574, 198, 14045, 220, 151707, 32079, 151704
target = <SPK_x> transcript [<TRANS> translation] <|im_end|>
```
Decode (`vapasr/hf/offline_decode.py:25-43`, eval call `EV:480-501`): embed `hdr`, overwrite positions `a0 = len(pre)+1 = 9 … 9+K-1` with
`adapter(feats)` (same causal features as streaming), one prefill, then greedy: mask blocked → argmax → stop at `<|im_end|>` (151645,
excluded) or after `max_new_tokens` = **K + 64 for ast** (`EV:501`; default K//2+32). Blocked set for s4 eval =
`offline_blocked_ids` (`offline_decode.py:13-22`: all `config.sp_ids` values ∪ `sem_registry` ∪ {151669,151670,151676} ∪ {151644, 151704},
minus 151645) **minus {SPK_A, SPK_B, TRANS}** (`EV:494-495`) ⇒ {151644, 151669, 151670, 151676, 151704, 151705, 151706, 151709–151725}.
Positions are again the plain running index (no attention_mask). Output is split with `split_runs` like streaming ⇒ yes, it yields the
translation too (eval reports `delta="final"` BLEU, e.g. FLEURS en→de 30.2 vs 24.3 at δ10: `output-q06-s4-ast-eval-20261013.md`).
Cost: one prefill of K+~25 positions + up to K+64 sequential steps (typically ≈ #transcript+#translation tokens, ~0.3–0.6 per frame
each). It is a post-endpoint "second pass", not real-time; it reuses the streaming encoder features (no re-encode).

---------------------------------------------------------------------------------------------------------------------------------
## (g) Reference harness on THIS Mac (no .nemo, no nemotron-encoder.pt)

Verified end-to-end on a tiny s4-shaped checkpoint (2-layer random thinker, random encoder): `scratch-decode/probe_encoder.py`,
`probe_thinker.py`, `make_golden_tiny.py`. Import chain OK in `~/anaconda3/envs/vapasr-local/bin/python`: `qwen_asr` 0.0.6
(`Qwen3ASRThinkerConfig`/`Qwen3ASRThinkerForConditionalGeneration`, imported inside `MV:52-54`), transformers 4.57.6, NeMo 3.1.0
(import ~4 s warm), safetensors. `load_model(..., device="cpu")` 5 s for the tiny model; `stream_decode`, `decode_batch` and an independent
merged-NEXT loop gave identical tokens/forced/flush counts.

Pitfalls found (each would hang or crash):
1. `scripts/local-env.sh:9-10` sets `VAPASR_ENCODER_CACHE=$HOME/Desktop/VAPKT-models/nemotron-encoder.pt` — a **dataless iCloud file**;
   `NemotronOnline` checks `os.path.exists` (true) and `torch.load`s it ⇒ hang. Always export your own `VAPASR_ENCODER_CACHE` **after**
   sourcing, or don't source it.
2. `NemotronOnline(path=None)` globs `*.nemo` in `MXC_NEMOTRON_DIR`/`NEMOTRON_DIR`/**CWD** (`online.py:90`). Unset both vars and pass
   `encoder_path=<empty dir>` to `load_model`.
3. `VAPASR_STAGE_DIR=""` (else `stage_dir` tries `/dev/shm/...`; `stage.py:10-11`).
4. `stream_decode` calls `torch.cuda.synchronize()` unconditionally (`MV:241`) and so does `realtime.StreamingEncoder` (`realtime.py:154-179`)
   ⇒ on a CUDA-less torch they raise. Patch `torch.cuda.synchronize = lambda *a, **k: None` before decoding, or use `decode_batch`
   (no sync). `realtime.GraphThinker` is CUDA-graph-only; `StreamingEncoder(..., cuda_graphs=False)` works on CPU with the patch.
5. Call `.eval()` (load_model does): `NemotronOnline.set_trainable(False)` only puts `enc` in eval; a training-mode preprocessor adds
   dither 1e-5 noise.
6. `stream_decode` wants `feats` of shape (1, K, 1024) (= `model.encode(...)[0]`), not (K, 1024).

Recipe (MacBook Pro, s4 at `~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync`, CPU fp32 for determinism):
```bash
cd /Users/taesookim/Desktop/VAPKT
export PYTHONPATH=$PWD VAPASR_STAGE_DIR="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset MXC_NEMOTRON_DIR NEMOTRON_DIR
CK=~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync          # must be materialized (ls -lO shows no 'dataless')
export VAPASR_ENCODER_CACHE=$PWD/tmp/s4-encoder-cache.pt        # built below
```
1. Build the cache file **from the checkpoint itself** (exact weights; the yaml supplies only the module configs):
```python
from omegaconf import OmegaConf; from safetensors import safe_open; import torch
cfg = OmegaConf.load(SP + "/hf-nemotron/model_config.yaml")
pre_cfg = OmegaConf.to_container(cfg.preprocessor, resolve=True); enc_cfg = OmegaConf.to_container(cfg.encoder, resolve=True)
pre_state, enc_state = {}, {}
with safe_open(f"{CK}/model.safetensors", "pt") as f:
    for k in f.keys():
        if k.startswith("encoder.pre."): pre_state[k[len("encoder.pre."):]] = f.get_tensor(k)     # featurizer.window, featurizer.fb
        elif k.startswith("encoder.enc."): enc_state[k[len("encoder.enc."):]] = f.get_tensor(k)   # 636 keys
torch.save(dict(pre_cfg=pre_cfg, enc_cfg=enc_cfg, pre_state=pre_state, enc_state=enc_state), cache_path)
```
   (Format = `online.py:20-30 _load_encoder_cache_file`. If `encoder.pre.*` keys are absent, instantiate
   `Serialization.from_config_dict(pre_cfg)` and use its own `state_dict()` — that is what `probe_encoder.py` did.)
   `from_pretrained` then re-applies `encoder.enc.*` from the safetensors (`MV:261-268`; verified "encoder weight from ckpt == loaded: True").

   **[critic] Fixes to the recipe.**
   - (i) **Do not write the 2.4 GB cache into `$PWD/tmp/`.** The repo sits on the iCloud-synced Desktop, which is at risk of
     eviction/dataless. Use a `.nosync` directory instead, for example `~/Desktop/VAPKT-models/cache.nosync/s4-encoder-cache.pt`, next
     to the `.nosync` checkpoint. On the 16 GB Mac mini, use `/Volumes/Samsung_T5`.
   - (ii) Iterate `sorted(glob(f"{CK}/*.safetensors"))` as `MV:265` does, instead of assuming a single `model.safetensors` (shards are
     possible).
   - (iii) Alternative: any `{pre_cfg, enc_cfg, pre_state, enc_state}` cache with the right key set works, because `from_pretrained`
     overwrites all 636 `enc` keys with s4's. That includes a materialised `~/Desktop/VAPKT-models/nemotron-encoder.pt` or
     `scratch-decode/nemotron-encoder-random.pt`.
     - This holds **only if** `config.encoder_trainable or config.encoder_saved` is true. Assert it, because otherwise the random or base
       encoder silently stays.
     - The checkpoint-built cache in step 1 is robust either way.
   - (iv) Misleading error: if the cache load raises for any reason (key mismatch, bad path), `_load_encoder_cache_file` prints
     `nemotron encoder 캐시 무시(...)` and returns None. `NemotronOnline` then raises
     `FileNotFoundError(".nemo 도 인코더 캐시도 없음")` (`online.py:29-30,92-93`). Read the first message.
   - (v) `local-env.sh:10` uses `${VAPASR_ENCODER_CACHE:-…}`, so exporting your own value **before** sourcing also works.
2. Load and decode:
```python
import torch; torch.cuda.synchronize = lambda *a, **k: None
from vapasr.hf.infer import load_model
from vapasr.data.ast_dataset import ast_head_text
from vapasr.data.ast_eval import split_runs
model, tok = load_model(CK, device="cpu", encoder_path=EMPTY_DIR, dtype=torch.float32)
assert model.encoder.enc.att_context_size == [56, 3] and model.config.ast_format == "ast-v1"
wav = np.pad(wav16k, (0, 16000)).astype(np.float32); n = len(wav); K = int(round(n / 16000 * 12.5))
feats = model.encode(torch.from_numpy(wav)[None], torch.tensor([n]), torch.tensor([K]))[0]      # (1,K,1024)
pfx = list(model.config.prefix_ids) + tok(ast_head_text("English", "<SPK_A>", "German"), add_special_tokens=False)["input_ids"] + [model.config.sp_ids["<DELAY_6>"]]
out, forced, ticks, rounds = model.stream_decode(feats, pfx, max_flush_rounds=8)
src, tr = split_runs(out, [model.config.sp_ids["<SPK_A>"], model.config.sp_ids["<SPK_B>"]], model.config.sp_ids["<TRANS>"])
```
   Final mode: copy `AstOfflineFormat` + `oblk` from `EV:484-501` (or `make_golden_tiny.py`).
   Memory: fp32 model ≈ 5 GB resident; `from_pretrained` additionally `load_file`s the whole 4.8 GB safetensors to extract encoder keys
   (`MV:265-267`) ⇒ peak ≈ 10 GB. Fine on 32 GB; on the 16 GB mini run it alone under the heavy lock.
3. Golden dumps for the MLX tests: save `mel` (128×n//160), `feats` (K×1024), `ce`, prefix ids, top-2 logits margins per decision,
   `emitted`, `forced`, `flush_rounds`, final-mode ids (template: `make_golden_tiny.py`).

Prebuilt fixtures (mechanics only — random weights, reconstructed config, NOT s4 quality):
- `SP/scratch-decode/tiny-s4-like/` — s4 key layout & config layout, 2 thinker layers, random encoder (3.0 GB; includes tokenizer files).
- `SP/scratch-decode/nemotron-encoder-random.pt` — yaml-built random encoder cache (2.3 GB, `VAPASR_ENCODER_CACHE` format).
- `SP/scratch-decode/golden-tiny/{en1-en-ge-d6, en1-en-as-d4, ko1-ko-en-d10}/` — `meta.json` (prefix, K, emitted, forced, flush, offline ids,
  decision margins), `mel.npy`, `feats.npy`, `ce.npy`. Random weights hit the per-round cap every frame and the total cap
  (`total_cap_hit: true`, 676 = 6·102+64), and min top-2 margins are 1e-4…6e-4 — good for exercising cap/flush logic, fragile for fp-exactness.

---------------------------------------------------------------------------------------------------------------------------------
## Checklist — invariants an implementation test must assert

Tokenizer / config
- [ ] `sp_ids` from config equal the tokenizer ids; table §a.1 holds (151705…151726); `<asr_text>`=151704, `<|im_end|>`=151645.
- [ ] `blocked_ids` used verbatim from config; log/assert vs §a.2; NEXT/SPK_A/SPK_B/TRANS never blocked in streaming.
- [ ] `runaway_cap == 32`, `max_flush_rounds == 8`, `encoder_right_context == 3`, `ast_format == "ast-v1"`, `lanes == 0`.
- [ ] Prefix ids for every (src, tgt, δ) match §b.2/§b.3 (19 ids for en→de δ6); `<SPK_A>` on both lines; δ ∈ 4..10.
Encoder / features
- [ ] K = `int(round(n/16000*12.5))` on n incl. the 16 000-sample tail (round-half-even); test n=30080 → 24.
- [ ] mel: no dither, `normalize NA`, valid length n//160, columns ≥ n//160 zero; window/fb from checkpoint; |mel − torch| ≤ 1e-4.
- [ ] encoder output truncated/zero-padded to K; padded rows are zeros *before* the adapter.
- [ ] MLX streaming encoder vs torch offline masked forward: max |Δ| ≲ 1e-4 (torch↔torch is 6–9e-6).
- [ ] Each utterance starts from fresh mel/encoder/KV state.
Thinker / loop
- [ ] adapter in fp32 (LN eps 1e-5, exact GELU); lm_head tied, 151 936 logits, fp32 before masking.
- [ ] RoPE position = running KV index from 0 (prefix first), plain RoPE θ=1e6, rotate-half convention.
- [ ] first-step logits after the prefix are discarded; NEXT is fed (alone or merged) before every audio/EMPTY input except the first audio.
- [ ] stop rule order (NEXT | n ≥ 32 | total ≥ 6K+64) checked before appending; forced token discarded.
- [ ] flush: ≤ 8 rounds of EMPTY, break after the first round with 0 emissions.
- [ ] `emitted` [(k, id)], `forced`, `flush_rounds` equal the torch CPU-fp32 `stream_decode` reference on the golden set; on mismatch,
      report the first diverging decision and its top-2 margin (expect divergences only at margins ≲ 1e-3 when MLX runs fp32; for
      fp16/bf16/quantized thinkers measure agreement rate + WER/BLEU instead of demanding identity).
- [ ] Live/chunked feeding of a file (any audio-chunk size) reproduces the offline-file tokens exactly when given digital zeros as tail.
Output
- [ ] `split_runs` semantics with run state across frames; tokens before the first tag → transcript.
- [ ] **[critic]** RoPE: the MLX cos/sin table equals the HF table bit for bit at positions 0..2047 (`rope_check.py`), and `nn.RoPE` is
      not used in the parity engine.
- [ ] **[critic]** lm_head is tied: logits use `embed_tokens.weight`. If the safetensors ever contains `thinker.lm_head.weight`, assert
      that it equals the embedding.
- [ ] **[critic]** `assert T' >= K` (offline) and `S >= K` (streaming); `keep_all_outputs == (idx+shift >= T)` per step; a non-first chunk
      with fewer than 8 mel rows is dropped.
- [ ] **[critic]** The live engine never cuts samples already consumed by finalised mel frames. Per-segment K is computed from the final
      sample count *including* the segmenter's zero tail.
- [ ] **[critic]** Tag rendering test: the §e example sequence produces transcript `it is about shared values` and translation
      `Es geht um gemeinsame Werte` with no extra tag before `values`.
- [ ] **[critic]** `config.runaway_cap == 32` is asserted at load; never hard-code 8.
- [ ] transcript/translation text = `tok.decode(run_ids, skip_special_tokens=True).strip()`; display decodes accumulated ids.
- [ ] final mode: header layout §f, audio positions 9…9+K−1, stop at 151645, cap K+64, blocked set §f.

## Unknowns

1. The actual s4 `config.json` (blocked_ids, runaway_cap, prefix_ids, sp_ids, encoder_saved) — reconstructed from code, not read. The
   q06-s0 starting checkpoint's creation (from_qwen version at that time) is not in the repo.
2. Whether the s4 safetensors indeed contain `encoder.pre.featurizer.{window,fb}` (inferred from `encoder_saved_keys: 638` = 636 + 2).
   If absent, freshly built NeMo buffers are used — equal to the .nemo ones only if the filterbank construction is version-stable.
   **[critic] Resolved; this is near-certain now.** The evidence chain:
   - `train-85952-events.txt` (s4 final parity) reports `encoder_keys: 638` in memory and `encoder_saved_keys: 638` in the file.
   - The critic probe counts 636 `enc` keys plus 2 `pre` keys (`featurizer.window` and `featurizer.fb`). `NemotronOnline.spec_aug` has
     no state.
   - The same parity record has `value_diff: []`. The reload takes `pre` from the base `.nemo`, so the saved window/fb equal the `.nemo`
     buffers to within 1e-6.
3. Cross-hardware identity: the s4 eval ran on H200 fp32 (TF32 off) with batched decode (B≤64) — never compared to Mac CPU/MLX. Near-tie
   argmax flips are possible; acceptance must be defined against a Mac CPU-fp32 torch reference.
4. Live-mic conditions (real room silence instead of digital zeros for the tail, mic gain, resampling) are outside the token-identity
   contract; FLEURS showed the model is sensitive to very low input level (`ast_dataset.py:14-16`, gain augmentation was English-only).
5. `max_total` cap is non-causal (depends on final K); exact only if never reached.
6. MLX `argmax` tie-breaking was checked on one toy vector only; `mx.fast.scaled_dot_product_attention`/RoPE numerics vs torch are
   unmeasured for this model.
   **[critic] Update:**
   - argmax: now checked on 300 vocab-size vectors, on GPU and CPU (item c.2-5). OK.
   - RoPE: measured. `nn.RoPE` deviates by up to 5.2e-4, while a table port has 0 deviation (item c.2-1).
   - Still unmeasured: `mx.fast.scaled_dot_product_attention` and MLX matmul accumulation order vs torch CPU on the real 28-layer
     thinker.
7. Behaviour for untrained directions (de↔ko) and δ < 4 is undefined; segment lengths > 31 s are untested (KV ≈ 19 + 2K + tokens).
8. Whether the checkpoint's own tokenizer files were saved by a different transformers version — harmless if §b.2 ids are asserted.
9. **[critic]** The SEM row was never trained to stay low. Every lineage stage after q06-s0 used `--asr-only` (`submit-q06-s4-ast.sh:28`,
   `semcommit_train.py:302`), so all rows were SEM-neutral and `sem_mask` removed `<SEM_END>` from the softmax (`MV:132-137`). Yet
   `<SEM_END>` is **unblocked** at decode (§a.2).
   - Its emission rate on real speech is unknown. The eval would include any emissions, since the reference has the same behaviour.
   - Keep it unblocked for parity. Log it if it is ever emitted (it is special, so it is invisible in the text).
10. **[critic]** Effect of a wrong or unsupported source-language prefix, and of de↔ko pairs: not measured (see §b.1).

---------------------------------------------------------------------------------------------------------------------------------
## [critic] Cross-spec consistency (vs `spec/audio-ui-spec.md` and `spec/encoder-spec.md`, as of 2026-10-16 00:41)

Those files belong to other agents and were not edited. Where this contract and they disagree, the source code below decides.

1. **Per-round cap.**
   - `audio-ui-spec.md:134` says "runaway caps (8 tokens/round, `config.runaway_cap`, `configuration_vapasr.py:17`)".
   - **That is wrong for s4.** 8 is only the class default. `add_ast_tokens` raises the cap to `max(old, 32)` (`ST:98`), and the s2 smoke
     showed `runaway_cap 32` (`2026-10-10-ast-eval-smoke-ck1000/README.md:14`).
   - The engine must read `config.runaway_cap` (32) and must never hard-code it. With 8, any frame carrying more than 8 tokens
     (transcript + translation + tags) would be force-cut, and tokens would diverge from the reference.
2. **Total cap.**
   - `audio-ui-spec.md:135` applies `6·K+64` with K = frames fed so far. This contract (§d.6) suggested a generous guard.
   - Neither equals the reference's `6·K_final+64`. Both are fine as runaway guards. Parity tests must use the exact formula with the
     final K and skip `total_cap_hit` rows (§d.6).
3. **Prefix length.**
   - `audio-ui-spec.md:132` says "prefix (≈ 25–40 ids)".
   - The actual s4 prefixes are **14–20 ids**: 8 header ids + 5–11 language-line ids + 1 delay id (§b.2). Examples: ASR en/de = 14,
     ASR ko = 15, en→de and de→en = 19, en→ko and ko→en = 20.
   - The KV budget there is therefore conservative, which is harmless.
4. **Zero tail.**
   - `audio-ui-spec.md:94` appends `max(0, 1000 − trailing_nonspeech_ms)` ms of zeros in live mode. Its `segmentation:"none"` mode
     appends exactly 16 000 zeros.
   - This is consistent with §d.5(c): identity is defined on the final sample sequence, and "wav + 16 000 zeros" is the parity protocol.
5. **Encoder end of utterance.**
   - `encoder-spec.md` (end-of-utterance, item 3) writes `keep_all_outputs=True` for every post-endpoint step, while `realtime.py:178`
     uses `idx+shift ≥ T`.
   - This is numerically equivalent for [56,3], since steps never exceed `valid_out_len` = 4. Prefer the realtime predicate.
   - The other overlapping claims agree with this contract and with the critic probe: mel finality `n ≥ 160t+256`, steps at
     `4096+5120c` samples, the drop of fewer than 8 trailing rows, S ≥ K, window/fb from the checkpoint, and 636+2 keys.
6. **Dither during training.** `encoder-spec.md:54` notes that training ran the preprocessor with dither on, because
   `NemotronOnline.train()` recurses into `pre` (`online.py:127-128`). The eval and this contract run with dither off (`.eval()`),
   which is the reference behaviour. This is consistent.

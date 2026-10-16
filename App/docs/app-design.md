# App design contract (binding for all implementers) — 2026-10-16

This file fixes the module layout and the APIs between modules so that several agents can implement in parallel.
Details of *behaviour* live in the four specs next to this file; this file wins on any conflict about file names / APIs.

- `decode-contract.md` — token-identical decode rules for s4 (prefix, caps, flush, K, tail, parsing, final mode, torch reference harness)
- `encoder-spec.md` — log-mel + Nemotron FastConformer cache-aware streaming, MLX reference `scratch-encoder/mlx_encoder_ref.py`
- `thinker-spec.md` — adapter + Qwen3 thinker in MLX, fixed KV, speculative audio batching, numerics
- `audio-ui-spec.md` — segmentation/VAD/gain, latency accounting, browser capture, WS protocol, server threads, UI, packaging

## 0. Facts confirmed from the real s4 checkpoint (2026-10-16)

- Location on this Mac mini: `~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/final/` (materialized, NOT dataless; `.nosync` ⇒ never
  evicted). Files: `config.json` (733 KB, mostly the `semcommit` fingerprint), `model.safetensors` (4,837,811,880 B, all F32),
  `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `added_tokens.json`, `special_tokens_map.json`.
  The run dir also has `../parity.json`, `../results.json`.
- Same path on the user's MacBook Pro (M3/M4 Pro or Max, ≥ 32 GB). The user wants to run the app there.
- safetensors keys: `encoder.enc.*` (636), `encoder.pre.featurizer.fb` (1,128,257), `encoder.pre.featurizer.window` (400,),
  `adapter.net.{0,1,3}.{weight,bias}` (LN 1024; Linear 1024→2048; Linear 2048→1024), `thinker.model.*` (310; embed 151936×1024, tied, no lm_head).
- config.json: `blocked_ids` = the 23 ids of decode-contract §a.2 (exact match), `runaway_cap` 32, `max_flush_rounds` 8, `delays` [4..10],
  `encoder_right_context` 3, `encoder_left_context` 56, `ast_format` "ast-v1", `prefix_ids` [151644, 8948, 198, 151645, 198, 151644, 77091, 198],
  `frame_hz` 12.5, `chunk_s` 0.08, `audio_pad_id` 151676, `sem_registry` {"<SEM_END>": 151723}, `thinker.text_config` = Qwen3 0.6B
  (28 layers, 1024, 16/8 heads, head_dim 128, ffn 3072, eps 1e-6, rope_theta 1e6, vocab 151936, tied).

## 1. Decisions

1. **No conversion step.** The engine loads the raw s4 checkpoint dir (`<dir>` or `<dir>/final`) directly with `mx.load` and casts per the
   selected precision at load time. (4.8 GB read from internal SSD takes seconds; fp32 "exact" mode needs the fp32 weights anyway.)
   There is no `convert.py`.
2. **Runtime = MLX only** (+ numpy, tokenizers, aiohttp, soxr). Never import torch / transformers / nemo / mlx_lm / vapasr at runtime.
   `tools/make_reference.py` is the only file that uses the research stack (torch, NeMo, vapasr) and runs only in the conda env.
3. Encoder config ships as `App/assets/nemotron_encoder.json` = `{"preprocessor": {...}, "encoder": {...}}` extracted from the public NeMo yaml
   (`scratchpad/hf-nemotron/model_config.yaml`, sections `preprocessor:` and `encoder:`), so the runtime needs no yaml.
4. Segment = one eval utterance (fresh mel/encoder/KV/prefix), ended with a zero tail so that ≥ 1.0 s of post-speech audio is decoded
   (audio-ui-spec §1.4 tail rule). `segmentation:"none"` + `gain:"off"` + 16 kHz input = exact eval protocol (wav + 16000 zeros).
5. Precision is a load-time option `--precision {auto,fp32,fp16}` for the thinker (+ `--encoder-precision {fp32,fp16}`).
   fp16 thinker = fp16 weights, fp32 residual stream/norms, fp32 logits, top-8 fp32 re-score, finiteness guard (thinker-spec §6).
   Defaults are decided by the integration stage from measured agreement + speed on real weights.
6. Speculative audio batching on by default (`--no-speculative` to disable). Exactness argument: thinker-spec §5.
7. One worker thread owns every MLX object (MLX arrays cannot cross threads). Event loop never touches MLX.
8. Port 8796 (scan 8796–8805, skip 8765 and 8790 — an old demo server is running on 8790; never kill it). Bind 127.0.0.1.
9. venv for the runtime: `~/.vapkt-app/venv` (outside the iCloud Desktop). `PYTHONPYCACHEPREFIX=~/.vapkt-app/pycache`.
10. UI text Korean; code comments Korean (short); identifiers English.

## 2. Layout and ownership

```
App/
  README.md                    Korean quickstart + architecture + measurements          (written at the end)
  requirements.txt             runtime pins (mlx, mlx-metal, numpy, tokenizers, aiohttp, soxr)            [W5]
  requirements-dev.txt         pytest, soundfile, safetensors (tools/tests only)                         [W5]
  setup.sh  run.sh                                                                                       [W5]
  server.py                    aiohttp app, WS protocol, worker thread, stats, logs                      [W5]
  audio_pipeline.py            Resampler(soxr), framer, VAD/Segmenter, gain — pure numpy                 [W4]
  text_stream.py               TextAssembler (tag routing + UTF-8-safe incremental text) — pure          [W4]
  assets/nemotron_encoder.json                                                                           [W1]
  engine/
    __init__.py                exports: Engine, SegmentDecoder, Round, find_model, ModelNotFound         [W3]
    checkpoint.py              find_model(), dataless-safe checks, load config/tokenizer, load weights   [W3]
    tokens.py                  SpecialIds, LANGS/PAIRS/DEFAULT_DELAYS, prefix_ids(), offline header, token_bytes table [W3]
    frontend.py                streaming log-mel                                                         [W2]
    encoder.py                 MLX FastConformer + EncoderStream                                         [W2]
    thinker.py                 Adapter + MlxThinker (fixed KV, forward(x,start,sel)->ids)                [W3]
    decoder.py                 SegmentDecoder (rounds, speculative batching, caps, flush, final mode)    [W3]
    fake.py                    FakeEngine (same API, deterministic, no MLX weights)                      [W4]
  web/  index.html app.js worklet.js style.css                                                           [W5]
  tools/
    make_reference.py          torch/NeMo fp32 CPU reference → golden dir (research env only)            [W1]
    verify_engine.py           MLX engine vs goldens: features, teacher-forced, free-running, precision table [W3 (+W2 encoder part)]
    bench.py                   file → engine timings (enc/dec ms, chunk p50/p95, RTF), real-time pacing  [W3]
    client_file.py             headless WS client: wav → server at 1×/max; prints deltas, latency, JSON  [W5]
  tests/                       pytest; no model weights needed (fake engine / random tiny weights)       [all]
```

## 3. Module APIs (exact signatures)

### 3.1 engine/frontend.py + engine/encoder.py  [W2]
```python
class EncoderModel:
    def __init__(self, weights: dict[str, "mx.array"], cfg: dict, right_context: int = 3, dtype=mx.float32): ...
        # weights: checkpoint tensors with the leading "encoder." removed, i.e. "pre.featurizer.fb", "pre.featurizer.window", "enc.<...>"
        # cfg: App/assets/nemotron_encoder.json content
    def new_stream(self) -> "EncoderStream": ...
    def warmup(self) -> None: ...                       # run first-step and steady-step shapes once
class EncoderStream:                                    # one per segment, fresh mel + caches
    samples_fed: int                                    # real + zero samples appended so far
    frames_out: int                                     # encoder frames emitted so far
    def feed(self, pcm: "np.ndarray") -> "mx.array": ...    # float32 16 kHz, any length; runs every chunk that became complete
                                                        # (4096 + 5120·c samples) → (n, 1024) fp32 new frames, n ∈ {0,4,8,...}
    def finish(self, zero_tail_samples: int) -> tuple["mx.array", int]: ...
        # append zeros, compute K = int(round(n_total/16000*12.5)), run remaining full chunks and the final short step
        # (keep_all_outputs), return (remaining frames truncated so that frames_out_total ≤ K, K). Caller zero-pads if total < K.
    last_step_ms: float                                 # wall ms of the most recent chunk step(s), measured after mx.eval
def offline_features(model: EncoderModel, wav: "np.ndarray", tail_samples: int = 16000) -> "mx.array": ...  # (K,1024), via the streaming path
```
Note decode-contract §d.5 and audio-ui-spec §5.2: frames emitted by `feed()` before the end are always < K (tail ≥ 200 ms); `finish` must not
return frames with index ≥ K.

### 3.2 engine/thinker.py  [W3]
```python
class Adapter:  def __init__(self, w: dict); def __call__(self, frames_fp32: "mx.array") -> "mx.array"   # (n,1024) fp32 → (n,1024) fp32
class MlxThinker:
    def __init__(self, w: dict, text_cfg: dict, blocked_ids: list[int], dtype=mx.float16, max_len: int = 2048,
                 resid_fp32: bool = True, rescore_k: int = 8): ...
    def embed(self, ids: list[int]) -> "mx.array": ...           # (n, 1024) in model dtype
    def forward(self, x: "mx.array", start: int, sel: list[int] | None = None) -> list[int]: ...
        # x (q,1024); writes KV at start..start+q-1; returns masked argmax ids at rows sel (default [q-1]); one host sync
    def forward_logits(self, x, start, sel) -> "np.ndarray": ...  # fp32 logits (for verification only)
    max_len: int
```

### 3.3 engine/decoder.py + engine/__init__.py  [W3]
```python
@dataclass
class Round:
    k: int                 # audio frame index (k ≥ K ⇒ flush round)
    ids: list[int]         # emitted ids this round (tags included; <NEXT_AUDIO> excluded)
    forced: bool
    is_tail: bool          # frame lies in the zero tail (k ≥ first tail frame) or flush
    is_flush: bool
    need_samples: int      # segment sample count this round needed (4096 + 5120*(k//4)); flush: n_total
    enc_ms: float          # encoder ms attributed to this round (first round of a chunk carries the chunk's enc time)
    dec_ms: float
    kv_full: bool = False

class Engine:              # constructed, used and destroyed on the worker thread only
    def __init__(self, model_dir: str | None = None, kv_capacity: int = 2048, precision: str = "auto",
                 encoder_precision: str = "auto", speculative: bool = True): ...
    info: dict             # name, dir, precision, encoder_precision, kv_capacity, load_ms, speculative, mlx version, device
    langs: list[str]; pairs: list[tuple[str,str]]; delays: list[int]; defaults: dict   # defaults per audio-ui-spec §2.1 [critic]
    spk_ids: tuple[int,int]; trans_id: int
    def warmup(self) -> dict: ...
    def new_segment(self, src_lang: str, tgt_lang: str | None, delay: int) -> "SegmentDecoder": ...
    def token_bytes(self, tid: int) -> bytes: ...     # b"" for every special/added id and ids ≥ len(tokenizer)
class SegmentDecoder:
    positions: int; kv_capacity: int; samples_fed: int; K: int | None
    def feed(self, pcm: "np.ndarray") -> list[Round]: ...
    def finish(self, zero_tail_samples: int) -> list[Round]: ...
    def final_redecode(self, max_new: int | None = None) -> tuple[list[int], float]: ...   # offline (final) mode on this segment's features; (ids, ms)
```
Caps: per round `config.runaway_cap`; total `6*K_cap+64` while streaming (K_cap from hard_max 30 s + 1 s tail), exact `6*K+64` inside
`finish()`; hard KV clamp (`kv_full`). Speculative batching per thinker-spec §5 (rollback = smaller `start`).

### 3.4 text_stream.py  [W4]
```python
class TextAssembler:
    def __init__(self, token_bytes: Callable[[int], bytes], spk_ids: tuple[int,int], trans_id: int, keep_tr: bool = True): ...
    def push(self, ids: list[int]) -> tuple[str, str]: ...    # (src_delta, tr_delta) display text appended (whitespace rules of audio-ui-spec §4.4)
    def finish(self) -> tuple[str, str, dict]: ...            # (src_text, tr_text, stats{replaced_bytes, tr_dropped}) ; src_text == "".join(deltas)
```

### 3.5 audio_pipeline.py  [W4]
```python
@dataclass
class SegParams:  # defaults = audio-ui-spec §1.4/§1.5
    segmentation: str = "vad"     # "vad" | "none"
    pre_roll_ms: int = 480; end_pause_ms: int = 800; soft_max_s: float = 18.0; end_pause_soft_ms: int = 256
    hold_from_s: float = 23.0; hold_ms: int = 640; hard_max_s: float = 25.0; tail_s: float = 1.0
    lead_in: bool = True; gain: str = "auto"; target_db: float = -26.0
@dataclass
class Cmd:
    kind: str                      # "seg_start" | "audio" | "seg_end"
    seg: int
    pcm: "np.ndarray | None" = None  # float32 16 kHz, gain already applied (seg_start may carry the pre-roll/lead-in burst)
    info: dict = field(default_factory=dict)
    # seg_start.info: t_start, reason, pre_roll_ms, lead_in_ms, gain_db
    # seg_end.info:   t_end, reason, zero_tail_samples, voiced_s
class AudioPipeline:
    def __init__(self, in_rate: int, params: SegParams): ...
    def push(self, pcm: "np.ndarray") -> list[Cmd]: ...           # native-rate float32 mono
    def stop(self, reason: str = "stop") -> list[Cmd]: ...        # flush soxr (last=True), end open segment
    def force_end(self, reason: str) -> list[Cmd]: ...            # e.g. "kv_guard" requested by the worker
    def vad_snapshot(self) -> dict: ...                           # fields of the `vad` event
    def counters(self) -> dict: ...                               # audio_rx_s, audio_seg_s, audio_skipped_s, audio_held_s, lead_in_s, zero_tail_s, limited_samples
```

### 3.6 engine/fake.py  [W4]
`FakeEngine` implements the `Engine`/`SegmentDecoder` API above with deterministic tokens (e.g. emits a few ASCII "words" + a `<TRANS>` run
per N frames), configurable per-round sleep, a tiny token_bytes table; no MLX weights. Used by server tests and `--engine fake`.

## 4. Verification data (golden) — produced by W1

`GOLD = /Volumes/Samsung_T5/vapkt-models/app-work/golden-s4/` (T5 — the internal disk has ~26 GB free; keep big files off it).
Per case `<GOLD>/<case>/`: `meta.json` (wav path, src, tgt, delta, n_real, n_total (with tail), K, prefix_ids, emitted [[k,id],...], forced,
flush_rounds, total_cap_hit, src_text, tr_text, final_ids/final_src/final_tr when computed, decision margins along the trajectory
(top-2 logit gap per decision row), per-layer max|h| (fp16 headroom), runtime), `feats.npy` (K,1024 fp32), `mel.npy` (T,128 fp32),
`wav.npy` (exact float32 samples incl. tail). Plus `<GOLD>/index.json` listing cases, and `<GOLD>/DONE` when complete.
Reference = torch fp32 **CPU**, `VapAsrForStreamingASR.stream_decode` with the s4 config caps (decode-contract §g recipe).

## 5. Rules for every agent

- Python for dev/tests: `~/anaconda3/envs/vapasr-local/bin/python` (has mlx 0.32.2, numpy, tokenizers, aiohttp, soxr, safetensors, pytest, torch, nemo).
  Do NOT `source scripts/local-env.sh` (it points VAPASR_ENCODER_CACHE at a dataless file → hang). Set env vars yourself.
- The ONLY readable path under `~/Desktop/VAPKT-models/` is `~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/` (and below).
  Everything else there is iCloud-dataless and hangs on read. Before reading any other file outside the repo, check `ls -lO` for `dataless`.
- Heavy processes (loading the s4 weights, NeMo, any 0.6B model, benchmarks) must run as
  `/usr/bin/lockf -k -t 14400 /private/tmp/claude-501/-Users-taesookim-Desktop-VAPKT/e6f0e8db-a78d-4321-bce1-f9fa25adbcf1/scratchpad/heavy.lock <cmd>`
  — one at a time on this 16 GB machine. Benchmarks are only meaningful when nothing else heavy runs (the lock guarantees that).
- Large files (> 200 MB) go to `/Volumes/Samsung_T5/vapkt-models/app-work/`, never the internal disk or the repo.
- Do not modify repo files outside `App/`. No git commit/push. Do not kill processes you did not start. Do not use ports 8765/8790.
- Write tests under `App/tests/` (pytest, fast, no weights) and keep heavy verifications in `App/tools/` scripts.
- Korean for UI strings and short code comments; English identifiers.

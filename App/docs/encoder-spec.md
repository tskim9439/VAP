# Nemotron 3.5 streaming FastConformer encoder: MLX porting spec

Scope: the audio frontend and encoder of the VapAsr Stage-2 AST checkpoint `q06-s4-ast` (the NeMo 3.1
`AudioToMelSpectrogramPreprocessor` and `ConformerEncoder`), ported to MLX for a live microphone app on Apple Silicon.
It covers the exact config in `scratchpad/hf-nemotron/model_config.yaml` (preprocessor ~L565, encoder ~L585) with
`att_context_size [56,3]` (s4). Every rule also holds for `[56,0]`. The adapter and thinker are out of scope.

- **Source of truth:** NeMo `3.1.0+6989ae9fc`, installed at
  `~/anaconda3/envs/vapasr-local/lib/python3.11/site-packages/nemo/collections/asr/` (abbreviated `nemo/asr/` below),
  together with how vapasr uses it. torch 2.14.0, numpy 2.4.6, mlx 0.32.2.
- **Evidence scripts** live in `scratchpad/scratch-encoder/`:
  - `encoder_probe.py` builds the NeMo modules from the yaml with random weights and fixed seeds. It covers the
    preprocessor, streaming cfg, state_dict, NeMo offline vs stream and MLX.
  - `encoder_probe2.py` uses the same weights. It covers the exact feeding rule on NeMo, MLX fp32/fp16/bf16 and timing.
  - `mlx_encoder_ref.py` is an MLX/numpy reference implementation of this spec. It is executable and verified.
  - Results are in `out/probe-results.json`, `out/probe2-results.json`, `out/state_dict.tsv`, `out/modules.txt`,
    `probe.log` and `probe2.log`.
- **Bottom line:** the streaming procedure below reproduces NeMo's offline masked forward. With random weights the
  maximum absolute difference was **1.07e-5** for NeMo streaming vs. offline (§7). The MLX fp32 reference matched
  offline to **1.1e-5**, including the full live path from PCM. One exception: for [56,0], NeMo's own streaming buffer
  is *not* exact at the start (§2).
- **Measured cost:** on a base M4, one 320 ms encoder step costs 35.6 ms in fp32 and 17 ms in fp16/bf16 (§8.3).
- **[critic] Review 2026-10-16.** An adversarial pass re-read every cited NeMo/vapasr line in the working tree (all
  §1–§5 citations check out) and re-ran independent checks in `scratchpad/scratch-critic-encoder/`:
  - `kcheck.py`: K formulas, stream frame counts for every n ≤ 500 000 and the rel_shift identity. No weights.
  - `kgate.py`: live K gating.
  - `sdpa_check.py`: MLX fp16 SDPA numerics.
  - `nemo_edge_probe.py` under the lock: a 2-layer yaml encoder with random weights. Results are in `nemo-edge-results.json`
    and `nemo_edge_probe.log`.

  The streaming math (§1–§4) is confirmed. The corrections, each marked **[critic]** where it is made, are:
  - **K formula.** Use the eval expression `int(round(n/16000*12.5))`, not `round(N/1280)`. The two differ by ±1 at 16
    lengths below 31 s (§4.4, §6).
  - **Live K gating.** Frames computed before the endpoint can include index K (§6, §8.5).
  - **Stop-time mel frames.** Finishing the utterance releases 0–1 more mel frames, not 1–2 (§1.3).
  - **Eval resampling.** Eval does not resample (§1.3).
  - **Target extrapolation.** It now includes the M3 Pro (§8.3).
  - **Smaller items.** Caveats on the CUDA Triton subsampling path and on where the fb/window buffers come from.
    Evidence that the uncommitted att_context-fixing code was in effect. Prerequisites for the acceptance test (§8.6).

---------------------------------------------------------------------------------------------------------------------

## 0. Pipeline at a glance

```
PCM float32 16 kHz (mono)
  └─ pre-emphasis y[0]=x[0], y[i]=x[i]-0.97·x[i-1]
  └─ STFT: zero center-pad 256 | hann(400, symmetric) centred in 512 | hop 160 | rfft 512 → |X|²   (257 bins)
  └─ slaney mel 128 (fb buffer) → ln(mel + 2^-24)                       → log-mel (T,128) @100 Hz, T = N//160
  └─ streaming chunks of mel frames: first 25 frames; then [9 cached | 32 new] = 41 frames
       └─ ConvSubsampling dw_striding ×8, causal (Conv2d k3 s2 → ReLU → [dw k3 s2 → pw 1×1 → ReLU]×2)
       └─ flatten (256 ch × 17 freq) → Linear 4352→1024 ; drop first 2 rows on non-first steps   → 4 frames
       └─ 24 × ConformerLayer (FF½ → rel-pos MHSA w/ 56-frame cache → causal conv k9 w/ 8-frame cache → FF½ → LN)
  → encoder frames (4,1024) every 320 ms @12.5 Hz (frame j ↔ 80·j ms) → adapter → thinker
```

---------------------------------------------------------------------------------------------------------------------

## 1. Preprocessor (eval mode): exact math

### 1.1 Facts

| item | value at inference | evidence |
|---|---|---|
| input dtype | cast to float32. The output is also float32 because the dtype sentinel is fp32. | `nemo/asr/modules/audio_preprocessing.py:85-95` |
| window / hop in samples | `int(0.025·16000)=400`, `int(0.01·16000)=160` | yaml `window_size 0.025, window_stride 0.01`; `audio_preprocessing.py:241-244` |
| n_fft | 512 | yaml `n_fft: 512`; `features.py:304` |
| exact_pad | False, so `stft_pad_amount=None` and STFT `center=True` | default `audio_preprocessing.py:223`; `features.py:305,385`; probe `featurizer.exact_pad=false` |
| center padding | **zeros** (`pad_mode="constant"`), n_fft//2 = **256** each side, applied *after* pre-emphasis. **Note:** the comment at `vapasr/hf/live.py:29` ("반사 패딩", i.e. reflect padding) is wrong. | `features.py:379-389` |
| window | `torch.hann_window(400, periodic=False)` **symmetric** Hann, w[n]=0.5−0.5·cos(2πn/399), w[0]=w[399]=0. torch.stft centres it inside 512 at offset (512−400)//2 = **56**. Stored as buffer `featurizer.window` (400,) fp32. | `features.py:318-320`; probe: diff vs periodic=False is 0.0, vs np.hanning 1.9e-7, vs periodic=True 6.0e-3 |
| dither | 1e-5 · randn, added **only when `self.training`**, so no dither at inference. **Caveat:** vapasr training ran the preprocessor in train mode (`NemotronOnline.train()` recurses into `pre`, `online.py:127-128`), so s4 was *trained with* dither. Probe effect: log-mel changes by ≤0.045 in speech frames and ≤0.13 in digital-silence frames. Harmless; keep dither off for determinism. | `features.py:427-429`; probe `dither` |
| pre-emphasis | 0.97 (NeMo default; the yaml does not set it; HF `processor_config.json` `preemphasis: 0.97` agrees). `y[0]=x[0]`, `y[i]=x[i]−0.97·x[i−1]` in fp32. Samples beyond `length` are zeroed, which is a no-op for batch 1. | `audio_preprocessing.py:213`; `features.py:431-435` |
| spectrum | `|X| = sqrt(re²+im²)` (guard 0 because `use_grads=False`), then `pow(mag_power=2.0)`, giving the **power** spectrum (257 bins). nb-augmentation is training only and its prob is 0. | `features.py:441-454` |
| mel fb | `librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, fmin=0, fmax=8000, norm='slaney')` (htk=False). `highfreq=None` gives sr/2. Stored as buffer `featurizer.fb` (1,128,257) fp32. Bin 0 (DC) and bin 256 (Nyquist) have zero weight. | `features.py:329-337`; probe: diff vs librosa slaney is 0.0, vs htk 0.051 |
| mel product | `fb @ power` in fp32 (autocast disabled) | `features.py:460-464` |
| log | natural log, guard type **"add"**, value **2⁻²⁴ = 5.9605e-8**: `ln(mel + 2^-24)`. Digital silence gives −16.6355. | defaults `audio_preprocessing.py:218-219`; `features.py:466-468`; probe `silence_value=-16.63553` |
| frame_splicing | 1, so none | yaml; `features.py:475-476` |
| normalize | yaml `"NA"`: `normalize_batch` matches no branch and returns x unchanged (`"fixed_mean" in "NA"` is False), so **no normalization** | yaml `normalize: NA`; `features.py:51-106,479-480` |
| length | `seq_len = (N + 2·256 − 512) // 160 = N // 160` | `features.py:406-410` |
| frames returned | STFT gives `1 + N//160` frames. Frames `≥ seq_len` (just the last one) are overwritten with `pad_value=0.0`. | `features.py:482-486`; probe: N=96000 returns 601 frames with length 600, and the last frame is all 0.0 |
| pad_to | 0, so no extra padding | yaml `pad_to: 0`; `features.py:488-494` |

**Valid frame count:** T = N // 160. Probe values: N=160, 255 and 256 give 1; 399, 400 and 401 give 2; 1279 gives 7;
1280 gives 8; 16159 gives 100; 16160 gives 101. Use exactly T frames: realtime.py does `f[:, :, :T]`
(`vapasr/hf/realtime.py:157`). Never feed the zeroed frame T. `live.py:37-38` with `final=True` takes `n//HOP+1` frames,
which includes that zero frame; this is a small deviation, so do not copy it.

Frame t (0-based) is centred on sample 160·t. With the zero center padding it covers raw samples
[160t−256, 160t+256). Because the window sits at 56..455 of the 512 frame and w[0]=w[399]=0, the non-zero support is
raw samples **[160t−199, 160t+198]**.

### 1.2 Offline formula (numpy; matches NeMo to ≤2.9e-5 in log domain, 0.0 in silence)

```python
def logmel_full(x, window400, fb):               # x float32 (N,), window400 = ckpt buffer, fb = ckpt buffer (1,128,257)
    y = np.concatenate([x[:1], x[1:] - np.float32(0.97) * x[:-1]])          # pre-emphasis, fp32
    yp = np.pad(y, (256, 256))                                               # center=True, constant(0)
    w = np.zeros(512, np.float32); w[56:456] = window400                     # torch.stft window centring
    T = len(x) // 160
    idx = np.arange(T)[:, None] * 160 + np.arange(512)[None]                 # frame t = yp[160t : 160t+512]
    X = np.fft.rfft(yp[idx] * w, n=512, axis=-1)                             # complex64 in numpy>=2
    P = X.real**2 + X.imag**2                                                # power (257)
    return np.log(P @ fb[0].T + np.float32(2**-24))                          # (T,128) float32  (NeMo layout is (128,T))
```

`mlx_encoder_ref.mel_full` is this code. Measured vs NeMo: max 2.86e-5, mean 2.8e-7 (FFT rounding), and 0.0 in silent
frames.

### 1.3 Incremental computation (live mic)

A frame is **final** once every sample it touches has arrived. Two exact rules apply after n received samples:

- **Conservative** (any window): frame t is final iff `n ≥ 160t + 256`.
  `F(n) = 0 if n < 256 else min(n//160, (n−256)//160 + 1)`. This is `vapasr/hf/live.py:16,38` and `MelStream`.
- **Tight** (uses w[0]=w[399]=0): frame t is final iff `n ≥ 160t + 199`.
  `F(n) = min(n//160, (n−199)//160 + 1)` for n ≥ 199. It saves 3.6 ms. Probe check against NeMo on the prefix
  `x[:n]`: n=4095 gives frame 24 identical (needs 4039), n=48095 gives frame 299 identical (needs 48039), while
  n=20000 differs by 0.13 (frame 124 needs 20039) and n=48000 differs by 0.54.

The probe compared NeMo run on the prefix `x[:n]` with NeMo on the full utterance for 16 values of n in [4095, 48096].
Every frame below F(n) under the conservative rule was **bitwise identical** (max diff 0.0), so final frames never
change. Pre-emphasis is causal (y[i] needs only x[i−1]), so keep the last raw sample as state.

**At utterance end (stop):** append the trailing 256 zeros of the center padding and emit the remaining frames
F(N)..N//160−1. These frames depend on the end padding, so they are only final at end of utterance.
**[critic]** There are **0 or 1** such frames, not 1–2. Under the conservative rule F(N) = (N−96)//160, and
N//160 − (N−96)//160 ∈ {0, 1} because 96 < 160; the tight rule gives the same. Probe data agrees: n=4095 has valid 25
and final 24, while n=4096 has 25 and 25 (`out/probe-results.json` `mel_incremental_vs_nemo_prefix`). The last frame
T−1 covers raw samples up to 160T+95, so it touches the end padding only when N < 160T+96.

```python
class MelStream:                     # mlx_encoder_ref.MelStream (verified: vs offline 9.5e-7, count rule violations 0)
    state: ypad (pre-emphasized, starts with 256 zeros), last raw sample, n, t (frames emitted)
    push(x):   y = x - 0.97*concat([last or x[0]], x[:-1]); if first chunk: y[0] = x[0]
               ypad += y; n += len(x); emit frames t..F(n)-1; trim ypad before 160*t
    finish():  ypad += 256 zeros; emit frames t..N//160-1
```

Cost: 32 frames (320 ms) take about **0.07 ms** in numpy on an M4. Port it to `mx.fft.rfft` if desired.
Tolerance against NeMo's torch FFT is about 1e-5 in log-mel, which is irrelevant downstream.

**Sample rate:** the model expects 16 kHz mono float in [−1,1]. Browser mics deliver 44.1 or 48 kHz. Resample with a
proper anti-aliased resampler (polyphase, soxr-class). Never use naive decimation.
**[critic]** The original text said eval used soundfile + `torchaudio.functional.resample` (`vapasr/hf/infer.py:34-42`).
That is wrong: eval does **not** resample. `experiments/eval_single_turn_asr.py:288-299` `read_audio` asserts
`sr == 16000` for files and reads `pcm_s16le` as int16/32768. `infer.load_audio` is the notebook helper, and it is the
code that uses torchaudio resampling. The app's resampler choice is therefore unconstrained by eval; `audio-ui-spec.md`
§3.2 picks soxr HQ.

---------------------------------------------------------------------------------------------------------------------

## 2. Subsampling: `ConvSubsampling('dw_striding', ×8, causal)`

Construction: `ConvSubsampling(subsampling='dw_striding', subsampling_factor=8, feat_in=128, feat_out=1024,
conv_channels=256, subsampling_conv_chunking_factor=1, activation=nn.ReLU(True), is_causal=True)`
(`nemo/asr/modules/conformer_encoder.py:421-431`; yaml `subsampling_conv_channels: 256`, `causal_downsampling: true`).
The Triton fused path (`fuse_triton`) is CUDA-only and False here.
**[critic]** "Numerically the same" is unverified. It is the same math in fp32 but a different kernel and accumulation
order. It probably **was** active in eval on the H200: `use_triton` defaults to None, giving `TRITON_AVAILABLE`
(`conformer_encoder.py:352,430`; `subsampling.py:446`). `NemotronOnline.set_trainable(False)` only turns it *off* when
training (`online.py:118-120`), and `_forward_fused` runs whenever `x.is_cuda` and there is no grad
(`subsampling.py:736-743`). The eval features were therefore likely produced by the fused kernel, while the MLX port
reproduces the PyTorch path. Expect fp32-rounding-level differences (the realtime path, on the same CUDA kernels, matched
eval to 5.7e-6). This cannot be tested on a Mac, and §8.6 compares against torch CPU/MPS, which is the PyTorch path.

Input layout: mel (B,128,T) is transposed to (B,T,128) and unsqueezed to (B,1,T,F=128), i.e. NCHW with H=time and W=freq
(`conformer_encoder.py:675`, `subsampling.py:730`).

| idx | module (key prefix `pre_encode.conv.<idx>`) | weight shape (torch) | bias | stride | padding |
|---|---|---|---|---|---|
| 0 | `CausalConv2D(1→256, k3, s2)` | (256,1,3,3) | yes (256) | 2 | **(T: top 2, bottom 1), (F: left 2, right 1)**, zeros |
| 1 | ReLU | | | | |
| 2 | `CausalConv2D(256→256, k3, s2, groups=256)` depthwise | (256,1,3,3) | yes | 2 | same causal pad |
| 3 | `Conv2d(256→256, k1)` pointwise | (256,256,1,1) | yes | 1 | 0 |
| 4 | ReLU | | | | |
| 5 | depthwise as idx 2 | (256,1,3,3) | yes | 2 | causal |
| 6 | pointwise as idx 3 | (256,256,1,1) | yes | 1 | 0 |
| 7 | ReLU | | | | |
| out | `Linear(256·17=4352 → 1024)` | (1024,4352) | yes (1024) | | |

Evidence: `subsampling.py:200-273` (layer list), `causal_convs.py:47-48,69`
(`F.pad(x,(left=k−1=2,right=s−1=1,left,right))` pads both time and freq), `subsampling.py:416-428` (Linear in =
256·calc_length(128)). `use_bias: false` in the yaml applies to the Conformer layers only. The subsampling convs and
`out` all carry biases (`out/state_dict.tsv`).

- **Size per stride-2 layer** (time and freq): `L → (L + 2 + 1 − 3)//2 + 1 = L//2 + 1` (`subsampling.py:645-655`
  with all_paddings 3 and floor; `:719-721`).
  - Freq: 128 → 65 → 33 → **17**.
  - Time over three layers: **T′ = ⌊(T + 14)/8⌋**, checked for T = 1..4999.
- **Flatten order:** x (B, C=256, T′, F=17) is transposed to (B, T′, 256, 17) and reshaped to (B, T′, 4352), so the
  column index is **c·17 + f** (`subsampling.py:497-500`). Then `out`.
- **Masking (`MaskedConvSequential`, `subsampling.py:724-769,822-826`):** before every layer, time rows ≥ the current
  length are zeroed. The length is updated after each stride layer, and a final mask is applied after the last layer.
  If the input tensor ends exactly at `length` (true for batch 1 offline after trimming to T, and for every streaming
  chunk built as in §6), all masks are no-ops: the zero right-pad of the conv equals the masked zero.
  The port therefore needs **no masking** provided it never feeds frames beyond the valid length.
- **Receptive field / alignment:** each layer's output i reads inputs [2i−2, 2i]. Encoder frame j therefore reads mel
  frames **[8j−14, 8j]**, and anything before mel 0 is the per-layer zero padding, *not* zero mel. Frame j is centred
  at 80·j ms and its newest sample is 1280·j+198.
  **[critic]** More precisely, the *newest* mel frame of encoder frame j (mel 8j) is centred at 80·j ms. The conv
  receptive field (mel 8j−14…8j) is centred near 80j−70 ms. With [56,3] attention, frame j also sees frames up to the end
  of its 4-frame chunk, i.e. mel up to 32⌊j/4⌋+24. So in the label convention (token time t lands in frame ⌊t/0.08⌋+δ),
  frame j's own 80 ms window [80j, 80j+80) is mostly *not* yet inside its conv receptive field; it becomes visible via
  the in-chunk lookahead or the next frames. This matters only for latency reasoning (see `audio-ui-spec.md` §2.1), not
  for the port.

**Streaming feed** (`streaming_cfg`, §3; NeMo buffer `nemo/asr/parts/utils/streaming_utils.py:1598-1696`;
`vapasr/hf/realtime.py:159-180`):

| step | mel input (rows) | length | after subsampling | drop | kept rows = encoder frames |
|---|---|---|---|---|---|
| first ([56,3]) | mel[0:25], no cache | 25 | 25→13→7→**4** | 0 | frames 0..3 |
| k ≥ 1 ([56,3]) | mel[25+32(k−1)−9 : 25+32k] = 9 cached + 32 new | 41 | 41→21→11→**6** | **2** | frames 4k..4k+3 |
| first ([56,0]) | mel[0:1] | 1 | 1→1→1→1 | 0 | frame 0 |
| k = 1 ([56,0]) | mel[0:9] (exact rule; NeMo zero-fills instead, see below) | 9 | 9→5→3→2 | 1 | frame 1 |
| k ≥ 2 ([56,0]) | mel[8(k−2) : 8k+1] = 9 cached + 8 new | 17 | 17→9→5→3 | 2 | frame k |

Why the kept rows equal offline: chunk-local row r reads chunk-local mel [8r−14, 8r]. Kept rows r=2..5 need local
mel ≥ 2, so they never touch the chunk's left zero padding (the 9 cached frames cover 7 needed plus 2 spare). Rows
r=0,1 do touch it and are dropped (`drop_extra_pre_encoded`, `conformer_encoder.py:683-685`). On the right, 41, 21
and 11 are odd, so the last row never reads the bottom pad. In the first step the zero padding *is* the utterance
start, exactly as offline.

**Exact general rule (implement this).** Let `idx` be the first new mel frame of the step, `size` its new-frame count
(1+8R for the first step, 8(R+1) after), and `emitted` the number of encoder frames already output. Then:

- `s = max(0, idx − 9)` is the start of the subsampling input `mel[s : idx+size]`. It is **never zero-filled**.
- `drop = emitted − s/8`. s is always a multiple of 8. Subsampled row r is encoder frame s/8 + r.

This gives drop 0 on the first step and drop 2 in steady state. For **[56,3] it is identical to NeMo's buffer and
realtime.py**, because idx ≥ 25 > 9 after the first step.

**For [56,0] it is not identical.** At step 1, idx=1 < 9, and NeMo's `CacheAwareStreamingAudioBuffer`
(`streaming_utils.py:1660-1688`), realtime.py (`:167-170`) and live.py (`:48-49`) all **zero-fill** the 8 missing
pre-encode cache frames with mel = 0. Zero *mel* is not conv zero-padding: after bias and ReLU the layer-1 rows are
non-zero. Encoder frame 1 is therefore wrong, and the error propagates through the K/V and conv caches for the rest of
the utterance. The probe measured max 0.037 and still ~0.015 at frame 74 with random weights (§7). The exact rule
feeds mel[0:9] with drop 1 at step 1 instead. `mlx_encoder_ref.StreamingEncoderMLX` implements the exact rule.

**Do not** use NeMo's alternative `pad_and_drop_preencoded=True` buffer mode (`streaming_utils.py:1604-1606,1645-1647`).
It zero-pads the first chunk to 41 rows, which shifts the frame↔mel alignment by 7 frames and is not offline-equivalent.
Do not emulate the first step by prepending zero *mel* frames either: zero mel ≠ zero conv padding because of the
biases and ReLU.

---------------------------------------------------------------------------------------------------------------------

## 3. Streaming configuration (measured: `setup_streaming_params` on the yaml-built encoder)

`set_default_att_context_size(ctx)` re-runs `setup_streaming_params()` (`conformer_encoder.py:993-1008`), which is
implemented at `:1010-1118`. With `chunked_limited`, lookahead = R and `cache_drop_size = 0` (`:1042-1044`).
`sampling_frames = [1, 8]` (`subsampling.py:448-449`), `pre_encode_cache_size = [0, 9]` (`:451-452`).

| ctx | chunk_size [first, later] (mel) | shift_size | valid_out_len | last_channel_cache_size | cache_drop_size | pre_encode_cache_size | drop_extra_pre_encoded |
|---|---|---|---|---|---|---|---|
| **[56,3]** | **[25, 32]** | **[25, 32]** | **4** | **56** | 0 | **[0, 9]** | **2** (pass 0 on the first step) |
| [56,0] | [1, 8] | [1, 8] | 1 | 56 | 0 | [0, 9] | 2 |
| [56,6] (info) | [49, 56] | [49, 56] | 7 | 56 | 0 | [0, 9] | 2 |
| [56,13] (info) | [105, 112] | [105, 112] | 14 | 56 | 0 | [0, 9] | 2 |

Initial cache from `get_initial_cache_state(batch_size=1)` (`conformer_encoder.py:1120-1158`), all zeros:
`cache_last_channel (24, 1, 56, 1024)`, `cache_last_time (24, 1, 1024, 8)`, `cache_last_channel_len (1,) = 0`.
The same shapes hold for [56,0], because the cache size is the left context (56), not the chunk.
`last_channel_num` and `last_time_num` are 0 (unused). The MHA and depthwise-conv `cache_drop_size` are set to 0.

Derivations: chunk_size = `sampling_frames + 8·R`, giving [1+24, 8+24]; shift_size equals chunk_size because
cache_drop is 0; `valid_out_len = (32−8)//8+1 = 4`; `drop_extra_pre_encoded = 1 + (9−1)//8 = 2`.

Step semantics in `cache_aware_stream_step` (`nemo/asr/parts/mixins/streaming.py:79-148`):

- It overrides `drop_extra_pre_encoded` for the call, then calls `forward()` and `streaming_post_process`
  (`conformer_encoder.py:574-597`).
- Post-processing keeps the last 56 cache entries. If `keep_all_outputs=False`, outputs are clipped to `valid_out_len`
  (4 or 1). With chunked_limited that clip is a no-op, because the step never produces more.
- The step returns `cache_last_channel_len = clamp(len + T_new, max=56)` (`conformer_encoder.py:809`).

---------------------------------------------------------------------------------------------------------------------

## 4. Conformer layers in a streaming step

Global, before layer 0 (`conformer_encoder.py:690-726`, `multi_head_attention.py:1136-1159`):

- `xscaling: false`, so `xscale=None` and x is **not** scaled (`conformer_encoder.py:403-406`; probe `xscale null`).
  `rel_pos` does **not** add any embedding to x. `pos_enc` returns `x` (dropout is identity in eval;
  `dropout_emb=0.0` creates no emb dropout) and `pos_emb`.
- `pos_emb`: sinusoid table at relative positions (L−1) … −(L−1) with **L = T_new + cache_len**. In streaming,
  `cache_len = last_channel_cache_size = 56` *regardless of how full the cache is*, so L = 60 for [56,3]
  (`conformer_encoder.py:691-694`). `pe(r)[2m] = sin(r·d_m)`, `pe(r)[2m+1] = cos(r·d_m)`,
  `d_m = exp(2m · (−ln 10000 / 1024))`, all fp32 (`multi_head_attention.py:1074-1087`; table `(1, 9999, 1024)` for
  max_len 5000, non-persistent buffer, `:1126-1134`). The values depend only on r, so compute them directly.
- No `out_proj` (`feat_out: -1`), no reduction, `layer_drop_probs` all 0, all dropout off in eval.

### 4.1 Layer (`nemo/asr/parts/submodules/conformer_modules.py:177-252`)

All LayerNorms are `torch.nn.LayerNorm(1024)` with affine weight and bias and **eps 1e-5**. `use_bias:false` removes
biases only from the Linear and Conv1d layers.

```
r  = x + 0.5 · FF1(LN_ff1(x))                     # fc_factor 0.5 (:95,191-194); FF = Linear(1024→4096,no bias) → SiLU → Linear(4096→1024,no bias) (:392-413)
a  = LN_att(r)                                    # ← this tensor is what NeMo caches (cache_last_channel)
r  = r + MHSA(a)                                  # (:196-211)
r  = r + ConvModule(LN_conv(r))                   # (:224-228)
r  = r + 0.5 · FF2(LN_ff2(r))                     # (:230-232)
y  = LN_out(r)                                    # (:234) — this is the layer output; layer 23's y is the encoder output
```

SiLU is `Swish` = x·sigmoid(x) (`nemo/asr/parts/utils/activations.py:47`; probe `ff_activation Swish`).

### 4.2 Relative-position MHSA, untied biases (`multi_head_attention.py:225-367`, non-SDPA path `:357-362`)

`use_pytorch_sdpa` is False (yaml default; probe), which selects the manual path.

Per layer, as separate parameters:

- `linear_q`, `linear_k`, `linear_v` and `linear_out` are each (1024,1024) with no bias.
- `linear_pos` is (1024,1024) with no bias (`:258`).
- `pos_bias_u` and `pos_bias_v` are (8,128) each. With `untie_biases: true` the encoder passes `None` and each layer
  creates its own (`conformer_encoder.py:450-459`, `multi_head_attention.py:261-267`; probe
  `pos_bias_shared=false`).
- Heads h=8, d_k=128, scale 1/√128.

```
q = a_new·Wqᵀ                       (T,8,128)          # queries: new frames only
K = [cache ; a_new]·Wkᵀ, V = [cache ; a_new]·Wvᵀ  (56+T, 8,128)   # update_cache (:213-218): key=value=cat(cache, new)
P = linear_pos(pe(rel))             (2L−1, 8, 128)     # rel = (L−1) … −(L−1)
ac[h,i,j] = (q[i,h] + u[h]) · K[j,h]
bd[h,i,j] = (q[i,h] + v[h]) · P[rel = (56 + i) − j, h]          # = rel_shift(...)[..., :L]  (:272-283,326-327,360)
s = (ac + bd) / √128 ; s[mask] = −10000 ; p = softmax(s) ; p[mask] = 0        # (:60,133-154)
out = (p · V) merged heads · Woutᵀ
```

The rel_shift identity, checked numerically for (C,T) = (56,4), (56,1), (0,75): `rel_shift(X)[i,j] = X[i, (T−1)−i+j]`.
Column m of `pos_emb` is position (L−1)−m, so the relative distance is `(cache_len+i) − j` = query absolute index
minus key absolute index (positive means the key is to the left). Offline: cache_len=0, so it is i−j.

**Cache semantics** (`multi_head_attention.py:213-218`, `conformer_encoder.py:691-696,720-748,801-810`):

- `cache_last_channel[l]` (56,1024) stores the **attention input `a = LN_att(r)` of the previous 56 frames** (pre
  projection), not K/V. Next cache = `cat(cache[T:], a_new)` with `cache_drop_size=0`.
- Since `linear_k` and `linear_v` have no bias, caching **projected K and V** ((8,56,128) each per layer) is exactly
  equivalent. It saves 2·56·1024² MACs per layer per step, which is what `mlx_encoder_ref` does.
- **Validity:** `cache_last_channel_len` starts at 0 and grows by T_new per step, capped at 56. Key slot s < 56 is
  valid iff `s ≥ 56 − len` (pad_mask offset `= cache_len − cache_last_channel_len`, `conformer_encoder.py:696,911-915`).
  Invalid slots are masked. The zero initial cache is therefore never attended.

**Attention mask, `chunked_limited`** (`conformer_encoder.py:847-928`):

- Offline: `chunk = R+1`, `left_chunks = 56 // (R+1)`. Query i sees key j iff `0 ≤ ⌊i/chunk⌋ − ⌊j/chunk⌋ ≤ left_chunks`
  and both are within `length`.
  - [56,3]: chunks of 4 with 14 left chunks. A query in chunk c sees frames 4(c−14) … 4c+3: up to 60 keys, including
    0–3 *future* frames of its own chunk. That is the 0–240 ms in-chunk lookahead.
  - [56,0]: keys i−56 … i.
- Streaming: the mask is built over [56 cache slots | T new]. The new rows fall in chunk 14 (because 56 % 4 == 0), and
  every key in chunks 0..14 is visible except invalid cache slots. Each step adds exactly 4 frames (1 for [56,0]), so
  chunk boundaries stay aligned to frame 0 and this equals the offline mask.
- Rows that are fully masked (only possible for padded queries, which this procedure never creates) output 0.

**MLX form:** precompute per layer `P_l = pe(rel) · W_pos,lᵀ` for rel = 59 … −3 (63 rows; this table also serves
[56,0], whose rel range is 56 … 0). Then:

- `bd = take_along_axis((q+v)·P_lᵀ, idx)` with `idx[i,j] = 3 − i + j`, i < T, j < 56+T.
- Feed `mask = bd/√128 + maskadd` to `mx.fast.scaled_dot_product_attention(q+u, K, V, scale=1/√128)`. MLX computes
  softmax(scale·q·kᵀ + mask) in fp32 regardless of input dtype.
  **[critic]** This was stated from memory; it is now measured with `scratch-critic-encoder/sdpa_check.py` on mlx 0.32.2.
  The setup was H=8, T=4, L=60, fp16 q/k/v, and an fp16 additive mask with −inf on 20 slots. Max error against an fp64
  reference was 9.8e-4 for fp16 SDPA (≈2 output ulps), 7.0e-3 for a naive fp16 `mx.softmax` path and 2.3e-6 for fp32.
  There were no NaNs. This is consistent with fp32 internal softmax and fp16 output rounding.
- `maskadd` is −1e9 on invalid cache slots and 0 elsewhere. In fp16 it becomes −inf, which is harmless because every
  query always has at least one valid key (itself). Never feed padded query rows: a fully masked row would turn into NaN
  under −inf and poison the caches.

NeMo recomputes `linear_pos` on 119 positions every step (~125 M MACs/layer); the precomputed table removes that.

### 4.3 Convolution module (`conformer_modules.py:255-376`)

- `conv_norm_type: layer_norm`, `conv_context_size: causal`, `conv_kernel_size: 9`.
- `pointwise_activation='glu_'` is not in `activation_registry`, so the module uses the GLU path and the depthwise
  conv has d_model channels (`:286-294`; probe `conv_pointwise_activation glu_`).

```
g  = LN_conv(r) · W_pw1ᵀ                (T, 2048)      # pointwise_conv1 Conv1d(1024→2048,k1,no bias) applied as linear (:340-347)
g  = g[:, :1024] * sigmoid(g[:, 1024:])                # nn.functional.glu(dim=-1) (:350-351)
g[pad rows] = 0                                        # pad_mask (:357-358) — no-op when no padded frames are fed
hc = [cache_last_time (8 frames) ; g]   (8+T, 1024)    # CausalConv1D padding [8,0] → _left 8, _right 0 (conformer_encoder.py:984-985; causal_convs.py:110-116)
dw[t,c] = Σ_{k=0..8} w_dw[c,0,k] · hc[t+k, c]          # depthwise groups=1024, no bias, cross-correlation (:305-313; causal_convs.py:132-148)
cache_last_time' = hc[-8:]                             # last 8 GLU outputs (cache_drop_size 0)
z  = SiLU(LayerNorm_1024(dw))                          # module attr name is "batch_norm" but it is nn.LayerNorm (:319-320,364-371)
out = z · W_pw2ᵀ                         (T, 1024)     # pointwise_conv2 Conv1d(1024→1024,k1,no bias)
```

NeMo stores `cache_last_time` channel-first as (1024, 8). The zero initial conv cache equals the offline left zero
padding.

### 4.4 Encoder output

Layer 23's `LN_out` output (T,1024) is transposed to (B,1024,T) in NeMo (`conformer_encoder.py:798`). There is no extra
final norm. In streaming, every step's output frames are new frames (4, or 1 for [56,0]). NemotronOnline then truncates
or zero-pads the offline output to `K = round(N/1280)` frames (`vapasr/features/online.py:9,139-141`).

**[critic] K formula.** K is an *argument* of `NemotronOnline.forward` (`online.py:129,139-141`). The caller that matters
for parity is eval, which uses `K = int(round(len(wav) / 16000 * model.config.frame_hz))` with frame_hz 12.5
(`experiments/eval_single_turn_asr.py:431`; same form in `modeling_vapasr.py:225` and `offline_decode.py:49`). This is
**not** the same as `round(N/1280)`. At exact half points N = 1280m+640, the float product lands slightly above or below
.5, so K differs by ±1. That happens at **16 lengths below 31 s** and 36 below 62.5 s (`scratch-critic-encoder/kcheck.py`).
Examples: N=69 760 gives 55 (eval) vs 54, and N=73 600 gives 57 vs 58. Also `audio-ui-spec.md` critic item 2. **Use the
eval expression verbatim** everywhere this spec says `round(N/1280)`. All frame-count conclusions below still hold with
it; see §6.

---------------------------------------------------------------------------------------------------------------------

## 5. State dict: keys, shapes, checkpoint prefixes

Measured by instantiating from the yaml (full list: `scratch-encoder/out/state_dict.tsv`):

- Preprocessor: 2 persistent buffers.
- Encoder: **636** keys and **609,141,760** parameters, of which pre_encode has 4,596,736. That is 2.44 GB in fp32.

| key (NeMo) | shape | note |
|---|---|---|
| `featurizer.window` | (400,) | pre buffer, persistent. Symmetric Hann. |
| `featurizer.fb` | (1,128,257) | pre buffer, persistent. Slaney mel. |
| `dtype_sentinel_tensor` | (0,) | pre, **non-persistent**. Ignore. |
| `pre_encode.conv.0.weight` / `.bias` | (256,1,3,3) / (256,) | |
| `pre_encode.conv.2.weight` / `.bias` | (256,1,3,3) / (256,) | depthwise |
| `pre_encode.conv.3.weight` / `.bias` | (256,256,1,1) / (256,) | pointwise |
| `pre_encode.conv.5.*`, `pre_encode.conv.6.*` | same as 2 and 3 | |
| `pre_encode.out.weight` / `.bias` | (1024,4352) / (1024,) | input column = c·17+f |
| `pos_enc.pe` | (1,9999,1024) | **non-persistent** buffer, not in the ckpt. Recompute. |
| `layers.{0..23}.norm_feed_forward1.{weight,bias}` | (1024,) | |
| `layers.{l}.feed_forward1.linear1.weight` | (4096,1024) | |
| `layers.{l}.feed_forward1.linear2.weight` | (1024,4096) | |
| `layers.{l}.norm_conv.{weight,bias}` | (1024,) | |
| `layers.{l}.conv.pointwise_conv1.weight` | (2048,1024,1) | |
| `layers.{l}.conv.depthwise_conv.weight` | (1024,1,9) | |
| `layers.{l}.conv.batch_norm.{weight,bias}` | (1024,) | LayerNorm |
| `layers.{l}.conv.pointwise_conv2.weight` | (1024,1024,1) | |
| `layers.{l}.norm_self_att.{weight,bias}` | (1024,) | |
| `layers.{l}.self_attn.pos_bias_u` / `pos_bias_v` | (8,128) | |
| `layers.{l}.self_attn.linear_{q,k,v,out}.weight` | (1024,1024) | |
| `layers.{l}.self_attn.linear_pos.weight` | (1024,1024) | |
| `layers.{l}.norm_feed_forward2.{weight,bias}`, `feed_forward2.linear{1,2}.weight` | as FF1 | |
| `layers.{l}.norm_out.{weight,bias}` | (1024,) | |

Per layer that is 26 keys, so 24·26 + 12 pre_encode keys = 636.

**Prefixes in the VapAsr checkpoint (`model.safetensors`):**

- `VapAsrForStreamingASR.encoder` is a `NemotronOnline` with submodules `pre` (preprocessor) and `enc`
  (ConformerEncoder) (`vapasr/hf/modeling_vapasr.py:71-76`, `vapasr/features/online.py:87-101`). Its keys are
  therefore `encoder.enc.<NeMo key>` (636) plus `encoder.pre.featurizer.window` and `encoder.pre.featurizer.fb`, for
  **638** in total.
- This matches the s4 save parity "인코더 638 키" ("638 encoder keys") (`wiki/outputs/output-q06-s4-ast-eval-20261013.md:92`;
  counted by `experiments/semcommit_train.py:447`).
- `save_pretrained` writes `encoder.*` when `encoder_trainable` or `encoder_saved` is set (`modeling_vapasr.py:249-254`).
  s4 trained the encoder (`slurm/submit-q06-s4-ast.sh:29` `--train-encoder`), so the fine-tuned weights are in the
  checkpoint and **must** come from there, not from the public `.nemo`.
- `from_pretrained` reads only `encoder.enc.*` and rebuilds `pre` from config (`modeling_vapasr.py:257-269`,
  `online.py:20-28`). The `fb` and `window` buffers are fixed: they equal the librosa/torch construction exactly.
  Prefer the checkpoint copies and fall back to recomputing them.
  - **[critic]** `pre` is not rebuilt "from config" alone. `NemotronOnline` takes it from `restore_from(.nemo)` or from
    the `VAPASR_ENCODER_CACHE` `.pt`, whose `pre_state` is the .nemo's preprocessor state dict (`online.py:26-28,45-46,
    94-98`). The checkpoint's `encoder.pre.featurizer.{window,fb}` are therefore NVIDIA's stored buffers.
  - **[critic]** "Equal the librosa/torch construction exactly" was verified only for a *freshly built* preprocessor
    (probe `fb_check`). The stored .nemo fb may come from another librosa version (differences ~1e-8 expected). Keep the
    rule "prefer the checkpoint copies".
  - **[critic]** Direct evidence for the 638 count: `raw/sources/experiments/2026-10-13-q06-s4-ast-eval/train-85952-events.txt:53,55`
    has `"encoder_keys": 638` and `"encoder_saved_keys": 638`, so the safetensors really holds 638 `encoder.*` keys.
    `n_enc` counts every `encoder.` key, pre buffers included (`semcommit_train.py:447`).
- No `encoder.enc.pos_enc.pe` key exists, and none of the thinker's `thinker.audio_tower.*` keys are used.

**MLX weight conversion** (implemented in `mlx_encoder_ref.EncoderWeights`):

| tensor | torch | MLX |
|---|---|---|
| conv0 | (256,1,3,3) OIHW | (256,3,3,1) = `w.transpose(0,2,3,1)` (OHWI) |
| depthwise 2D | (256,1,3,3) | (256,3,3,1), used with `mx.conv2d(..., groups=256)` (verified on GPU, max diff 1.9e-6) |
| pointwise 2D | (256,256,1,1) | (256,256) matmul `h @ W.T + b` |
| `pre_encode.out.weight` | (1024, c·17+f) | permute columns to f·256+c (`W.reshape(1024,256,17).transpose(0,2,1).reshape(1024,4352)`) so that the NHWC output (T,17,256) can be reshaped directly |
| Linear | (out,in) | keep and use `x @ W.T` (or pre-transpose) |
| `pointwise_conv1/2` | (O,I,1) | (O,I) |
| `depthwise_conv` | (1024,1,9) | (9,1024) taps: `y = Σ_k hc[k:k+T] * w[k]`. Alternatively `mx.conv1d` with weight (1024,9,1) and groups=1024. |
| `pos_bias_u/v` | (8,128) | (8,1,128) |
| `linear_pos` | (1024,1024) | fold into the constant table `P_l` (8,128,63) at load |

---------------------------------------------------------------------------------------------------------------------

## 6. Offline vs streaming: what training used, what realtime.py does, edge handling

- **Training (s4):** offline masked forward over the whole utterance, `NemotronOnline.forward`
  (`online.py:129-142`): `pre(wav, len)` produces T+1 frames with length T, then `enc(f, T)` with
  `att_context [56,3]` fixed. The fixing comes from `--encoder-right-context 3 --encoder-context-sampling fixed` in
  `slurm/submit-q06-s4-ast.sh:31`, which sets `att_context_size_all=[[56,3]]` (`experiments/semcommit_train.py:241-249`).
  Without it, NeMo would sample among the 4 contexts in train mode (`conformer_encoder.py:669-672`).
  - **[critic]** The `--encoder-right-context` / `--encoder-context-sampling` code is **uncommitted**: `git diff
    experiments/semcommit_train.py` adds both args and lines 241-249. Only the launch script `submit-q06-s4-ast.sh` is
    committed (00500c2, 2026-10-12). argparse would reject unknown flags, so the mxc copy that ran s4 had this code.
    Independent evidence it was active in the lineage: the Stage-1 job 82738 startup log
    (`raw/sources/experiments/2026-10-08-stage1-trilingual-d10-prep/job-82738-startup.log:3-4`, 2026-10-08) prints
    "인코더 att_context [56,0] → [56, 3]" and "인코더 학습 att_context 고정 [56, 3]" ("encoder att_context fixed at
    [56, 3] for training"). No s4 startup log is on this machine.
  - **[critic]** Load-time contract for the app: read `config.json` and assert `encoder_left_context == 56` and
    `encoder_right_context == 3`. 56 is in fact hard-coded at `online.py:101`; the right context comes from
    `modeling_vapasr.py:73`. Also assert `encoder_trainable or encoder_saved`, so the `encoder.enc.*` weights are present
    (`modeling_vapasr.py:261`).
  - Encoder trainable: bf16 autocast on CUDA during training (`online.py:135`).
  - SpecAugment and dither in training only.
  - Features are truncated or zero-padded to `K = round(N/1280)` (`online.py:139-141`).
    **[critic]** In training K is the item's `K` from the manifest/dataset (`semcommit_dataset.py:369-372`; speed-perturbed items `:508-509`). It may be
    extended to `max(K0, last emission + tail_margin)`, with the audio duration set to `K·0.08` (training tail padding).
    It is not recomputed from N here; for the app, only the eval expression in §4.4 matters.
- **Inference / eval:** `load_model` calls `set_trainable(False)`, so the encoder runs **fp32** without autocast
  (`vapasr/hf/infer.py:30`). Eval appends **1 s of digital silence** to every utterance and uses
  K = round(len·12.5/16000) including that tail (`experiments/eval_single_turn_asr.py:288-299,431`).
  **[critic]** The exact expression is `int(round(len(wav) / 16000 * 12.5))`, divide first then multiply. The operation
  order matters at half points (§4.4). Eval also disables TF32 (`eval_single_turn_asr.py:309-310`), and the
  sbatch passes `--tail-s 1.0` (`slurm/eval-single-turn-queue.sbatch:44`).
- **realtime.py** (`vapasr/hf/realtime.py:136-183`): mel computed once from the whole file, then the §2 feeding with
  `cache_aware_stream_step`, `keep_all_outputs=(idx+shift ≥ T)`, a CUDA graph for steady steps, and the output cut or
  zero-padded to K.
  - Measured on H200 with real s3 weights: max |stream − full| = **5.7e-6**, and decoding is token-identical to the eval
    path over 192 decodes (`wiki/outputs/output-simulst-baselines-20261012.md:163,166`).
  - `vapasr/hf/live.py` (Sep-10 demo, [56,0]) uses the same idea with incremental mel, but its step-1 zero-filled
    pre-encode cache is not offline-exact for [56,0] (§2).
- **Start of utterance:** the first step is special: 25 mel frames (1 for [56,0]), no pre-encode cache, drop 0, and
  `cache_last_channel_len=0`, so every cache slot is masked. The K/V/conv cache *shapes* are the same as in steady state,
  so the layer stack keeps its static shapes and only the subsampling input differs (25 vs 41 rows).
- **End of utterance** (realtime.py / NeMo buffer rules):
  1. Make all mel frames up to T = N//160 final using the end zero padding (§1.3).
  2. While `idx < T`: `chunk = mel[idx : min(idx+size, T)]`. If it has fewer rows than `sampling_frames` (1 first,
     **8** later), stop and drop the tail (`realtime.py:165-166`, `streaming_utils.py:1629-1639`).
  3. Otherwise feed `mel[s : end]` with `s = max(0, idx−9)` (the exact rule, §2; for [56,3] this is the 9 cached
     frames), `length = end − s`, `drop = emitted − s/8` (2 in steady state) and `keep_all_outputs=True`. Then
     `idx += size`.
  4. The final step therefore produces `⌊(r+7)/8⌋` frames for r = 8..32 remaining mel frames (1–4 frames). It is a
     different shape, so run it uncompiled.
  5. There are never padded rows, so no mask is needed.
- **How many frames:**
  - With T = 25+32m+r, streaming yields S = 4+4m+⌊(r+7)/8⌋ for r ≥ 8, and 4+4m for r < 8.
  - Offline yields ⌊(T+14)/8⌋, which is one more than S when r ∈ [1,7]. That extra frame is computed largely from the
    right zero padding.
  - Since K = round(N/1280) ≤ ⌊(T+4.96)/8⌋, **K ≤ S always**, so streaming always covers the K frames training uses.
    Verified: r = 0, 5, 8 and 31 give (K, S, offline) = (71,72,72), (72,72,73), (72,73,73) and (75,76,76).
    Also brute-forced for every N in [400, 640000) in steps of 7, for both [56,3] and [56,0]: no case with K > S, and
    S never exceeds the offline count.
    **[critic]** The constant is 4.994, not 4.96. N ≤ 160T+159 gives K ≤ N/1280 + 0.5 ≤ (T + 0.994 + 4)/8; the
    conclusion is unchanged. Re-verified with the **eval** K expression for **every** n in [0, 500 000], for both [56,3]
    and [56,0] (realtime.py feeding, tail drop included): 0 cases of K > S and 0 cases of S > offline
    (`scratch-critic-encoder/kcheck.py`).
  - **Truncate to K** to mimic training/eval exactly. Live, this only matters at stop: frames beyond K simply are not
    fed.
    **[critic] Wrong as stated: live frames can reach index K *before* the stop.** Chunk c becomes computable at
    n = 4096 + 5120c samples, and then K(n) = round(3.2 + 4c) = 4c+3. So its last frame 4c+3 is **not yet** < K.
    - If the endpoint falls there and the appended tail is short or zero, eval never decodes frame 4c+3. In
      `audio-ui-spec.md` the zero tail is `max(0, 1000 − trailing_nonspeech_ms)`, so it can be 0, and the real pause
      audio is fed live.
    - With no tail, this happens at **7.5 %** of endpoint lengths (`scratch-critic-encoder/kgate.py`).
    - **Exact gating rule:** K(n) is monotone non-decreasing in n (checked for n ≤ 500 000), so pass encoder frame k
      to the thinker only once **k < K(n_received)**, where n_received counts the segment's samples so far.
    - Frames 4c..4c+2 pass immediately. Frame 4c+3 passes after **384–385 more samples (24 ms)**.
    - At the endpoint, decode the remaining frames with k < K(n_final) and discard the rest.
    - Alternatively, always append ≥ 385 zero samples before finishing. The eval protocol appends 16 000. This keeps
      every *pre-endpoint* frame below K, but `finish()` must still cut its own output at K.
- **Latency of availability:** encoder chunk c (frames 4c..4c+3) needs mel frame 32c+24 to be final.
  - Conservative rule: n ≥ 4096 + 5120c samples, i.e. **256 + 320·c ms** of audio plus compute.
  - Tight rule: 4039 + 5120c samples (252.4 + 320c ms).
  - [56,0]: frame k at 16 + 80k ms (conservative).
  - In the probe's MLX live pipeline, blocks arrive in random sizes, so each emission lands at the first block
    boundary at or after its threshold:
    - [56,3]: emitted at [4829, 10861, 14541, 21001, 25661] samples (thresholds 4096, 9216, 14336, 19456, 24576).
    - [56,0]: emitted at [1394, 3055, 3055, 4829, 5823] samples (thresholds 256, 1536, 2816, 4096, 5376).
- **Offline ("final") mode:** the streamed frames equal the offline features to about 1e-5, so a final or offline
  thinker decode can reuse the frames already computed. No second encoder pass is needed.

---------------------------------------------------------------------------------------------------------------------

## 7. Empirical check (random weights; seeds: torch 0 for init, randomized LN affine/pos_bias with generator 1, audio seed 7)

Scripts: `scratch-encoder/encoder_probe.py` and `encoder_probe2.py`, both run under the heavy lock. probe2 took 170 s.
The first run of probe 1 aborted in its MLX section on a since-fixed reference bug; see the note below.

Audio: a synthetic harmonic "voice" (f0 120±40 Hz, 11 harmonics, AM) plus noise at 0.01, with 0.5 s of exact digital
silence at 2.0–2.5 s.

NeMo offline is `pre` followed by `enc(f, T)`. NeMo streaming follows the realtime.py feeding on CPU. The table reports
the max over frames of max_d |Δ|. The offline output RMS is about 1, because the final LN is roughly unit-scale.

| ctx | N (samples) | T mel | r = (T−25) mod 32 | K = round(N/1280) | offline frames | stream frames | feeding | max over compared frames | max over frames < K |
|---|---|---|---|---|---|---|---|---|---|
| [56,3] | 96000 | 600 | 31 | 75 | 76 | 76 | realtime.py / NeMo buffer | **1.07e-5** | 1.07e-5 |
| [56,3] | 91120 | 569 | 0 | 71 | 72 | 72 | same | 1.07e-5 | 1.07e-5 |
| [56,3] | 91920 | 574 | 5 | 72 | 73 | 72 (tail of 5 mel frames dropped) | same | 1.07e-5 | 1.07e-5 |
| [56,3] | 92400 | 577 | 8 | 72 | 73 | 73 (last step: 17 rows, 1 frame) | same | 1.07e-5 | 1.07e-5 |
| [56,3] | 96000 | 600 | | 75 | 76 | 76 | exact rule (§2) | 1.07e-5 | |
| [56,0] | 96000 / 91120 / 91920 / 92400 | | | 75 / 71 / 72 / 72 | 76 / 72 / 73 / 73 | 75 / 72 / 72 / 73 | realtime.py / NeMo buffer (zero-filled pre-encode cache at step 1) | **0.0368** (frame 0: 9e-6; frames 1..: 0.015–0.027, still 0.0146 at the last frame) | 0.0368 |
| [56,0] | 96000 / 91920 | | | 75 / 72 | 76 / 73 | 75 / 72 | **exact rule** (step 1 = mel[0:9], drop 1) | **1.10e-5** | 1.10e-5 |

All [56,3] steps had 25 rows with drop 0, then 41 rows with drop 2, and `cache_len_after` grew 4, 8, 12, … up to 56.
The final step of the r=31 case had 40 rows and `keep_all=True` and produced 4 frames.
NeMo CPU timing on this Mac (fp32, eager, 8 torch threads): offline 6 s took 0.75 s, and the realtime.py-style stream
took 12.4 s for 19 steps (NeMo's cache handling is slow on CPU; irrelevant for the port).

MLX reference (`mlx_encoder_ref.py`, GPU) vs NeMo, fed the NeMo mel (`vs stream`, `vs offline[:K]`) and the full live
pipeline (PCM in random 64–2048-sample blocks → `MelStream` → `StreamingEncoderMLX`):

| dtype | ctx | input | vs NeMo offline[:K]: max / rel-L2 / min cos | vs NeMo stream (same feeding) |
|---|---|---|---|---|
| fp32 | [56,3] | NeMo mel, N = 96000 / 91120 / 91920 / 92400 | **9.3e-6 / 1.8e-6 / 1−3e-12** (≤1.02e-5 all lengths) | max 1.17e-5 |
| fp32 | [56,3] | **live: PCM → MelStream → MLX** (random 64–2048 blocks) | **1.11e-5** / 1.9e-6 | |
| fp32 | [56,0] | NeMo mel, N = 96000 / 91920 | 1.14e-5 / 1.8e-6 | (vs NeMo realtime-feeding stream: ≈0.037, i.e. NeMo is the one off) |
| fp32 | [56,0] | live pipeline | 1.04e-5 | |
| fp16 (weights + activations) | [56,3] | NeMo mel / live | 0.0153 / 0.0146 max, rel-L2 2.6e-3, min cos 0.999995 | |
| fp16 | [56,0] | NeMo mel / live | 0.0156 max, rel-L2 2.7e-3, min cos 0.999994 | |
| bf16 | [56,3] | NeMo mel / live | **0.155** max, rel-L2 2.1e-2, min cos 0.99972 | |
| bf16 | [56,0] | NeMo mel / live | 0.133 max, rel-L2 2.2e-2, min cos 0.99961 | |

Sources: `out/probe2-results.json` and the re-run `out/probe-results.json` (all four lengths). Output scale is RMS ≈ 1.01.

Note on run 1: the first version of the reference fed `mel[idx−9:…]` with a negative start for [56,0] (idx=1). The
resulting empty Metal arrays killed the process (lockf exit 70) after all NeMo checks and the MLX fp32 [56,3] checks
had been written. That partial output is kept as `out/probe-results-run1-crashed.json` and `probe-run1-crashed.log`.
The reference now uses the exact rule (§2), and `encoder_probe.py` was re-run with it: 283 s, exit 0, results in
`out/probe-results.json` and `probe.log`. The re-run reproduces every number above. In particular, MLX [56,0] (exact
rule) differs from NeMo's realtime-style stream by 0.0368 but from NeMo offline by only 1.14e-5, so the NeMo buffer
is the one that is off.

Conclusion: the procedure in §1–§4 is exact to fp32 rounding. The fp32 MLX port can be accepted at a tolerance of
max |Δ| < 1e-3 on real weights; the expectation is about 1e-5. For real-checkpoint acceptance criteria see §8.6.

**[critic] Independent re-check** with `scratch-critic-encoder/nemo_edge_probe.py`, run under the lock (9.5 s). It uses
its own feeding code and a NeMo 3.1 yaml encoder with `n_layers=2`, random weights, and LN affine/pos_bias perturbed
(σ 0.2). The audio is a synthetic FM tone plus noise, CPU fp32. Results are in `nemo-edge-results.json`.

- **A. [56,3] realtime.py feeding vs offline.**
  - Final remainders r ∈ {0,3,7,8,9,16,24,25,31} (T = 185…216): stream/offline frame counts are 24/24, 24/25, 24/25,
    25/25, 26/26, 26/26, 27/27, 28/28, 28/28. Last-step rows/out: r=8 gives 17/1, r=9 18/2, r=16 25/2, r=24 33/3,
    r=25 34/4, r=31 40/4. Max |Δ| ≤ **6.7e-6** for all.
  - Short utterances T ∈ {1,7,10,24}, a single first step of T rows: 1/2/3/4 frames, max |Δ| ≤ 4.2e-6.
  - This confirms the §6 end-of-utterance rules and the ⌊(r+7)/8⌋ count for the r values not covered by §7.
- **B. Cache contents (step 3, layer 0).** The tail of `cache_last_channel` equals the `norm_self_att` output of the new
  frames (|Δ| = **0.0**), so the cache holds attention inputs, not K/V. `cache_last_time` equals the last 8 GLU outputs
  fed to `depthwise_conv` (|Δ| = **0.0**).
- **C. [56,0], two lengths.**
  - NeMo/realtime zero-filled step 1 (17 rows, drop 2): frame 0 is 9e-6, frame 1 is **0.074**, and the last frame is
    still 0.010–0.011.
  - Exact rule (step 1 = 9 rows, drop 1): max **9.1e-6**.
  - This independently confirms the §2 [56,0] finding.
- The §7 table's N values (96 000 / 91 120 / 91 920 / 92 400) are not half points, so their K columns are the same under
  the eval K expression.

---------------------------------------------------------------------------------------------------------------------

## 8. MLX porting notes

### 8.1 Op mapping

| NeMo / torch | MLX |
|---|---|
| `F.pad` + `Conv2d` NCHW (CausalConv2D) | `mx.pad(h, [(0,0),(2,1),(2,1),(0,0)])` + `mx.conv2d(h, w_OHWI, stride=2)` on NHWC (1,T,F,C). MLX `padding=` is symmetric only, so always pad explicitly. |
| depthwise Conv2d (groups=256) | `mx.conv2d(..., groups=256)` with weight (256,3,3,1). Verified: max diff 1.9e-6 vs torch. |
| pointwise Conv2d 1×1 | `h @ W.T + b` on the channel axis |
| flatten (c,f) + Linear | reshape NHWC (T,17,256) to (T,4352) with the column-permuted `out.weight` |
| Linear (no bias) | `x @ W.T`. Pre-transposing at load is optional; MLX handles transposed operands. |
| LayerNorm | `mx.fast.layer_norm(x, w, b, 1e-5)` |
| SiLU / GLU / ReLU | `x*mx.sigmoid(x)` / `a*mx.sigmoid(b)` / `mx.maximum(x,0)` |
| rel-pos bd + rel_shift | constant table `P_l` (8,128,63) per layer, then `(q+v) @ P_l` and `mx.take_along_axis(…, idx)` with `idx[i,j]=3−i+j` |
| masked softmax attention | `mx.fast.scaled_dot_product_attention(q+u, K, V, scale=1/√128, mask=bd/√128 + maskadd)`. The additive mask is supported, verified at 8.3e-7. |
| depthwise causal Conv1d k9 with cache | `hc = concat(cache8, g)`, then `y = Σ_k hc[k:k+T]*w[k]` (9 FMAs) or `mx.conv1d(hc[None], w(1024,9,1), groups=1024)` |
| caches | per layer: K,V (8,56,128), conv (8,1024), plus an integer `valid` (0..56). Update: K ← concat(K, k_new)[:, −56:]. |

### 8.2 Static shapes and `mx.compile`

- **Layer stack (24 layers)**: inputs x (4,1024), K/V caches (24×(8,56,128)), conv caches (24×(8,1024)), and
  maskadd (1,60). These shapes are identical for the first step and every steady step, so **one compiled graph**
  serves the whole utterance except the final short step. Pass maskadd as an *input* array (never a Python constant)
  so that changes in cache fill do not retrace. Weights can be closure constants.
- **Subsampling**: two shapes, 25 rows (first) and 41 rows (steady). Compile both, or compile only the steady one.
- **Final step**: T_new ∈ {1..4}. Run it eagerly (once per utterance).
- **[56,0]**: x (1,1024), keys 57, subsampling 17 rows. Also static.
- MLX reuses compiled graphs per input shape automatically.

### 8.3 Compute and memory per step ([56,3], 4 frames per 320 ms)

Per frame per layer:

- FF1 + FF2: 2·2·1024·4096 = 16.8 M MACs
- q, k, v, out: 4.2 M
- pw1 + pw2: 3.1 M
- attention: 8·60·128·2 + 8·63·128 ≈ 0.19 M
- depthwise: 9 K

That is ≈ 24.3 M MACs per frame per layer, **≈ 2.33 G MACs (4.7 GFLOP) per 320 ms step** over 24 layers × 4 frames.
Subsampling adds ≈ 0.06 G MACs.

NeMo's own step does about 3.5× more work, because each step re-projects K/V for the 56 cached frames and runs
linear_pos over 119 positions. The cached-K/V + `P_l` design avoids both.

The step is **memory-bandwidth bound**: every step streams all weights once.

Ideal bandwidth-bound estimate (weights / peak bandwidth):

| weights | size per step | M4 (120 GB/s) | M3/M4 Pro (150–273 GB/s) | M3/M4 Max (300–546 GB/s) |
|---|---|---|---|---|
| fp32 | 2.44 GB | ≈ 20 ms | 9–16 ms | 4.5–8 ms |
| fp16/bf16 | 1.22 GB | ≈ 10 ms | 4.5–8 ms | 2.2–4 ms |

Measured on this Mac mini M4 (16 GB) with MLX 0.32.2. Each figure is one steady [56,3] step (4 frames, 41 mel rows
in), with `mx.eval` on the outputs and caches. The run was under the heavy lock, so no other heavy process was active,
but the machine was not otherwise idle:

| dtype | compiled p50 / p95 (ms) | eager p50 (ms) | subsampling alone (41 rows) | [56,0] step (1 frame) p50 |
|---|---|---|---|---|
| fp32 | **35.6 / 38.6** (re-run: 32.7 / 33.9) | 37.4 | 0.89 ms | 32.5 ms |
| fp16 | **17.6 / 19.4** | 18.2 | 0.49 ms | 16.6 ms |
| bf16 | **16.7 / 17.6** | 17.6 | 0.50 ms | 16.2 ms |

These figures confirm the bandwidth-bound model. Cost per step barely depends on the frame count (4 vs 1). The
effective bandwidth is about 68–73 GB/s on this M4 (2.44 GB / 35.6 ms, 1.22 GB / 16.7 ms). Extrapolating at the same
efficiency gives M4 Pro (273 GB/s) ≈ 15 ms fp32 / 7.5 ms fp16, and M4 Max ≈ 8 / 4 ms. **[56,3] fp32 therefore uses
11 % of the 320 ms budget on a base M4.** [56,0] fp32 would use 41 % of its 80 ms budget, so use fp16 there.
`mx.compile` gains only about 5 %, because the step is dominated by weight reads.
MelStream costs 0.07 ms per 320 ms of audio in numpy.

Caches: K+V = 24·2·56·1024·4 B = 11 MB (fp32), plus 0.8 MB conv. On the target MacBook Pro (M3/M4 Pro or Max,
32 GB+), the extrapolated fp32 step is 8–15 ms, about 3–5 % of the 320 ms budget. Almost all of the budget is left
for the thinker.
**[critic]** That range omits the **M3 Pro (150 GB/s)**. At the measured ≈57 % efficiency (68.5 GB/s of 120 GB/s on the
base M4), the extrapolations are:

| chip (bandwidth) | fp32 step | fp16 step |
|---|---|---|
| M3 Pro (150 GB/s) | ≈ **28 ms** | ≈ 14 ms |
| M4 Pro (273 GB/s) | ≈ 16 ms | ≈ 8 ms |
| M3 Max (300/400 GB/s) | ≈ 14/11 ms | |
| M4 Max (410/546 GB/s) | ≈ 10/8 ms | |

The correct fp32 range is therefore **8–29 ms (3–9 % of 320 ms)**. These are extrapolations, not measurements. Measure
on the actual MacBook with `encoder_probe2.py`'s timing section.

### 8.4 Precision

- The inference reference is fp32 (`infer.py:30`). Training ran the encoder under bf16 autocast (`online.py:135`), so
  a bf16/fp16 encoder is within the training numerics, but it must be validated on the real checkpoint (§8.6).
- Random-weight results vs NeMo fp32: fp16 max |Δ| 0.015, rel-L2 2.6e-3, min cos 0.999995; bf16 max |Δ| 0.155, rel-L2 2.1e-2, min cos 0.9997. That is 10× less accurate than fp16 at the same speed. **Prefer fp16 over bf16** if the real activations fit fp16 range (check max |activation| per layer on real audio; LayerNorms bound the residual stream reads, but the residual stream itself can be large in trained Conformers).
- Recommendation: **start fp32**, which keeps parity. It measured 35.6 ms per 320 ms step on a base M4 and is
  estimated at ≤15 ms on an M4 Pro.
- **Switch to fp16 or bf16 weights only if the thinker needs the headroom** and the real-checkpoint check passes.
  Keep LayerNorm statistics and softmax in fp32 (MLX's fast kernels do this internally).
- Keep the mel in numpy fp32.

### 8.5 Suggested app-side API (mirrors `mlx_encoder_ref`)

```python
mel = MelStream(window, fb)                      # ckpt buffers encoder.pre.featurizer.window / .fb
enc = StreamingEncoderMLX(EncoderWeights(sd_enc, C=56, Tm_max=4, dtype=mx.float32), att_context=(56, 3))
# every audio callback (any block size, 16 kHz float32):
for x in enc.push_mel(mel.push(pcm_block)):      # x: (4,1024) mx.array → adapter → thinker audio positions k..k+3
    ...
# on stop:
tail = enc.push_mel(mel.finish()) + enc.finish() # then cut the total frame count to K = round(N/1280) (training convention)
# eval convention additionally appends 1 s of digital silence before finishing (eval_single_turn_asr.py:299)
```

Here `sd_enc` is `{k[len("encoder.enc."):]: v for k, v in safetensors if k.startswith("encoder.enc.")}`. Read it with
`safetensors.numpy` or `mx.load`. The ckpt is fp32, 4.8 GB in total.

**[critic] Corrections to this API sketch:**
- **K.** Cut to `K = int(round(n / 16000 * 12.5))` (the eval expression, §4.4), not `round(N/1280)`.
- **Gating.** Gate each yielded frame before it reaches the thinker. Frame index k may be decoded only when
  `k < K(n_received)`, which is monotone in n (§6). The loop above must therefore keep a small pending queue: frames
  4c..4c+2 pass at once, and frame 4c+3 passes once 385 more samples have arrived.
- **Stop.** Decode pending frames with k < K(n_final) and drop the rest.
- **Load-time asserts.** Before building the encoder, assert `config.encoder_left_context == 56`,
  `config.encoder_right_context == 3`, and `config.encoder_trainable or config.encoder_saved` (§6). Also assert there are
  636 `encoder.enc.*` keys plus the 2 `encoder.pre.featurizer.*` buffers (§5).
- **Utterance boundaries.** `MelStream.reset()` and `StreamingEncoderMLX.reset()` per segment. Every utterance or segment
  starts from zero caches, `valid = 0`, pre-emphasis `y[0]=x[0]` and 256 leading zeros (`decode-contract.md` §d item 1).

### 8.6 Acceptance test once the real checkpoint is on the MacBook

1. Load the real `encoder.enc.*` weights into torch `NemotronOnline` (NeMo, CPU or MPS) and into the MLX port.
2. Run 3–5 real utterances (ko, en, de, 5–15 s) **plus a 1 s zero tail**.
3. Check `max |MLX_stream − NemotronOnline.forward[:K]|`: below 1e-3 (fp32; expected ~1e-5), and min cosine above
   0.9999.
4. For fp16 or bf16, compare thinker greedy tokens against the fp32 encoder on the same utterances. They should be
   token-identical or nearly so (a WER/BLEU delta of 0).

**[critic] Prerequisites for step 1:**
- **No .nemo on the MacBook.** `NemotronOnline` needs either a `.nemo` or a `VAPASR_ENCODER_CACHE` `.pt` holding
  pre/enc cfg+state (`online.py:20-36,87-98`). Without a `.nemo` it uses only the cache file.
- **local-env.sh trap.** `scripts/local-env.sh:9-10` defaults `VAPASR_ENCODER_CACHE` to
  `~/Desktop/VAPKT-models/nemotron-encoder.pt`, which is an iCloud **dataless** file on this machine; loading it hangs.
  Build an explicit cache from the yaml plus the checkpoint's `encoder.pre.*`/`encoder.enc.*` tensors
  (`decode-contract.md` §g) and export `VAPASR_ENCODER_CACHE` *after* sourcing `local-env.sh`.
- **Run on CPU or MPS.** `realtime.StreamingEncoder` calls `torch.cuda.synchronize()` unconditionally
  (`realtime.py:154-179`), so use plain `NemotronOnline.forward` or `cache_aware_stream_step` without those syncs.

---------------------------------------------------------------------------------------------------------------------

## Unknowns / risks

1. **Real-checkpoint parity is unverified.** Everything above uses random weights. The s4 checkpoint is not on this
   machine. The s4 encoder was fine-tuned (`--train-encoder`), so the weights **must** come from
   `model.safetensors` `encoder.enc.*`, not from the public `.nemo` / HF Nemotron. Run the §8.6 acceptance test on the
   MacBook.
2. **fp16/bf16 behaviour on real weights is unknown.** Random-weight error (fp16 2.6e-3 rel, bf16 2.1e-2 rel) says
   nothing about real activation ranges. A trained FastConformer can have large residual-stream values, which risks
   fp16 overflow or underflow in FF intermediates. Validate token-level parity of the full system before switching
   from fp32. bf16 has the range but is 10× noisier; fp16 has the precision but needs a range check. Training used bf16
   autocast, which argues bf16 is tolerable.
3. **[56,0] start artifact in NeMo, realtime.py and live.py:** the zero-filled pre-encode cache at step 1 is not
   offline-exact (0.037 with random weights, persisting through the caches). It is irrelevant for s4 ([56,3]), but
   anyone reusing `vapasr/hf/live.py` for [56,0] inherits it. Use the exact rule (§2).
4. **Audio front-end mismatch, the most likely real-world quality risk.** Browser `getUserMedia` defaults turn on
   echoCancellation, noiseSuppression and autoGainControl. Disable them to get the raw signal closer to training data.
   Browsers run at 44.1 or 48 kHz, so resample to 16 kHz with an anti-aliased resampler (§1.3). Clipping and levels:
   s4 trained with gain augmentation (`--gain-augment 0.5,-70,-20` for English, `slurm/submit-q06-s4-ast.sh:31`).
   A *muted* mic delivers exact zeros, giving log-mel −16.64 everywhere. Training always had dither noise there, but
   eval used digital-zero tails successfully (§6), so this is likely fine.
5. **Utterance-end conventions:**
   - Training and eval truncate features to K = round(N/1280).
     **[critic]** Eval uses `int(round(N/16000*12.5))`, which differs by ±1 at 16 lengths below 31 s (§4.4). Training
     uses the item's K (§6).
   - Eval appends 1 s of digital silence before decoding (`eval_single_turn_asr.py:288-299,431`).
   - A live app has no natural "end". Either keep the encoder running (state is bounded: 56-frame K/V cache plus an
     8-frame conv cache, and positions are relative, so there is **no length limit in the encoder**), or on stop
     append 1 s of zeros, call `finish()` and cut to K.
   - The thinker's context length is the real limit (out of scope).
6. **Dither:** training ran the preprocessor in train mode (dither 1e-5) and inference does not. Measured effect ≤0.045
   on speech frames and ≤0.13 in digital silence (§1.1). This is believed harmless; adding dither at inference would
   break determinism and parity.
7. **Tight finality rule** (n ≥ 160t+199) depends on the window's exact zero endpoints. The checkpoint `window` buffer
   is the symmetric Hann with w[0]=w[399]=0.0 (probe), but if in doubt use the conservative rule (n ≥ 160t+256), which
   costs 3.6 ms.
8. **MLX version drift:** verified on mlx 0.32.2 (conv2d `groups`, SDPA additive-mask semantics,
   `take_along_axis`, `mx.compile` with list pytrees). Re-run `encoder_probe2.py`'s MLX section (or the §8.6 test)
   if the MacBook has another version.
9. **Scheduling:** the encoder and the thinker share the GPU. Run encoder steps and thinker decode on one worker thread
   (or serialize MLX calls) so neither starves the other. The encoder step must start as soon as a chunk is final, at
   256 + 320c ms.
10. **pos-emb numerics:** NeMo builds `pe` in torch fp32; the reference uses numpy fp32 sin/cos, which can differ by
    one ulp. This is included in the measured 1e-5 total and is negligible.
11. **Mel FFT numerics:** numpy/MLX FFT vs torch.stft differ by ≤2.9e-5 in log-mel (§1.2), which is included in the
    live-pipeline measurement (1.1e-5 at the encoder output).
12. **[critic] Live frame index K before the endpoint** (§6). Without gating, 7.5 % of zero-tail endpoints decode an
    encoder frame that eval never decodes. That is a parity break visible as an extra round of tokens. The gate
    `k < K(n_received)` costs ≤ 24 ms on every 4th frame. Cross-spec: `audio-ui-spec.md` §5 still writes
    `K = round(n_samples / 1280)` (its lines ~417 and ~433) despite its own critic item 2. Both specs must use the eval
    expression.
13. **[critic] Eval features came from the CUDA Triton fused subsampling kernel** (probably; §2), while the MLX port and
    any Mac torch reference follow the PyTorch path. Expected difference is fp32-rounding level, but it is unmeasured.
    If §8.6 tokens differ from published eval tokens, check here first, before blaming the port.
14. **[critic] Provenance of `fb` and `window`.** The checkpoint buffers are NVIDIA's .nemo copies (§5). Recomputing
    them with the local librosa is verified identical only to a fresh NeMo build, not to the .nemo copies. Always load
    them from the checkpoint.
15. **[critic] Training context evidence is indirect.** The att_context-fixing code is uncommitted (§6). That s4 trained
    with fixed [56,3] rests on the launch flags plus the 2026-10-08 Stage-1 log; there is no s4 log locally. If the
    eval or real-checkpoint parity looks off, confirm `config.json` → `semcommit.encoder_context_sampling == "fixed"`
    and `semcommit.encoder_right_context == 3` (fingerprint keys, `semcommit_train.py:343-344`; `config.json["semcommit"]`
    is checked at `:455`). This is the first thing to read once the checkpoint is on the MacBook.

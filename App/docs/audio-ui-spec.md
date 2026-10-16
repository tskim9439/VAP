# VAPKT Live App: audio I/O, segmentation, protocol and UI spec (v1, 2026-10-16)

Scope: the product and runtime layer of `App/`, meaning everything around the MLX engine. Model internals (mel, encoder, adapter, thinker, weight
conversion format) belong to the engine spec. This document states only what the app needs from the engine (§5.3).
Notation: `file:line` cites the working tree of `/Users/taesookim/Desktop/VAPKT` unless a path is absolute. "Measured" means it was run on
this Mac mini M4 on 2026-10-16. The throwaway scripts are in `scratchpad/scratch-audioui/`.

> **[critic] Review pass (2026-10-16).** An adversarial review re-derived the load-bearing claims from the working tree and re-ran the
> probes. Its scripts are in `scratchpad/scratch-critic/`. Every change is marked **[critic]** in place. The most important ones:
> 1. **Off-by-one at segment end (parity-breaking).** If the zero tail goes through `feed()`, the chunks it completes can contain encoder
>    frame index **K**, and eval never decodes that frame. This happens for 7.6 % of lengths (`kdiff.py`). Fix: `finish(zero_tail_samples)`
>    appends the tail itself, computes K, and decodes only k < K (§5.2, §5.3).
> 2. **K formula.** `round(n/1280)` ≠ eval's `int(round(n/16000*12.5))` at 36 lengths below 45 s (e.g. n = 69 760 gives 54 vs 55). Use the eval expression.
> 3. **`runaway_cap` is 32 for s4, not 8** (`vapasr/data/semcommit_tokens.py:86,98`, called at `experiments/semcommit_train.py:215-217`).
>    The prefix is 14–20 ids, not 25–40 (§1.6).
> 4. **δ rationale was wrong.** For s4, δ8 and δ10 *are* better than δ6 for EN/DE ASR, and translation BLEU is *not* monotone in δ (en→ko flat, ko→en peaks at δ8).
>    The defaults were re-justified and ko→en now defaults to δ8 (§2.1).
> 5. **The fast-file flow control deadlocks on silent stretches** (`audio_proc_s` never counts skipped audio). It now uses `audio_done_s` (§4.5).
> 6. **The forced-cut search window** is the last 640 ms (unreleased frames), not [23 s, 25 s]. That is what the prototype actually implements (§1.4).
> 7. Smaller items:
>    - causal noise-floor initialization
>    - lead-in default off, with a no-history guard
>    - a minimum zero tail and a minimum `tail_s` per δ
>    - flush the resampler at stop
>    - a hard KV clamp
>    - Python interpreter arch check (this box has an x86_64 `python3.12` first on PATH)
>    - `tokenizer.json` requirement
>    - CSP inline gotcha
>    - checkpoint `final/` subdirectory
>    - `pytest` dependency

---

## 0. Decisions at a glance

| Topic | Decision |
|---|---|
| Unit of decoding | **Segment** = one utterance-like span cut from the mic stream by an energy VAD. Each segment gets a fresh encoder cache, fresh thinker KV and a fresh prefix, so it is decoded exactly like one eval utterance. |
| Segment shape | 480 ms pre-roll; ends after an 800 ms pause. After 18 s the pause threshold drops to 256 ms (speech-relative). Hard max is 25 s, with a cut at the quietest point inside a 640 ms hold-back window. Before flush the decoder always sees at least 1.0 s of post-speech audio (real pause plus digital zeros), which is the eval protocol. |
| Level | Server-side **per-segment constant gain** toward −26 dBFS median voiced-frame level (deadband ±6 dB, clamp −10..+30 dB). On by default. Browser EC/NS/AGC are **off** by default (toggles). |
| Capture | AudioContext at the **device rate**. An AudioWorklet sends **Float32 mono, 20 ms packets**. The server resamples to 16 kHz with **soxr HQ** (streaming). The old demo's naive decimation aliased, see §3.2. |
| Transport | One WebSocket. JSON control plus raw binary PCM. Revision-free append-only text deltas for two streams (`src`, `tr`). Audio is never dropped: backlog is reported, and the session errors out explicitly only at 600 s of backlog. |
| Server | aiohttp on the asyncio thread (I/O, resample, VAD, gain) plus **one inference worker thread that owns every MLX object**. MLX arrays cannot cross threads; this was measured. One active session at a time, with takeover. |
| Port / bind | `127.0.0.1:8796`. If busy, scan 8796–8805 and skip 8765 and 8790. Auto-open `http://127.0.0.1:<port>/`. |
| UI | One static page (`web/index.html` + `app.js` + `worklet.js` + `style.css`), no build step and no CDN, Korean labels. Two aligned panes (원문 전사 / 번역), one row per segment. Status bar shows algorithmic latency, processing lag, backlog and RTF. |
| Final re-decode | Implemented as an optional P1 feature, **default off**, low priority. It is shown under the streaming text and never replaces it. |
| Packaging | `App/setup.sh` creates `~/.vapkt-app/venv` with pinned deps (mlx, numpy, safetensors, tokenizers, aiohttp, soxr, soundfile) and honors `~/.vapkt-ca-bundle.pem`. Converted weights go next to the checkpoint in a `*-mlx.nosync` dir (or on the T5). `App/run.sh` runs under `caffeinate`. |

---

## 1. Training-distribution constraints and live segmentation

### 1.1 Evidence

| Fact | Value | Source |
|---|---|---|
| Training item = stream, not a sentence | English LibriSpeech: same-chapter utterances concatenated into **20–30 s** streams (target 25 s), lead 0.3–1.0 s, inter-utterance gaps **0.3–1.5 s**, tail 0.3 s | `experiments/s1_build_manifest.py:4,35,96-109` |
| Single-utterance rows (Kspon etc.) | lead and trail **U(0.3, 1.0) s** each | `experiments/s1_build_manifest.py:5,112-119` |
| Approved speechlm packs (KO) | target 22 s, **max 25 s**, silences 0.3–0.8 s | `experiments/selection_prepare_training.py:89-90` |
| speechlm-partial rows | 8–30 s recipe, max 30 s, `offset_s=0, silence_before_s=0` (natural lead only); a separate short pool (<8 s) | `experiments/semcommit_build_speechlm_candidates.py:85,98,171`; `experiments/semcommit_train.py:97` |
| CV4 English (s4 addition) | 1–30 s clips (232,845 rows / 351 h → mean ≈ 5.4 s) | `experiments/cv_en_align.py:331-332`; `slurm/submit-q06-s4-ast.sh:6` |
| German YODAS | 1–30 s plus a long pass of 30–40 s | `experiments/yodas_de_align.py:15,382`; `wiki/decisions/decision-yodas-de-asr-selection.md:45` |
| Corpus mix (q06 Stage-1 base of s4) | EN 456,893 streams / 2,143 h (mean 16.9 s), KO 2.96 M / 4,990 h (6.1 s), KO short 945 k / 1,122 h (4.3 s), DE 1.32 M / 2,125 h (5.8 s) | `wiki/outputs/output-stage1-trilingual-d10-prep-20261008.md:78` |
| AST rows = the same Stage-1 rows | translation targets were generated for all Stage-1 training rows (5.68 M rows), so translation mode also saw 20–30 s multi-sentence streams | `wiki/decisions/decision-ast-monotonic-data.md:31,45` |
| Synthetic silence is **not zero** | gaps and pads are the neighbour's 20 ms background, tiled with random order and sign, ×0.7 | `vapasr/data/streams.py:3,67-86` |
| Tail padding | every item is padded with background up to `max(K, last emission chunk at δmax + tail_margin 2)`. **[critic]** In frames that is K' ≥ ⌊t_last/0.08⌋ + δmax + 2, so the audio after the last token time is **(0.88, 0.96] s** for δmax = 10 (frame quantization), not "≥ 0.96 s". It is longer when the natural tail is longer. The training-side mean added padding is 0.25–0.66 s | `vapasr/data/semcommit_dataset.py:283,370-372` (working tree); `vapasr/data/ast_dataset.py:170-174` (speed-perturbed items); `wiki/decisions/decision-stage1-lookahead-320-1040ms.md:45` |
| **[critic]** Bucketing rule | token time t → chunk `k = min(n_chunks, int(t/0.08) + δ)`; s4 delays 4–10 | `vapasr/data/interleave.py:29`; `vapasr/data/semcommit_dataset.py:19` (doc); `vapasr/data/ast_dataset.py:6` |
| **[critic]** Per-round cap | `runaway_cap = max(old or 8, 32) = 32` for every ast-v1 model, including s4 | `vapasr/data/semcommit_tokens.py:86,98`; `experiments/semcommit_train.py:215-217` |
| Eval protocol (all reported numbers) | raw utterance **+ 1.0 s digital silence**. No leading silence, no VAD. Eval manifests are ≤ 30 s. Flush ≤ 8 rounds | `experiments/eval_single_turn_asr.py:4-6,288-299,577-578`; `slurm/submit-eval-q06-s4-ast.sh:24-30` |
| Prefix | `language <SPK_A>{src}[\ntranslate <SPK_A>{tgt}]<asr_text><DELAY_δ>`. Training picks A or B at random; eval uses A | `vapasr/data/ast_dataset.py:4,40-42,233`; `experiments/eval_single_turn_asr.py:375-379` |
| Pairs trained | en→de, en→ko, de→en, ko→en; ASR for en, ko, de | `wiki/decisions/decision-ast-monotonic-data.md:31`; `experiments/ast_build_targets.py:31` |
| Front end has no level normalization | Nemotron preprocessor `normalize: NA`, dither 1e-5 in training mode only | `scratchpad/hf-nemotron/model_config.yaml:568,576`; NeMo `features.py:427-429` |
| Quiet-audio failure | FLEURS-en test RMS ≈ −60 dBFS (64 % with peak < 0.02). The streaming model emitted nothing on 25 % of utterances (en→de δ4). CoVoST had 65 silent cases at RMS −55 dBFS | `wiki/outputs/output-q06-s2-ast-eval-20261011.md:130-135` |
| Gain augmentation (s3/s4) | p = 0.5, RMS uniform in **[−70, −20] dBFS**, **English only**, applied after noise and reverb as a constant gain per item | `slurm/submit-q06-s4-ast.sh:31`; `vapasr/data/ast_dataset.py:106-113,241-242`; `experiments/semcommit_train.py:57-58` |
| Other augmentation | speed 0.9/1.0/1.1, SpecAugment, noise p 0.4 at SNR 5–30 dB, reverb p 0.2 | `slurm/submit-q06-s4-ast.sh:29`; `experiments/semcommit_train.py:100,104-107` |

Measured levels of local corpus samples (median of voiced 32 ms frames ≈ "active RMS"; script `scratch-audioui/levels.py`):
LibriSpeech dev streams −22…−25 dBFS (lead 0.7–1.4 s, trail 0.5–0.8 s), VoxPopuli −21…−26, NIKL −27…−32,
**KsponSpeech −34** (lead 0.35–0.48 s), AMI far-field −40…−64. Training corpora therefore sit at about **−21 to −35 dBFS**. Korean and German
have no gain augmentation, so a quiet Korean mic could reproduce the FLEURS silence.
The `scratchpad/testaudio/*.wav` clips are peak-normalized (−14…−17 dBFS) and **tightly cropped (lead/trail ≈ 0)**, so file replay of them depends on the 1 s tail rule.

**[critic] Not every language saw long streams equally.** The multi-sentence 20–30 s streams are mainly English LibriSpeech (EN mean 16.9 s).
- **Korean** items average 6.1 s (main) and 4.3 s (short pool, which s4 uses: `slurm/semcommit-train-apex.sbatch:45`). The long Korean items are the approved speechlm packs (target 22 s, max 25 s).
- **German** averages 5.8 s.
- The **CV4 English** clips added in s4 average about 5.4 s.

Long KO and DE segments are therefore in range but less typical than for English. Keep `end_pause_ms` at 800 so normal conversational turns stay short, and treat KO/DE quality on segments over 15 s as something to watch in G2.

**Known s4 weakness:** in ko→en translate mode the Korean transcript sometimes stops after the first few words. This happened in 3 → 9 FLEURS sentences, with pooled CER 8.28 → 10.18 % versus s3 (`wiki/outputs/output-q06-s4-ast-eval-20261013.md:62-67`). The UI README should say so.

**[critic]** The level column above comes from `scratch-audioui/levels.py`, which takes the **mean power of frames > p10 + 12 dB**, not a median. The live `L_spk` (§1.3) is a median of per-frame dB over `T_on`. On the six testaudio clips the two measures differ by −1.5…+0.3 dB (`scratch-critic/lvl_cmp.py`), so the target and the deadband below are unaffected. Training `apply_gain` instead uses **whole-waveform RMS including pauses** (`vapasr/data/ast_dataset.py:106-113`), which reads about 1–1.5 dB lower than active level on these clips.

### 1.2 Rules that follow

- **R1, segments not one endless stream.** The thinker KV and positions grow without bound and training never exceeded about 30–40 s. Each segment resets
  encoder cache + KV + prefix, so it is an exact eval utterance.
- **R2, multi-sentence segments are in-distribution.** Training streams contain pauses up to 1.5 s, so the end-of-segment pause can be long (800 ms).
  Splitting at every 300 ms hesitation would throw away cross-sentence context, which matters for ko→en reordering.
- **R3, length cap 25 s** (KO pack max 25 s, LS 20–30 s, eval ≤ 30 s). This also bounds KV positions (§1.6).
- **R4, leading non-speech ≥ about 0.3 s** (training lead 0.3–1.0 s), hence a 480 ms pre-roll.
- **R5, ≥ 1.0 s of post-speech audio before flush**: training pads 0.88–0.96 s after the last token time **[critic: was "(δmax+2)·80 ms = 0.96 s"]**,
  and eval adds 1.0 s. A real pause plus digital zeros (zeros are the eval protocol) satisfy both. This costs compute only, not wall-clock waiting.
  **[critic]** The post-speech audio must cover the emission point of the last word: ≥ (δ+2)·80 ms. Shorter tails push the last tokens into
  `<EMPTY_AUDIO>` flush rounds, which training never targets because the dataset pads the tail precisely to avoid that (`ast_dataset.py:10`).
- **R6, absolute level matters** (`normalize: NA`). Apply one constant gain per segment, as in training (`apply_gain` is per item), never a time-varying AGC
  inside a segment.

### 1.3 Server audio framing and VAD

All VAD and segmentation runs **on the 16 kHz audio clock** (resampled samples counted since `start`), never on wall clock. The same audio
therefore yields the same segments whether it arrives at 1×, 4× or in bursts.

- **Frame** = 512 samples = 32 ms. Segment boundaries lie on this grid. The encoder's chunk *c* becomes computable once the segment
  holds 4096 + 5120·c samples (first chunk 25 mel frames, then 32, plus 256-sample STFT tail; §2.1), and 4096 + 5120c = 512·(8 + 10c).
  Chunk completion therefore always coincides with a frame boundary and framing adds no wait.
- **VAD feature**: first-order 80 Hz high-pass (VAD path only; the model gets unfiltered audio), then `E = 10·log10(mean(x²) + 1e-10)` dBFS per frame.
  **[critic]** scipy is not a dependency. The prototype's per-sample Python IIR (`seg_proto.py:hp`) costs 2.7 ms CPU and holds the GIL per second of audio,
  so at "최대" file speed (tens of ×) it competes with the worker's token loop. Use the vectorized equivalent instead: `rfft` of each 512-sample frame, zero bins 0–2
  (< 94 Hz), and sum the power. That costs 0.24 ms per second of audio (`scratch-critic/hp_cost.py`). Unit-test the thresholds on whichever implementation ships.
- **Noise floor** (minimum tracking with slow rise): `N ← clip(min(E, N + 0.016 dB), −85, −35)`, a rise of 0.5 dB/s.
  Initialized to min E over the first 16 frames (0.5 s). The UI asks for a moment of quiet at start ("주변 소음 측정 중…").
  **[critic] Causality:** the prototype initializes N from `e[:16]` before processing frame 0 (`seg_proto.py:run`), which is non-causal. Live, frames 0–15 are buffered
  until N is known and then run through the state machine in order. The pre-roll ring must therefore hold ≥ 16 + pre_roll frames at session start.
  This costs a one-time 0.5 s decision delay and changes no boundary relative to the prototype. File replay and live behave identically.
- **Thresholds**: `T_on = max(N + 10, −60)`, `T_off = max(N + 6, −64)` dBFS.
- **Speech level** `L_spk` = median E of the last 150 frames with `E ≥ T_on` (about 4.8 s of speech). It is used by the soft end rule and by gain.

### 1.4 Segmenter state machine (parameters are UI/`start.options` overridable; defaults shown)

| Param | Default | Range | Why |
|---|---|---|---|
| `pre_roll_ms` | 480 (15 fr) | 160–960 | R4. Also covers the 96–128 ms onset-confirm delay and soft onsets (s, h). |
| onset rule | ≥ 3 of the last 4 frames ≥ `T_on`; onset = first of them | fixed | rejects clicks; adds ~0.1 s, which the pre-roll absorbs |
| `end_pause_ms` | 800 (25 fr) | 300–2000 | R2. A pause is frames with `E < T_off`. |
| `soft_max_s` / `end_pause_soft_ms` | 18 s / 256 (8 fr) | — | After 18 s a pause is `E < max(T_off, L_spk − 15 dB)` for 256 ms (micro-pauses in continuous speech). **[critic]** Derive it from the hard max: `soft_max_s = hard_max_s − 7` (≥ 0.6·hard_max). A fixed 18 s is meaningless when the user sets `hard_max_s` < 18. |
| `hold_from_s` / `hold_ms` | 23 s / 640 (20 fr) | — | From 23 s the worker receives audio 640 ms late, so a cut point can still be chosen. **[critic]** Derive it as `hold_from_s = hard_max_s − 2`. Starting the hold at that moment stalls the worker once for 640 ms, and the lag then persists to the segment end, adding 640 ms to the displayed latency (§2.3). |
| `hard_max_s` | 25 | 10–30 | R3. Cut at argmin of the 3-frame moving average of E over **unreleased** frames. **[critic]** Because the release pointer lags by a sliding 640 ms, the unreleased frames at the moment of hard max are only **[hard_max − 0.64 s, hard_max] = [24.36 s, 25 s]**, not [23 s, 25 s]. Frames before that were already decoded. The prototype does exactly this: `lo = max(start + hold_from, t + 1 − hold)` (`scratch-audioui/seg_proto.py`). NIKL cut at 24.76 s. A wider search would need a longer hold, which costs latency. |
| tail rule | `zero_tail_ms = max(200, tail_ms − trailing_nonspeech_ms)` with `tail_ms = 1000` **[critic: floor was 0]** | `tail_s` **[critic]** `max(0.5, 0.08·(δ+2))`–1.5 | R5 (pause end → +200 ms; soft → +744; forced/kv → +1000). **[critic]** The 200 ms floor matters: with `end_pause_ms` ≥ 1000 (the UI allows 2000) the old rule gave a 0 ms tail, and K = round(n/1280) can then be ≤ the index of a frame already decoded from the real pause (same arithmetic as §5.3). Any tail ≥ 104 ms (1664 samples) guarantees that every frame decoded before the end decision has k < K; 200 ms rounds this up. With a 0 ms tail, 37 252 of 491 904 lengths violate it, and with ≥ 100 ms none do (`scratch-critic/tailfloor.py`). A `tail_s` below (δ+2)·80 ms (e.g. 0.5 s at δ10) pushes the last word's tokens into flush rounds, which training never targets (R5). |
| lead-in | if real pre-roll < 160 ms (soft or forced cut, or a back-to-back onset): prepend 320 ms of background = tile of the 3 quietest frames of the last 3 s × 0.7 (as `streams.silence_like`) | on/off, **[critic] default OFF** | training items mostly start with background (R4). Effect unverified, see acceptance C6. **[critic]** Default off because every reported number used **no leading silence** (`experiments/eval_single_turn_asr.py:4-6`), speechlm-partial training rows start with `silence_before_s = 0` (`experiments/semcommit_build_speechlm_candidates.py:171`), and the prototype's cropped test clips start at sample 0. When on, it must be skipped if fewer than 3 s of session history exist or if the 3 quietest frames are ≥ `T_on`, or it would tile *speech* (e.g. at file start). Never apply it in `segmentation:"none"`. |
| `kv_guard` | force end if engine positions > 0.85 × KV capacity | — | §1.6 |
| noise segment | segments with < 6 voiced frames **and** no text are hidden in the UI (kept in the log) | — | coughs and clicks |

State machine (per frame *t*):

```
SILENCE --(>=3/4 frames >= T_on)--> open segment: start = max(onset - pre_roll, prev_end, 0)
   (the frames start..t are released to the worker at once: a burst of about 0.6 s)
SPEECH  : voiced := E >= thr (thr = T_off; after soft_max: max(T_off, L_spk-15));  last_voiced updated
          t - last_voiced >= end_pause(_soft)          -> END(reason="pause" | "soft_max"), end = t+1
          length >= hold_from                          -> release pointer lags received by hold_ms
          length >= hard_max                           -> END(reason="max_len") at quietest unreleased frame;
                                                          the next segment starts at the cut (flag forced_start), inherits the held audio
          engine.positions > 0.85*capacity             -> END(reason="kv_guard") at t+1; next segment starts at t+1
          stop / file end                              -> END(reason="stop" | "file_end") at last received sample
```

Invariants: **every received sample belongs to at most one segment**. B's start is never earlier than A's end, so no duplicated words from overlap.
Samples outside segments are counted as `skipped_s` and never decoded. `received = in_segments + skipped + held`, and `held = 0` at session end.
**[critic]** Two clarifications:
- Synthetic samples (lead-in background, zero tail) are *not* received samples. They are counted separately (`lead_in_ms`, `zero_tail_ms`).
- `held` includes the VAD look-back (pre-roll ring and onset-confirm frames while in SILENCE) as well as the hold-back.
- **No retroactive cut** (`decode-contract.md` §d.5(d)). An end point is never earlier than the worker's release pointer. Samples already released may have entered
  finalized mel frames, which use samples up to 160t+255. Pause, soft, stop and kv ends are at the newest received frame. The forced cut is chosen only among unreleased (held) frames,
  so this holds by construction. A unit test must assert `cut ≥ released` for every forced end.

**Segmentation `none`** (file replay only, eval-parity mode): the whole input is one segment from sample 0 to the end, then exactly 1.0 s of zeros,
with gain off and no lead-in. That is the eval protocol `np.pad(wav, (0, 16000))` (`experiments/eval_single_turn_asr.py:299`). It is refused for inputs > 30 s.

### 1.5 Gain normalization (server, default **on**)

- Measure on raw (pre-gain) audio. Target `T_L = −26 dBFS` median voiced-frame level, which sits mid-way in the −21…−35 corpus range of §1.1.
- At **segment start**: `g = T_L − L_spk`. If |g| ≤ 6 dB then g = 0 (deadband: corpus-range levels are left untouched). Clamp g to [−10, +30] dB.
  Then cap g so that (peak of the last 3 s) × 10^(g/20) ≤ −1 dBFS. For the first segment, with no history, use the voiced frames of the onset window and cap at +20 dB.
- g is **frozen for the whole segment**, the same as training `apply_gain` (one gain per item, `vapasr/data/ast_dataset.py:106-113`). A safety soft-limiter
  (`y = sign(x)·(0.95 + 0.05·tanh((|x|−0.95)/0.05))` above 0.95) is counted as `limited_samples` and should be ~0.
- Rationale: English is robust down to −70 dBFS (augmented). Korean and German are not augmented and Kspon is already at −34 dBFS, so a quiet MacBook mic
  without browser AGC can fall below the trained range. Zeros (tails) stay zero. SNR is unchanged.
- UI: toggle "서버 음량 자동 맞춤". Each segment header shows its gain ("이득 +12 dB").

### 1.6 KV and position budget (the app sizes, the engine enforces)

Per segment: prefix + 2 positions per 80 ms frame (audio + pending `<NEXT_AUDIO>`) + text.
- **[critic]** The prefix is **14–20 ids** (8 header + 5–11 language line + 1 `<DELAY_δ>`; e.g. ASR en 14, en→de 19, ko→en 20; `decode-contract.md` §b.2–b.3), not "≈ 25–40".
- At 25 s, K ≈ 313, plus 12.5 tail frames and ≤ 8 flush rounds, gives ≈ 670 audio/NEXT positions.
- AST text is about 3 transcript and 3–5 translation tokens per second plus tags, so ≈ 200 ⇒ **≈ 900**. At `hard_max_s = 30`, ≈ 1 060.
- A 2048 KV leaves about 2× headroom.

The runaway caps are:
- per round: **[critic] `config.runaway_cap` = 32 for s4**, not 8. `add_ast_tokens` raises it to `max(old or 8, 32)` (`vapasr/data/semcommit_tokens.py:86,98`), and s4 training called it (`experiments/semcommit_train.py:215-217`). 8 is only the class default (`vapasr/hf/configuration_vapasr.py:17`). Read it from `config.json`.
- total: `6·K + 64` (`vapasr/hf/modeling_vapasr.py:231`).

**[critic] Causal total cap, aligned with `decode-contract.md` §d.6.** During streaming use `max_total = 6·K_cap + 64`, where `K_cap = int(round((hard_max_s + tail_s)·12.5))` (= 325 at the defaults). When `finish()` knows the final K, it switches to the exact `6·K + 64` for the remaining tail and flush rounds. This is never stricter than eval, so it never cuts a token eval would keep. Tests skip references with `total_cap_hit`.

**[critic] Hard KV clamp.** With a 32-token round cap, a runaway frame uses 34 positions, so the 0.85 `kv_guard` (an end decision on the asyncio thread) is not enough by itself: the tail plus flush can need up to (13 + 8)·34 ≈ 714 positions. The engine must also refuse to write past `kv_capacity`. When `positions + 2 > kv_capacity` it ends the round as forced, skips the remaining rounds, reports `kv_full` in the Round, and the app ends the segment with reason `kv_guard`. This only happens in pathological runaway.

### 1.7 Prototype check on local audio (energy VAD exactly as §1.3–1.4, `scratch-audioui/seg_proto.py`)

| File | Result (start–end s, reason) |
|---|---|
| LS dev-clean 34 s stream | 0.48–3.84 pause · 4.35–18.24 pause · 18.56–27.42 pause · 28.45–33.98 eof. Gaps < 0.8 s merge (multi-utterance segments as in training); pre-roll 0.48 s each |
| LS dev-other 34 s | 4 segments, all pause/eof |
| VoxPopuli 28.7 s continuous | 0.00–21.57 **soft_max** (cut at a 0.29 s micro-pause found only by the speech-relative rule; `T_off` alone found none) · 21.57–28.64 |
| NIKL 35 s (overlapped talk, quiet −46 dBFS, 4.3 s digital-zero lead) | 3.78–28.54 **max_len** (no pause at all) · 28.54–34.98 forced_start |
| AIHub 71631 24 s | 5 segments, pause-ended |
| Kspon 4 s / 8 s | one segment each |
| testaudio de/en/ko (cropped) | one segment each, starting at 0 (pre-roll 0, so the 1 s tail rule matters) |

---

## 2. Latency accounting

### 2.1 Algorithmic latency (model plus front end, zero compute)

- Training rule: a token for a word ending at time *t* is emitted at decoder step `s = ⌊t/80 ms⌋ + δ` (`vapasr/data/semcommit_dataset.py` doc, `build_interleaved_events`).
- Encoder `[56,3] chunked_limited`: `chunk_size = [1 + 8·3, 8 + 8·3] = [25, 32]` mel frames, `pre_encode_cache = [0, 9]` (NeMo
  `conformer_encoder.py:1043,1074-1077,1097`; `subsampling.py:448-452`; same as `vapasr/hf/realtime.py:11-13`). STFT `center=True` zero-pads 256 samples, so
  mel frame *m* is final at sample 160·m + 256 (NeMo `features.py:385-388`; `vapasr/hf/live.py:16`).
  ⇒ decoder step *s* is computable once the segment holds **4096 + 5120·⌊s/4⌋ samples** (0.256 s + 0.32 s·⌊s/4⌋).
- Lag from word end to "computable", with *u* the position of *t* inside its 80 ms frame:
  `0.256 − 0.08·(s mod 4) + 0.08·δ − u`, giving **mean ≈ 80δ + 96 ms, range ≈ [80δ − 64, 80δ + 256] ms**.
  The wiki convention (`decision-stage1-lookahead-320-1040ms.md:29`: 80δ … 80δ + 240, mean 80δ + 120) is the same quantity measured at frame
  boundaries. Ours is 24–64 ms earlier because of the causal 16 ms mel offset (the wiki notes "특징 추출까지 따지면 약 0.06 s 이르다",
  `output-simulst-baselines-20261012.md:35`).
  **[critic] The explanation above is imprecise. The formula and table are correct (re-derived).** The wiki quantity runs from the **end of the word's 80 ms
  frame** to the **chunk boundary** 0.32·(⌊k/4⌋+1) s. Ours runs from the **word end t** to the sample at which the chunk is computable,
  0.256 + 0.32·⌊k/4⌋ s. That sample is 64 ms before the boundary because encoder frame 4c+3 needs mel frames only up to 32c+24 (`encoder-spec.md` §6), not the full chunk.
  So ours = wiki + (80 − u) − 64 ms. The mean is 120 + 40 − 64 = 96, and the range runs from wiki_min + 0 − 64 to wiki_max + 80 − 64, i.e. [80δ − 64, 80δ + 256].
  The upper bound is therefore **16 ms later** than the wiki's, not earlier.

| δ | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|
| mean (ms) | 416 | 496 | 576 | 656 | 736 | 816 | 896 |
| range (ms) | 256–576 | 336–656 | 416–736 | 496–816 | 576–896 | 656–976 | 736–1056 |

- Translation adds the model's own reordering lag. The eval LAAL already includes the chunk wait and excludes compute
  (`output-q06-s4-ast-eval-20261013.md`, δ4 / δ10, FLEURS): en→de 1.52 / 2.00 s, en→ko 3.06 / 3.65, de→en 1.55 / 1.90, ko→en 2.11 / 2.72.
- Input path (measured or by construction): worklet packet 20 ms (avg 10), soxr HQ block buffering ≤ 37 ms at 48 k (avg ~18, §3.2), WS on localhost < 1 ms.
  ≈ **30–60 ms**, shown as a constant.
  **[critic] Re-measured** (`scratch-critic/soxr_lag.py`, 20 ms packets, 10 s noise), HQ buffering is mean / max:
  - 48 kHz: 21 / 37 ms
  - 44.1 kHz: 23 / 40 ms
  - 96 kHz: 43 / 79 ms

  Show the constant per input rate: ≈ 31–57 ms at 44.1/48 k and ≈ 53–99 ms at 96 k, plus the browser's own `track.getSettings().latency` / `ctx.baseLatency` when reported.
- Defaults: **ASR δ6** (s4 LibriSpeech test-other δ4 7.43 % vs δ6 6.54 %; δ8–10 are no better than δ6, `output-q06-s4-ast-eval-20261013.md`,
  `decision-stage1-lookahead-320-1040ms.md:67`). **Translate δ10**: BLEU and xCOMET rise monotonically with δ. The UI remembers δ per mode.
  **[critic] Both rationales are wrong for s4.** The cited `:67` line is about the Stage-1 model q06-s1m3-d10, not s4. s4 numbers come from
  `raw/sources/experiments/2026-10-13-q06-s4-ast-eval/summaries.json`.

  ASR error rate %:

  | set | δ4 | δ6 | δ8 | δ10 | final |
  |---|---|---|---|---|---|
  | LS test-other WER | 7.43 | 6.54 | 6.28 | 6.14 | 5.23 |
  | LS test-clean WER | 3.20 | 2.67 | 2.58 | 2.57 | 2.22 |
  | YODAS-de WER | 4.11 | 3.38 | 3.13 | 3.03 | 2.10 |
  | Kspon clean CER | 9.17 | 8.32 | 8.25 | 8.24 | 7.30 |
  | VoxPopuli WER | 7.36 | 6.78 | 6.92 | 6.87 | 7.12 |

  - δ8/δ10 are **4–10 % relatively better than δ6 for EN read speech and DE**, and flat for KO and VoxPopuli.
  - δ6 is kept as the ASR default because it gets 70–100 % of the δ4→δ10 gain at 0.32 s less algorithmic latency. The δ hint should say "δ8–10: 영어·독일어 약간 더 정확".

  Translation, FLEURS BLEU (raw LAAL s, without the +0.11–0.17 s chunk correction):

  | pair | δ4 | δ6 | δ8 | δ10 |
  |---|---|---|---|---|
  | en→de | 21.1 (1.42) | 23.5 (1.56) | 23.7 (1.76) | **24.3** (1.89) |
  | en→ko | 29.7 (2.89) | 29.2 (3.04) | 29.5 (3.27) | **29.7** (3.52) |
  | de→en | 26.1 (1.43) | 27.3 (1.51) | 28.2 (1.66) | **28.8** (1.78) |
  | ko→en | 14.5 (1.99) | 14.8 (2.19) | **15.5** (2.43) | 14.9 (2.59) |

  - BLEU is **flat for en→ko** and **peaks at δ8 for ko→en**. xCOMET is only reported for δ4/δ10 and favours δ10 in every pair (`output-q06-s4-ast-eval-20261013.md:42-49`).
  - Defaults per pair: **δ10 for en→de, de→en and en→ko** (en→ko on xCOMET 66.2 vs 64.1), and **δ8 for ko→en** (best BLEU and 0.16 s earlier). The ko→en choice is weak evidence: there is no δ8 xCOMET or bootstrap, and ko→en δ10 is the one cell that significantly regressed s3→s4 (`:62`).
  - `hello.defaults.translate_delay` therefore becomes a per-pair map (§4.3).
  - The UI LAAL hint uses the corrected wiki values for δ4/δ10 (en→de 1.52/2.00, en→ko 3.06/3.65, de→en 1.55/1.90, ko→en 2.11/2.72) and raw + 0.12 s for δ6/δ8, labelled "≈".

### 2.2 Measured quantities (server; every MLX timing taken after the result is evaluated, e.g. `.item()`/`mx.eval`)

| Name | Definition |
|---|---|
| `enc_ms` | wall time of one encoder chunk step (320 ms of audio) |
| `dec_ms` | wall time of one decoder round (one 80 ms audio position + its tokens until `<NEXT_AUDIO>`) |
| `chunk_ms` | `enc_ms` + its 4 rounds. **Real-time condition: `chunk_ms` < 320 ms** (`experiments/stream_rtf.py:6-8`) |
| `rtf` | Σ(enc + dec) / audio seconds decoded (incl. tail), the same as `stream_rtf.py` / wiki (`output-simulst-baselines-20261012.md:152-160`). Rolling 10 s and per session |
| `load` | Σ compute / wall over the last 10 s (≈ GPU busy fraction, includes idle silence) |
| `proc_lag_ms` (per round) | wall(round result ready) − wall(arrival at the server of the last sample that round needed). Tail and flush rounds use wall(end decision). It is queue wait plus compute. **[critic]** Rounds released during the ≥ `hold_from_s` hold-back include the deliberate 640 ms hold. Report them separately (`held: true`) and exclude them from the p50/p95 used for the colour thresholds and F1. |
| **[critic]** `audio_done_s` | session-clock seconds whose fate is final: skipped, or consumed by the engine (real samples only, excluding zero tail and lead-in). Equals `audio_rx_s − backlog_s − audio_held_s`. This is the flow-control signal (§4.5) |
| `backlog_s` | (16 k samples released to the worker − samples consumed by the engine) / 16000, i.e. how far the model is behind real time |
| `finalize_ms` (per segment) | wall(`segment_end`) − wall(end decision): zero tail, flush and final text |
| `first_text_ms` (per segment) | wall(first non-empty delta) − wall(arrival of onset frame) |
| `loop_lag_ms` | asyncio heartbeat jitter (guards against GIL hogging) |

Arrival wall time is recorded per received packet in a ring of (audio_clock, wall) pairs. In **file mode faster than 1×**, `proc_lag` and `backlog` grow by design, so the UI
shows throughput ("처리 속도 3.2×") instead of lag.

### 2.3 What the UI shows

- Next to δ: "알고리즘 지연 ≈ 0.58 s (0.42–0.74)". In translate mode also "번역 기준 LAAL ≈ 2.0 s (평가)".
- Status bar (rolling 10 s): `처리 지연 p50/p95`, `밀림 {backlog_s}`, `RTF`, `인코더 ms/청크 · 디코더 ms/라운드` (p50), a `청크 p95 / 320 ms` budget bar,
  `토큰/s`, `KV %`, `메모리 GB` (`mx.get_active_memory`), `입력 경로 ≈ 40 ms`.
- Colors: RTF < 0.5 green, 0.5–0.9 amber, ≥ 0.9 red. `proc_lag` p95 < 200 ms green, < 500 amber, else red. `backlog_s` > 1 s shows the warning "처리가 실시간보다 {x} s 늦습니다".
- Per segment row: time range, duration, end reason, `finalize_ms`, `first_text_ms`, gain, δ.
- **[critic]** While a segment is past `hold_from_s` (23 s), show "긴 발화: +0.64 s 대기" next to the latency hint. The hold adds that much to every token in the rest of the segment.

---

## 3. Browser audio capture

### 3.1 getUserMedia

```js
navigator.mediaDevices.getUserMedia({audio: {deviceId: sel ? {exact: sel} : undefined, channelCount: 1,
  echoCancellation: ui.ec /*false*/, noiseSuppression: ui.ns /*false*/, autoGainControl: ui.agc /*false*/}})
```

- Chrome turns all three on when unspecified. The old demo used EC on, NS off, AGC on (`experiments/live/index.html:44`).
  **New defaults: all off.** Reasons:
  - NS artifacts are out of distribution (training *adds* noise; it never removes it).
  - AGC is a time-varying gain inside a segment, against R6. Chrome's AGC may also move the macOS input volume.
  - EC is unnecessary because the app never plays to the speakers while the mic is live. In Safari, EC switches to Apple voice processing, which also does AGC and NS.
- Toggles live under 고급. The page shows `track.getSettings()` (sampleRate, channelCount, echoCancellation, autoGainControl, noiseSuppression,
  latency) because browsers silently ignore unsupported constraints.
- Device list: `enumerateDevices()` after permission (labels only appear after permission), plus a `devicechange` listener. If a Bluetooth headset is selected,
  show the warning "블루투스 헤드셋 마이크는 협대역(8–16 kHz 코덱)이라 정확도가 떨어집니다".

### 3.2 Sample rate and resampling: native-rate capture, **server-side soxr HQ**

- `new AudioContext({latencyHint: 'interactive'})` with **no** `sampleRate`. Forcing 16 kHz gave silence in some Chrome builds (the old demo's own note,
  `experiments/live/index.html:47`), and engines differ in whether a MediaStream source can feed a context at another rate.
- The old worklet decimated by picking samples with no low-pass (`experiments/live/index.html:51-53`). Measured: a 12 kHz tone at −9 dBFS aliases to 4 kHz
  **at −9 dBFS** (`scratch-audioui/soxr_test.py`).
- Server: `soxr.ResampleStream(sr, 16000, 1, dtype='float32', quality='HQ')`, created per session and bypassed when sr == 16000 (bit-exact path for parity).
  Measured at 48 k, 20 ms packets:
  - output is delay-compensated (an impulse lands exactly at 8000)
  - block buffering ≤ 37 ms
  - flat to 7 kHz, −0.43 dB at 7.5 kHz
  - alias residual −146 dBFS
  - 0.3 ms CPU per 2 s of audio

  LQ buffers only 20 ms but is −12.8 dB at 7 kHz, which would starve the top mel bins (128 mel up to 8 kHz). Do not use it. QQ aliases.
  **[critic] Re-run** (`scratch-audioui/soxr_test.py`, `scratch-critic/soxr_resp.py`):
  - The impulse lands at 8000 at both 48 k and 44.1 k.
  - 12 kHz alias residual: −236 dBFS (48 k), −153 dBFS (44.1 k).
  - HQ response is −0.43 / −0.40 dB at 7.5 kHz and −24 dB at 7.8 kHz; LQ is −12.8 / −13.3 dB at 7 kHz. Confirmed.
  - 44.1 k buffers up to 40 ms and 96 k up to 79 ms (§2.1).
- **[critic] Flush at end.** soxr holds 20–80 ms of input. On `stop` (and on file end), call `resample_chunk(np.zeros(0, np.float32), last=True)` and push its output
  through the framer *before* ending the open segment. Otherwise the last 20–80 ms of every session are silently dropped, which violates `dropped_s == 0`.
  Pad the trailing partial 512-sample frame with zeros for the VAD only; the engine gets the exact samples. After `last=True` the stream object is spent, so create a new one per session.

### 3.3 AudioWorklet and transport

- Serve `/static/worklet.js` same-origin (not a Blob URL). Node: `new AudioWorkletNode(ctx, 'capture', {numberOfInputs: 1, numberOfOutputs: 1,
  processorOptions: {packet: Math.round(ctx.sampleRate * 0.02)}})`. Graph: `source → node → GainNode(gain 0) → destination`. The zero-gain sink keeps the node
  pulled in every engine without playing the mic back. The old demo connected straight to destination (`index.html:58`).
- `process()` averages the channels to mono and fills a Float32Array of 20 ms (960 at 48 k; 128-frame render quanta), then posts it with transfer plus
  `{peak, rms}`. The main thread calls `ws.send(buffer)`. WebSocket does not exist in the worklet scope.
- **Wire format: Float32 LE mono at the declared `sample_rate`, raw (no header), any length.** Recommend 20 ms for mic and 1× file, and 320 ms for fast file.
  Float32 needs no int16 clipping or scaling and keeps values above 1.0 visible to the server's clip detector. 192 kB/s is trivial on localhost. `pcm_s16le` is also accepted
  (Python CLI client, old `client_wav.py` style).
- Why not 80 or 320 ms mic packets: a packet adds its length to latency. 320 ms would double the encoder chunk wait.
- Start sequence, all inside the Start click handler (autoplay policy, Safari): create the context, `await ctx.resume()`, `addModule`, getUserMedia, connect,
  open WS, `start`. `track.onmute` / `onended` trigger UI warnings.
- **Permission**: `http://localhost` and `http://127.0.0.1` are secure contexts, so getUserMedia and AudioWorklet work without TLS. A LAN IP over http does not
  (`navigator.mediaDevices` is undefined), so keep the old `--tls` self-signed option (`experiments/live/server.py:13,69-77`) as P2.
  macOS also needs System Settings › 개인정보 보호 및 보안 › 마이크 for the browser. If the first 1 s is exact zeros, show
  "마이크 입력이 0 입니다 — 시스템 설정 > 개인정보 보호 및 보안 > 마이크에서 브라우저를 허용하세요".
- Open `http://127.0.0.1:<port>/`, not `localhost`. The server binds IPv4 127.0.0.1, and `localhost` may resolve to ::1 first.

### 3.4 File replay (two paths, same protocol)

1. **실시간 재생 (1×, realistic)**: `decodeAudioData` into the live AudioContext, then an `AudioBufferSourceNode` → **the same capture worklet** as the mic (and
   optionally also → destination when "소리 듣기" is on). The audio clock paces it exactly (no setTimeout drift, no background-tab throttling), and it exercises
   the identical client path. `source:"file"`, `speed:1`.
2. **빠르게 (2× / 4× / 최대) and 평가 방식**: decode, then `OfflineAudioContext(1, ceil(dur·16000), 16000)` render (browser sinc resampler, stereo
   downmix), then send Float32 at **16000** (server bypasses soxr) in 320 ms packets:
   - 2× / 4×: absolute schedule `t0 + i/16000/speed`
   - 최대: unpaced, with flow control (§4.5)
   - "분할 없음(평가 방식)" uses this path with `segmentation:"none"`, `gain:"off"`, speed 최대
3. After the last packet the client sends `stop`. The server ends the open segment with reason `file_end`, applies the tail rule, flushes, then sends `session_end`.

Formats: whatever `decodeAudioData` supports (wav/mp3/m4a everywhere; flac/ogg depend on the browser). `tools/client_file.py` (soundfile) covers the rest and runs headless tests.

### 3.5 Level meter

- Client: per packet `peak` and `rms` (dBFS, raw mic before resample). Drawn with rAF. Instant attack, 20 dB/s release, 1 s peak hold. Scale −90…0 dBFS. Clip LED when peak ≥ 0.999.
- Overlays from the server `vad` event (10 Hz): noise floor N, `T_on`, VAD state color (silence grey / speech green / hangover amber), applied gain.
- Warnings:
  - If L_spk < −50 dBFS for 3 s while gain is off: "입력이 너무 작습니다 — 마이크 음량을 올리거나 가까이에서 말하세요".
  - If gain is on and hits +30 dB: the same text with "(최대 이득)".

---

## 4. WebSocket protocol (`/ws`, `proto: 1`)

### 4.1 Conventions

- Text frames are JSON objects with `type`. Binary frames are PCM only. Times `t*` are **seconds on the session audio clock** (16 kHz samples since
  `start` ÷ 16000, 3 decimals). Wall-derived values are `*_ms`. The server sets `compress=False` (no permessage-deflate on audio), `heartbeat=10`,
  and `max_msg_size=1 MiB`.
- Order: TCP keeps binary and text frames ordered, so `stop` always comes after the last audio. The server keeps that order end to end through one queue.

### 4.2 Client → server

| type | fields | semantics |
|---|---|---|
| `start` | `src_lang` ∈ {Korean, English, German}; `mode` ∈ {asr, translate}; `tgt_lang` (translate only; must be a trained pair: English→German/Korean, German→English, Korean→English); `delay` 4–10; `source` ∈ {mic, file}; `sample_rate` 8000–96000; `format` ∈ {f32le, s16le}; `takeover` bool; `options`: {`segmentation`: vad\|none, `pre_roll_ms`, `end_pause_ms`, `hard_max_s`, `gain`: auto\|off, `lead_in`: bool, `final_redecode`: bool, `debug_tokens`: bool, `file_name`, `file_duration_s`, `speed`} | Validates (else `error bad_config`) and opens a session. On a socket with an active session it is treated as `stop` then `start`. |
| *(binary)* | PCM per `start` | Appended to the session. Before `start` or after `stop` it is ignored, with one `warning binary_outside_session` carrying the byte count. |
| `update` | any of `src_lang, mode, tgt_lang, delay, options.{end_pause_ms, gain, final_redecode}` | Takes effect at the **next segment start**, since each segment has its own prefix. Acknowledged by `updated`. |
| `stop` | — | Graceful: ends the open segment (`stop`/`file_end`), drains every queued sample, flushes, then `session_end`. |
| `abort` (alias `reset`) | — | Explicit user cancel: discards queued audio (the only allowed discard; counted in `discarded_s`), cancels the open segment (`segment_end reason:"aborted"`), returns to idle in < 300 ms. |
| `ping` | `t` (client ms) | → `pong {t, server_ms}` (RTT shown under 고급) |

### 4.3 Server → client

| type | fields |
|---|---|
| `hello` | `proto`, `state` loading\|ready, `model` {name, dir, dtype, kv_capacity}, `langs`, `pairs`, `delays`, `defaults` {asr_delay 6, translate_delay **[critic]** per pair `{"English>German":10,"English>Korean":10,"German>English":10,"Korean>English":8}` (§2.1), options…}, `busy` (another session active), `latency_table` (§2.1), **[critic]** `laal_table` {pair: {δ: s, approx: bool}} |
| `status` | `state`, `message` (load progress, e.g. "가중치 1.2/2.5 GB", warmup) |
| `ready` | `load_ms`, `warmup_ms` |
| `started` | `session_id`, resolved config, `input_rate`, `resampler` |
| `updated` | resolved pending config |
| `segment_start` | `seg` (1-based), `t_start`, `reason` vad\|forced\|soft\|kv\|file\|none, `pre_roll_ms`, `lead_in_ms`, `gain_db`, `config` {src_lang, mode, tgt_lang, delay} |
| `delta` | `seg`, `k` (step index; **[critic]** k ≥ K means a flush round; zero-tail frames have k < K but are flagged `tail: true`), `src` (append text), `tr` (append text), `t_avail` (audio time this step needed), `proc_lag_ms`, optional `tokens` [[piece, stream]…] and `forced` when `debug_tokens` |
| `segment_end` | `seg`, `t_end`, `reason` pause\|soft_max\|max_len\|kv_guard\|stop\|file_end\|aborted, `src_text`, `tr_text` (full normalized; must equal the concatenated deltas), `audio_s`, `zero_tail_ms`, `n_tokens` {src, tr}, `forced_rounds`, `flush_rounds`, `finalize_ms`, `first_text_ms`, `enc_ms_sum`, `dec_ms_sum`, `voiced_s`, `hidden` (noise segment) |
| `segment_final` | `seg`, `src`, `tr`, `ms`, or `skipped: "lagging"` (§7) |
| `stats` (4 Hz) | `audio_rx_s`, `audio_seg_s`, `audio_skipped_s`, `audio_held_s`, `audio_proc_s`, **[critic]** `audio_done_s` (§2.2), `backlog_s`, `discarded_s` (abort only), `dropped_s` (**always 0**), `proc_lag_ms` {p50,p95,max}, `enc_ms` {p50,p95}, `dec_ms` {p50,p95}, `chunk_ms` {p50,p95}, `rtf_10s`, `rtf_session`, `load_10s`, `tok_per_s`, `kv_pct`, `mem_gb`, `loop_lag_ms` p99, `throughput_x` (file) |
| `vad` (10 Hz) | `t`, `level_db`, `floor_db`, `on_db`, `off_db`, `state` silence\|speech\|hangover, `gain_db`, `seg` or null, `limited` |
| `session_end` | `session_id`, `audio_s`, `wall_s`, `segments`, `rtf`, `backlog_max_s`, `proc_lag_p95_ms` |
| `warning` | `code` lagging\|quiet_input\|zero_input\|clipping\|binary_outside_session\|kv_guard\|unusual_rate, `message` (Korean) |
| `error` | `code` busy\|preempted\|bad_config\|overload\|model_not_ready\|internal, `message` (Korean), `fatal` |

### 4.4 Text assembly (server-side, revision-free)

- Tag routing is the same as `split_runs` (`vapasr/data/ast_eval.py:15-24`): `<SPK_A>`/`<SPK_B>` switch to `src`, `<TRANS>` to `tr`, tokens before any tag count as
  `src`, tags are not shown. State carries across rounds within a segment and resets per segment.
- Byte-level BPE: tokens map to raw bytes (special tokens give `b""`), with one UTF-8 byte buffer per stream. Only the **complete-character prefix** is emitted.
  A Hangul syllable can span two tokens. Incomplete bytes left at `segment_end` are decoded with `errors="replace"` and counted.
- Whitespace (so that concatenated deltas equal the final `re.sub(r"\s+"," ",s).strip()` used in `vapasr/hf/live.py:115` and `infer.py:63`):
  collapse runs to one space, drop leading space while the stream is empty, drop a leading space if the stream already ends in a space, and **hold back
  a trailing space** until non-space text follows.
- `src` is the normalized form (lowercase, no punctuation, numbers spelled out; `vapasr/data/ast_dataset.py:7,124`). `tr` is the written form. The UI explains this.
- `delta` is sent only when `src` or `tr` is non-empty, or in debug mode.
- **[critic] ASR mode.** `<TRANS>` is not blocked (`decode-contract.md` §a.2), so an ASR-only prefix can in principle open a translation run. Eval keeps only run 0
  as the hypothesis (`experiments/eval_single_turn_asr.py:460-461`). In `mode:"asr"` the server therefore drops `tr` text from `delta`/`segment_end` and counts
  the ids in `segment_end.n_tokens.tr_dropped`. They are visible only in the debug token log.
- **[critic] Byte mapping edge cases**, verified with `tokenizers` 0.22.2 on the checkpoint-layout `tokenizer.json` (`scratch-critic/tok_check.py`):
  - ids 151727–151935 (unused embedding rows, *not* blocked, `decode-contract.md` §a.1) have no token and decode to `""`, so `token_bytes` returns `b""` for them.
  - `<asr_text>` is a non-special added token: `decode(skip_special_tokens=True)` returns the literal `"<asr_text>"`. It is blocked so it never appears, but `token_bytes` must map every id in `sp_ids`/added tokens to `b""` rather than rely on the special flag.
  - Hangul splits are real: `안녕하세요 반갑습니다` → `안|녕|하세요| 반|갑|습니다`, i.e. multi-byte tokens. Keep the holdback test (E6) with byte-split tokens.
- `clean_up_tokenization_spaces` is `false` in the Qwen3-ASR tokenizer config, so byte concatenation equals `tok.decode` (no space cleanup) **[critic, verified]**.

### 4.5 Flow control and backpressure

- The server **always reads** the socket. Resampling, VAD and gain run on the event loop (µs per packet) and audio goes into an unbounded worker queue.
  Nothing is ever dropped (`dropped_s` stays 0). If `backlog_s` > 2 s, send `warning lagging` once per 10 s. If `backlog_s` > 600 s (≈ 38 MB at 16 k float32), send `error overload fatal`
  and stop the session. This is explicit failure, not silent loss.
- Fast file sender: keep `sent_audio_s − stats.audio_proc_s ≤ 10 s` **and** `ws.bufferedAmount ≤ 512 KB`, polling every 50 ms. At 2× or 4× the schedule
  still applies and flow control only pauses it.
  **[critic] Bug: this deadlocks.** `audio_proc_s` counts only audio the engine consumed. Skipped (non-speech) audio never enters it, so a file with a
  silent stretch > 10 s pauses the sender forever: the server has nothing left to process and `audio_proc_s` never rises. Use
  **`sent_audio_s − stats.audio_done_s ≤ 10 s`** (§2.2; done = skipped + engine-consumed real samples). The pending amount is then bounded by backlog + hold +
  VAD look-back and always drains. E1 must include a file with ≥ 15 s of silence.
  - Converting `sent_audio_s` to the 16 kHz session clock: the client counts samples at the declared `sample_rate`, divided by `sample_rate`.
  - The ≤ 80 ms the resampler holds (§3.2) is inside the 10 s margin.
- `abort` is the only path that discards queued audio.

### 4.6 Example (translate ko→en, δ10)

```
C {"type":"start","src_lang":"Korean","mode":"translate","tgt_lang":"English","delay":10,"source":"mic","sample_rate":48000,"format":"f32le","options":{}}
S {"type":"started","session_id":"20261016-101503-3f2a","input_rate":48000,"resampler":"soxr-HQ", ...}
C <binary 3840 B> × n
S {"type":"vad","t":1.216,"level_db":-31.2,"floor_db":-62.0,"on_db":-52.0,"off_db":-56.0,"state":"speech","gain_db":0,"seg":null}
S {"type":"segment_start","seg":1,"t_start":0.736,"reason":"vad","pre_roll_ms":480,"lead_in_ms":0,"gain_db":0,"config":{"src_lang":"Korean","mode":"translate","tgt_lang":"English","delay":10}}
S {"type":"delta","seg":1,"k":23,"src":"안녕하세요","tr":"","t_avail":2.592,"proc_lag_ms":41}
S {"type":"delta","seg":1,"k":31,"src":"","tr":"Hello,","t_avail":3.232,"proc_lag_ms":38}
S {"type":"segment_end","seg":1,"t_end":4.928,"reason":"pause","src_text":"안녕하세요 반갑습니다","tr_text":"Hello, nice to meet you.","zero_tail_ms":200,"finalize_ms":46, ...}
C {"type":"stop"}
S {"type":"session_end", ...}
```

---

## 5. Server architecture

### 5.1 Threads

```
asyncio thread (aiohttp)                         worker thread (owns ALL MLX state; QoS USER_INTERACTIVE)
 ├ HTTP: / , /static/*, /api/health              │ load weights → warmup → ready
 ├ WS reader → resampler(soxr) → framer(512)     │ loop: cmd = in_q.get()
 │    → VAD/segmenter → gain → in_q.put(cmd) ───►│   SEG_START(cfg) → engine.new_segment()
 │    (cmds: SEG_START, AUDIO, SEG_END,           │   AUDIO(pcm)     → seg.feed() → rounds
 │     STOP, ABORT, UPDATE)                       │   SEG_END(tail)  → seg.finish(zero_tail_samples)   [critic: was seg.feed(zeros); seg.finish()]
 ├ out_q (asyncio.Queue) ◄──── call_soon_threadsafe ── events (rounds, timings)
 └ WS writer: text assembly (§4.4), stats 4 Hz, vad 10 Hz
```

- **Never call MLX on the event loop.** Measured on mlx 0.32.2: evaluating an array built on another thread raises
  `RuntimeError: There is no Stream(gpu, 0) in current thread.` So load, warmup, inference and final re-decode all run on the worker thread, and only
  numpy arrays and Python ints cross the boundary.
- `mx.eval` releases the GIL. Measured: event-loop lag p99 was 1.34 ms while a worker ran heavy matmuls (`scratch-audioui/mlx_thread.py`). One worker thread is
  therefore enough and no subprocess is needed. Acceptance checks `loop_lag_ms` p99 < 10 ms during real decoding.
- The worker thread sets `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE=0x21, 0)` via ctypes (verified 0x15 → 0x21) so that the
  Python token loop is not parked on efficiency cores.
- Abort and stop checks happen between rounds, never inside one.
- **One active session.** A second `start` gets `error busy` unless it sets `takeover:true`; the old socket then gets `error preempted` and its session is aborted.
  A closed socket aborts its session (there is nobody to deliver to). The text log is still written.

### 5.2 Worker scheduling

- Feed the engine as soon as audio is released. Chunk boundaries fall on 512-sample frames (§1.3), so a chunk starts the moment its last frame arrives.
- At segment start the pre-roll burst (~0.6 s) is processed immediately. This is a short catch-up.
- `SEG_END`: append `zero_tail` samples, call `finish()` (final mel padding, remaining frames up to **K = round(n_samples / 1280)**, the same K rule as eval,
  `experiments/eval_single_turn_asr.py:431`), then flush rounds with `<EMPTY_AUDIO>` until an empty round or 8 rounds (`max_flush`, `eval_single_turn_asr.py:578`).
  **[critic] Corrected procedure** (`SEG_END` → `seg.finish(zero_tail_samples)`):
  1. Append the zeros *inside the engine*, so no round runs before the final length is known.
  2. Compute **`K = int(round(n_total / 16000 * frame_hz))`** with `frame_hz = 12.5` from config. This is exactly `eval_single_turn_asr.py:431` and `modeling_vapasr.py:225`.
     `round(n/1280)` is *not* the same: it differs at 36 lengths below 45 s, e.g. n = 69 760 gives 54 vs eval's 55 (`scratch-critic/kdiff.py`, binary rounding of n/16000·12.5 at exact halves).
  3. Run the remaining complete chunks, then the final short step with `keep_all_outputs` (`encoder-spec.md` §6).
  4. **Decode only frames k < K that were not yet decoded.** Frames ≥ K are dropped, and frames missing below K are zero-padded before the adapter.
  5. Run the flush rounds (`config.max_flush_rounds`, 8; stop after the first empty round).

  Why step 1 matters: chunk c completes at 4096 + 5120c samples and yields frames up to 4c+3. With n_total = 5120c + 4096 + r and r < 384, K = 4c+3, so a
  `feed()` that eagerly decoded the tail-completed chunk would run one **extra audio round at k = K**. Eval never runs that round, so parity breaks. This happens for
  **7.6 % of all lengths** (`kdiff.py`: 37 252 of 491 904 lengths up to 31 s). Frames completed *before* the end decision are always < K, given the ≥ 200 ms zero-tail floor (§1.4).
- Final re-decode jobs (§7) run only when no streaming work is pending, one token step at a time, checking `in_q` between steps.
- Optional speed-up for the engine: precompute the prefix KV per (src, tgt, δ) and copy it at segment start. This saves one prefill per segment.

### 5.3 Contract with the engine (the app needs exactly this)

```python
class Engine:                                    # constructed, used and destroyed on the worker thread only
    def __init__(self, model_dir: str, kv_capacity: int = 2048): ...
    def warmup(self) -> dict: ...                # compile/run every shape: first chunk, steady chunk, q=1/q=2 decoder, 7 prefixes
    def new_segment(self, src_lang: str, tgt_lang: str | None, delay: int) -> "SegmentDecoder": ...
    def token_bytes(self, tid: int) -> bytes: ...   # b"" for special/added tokens
    spk_ids: tuple[int, int]; trans_id: int
class SegmentDecoder:
    def feed(self, pcm16k: "np.ndarray[float32]") -> list["Round"]: ...   # runs every encoder chunk now complete (4096+5120c samples) and its 4 rounds
    def finish(self, zero_tail_samples: int) -> list["Round"]: ...   # [critic] engine appends the zeros itself; K = int(round(n_total/16000*12.5));
                                                # remaining chunks + final short step; decode only k < K; then ≤ config.max_flush_rounds flush rounds
    # [critic] caps: per round config.runaway_cap (32 for s4); total 6*K_cap+64 while streaming, exact 6*K+64 inside finish();
    #          hard KV clamp at kv_capacity (Round.kv_full). token_bytes(): b"" for every added/special id and for ids >= len(tokenizer) (151727..151935).
    positions: int; kv_capacity: int; samples_fed: int
    def features(self) -> "mx.array": ...       # (K,1024) causal features, for the optional final re-decode
    def offline_final(self) -> list[int]: ...   # optional (§7)
@dataclass
class Round: k: int; ids: list[int]; forced: bool; is_tail: bool; need_samples: int; enc_ms: float; dec_ms: float; kv_full: bool = False   # [critic]
```

Parity requirement (engine spec owns the proof): `segmentation:"none"` + `gain:"off"` + 16 kHz input reproduces the token ids of the reference decode
(`stream_decode` / `realtime_decode`) on the same wav + 1.0 s zeros.
**[critic]** The parity set must include lengths where `(n_total − 4096) mod 5120 < 384`, the K-boundary case of §5.2. It must also include one length from the
36 where `round(n/1280)` ≠ eval K (e.g. n_total = 69 760 or 129 920). Otherwise the two off-by-one bugs fixed above go untested.

### 5.4 Startup

1. Parse args: `--model DIR`, `--port 8796`, `--host 127.0.0.1`, `--no-browser`, `--kv-capacity 2048`, `--engine mlx|fake`, `--tls`, `--save-audio`, `--debug`.
   `--engine fake` emits deterministic tokens with a configurable per-round delay. It lets protocol, UI, backpressure and segmentation be tested on the dev box,
   where s4 is not present.
2. Environment checks, failing early with Korean messages:
   - arm64 and not Rosetta (`sysctl -n sysctl.proc_translated` is 0)
   - macOS ≥ 14. **[critic]** This is unverified for the pinned build: the dev box (macOS 26.3) received `mlx-0.32.2 cp311-cp311-macosx_26_0_arm64` and
     `mlx_metal-0.32.2 py3-none-macosx_26_0_arm64` (`*.dist-info/WHEEL`). MLX publishes per-macOS wheels, so whether 0.32.2 has a wheel for the MacBook Pro's macOS is
     checked by `setup.sh` (the `--only-binary` install fails fast). The real minimum is whatever pip resolves there; print `sw_vers` in the error.
   - `mx.metal.is_available()`
   - `hw.memsize` (warn < 24 GB)
   - Python ≥ 3.10
3. Model discovery, first valid wins:
   - `--model` or `$VAPKT_APP_MODEL`
   - converted dirs:
     - `/Volumes/Samsung_T5/vapkt-models/semcommit-q06-s4-ast-mlx`
     - `~/Desktop/VAPKT-models/semcommit-q06-s4-ast-mlx.nosync`
     - `~/VAPKT-models/semcommit-q06-s4-ast-mlx`
   - raw dirs (these mean "convert first"; `run.sh` does it):
     - `/Volumes/Samsung_T5/vapkt-models/semcommit-q06-s4-ast`
     - `~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync`
     - `~/Desktop/VAPKT-models/semcommit-q06-s4-ast`

   Before opening **any** file, `os.stat(p).st_flags & 0x40000000` (SF_DATALESS) must be 0. If it is set, refuse with
   "iCloud 에서 내려받지 않은 파일입니다(dataless): … — Finder 에서 다운로드하거나 `brctl download <경로>` 후 다시 실행하세요. 체크포인트는 *.nosync 폴더에 두세요".
   Reading a dataless file can hang forever (user memory: iCloud eviction). Also validate the converted dir's manifest (source fingerprint, format version).
   **[critic] Additions:**
   - Check `st_flags` on each **directory** before `listdir`/`glob`, not only on files. Enumerating a dataless directory also materializes it.
     Use `os.stat` (it does not read file data) on the exact expected names (`config.json`, `model.safetensors`, `tokenizer.json`, …) rather than globbing.
   - Accept `<dir>/config.json` **or** `<dir>/final/config.json`. Training writes the HF checkpoint to `<run>/final/` (`experiments/semcommit_train.py:433`;
     `slurm/submit-q06-s4-ast.sh:20` uses `.../semcommit-q06-s3-ast/final`), so a copied run directory has the weights one level down.
   - Require `tokenizer.json` in the checkpoint. The runtime has `tokenizers` but not `transformers`, and the public Qwen3-ASR repo ships only `vocab.json` + `merges.txt`
     (`scratchpad/hf-qwen3-asr/`). `tok.save_pretrained(fin)` (`semcommit_train.py:433`) writes `tokenizer.json` with the fast tokenizer, as the checkpoint-layout fixture
     `scratch-decode/tiny-s4-like/` shows. `tokenizers.Tokenizer.from_file` on it reproduces the `decode-contract.md` §b.2 prefix ids exactly (`scratch-critic/tok_check.py`).
     If it is missing, `convert.py` stops with a one-line fix to run once in the dev env:
     `PYTHONPATH=<repo> python -c "from vapasr.hf import load_tokenizer as L; L('<ck>').save_pretrained('<out>')"`. `load_tokenizer` re-adds the special tokens
     in the mandatory order and asserts the ids against `config.sp_ids` (`vapasr/hf/__init__.py:16-41`).
   - At load, assert the 7 prefixes against the `decode-contract.md` §b.2 table and `config.sp_ids`.
4. Bind the port: try 8796…8805, skipping 8765 and 8790. Print the URL and `webbrowser.open("http://127.0.0.1:<port>/")` once listening, **before** model load.
   The page shows "모델 로딩 중…" from `hello`/`status`.
5. Start the worker: load, warm up (one synthetic 2 s segment of −50 dBFS noise + 1 s zeros at δ6, and every prefix), then broadcast `ready` with `load_ms` and `warmup_ms`.
   WS `start` before ready gets `error model_not_ready`.

### 5.5 Shutdown and logging

- SIGINT/SIGTERM: send `error server_shutdown` to the session, put `ABORT, SHUTDOWN` on the queue, join the worker with a 5 s timeout, release MLX memory, exit 0.
- Logs: `~/.vapkt-app/logs/server-<ts>.log`.
- Session transcript JSONL (config, segments, final texts, timings; no audio): `~/.vapkt-app/sessions/<session_id>.jsonl`, on by default.
- `--save-audio`, or the UI toggle "세션 오디오 저장", writes the post-resample 16 kHz input wav plus segment boundaries for replay. Off by default.
- HTTP: `Cache-Control: no-store` on the page and static files. CSP `default-src 'self'; connect-src 'self' ws: wss:`. No external requests, so the app works offline.
  **[critic] CSP gotcha.** With only `default-src 'self'`, inline `<script>`, inline event handlers (`onclick=`), `<style>` blocks and `style="…"` attributes in
  `index.html` are **blocked silently**; only an error is logged to the console. All JS must live in `app.js`/`worklet.js`, all CSS in `style.css`, and handlers must be bound
  with `addEventListener`. Dynamic `el.style.x = …` from JS (CSSOM) is allowed. `connect-src` should be `'self' ws://127.0.0.1:* ws://localhost:*`
  rather than the scheme-wide `ws:`. Add a test (H3) that loads the page with the header and checks for zero CSP violations.

---

## 6. UI (Korean, `web/index.html` + `app.js` + `worklet.js` + `style.css`, system fonts, light/dark via `prefers-color-scheme`, works from 360 px)

```
┌ VAPKT 실시간 전사·번역   [q06-s4-ast · MLX]   ● 준비됨 ─────────────────────────────────────────┐
│ 원문 [한국어▾]  모드 (전사 | 전사+번역)  번역 [영어▾]  지연 δ [6 ▾] 알고리즘 지연 ≈ 0.58 s       │
│ 마이크 [MacBook Pro 마이크▾]   [● 마이크 시작]   파일 [선택] [실시간 재생] [빠르게 ▾] □ 소리 듣기  │
│ 입력 ▕████████▏──|─────  −31 dBFS · 바닥 −62 · 이득 0 dB · 말하는 중          ▸ 고급            │
├──────────────┬─────────────────────────────────┬─────────────────────────────────┤
│ #1 00:00.7–  │ 원문 전사                        │ 번역 (영어)                      │
│   00:04.9    │ 안녕하세요 반갑습니다             │ Hello, nice to meet you.         │
│ 멈춤 · 46 ms │                                  │ final: … (optional, muted)       │
│ #2 00:06.1–… │ 오늘은 날씨가▌                   │ Today the weather▌               │  ← live row
├──────────────┴─────────────────────────────────┴─────────────────────────────────┤
│ 처리 지연 41/80 ms · 밀림 0.00 s · RTF 0.21 · 인코더 9 ms/청크 · 디코더 11 ms/라운드 · KV 18 %  │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

- **Controls.**
  - 원문 언어: 한국어/영어/독일어, mapping to Korean/English/German.
  - 모드: 전사 / 전사 + 번역.
  - 번역 언어 lists only trained pairs (한국어→영어; 영어→독일어·한국어; 독일어→영어) and is disabled in 전사 mode.
  - δ select 4–10, each option labelled `δ6 · 약 0.58 s`, with the range in a tooltip. In translate mode the hint adds the eval LAAL for the pair. Defaults are δ6 (전사) and δ10 (번역). **[critic]** 번역 기본값은 짝마다 다르다: 한→영 δ8, 나머지 δ10 (§2.1). In 한→영 mode, show the note "한→영 번역 모드에서는 한국어 전사가 앞부분에서 끊기는 경우가 있습니다(s4 알려진 약점)".
  - Changing language, mode or δ during a session shows the chip "다음 세그먼트부터 적용" (`update`).
  - Buttons: "● 마이크 시작" / "■ 정지" (Space). File mode adds "■ 중단" (Esc, `abort`).
- **고급 (collapsible).**
  - 브라우저 에코 제거 / 잡음 억제 / 자동 이득 (off)
  - 서버 음량 자동 맞춤 (on)
  - 분할: 자동(VAD) | 없음(평가 방식, 파일만)
  - 끝 멈춤 300–2000 ms (800)
  - 최대 길이 10–30 s (25)
  - 세그먼트 끝 final 재디코드 (off)
  - 토큰 로그 표시 (off)
  - 세션 오디오 저장 (off)
  - track settings readout, RTT
- **Panes.** A CSS grid with one row per segment and columns `meta | 원문 전사 | 번역`, so both texts of a segment share the row height (synchronized).
  - In 전사 mode the 번역 column is hidden.
  - Below 720 px each row stacks meta, then src, then tr (tinted).
  - meta shows `#n`, `mm:ss.s–mm:ss.s`, end reason (멈춤/긴 발화/최대 길이/정지/파일 끝), `finalize_ms`, gain and δ. The tooltip adds `first_text_ms`.
  - The open segment is the bottom "live" row with a caret. Newly appended text gets a `fresh` highlight fading over 1.2 s.
  - Rows are append-only: text is never revised. `segment_final`, when enabled, adds a muted line labelled `final`.
  - Auto-scroll is on unless the user scrolls up; then a "맨 아래로" button appears.
  - Hidden noise segments are not rendered.
  - A note under the header reads "전사는 정규형(소문자·구두점 없음, 숫자는 말한 대로)으로 나옵니다".
- **Status bar.** As in §2.3, with colors. During model load it shows "모델 로딩 중… n s". During warmup it shows "준비 중(커널 컴파일)".
- **Banners.** zero input, quiet input, lagging, busy ("다른 탭에서 사용 중입니다 — [가져오기]"), disconnected (auto-reconnect with backoff; any running session ends).
- **Export.** "텍스트 복사" copies `[mm:ss.s] 원문\n→ 번역` per segment. "JSON 저장" saves all events of the session.
- **Debug panel (토큰 로그).** Per round: `k`, pieces with stream color, forced flag and lag, the same information as the old demo's chunk log (`experiments/live/index.html:39`).
- Plain JS modules, no framework, no build step, no CDN. `fetch`/WS go only to the same origin.

---

## 7. Optional per-segment final (offline) re-decode

- What: after `segment_end`, decode the segment again in final mode (Qwen3-ASR original format, `ast_offline_sequence`, `vapasr/data/ast_dataset.py:178-191`) from the
  **same causal encoder features**, as eval does (`experiments/eval_single_turn_asr.py:480-503`, `offline_decode.py`).
- Worth: s4 final is much better than streaming. BLEU δ10 → final: en→de 24.3 → 30.2, en→ko 29.7 → 34.2, de→en 28.8 → 33.5, ko→en 14.9 → 19.9
  (`output-q06-s4-ast-eval-20261013.md`). It shows the user the quality ceiling of the same model.
- Cost: one prefill of about K + 40 positions **[critic: K + 20…26 — 8 pre + 1 + K + 1 + 5 mid + 5–11 head, `decode-contract.md` §f; `max_new_tokens = K + 64` for ast, `eval_single_turn_asr.py:501`]** plus autoregressive tokens for transcript + translation (about 6–9 tokens/s of speech). That is about 25–40 % on top of the
  segment's streaming compute (estimate; measure on the MacBook Pro). It runs only when the worker is otherwise idle (§5.2) and is skipped (`skipped:"lagging"`) if
  `backlog_s` > 0.5 s.
- Decision: **implement as P1, default off**, enabled by a UI toggle. Turn it on by default only if the MacBook Pro acceptance F1 still passes with it on.

---

## 8. Packaging (MacBook Pro)

```
App/
  README.md          Korean quickstart (setup → run → browser)
  requirements.txt   exact pins (below)
  setup.sh           venv in ~/.vapkt-app/venv
  run.sh             model discovery → convert if needed → server under caffeinate
  convert.py         HF ckpt → MLX dir (format defined by the engine spec; placement rules here)
  server.py          aiohttp app, session, segmenter glue, worker thread
  audio_pipeline.py  resampler, framer, VAD/segmenter, gain (pure numpy, unit-tested)
  text_stream.py     tag routing + UTF-8 incremental assembly (pure, unit-tested)
  engine/            MLX engine (engine spec) + fake engine
  web/               index.html app.js worklet.js style.css
  tools/client_file.py   headless WS client: wav → server at 1×/max, prints deltas/latency, writes JSON
  tests/             test_segmenter.py test_text_stream.py test_protocol.py (fake engine) — no model needed
```

**requirements.txt.** Exact versions verified pip-installed in `vapasr-local` on 2026-10-16, Python 3.11.16 arm64:
`mlx==0.32.2`, `mlx-metal==0.32.2`, `numpy==2.4.6`, `safetensors==0.8.0`, `tokenizers==0.22.2`, `aiohttp==3.14.3`, `soxr==1.1.0`, `soundfile==0.14.0`.
Also commit a `requirements.lock` made by installing these into a fresh venv on the dev box and running `pip freeze` (this pins aiohttp's transitive deps).
The runtime must **not** import torch, transformers, nemo or mlx_lm (mlx_lm pulls in transformers). `safetensors` is used only by convert.py.
`soundfile` is used only by tools and tests.
**[critic]** `setup.sh` runs `pytest`, but pytest is not in the list. Add `requirements-dev.txt` with `pytest==9.1.1` (the version in `vapasr-local`), installed by `setup.sh`.
Transitive pins seen here for the lock: `multidict 6.8.0`, `yarl 1.24.5`, `frozenlist 1.8.0`, `aiosignal 1.4.0`, `attrs 26.1.0`, `propcache 0.5.2`,
`aiohappyeyeballs 2.7.1`, `cffi 2.1.1`, `pycparser 3.0`. The `mlx`/`mlx-metal` wheels are per-macOS (§5.4), so the lock pins versions, not wheel files or hashes.

**setup.sh.**
- `set -euo pipefail`.
- Pick the Python from `$PYTHON`, then python3.12, python3.11, python3.13, then `python3`. Require ≥ 3.10 and `platform.machine()=="arm64"`; the CLT `/usr/bin/python3` 3.9 is rejected with a hint (`brew install python@3.12`).
  **[critic] It must iterate, not stop at the first name on PATH.** On this Mac `command -v python3.12` resolves to **`/usr/local/bin/python3.12`, an x86_64 (Intel Homebrew/Rosetta)
  build**. `/opt/homebrew/bin/python3.12` (arm64) comes later, and `/Library/Frameworks/…/3.7/bin/python3` (x86_64) is second for `python3`. The user's MacBook may have the same layout.
  Candidate order:
  1. `$PYTHON`
  2. `/opt/homebrew/bin/python3.{12,11,13}`
  3. `~/anaconda3/bin/python3`, `~/miniconda3/bin/python3` (arm64 3.11.4 here)
  4. every `python3.{12,11,13}` and `python3` from `which -a`

  Select the **first** whose `platform.machine()=="arm64"`, `sys.version_info ≥ (3,10)`, `venv` and `ensurepip` import, and `sysctl.proc_translated` is 0. Print every rejected candidate with the reason.
  A venv made from conda's Python works (`venv` is stdlib). Running inside an active conda env is fine because the venv isolates packages.
- `python -m venv ~/.vapkt-app/venv` (`--recreate` to rebuild).
- If `~/.vapkt-ca-bundle.pem` exists, export `PIP_CERT`, `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE` and `CURL_CA_BUNDLE` to it.
- `pip install --only-binary :all: -r requirements.lock`.
- Verify `mx.metal.is_available()` and print the versions.
- Run `pytest App/tests -q` with the fake engine.
- Idempotent. Never touches `~/Desktop/VAPKT-models`.

**convert.py placement.**
- Source: `--src`, or discovery (§5.4).
- Default output:
  - `<parent>/<name without .nosync>-mlx.nosync` when the parent is under `~/Desktop` or `~/Documents` (iCloud-synced; `.nosync` keeps the files local and never evicted)
  - `<parent>/<name>-mlx` on `/Volumes/*` (T5, exFAT)
- Refuse an iCloud-synced output path without `.nosync` unless `--force`. Check dataless flags on every input before reading. Check free space (fp16 output ≈ 2.5 GB).
- Read tensors one at a time (safetensors `safe_open(framework="numpy")`) to keep peak RAM ≈ 3 GB. Write to `<out>.partial-<pid>`, then rename atomically.
- Copy `config.json` and the tokenizer files into the output. **[critic]** `tokenizer.json` is mandatory (§5.4).
- Write `vapkt-app-manifest.json`: format version, source dir, sha256 of `config.json`, per-file size and mtime of the source weights, dtype per component, tensor count and app version.
  The server checks it, and `run.sh` reconverts when the source fingerprint changed.

**run.sh.**
- Require the venv (else print "먼저 bash App/setup.sh").
- `export PYTHONPYCACHEPREFIX=~/.vapkt-app/pycache PYTHONUNBUFFERED=1`, which keeps `__pycache__` out of the iCloud Desktop repo.
- Warn if any `App/` file is dataless (the repo itself lives on the iCloud Desktop).
  **[critic]** It must **refuse**, not warn, and list the files. Importing a dataless `.py` or serving a dataless `web/` file blocks forever when iCloud is unreachable
  (user memory: Desktop eviction, unrecoverable on the corporate network). Check with `stat -f %Xf` (or Python `st_flags & 0x40000000`) and suggest `brctl download App/`.
- **[critic]** `cd "$(dirname "$0")/.."` first (or use absolute paths), so `App/server.py` resolves no matter where `run.sh` is invoked from.
- If discovery finds only a raw checkpoint, run `convert.py` once (logged).
- `exec /usr/bin/caffeinate -i ~/.vapkt-app/venv/bin/python App/server.py "$@"`. `caffeinate` prevents idle sleep during long sessions.

---

## 9. Acceptance checklist

**A. Packaging and startup**
- [ ] A1 `bash App/setup.sh` on a clean MacBook Pro (with and without the corporate CA) creates `~/.vapkt-app/venv`. `pip freeze` equals `requirements.lock`, and torch, transformers, nemo and mlx_lm are absent.
- [ ] A2 `bash App/run.sh` with only `~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync` converts once into `semcommit-q06-s4-ast-mlx.nosync` (manifest present) and skips conversion on the second run.
- [ ] A3 The dataless guard: a unit test with a mocked `st_flags` shows the refusal message, and no file under a dataless path is opened.
- [ ] A4 The server binds 127.0.0.1:8796 (or the next free port; never 8765/8790) and the browser opens before model load. The page shows loading → ready with `load_ms`/`warmup_ms`.
- [ ] A5 The page makes no network request outside its own origin (DevTools), and `grep -R "http" App/web` finds no external URL.
- [ ] A6 No MLX call on the event loop: an assertion in engine entry points that `threading.get_ident() == worker_id`. `loop_lag_ms` p99 < 10 ms during decoding.
- [ ] **[critic] A7** `setup.sh` on a Mac whose first `python3.12` on PATH is x86_64 (as on the dev box: `/usr/local/bin/python3.12`) skips it with a printed reason and builds an arm64 venv. `pytest` is installed from `requirements-dev.txt`.
- [ ] **[critic] A8** Discovery accepts a copied run dir with weights in `final/`. A checkpoint without `tokenizer.json` stops with the one-line fix. Dataless *directories* are refused before enumeration, and `run.sh` refuses (not warns) on dataless `App/` files.

**B. Capture**
- [ ] B1 Chrome and Safari prompt for the mic on `http://127.0.0.1`. `getSettings()` shows EC/NS/AGC off by default, and the toggles change them (reflected in the readout).
- [ ] B2 OS mic permission removed → the zero-input banner appears within 1.5 s.
- [ ] B3 Resampler unit test (48 k and 44.1 k → 16 k, streaming in 20 ms packets): impulse aligned exactly, alias residual of a 12 kHz tone < −100 dBFS, ≥ −0.5 dB at 7.5 kHz.
      16 k input is bit-exact passthrough.
- [ ] B4 Level meter and server VAD state follow speech. The clip LED lights at full scale.
- [ ] **[critic] B5** Resampler end flush: a 48 kHz stream of N samples yields exactly round(N/3) 16 kHz samples after `stop` (`last=True`), i.e. no lost tail.

**C. Segmentation and gain** (tests use synthetic signals plus the §1.7 files; no model needed)
- [ ] C1 Onset gives a 480 ms pre-roll. An end fires exactly 800 ms after the last voiced frame. Gaps < 800 ms merge.
- [ ] C2 Soft max (≥ 18 s) ends at a ≥ 256 ms speech-relative pause: the VoxPopuli 28.7 s file cuts near 21.6 s.
      Continuous speech forces a cut in **[critic: [24.36, 25] s, i.e. the unreleased 640 ms; was "[23, 25] s"]** at the quietest unreleased frame (NIKL file: 24.76 s) and the next segment has `forced_start`.
- [ ] C3 Sample accounting: `received = in_segments + skipped` at session end, no overlap between segments, `dropped_s == 0`.
- [ ] C4 Determinism: the same file at 1×, 4× and max gives identical segment boundaries and (with the real engine) identical tokens.
- [ ] C5 Tail rule: `zero_tail_ms` is 200 / 744 / 1000 for pause / soft / forced ends. `segmentation:"none"` appends exactly 16000 zeros.
      **[critic]** With `end_pause_ms = 2000` the tail is still 200 ms (floor). `tail_s` below 0.08·(δ+2) is rejected with `bad_config`. The zero tail reaches the engine only via `finish(zero_tail_samples)`.
- [ ] C6 Lead-in A/B: replay files cut by soft/forced rules with lead-in on and off, and compare first-word recall. Keep the default only if it is not worse.
- [ ] C7 Gain: a Korean or English file attenuated to −55 dBFS produces text with gain on (gain ≈ +29 dB reported). With gain off the failure is reproduced or logged, as a diagnostic.
- [ ] **[critic] C8** Causal start: a file whose speech starts at sample 0 gives the same boundaries live (frames 0–15 buffered until the floor is known) as the offline prototype. Lead-in is off by default and is never applied with < 3 s of history.
- [ ] **[critic] C9** `hard_max_s = 10` derives `soft_max_s = 3` (clamped up to 0.6·10 = 6) and `hold_from_s = 8`. The forced cut still has a 640 ms search window.
      Corpus-level input (−21…−32 dBFS) gets 0 dB.

**D. Parity**
- [ ] D1 `tools/client_file.py --segmentation none --gain off` on 16 kHz wavs (testaudio de/en/ko, LS and Kspon samples): token ids equal the engine's reference decode of
      wav + 1.0 s zeros for δ4, δ6 and δ10, in ASR mode and in every trained translate pair.
      **[critic]** Include at least two lengths with `(n_total − 4096) mod 5120 < 384` (K-boundary, §5.2) and one with `round(n/1280) ≠ int(round(n/16000*12.5))`
      (e.g. crop a wav so that n_total = 69 760). Also include a 1× mic-path replay of the same wav at 48 kHz: its tokens may legitimately differ because of resampling. Report agreement, do not require identity.

**E. Protocol and backpressure**
- [ ] E1 File at max speed: client in-flight stays ≤ 10 s, server RSS stays flat, `dropped_s = 0`, and final texts equal the concatenated deltas.
      **[critic]** Must include a file with a ≥ 15 s silent stretch: the sender must not stall (flow control on `audio_done_s`, §4.5). After `stop`, the resampler tail is flushed, so `audio_rx_s` equals the file duration ± 1 sample at 48 k input.
- [ ] E2 Fake engine with an injected 120 ms/round delay (RTF > 1): `backlog_s` grows, the lagging warning shows, nothing is dropped, and lag recovers once the delay is removed.
- [ ] E3 `stop` ends with `segment_end(stop)`, then `session_end`. `abort` returns to idle in < 300 ms with `discarded_s` reported.
- [ ] E4 A second tab gets `busy`. Takeover preempts the first tab, which shows "preempted".
- [ ] E5 `update` (δ, language, mode) mid-session applies from the next `segment_start.config` and never to the open segment.
- [ ] E6 Korean text never shows broken characters mid-stream (UTF-8 holdback test with tokens that split syllables).
- [ ] **[critic] E7** In `mode:"asr"` an injected `<TRANS>` run (fake engine) never reaches `tr`/`src_text` and is counted in `tr_dropped`. Ids 151727–151935 and `<SEM_END>` produce no text.

**F. Performance on the MacBook Pro (real s4 weights)**
- [ ] F1 5-minute mic sessions, Korean ASR δ6 and en→ko translate δ10: `chunk_ms` p95 < 160 ms (RTF < 0.5), `proc_lag_ms` p95 < 200 ms, `backlog_s` max < 0.3 s, no `kv_guard`.
- [ ] F2 `finalize_ms` p95 < 150 ms. `first_text_ms` ≈ algorithmic latency + < 150 ms.
      **[critic]** As written, the second part is ill-defined: `first_text_ms` starts at the onset frame, but the first token can only follow the *first word's end*, so it
      includes the first word's duration. Test it on file replay with known word times (aligned manifests): (first-token wall time − segment-start wall time) −
      first-word end − lead-in ≤ 80δ + 256 + 150 ms. Equivalently, the measured per-token lag ≤ the §2.1 range max + compute.
- [ ] F3 The UI latency table matches §2.1, and the measured `t_avail` of each step equals `seg_start + (4096 + 5120·⌊k/4⌋)/16000`.
      **[critic]** This holds only for non-tail steps with no lead-in, before the hold-back. With lead-in subtract `lead_in_ms`. Past `hold_from_s` the release adds 0.64 s.
      Tail and final-step frames take `t_avail` = end decision.

**G. Quality smoke (human)**
- [ ] G1 Spoken Korean, English and German sentences give sensible normalized transcripts. en→de, en→ko, de→en and ko→en give translations. 전사 mode hides the 번역 column.
- [ ] G2 A 20–40 s monologue reads naturally across segments: no duplicated or lost words at soft or forced cuts beyond the cut word itself.

**H. UI**
- [ ] H1 Korean labels throughout, two panes aligned per segment, live row with highlight, timestamps, auto-scroll and "맨 아래로".
- [ ] H2 Usable at 1440 px and 390 px, in light and dark. Space and Esc shortcuts work. Copy and JSON export work.
- [ ] **[critic] H3** Served with the §5.5 CSP header, the page logs zero CSP violations in Chrome and Safari (no inline script, style or handlers).

---

## 10. Open items and assumptions to verify

1. The lead-in default (C6) and the gain target of −26 dBFS rest on 13 local corpus samples and on reasoning, not on a decoded A/B. Verify with the real s4 on the MacBook Pro.
2. Browser behaviour was not exercised in a real browser here: Safari's handling of `autoGainControl:false` and EC→voice processing, Chrome AGC moving the OS input volume, and
   Safari `decodeAudioData` format coverage. Acceptance B1 settles it. Show `getSettings()` regardless.
3. The final re-decode cost (§7) is an estimate.
4. The engine must apply the runaway total cap with K = frames fed so far (eval knows K up front). This differs from eval only in pathological runaway cases.
   **[critic] Superseded by §1.6:** use `6·K_cap + 64` while streaming and the exact `6·K + 64` inside `finish()`, as `decode-contract.md` §d.6 says. Frames-fed-so-far is *stricter* than eval and could cut tokens eval keeps.
5. The s4 `config.json` is not on this machine, so `runaway_cap`/`max_flush_rounds` defaults (8/8) are taken from `configuration_vapasr.py:17` and the eval args. Confirm against the checkpoint at load.
   **[critic] Corrected:** the expected s4 values are `runaway_cap = 32` (`semcommit_tokens.py:98` via `semcommit_train.py:215-217`) and `max_flush_rounds = 8`. Read both from `config.json` and log them.
6. **[critic]** The per-pair translate δ defaults (ko→en δ8) rest on BLEU only. There is no δ6/δ8 xCOMET and no bootstrap. Revisit with a quick xCOMET on FLEURS ko→en δ8 if that run is cheap.
7. **[critic]** Safari: support for `OfflineAudioContext` at 16 000 Hz (file fast path) and whether Safari treats `http://127.0.0.1` as a secure context for `getUserMedia` are
   unverified here. Fallback for the first: render at the context rate and let the server resample. B1 settles the second.
8. **[critic]** HQ soxr rolls off above 7.5 kHz (−24 dB at 7.8 kHz), while training audio resampled with `torchaudio.functional.resample` (`vapasr/hf/infer.py:34-42`) keeps a
   little more top band. The top 1–2 mel bins may sit slightly lower than in training for 44.1/48 k sources. This is expected to be negligible but untested.
9. **[critic] Cross-spec consistency.**
   - `encoder-spec.md` §4.4, §6 and §8.5 also write `K = round(N/1280)`. Like this spec's §5.2, they should use eval's `int(round(N/16000*12.5))`, which differs at 36 lengths below 45 s.
   - `decode-contract.md` §d.5 describes the *eval-parity* endpoint (exactly 16 000 zeros). In VAD mode this spec deliberately uses real pause + ≥ 200 ms zeros
     (≥ 1.0 s total, R5), so only `segmentation:"none"` is held to token identity. VAD-mode segments are judged on quality, not parity.

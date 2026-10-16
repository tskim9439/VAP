# W6 integration report: VAPKT live transcription and translation app (q06-s4-ast, all-MLX)

- Date: 2026-10-16.
- Machine: Mac mini, base M4, 16 GB, macOS 26.3, mlx 0.32.2. The dev Python is the vapasr-local env (3.11.16); the packaged venv (3.12.14) was also tested.
- The machine was not idle. Other user apps were resident, and swap was already about 22.7 of 23.5 GB used before I started. Every heavy run went through `lockf heavy.lock`, one at a time.
- Scratch directory: `work-w6/` (all scripts, logs and JSON referenced below). Portable copy for the MacBook: `/Volumes/Samsung_T5/vapkt-models/app-work/w6-live-suite/`.

## 0. Summary

| item | result |
|---|---|
| Real server (`--engine mlx`), golden parity through WebSocket (39 cases, segmentation none, gain off, 16 kHz, max speed, with final re-decode) | **39/39 identical** for streaming (k, id), src text, tr text, K and final-mode ids, in **all four** precision combinations (fp32/fp32, fp16/fp16, fp16/fp32, fp32/fp16 for thinker/encoder) |
| 1× VAD through the mic path (48 kHz → soxr → VAD/gain → engine), 20 sessions / 58 segments per configuration | fp32/fp32 and fp16/fp16 give **identical** segment boundaries, src and tr texts (58/58), and identical final re-decode texts (14/14) |
| Default precision | `--precision auto --encoder-precision auto` (new startup micro-benchmark in `Engine`). On this base M4 it chooses **fp16 thinker + fp16 encoder** (7/7 startups). On faster Macs it chooses the most exact combination that passes the real-time margin rule. The decision is logged, printed, and shown in the UI tooltip |
| Default on base M4, 1×, ASR | RTF 0.23–0.28, chunk p95 121–154 ms, 0/680 chunks over 320 ms, proc_lag p95 132–162 ms, backlog ends at 0 |
| Default on base M4, 1×, translation | RTF 0.33–0.38, chunk p95 168–350 ms, 26/680 (3.8 %) chunks over 320 ms, proc_lag p95 201–333 ms, backlog ends at 0 |
| fp32/fp32 on base M4 | ASR is fine (RTF 0.38–0.44, chunk p95 ≤ 238 ms). Translation is **not real-time-safe** (chunk p95 303–620 ms, 13.7 % of chunks over 320 ms, proc_lag p95 up to 634 ms) |
| Bugs fixed across modules | (1) MLX buffer cache grew without bound: the fp32 server footprint reached **11 GB** on the 16 GB machine and caused 1–1.5 s worker stalls when final re-decode was on. Fixed with a 1 GB cache cap (fp32 footprint now ≤ 6.4 GB, no stalls). (2) The server called the blocking `final_redecode()`; it now runs `FinalJob` one step at a time (§5.2). (3) Auto-precision selection is now memory-lean (load peak 7.83 → 6.65 GB) |
| Tests | `pytest App/tests`: **254 passed** (twice in vapasr-local py3.11, once in `~/.vapkt-app/venv` py3.12); the baseline was 245. No forbidden imports |

## 1. Changes I made (all under `App/`)

| file | change |
|---|---|
| `engine/__init__.py` | **Auto precision.** `AUTO_RULE`, `AUTO_ORDER`, `predict_chunk_ms`, `choose_precision`, `auto_note`, `bench_thinker`, `bench_encoder`. `Engine(precision="auto", encoder_precision="auto")` times each component on its own, holding one measured model at a time, and picks the first passing combination in exactness order, or the fastest one if none passes. The final models are built by *consuming* the fp32 sources, and the last measured model is reused when it matches the choice. `info["auto"]` (table, measurements, rule, ms) and `info["precision_note"]` (Korean one-liner). **MLX cache cap**: new kwarg `cache_limit_gb=1.0` calls `mx.set_cache_limit`; `info["cache_limit_gb"]` records it. The predecessor's (W6 run 1) uncommitted draft of the heuristic was reviewed and rewritten (its peak was 7.83 GB) |
| `server.py` | **Step-wise final re-decode.** `Worker._final_step` runs one `FinalJob.step()` only when the input queue is empty, including while the next segment streams (the job has its own KV). The backlog check (> 0.5 s → `skipped:"lagging"`) happens at job start. `_BlockingFinal` falls back to engines that have only `final_redecode`. At most 3 pending jobs; the oldest *unstarted* job is the one dropped. On `stop`, that session's jobs are run to completion before `session_end`. The worker loop was refactored into `_loop`/`_handle` plus a `_slow()` diagnostic: any worker call over 400 ms is logged at INFO, over 100 ms at DEBUG. **Summary metrics**: `session_end` now carries `proc_lag_p50_ms`, `n_lag`, `chunk_ms{p50,p95,max,n,over_320}`, `finalize_ms{p50,p95,max}`, `first_text_ms{p50,max}`, `mem_gb`, `mem_peak_gb`. Stats carry `mem_peak_gb`; the ready log carries the load peak, then the peak is reset. **Precision note** is logged, printed to the terminal and exposed in `hello.model.precision_note`. **Nonfinite guard surfaced**: the worker reports `thinker.nonfinite` and the session sends one Korean `warning nonfinite` suggesting `--precision fp32` |
| `engine/fake.py` | `FakeSegmentDecoder.final_job()` plus `FakeFinalJob` (one id per step), the same shape as `decoder.FinalJob` |
| `web/app.js` | The model-chip tooltip shows `precision_note` (one line) |
| `tools/bench.py` | The file-mode summary prints the resolved precisions, e.g. `fp16/fp16 (요청 auto/auto)` |
| `tests/test_auto_precision.py` (new, 7 tests) | Formula, early exit, base-M4 fallback, a mid machine choosing fp32 thinker + fp16 encoder, the mean and burst limits, the thinker-only path, and the Korean note |
| `tests/test_protocol.py` (+2 tests) | `test_final_redecode_stepwise_interleaves`: the final for segment 1 arrives before segment 2 ends, runs in several `FinalJob` steps, and its text equals the segment text. `test_nonfinite_warning` |
| `README.md` (new) | Korean quickstart, options, the precision policy, architecture, measured numbers, MacBook commands and limitations |

Not changed: encoder, frontend, thinker, decoder, tokens, checkpoint, audio_pipeline, text_stream and the web UI apart from one tooltip line. No cross-module correctness bug was found in token decoding; all four golden sweeps through the full server path are identical.

## 2. Exactness

### 2.1 Golden parity through the real server (client_file.py --golden, max speed, final re-decode on)

| thinker / encoder | streaming (k,id) | src text | tr text | K | final ids | final src | wall | per-case RTF min/median/max | final re-decode ms median/max |
|---|---|---|---|---|---|---|---|---|---|
| fp32 / fp32 | 39/39 | 39/39 | 39/39 | 39/39 | 39/39 | 39/39 | 248 s | 0.345 / 0.511 / 0.752 | 1477 / 7386 |
| fp16 / fp16 (auto, run 1) | 39/39 | 39/39 | 39/39 | 39/39 | 39/39 | 39/39 | 122 s | 0.163 / 0.240 / 0.385 | 745 / 3791 |
| fp16 / fp16 (auto, final code) | 39/39 | 39/39 | 39/39 | 39/39 | 39/39 | 39/39 | 122 s | 0.161 / 0.239 / 0.396 | 739 / 3753 |
| fp16 / fp32 | 39/39 | 39/39 | 39/39 | 39/39 | 39/39 | 39/39 | 143 s | 0.216 / 0.295 / 0.451 | 759 / 3919 |
| fp32 / fp16 | 39/39 | 39/39 | 39/39 | 39/39 | 39/39 | 39/39 | 218 s | 0.256 / 0.403 / 0.688 | 1368 / 7230 |

- The cases include the K-boundary cases (edge-k4c3a, edge-k4c3b, three natural K=4c+3 clips) and the n=69760 case where `round(n/1280)` ≠ eval K.
- `consistency` (joined deltas == segment text) held in every case, and the client exited with code 0 in all five runs.
- W3's earlier fp16 final-mode flip (epst-de2-de2en-d10, at reference margin 0.0101) appeared only with *torch* features; with MLX-encoder features it does not occur.

### 2.2 Teacher-forced decision margins (6402 golden decision rows, `work-w6/tf_margins2.py`)

Each golden trajectory (`tf_seq`) is replayed in one full-sequence forward. Δm = |m_mlx − m_ref|, where m is the logit of the reference id minus the best other logit.

| thinker / features | argmax agree | max abs Δm | median over cases of the per-decision median abs Δm | cases with max abs Δm > 1e-2 | smallest MLX margin |
|---|---|---|---|---|---|
| fp32 / torch golden feats | 6402/6402 | 1.19e-4 | 1.5e-5 | 0 | 0.00362 |
| fp32 / MLX enc fp32 | 6402/6402 | 1.37e-4 | 1.4e-5 | 0 | 0.00361 |
| fp32 / MLX enc fp16 | 6402/6402 | 1.05e-2 | 5.1e-4 | 1 | 0.00288 |
| fp16 / torch golden feats | 6402/6402 | 2.39e-2 | 2.0e-3 | 36 | 0.00280 |
| fp16 / MLX enc fp32 | 6402/6402 | 2.81e-2 | 2.1e-3 | 39 | **4.96e-5** (a near-flip) |
| fp16 / MLX enc fp16 | 6402/6402 | 2.77e-2 | 2.4e-3 | 38 | 0.00253 |

- This gives the exactness order fp32/fp32 ≫ fp32 thinker + fp16 encoder (about 36× more perturbation, still 4× less than any fp16 thinker) > fp16 thinker with either encoder. `AUTO_ORDER` follows this order.
- Among the fp16-thinker variants the encoder precision makes no measurable difference: fp16/fp32 even produced the closest near-flip. So choosing fp16/fp16 over fp16/fp32 costs no exactness and saves about 17 ms per chunk.

### 2.3 VAD-mode identity across precisions (1×, mic path)

- fp32/fp32 vs fp16/fp16 had identical boundaries, src and tr in 20/20 sessions and 58/58 segments, before the cache cap (runs `fp32`, `auto`) and after it (runs `fp32b`, `auto2`).
- auto vs auto2, which are the same configuration with different code, were also 58/58 identical.
- Final re-decode texts were identical between fp32 and fp16 in 14/14 segments.

## 3. Precision policy (`auto`)

**Rule.** The chunk cost model uses golden chunk statistics from `work-w6/chunk_stats.py`. For translation chunks: tokens mean 2.85, p95 13, p99 17, max 20; emitting frames 0.73 mean, 2 at p95. For ASR chunks: tokens mean 1.16, p95 4. With enc = the encoder's steady 320 ms step, t1 = a q=1 forward and t8 = a speculative q=8 forward with 4 selected rows at context 320:

- p95 = enc + 2·t8 + 13·t1 ≤ 256 ms (0.8 × 320)
- mean = enc + 1.73·t8 + 2.85·t1 ≤ 160 ms (RTF 0.5)
- burst = enc + t8 + 23·t1 ≤ 450 ms (the thinker-spec §6.4 p99 group)

Candidates are tried in exactness order: (fp32,fp32) → (fp32,fp16) → (fp16,fp32) → (fp16,fp16). The first that passes is chosen; if none passes, the one with the smallest predicted p95 is chosen. Only components that are actually needed get measured, one model at a time.

**Measured on this M4.** 7 startups: 3 + 2 from `auto_probe*.py`, plus the 2 server runs.

| component | t1 / step ms | t8 ms |
|---|---|---|
| thinker fp32 | 30.3–32.2 | 37.1–38.9 |
| thinker fp16 (resid32 + rescore8) | 16.3–18.2 | 19.6–20.6 |
| encoder fp32 step | 33.7–36.1 | |
| encoder fp16 step | 17.0–19.8 | |

| candidate | predicted p95 / mean / burst (ms) | passes? |
|---|---|---|
| fp32/fp32 | 514–524 / 189–191 | no |
| fp32/fp16 | 498–506 / 172–174 | no |
| fp16/fp32 | 289–304 / 116–120 | no (p95) |
| fp16/fp16 | 270–296 / 99–106 / 420–443 | no (p95) → **chosen as fastest**, reason `fallback_fastest` |

The model is well calibrated against the 1× measurements:
- fp16/fp16 predicted mean 99–106 ms vs measured translation mean 0.33–0.38 × 320 = 105–122 ms. Predicted p95 270–296 ms vs measured translation chunk p95 168–350 ms.
- fp32/fp32 predicted mean 190 ms vs measured 0.53–0.63 × 320 = 170–202 ms.

**Cost.**

| config | load ms | MLX peak at load | MLX resident | process footprint (top MEM) during 1× sessions |
|---|---|---|---|---|
| auto (→ fp16/fp16) | 4879–5885, of which auto measurement 4287–5170 | 6.65 GB | 3.25 GB | 3.4–4.4 GB (cache cap 1 GB) |
| fixed fp16/fp16 | 1888 | 5.69 GB | 3.25 GB | |
| fixed fp32/fp32 | 1788 | 5.64 GB | 5.22 GB | 5.3–6.4 GB (cache cap); **up to 11 GB without the cap** |
| fixed fp16/fp32 | 1823 | 4.83 GB | 4.41 GB | |

Warmup takes 0.8–1.0 s for fp16/fp16, 1.2 s for fp16/fp32 and 1.9 s for fp32/fp32. The runtime MLX peak during sessions was 3.85 GB for auto and 5.25–5.8 GB for fp32.

**Why this default.**
- On this M4, fp32 thinker is not real-time-safe for translation (§4). fp16/fp16 is the fastest combination that is still token-identical on all 39 goldens and all 58 VAD segments, and section 2.2 shows it is no less exact than fp16/fp32.
- On a MacBook Pro the same rule should pick an fp32 thinker when t1 ≲ 15 ms. That is plausible on M4 Pro/Max but **not measured**. The README tells the user to read the startup log line.
- Override with `--precision fp32|fp16 --encoder-precision fp32|fp16`.

## 4. Real-time 1× results (VAD, mic path)

**Inputs** (`work-w6/make_audio.py`; 48 kHz float WAV, sent with `--source mic --speed 1` in 20 ms packets):
- `clip-*`: each of the 6 TTS clips with 0.5 s lead and 1.2 s tail of −70 dBFS noise. Each ran as ASR δ6 and as translation (en1→de, en2→ko, de→en at δ10; ko→en at δ8).
- `conv-ko`: 63.5 s; ko1, kspon1, ko2, kspon2, kspon3 with 0.6–1.8 s pauses and a −70 dBFS floor.
- `conv-en`: 60.6 s; en1, epst-en1, en2, epst-en2 ×2 with 0.5–2.0 s pauses. conv-ko and conv-en each ran as ASR and translation (ko→en δ8, en→ko δ10).
- `chain-en`: 44.3 s of 7 trimmed English clips with no pauses, ASR and en→de δ10.
- Plus `conv-en-tr` and `conv-ko-asr` again with final re-decode on.

**Definitions.**
- chunk = round-attributed enc+dec per 320 ms chunk.
- proc_lag = round ready time − arrival of the last needed sample. As the spec says, burst rounds (segment-start pre-roll) and held rounds are excluded.
- steady backlog = max backlog excluding the first 1.5 s after each segment start.
- RTF, finalize_ms and first_text_ms are as in audio-ui-spec §2.2.

**Aggregate.** Definitive runs `auto2` and `fp32b`, final code with the 1 GB cache cap. 9 ASR + 9 translation sessions per config, 217 s each.

| config | mode | RTF | chunk p95 ms | chunks > 320 ms | proc_lag p50 / p95 ms | steady backlog max | backlog at end | finalize p50 ms | loop-lag p99 |
|---|---|---|---|---|---|---|---|---|---|
| auto = fp16/fp16 | ASR | 0.234–0.277 | 121–154 | 0 / 680 | 73–93 / 132–162 | 0.10–0.16 s (0.67 s only at a hold-back release) | 0 | 60–188 | ≤ 4.1 ms |
| auto = fp16/fp16 | translate | 0.325–0.375 | 168–350 | 26 / 680 (3.8 %) | 79–113 / 201–333 | 0.16–0.67 s | 0 | 185–414 | ≤ 4.0 ms |
| fp32/fp32 | ASR | 0.375–0.442 | 194–238 | 0 / 680 | 99–139 / 198–244 | 0.13–0.67 s | 0 | 107–350 | ≤ 11.5 ms (one blip in the first session; others ≤ 4.2) |
| fp32/fp32 | translate | 0.528–0.633 | 303–620 | 93 / 680 (13.7 %) | 101–203 / 366–634 | 0.42–0.77 s | 0 | 331–885 | ≤ 4.2 ms |

**Before the cache cap** (runs `auto` and `fp32`, same audio): auto was the same within noise (ASR chunk p95 118–143 ms, translation 201–335 ms, 4.4 % of chunks over 320 ms). fp32 translation was slightly worse (15.1 % over, chunk p95 314–628 ms) under memory pressure.

**Final re-decode on (default off).**

| config | session | proc_lag p95 off → on | stalls > 0.6 s with finals on | final ms per segment |
|---|---|---|---|---|
| auto2 | conv-ko-asr | 156 → 152 ms | none | 1.37 s for the 19 s segment |
| auto2 | conv-en-tr | 296 → 322 ms | none | |
| fp32, before the cap | conv-ko-asr | 245 → 392 ms | 1.5 s at a segment boundary | 4.2 s for the 19 s segment |
| fp32b, after the cap | conv-ko-asr | 236 → 245 ms | none | 2.46 s for the 19 s segment |

Isolated FinalJob step timing from `final_steps.py`:

| config | segment | prefill ms | token step ms | total |
|---|---|---|---|---|
| fp32 | 19 s, K=248, 71 ids | 140–163 | 33 | 2.5 s |
| fp32 | 5.6 s, K=72 | 60–66 | 32–33 | 1.1 s |
| fp16 | 19 s | 112 | 17.5 | 1.35 s |
| fp16 | 5.6 s | 47–50 | 17 | 0.6 s |

- No single step exceeds about 165 ms, so a step-wise job delays a streaming chunk by at most one step.
- I recommend keeping finals off by default as the spec says. Turning them on is fine with auto on this M4: streaming latency is unchanged.

**Example outputs** (auto2, identical in fp32b):
- conv-ko segment 1: "오늘 회의에서는 다음 분기 예산과 신규 채용 계획에 대해 논의하겠습니다 각 팀장님들은 자료를 미리 준비해 주세요".
- conv-en-tr segment 3: "the financial crisis rages on and european eurozone countries are facing recession" → "금융 위기가 계속 이어지고 있고, 유럽 유로존 국가들은 경기 침체에 직면해 있어요."
- chain-en-tr segment 1 → "Das Europäische Parlament wird am nächsten Dienstag über das neue Klimaabkommen abstimmen, …".
- chain-en: 0–19.26 s soft_max and 19.26–43.55 s soft_max. The 24.3 s second segment passed hold_from, so the 640 ms hold-back was released at the cut. This is the 0.67 s backlog seen in chain-en.

**Model-quality observations.** These are identical across precisions, so they are model behaviour, not engine bugs:
- In chain-en **translate** mode, segment 1's src stream skipped the spliced Europarl sentence "free markets are not synonymous…" entirely. In ASR mode the same sentence reads "three markets are not synonymous with markets with deregulated markets".
- In conv-en-tr segment 4, "train station" became "역" (the word for "train" was dropped).
- In conv-en segment 2 (0.5 s gap < 800 ms end pause), two utterances were merged, as designed.

### 4.x Per-session tables (definitive runs, final code)

Column notes: chunk is the round-attributed sum per 320 ms chunk. backlog is given as max / steady (excluding 1.5 s after segment starts) / end. Loop is the maximum of the rolling 10 s event-loop lag p99.

#### auto2

| session | audio s | segs | RTF | chunk p50/p95/max ms | >320 | proc_lag p50/p95 ms | backlog max / steady / end s | finalize p50/max ms | first_text p50/max ms | loop p99 max ms |
|---|---|---|---|---|---|---|---|---|---|---|
| clip-en1-asr (en δ6) | 8.9 | 1 | 0.243 | 76.7/120.5/127.8 | 0/27 | 91.0/134.5 | 0.64 / 0.128 / 0.0 | 69.4/69.4 | 866.4/866.4 | 2.63 |
| clip-en1-tr (en→de δ10) | 8.9 | 1 | 0.375 | 91.9/274.0/437.6 | 1/27 | 113.2/274.4 | 0.672 / 0.32 / 0.0 | 252.2/252.2 | 1184.5/1184.5 | 2.76 |
| clip-en2-asr (en δ6) | 6.9 | 1 | 0.264 | 91.1/121.5/125.3 | 0/21 | 92.6/140.3 | 0.16 / 0.16 / 0.0 | 60.3/60.3 | 879.1/879.1 | 2.76 |
| clip-en2-tr (en→ko δ10) | 6.9 | 1 | 0.353 | 100.9/285.8/350.2 | 1/21 | 93.6/332.8 | 0.672 / 0.192 / 0.0 | 366.9/366.9 | 1210.3/1210.3 | 2.36 |
| clip-ko1-asr (ko δ6) | 9.9 | 1 | 0.274 | 90.6/129.8/135.9 | 0/31 | 83.7/149.1 | 0.64 / 0.096 / 0.0 | 120.5/120.5 | 878.8/878.8 | 2.53 |
| clip-ko1-tr (ko→en δ8) | 9.9 | 1 | 0.332 | 102.1/168.1/341.3 | 1/31 | 89.3/200.8 | 0.608 / 0.256 / 0.0 | 188.0/188.0 | 1268.4/1268.4 | 3.18 |
| clip-ko2-asr (ko δ6) | 7.4 | 1 | 0.277 | 90.6/150.3/169.7 | 0/23 | 82.4/161.3 | 0.608 / 0.096 / 0.0 | 116.7/116.7 | 870.9/870.9 | 3.19 |
| clip-ko2-tr (ko→en δ8) | 7.4 | 1 | 0.339 | 84.3/219.5/323.9 | 1/23 | 104.8/278.9 | 0.608 / 0.352 / 0.0 | 279.5/279.5 | 1230.1/1230.1 | 3.18 |
| clip-de1-asr (de δ6) | 9.9 | 1 | 0.254 | 63.3/154.3/173.2 | 0/31 | 78.5/160.9 | 0.608 / 0.128 / 0.0 | 100.2/100.2 | 860.4/860.4 | 2.65 |
| clip-de1-tr (de→en δ10) | 9.9 | 1 | 0.334 | 90.6/226.6/353.8 | 1/31 | 78.6/247.9 | 0.608 / 0.16 / 0.0 | 374.5/374.5 | 1282.5/1282.5 | 2.47 |
| clip-de2-asr (de δ6) | 5.9 | 1 | 0.237 | 55.0/127.3/149.7 | 0/18 | 88.8/149.3 | 0.64 / 0.096 / 0.0 | 70.2/70.2 | 1548.5/1548.5 | 2.47 |
| clip-de2-tr (de→en δ10) | 5.9 | 1 | 0.337 | 51.1/227.6/268.6 | 0/18 | 81.4/269.9 | 0.576 / 0.192 / 0.0 | 215.4/215.4 | 2009.3/2009.3 | 2.35 |
| conv-ko-asr (ko δ6) | 63.5 | 8 | 0.268 | 86.8/142.1/174.5 | 0/199 | 74.8/156.0 | 0.448 / 0.128 / 0.0 | 94.3/223.7 | 1020.5/1236.8 | 4.08 |
| conv-ko-tr (ko→en δ8) | 63.5 | 8 | 0.358 | 88.7/312.1/423.4 | 8/199 | 82.7/326.9 | 0.672 / 0.352 / 0.0 | 185.2/509.3 | 1260.5/1558.9 | 3.6 |
| conv-en-asr (en δ6) | 60.6 | 6 | 0.234 | 71.5/121.0/149.7 | 0/188 | 72.7/131.5 | 0.672 / 0.16 / 0.0 | 65.5/120.3 | 888.7/1174.6 | 3.76 |
| conv-en-tr (en→ko δ10) | 60.6 | 6 | 0.325 | 76.7/350.3/441.0 | 10/188 | 79.2/296.4 | 0.704 / 0.352 / 0.0 | 367.2/475.3 | 1186.8/1471.7 | 3.35 |
| chain-en-asr (en δ6) | 44.3 | 2 | 0.255 | 81.7/130.6/188.0 | 0/142 | 89.0/161.6 | 0.672 / 0.672 / 0.0 | 187.9/371.3 | 607.3/860.4 | 3.43 |
| chain-en-tr (en→de δ10) | 44.3 | 2 | 0.363 | 87.8/244.5/422.7 | 3/142 | 86.3/265.6 | 0.672 / 0.672 / 0.0 | 413.7/681.0 | 942.6/1211.6 | 4.04 |
| conv-en-tr-final (en→ko δ10) | 60.6 | 6 | 0.324 | 79.6/342.5/451.4 | 11/188 | 77.0/321.5 | 0.64 / 0.384 / 0.0 | 359.1/485.9 | 1188.3/1478.4 | 3.17 |
| conv-ko-asr-final (ko δ6) | 63.5 | 8 | 0.263 | 84.8/140.3/169.2 | 0/199 | 75.0/152.3 | 0.576 / 0.16 / 0.0 | 83.4/234.2 | 1018.3/1209.6 | 3.95 |

MLX memory (active, runtime peak) seen: [(3.27, 3.85)]

#### fp32b

| session | audio s | segs | RTF | chunk p50/p95/max ms | >320 | proc_lag p50/p95 ms | backlog max / steady / end s | finalize p50/max ms | first_text p50/max ms | loop p99 max ms |
|---|---|---|---|---|---|---|---|---|---|---|
| clip-en1-asr (en δ6) | 8.9 | 1 | 0.393 | 132.1/199.9/205.9 | 0/27 | 127.9/198.1 | 0.608 / 0.16 / 0.0 | 112.2/112.2 | 942.1/942.1 | 11.45 |
| clip-en1-tr (en→de δ10) | 8.9 | 1 | 0.633 | 167.0/458.1/757.3 | 4/27 | 203.1/610.0 | 0.768 / 0.768 / 0.0 | 475.0/475.0 | 1275.8/1275.8 | 2.82 |
| clip-en2-asr (en δ6) | 6.9 | 1 | 0.434 | 141.0/210.2/211.1 | 0/21 | 138.6/215.9 | 0.192 / 0.16 / 0.0 | 107.2/107.2 | 971.2/971.2 | 4.21 |
| clip-en2-tr (en→ko δ10) | 6.9 | 1 | 0.602 | 165.5/526.7/659.9 | 3/21 | 137.7/623.6 | 0.672 / 0.48 / 0.0 | 695.9/695.9 | 1266.2/1266.2 | 4.21 |
| clip-ko1-asr (ko δ6) | 9.9 | 1 | 0.442 | 144.7/206.4/247.5 | 0/31 | 108.2/217.6 | 0.64 / 0.224 / 0.0 | 177.7/177.7 | 937.6/937.6 | 3.42 |
| clip-ko1-tr (ko→en δ8) | 9.9 | 1 | 0.577 | 164.1/303.1/592.5 | 2/31 | 151.9/365.5 | 0.608 / 0.48 / 0.0 | 330.6/330.6 | 1404.5/1404.5 | 2.83 |
| clip-ko2-asr (ko δ6) | 7.4 | 1 | 0.440 | 140.1/220.9/254.3 | 0/23 | 107.8/226.8 | 0.224 / 0.224 / 0.0 | 191.6/191.6 | 928.2/928.2 | 3.49 |
| clip-ko2-tr (ko→en δ8) | 7.4 | 1 | 0.574 | 144.3/378.7/627.3 | 4/23 | 176.2/634.2 | 0.672 / 0.672 / 0.0 | 465.6/465.6 | 1311.7/1311.7 | 3.49 |
| clip-de1-asr (de δ6) | 9.9 | 1 | 0.420 | 112.0/217.4/298.0 | 0/31 | 99.9/237.4 | 0.64 / 0.288 / 0.0 | 193.4/193.4 | 939.5/939.5 | 2.76 |
| clip-de1-tr (de→en δ10) | 9.9 | 1 | 0.549 | 141.7/375.9/618.9 | 3/31 | 100.5/510.7 | 0.64 / 0.416 / 0.0 | 667.2/667.2 | 1376.5/1376.5 | 3.02 |
| clip-de2-asr (de δ6) | 5.9 | 1 | 0.375 | 78.2/203.6/273.5 | 0/18 | 119.5/221.5 | 0.608 / 0.128 / 0.0 | 187.6/187.6 | 1674.2/1674.2 | 3.02 |
| clip-de2-tr (de→en δ10) | 5.9 | 1 | 0.528 | 77.8/370.1/393.7 | 4/18 | 143.8/400.9 | 0.576 / 0.416 / 0.0 | 366.8/366.8 | 2133.7/2133.7 | 2.49 |
| conv-ko-asr (ko δ6) | 63.5 | 8 | 0.431 | 140.6/238.0/274.8 | 0/199 | 100.0/236.1 | 0.768 / 0.256 / 0.0 | 149.1/383.9 | 1111.9/1333.3 | 3.79 |
| conv-ko-tr (ko→en δ8) | 63.5 | 8 | 0.594 | 142.3/554.0/774.4 | 27/199 | 166.7/626.3 | 0.896 / 0.768 / 0.0 | 463.2/875.4 | 1381.4/1705.9 | 3.72 |
| conv-en-asr (en δ6) | 60.6 | 6 | 0.375 | 109.5/194.0/251.1 | 0/188 | 99.1/197.9 | 0.736 / 0.224 / 0.0 | 115.8/191.5 | 962.4/1258.6 | 3.41 |
| conv-en-tr (en→ko δ10) | 60.6 | 6 | 0.546 | 134.3/620.2/754.9 | 20/188 | 124.0/582.8 | 0.8 / 0.608 / 0.0 | 688.1/845.5 | 1259.3/1557.9 | 3.74 |
| chain-en-asr (en δ6) | 44.3 | 2 | 0.413 | 137.7/221.8/288.0 | 0/142 | 120.3/244.1 | 0.704 / 0.672 / 0.0 | 350.4/623.5 | 647.6/926.0 | 3.6 |
| chain-en-tr (en→de δ10) | 44.3 | 2 | 0.631 | 152.2/441.4/699.7 | 26/142 | 179.7/621.1 | 0.832 / 0.704 / 0.0 | 885.0/1619.3 | 1142.9/1274.1 | 3.56 |
| conv-en-tr-final (en→ko δ10) | 60.6 | 6 | 0.548 | 134.3/613.5/808.2 | 19/188 | 129.4/594.1 | 0.864 / 0.704 / 0.0 | 646.6/888.6 | 1263.6/1566.3 | 3.39 |
| conv-ko-asr-final (ko δ6) | 63.5 | 8 | 0.436 | 142.6/243.1/281.6 | 0/199 | 105.9/244.5 | 1.12 / 0.256 / 0.0 | 150.4/418.3 | 1120.6/1334.7 | 3.68 |

MLX memory (active, runtime peak) seen: [(5.23, 5.25), (5.23, 5.26), (5.23, 5.68), (5.23, 5.8)]

## 5. Diagnosis: worker stalls with final re-decode on fp32 were caused by memory pressure (fixed)

1. **Symptom.** In the first fp32 run (`runs/fp32`, conv-ko-asr-final), `audio_done_s` stopped advancing for 1.5 s right after a soft-max cut. Segment 6 then took `first_text` 2.7 s and `finalize` 1.1 s, steady backlog reached 1.57 s and proc_lag p95 rose 245 → 392 ms. conv-en-tr-final had two 1.0 s stalls.
2. **Ruled out.** The event loop was not starved: loop-lag p99 stayed ≤ 6 ms. Final steps in isolation are ≤ 163 ms (§4).
3. **Cause.** The same 19 s final took 4.2 s inside the server against 2.5 s in isolation. The fp32 server's footprint (`top` MEM) grew 5.3 → 11 GB over the run. MLX keeps freed buffers in a cache that by default can grow to the whole memory limit, and the final-mode KV and prefill buffers have a different size every segment, so they pile up. On a 16 GB machine that was already deep in swap, this causes GPU paging stalls. The fp16 server showed the same growth (3.4 → 7.0 GB) but less pressure.
4. **Fix.** `Engine(cache_limit_gb=1.0)` calls `mx.set_cache_limit(1 GiB)`.
5. **Diagnostic rerun.** `runs/fp32-diag`, fp32 with `--debug`:
   - footprint ≤ 6.3 GB
   - conv-ko-asr-final had no stalls; proc_lag p95 242 ms and the 19 s final took 2.46 s (= isolated)
   - no final step over 400 ms; the only slow worker calls were fp32 translation bursts (600–800 ms `audio` or `seg_end` calls)
6. **Confirmation.** The full reruns `auto2` and `fp32b` confirm the result with no speed regression; auto2 even has a slightly lower over-budget share (3.8 % vs 4.4 %).

## 6. Known limitations and open issues

1. **Everything was measured on a base M4 (16 GB, busy, swapped).** MacBook Pro numbers are not measured. Use §7.
2. **Translation on the base M4 is at the real-time edge even with fp16/fp16.** Chunk p95 is up to 350 ms; 3.8 % of chunks exceed 320 ms, during bursts of 13 or more tokens per chunk. Backlog always returned to 0, "chunk p95 < 320 ms" holds for all 9 ASR sessions and 8 of 9 translation sessions. It misses only for conv-en-tr (350 ms), and for that session's final-on repeat (342 ms). There are no further exact levers without new engine work:
   - async token pipelining (thinker-spec §4.6, about 0.8 ms/token)
   - a two-stage lm_head (about 2.5 ms per forward, not prototyped)

   q8 is not exact (W3v: 34/39).
3. **F1 as written (`backlog_s max < 0.3 s`) cannot pass on any machine.** The VAD pre-roll burst (0.5–0.7 s) and the hold-back release (0.64 s) are released at once by design. I report the steady backlog (excluding 1.5 s after each segment start) and the backlog at end instead. The burst still makes the first rounds of a segment wait; on fp32 this is visible as 0.75–1.3 s "stalls" at segment starts in `stalls.py`. These are not engine stalls.
4. **Auto adds about 4.3–5.2 s to startup.** It measures every time; no cached decision. The measurement is pessimistic if the machine is busy at startup, which only pushes it toward fp16.
5. **auto falls back to the fastest combination even when the fastest still misses the p95 rule.** It never chooses quantized or other non-exact modes.
6. Final re-decode stays default off. With auto on this M4 it does not disturb streaming. With fp32 translation on the M4 the machine is already over budget.
7. **Not exercised here:** Safari, a hardware mic, Bluetooth, real room noise, sessions longer than 64 s, 5-minute F1 sessions, the lead-in A/B (C6), and gain on quiet real speech (C7).
8. **kv_guard at max file speed fires late** (W5v). Unchanged; it only matters for a pathological runaway.
9. **One loop-lag p99 blip of 11.5 ms** in the first fp32b session, above the A6 target of 10 ms. All other sessions were ≤ 4.5 ms.
10. **Pre-existing `__pycache__` dirs** under App/engine, tests and tools predate this run (other agents). I created and then removed App/`__pycache__` and six cpython-312 `.pyc` files from one import check. One older `engine/__pycache__/checkpoint.cpython-312.pyc` is not mine and was left.

## 7. Reproducing on the MacBook Pro

```bash
# 0) Repo at ~/Desktop/VAPKT, model at ~/Desktop/VAPKT-models/semcommit-q06-s4-ast.nosync/final (or --model DIR)
bash App/setup.sh                                   # venv ~/.vapkt-app/venv + pytest (expect 254 passed)
bash App/run.sh                                     # look for the line: "자동 정밀도: thinker … · 인코더 … (…)"
# Fixed precision instead of auto:
bash App/run.sh --precision fp32 --encoder-precision fp32

# 1) Engine-only timings (no server; stop the app first)
~/.vapkt-app/venv/bin/python App/tools/bench.py --micro                         # t1/t8 per precision and context
~/.vapkt-app/venv/bin/python App/tools/bench.py --wav <en.wav> --src English --tgt German --pace 0 --final --precision fp32
~/.vapkt-app/venv/bin/python App/tools/bench.py --wav <en.wav> --src English --tgt German --pace 0 --final   # auto

# 2) Full-server suite (T5 attached). Terminal A: the server. Terminal B: the suite.
bash App/run.sh --no-browser --port 8799 [--precision fp32 --encoder-precision fp32]
S=/Volumes/Samsung_T5/vapkt-models/app-work/w6-live-suite
APP=$HOME/Desktop/VAPKT/App PY=$HOME/.vapkt-app/venv/bin/python bash $S/live_suite.sh ~/w6-mbp-auto 8799 all
#   golden parity (39 cases, expect "parity: 39/39") + 20 1× VAD sessions (about 11 min) + summary table
~/.vapkt-app/venv/bin/python $S/stalls.py ~/w6-mbp-auto
~/.vapkt-app/venv/bin/python $S/cmp_runs.py $S/reference-m4/auto2 ~/w6-mbp-auto      # texts vs this M4 run
```

Accept fp32 as the MacBook default if the startup line says `thinker fp32` (rule passed) **and** the suite shows translation chunk p95 < 256 ms with backlog at end 0. Otherwise keep auto's choice.

## 8. Artifacts and exact commands used here

- **Scripts** (`work-w6/`):
  - `run_suite.sh <tag> <port> <golden|vad|final|all> -- <server args>` starts `server.py --engine mlx` under `lockf heavy.lock`, waits for health, runs `run_clients.sh`, logs `top` MEM every 5 s, and stops the server with SIGINT.
  - `run_clients.sh` holds the client_file commands:
    `client_file.py --golden <GOLD> --speed 0 --final-redecode --quiet --json golden.json` and
    `client_file.py audio/<wav>.wav --src <S> [--mode translate --tgt <T>] --source mic --speed 1 --quiet --json <name>.json [--final-redecode]`.
  - `make_audio.py` builds the test audio; `analyze.py`, `stalls.py`, `cmp_runs.py` and `tables.py` summarize.
  - `tf_margins2.py` (§2.2), `auto_probe.py` and `auto_probe2.py` (§3), `final_steps.py` (§4), `chunk_stats.py` (§3 constants).
- **Results** (`work-w6/runs/<tag>/`: summary.json, per-session JSON, server.log, mem.log):
  - `fp32` and `auto`: before the cache cap.
  - `fp16-fp32` and `fp32-fp16`: golden only.
  - `fp32-diag`: the stall diagnosis.
  - `auto2` and `fp32b`: final code, definitive.
  - `fake-suite`: the portable-suite check against the fake engine.
  - `tf_margins.json`, `auto_probe*.json`, `final_steps.json`, `bench-en1-auto.json`.
- **Verification runs:**
  - `pytest App/tests -q`: 254 passed in 20.2 s (×2, py3.11). In `~/.vapkt-app/venv` (py3.12.14): 254 passed in 20.0 s. Importing engine and server loads none of torch, transformers, nemo, mlx_lm or vapasr.
  - `tools/bench.py --wav testaudio/en1.wav --src English --tgt German --pace 0 --final` (auto → fp16/fp16): RTF 0.349, chunk wall p95 300 ms, final 1221 ms (70 tokens).

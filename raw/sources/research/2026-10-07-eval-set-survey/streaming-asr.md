# Evaluation-set survey: streaming-asr (2026-10-07, research agent report, verbatim)

I could open every source on your list, plus five recent streaming LLM ASR papers, using arXiv HTML, the Hugging Face cards and APIs, OpenSLR, the AI Hub page and the AA methodology page. I didn't modify anything in the repo; scratch copies are in `/private/tmp/claude-501/-Users-taesookim-Desktop-VAPKT/49169b81-fc5f-478d-acc2-17601e549321/scratchpad/lit/`.

The most important finding is a near-copy of your setup posted three days ago. Factorized DSM (arXiv 2610.04333, 2026-10-03) uses the Nemotron-3.5-ASR-Streaming-0.6B FastConformer encoder at 12.5 Hz with a Qwen3.5-0.8B LLM and delayed emission. It evaluates only on LibriSpeech and CSJ (a Japanese corpus), at target delays of 160–800 ms. It is your closest prior art for both novelty and comparison.

## (1) Per-work table

| Work | Languages | Eval sets (splits) | Quality metric and normalization | Latency metric and operating points | Link |
|---|---|---|---|---|---|
| Nemotron 3.5 ASR streaming 0.6B (cache-aware FastConformer-RNNT) | 40 locales, including en, de, ko | Card lists FLEURS, MCV (Common Voice), MLS and internal sets; it only publishes FLEURS test numbers (configs `en_us`, `de_de`, `ko_kr`) | WER; CER for ja/ko/zh. "Text normalization that aligns reference and hypothesis", but no specific normalizer is named | Chunk size = current frame + right context: 80, 160, 320, 560, 1120 ms (att_context [56,{0,1,3,6,13}]). Also a figure of median final-token latency against concurrency on one H100. FLEURS with language ID given, at 80/160/320/560/1120 ms: **de 9.81 / 9.21 / 8.83 / 8.42 / 8.31 WER; ko 7.59 / 7.70 / 7.27 / 7.18 / 7.12 CER**; en 9.43 → 7.91 | https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b |
| Nemotron Speech Streaming EN 0.6B | en | Open ASR Leaderboard 8 sets: AMI, Earnings22, GigaSpeech, LS test-clean, LS test-other, SPGI, TEDLIUM, VoxPopuli | WER after `whisper-normalizer` 0.1.12, without punctuation and casing | Chunk 80/160/560/1120 ms ([70,{0,1,6,13}]). Average WER 8.43 / 7.67 / 7.07 / 6.93 | https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b |
| Cache-aware streaming FastConformer (Noroozi et al., ICASSP 2024) | en | LS test-other (main). Multi-domain table: LS test-other/test-clean, SPGI, Earnings22, GigaSpeech, TED-LIUM, MCV (version not given), VoxPopuli, AMI | WER | Average encoder-induced algorithmic latency ("average time needed for each word to get predicted… ignoring inference time"): 0, 40, 240, 520, 680, 1360 ms. Buffered baseline at 2000 ms | https://arxiv.org/abs/2312.17279 |
| Fast Conformer (Rekesh et al., ASRU 2023) | en | LS test-other, MCV 8, MLS-en, WSJ-92; Open ASR Leaderboard sets (MCV 9 test); long-form TED-LIUM v3 and Earnings21 | WER, Whisper normalizer | Offline or buffered only (20 s buffers); RTF | https://arxiv.org/abs/2305.05084 |
| Whisper-Streaming (Macháček et al., 2023) | en, de, cs | ESIC dev set: European Parliament speeches, with de/cs from simultaneous interpreters | WER with punctuation and casing removed | Word emission time minus gold word time (ESIC timestamps, edit-distance alignment), reported compute-aware and compute-unaware. MinChunkSize 0.1/0.5/1.0/2.0 s. At 1 s: 3.3 s en, 4.4 s de, 4.8 s cs. **de (interpreter speech): WER 9.3–12.8, latency 3.83–5.94 s** | https://arxiv.org/abs/2307.14743 |
| Kyutai DSM-ASR / STT | Paper: en. Cards: stt-2.6b-en (en), stt-1b-en_fr (en+fr) | Short-form: Open ASR Leaderboard 8 sets. Long-form: TED-LIUM, Meanwhile, Rev16, Earnings21. Latency: LS test-clean with MFA pseudo-timestamps (Lugosch et al.) | Micro-averaged WER with the official Open ASR Leaderboard code and Whisper English normalizer | Fixed delay 2.5 s (2.6B) or 0.5 s (1B), plus a delay-conditioned variant. Latency = mean delay between the true word time and its transcription. Throughput = RTF × batch. Also re-evaluates Whisper-Streaming (2.5 s), SeamlessStreaming and chunked Parakeet on the same 8 sets. Cards give no evaluation tables | https://arxiv.org/abs/2509.08753 ; https://huggingface.co/kyutai/stt-2.6b-en ; https://huggingface.co/kyutai/stt-1b-en_fr |
| Moshi | en | LS test-clean | WER | 2 s text delay. 5.7% WER, against 3.6% for streaming FastConformer at "similar look-ahead" | https://arxiv.org/abs/2410.00037 |
| Qwen3-ASR | 30 languages, including de, ko | English: LS clean/other, GigaSpeech, CV-en, Fleurs-en, MLS-en, Tedlium, VoxPopuli. Multilingual: MLS (de), CommonVoice (de, ko; version not given), MLC-SLM (de, ko), Fleurs (de, ko) | WER; CER for zh, yue, ko; macro averages. No normalizer stated | **Streaming is evaluated only on LS, Fleurs-en and Fleurs-zh**: 2 s chunks, 5-token fallback, last 4 chunks unfixed. TTFT and RTF under vLLM concurrency. No de/ko streaming numbers | https://arxiv.org/abs/2601.21337 |
| Speech ReaLLM (Interspeech 2024) | en | LS test and dev, clean/other | WER | 240 ms input chunks, 960 ms encoder right context; tested a 480 ms label delay. No measured latency metric | https://arxiv.org/abs/2406.09569 |
| SeamlessStreaming | about 100 languages | ASR: only an average over 90 FLEURS languages ("Fleurs-90"); the split isn't stated in that table | WER, Whisper normalizer (metric card) | AL and LAAL in seconds over SentencePiece tokens, at EMMA threshold 0.4–0.7. ASR: WER 31.3–30.9, AL 1.19–1.29 s. Start/end silence trimmed | https://arxiv.org/abs/2312.05187 |
| Canary-1B-v2 / Parakeet-TDT-0.6B-v3 (offline) | 25 European languages | FLEURS (25), MLS (5, not de), CoVoST2 (13, includes de), Open ASR Leaderboard 8 sets | Open ASR Leaderboard English normalizer; "multilingual normalizer" for other languages | RTFx only | https://arxiv.org/abs/2509.14128 |
| Open ASR Leaderboard paper | en; multilingual track de/fr/it/es/pt | English short-form: the 8 sets (Earnings22 is a 5 h subset). Multilingual: FLEURS, CoVoST-2, MLS (MLS only fr/it/es/pt). Long-form: TED-LIUM v3, Earnings21/22, CORAAL. Data at `hf-audio/esb-datasets-test-only-sorted`, split=test | WER after a Whisper-style English normalizer (punctuation, casing and fillers removed; numbers normalized) | RTFx only; no streaming latency. No Korean | https://arxiv.org/abs/2510.06961 |
| Artificial Analysis STT | en only | AA-AgentTalk (proprietary, 50%), VoxPopuli-Cleaned-AA (English, 628 clips, 25%), Earnings22-Cleaned-AA (6 calls, 25%) | Duration-weighted WER, Whisper normalizer plus their own extra rules (AA-WER v2.2) | Streaming: Time to Final Transcript and Time to First Partial, measured after SileroVAD end-of-speech, with forced endpointing; 20 ms real-time chunks; network delay included | https://artificialanalysis.ai/speech-to-text/methodology |
| Voxtral Realtime (DSM-style LLM streaming) | 13 languages, including de, ko | En-Short: LS-C/O, GigaSpeech, VoxPopuli, SwitchBoard, CallHome, CHiME-4, SPGI, TED, E22 (no AMI). En-Long: Meanwhile, Earnings21/22, TED-LIUM. FLEURS and MCV in 13 languages (MCV version not given) | WER; CER only for zh/ja, **so Korean is scored as WER**. No normalizer stated | Target delay 240/480/960/2400 ms; also reruns Nemotron EN at 560/1120 and DSM at 500/2500. **FLEURS de 8.15 / 6.19 / 4.87 / 4.15; ko 17.56 / 15.74 / 14.90 / 14.30 (WER). MCV de 11.13 / 8.70 / 6.85 / 5.66; ko 33.47 / 31.37 / 27.24 / 25.26** | https://arxiv.org/abs/2602.11298 |
| F-DSM (Factorized DSM) | en, ja | LS test-clean/test-other; CSJ eval1–3 | WER (LS), CER (CSJ) | Target delay 160–800 ms (main results at 480 ms); RTF | https://arxiv.org/abs/2610.04333 |
| VibeVoice-ASR-Streaming | 9 MLC languages, including de, ko; zh | MLC-Challenge (two-speaker conversations, recordings capped at 480 s), AMI-IHM/SDM, AliMeeting, AISHELL-4; also LS clean/other, GigaSpeech, AISHELL-1 | WER and cpWER via MeetEval; CER for zh/ja/ko | Expected algorithmic delay = C/2 + lookahead: 2.9 s chunks → 2.00 s, 2.0 s chunks → 1.53 s; wall-clock for APIs. MLC: **de 21.83 WER, ko 9.09 CER** | https://arxiv.org/abs/2609.02812 |
| Uni-ASR | zh, en | AISHELL-1/2, LS clean/other, FLEURS en/zh, WeNetSpeech | CER/WER | Chunk length 1000/640/320 ms | https://arxiv.org/abs/2603.11123 |
| Alignment-path distillation | zh, en | AISHELL, KeSpeech, WeNetSpeech, LibriSpeech (split not checked) | CER/WER | Mean emission delay against the forced-alignment word endpoint, plus flicker; 240 ms chunks | https://arxiv.org/abs/2609.20121 |
| StreamSpeech | fr (ASR) | CVSS-C Fr→En test | WER | AL in ms via SimulEval, about 109–758 ms | https://arxiv.org/abs/2406.03049 |

## (2) Which sets are standard, and which you need for each baseline

**English**
- **LibriSpeech test-clean/test-other** is near-universal: 11 of the 13 streaming works above use it. DSM also measures word latency on test-clean using MFA timestamps.
- **The 8 Open ASR Leaderboard sets** (AMI, Earnings22, GigaSpeech, LS clean/other, SPGI, TED-LIUM v3, VoxPopuli), scored with the Whisper normalizer, are needed to compare with Nemotron-EN, Kyutai DSM, and the Whisper-Streaming and SeamlessStreaming reruns that DSM reports. Voxtral Realtime uses a variant without AMI.
- **FLEURS `en_us` test** is needed for Nemotron 3.5, Qwen3-ASR streaming and Voxtral Realtime.

**German**
- **FLEURS `de_de` test** is the only German set with published *streaming* numbers from more than one strong baseline: Nemotron 3.5 (5 chunk sizes) and Voxtral Realtime (4 delays). Qwen3-ASR, Canary and the leaderboard's multilingual track report it offline.
- Secondary options:
  - **Common Voice de**: Voxtral Realtime streaming, Qwen3-ASR offline.
  - **MLS de**: Qwen3-ASR offline only.
  - **CoVoST2 de**: Canary and the leaderboard.
  - **MLC-SLM de**, which is conversational: Qwen3-ASR offline, VibeVoice streaming.
  - **ESIC de**: Whisper-Streaming only, on interpreter speech.

**Korean**
- **FLEURS `ko_kr` test** is the de-facto set: Nemotron 3.5 streaming (CER), Voxtral Realtime streaming (WER), Qwen3-ASR offline (CER).
- Common Voice ko and MLC-SLM ko are secondary.
- **None of the covered baselines report KsponSpeech or Zeroth-Korean.** KsponSpeech eval_clean/eval_other is the Korean-community standard (the ESPnet recipe has 3000 utterances each), so it adds Korean credibility but no baseline comparison.

**Per baseline**
- **Nemotron:** FLEURS en/de/ko plus the 8 English sets. Its chunk grid of 80/160/320/560/1120 ms matches your 80 ms encoder exactly.
- **Kyutai STT:** English only. Use the 8 sets plus long-form, at 0.5 s and 2.5 s delays.
- **Whisper-Streaming:** to get German numbers on a common set you have to run it yourself; DSM's English rerun covers English.
- **SeamlessStreaming:** the paper has no per-language ASR numbers, so you would rerun it.
- **Qwen3-ASR:** streaming numbers exist only for LS and Fleurs-en, so you would have to rerun its streaming mode for de/ko.
- **Conversational setting:** MLC-SLM (de/ko/en, two-speaker) and AMI are the only conversational sets that streaming baselines report.

**Latency conventions to report**
- The configured chunk size or target delay.
- Measured word emission delay against forced-alignment timestamps, as DSM, Whisper-Streaming and alignment-path distillation do.
- RTF or RTFx, and optionally TTFT.

## (3) Access and licence notes for the recommended sets

| Set | Licence / access |
|---|---|
| FLEURS | CC-BY-4.0, ungated (`google/fleurs` on HF) |
| LibriSpeech | CC BY 4.0 (OpenSLR SLR12) |
| MLS | CC BY 4.0 (OpenSLR SLR94) |
| AMI | CC-BY-4.0 |
| Earnings22 | CC-BY-SA-4.0 |
| GigaSpeech | apache-2.0, but HF access is gated |
| SPGISpeech | "User Agreement"; gated on HF |
| TED-LIUM v3 | CC-BY-NC-ND 3.0 |
| VoxPopuli | CC0 |
| CoVoST-2 | CC-BY-NC-4.0 |
| Common Voice | Since October 2025 only distributed through Mozilla Data Collective. CC0 per an HF mirror card, not an official Mozilla page |
| Zeroth-Korean | CC BY 4.0; test set is 457 utterances, 1.2 h, 10 speakers (OpenSLR SLR40) |
| KsponSpeech | AI Hub states "only Korean nationals can apply" |
| MLC-SLM | Eval transcripts are on HF (`bsmu/MLC-SLM-Eval`, cc-by-sa-4.0, text only). Audio needs Nexdata registration and a data-use agreement limited to the challenge |
| AA cleaned sets | VoxPopuli-Cleaned-AA and Earnings22-Cleaned-AA are apache-2.0 |
| Open ASR test bundle | `hf-audio/esb-datasets-test-only-sorted`; no licence field |

Sources for these licences:
- Open ASR Leaderboard paper, Table 1 (AMI, Earnings22, GigaSpeech, SPGISpeech, TED-LIUM v3, VoxPopuli, CoVoST-2): https://arxiv.org/abs/2510.06961
- FLEURS: https://huggingface.co/datasets/google/fleurs
- LibriSpeech: https://www.openslr.org/12/
- MLS: https://www.openslr.org/94/
- Zeroth-Korean: https://www.openslr.org/40/
- Common Voice: https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0 and https://huggingface.co/datasets/fsicoli/common_voice_17_0
- KsponSpeech: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=123 and the ESPnet recipe https://github.com/espnet/espnet/tree/master/egs2/ksponspeech/asr1
- MLC-SLM: https://huggingface.co/datasets/bsmu/MLC-SLM-Eval and https://www.nexdata.ai/competition/mlc-slm
- AA sets: https://huggingface.co/datasets/ArtificialAnalysis/VoxPopuli-Cleaned-AA and https://huggingface.co/datasets/ArtificialAnalysis/Earnings22-Cleaned-AA
- GigaSpeech and SPGISpeech gating: the HF API entries for `speechcolab/gigaspeech` and `kensho/spgispeech`

## (4) Uncertainties

- **Korean scoring differs.** Nemotron 3.5, Qwen3-ASR, MLC-SLM and VibeVoice use CER; Voxtral Realtime uses WER. Their published numbers are not directly comparable.
- **Normalizers are often missing.** Nemotron 3.5, Qwen3-ASR and Voxtral Realtime don't name one.
- **Common Voice versions are unstated** in Qwen3-ASR, Voxtral Realtime and the cache-aware paper.
- **Nemotron 3.5 lists MCV and MLS** as evaluation sets but publishes only FLEURS numbers.
- **SeamlessStreaming per-language ASR results** are on GitHub and I didn't check them. The split in its Table 29 isn't stated.
- **Not extracted:** DSM's per-delay latency values are only in a figure, and I didn't take Qwen3-ASR's MLC-SLM eval subset. VibeVoice's MLC numbers come from recordings capped at 480 s.
- **Not verified:** the AMI condition (IHM or SDM) in the leaderboard bundle, the TED-LIUM OpenSLR page (it didn't load; the licence comes from the leaderboard paper), the KsponSpeech paper (MDPI returned 403), the ESIC licence, and the LibriSpeech split in the alignment-path distillation paper.
- **MLC-SLM terms may block paper use.** The Nexdata page restricts use to the challenge, while the HF transcripts are CC-BY-SA. This needs clarifying before you use it.
- **Tooling:** WebSearch wasn't used and Semantic Scholar was rate-limited. The "recent papers" are the arXiv-API hits I opened, not an exhaustive survey.

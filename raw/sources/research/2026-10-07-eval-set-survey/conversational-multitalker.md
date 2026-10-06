# Evaluation-set survey: conversational-multitalker (2026-10-07, research agent report, verbatim)

I couldn't find a public test set of two-person conversations in two languages with speaker-attributed translations. JSTAR's RealConv (Spanish–English) is the closest match, and it is internal. For ASR plus diarization, the standard two-speaker, single-channel comparisons are CALLHOME part 2 (2-speaker subset), CH109 and the Fisher test split. Everything below comes from fetched arXiv, GitHub, Hugging Face and LDC pages; I modified no repository files.

## 1. What each work evaluates on

| Work | Task | Eval sets (splits, speakers, channels) | Streaming? | Metrics and key numbers | Link |
|---|---|---|---|---|---|
| SOT (kanda2020serialized) | Multi-talker ASR | LibriSpeech dev-clean/test-clean mixed into 1-, 2- and 3-speaker sets. Single-channel, simulated. | Offline | WER minimised over hypothesis order, plus speaker-counting accuracy. 2-speaker test WER 11.2% (1024-dim with SAA). | https://arxiv.org/abs/2003.12687 |
| LibriSpeechMix (data) | Benchmark | dev-clean and test-clean, each as 1mix/2mix/3mix. Partially overlapped. "Each utterance … is used exactly N times in the N-speaker set." | n/a | WER | https://github.com/NaoyukiKanda/LibriSpeechMix |
| SA-ASR Transformer (kanda2021end) | Speaker-attributed ASR | LibriSpeechMix with 8 speaker profiles. LibriCSS first channel only, 8 speakers per recording, segmented with WebRTC VAD. | Offline | SER, WER, SA-WER; cpWER. LibriCSS cpWER 11.9% with profiles; 13.3% / 16.3% with clustering (oracle / estimated speaker count). | https://arxiv.org/abs/2104.02128 |
| t-SOT (kanda2022streaming) | Streaming multi-talker ASR, speaker-agnostic | LibriSpeechMix 1-speaker and 2-speaker test. LibriCSS first channel, sessions 1–9 (session 0 is dev), continuous-input setting on 1–2 min pre-segmented audio. | Yes, 40–2560 ms algorithmic latency | Permutation WER; speaker-agnostic WER (SAgWER) on LibriCSS. LibriSpeechMix 3.3% / 4.4% (1 / 2 speakers) at 2560 ms. LibriCSS average 9.0% at 160 ms, 7.6% at 2560 ms. | https://arxiv.org/abs/2202.00842 |
| t-SOT speaker-attributed ASR (kanda2022streamingspeaker) | Streaming speaker-attributed ASR | LibriSpeechMix with 8 profiles. LibriCSS first channel. | Yes | SA-WER; cpWER. LibriCSS cpWER 12.0% (ASR + speaker ID) and 12.4% (ASR + diarization, oracle speaker count). | https://arxiv.org/abs/2203.16685 |
| LibriCSS (data) | Benchmark | 10 sessions × six 10-min mini-sessions; overlap 0S/0L/10–40%; 8 speakers per mini-session; 7-ch array. Session 0 is dev. | n/a | WER via asclite | https://arxiv.org/abs/2001.11482 , https://github.com/chenzhuo1011/libri_css |
| Sortformer (park2025sortformer) | Diarization + multi-speaker ASR | DIHARD3 eval (≤4 speakers, collar 0); CALLHOME part 2 (collar 0.25 s); CH109 (2 speakers, collar 0.25 s). For ASR: AMI test and CH109 cut into 90 s segments with ≤4 speakers, and LibriSpeechMix. Overlap included in all DER. | Offline | DER; cpWER. CH109 cpWER 21.45; AMI cpWER 26.71; LibriSpeechMix 2mix 4.61. | https://arxiv.org/html/2409.06656 |
| Streaming Sortformer (medennikov2025streaming) | Streaming diarization, max 4 speakers | DIHARD III eval (collar 0); CALLHOME part 2 by speaker count (Kaldi part1/part2 split, post-processing tuned on part 1); CH109. Collar 0.25 s, overlap included. | Yes: 0.32 / 1.04 / 10 s | DER on CALLHOME part 2, 2-speaker subset, without / with post-processing: 8.60 / 6.86 at 0.32 s; 7.35 / 6.43 at 1.04 s; 6.80 / 6.06 at 10 s. CH109 at 1.04 s: 5.59 / 5.09. | https://arxiv.org/html/2507.18446 |
| NVIDIA multitalker-parakeet-streaming (Wang et al., arXiv 2506.22646) | Streaming multi-talker ASR on top of Streaming Sortformer | Model card: AMI IHM/SDM, CH109, Mixer 6. Paper: LibriSpeechMix 1/2/3-mix, CH109. | Yes, 80 ms to 1.12 s | cpWER at 1.12 s (card): CH109 15.81, Mixer 6 23.81, AMI SDM 37.44. Paper: CH109 26.21 at 1120 ms; LibriSpeechMix 2-mix 5.6 at 560 ms. | https://huggingface.co/nvidia/multitalker-parakeet-streaming-0.6b-v1 , https://arxiv.org/html/2506.22646 |
| pyannote 3.1 / community-1 | Diarization | AISHELL-4, AliMeeting, AMI IHM/SDM, CALLHOME part 2, DIHARD3, VoxConverse and others. | Offline (no streaming claim on the cards) | DER with no collar, overlap scored, fully automatic. CALLHOME part 2: 28.5 (3.1), 26.7 (community-1). AMI SDM: 22.7 / 19.9. | https://github.com/pyannote/pyannote-audio , https://huggingface.co/pyannote/speaker-diarization-3.1 |
| diart (Coria et al., ASRU 2021) | Online diarization built on pyannote | AMI, DIHARD, VoxConverse | Yes, 500 ms to 5 s | DER (numbers not captured) | https://arxiv.org/abs/2109.06483 , https://github.com/juanmc2005/diart |
| DiarizationLM (wang2024diarizationlm) | LLM post-processing of ASR + diarization | Fisher test (172 conversations, 28.7 h); Callhome American English test (20 conversations, 1.7 h). Both 2-speaker telephone. | Offline | WER, WDER, cpWER, speaker-count error. After PaLM 2 correction: Fisher WDER 2.37 / cpWER 16.93; Callhome 4.25 / 20.22. | https://arxiv.org/html/2401.03506 |
| JEDIS-LLM (arXiv 2511.16046) | Joint ASR + diarization speech-LLM | AMI test (IHM-Mix, ≤4 speakers); CH109; Fisher test | Streamable, ~10 s chunks | WDER, cpWER, SA-WER. Long-form: CH109 test 1.73 / 18.20; Fisher test 2.05 / 15.88. | https://arxiv.org/html/2511.16046 |
| SpeakerLM (arXiv 2508.06372) | Joint ASR + diarization speech-LLM | AliMeeting eval (2–4 speakers); AISHELL-4 eval; AISHELL-5 eval (2 speakers). First far-field channel. | Offline, 40–50 s clips | CER, cpCER, saCER | https://arxiv.org/html/2508.06372 |
| TagSpeech (arXiv 2601.06896) | Joint ASR + diarization | AMI SDM; AliMeeting first channel; 1–4 speakers | Offline | DER (collar 0, with 0.25 s in appendix), cpWER | https://arxiv.org/html/2601.06896 |
| VibeVoice-ASR-Streaming (arXiv 2609.02812) | Streaming speaker-attributed ASR | AISHELL-4, AliMeeting, AMI IHM/SDM, MLC-SLM (9 languages); sessions capped at 480 s | Yes, about 2 s expected speaker latency | WER/CER, cpWER/cpCER. Compared against Gemini, GPT, ElevenLabs, Azure and Google streaming APIs. | https://arxiv.org/html/2609.02812 |
| NAR-LLM timestamps and speaker attribution (arXiv 2609.15218) | Speaker attribution | Fisher, CallHome English, AMI SDM (2 min and 5 min segmentations) | Offline | WDER, cpWER | https://arxiv.org/html/2609.15218 |
| MLC-SLM challenge (arXiv 2509.13785) | Joint diarization + ASR (Task 2) | About 20-min two-speaker conversations, 11 languages, recorded on phones at 16 kHz. Dev, Eval-1 and Eval-2 are about 32 h each. | Offline challenge | tcpWER/tcpCER (tcpMER) via MeetEval. Best Task 2 entry: 16.53 on Eval-2. | https://arxiv.org/html/2509.13785 , https://huggingface.co/datasets/bsmu/MLC-SLM-Eval |
| NOTSOFAR-1 (vinnikov2024notsofar) | Distant meeting transcription | About 315 meetings, 4–8 speakers; single-channel (commercial devices) and multi-channel tracks | Not required | Speaker-attributed tcpWER (ranking); tcORC-WER (supplementary) | https://arxiv.org/html/2401.08887 , https://github.com/microsoft/NOTSOFAR1-Challenge |
| CHiME-6 / 7 / 8 DASR | Distant ASR + diarization | CHiME-6 and DiPCo (4 speakers); Mixer 6 (2-speaker interviews); NOTSOFAR-1 (4–8). All multi-device. | Offline | CHiME-7: DA-WER (cpWER-style), DER/JER with 0.25 s collar. CHiME-8: macro tcpWER, 5 s collar. | https://arxiv.org/html/2306.13734 , https://arxiv.org/html/2407.16447 , https://arxiv.org/abs/2004.09249 |
| MeetEval (toolkit) | Scoring | n/a | n/a | cpWER, ORC-WER, MIMO-WER, tcpWER, tcORC-WER, DI-cpWER; also wraps md-eval for DER | https://github.com/fgnt/meeteval |
| DiariST (yang2024diarist) | Streaming speech translation + diarization, zh→en | DiariST-AliMeeting: dev 8 sessions / 4 h, test 20 sessions / 10 h, cut into 195 test mini-sessions of 3–6 min. 2–4 speakers. Conditions SDM (first channel), IHM-MIX, IHM-CAT. Human translations for dev/test. | Yes, 1 s chunks | SAgBLEU, SAtBLEU (best speaker permutation), DER with 0.25 s collar. SDM test: SAtBLEU 11.67 vs 9.02 for the cascade. | https://arxiv.org/html/2309.08007 , https://github.com/Mu-Y/DiariST |
| STAC-ST (zuluagagomez2023end) | Speaker-turn-aware conversational speech translation, es→en | Fisher-CALLHOME with the two telephone channels merged to one. Multi-turn multi-speaker splits: Fisher dev/dev2/test about 4.1 h each; CALLHOME dev 3.5 h, test 1.7 h. Human-annotated chunks of up to 30 s. | Offline | BLEU (SacreBLEU, 4 references Fisher / 1 CALLHOME, speaker tokens removed); WER; speaker-change FAR/MDR/F1 at 0.25 s. Fisher test 50.0 BLEU / 23.5 WER; CALLHOME 21.0 / 38.5. | https://arxiv.org/html/2311.00697 , https://github.com/amazon-science/stac-speech-translation |
| JSTAR (moritz2025transcribing) | Joint ASR + ST on smart glasses, SELF/OTHER labels | RealConv: internal Spanish–English real conversations. MC-FLEURS: simulated 5-channel two-talker data. | Yes | SA-WER, BLEU, P50 latency of first/last finalised token | https://arxiv.org/html/2412.15415 |
| Wang et al. 2025 (wang2025streaming) | Speaker change + gender for multi-talker ST, en→de/es/hi/it/ru | 5 real long recordings (up to 8 speakers, origin not disclosed); VoxCeleb for gender | Yes, 1 s chunks | Speaker-change P/R/F1 at ±2 s; gender accuracy; no BLEU reported | https://arxiv.org/html/2502.02683 |

**DER conventions differ and the numbers are not comparable across them:**
- **EEND, Sortformer, Streaming Sortformer on CALLHOME/CH109:** 0.25 s collar, overlap included. Source: https://arxiv.org/abs/2005.09921
- **Kaldi x-vector CALLHOME recipe:** `md-eval -1 -c 0.25`. The `-1` flag scores only single-speaker regions, so overlap is excluded. Sources: https://github.com/kaldi-asr/kaldi/blob/master/egs/callhome_diarization/v2/run.sh , https://raw.githubusercontent.com/nryant/dscore/master/scorelib/md-eval-22.pl
- **DIHARD:** collar 0, plus JER.
- **pyannote:** collar 0, overlap included.

## 2. Recommendation

**Standard and practical for two-speaker, single-channel streaming ASR + diarization**
1. **CALLHOME part 2, 2-speaker subset.** This is NIST SRE 2000 Disc8 (LDC2001S97), 148 test recordings, 8 kHz telephone. It is the most direct comparison with Streaming Sortformer, which has per-latency DER (0.32 / 1.04 / 10 s). Report DER with 0.25 s collar and overlap included. Also report collar 0, because pyannote uses that and only publishes full part 2, so pyannote would have to be re-run on the 2-speaker subset.
2. **CH109.** These are 109 two-speaker sessions from CALLHOME American English (LDC97S42). It supports both DER and cpWER, with streaming baselines from Streaming Sortformer, multitalker-parakeet and the self-speaker-adaptation paper, plus JEDIS-LLM and DiarizationLM.
3. **Fisher English test.** 172 conversations; the split list is at https://github.com/google/speaker-id/blob/master/publications/ScdLoss/eval/fisher.txt. Report WDER and cpWER against DiarizationLM and JEDIS-LLM.
4. **LibriSpeechMix test-clean-2mix.** Free and cheap to run, and t-SOT, Sortformer-MS-Canary and the self-speaker-adaptation paper all report on it, but it is read speech with short two-utterance mixtures. cpWER is the natural metric for our model; SA-WER would require speaker profiles.

**Poor fits because of speaker count or channel setup**
- **LibriCSS:** 8 speakers per mini-session. Only a speaker-agnostic WER comparison with t-SOT is meaningful.
- **AMI SDM** (Full-corpus-ASR test, groups ES2004, IS1009, TS3003, EN2002): meetings with more than two speakers.
- **NOTSOFAR-1:** 4–8 speakers.
- **CHiME Mixer 6:** two speakers, but far-field multi-mic and LDC-licensed.
- **MLC-SLM Eval-2:** two-speaker, 16 kHz phone recordings, multilingual, with tcpWER. It is attractive, but each conversation is in one language, and how the two speakers were recorded is not stated.

**Conversational speech translation as an extra comparison**
- **Fisher-CALLHOME es→en (STAC-ST setup).** Usable if our model supports es→en. It only exercises one translation direction because both speakers speak Spanish. STAC-ST is offline with oracle segments of up to 30 s, so compare BLEU and WER, and optionally speaker-change F1 at 0.25 s, as an offline reference point.
- **DiariST-AliMeeting zh→en (SDM).** Public, streaming, with human references and SAtBLEU. Sessions have 2–4 speakers and are monolingual Mandarin meetings, so it is a stretch for a two-speaker model; filtering to two-speaker sessions would be non-standard.
- **SAgBLEU/SAtBLEU** are also the closest existing metrics to our setting, so it is worth reporting them on our own benchmark.

## 3. Access and licences (as stated)

- **LDC licence required:**
  - Speech: LDC2001S97, LDC97S42, Fisher English (LDC2004S13), Fisher Spanish (LDC2010S01), CALLHOME Spanish (LDC96S35).
  - Translations: LDC2014T23.
  - Mixer 6: LDC2013S03; LDC provided free access only during CHiME.
  - DiarizationLM's README notes Fisher and Callhome need LDC permission.
- **CC BY 4.0:** AMI, NOTSOFAR-1 data, pyannote community-1, Streaming Sortformer v2 weights.
- **CC BY-SA 4.0:** AliMeeting and DiariST-AliMeeting.
- **MIT:** pyannote-audio code, MeetEval, diart.
- **Apache-2.0:** STAC-ST scripts.
- **NVIDIA Open Model License:** multitalker-parakeet.
- **pyannote 3.1:** MIT, but the files are gated behind accepting conditions on Hugging Face.
- **LibriSpeechMix, LibriCSS:** the repos have a LICENSE file, but I did not capture its terms.

## 4. Uncertainties

- **Mono conversion:** CALLHOME American English and Fisher are 2-channel, 8 kHz. None of the papers I read state how they made mono input; only STAC-ST says it merged the channels.
- **CH109 overlap:** CH109 ("Full") may overlap the 20-conversation Callhome test used by DiarizationLM and JEDIS-LLM ("CH109 Test"). I did not verify the overlap.
- **pyannote CALLHOME numbers:** I could not confirm that pyannote's "CALLHOME (part 2)" is the same 250-recording, 2–6 speaker part 2 used by EEND and Sortformer.
- **Sortformer Table 1 header:** the HTML version garbled the column headers (CH109 showed 3 and 4 speakers), so I used Streaming Sortformer's table instead.
- **Model card latency:** I did not capture the latency setting behind the Streaming Sortformer v2 card's 6.57% DER on 2-speaker CALLHOME part 2.
- **MLC-SLM data:**
  - The summary paper says "CC Zero" but the Hugging Face card says CC-BY-SA-4.0.
  - The card says audio is included, but the repo is only 6.55 MB, so audio availability is unclear.
  - The recording setup (shared or separate phones) is not stated.
- **JSTAR:** the SA-WER definition is only cited, and MC-FLEURS is not stated to be released.
- **Wang et al. 2025:** the origin of the 5 test recordings is not disclosed.
- **diart:** I captured no DER numbers or collar convention.
- **Recent speech-LLM papers:** DM-ASR (2604.22467), G-STAR (2603.10468), arXiv 2604.11269 and arXiv 2606.13095 were found in arXiv listings but not read.
- **SLT 2026 SmartGlasses Challenge (2608.12034):** it has a two-person dialogue track but uses 4-channel audio; language and licence were not captured.

# Evaluation-set survey: korean-asr-st (2026-10-07, research agent report, verbatim)

I found standard Korean ASR test sets and a usable set of Korean↔English speech-translation test sets. There is no public Ko↔En simultaneous-translation benchmark; the only system I found reporting Ko↔En streaming results is SeamlessStreaming, which uses FLEURS. I modified nothing in the repo.

## (1) Datasets

| Dataset | Task | Test split | Style | Licence / access | Used by |
|---|---|---|---|---|---|
| **KsponSpeech** (ETRI/AI Hub) | Korean ASR | eval-clean: 3,000 utterances (2.6 h); eval-other: 3,000 (3.8 h). Same 60 speakers, none in train; split by language-model perplexity | spontaneous two-person dialogue, 969 h | AI Hub page says "내국인만 데이터 신청이 가능합니다" (Korean nationals only). AI Hub terms below | Paper baseline CER 7.6/8.5, WER 21.1/25.5, sWER 13.4/15.4; ESPnet CER 8.7/9.4; OWSM v3 also reports Whisper small/medium zero-shot CER (24.0/15.4, 17.6/12.8); K-wav2vec; KoALa-Bench |
| **FLEURS ko_kr / en_us** | Korean ASR; Ko→En and En→Ko speech-to-text translation (n-way parallel via FLoRes sentence ids) | ko_kr: 382; en_us: 647 | read Wikipedia sentences | CC-BY-4.0 | Whisper, SeamlessM4T v2, SeamlessStreaming, Qwen3-ASR, Nemotron 3.5 ASR (numbers below) |
| **Zeroth-Korean** (OpenSLR 40) | Korean ASR | 457 utterances, 1.2 h, 10 speakers | read news (3,000 sentences in total) | CC BY 4.0 | KoALa-Bench; kosp2e re-split it so test holds only unseen sentences |
| **Common Voice ko** | Korean ASR | v17: 339 clips; v27 (2026-09): 579; only 2.56 validated hours | read, crowd-sourced | described as public domain; since Oct 2025 only via Mozilla Data Collective | Qwen3-ASR (version not stated); KoALa-Bench (523 samples); not in Whisper's Common Voice 9 table |
| **ClovaCall** | Korean ASR | 1,084 utterances (1.66 h raw / 0.88 h clean) | phone calls, restaurant reservations | non-commercial research only, application form for academic organisations | K-wav2vec (test CER 6.41) |
| **MLC-SLM** | Korean ASR | not checked | conversational (11 languages, 1,500 h) | challenge registration (not verified) | Qwen3-ASR (ko CER 10.31 / 8.61) |
| **kosp2e** | Ko→En | 2,320 utterances (paper); the README table sums to 2,400 | read/recorded scripts: news, textbook, AI-agent commands, diary | per subset: CC-BY 4.0 / CC-BY-SA 4.0 / CC-BY-NC-SA 4.0 (two academic-only). Audio and English text free; Korean scripts on request | Interspeech 2021 baselines BLEU 21.3 (ASR + Papago) / 18.0 (end-to-end); EACL-Findings 2024 (training) |
| **ETRI EnKoST-C** | En→Ko | tst-COMMON: 2,532 segments (4.03 h); tst-HE: 544 (1.07 h) | TED talks, MuST-C-style alignment, 559 h | ETRI e-PreTX portal (separate English sign-up for foreign users); dataset licence not verified | KoALa-Bench |
| AI Hub **71379** 방송콘텐츠 한국어-영어 통번역 음성 | Ko→En (also Es, Ru) | only Training / Validation released | broadcast speech, 600 h (documentary, entertainment, drama, interview) | Korean nationals only | AI Hub's own validation only |
| AI Hub **71693** 국제 학술대회용 전문분야 한영/영한 통번역 | Ko→En and En→Ko | only Training / Validation in the file tree, though the page describes an 80/10/10 split | academic-conference video; 2,017 h; about 450k sentences per direction | Korean nationals only | — |
| AI Hub **71668** 라이브 스트리밍 영상 영어 통번역 | Ko→En and En→Ko | only Training / Validation | live-streaming | Korean nationals only | — |
| AI Hub **71524** 다국어 통·번역 낭독체 | Ko ↔ En / Ja / Es | only Training / Validation | read book text, 4,107 h, led by HUFS | Korean nationals only | probably the ICASSP 2025 interpretation corpus (my inference from matching sizes) |
| AI Hub **71384** / **71686** | **not English**: Ko↔De/Fr/It broadcast; Ko↔Es/Fr/Ru daily conversation | — | — | Korean nationals only | — |
| CoVoST 2 / MuST-C (all releases) / mTEDx | **no Korean** | — | — | — | covost repo, archived MuST-C pages, OpenSLR 100 |

**AI Hub terms** (usage-policy page):
- You must credit NIA (한국지능정보사회진흥원).
- Organisations or individuals abroad need a separate agreement, and so does taking the data out of Korea (국외 반출).
- No redistribution.
- Commercial sale needs a separate agreement.
- One clause says the data may be used "인공지능 학습모델의 학습용으로만" (only for training AI models).

**Published Korean numbers, and the metric each uses**

| System | Korean ASR | Ko→En | En→Ko |
|---|---|---|---|
| Whisper large-v2 (paper) | FLEURS 14.3, as **WER** (Korean is not split into characters) | FLEURS 21.3 BLEU | — (Whisper translates into English only) |
| Whisper README chart | Korean in italics, i.e. **CER**: Common Voice 15 = 5.2, FLEURS = 3.1 (read from the SVG; probably large-v3) | — | — |
| SeamlessM4T v2 Large | FLEURS 18.19 (**WER**) | 24.11 BLEU | 12.82 BLEU |
| SeamlessStreaming | FLEURS normalised WER 45–47 | 20.8–21.2 BLEU, AL 3.3–3.6 s | 8.3–8.7 BLEU, AL 2.1–2.3 s |
| Qwen3-ASR 0.6B / 1.7B / Flash | **CER**: FLEURS 3.72 / 2.57 / 2.07; Common Voice 8.48 / 5.88 / 3.82; no Korean streaming results | — | — |
| Nemotron 3.5 ASR (streaming, 0.6B) | Korean supported (ko-KR, OpenMDW-1.1 licence). FLEURS **CER** 7.12–7.59 with language given, 7.30–8.31 with auto-detect, across 1.12 s–80 ms chunks | — | — |

Seamless BLEU uses SacreBLEU's default 13a tokeniser for Korean output (character-level only for zh/ja/th/lo/my). I found no dedicated Korean ASR leaderboard: HF's Open ASR Leaderboard multilingual track covers only de/fr/it/es/pt, and AI Hub's leaderboard is for LLMs.

**KsponSpeech normalisation conventions.** Each utterance has a dual transcript, written "(spelling)/(phonetic)", e.g. "(70%)/(칠 십 퍼센트)".
- **Paper and ESPnet:** use the spelling form ("char" = the first alternative). They strip punctuation and noise tags but keep filler and repeated words, removing only the "/" and "+" markers. Paper CER counts spaces. It also proposes sWER, which fixes hypothesis spacing to match the reference.
- **KoSpeech:** defaults to the phonetic form. K-wav2vec reports phonetic results.
- **KoALa-Bench:** CER with all spaces and punctuation removed, keeping only Hangul and Arabic digits.

Numbers under different conventions are not comparable.

## (2) Recommendations

- **Korean ASR:**
  - Primary: KsponSpeech eval-clean and eval-other (the de-facto standard, spontaneous) plus FLEURS ko_kr test (the only set Whisper, Seamless, Qwen3-ASR and Nemotron all report).
  - Reference: use the spelling transcript, since multilingual models output digits and standard spelling.
  - Metrics: report space-free CER as primary and with-space CER for comparison with the KsponSpeech paper. On FLEURS, also report Whisper-normalised WER so Whisper and SeamlessM4T numbers are comparable.
  - Optional: Zeroth (CC BY, read news).
- **Ko→En translation:** FLEURS ko→en (direct comparison with Whisper, SeamlessM4T v2 and SeamlessStreaming) plus the kosp2e test set (open, multi-domain).
- **En→Ko translation:** FLEURS en→ko plus EnKoST-C tst-COMMON. EnKoST-C is TED built the MuST-C way, so it mirrors whatever MuST-C En→De test you use. State the BLEU tokeniser explicitly (13a vs character vs ko-mecab) and add chrF++/COMET.
- **Simultaneous:** use FLEURS for comparability with SeamlessStreaming (AL/LAAL). For long-form, EnKoST-C TED talks work for En→Ko. For Ko→En, an AI Hub validation subset (71693 or 71379) is possible, but those numbers can't be compared with anyone else's.

## (3) Uncertainties

- **AI Hub sets you named:** 71384 and 71686 contain no English, so they don't serve Ko↔En.
- **AI Hub access:** all AI Hub sets, KsponSpeech included, show the Korean-nationals-only notice. It is unclear whether evaluation counts as "training use".
  - Before staging AI Hub data onto mxc, confirm the Azure Blob / mxc region is in Korea; otherwise the 국외 반출 clause applies.
  - Foreign groups (CMU's OWSM, ESPnet) have published KsponSpeech results.
- **EnKoST-C:** dataset licence and access terms are unverified; the ETRI Journal and e-PreTX pages were blocked or needed login.
- **Model attribution:** the Whisper README values are read from SVG labels, and attributing them to large-v3 is my inference.
- **71524 = ICASSP 2025 corpus:** inferred, not confirmed.
- **Benchmark versions:** Qwen3-ASR doesn't say which Common Voice version it used.
- **Nemotron metric label:** the card's metadata labels the Korean score WER, while its text says CER.
- **FLEURS pairing:** the number of usable Ko↔En pairs (sentence ids shared across languages) wasn't computed.
- **MLC-SLM:** licence not checked.

## Key URLs
- KsponSpeech paper (MDPI blocks direct fetches; read via archive.org): https://doi.org/10.3390/app10196936 · AI Hub: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=123
- ESPnet recipe: https://github.com/espnet/espnet/tree/master/egs2/ksponspeech/asr1
- OWSM: https://arxiv.org/abs/2309.13876 · K-wav2vec: https://arxiv.org/abs/2110.05172 · KoSpeech: https://arxiv.org/abs/2009.03092
- Whisper: https://arxiv.org/abs/2212.04356 · chart: https://github.com/openai/whisper (language-breakdown.svg)
- Seamless: https://arxiv.org/abs/2312.05187 · metrics: https://dl.fbaipublicfiles.com/seamless/metrics/seamlessM4T_large_v2.zip and https://dl.fbaipublicfiles.com/seamless/metrics/streaming/seamless_streaming.zip
- Qwen3-ASR: https://arxiv.org/html/2601.21337 · Nemotron: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b
- FLEURS: https://huggingface.co/datasets/google/fleurs · Zeroth: https://www.openslr.org/40/ · ClovaCall: https://github.com/clovaai/ClovaCall
- Common Voice stats: https://github.com/common-voice/cv-dataset
- kosp2e: https://arxiv.org/abs/2107.02875 and https://github.com/warnikchow/kosp2e
- EnKoST-C: https://doi.org/10.4218/etrij.2021-0336 · KoALa-Bench: https://arxiv.org/abs/2604.19782 · ETRI portal: https://epretx.etri.re.kr
- AI Hub pages: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71379 (likewise 71693, 71668, 71524, 71384, 71686) · file trees: https://api.aihub.or.kr/info/71693.do · policy: https://aihub.or.kr/intrcn/guid/usagepolicy.do
- Open ASR Leaderboard: https://arxiv.org/abs/2510.06961 · CoVoST: https://github.com/facebookresearch/covost
- ICASSP 2025 corpus: https://doi.org/10.1109/icassp49660.2025.10887990 · CLASI (zh↔en only): https://arxiv.org/abs/2407.21646

# Evaluation-set survey: simul-s2tt (2026-10-07, research agent report, verbatim)

I covered every work you listed plus four recent LLM-based SimulST papers, reading the full text of each (findings papers, system papers, arXiv preprints), not just the abstracts. Every number below was read from a source. I modified no files in the repo; scratch PDFs and text extractions are in `/private/tmp/claude-501/-Users-taesookim-Desktop-VAPKT/49169b81-fc5f-478d-acc2-17601e549321/scratchpad/simulst/`.

**Bottom line:**
- **En→De:** follow the IWSLT 2026 protocol (long-form MCIF, XCOMET-XL, LongYAAL, low/high regimes) plus MuST-C tst-COMMON and FLEURS for backward comparison.
- **De→En:** there is no IWSLT track; use FLEURS, CoVoST 2 / CVSS-C, and Europarl-ST plus Audio-NTREX-4L.
- **Korean:** no IWSLT set, and none of MuST-C, ACL 60/60, MCIF or CoVoST 2 covers it. FLEURS (both directions) and Kosp2e (Ko→En) are the only public options.

## 1. Evaluation table

CU = computation-unaware, CA = computation-aware latency.

| Work | Directions | Eval sets (splits) | Input | Quality metrics | Latency metrics + regimes (as reported) | Link |
|---|---|---|---|---|---|---|
| IWSLT 2023 Simul | En→De/Zh/Ja | Blind "Common TED talks" test, plus Non-Native test (En→De). Latency qualified on MuST-C v2.0 tst-COMMON | Segmented | BLEU (ranking), BLASER for speech output, human eval | AL, LAAL, ATD, AP, DAL, both CU and CA. One constraint: AL ≤ 2 s. Best En→De: HW-TSC 29.63 BLEU, LAAL 2.26 (CA 3.93) | iwslt.org/2023/simultaneous; aclanthology.org/2023.iwslt-1.1 |
| IWSLT 2024 Simul | En→De/Zh/Ja, Cs→En (new) | En→X: Common TED talks (blind). Cs→En: dev = ParCzech 3.0 + ELITR; test = MockConf (not released). Latency on MuST-C tst-COMMON | Segmented | BLEU (ranking); ASR-BLEU for speech output | AL ≤ 2 s; LAAL/AL/AP/DAL/ATD reported CU and CA, not used for ranking. Best En→De: HW-TSC 26.39 BLEU, LAAL 2.17 (CA 4.19) | iwslt.org/2024/simultaneous; aclanthology.org/2024.iwslt-1.1 |
| IWSLT 2025 Simul (abdulmumin2025findings) | En→De/Zh/Ja, Cs→En | En→X: dev = ACL 60/60; test = IWSLT25Instruct ACL talks (the web page calls it "MCIF"), plus Accented English test (En→De). Cs→En: ParCzech + ELITR dev; ParCzech + non-native A2-exam test | Unsegmented (long-form) | Ranking by BLEU; COMET (Rei et al. 2022a); human Continuous Rating | Non-CA StreamLAAL. En-De and Cs-En: low 0–2 s, high 2–4 s. En-Zh: 2.5 / 4 s. En-Ja: 3.5 / 5 s. Regime fixed on dev. En→De best high: CUNI 35.25 BLEU / 0.790 COMET / 3.32 s. Best low: CMU 22.63 / 0.697 / 1.47 s (CA 1.81) | iwslt.org/2025/simultaneous; aclanthology.org/2025.iwslt-1.44 |
| IWSLT 2026 Simul (adelani2026speech) | En→De/Zh/It, Cs→En | En→X: dev = MCIF; main test = ACL Talks (up to 2.5 h; Extra-Context sub-track gets paper PDFs); optional Bloomberg (~2 h) and YODAS (5 × 10–30 min). Cs→En: Chamber of Deputies dev; "Media and Ukraine" conference test | Unsegmented | Primary XCOMET-XL (Unbabel/XCOMET-XL); also SacreBLEU and chrF. Computed with OmniSTEval after SoftSegmenter resegmentation | Primary CU LongYAAL; also StreamLAAL (mwerSegmenter). Low 0–2 s, high 2–4 s for all pairs, measured on dev. Hardware: one H100 80 GB. En→De ACL test best: NeMo high 0.93 / 45.01 BLEU / 4.7 s; low 0.92 / 44.94 / 2.6 s. MCIF dev low: 0.93 / 37.91 / 2.0 s | iwslt.org/2026/simultaneous; aclanthology.org/2026.iwslt-1.39 |
| StreamSpeech (zhang2024streamspeech) | Fr/Es/De→En | CVSS-C test (built on CoVoST 2) | Segmented (sentence) | ASR-BLEU (SacreBLEU) and BLASER 2.0. Simul-S2TT shown only as a figure (Fr→En) | SimulEval AL (ms), plus AP/DAL/Start/EndOffset/LAAL/ATD, CU and CA (RTX 3090). Chunk sizes 320 ms to 10 s; e.g. De→En S2ST at 320 ms: AL 1688 ms. "AL ≈ 2000 ms" for good S2TT | aclanthology.org/2024.acl-long.485 |
| InfiniSST (ouyang2025infinisst) | En→Es (v1), En→De (v1), En→Zh (v2) | MuST-C tst-COMMON as 27 complete TED talks (3–23 min) | Unsegmented | SacreBLEU; COMET = mean of XCOMET-XL and XCOMET-XXL | StreamLAAL, StreamLAAL_CA, RTF (LAAL for segmented baselines). Multiplier m = 1–5; CA axis ~1.5–4.5 s | aclanthology.org/2025.findings-acl.157 |
| CMU IWSLT25 (ouyang2025cmu) | En→De/Zh | ACL 60/60 dev, plus the IWSLT25 test | Unsegmented | BLEU | StreamLAAL / StreamLAAL_CA. En-De dev: 25.1 BLEU, 1.689 s / CA 2.306 s. Low regime only | aclanthology.org/2025.iwslt-1.31 |
| AlignAtt (papi2023alignatt) | En→de,es,fr,it,nl,pt,ro,ru | MuST-C v1.0 tst-COMMON | Segmented | sacreBLEU 1.5.1, 13a | CA LAAL (results capped at LAAL_max 3.5 s; plots ~2–3.5 s); K80 GPU | arxiv.org/abs/2305.11408 |
| EDAtt (papi2023attention) | En→De, En→Es | MuST-C tst-COMMON | Segmented | sacreBLEU 1.5.1, 13a | AL and AL_CA (plots 0.5–5 s); LAAL/DAL in appendix | aclanthology.org/2023.acl-long.745 |
| StreamAtt (papi2024streamatt) | 8 MuST-C v1.0 pairs | tst-COMMON full talks (StreamST); segmented for the AlignAtt comparison | Both | sacreBLEU 2.3.1, 13a | Introduces StreamLAAL (mwerSegmenter), NCA and CA. 8-language average: 1.42–2.30 s NCA (CA 2.84–3.62) at 22.3–25.6 BLEU | aclanthology.org/2024.acl-long.202 |
| papi2025how (TACL) | survey of 110 papers | – | – | – | 81.8% of papers use pre-segmented input; 91.8% don't say they assume gold segmentation. Recommends: always report CU latency (CA optionally, same hardware); state input type; use at least automatic segmentation; build unbounded-speech evaluation | aclanthology.org/2025.tacl-1.14 |
| simulstream (gaido2025simulstream) | MuST-C 8 pairs; MCIF En→De/It/Zh | Whole talks | Unsegmented | BLEU and COMET (default wmt22-comet-da) after mweralign | StreamLAAL / StreamLAAL_CA, NE (flicker), RTF. Character-level latency for Zh/Ja/Ko. Deliberately avoids CoVoST 2 and FLEURS (sentence-level) | arxiv.org/abs/2512.17648 |
| LongYAAL / OmniSTEval (Polák et al.) | meta-evaluation | IWSLT 2022–25 logs; ACL 60/60; IWSLT25 test | Both | – | YAAL for short-form; LongYAAL + SoftSegmenter for long-form; StreamLAAL is least accurate (82%). Recommends long-form evaluation | arxiv.org/abs/2509.17349 |
| SeamlessStreaming (seamless2023seamless) | X→eng (101), eng→X (87, includes kor) | FLEURS test | Segmented, start/end silence removed | sacreBLEU 13a (char tokenization only for cmn/jpn/tha/lao/mya; Korean uses 13a); ASR-BLEU for speech | AL and LAAL in seconds over SentencePiece tokens; Ending Offset for speech. Threshold 0.4–0.7. Averages: X-eng AL 1.59–1.84; eng-X AL 1.91–2.10 | arxiv.org/abs/2312.05187 |
| Hibiki (labiausse2025highfidelity) | Fr→En | CVSS-C Fr-En test (short); Audio-NTREX (long, 10 h, ~50 s/utterance, human-read) | Both | Text BLEU, ASR-BLEU (Whisper-medium, normalized) | End Offset and LAAL on the generated speech. Hibiki: CVSS 2.9 / 3.4 s; Audio-NTREX 2.7 / 5.0 s | arxiv.org/abs/2502.03382 |
| Hibiki-Zero (labiausse2026simultaneous) | Fr/Es/Pt/De→En (+It adaptation) | Europarl-ST filtered 2–20 s (1024 test samples per language); Audio-NTREX-4L (TTS, ~45 s/sample, 1800-sample test split) | Both | Text BLEU, ASR-BLEU, ASR-COMET (XCOMET-XL) | End Offset and LAAL (speech output). De→En short: 28.7 BLEU, LAAL 2.8; long: 29.1 BLEU, LAAL 5.9. Seamless De long: LAAL 7.3 | arxiv.org/abs/2602.11072 |
| CLASI (cheng2024towards) | Zh↔En | RealSI (10 × ~5 min per direction); BSTC, CoVoST2 zh-en, MuST-C en-zh, GigaST | Long-form and sentence-level | SacreBLEU, BLEURT, COMET (doc and sentence level); human VIP | AL, LAAL, FLAL. CoVoST2 zh-en: AL 2.63 / LAAL 2.83 s | arxiv.org/abs/2407.21646 |
| Seed LiveInterpret 2.0 (cheng2025seed) | Zh↔En | RealSI; unnamed public + proprietary sentence-level sets | Both | VIP/SVIP (human), BLEURT, COMET | AL, LAAL, FLAL. S2T zh-en RealSI: AL 2.58, FLAL 2.37 | arxiv.org/abs/2507.17527 |
| StreamUni (guo2025streamuni) | Fr→En, En→Zh (CoVoST2); En→De/Es (MuST-C) | Test sets as listed | SimulST: segmented. StreamST: MuST-C full talks | SacreBLEU, COMET | AL/LAAL (SimulEval); StreamLAAL via mwerSegmenter; chunks 320/640 ms | arxiv.org/abs/2507.07803 |
| AlignAtt4LLM = CUNI-ALIGN (fuxa2026alignatt4llm) | En→De/It/Zh | MCIF dev (21 talks, ~2.1 h); IWSLT26 tests | Unsegmented | BLEU, chrF, XCOMET-XL | LongYAAL CU and CA. En-De dev: low 2.00 s CU (0.875 XCOMET); high 3.53 s (0.902) | aclanthology.org/2026.iwslt-1.32 |
| CUNI-POCKET (CUNI IWSLT 2026) | Cs→En, En→De/It | MCIF dev; IWSLT26 Cs-En dev | Unsegmented | BLEU, chrF, XCOMET-XL | CU LongYAAL, low < 2 s / high < 4 s. En-De high: 31.73 BLEU / 0.8776 / 3761 ms | aclanthology.org/2026.iwslt-1.22 |
| EASiST (2025) | En→De/Es | MuST-C v1 tst-COMMON; Europarl-ST En→De/Es (out-of-domain) | Segmented | SacreBLEU | LAAL and LAAL-CA | arxiv.org/abs/2504.11809 |
| SimulMEGA (2025) | 6 languages many-to-many | CoVoST2 (5 X→En, 2 En→X); FLEURS (30 pairs) | Segmented | BLEU | AL, LAAL | arxiv.org/abs/2509.01200 |
| HPO (CMU/NVIDIA, 2026) | En→Zh/De/Ja | ACL 60/60 dev (long-form); RealSI En→Zh | Unsegmented | BLEU, BLEURT-20, XCOMET-XXL, MetricX-24, LLM judge | StreamLAAL, but using the SEGALE segmenter instead of mwerSegmenter; 10 s penalty for unaligned sentences | arxiv.org/abs/2604.21045 |
| RASST (2026) | En→Zh/De/Ja | ACL 60/60 dev (unsegmented); ESO oncology test | Unsegmented | SacreBLEU, terminology accuracy | StreamLAAL ("following IWSLT 2025") | arxiv.org/abs/2601.22777 |

## 2. Recommendations

**En→De (main)**
- **IWSLT-comparable (primary):** long-form MCIF En→De, the IWSLT 2026 dev set (CC BY 4.0, huggingface.co/datasets/FBK-MT/MCIF).
  - Add the IWSLT26 ACL test if its references are obtainable, and ACL 60/60 dev, which CMU 2025, HPO and RASST use.
  - Evaluate with OmniSTEval (SoftSegmenter): XCOMET-XL primary, plus BLEU and chrF; CU LongYAAL primary, plus CA LongYAAL and StreamLAAL.
  - Report one system in each regime: low (≤ 2 s) and high (2–4 s).
- **Literature-comparable:** MuST-C v1.0 En-De tst-COMMON in two forms.
  - Full talks (StreamLAAL CU/CA, BLEU, COMET) to compare with StreamAtt, InfiniSST and simulstream.
  - Gold-segmented (SimulEval AL and LAAL, sacreBLEU 13a) to compare with AlignAtt, EDAtt, EASiST and StreamUni.
- **SeamlessStreaming-comparable:** FLEURS en_us→de_de test. Its released per-direction metrics show threshold 0.4–0.7 → AL 1.87–2.05 s, LAAL 1.99–2.16 s, BLEU 29.8–30.2 (dl.fbaipublicfiles.com/seamless/metrics/streaming/seamless_streaming.zip).

**De→En (no IWSLT track)**
- **FLEURS de→en:** SeamlessStreaming gets AL 1.68–1.85 s, LAAL 1.90–2.05 s, BLEU 34.0–34.4.
- **CoVoST 2 / CVSS-C De→En test:** the StreamSpeech family. StreamSpeech only gives S2ST numbers for De→En, so an S2TT comparison means re-running its released checkpoint.
- **Europarl-ST de→en (short-form) and Audio-NTREX-4L de→en (long-form, TTS):** Hibiki-Zero. Its LAAL is measured on the output speech, so report text-side LAAL and End Offset separately.

**En→Ko and Ko→En**
- **FLEURS en_us↔ko_kr:** the only standard set a major streaming system has reported (SeamlessStreaming).
  - En→Ko: AL 2.06–2.30 s, LAAL 2.30–2.53 s, BLEU13a 8.3–8.7. The low BLEU reflects 13a tokenization on Korean.
  - Ko→En: AL 3.26–3.58 s, LAAL 3.44–3.77 s, BLEU 20.9–21.2.
- **Kosp2e test (Ko→En, sentence-level):** 461 + 512 + 800 + 627 test utterances.
- **Korean-target scoring:** make XCOMET-XL primary. Report BLEU with sacreBLEU `ko-mecab`, which is the default when the target language is ko, and also 13a for Seamless comparability.
- **Latency unit:** state it explicitly. simulstream uses characters for Korean; Seamless used SentencePiece tokens.
- **Not from the literature:** two of my own suggestions with no precedent. One is a pseudo-long-form Korean set made by concatenating FLEURS utterances per FLoRes document. The other is reusing IWSLT 2025's En→Ja regimes (low 0–3.5 s, high 3.5–5 s) as a second Korean setting, since Korean, like Japanese, is verb-final.

**Latency reporting in general**
- Always report CU latency, and CA latency with the hardware stated (papi2025how).
- Long-form: LongYAAL, plus StreamLAAL for comparison with IWSLT 2025, InfiniSST and StreamAtt.
- Short-form: AL and LAAL, plus YAAL.
- Plot several operating points per system rather than a single one.

## 3. Licences, as stated on official pages

| Dataset | Licence | Source |
|---|---|---|
| MuST-C | CC BY-NC-ND 4.0 (same as the TED talks). Releases: v1.0 (8 directions), v1.2 (14), v2.0 (En-De/Zh/Ja), v3.0 (En-De). The live page now returns 404; this is from the Jan 2024 archive | web.archive.org/web/20240110022813/https://mt.fbk.eu/must-c/ |
| CoVoST 2 | GitHub: "CoVoST data" CC0, anything else CC BY-NC 4.0. The HF card says cc-by-nc-4.0 | github.com/facebookresearch/covost |
| CVSS | CC BY 4.0 | github.com/google-research-datasets/cvss |
| FLEURS | CC BY 4.0 (has a ko_kr config) | huggingface.co/datasets/google/fleurs |
| MCIF | CC BY 4.0 | mt.fbk.eu/mcif |
| Europarl-ST | Corpus work CC BY-NC 4.0; underlying data rights belong to the EU | mllp.upv.es/europarl-st/README.md |
| NTREX-128 | CC BY-SA 4.0 | github.com/MicrosoftTranslator/NTREX |
| Audio-NTREX-4L | HF tag just says "cc" | huggingface.co/datasets/kyutai/Audio-NTREX-4L |
| RealSI | CC BY 4.0 annotations; videos linked, not owned | github.com/byteresearchcla/RealSI |
| Kosp2e | Per sub-corpus: Zeroth CC BY 4.0, StyleKQC CC BY-SA 4.0, KSS and Covid-ED CC BY-NC-SA 4.0 (academic use only) | github.com/warnikchow/kosp2e |
| ACL 60/60 | Not stated in the paper | aclanthology.org/2023.iwslt-1.2 |

## 4. Uncertainties
- **IWSLT 2025 test-set name:** the web page calls the En→X test "MCIF", the findings call it "IWSLT25Instruct" (ACL talks), and MCIF is the 2026 dev set. The 2025 test and 2026 dev may overlap; I did not verify this.
- **IWSLT 2026 ACL test references:** the page gives an audio download, but I could not confirm the references are public.
- **OmniSTEval bug:** v0.1.10 (2026-06-02) fixes LongYAAL underestimation for SimulStream logs, so use that version or later. The 2026 findings mention "latency measurement issues" without saying what they were; this bug is likely, but not confirmed, to be the cause.
- **CA below CU:** AlignAtt4LLM shows CA LongYAAL lower than CU because OmniSTEval's CA mode replaces chunk increments with wall-clock time.
- **Source discrepancies:** IWSLT 2025 hardware is an A100 80 GB on the web page but an H200 in the findings. The IWSLT 2024 page says AL is the primary metric, but the findings say latency was not used for ranking.
- **BLEU signatures:** the simultaneous-track findings do not give sacreBLEU signatures.
- **MuST-C versions:** papers use v1.0 vs v2.0 tst-COMMON for En-De; I did not check whether the two differ.
- **Seamless per-direction CSV:** it does not name its test set. I assumed FLEURS, consistent with the paper's Table 28.
- **LongYAAL paper:** I only confirmed it as arXiv 2509.17349, not a venue.

EVIDENCE SUMMARY FOR V3 (all from this project, 2026-10-21)

A. TAXI decomposition (same normalization; cp / speaker-agnostic agn / attribution share): VibeVoice-Streaming-7B 20.5/20.1/0.4, VibeVoice offline 16.8/16.5/0.3, ours V1 (st4-m2a ck2000, whole session, delta10) 26.4/18.4/8.0, ours V2 30.6/20.3/10.2. Our recognition already beats VibeVoice streaming; the gap is attribution only.

B. TAXI error anatomy (V1 d10; aggregate, read-only analysis):
- 640 turns, 540 speaker changes, no overlap, gaps 0-1 s (median 0.22 s), German dispatcher opens 85/86 sessions, slot A=EN in 45 / DE in 41 sessions. Inter-speaker level difference median 6.35 dB (q90 14 dB). Both sides 8 kHz telephone.
- 94% of wrong words are in whole-wrong turns; 98% in change turns; 89% follow a speaker tag that was never emitted at the turn onset (missed toggle). The model under-switches (496 hypothesis slot changes vs 540 reference; VibeVoice 548) and almost never switches falsely (0.3% of within-turn word pairs).
- Typical error: an isolated swallowed turn (A,B,A all in A's slot); 72% re-sync by also missing the next change. Inverted streaks only ~6-10%. First turn: 0/86 wrong (explicit first choice always correct, including the 44 sessions where the first speaker is in slot B). Slot initialization/flip irrelevant (7.9% vs 7.8%).
- Position in turn flat (no late switching); no gap dependence; no drift over time; worst 10 sessions = 75% of wrong words but structurally ordinary.
- Direction: English client turns swallowed into the German dispatcher slot 2.7x more (9.3% vs 3.5%). Short turns (<=3 words) whole-wrong 14.9% vs 7.9%.
- Language-forcing: in swallowed turns ~half of the words are correct own-language transcripts, ~half translated/paraphrased into the wrong slot's declared language. Oracle relabel: cp 26.5 -> 18.3 (=agn); fixing the translation leak too -> ~16 (beats VibeVoice offline 16.8).
- Toggle recall at changes: V1 87.4%, V1 d4 ~87%, V1 40-s windows 85.2%, V2 79.6%, V2 with tap disabled at inference 75.2%, VibeVoice 98.1%. Delta 4 vs 10 same attribution. V2 learned a bias against switching into slot B.
- OPEN: whether swallowed turns also lack an <EOT> for the previous speaker (being measured now with an EOT-logging decode).

C. Pipeline review (code facts):
- Tags are state-carrying, emitted only before the first emission and at changes (vapasr/data/cst_dialogue.py, semcommit_dataset.build_interleaved_events add_spk_tags). Missing a change is locally free.
- Same-language windows: slot A = first speaker; cross-language: slots by language-name order with random flip (slot = language). ~73-77% of dialogue steps are same-language.
- Speaker-activity head (BCE) on LLM hidden at audio positions, labels at current chunk (not delay-aligned); never used in decoding except a hard insertion heuristic that hurt (+4 pt). Tag tokens get loss weight 1 in mono Stage 3; pos_weight exists per position (dataset can set it; used for <EOT> decision positions).
- Cross-language synthetic (xl) pairs always come from different corpora (EN pools incl. 8 kHz Switchboard/read speech vs German KEC/ZTC/NSC), so channel/corpus difference coincides with speaker change; phone augmentation p=0.3 is per window, identical for both speakers; level jitter at most +-3 dB per speaker; synthetic turn timing has 30% same-speaker holds and real FTO with 23% overlap and backchannels, unlike TAXI's strict alternation.
- Recover training (synthetic span flips) never applied to cross-language/translation-mode windows; on same-language data it had no effect.
- Switchboard sides are used only as single-speaker material, not as real 2-line telephone conversations; CallHome-de used as single-speaker replay.
- In overlaps, tags may split words (token-level interleave) - minor.

D. Literature (verified by research agents; see claims being re-verified):
- Missed changes are the known weakness of change-token systems; fixes: recall-oriented weighting of speaker/change tokens (Google FA/FR loss: recall 55.8->65.2; G-STAR 2x speaker-token CE; SA-DiCoW 5x), decode-time turn-token posterior scaling (F1 64.6->75.3), virtual same-speaker change tokens / per-turn re-assertion (Zheng et al. 2026, arXiv:2606.13095), per-token speaker decisions with 1-2 word hysteresis (t-vector; LibriCSS cpWER 12.4 at 2 words vs 19.7 at 8).
- Feeding resolved speaker evidence (activity posteriors / tracker embeddings) into the decoder helps (Sortformer kernels AMI cpWER 32.94->28.17; G-STAR Fisher 14.23->10.29); raw features without a speaker objective do not (our V2-a). Encoder-side training-only word-level speaker supervision helps (JEDIS: AMI WDER 10.87->6.97).
- Exposure bias: on-policy correction from the model's own hypotheses works better than synthetic flips (DiarizationLM hyp2ora; Google N-best).
- Memory/caches (AOSC) matter for long sessions; TAXI (<=72 s) fits the context.
- Data: realistic turn-taking simulation and real telephone 2-speaker data matter (Landini; Fisher used by all strong systems). Fisher/MLC-SLM would need the user's download.
- VibeVoice streaming decides speakers at 1.5-2.0 s latency with a 7B LLM and ~420k h of streaming pretraining.

E. Constraints: one E2E model at inference; latency <= 0.8 s (delta10) for words (whether speaker labels may lag 1-2 words is a user decision); 80 ms chunks; greedy decoding; ~1 h runs (3000 steps on 2x8 H200), user submits SLURM jobs; TAXI is eval-only (no tuning on TAXI); keep recognition, EOT, translation and single-speaker metrics from regressing; Korean/English/German same-language dialogue attribution also matters (current share 7.4 pt vs VibeVoice 5.0 streaming / 2.2 offline). Baseline recipe for comparisons: V1 = st4-m2a (2 slots, 20-40 s windows, phone aug 0.3, split v2, 2500 steps). A tap-less V2 control (v2-p1-notap) is pending.

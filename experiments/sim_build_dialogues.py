#!/usr/bin/env python
"""합성 2 인(V2: --speakers 로 3–4 인도) 대화 셋 만들기(Stage 3 화자 귀속 보강) — vapasr/data/sim_dialogue 로 세션을 합성하고 mono 혼합(+잡음·잔향)을 캐시한다.

  python experiments/sim_build_dialogues.py --lang English --out <dir> --sessions 2000 \\
      --phase2 <phase2 dir> --split <v2.json> --phase2-corpora otoSpeech --words @<list> --words @<list> [--words-frac 0.15] \\
      --pool-weights otoSpeech:0.3,swbd:0.3,cv:0.15,vp:0.15,ls:0.1 [--noise-bank <dir>] [--workers 16]

재료 풀: Phase 2 정제 대화(split 학습 분할만 — held-out 대화는 쓰지 않는다)의 코퍼스별 화자 발화 + Stage 1·2 words(화자 키를 아는 세트: ls·swbd·vp·cv).
  풀 이름 = Phase 2 코퍼스 이름 | words 키 접두어. 화자 키마다 발화 --min-units 개 이상만 쓴다.
시간 구조: --stats-corpora(기본 = --phase2-corpora)의 학습 분할 실제 대화에서 잰다(sim_dialogue.timing_stats).
세션 i(난수 = seed:i): 풀을 가중치로 고르고, 그 풀에서 다른 사람인 화자 키 둘을 고른다(Phase 2 는 다른 대화이면서 person_ids 가 겹치지 않게).
  각자 발화를 원래 순서로 임의 위치부터 꺼내 --dur 범위 길이로 합성한다(synth_session). 발화 오디오를 읽어 말소리 구간(speech_span)만 놓는다.
  --split-s 보다 긴 발화는 안쪽 쉼에서 구 단위로 나눠(split_points) 앞 조각부터 그 화자의 턴들에 쓴다. 재개 확률 --p-hold, 맞장구 배율 --bc-scale.
mono: 화자별 채널(놓인 말소리)을 dialogue_mix.mix_dialogue 와 같은 규칙(화자 발화 구간 RMS → −23 dBFS, ± --jitter-db, 피크 0.98)으로 맞춰 더하고,
  --noise-bank 가 있으면 섞은 뒤 잡음·잔향을 둘에 같이 입힌다(녹음 조건 차이가 화자 단서가 되지 않게).
  저장: <out>/mono/sim_<xx>/<conv>.npy(float16) — CstDialogueDataset(mono_cache_dir=<out>/mono) 의 캐시 규약이라 학습이 그대로 읽는다.
출력: <out>/sim_<xx>.refined.dialogues.jsonl, <out>/sim_<xx>.stats.json(세션·시간·풀별 수·겹침·교대·맞장구·시간 구조 요약·인자). 아무것도 지우지 않는다.

교차 언어(Stage 4, decision-stage4-prep-20261019 — 이어 붙인 합성 대화): --lang2 와 그 재료(--phase2-corpora2·--words2·--pool-weights2)를 주면
  화자 a 는 --lang 풀, 화자 b 는 --lang2 풀에서 고른다(코퍼스가 달라 사람 겹침 검사는 없다). 시간 구조는 두 언어의 실제 대화에서 함께 잰다.
  이름 xx = '<l1>-<l2>'(예 sim_ko-en), Dialogue.lang = '<L1>+<L2>', meta.spk_lang = {a: L1, b: L2} — CstDialogueDataset 이 슬롯을 언어로 고정한다.
  번역(--trans, 반복·쉼표·@목록 — ast_build_targets 산출: Stage 2 단일 화자 타깃(words 행 id)·S1 대화 타깃(conv|화자|utt)): 발화 Unit 에 붙여 놓고
  (sim_dialogue 모듈 설명), <out>/targets-sim_<xx>/<xx>/u-*.jsonl.gz(load_ast_targets 형식, 행 id = <conv>|<화자>|<utt_id 의 # 앞>)로 쓴다.
  교차 언어는 상대 언어 번역이 있는 발화만 재료로 쓴다(--allow-untranslated 로 끔).
--heldout: 평가 셋 — split held-out 대화만 재료로 쓰고 words 는 쓰지 않는다(학습 합성 셋은 held-out 을 쓰지 않으므로 겹치지 않는다).

2 회선 전화 조건(V3 C2 — sim_en-de-tel; 기본값이면 산출·난수 소비가 예전과 바이트 동일):
  --fto-clip lo,hi · --intra-pause lo,hi: SimConfig 의 교대 간격 자르기·턴 안 발화 사이 쉼(기본 = SimConfig 기본 -2,3 · 0.1,0.5). lo > 0 이면 겹침이 없고,
    hi < 0.25 s(dialogue.GAP_S)면 합성 턴 하나가 episode 하나다(턴 안에 같은 화자 <EOT> 가 생기지 않는다). 맞장구 끄기는 --bc-scale 0, 재개는 --p-hold.
  --phone-per-speaker P: 화자 트랙마다 따로 전화 채널(phone_aug.PhoneAugment(p=P, mulaw_p=0.5) — 대역·μ-law 결정)을 말소리를 놓은 뒤·gain 전에 건다.
    난수는 세션 rng 와 따로 random.Random(f'{seed}:{i}:chan:{화자}')(세션 난수열 그대로). gain 은 처리된 트랙의 말소리 구간 RMS 로 다시 재서 레벨 목표(−23 dBFS ± --jitter-db)를 지킨다.
    --phone-same-frac F: 이 비율의 세션은 모든 화자가 같은 채널 파라미터(첫 화자 시드)를 쓴다('회선이 다르면 화자가 다르다' 지름길을 막는다).
    meta.chan = {same, spk: {화자: {phone, low, high, mulaw, level_db}}}(켰을 때만).
  --name-suffix SUF: 교차 언어 키 sim_<l1>-<l2>-SUF, 같은 언어 키 sim_<l>-SUF(세 글자 이상, 2026-10-26) — 산출 이름(jsonl·stats·targets-·mono/)·conv_id 가 따른다.
  위 옵션이 하나라도 기본이 아니면 stats.json 에 tel(화자 레벨 차 |Δ|·교대/재개 간격·같은 화자 쉼 히스토그램, hold 비율, 채널 집계)을 더한다.

회의형 겹침·원거리(2026-10-26 데이터 확장 D6 — 기본값이면 예전과 바이트 동일):
  --p-interrupt P · --interrupt-fto lo,hi: 교대 때 확률 P 로 교대 간격을 U(lo, hi)(음수 = 앞 턴과 겹침)에서 뽑는다(SimConfig.p_interrupt).
  --far-per-speaker P: 세션마다 확률 P 로 한 방의 화자별 RIR(far_aug.FarFieldAugment.pick_rirs — 모의 방 RoomNNN 의 다른 위치, 일부 실제 RIR)을
    화자 트랙에 건다(직접음 t=0, 섞기 전·gain 전, 난수는 random.Random(f'{seed}:{i}:far')). 그 세션은 섞은 뒤 잔향을 다시 걸지 않고 잡음만(--noise-p) 더한다.
    --noise-bank 가 필요하다. meta.far = {화자: RIR 번호}(켰을 때만)."""
import argparse, collections, hashlib, json, math, os, random, re, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHORT = {"Korean": "ko", "English": "en", "German": "de"}
TEL_DEFAULTS = dict(fto_clip=(-2.0, 3.0), intra_pause=(0.1, 0.5), phone_per_speaker=0.0, phone_same_frac=0.0, name_suffix=None, level_offset_db="")   # C2 옵션 기본(fto·intra = SimConfig 기본) — 이 값이면 빌드·stats 가 예전과 같다
G = {}


def _paths(vals):
    return [q for x in vals for q in ([l.strip() for l in open(x[1:]) if l.strip()] if x.startswith("@") else x.split(",")) if q]


def _keep(p, frac):
    return frac >= 1.0 or int(hashlib.sha1(p.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def _person_of_words_key(k):
    fam, rest = k.split(":", 1)
    return {f"ls:{rest.split('-')[0]}"} if fam == "ls" else {k}


def load_pools(a, lang=None, corpora=None, words=None, stats_corpora=None, TR=None, need=None, min_keys=2):
    """→ (풀 {이름: {화자 키: [Unit…]}}, 사람 id, 시간 구조용 실제 대화, 정보). 기본 인자는 --lang 쪽. TR: 번역 타깃({행 id: {목표 언어: (ids, 시각)}}),
    need: 그 목표 언어 번역이 있는 발화만 남긴다(교차 언어). min_keys: 풀에 남길 화자 키 수(같은 언어는 한 풀에서 둘을 고르므로 2)."""
    from vapasr.data.dialogue import Dialogue
    from vapasr.data.sim_dialogue import units_from_dialogue, units_from_words_row, person_ids
    lang = lang or a.lang; corpora = a.phase2_corpora if corpora is None else corpora; words = a.words if words is None else words
    stats_corpora = a.stats_corpora if stats_corpora is None else stats_corpora
    split = json.load(open(a.split, encoding="utf-8")); pools = collections.defaultdict(lambda: collections.defaultdict(dict)); persons = {}; stats_d = []
    info = {}; n_drop = collections.Counter()
    def keep(u, pool):
        if need is None or need in u.trans: return True
        n_drop[pool] += 1; return False
    for c in corpora:
        held = set(split["corpora"].get(c, {}).get("heldout", [])); n_d = n_u = 0
        for line in open(os.path.join(a.phase2, f"{c}.refined.dialogues.jsonl"), encoding="utf-8"):
            d = Dialogue.from_json(line)
            if (d.conv_id not in held if a.heldout else d.conv_id in held) or d.lang != lang: continue
            n_d += 1
            if c in stats_corpora: stats_d.append(d)
            for u in units_from_dialogue(d, trans=TR):
                if keep(u, c): pools[c][u.key][u.uid] = u; persons[u.key] = person_ids(d, u.key.split("|")[2]); n_u += 1
        info[c] = dict(dialogues=n_d, units=n_u)
    files = [] if a.heldout else [p for p in _paths(words) if _keep(p, a.words_frac)]; n_rows = 0              # 평가 셋(--heldout)은 학습 words 를 쓰지 않는다
    for p in files:
        for line in open(p, encoding="utf-8"):
            if f'"lang": "{lang}"' not in line: continue
            r = json.loads(line); n_rows += 1
            for u in units_from_words_row(r, trans=TR.get(r["id"]) if TR else None):
                fam = u.key.split(":", 1)[0]
                if keep(u, fam): pools[fam][u.key][u.uid] = u; persons[u.key] = _person_of_words_key(u.key)
    info["words"] = dict(files=len(files), rows=n_rows)
    out = {}
    for name, keys in pools.items():
        kk = {k: sorted(v.values(), key=lambda u: u.order) for k, v in keys.items() if len(v) >= a.min_units}
        if len(kk) >= min_keys: out[name] = kk
        info.setdefault(name, {}).update(keys_all=len(keys), keys_used=len(kk), units_used=sum(len(v) for v in kk.values()), units_untranslated=n_drop[name])
    return out, persons, stats_d, info


def pick_keys(rng, keys, persons, tries=60, n=2):
    """다른 사람인 화자 키 n 개(쌍마다 person_ids 가 겹치지 않고 Phase 2 는 다른 대화). n = 2 는 예전과 같은 난수열."""
    ks = list(keys)
    if n == 2:
        for _ in range(tries):
            a, b = rng.sample(ks, 2)
            if persons[a] & persons[b]: continue
            if "|" in a and a.split("|")[1] == b.split("|")[1]: continue
            return a, b
        return None
    if len(ks) < n: return None
    for _ in range(tries):
        got = rng.sample(ks, n)
        if any(persons[x] & persons[y] or ("|" in x and x.split("|")[1] == y.split("|")[1]) for i, x in enumerate(got) for y in got[i + 1:]): continue
        return tuple(got)
    return None


def mix_arrays(d, audio, spans, rng, jitter_db, target_dbfs=-23.0, peak=0.98, fx=None, levels=None, offsets=None):
    """dialogue_mix.mix_dialogue 와 같은 규칙을 이미 읽은 발화 오디오로: 화자별 채널에 말소리 구간을 놓고, 화자 발화 구간 RMS 로 target_dbfs(± jitter) 에 맞춰 더한다.
    fx(ch, 화자) → 처리한 채널(C2 화자별 전화 채널 — phone_fx): 말소리를 놓은 뒤·gain 전에 걸고, gain 은 처리된 채널의 같은 말소리 구간 RMS 로 다시 잰다.
    levels: 주면 화자별 말소리 구간 레벨(dBFS, gain 뒤·피크 제한 전 = target_dbfs + jitter)을 채운다. 둘 다 None 이면 예전과 같은 계산·난수 소비.
    offsets: {화자: dB} 를 주면 레벨 목표·jitter 뒤 gain 에 곱한다(진단 --level-offset-db — 난수를 더 뽑지 않아 같은 시드면 같은 대화를 레벨만 바꿔 섞는다)."""
    import numpy as np
    from vapasr.data.sim_dialogue import SR
    T = int(round(d.duration_s * SR)); mono = np.zeros(T, np.float32); gains = {}
    for spk in d.speakers:
        ch = np.zeros(T, np.float32); acc = 0.0; n = 0; rr = []
        for u in d.utterances:
            if u.speaker != spk: continue
            uid = u.utt_id.split(":", 1)[1]; a, b = spans[uid]; x = audio[uid]; i0 = int(round(u.start * SR)); m = int(round((u.end - u.start) * SR))
            seg = x[int(round(a * SR)): int(round(a * SR)) + m]; seg = seg[: max(0, T - i0)]; ch[i0: i0 + len(seg)] += seg; acc += float(np.sum(seg.astype(np.float64) ** 2)); n += len(seg)
            rr.append((i0, len(seg)))
        if fx is not None:                                                         # 대역 제한·μ-law 가 말소리 RMS 를 바꾼다 → 처리된 채널의 같은 구간으로 다시 잰다
            ch = fx(ch, spk); acc = sum(float(np.sum(ch[i0: i0 + m].astype(np.float64) ** 2)) for i0, m in rr)
        rms = math.sqrt(acc / n) if n else 1e-6
        g = (10 ** (target_dbfs / 20)) / max(rms, 1e-6) * (10 ** (rng.uniform(-jitter_db, jitter_db) / 20) if jitter_db else 1.0)
        if offsets: g *= 10 ** (offsets.get(spk, 0.0) / 20)
        gains[spk] = round(g, 4); mono += ch * g
        if levels is not None: levels[spk] = round(20 * math.log10(max(g * rms, 1e-9)), 3)
    pk = float(np.max(np.abs(mono))) if T else 0.0
    return (mono * (peak / pk) if pk > peak else mono), gains


def phone_fx(aug, seed, i, speakers, same_frac=0.0):
    """C2 화자별 전화 채널(--phone-per-speaker): 화자 트랙마다 따로 aug(PhoneAugment — 대역·μ-law 결정)를 건다 → (fx(ch, 화자) → 처리한 채널, meta.chan).
    난수는 세션 rng 와 따로 random.Random(f'{seed}:{i}:chan:{화자}') — 세션 난수열을 건드리지 않는다. same_frac 비율의 세션(판정 난수 f'{seed}:{i}:chan',
    same_frac 0 이면 뽑지 않는다)은 모든 화자가 첫 화자의 시드를 써서 같은 대역·μ-law 결정을 받는다. meta.chan = {same, spk: {화자: {phone, low, high, mulaw}}}."""
    same = same_frac > 0 and random.Random(f"{seed}:{i}:chan").random() < same_frac; chan = dict(same=same, spk={})
    def fx(ch, spk):
        info = {}; y = aug(ch, random.Random(f"{seed}:{i}:chan:{speakers[0] if same else spk}"), info=info); chan["spk"][spk] = dict(phone=bool(info), **info)
        return y
    return fx, chan


def tel_row(d, levels, chan):
    """C2 stats 의 세션 행: 화자 레벨 차(최대 − 최소 dB — 2 인이면 |Δ|), episode 기준 교대·재개 간격과 같은 화자 쉼(sim_dialogue.turn_gaps),
    합성 턴 대비 남는 episode 수(split_turns — 턴 안 쉼이 GAP_S 를 넘어 갈라진 턴; 재개가 앞 턴과 붙으면 음수), 채널(meta.chan)."""
    from vapasr.data.sim_dialogue import turn_gaps
    lv = list(levels.values()); tg = turn_gaps(d)
    return dict(level_diff=round(max(lv) - min(lv), 3) if len(lv) > 1 else None, change=tg["change"], hold=tg["hold"], same=tg["same"],
                split_turns=tg["episodes"] - d.meta["turns"] - d.meta["bc"], chan=chan)


def _hist(xs, lo, hi, w):
    """[lo, hi) 를 폭 w 칸으로 센 막대 + 범위 밖 수(stats.json 의 C2 레벨 차·타이밍 분포)."""
    k = int(round((hi - lo) / w)); n = [0] * k; under = over = 0
    for x in xs:
        j = int(math.floor((x - lo) / w + 1e-9))
        if j < 0: under += 1
        elif j >= k: over += 1
        else: n[j] += 1
    return dict(lo=lo, w=w, n=n, under=under, over=over)


def far_fx(fa, p, seed, i, speakers):
    """D6 화자별 원거리(--far-per-speaker): 확률 p 로 한 방의 화자별 RIR 을 고른다 → (fx(ch, 화자) → 잔향 채널, meta {화자: RIR 번호}) 또는 None.
    난수는 세션 rng 와 따로(세션 난수열 그대로)."""
    r = random.Random(f"{seed}:{i}:far")
    if r.random() >= p: return None
    m = dict(zip(speakers, fa.pick_rirs(len(speakers), r)))
    return (lambda ch, spk: fa.na.reverb(ch, r, m[spk])), {s: int(k) for s, k in m.items()}


def make_session(i):
    import numpy as np
    from vapasr.data.streams import load_utt_audio
    from vapasr.data.sim_dialogue import synth_session, speech_span, overlap_stats, split_points, split_unit, SimConfig, SR
    from vapasr.data.dialogue_dataset import mono_cache_path
    a = G["args"]; xx = G["xx"]; errs = []
    for attempt in range(5):
        rng = random.Random(f"{a.seed}:{i}:{attempt}")
        if G.get("pools2") is None:
            names = [n for n in G["weights"] if n in G["pools"]]
            pool = rng.choices(names, weights=[G["weights"][n] for n in names])[0]
            n_spk = rng.choices(list(G["n_spk"]), weights=list(G["n_spk"].values()))[0] if len(G["n_spk"]) > 1 else next(iter(G["n_spk"]))   # 2 만이면 난수를 더 뽑지 않는다
            ks = pick_keys(rng, G["pools"][pool], G["persons"], n=n_spk)
            if ks is None: continue
            src = [G["pools"][pool][k] for k in ks]
        else:                                                                      # 교차 언어: 화자 a = --lang 풀, b = --lang2 풀(코퍼스가 달라 사람이 겹치지 않는다)
            picks = []
            for P, W in ((G["pools"], G["weights"]), (G["pools2"], G["weights2"])):
                names = [n for n in W if n in P]; pn = rng.choices(names, weights=[W[n] for n in names])[0]; k = rng.choice(list(P[pn])); picks.append((pn, k, P[pn][k]))
            pool = "+".join(x[0] for x in picks); ks = [x[1] for x in picks]; src = [x[2] for x in picks]
        us = []
        for v in src:                                                              # 행 첫 조각에서 시작한다(이어지는 조각 Unit.cont 를 건너뜀 — sim_dialogue 행 무결성)
            i0 = rng.randrange(len(v)); j = 0
            while v[i0].cont and j < len(v): i0 = (i0 + 1) % len(v); j += 1
            us.append((v[i0:] + v[:i0])[: a.max_units])
        audio, spans = {}, {}
        def load(u):
            if u.uid not in audio:
                audio[u.uid] = np.asarray(load_utt_audio(u.path, u.src_offset, u.dur) if u.src_offset is not None else load_utt_audio(u.path), dtype=np.float32)
            return audio[u.uid]
        def span(u):
            if u.uid not in spans: spans[u.uid] = speech_span(load(u), u.tokens)
            return spans[u.uid]
        def expand(u):                                                             # 긴 발화 → 쉼에서 구 단위 조각(조각 오디오는 부모를 잘라 캐시)
            x = load(u); parts = split_points(x, u.tokens, max_s=a.split_s); subs = split_unit(u, parts)
            for sub in subs[1:] if len(subs) == 1 else subs:
                k = int(sub.uid.rsplit("~", 1)[1]); p0, p1 = parts[k]; audio[sub.uid] = x[int(round(p0 * SR)): int(round(p1 * SR))]
            return subs
        try:
            d = synth_session(us, G["stats"], rng, rng.uniform(*a.dur), span, f"sim_{xx}_{i:06d}", f"sim_{xx}", G["lang_name"],
                              cfg=SimConfig(p_hold=a.p_hold, bc_scale=a.bc_scale, p_back=a.p_back, fto=a.fto_clip, intra_pause=a.intra_pause,
                                            **({"p_interrupt": a.p_interrupt, "interrupt_fto": tuple(float(v) for v in a.interrupt_fto.split(","))} if a.p_interrupt > 0 else {})),
                              meta=dict(pool=pool, keys=list(ks), attempt=attempt, **({"spk_lang": {"a": a.lang, "b": a.lang2}} if a.lang2 else {})), expand=expand)
            if d.meta["turns"] < 4 or d.duration_s < 25.0: errs.append("too_short"); continue
            if len(us) > 2 and min(collections.Counter(u.speaker for u in d.utterances).get(x, 0) for x in d.speakers) < 2: errs.append("speaker_too_few"); continue   # 3 명 이상: 모두 발화 2 개 이상
            tr = d.meta.pop("trans", None)
            fx, chan = phone_fx(G["phone"], a.seed, i, d.speakers, a.phone_same_frac) if G.get("phone") is not None else (None, None); lv = {} if G.get("tel") else None
            far = far_fx(G["far"], a.far_per_speaker, a.seed, i, d.speakers) if G.get("far") is not None else None
            if far is not None:
                ffx, d.meta["far"] = far; pfx = fx
                fx = (lambda ch, spk, f1=ffx, f2=pfx: f2(f1(ch, spk), spk)) if pfx is not None else ffx
            LO = G.get("level_offset") or {}; offs = {s: LO.get((d.meta.get("spk_lang") or {}).get(s, a.lang), 0.0) for s in d.speakers} if LO else None
            y, gains = mix_arrays(d, audio, spans, rng, a.jitter_db, fx=fx, levels=lv, offsets=offs); d.meta["gains"] = gains
            if offs and any(offs.values()): d.meta["level_offset_db"] = offs
            if chan is not None:
                for s, v in chan["spk"].items(): v["level_db"] = lv[s]
                d.meta["chan"] = chan
            if G.get("noise") is not None:
                ra = random.Random(f"{a.seed}:{i}:aug")
                if far is None: y = G["noise"](y, ra)
                elif G["noise"].p_noise > 0 and ra.random() < G["noise"].p_noise: y = G["noise"].add_noise(y, ra)   # 화자별 원거리 세션: 잔향은 이미 — 잡음만
            f = mono_cache_path(os.path.join(a.out, "mono"), d); os.makedirs(os.path.dirname(f), exist_ok=True)
            tmp = f[:-4] + f".{os.getpid()}.tmp.npy"; np.save(tmp, np.asarray(y, dtype=np.float16)); os.replace(tmp, f)
            return i, d.to_json(), dict(overlap_stats(d), pool=pool, bc=d.meta["bc"], turns=d.meta["turns"], units=d.meta["units"], attempt=attempt, errors=errs,
                                        utts=len(d.utterances), utts_translated=len(tr or {}), **({"tel": tel_row(d, lv, chan)} if lv is not None else {})), (d.conv_id, tr)
        except Exception as e:
            errs.append(f"{type(e).__name__}: {str(e)[:80]}")
    return i, None, dict(failed=True, errors=errs), None


def write_targets(out_dir, xx, trans_all):
    """세션 번역 → <out>/targets-sim_<xx>/<xx>/u-NNNNN.jsonl.gz(load_ast_targets 형식). 행 id = <conv>|<화자>|<utt_id 의 # 앞>(CstDialogueDataset 의 행 —
    정제 조각 #n 이 한 세션에 같이 놓이면 한 행으로 합친다). → (행 수, 토큰 수)."""
    import gzip
    td = os.path.join(out_dir, f"targets-sim_{xx}", xx); os.makedirs(td, exist_ok=True); buf = []; k = [0]; n = [0, 0]
    def flush():
        if not buf: return
        f = os.path.join(td, f"u-{k[0]:05d}.jsonl.gz"); tmp = f + ".partial"
        with gzip.open(tmp, "wt", encoding="utf-8") as g: g.write("\n".join(buf) + "\n")
        os.replace(tmp, f); k[0] += 1; buf.clear()
    for i in sorted(trans_all):
        cid, tr = trans_all[i]; rows = {}
        for uid, v in tr.items():
            for g, toks in v.items(): rows.setdefault((f"{cid}|{uid.split(':', 1)[0]}|{uid.split('#')[0]}", g), []).extend(toks)
        for (rid, g), toks in rows.items():
            toks.sort(key=lambda z: z[1]); buf.append(json.dumps(dict(id=rid, tgt=g, tokens=toks), ensure_ascii=False)); n[0] += 1; n[1] += len(toks)
            if len(buf) >= 50000: flush()
    flush()
    return n[0], n[1]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--lang", required=True, choices=sorted(SHORT)); ap.add_argument("--out", required=True)
    ap.add_argument("--sessions", type=int, default=2000); ap.add_argument("--dur", default="60,120"); ap.add_argument("--phase2", required=True); ap.add_argument("--split", required=True)
    ap.add_argument("--phase2-corpora", default=""); ap.add_argument("--stats-corpora", default=None); ap.add_argument("--words", action="append", default=[])
    ap.add_argument("--words-frac", type=float, default=0.15); ap.add_argument("--pool-weights", required=True); ap.add_argument("--min-units", type=int, default=10)
    ap.add_argument("--max-units", type=int, default=80); ap.add_argument("--jitter-db", type=float, default=3.0); ap.add_argument("--noise-bank", default=None)
    ap.add_argument("--noise-p", type=float, default=0.5); ap.add_argument("--noise-snr", default="10,30"); ap.add_argument("--rir-p", type=float, default=0.3)
    ap.add_argument("--p-hold", type=float, default=0.3, help="같은 화자 재개 확률(음수 = 실제 대화 값)"); ap.add_argument("--bc-scale", type=float, default=1.5)
    ap.add_argument("--split-s", type=float, default=4.0, help="이보다 긴 발화는 안쪽 쉼에서 구 단위로 나눈다")
    ap.add_argument("--speakers", default="2:1", help="세션 화자 수 분포 'n:가중,…'(V2 예 2:0.2,3:0.4,4:0.4 — 같은 언어만; 기본 2 인 = 예전과 같은 난수열)")
    ap.add_argument("--p-back", type=float, default=0.2, help="3 명 이상: 교대 때 앞앞 턴 화자에게 되돌아갈 확률(sim_dialogue.SimConfig)")
    ap.add_argument("--p-interrupt", type=float, default=0.0, help="교대 때 끼어들기 확률(SimConfig.p_interrupt — 기본 끔)"); ap.add_argument("--interrupt-fto", default="-2.0,-0.3")
    ap.add_argument("--far-per-speaker", type=float, default=0.0, help="세션마다 화자별 RIR(한 방) 확률 — --noise-bank 필요, 기본 끔")
    ap.add_argument("--far-real-p", type=float, default=0.3, help="--far-per-speaker 에서 실제 RIR 풀을 고를 확률")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--seed", type=int, default=0)
    g = ap.add_argument_group("교차 언어·번역(Stage 4)")
    g.add_argument("--lang2", default=None, choices=sorted(SHORT), help="화자 b 의 언어(주면 교차 언어 세션)"); g.add_argument("--phase2-corpora2", default="")
    g.add_argument("--stats-corpora2", default=None); g.add_argument("--words2", action="append", default=[]); g.add_argument("--pool-weights2", default="")
    g.add_argument("--trans", action="append", default=[], help="번역 타깃(ast_build_targets 산출, 반복·쉼표·@목록) — Stage 2 단일 화자·S1 대화")
    g.add_argument("--allow-untranslated", action="store_true", help="교차 언어에서 상대 언어 번역이 없는 발화도 재료로 쓴다")
    g.add_argument("--heldout", action="store_true", help="평가 셋: split held-out 대화만 재료로 쓴다(words 없음) — 학습 셋과 겹치지 않는다")
    t = ap.add_argument_group("2 회선 전화 조건(V3 C2 — 기본값이면 예전과 바이트 동일)")
    t.add_argument("--fto-clip", default=None, help="교대 간격을 'lo,hi' s 로 자른다(SimConfig.fto, 기본 -2,3 — 음수는 --fto-clip=-1,2 처럼). lo > 0 이면 겹침 없음")
    t.add_argument("--intra-pause", default=None, help="턴 안 발화 사이 쉼 'lo,hi' s(SimConfig.intra_pause, 기본 0.1,0.5). hi < 0.25 면 턴 하나가 episode 하나")
    t.add_argument("--phone-per-speaker", type=float, default=0.0, help="화자 트랙마다 따로 전화 채널을 걸 확률(PhoneAugment p, mulaw_p 0.5 — 섞기 전·gain 전; 0 = 끔)")
    t.add_argument("--phone-same-frac", type=float, default=0.0, help="--phone-per-speaker 세션 중 모든 화자가 같은 채널 파라미터를 쓰는 비율")
    t.add_argument("--name-suffix", default=None, help="교차 언어 키 접미사(예 tel → sim_en-de-tel; 산출 이름·conv_id 가 따른다)")
    t.add_argument("--level-offset-db", default="", help="진단: 언어별 화자 레벨 오프셋 'LANG:DB[,LANG:DB]'(예 English:-12) — 레벨 목표·jitter 뒤 gain 에 곱한다. "
                                                        "난수를 더 뽑지 않아 같은 시드·인자면 같은 대화를 그 언어 화자 레벨만 바꿔 다시 섞는다(기본 끔)")
    a = ap.parse_args()
    pair = lambda s, k: tuple(float(x) for x in s.split(",")) if s else TEL_DEFAULTS[k]
    a.fto_clip, a.intra_pause = pair(a.fto_clip, "fto_clip"), pair(a.intra_pause, "intra_pause")
    if len(a.fto_clip) != 2 or a.fto_clip[0] > a.fto_clip[1] or len(a.intra_pause) != 2 or not 0 <= a.intra_pause[0] <= a.intra_pause[1]:
        sys.exit(f"--fto-clip·--intra-pause 는 'lo,hi'(lo ≤ hi, 쉼 ≥ 0): {a.fto_clip} {a.intra_pause}")
    if not (0 <= a.phone_per_speaker <= 1 and 0 <= a.phone_same_frac <= 1) or (a.phone_same_frac > 0 and a.phone_per_speaker <= 0):
        sys.exit(f"--phone-per-speaker·--phone-same-frac 는 [0, 1], same-frac 은 per-speaker > 0 일 때만: {a.phone_per_speaker} {a.phone_same_frac}")
    if a.name_suffix is not None and not re.fullmatch(r"[a-z0-9]+" if a.lang2 else r"[a-z0-9]{3,}", a.name_suffix):   # 같은 언어는 세 글자 이상(sim_en-xx 는 교차 언어 키)
        sys.exit(f"--name-suffix: 교차 언어 [a-z0-9]+, 같은 언어 [a-z0-9]{{3,}}(2026-10-26 — 예 meet → sim_en-meet): {a.name_suffix!r}")
    if a.p_hold < 0: a.p_hold = None
    try: LO = {k.strip(): float(v) for k, v in (x.split(":") for x in a.level_offset_db.split(",") if x.strip())}
    except ValueError: sys.exit(f"--level-offset-db 'LANG:DB[,LANG:DB]': {a.level_offset_db!r}")
    if any(k not in SHORT for k in LO): sys.exit(f"--level-offset-db 언어는 {sorted(SHORT)} 중: {sorted(LO)}")
    G["level_offset"] = {k: v for k, v in LO.items() if v}
    a.dur = tuple(float(x) for x in a.dur.split(",")); a.phase2_corpora = [c for c in a.phase2_corpora.split(",") if c]
    a.stats_corpora = [c for c in (a.stats_corpora.split(",") if a.stats_corpora else a.phase2_corpora) if c]
    a.phase2_corpora2 = [c for c in a.phase2_corpora2.split(",") if c]
    a.stats_corpora2 = [c for c in (a.stats_corpora2.split(",") if a.stats_corpora2 else a.phase2_corpora2) if c]
    if a.lang2 and (a.lang2 == a.lang or not a.pool_weights2): sys.exit("--lang2 는 --lang 과 다른 언어이고 --pool-weights2 가 필요하다")
    weights = {k: float(v) for k, v in (x.split(":") for x in a.pool_weights.split(",") if x)}
    weights2 = {k: float(v) for k, v in (x.split(":") for x in a.pool_weights2.split(",") if x)}
    xx = SHORT[a.lang] + (f"-{SHORT[a.lang2]}" if a.lang2 else "") + (f"-{a.name_suffix}" if a.name_suffix else ""); lang_name = a.lang + (f"+{a.lang2}" if a.lang2 else "")
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    from vapasr.data.sim_dialogue import timing_stats
    TR = None
    if a.trans:
        from vapasr.data.ast_dataset import load_ast_targets
        TR, tr_st, _ = load_ast_targets(_paths(a.trans)); print(f"번역 타깃 {dict(tr_st)} · {time.time() - t0:.0f}s", flush=True)
    need = (lambda lg: lg) if a.lang2 and not a.allow_untranslated else (lambda lg: None)
    pools, persons, stats_d, info = load_pools(a, TR=TR, need=need(a.lang2), min_keys=1 if a.lang2 else 2); pools2 = None   # 3–4 인 세션에 키가 모자란 풀은 pick_keys 가 None → 다른 시도(풀·화자 수)
    if a.lang2:
        pools2, persons2, stats_d2, info2 = load_pools(a, lang=a.lang2, corpora=a.phase2_corpora2, words=a.words2, stats_corpora=a.stats_corpora2, TR=TR, need=need(a.lang), min_keys=1)
        persons.update(persons2); stats_d += stats_d2; info = {"a": info, "b": info2}
    stats = timing_stats(stats_d)
    show = lambda P: {k: (len(v), sum(len(u) for u in v.values())) for k, v in P.items()}
    print(f"풀 {show(pools)}" + (f" · 풀2 {show(pools2)}" if pools2 is not None else "") + f" · 시간 구조 대화 {stats.n_dialogues} · 교대 쌍 {stats.n_pairs} · 재개 {stats.p_hold:.2f} · "
          f"맞장구 {stats.bc_rate:.3f}/s · {time.time() - t0:.0f}s", flush=True)
    missing = [k for k, w in weights.items() if w > 0 and k not in pools] + [f"2:{k}" for k, w in weights2.items() if a.lang2 and w > 0 and k not in pools2]
    if missing: sys.exit(f"가중치 > 0 인데 쓸 수 있는 풀이 없다: {missing} (있는 풀 {sorted(pools)}{' / ' + str(sorted(pools2)) if pools2 is not None else ''})")
    n_spk = {int(k): float(v) for k, v in (x.split(":") for x in a.speakers.split(",") if x)}
    if not n_spk or not all(2 <= k <= 4 and v > 0 for k, v in n_spk.items()) or (a.lang2 and set(n_spk) != {2}): sys.exit(f"--speakers 2–4 의 'n:가중' (교차 언어는 2 만): {a.speakers}")
    G.update(n_spk=n_spk)
    G.update(args=a, pools=pools, persons=persons, stats=stats, weights={k: w for k, w in weights.items() if w > 0}, xx=xx, lang_name=lang_name,
             pools2=pools2, weights2={k: w for k, w in weights2.items() if w > 0})
    if a.noise_bank:
        from vapasr.data.noise_aug import from_args
        G["noise"] = from_args(a.noise_bank, a.noise_p, a.noise_snr, a.rir_p)
    G["tel"] = any(getattr(a, k) != v for k, v in TEL_DEFAULTS.items() if k != "name_suffix"); G["phone"] = None   # tel: stats 에 C2 점검(레벨 차·타이밍)을 더한다
    if a.phone_per_speaker > 0:
        from vapasr.data.phone_aug import PhoneAugment
        G["phone"] = PhoneAugment(p=a.phone_per_speaker, mulaw_p=0.5)
    if a.far_per_speaker > 0:                                                      # D6 화자별 원거리(--noise-bank 의 RIR·방 정보)
        if not a.noise_bank: sys.exit("--far-per-speaker 는 --noise-bank 와 함께")
        from vapasr.data.far_aug import FarFieldAugment
        G["far"] = FarFieldAugment(a.noise_bank, p=1.0, p_noise=0.0, p_real=a.far_real_p)
    import multiprocessing as mp
    lines, rows, trans_all = {}, [], {}
    with mp.get_context("fork").Pool(a.workers) as pool:
        for k, (i, line, st, tr) in enumerate(pool.imap_unordered(make_session, range(a.sessions), chunksize=2)):
            if line: lines[i] = line
            if line and tr and tr[1]: trans_all[i] = tr
            rows.append(st)
            if (k + 1) % 100 == 0: print(f"  {k + 1}/{a.sessions} · {time.time() - t0:.0f}s", flush=True)
    path = os.path.join(a.out, f"sim_{xx}.refined.dialogues.jsonl")
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for i in sorted(lines): f.write(lines[i] + "\n")
    os.replace(path + ".tmp", path)
    n_tr_rows, n_tr_tok = write_targets(a.out, xx, trans_all) if trans_all else (0, 0)
    ok = [r for r in rows if not r.get("failed")]; q = lambda xs, p: round(float(sorted(xs)[int(p * (len(xs) - 1))]), 3) if xs else None
    summ = dict(lang=a.lang, lang2=a.lang2, sessions=len(ok), trans=dict(rows=n_tr_rows, tokens=n_tr_tok, utts=sum(r["utts"] for r in ok),
                utts_translated=sum(r["utts_translated"] for r in ok)) if TR else None, failed=len(rows) - len(ok), hours=round(sum(r["duration_s"] for r in ok) / 3600, 2),
                speech_hours=round(sum(r["speech_s"] for r in ok) / 3600, 2), overlap_ratio_mean=round(sum(r["overlap_ratio"] for r in ok) / max(1, len(ok)), 4),
                changes_per_min_mean=round(sum(r["changes_per_min"] for r in ok) / max(1, len(ok)), 2), bc_per_min=round(60 * sum(r["bc"] for r in ok) / max(1e-6, sum(r["duration_s"] for r in ok)), 2),
                self_overlap=sum(r["self_overlap"] for r in ok), pools=dict(collections.Counter(r["pool"] for r in ok)),
                timing=dict(dialogues=stats.n_dialogues, pairs=stats.n_pairs, p_hold=round(stats.p_hold, 3), bc_rate_per_s=round(stats.bc_rate, 4),
                            fto_q10_50_90=[q(stats.fto, p) for p in (0.1, 0.5, 0.9)], hold_q50=q(stats.hold, 0.5), turn_q10_50_90=[q(stats.turn, p) for p in (0.1, 0.5, 0.9)],
                            fto_negative_share=round(sum(x < 0 for x in stats.fto) / max(1, len(stats.fto)), 3)),
                pool_info=info, errors=dict(collections.Counter(e for r in rows for e in r.get("errors", [])).most_common(20)),
                args={k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(a).items() if k not in TEL_DEFAULTS or v != TEL_DEFAULTS[k]}, sec=round(time.time() - t0, 1))   # C2 옵션은 기본이 아닐 때만
    if G["tel"]:                                                                   # C2 점검: 화자 레벨 차 |Δ|·교대/재개 간격·같은 화자 쉼·채널
        from vapasr.data.dialogue import GAP_S
        R = [r["tel"] for r in ok]; ld = [r["level_diff"] for r in R if r["level_diff"] is not None]; cg, hg, sg = ([x for r in R for x in r[k]] for k in ("change", "hold", "same"))
        CH = [r["chan"] for r in R if r["chan"]]; sp = [v for c in CH for v in c["spk"].values()]; on = [v for v in sp if v["phone"]]; q3 = lambda xs: [q(xs, p) for p in (0.1, 0.5, 0.9)]
        summ["tel"] = dict(level_diff_db=dict(q10_50_90=q3(ld), mean=round(sum(ld) / max(1, len(ld)), 3), hist=_hist(ld, 0.0, 20.0, 1.0)),
                           hold_ratio=round(len(hg) / max(1, len(hg) + len(cg)), 4), changes=len(cg), holds=len(hg), split_turns=sum(r["split_turns"] for r in R),
                           change_gap_s=dict(q10_50_90=q3(cg), min=min(cg, default=None), hist=_hist(cg, -2.0, 3.0, 0.1)), hold_gap_s=dict(q10_50_90=q3(hg), hist=_hist(hg, 0.0, 3.0, 0.25)),
                           same_spk_gap_s=dict(under_gap=int(sum(x < GAP_S for x in sg)), hist=_hist(sg, 0.0, 1.0, 0.05)),
                           chan=dict(sessions=len(CH), same=sum(c["same"] for c in CH), tracks=len(sp), phone=len(on), mulaw=sum(v["mulaw"] for v in on),
                                     low_q10_50_90=q3([v["low"] for v in on]), high_q10_50_90=q3([v["high"] for v in on])) if CH else None)
    json.dump(summ, open(os.path.join(a.out, f"sim_{xx}.stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k not in ("args", "pool_info")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

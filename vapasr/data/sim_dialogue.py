"""합성 2 인 대화(Stage 3 화자 귀속 보강) — 서로 다른 단일 화자 발화를 두 화자의 턴으로 번갈아 놓아 Dialogue 를 만든다.

  재료 Unit = 한 화자의 발화 하나: 오디오 참조(path, src_offset, dur) + 토큰 [(id, 발화 오디오 안 끝 시각)] + 화자 키(같은 키 = 같은 사람·같은 녹음).
    · Stage 1·2 words 행(units_from_words_row): 화자 키를 아는 세트만 쓴다(SPEAKER_RULES) — LibriSpeech(화자-장), Switchboard(대화 쪽),
      VoxPopuli(연설), CommonVoice(speaker). 여러 구간 행은 구간마다 나누고 토큰은 시각으로 나눈다. 한국어 행은 replay 가 학습 때 KO 정규화를
      거치므로 여기서 쓰지 않는다(한국어는 Phase 2 대화 재료).
    · Phase 2 정제 대화(units_from_dialogue): 전사 감독 발화(tokens 있고 flags 없음), 화자 키 = 코퍼스|대화|화자. 화자 채널(또는 조각)의 그 구간.
      같은 사람이 여러 대화에 나오므로(AI Hub 화자 번호·ZTC 이니셜·NSC 상대역) 짝은 persons_of 가 겹치지 않는 두 대화에서 고른다.
  시간 구조(TimingStats, timing_stats): 같은 언어 실제 대화의 episode 에서 잰다 — 화자 교대 간격(FTO, 음수 = 겹침), 같은 화자 재개 간격·비율,
    턴(episode) 길이, 맞장구(≤ 1 s · ≤ 3 토큰) 빈도(상대 말 1 초당)·상대 턴 안 위치.
  합성(synth_session): 화자 둘(같은 풀의 다른 키)의 발화를 각자 순서대로 꺼내 턴 길이 목표만큼 이어 붙인 턴을, 재개 확률(SimConfig.p_hold — 실제
    대화의 재개 비율은 쉼으로 나뉜 같은 화자 말까지 세서 합성 턴에 쓰면 교대가 성기다)로 같은 화자 또는 교대로 놓는다. 교대 간격·재개 간격·턴 길이·
    맞장구 위치는 실제 분포에서 뽑고 맞장구 빈도는 bc_scale 배. 긴 발화(LibriSpeech·VoxPopuli)는 안쪽 쉼에서 구 단위로 나눈다(split_points). 발화 앞뒤 무음은 에너지로 잘라(speech_span — 토큰 시각 보호) 간격·겹침이 말소리
    기준이 되게 한다. 한 화자는 자기 말과 겹치지 않는다(episode 병합 gap 0.25 s 도 지킨다).
  결과 Dialogue: channels = 화자별 조각 [(path, 대화 시각, src_offset, dur)](dialogue_mix.load_channel 규약), utterances = 놓인 발화(토큰은
    대화 시각), meta = synthetic "sim" · mask_eot(턴 타이밍이 인공적이라 EOT 감독 없음 — dialogue_stitch 와 같은 원칙) · 화자 키 · 풀.
  번역(Stage 4, 선택): Unit.trans = {목표 언어: [(id, 발화 오디오 안 확정 시각)]} — words 행은 Stage 2 타깃(행 id = words 행 id)을 구간 시각으로,
    Phase 2 발화는 S1 대화 타깃(행 id = conv|화자|원 utt_id)을 조각 구간으로 나눠 붙인다. 토큰과 같은 규칙으로 나누고(split_unit) 놓는다(synth_session) —
    결과는 meta.trans = {utt_id: {목표 언어: [[id, 대화 시각], …]}}(sim_build_dialogues 가 떼어 번역 타깃 파일로 쓴다).
    행 무결성: 한 행의 번역은 뒤 조각에서 확정되기도 한다(한국어는 행 끝에 몰린다). 그래서 이어지는 조각(Unit.cont)은 세션 시작(sim_build_dialogues)·
    맞장구로 쓰지 않고, 조각이 빠진 행은 번역을 떼어 낸다 — 말하지 않은 내용의 번역이 끼지 않게(2026-10-19 시험 합성에서 확인).
"""
import math, random, re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .dialogue import Dialogue, Utterance, ChannelRef, build_episodes, clamp_to_spans, GAP_S

SR = 16000


@dataclass
class Unit:
    key: str; order: float; path: str; src_offset: Optional[float]; dur: float
    tokens: List[Tuple[int, float]]                       # (id, 발화 오디오(src_offset 부터) 안 끝 시각 s)
    text: str = ""; uid: str = ""; pool: str = ""
    trans: Dict[str, List[Tuple[int, float]]] = field(default_factory=dict)   # 목표 언어 → [(id, 발화 오디오 안 확정 시각 s)](Stage 4, 없으면 비움 — 키만 있고 빈 목록 = 이 조각에서 확정된 번역 없음)
    cont: bool = False                                                         # 앞 조각에 이어지는 조각(정제 #n 의 둘째 이후·split_unit 의 둘째 이후) — 세션 시작·맞장구로 쓰지 않는다

    def short(self, max_tokens: int = 3, max_s: float = 1.6) -> bool:
        """맞장구 후보: 토큰 ≤ max_tokens 이고 마지막 토큰이 max_s 안(앞 무음 포함 대략치)."""
        return len(self.tokens) <= max_tokens and self.tokens[-1][1] <= max_s


# ── Stage 1·2 words 행
def _swbd_time(s: str, e: str, dur: float) -> float:
    """Switchboard utt id 의 시각(소수점 없는 숫자열 — 정수 자리 수가 다르다) → 길이가 dur 에 가장 맞는 시작 시각(s)."""
    best = (1e9, 0.0)
    for i in range(1, min(5, len(s))):
        for j in range(1, min(5, len(e))):
            a, b = float(s[:i] + "." + s[i:]), float(e[:j] + "." + e[j:])
            best = min(best, (abs(b - a - dur), a))
    return best[1]


def _ls(utt, row, seg):
    m = re.fullmatch(r"(\d+)-(\d+)-(\d+)", utt); return (f"ls:{m.group(1)}-{m.group(2)}", float(m.group(3))) if m else None


def _swbd(utt, row, seg):
    m = re.fullmatch(r"(sw\d+[AB])_(\d+)_(\d+)", utt); return (f"swbd:{m.group(1)}", _swbd_time(m.group(2), m.group(3), seg["dur_s"])) if m else None


def _vp(utt, row, seg):
    m = re.fullmatch(r"(.+)_(\d+)", utt); return (f"vp:{m.group(1)}", float(m.group(2))) if m else None


def _cv(utt, row, seg):
    s = row.get("speaker"); return (f"cv:{s}", float(abs(hash(utt)) % 100000)) if s else None


SPEAKER_RULES: Dict[str, Callable] = {"librispeech-960": _ls, "librispeech-960-rg": _ls, "swbd-train": _swbd, "swbd-train-rg": _swbd,
                                       "voxpopuli-train": _vp, "cv4-en": _cv}


def units_from_words_row(r: dict, trans: Optional[Dict[str, tuple]] = None) -> List[Unit]:
    """words 행 → 구간별 Unit(화자 키를 아는 세트만). 토큰은 행 시각 (offset, offset + dur + 0.08] 로 구간에 나눈다.
    trans: 이 행의 번역 타깃 {목표 언어: (ids, 행 시각)}(Stage 2 — 번역 단위는 구간을 넘지 않는다) — 토큰과 같은 규칙으로 구간에 나눈다."""
    rule = SPEAKER_RULES.get(r.get("set"))
    if rule is None or r.get("training_eligible") is False: return []
    toks = sorted(((int(t), float(x)) for t, x in (r.get("tokens") or [])), key=lambda z: z[1]); out = []; used = 0
    trs = {g: sorted(((int(i), float(t)) for i, t in zip(*v)), key=lambda z: z[1]) for g, v in (trans or {}).items()}; tr_used = {g: 0 for g in trs}
    def take(seq, j, a, b):
        got = []
        while j < len(seq) and seq[j][1] <= b + 0.08:
            if seq[j][1] > a - 1e-3: got.append((seq[j][0], round(max(0.0, seq[j][1] - a), 3)))
            j += 1
        return got, j
    for g in r["segments"]:
        k = rule(g["utt_id"], r, g)
        a, b = float(g["offset_s"]), float(g["offset_s"]) + float(g["dur_s"])
        tt, used = take(toks, used, a, b); tr = {}
        for lg, seq in trs.items(): tr[lg], tr_used[lg] = take(seq, tr_used[lg], a, b)
        if k is None or not tt: continue
        out.append(Unit(key=k[0], order=k[1], path=g["path"], src_offset=g.get("src_offset_s"), dur=float(g["dur_s"]), tokens=tt, text=g.get("raw_text", ""),
                        uid=g["utt_id"], pool=r["set"], trans=tr))
    return out


# ── Phase 2 정제 대화
def dialogue_piece_trans(d: Dialogue, trans: Dict[str, Dict[str, tuple]]) -> Dict[str, Dict[str, List[Tuple[int, float]]]]:
    """S1 대화 번역 타깃(행 = 같은 화자의 원 utt_id 로 묶은 정제 조각 #n)을 조각마다 나눈다 → {utt_id: {목표 언어: [(id, 대화 시각)]}}.
    확정 시각은 행 조각 구간으로 자르고(clamp_to_spans — cst_dialogue 번역 흐름과 같은 규칙) 그 시각 이후 처음 끝나는 조각에 준다.
    번역이 있는 행의 조각은 확정 토큰이 없어도 빈 목록을 받는다(한국어처럼 행 끝에서 몰아 확정되는 경우 — 번역 없음과 구별)."""
    rows: Dict[Tuple[str, str], list] = {}
    for u in d.utterances:
        if u.tokens and not u.flags and u.end > u.start: rows.setdefault((u.speaker, u.utt_id.split("#")[0]), []).append((u.start, u.end, u.utt_id))
    out: Dict[str, Dict[str, list]] = {}
    for (spk, base), ps in rows.items():
        tt = trans.get(f"{d.conv_id}|{spk}|{base}")
        if not tt: continue
        ps.sort(); spans = [(a, b) for a, b, _ in ps]
        for g, (ids, ts) in tt.items():
            for _, _, uid in ps: out.setdefault(uid, {}).setdefault(g, [])
            for i, t in zip(ids, ts):
                c = clamp_to_spans(float(t), spans); j = next((j for j, (_, b) in enumerate(spans) if c <= b + 1e-6), len(spans) - 1)
                out.setdefault(ps[j][2], {}).setdefault(g, []).append((int(i), c))
    return out


def units_from_dialogue(d: Dialogue, margin: float = 0.1, trans: Optional[Dict[str, Dict[str, tuple]]] = None) -> List[Unit]:
    """전사 감독 발화(tokens 있고 flags 없음) → Unit. 오디오 = 화자 채널(또는 그 시각에 가장 가까운 조각)의 [start − margin, end + margin].
    trans: S1 대화 번역 타깃 전체({행 id: {목표 언어: (ids, 대화 시각)}}) — 조각마다 나눠(dialogue_piece_trans) Unit.trans 로 붙인다."""
    out = []; pt = dialogue_piece_trans(d, trans) if trans else {}; first = {}; broken = set()
    for u in sorted((u for u in d.utterances if u.tokens and not u.flags and u.end > u.start), key=lambda u: u.start):
        first.setdefault((u.speaker, u.utt_id.split("#")[0]), u.utt_id)
    for u in d.utterances:
        if not u.tokens or u.flags or u.end <= u.start: continue
        ref = d.channels.get(u.speaker)
        if ref is None: continue
        if ref.pieces:
            pc = min(ref.pieces, key=lambda p: abs(p[1] - u.start)); path, off = pc[0], float(pc[1]); base = float(pc[2]) if len(pc) >= 4 else 0.0
            s0 = max(0.0, u.start - off - margin); src = base + s0; t0 = off + s0
        elif ref.path:
            path = ref.path; src = max(0.0, u.start - margin); t0 = src
        else: continue
        dur = (u.end + margin) - t0
        tt = [(int(t), round(float(x) - t0, 3)) for t, x in u.tokens]
        row = (u.speaker, u.utt_id.split("#")[0])
        if dur <= 0.1 or not tt or tt[0][1] < 0: broken.add(row); continue
        tr = {g: [(i, round(t - t0, 3)) for i, t in v] for g, v in pt.get(u.utt_id, {}).items()}
        out.append(Unit(key=f"{d.corpus}|{d.conv_id}|{u.speaker}", order=u.start, path=path, src_offset=round(src, 3), dur=round(dur, 3), tokens=tt,
                        text=u.text, uid=u.utt_id, pool=d.corpus, trans=tr, cont=first.get(row) != u.utt_id))
    for x in out:                                                                              # 조각이 빠진 행은 번역을 떼어 낸다(빠진 조각의 내용이 다음 조각 번역에 섞인다)
        if x.trans and (x.key.split("|")[2], x.uid.split("#")[0]) in broken: x.trans = {}
    return out


def persons_of(d: Dialogue) -> set:
    """대화에 나오는 사람 id 집합 — AI Hub 대화 id 의 화자 번호(예 6450G2A5) + 숫자만이 아닌 화자 id(ZTC·KEC·NSC). 비면 대화 id 로 대신."""
    ps = set(re.findall(r"[0-9A-Z]{4}G\dA\d", d.conv_id)) | {s for s in d.speakers if not s.isdigit()}
    return ps or {d.conv_id}


def person_ids(d: Dialogue, speaker: str) -> set:
    """화자 한 명의 사람 id — NSC 는 역할 이름(speaker·interlocutor)이라 세션 이름 '<화자>_<상대역>' 에서 읽는다(같은 상대역이 여러 세션에 나온다),
    이름 붙은 화자 id(KEC·ZTC)는 그대로, 위치 번호(0·1 — AI Hub·otoSpeech)는 어느 사람인지 몰라 대화의 사람 전부(persons_of)."""
    if d.corpus == "nsc" and speaker in ("speaker", "interlocutor") and "_" in d.conv_id.split(":", 1)[-1]:
        sp, inter = d.conv_id.split(":", 1)[-1].split("_", 1); return {f"nsc:{sp if speaker == 'speaker' else inter}"}
    if not speaker.isdigit(): return {f"{d.corpus}:{speaker}"}
    return persons_of(d)


def speech_span(x: np.ndarray, tokens: Sequence[Tuple[int, float]], sr: int = SR) -> Tuple[float, float]:
    """발화 오디오의 말소리 구간(s): 20 ms 창 에너지가 max(바닥 + 10 dB, 최대 − 40 dB) 이상인 첫·끝 창 ± 여유.
    토큰 보호: 시작은 첫 토큰 끝 − 0.25 s 보다 늦지 않고, 끝은 마지막 토큰 끝 + 0.1 s 보다 이르지 않다."""
    T = len(x) / sr; fl = int(0.02 * sr); hp = int(0.01 * sr)
    if len(x) < fl * 2: return 0.0, T
    n = 1 + (len(x) - fl) // hp; idx = np.arange(fl)[None] + hp * np.arange(n)[:, None]
    db = 10 * np.log10(np.mean(x[idx].astype(np.float64) ** 2, axis=1) + 1e-12)
    thr = max(float(np.percentile(db, 10)) + 10.0, float(db.max()) - 40.0); on = np.nonzero(db >= thr)[0]
    if len(on) == 0: return 0.0, T
    a = max(0.0, on[0] * hp / sr - 0.05); b = min(T, (on[-1] * hp + fl) / sr + 0.08)
    first, last = tokens[0][1], tokens[-1][1]
    return round(min(a, max(0.0, first - 0.25)), 3), round(max(b, min(T, last + 0.1)), 3)


# ── 실제 대화의 시간 구조
@dataclass
class TimingStats:
    fto: List[float]; hold: List[float]; p_hold: float; turn: List[float]; bc_rate: float; bc_pos: List[float]
    n_dialogues: int = 0; n_pairs: int = 0; n_bc: int = 0
    def pick(self, rng, xs, lo, hi): return min(hi, max(lo, xs[rng.randrange(len(xs))]))


def timing_stats(dialogues: Sequence[Dialogue], bc_s: float = 1.0, bc_tokens: int = 3) -> TimingStats:
    """episode(전사 없는 발화 포함 — 말소리 시각) 로 잰다. 맞장구 = ≤ bc_s 초·≤ bc_tokens 토큰. 나머지(주 episode) 의 연이은 쌍에서
    화자가 바뀌면 FTO(다음 시작 − 앞 끝), 같으면 재개 간격. 맞장구는 겹치거나 0.5 s 안에 붙는 상대 주 episode 를 host 로 상대 위치를 잰다."""
    fto, hold, turn, pos = [], [], [], []; n_pairs = n_hold = n_bc = 0; speech = 0.0
    for d in dialogues:
        eps = build_episodes(d); bc = [e.end - e.start <= bc_s and len(e.tokens) <= bc_tokens for e in eps]
        main = [e for e, b in zip(eps, bc) if not b]
        for p, q in zip(main, main[1:]):
            n_pairs += 1
            if q.speaker != p.speaker: fto.append(q.start - p.end)
            else: hold.append(q.start - p.end); n_hold += 1
        turn += [e.end - e.start for e in main]; speech += sum(e.end - e.start for e in main)
        for e, b in zip(eps, bc):
            if not b: continue
            h = next((m for m in main if m.speaker != e.speaker and m.start <= e.start <= m.end + 0.5), None)
            if h is not None: n_bc += 1; pos.append((e.start - h.start) / max(0.3, h.end - h.start))
    return TimingStats(fto=fto or [0.3], hold=hold or [0.5], p_hold=n_hold / max(1, n_pairs), turn=turn or [3.0], bc_rate=n_bc / max(1.0, speech),
                       bc_pos=pos or [0.5], n_dialogues=len(dialogues), n_pairs=n_pairs, n_bc=n_bc)


# ── 합성
def split_points(x: np.ndarray, tokens: Sequence[Tuple[int, float]], max_s: float = 4.0, min_pause: float = 0.2, sr: int = SR) -> List[Tuple[float, float]]:
    """긴 발화를 안쪽 쉼(에너지가 말소리 문턱 아래인 ≥ min_pause 구간)에서 구 단위로 나눌 구간 [(a, b)] — 각 구간 ≤ max_s 가 되도록 앞에서부터
    가장 늦은 쉼에서 자른다. 쉼 가운데를 경계로 하고 토큰이 없는 조각은 버린다. 나눌 곳이 없으면 [(0, 길이)]."""
    T = len(x) / sr; fl = int(0.02 * sr); hp = int(0.01 * sr)
    if T <= max_s or len(x) < fl * 4: return [(0.0, T)]
    n = 1 + (len(x) - fl) // hp; idx = np.arange(fl)[None] + hp * np.arange(n)[:, None]
    db = 10 * np.log10(np.mean(x[idx].astype(np.float64) ** 2, axis=1) + 1e-12); thr = max(float(np.percentile(db, 10)) + 10.0, float(db.max()) - 40.0)
    quiet = db < thr; pauses = []; i = 0
    while i < n:
        if quiet[i]:
            j = i
            while j + 1 < n and quiet[j + 1]: j += 1
            a, b = i * hp / sr, (j * hp + fl) / sr
            if b - a >= min_pause and a > 0.3 and b < T - 0.3: pauses.append((a + b) / 2)
            i = j + 1
        else: i += 1
    cuts, start = [], 0.0
    while True:
        cand = [p for p in pauses if start + 0.8 <= p <= start + max_s]
        if T - start <= max_s or not cand: break
        start = cand[-1]; cuts.append(start)
    bounds = [0.0] + cuts + [T]; ts = [t for _, t in tokens]
    return [(round(a, 3), round(b, 3)) for a, b in zip(bounds, bounds[1:]) if any(a < t <= b for t in ts)] or [(0.0, T)]


def split_unit(u: "Unit", parts: Sequence[Tuple[float, float]]) -> List["Unit"]:
    """split_points 의 구간으로 Unit 을 나눈다(src_offset·dur·토큰 시각을 그 구간 기준으로). 한 조각이면 그대로."""
    if len(parts) <= 1: return [u]
    out = []
    for k, (a, b) in enumerate(parts):
        tt = [(tid, round(t - a, 3)) for tid, t in u.tokens if a < t <= b]
        lo = -1e9 if k == 0 else a                                                              # 번역 확정 시각은 첫 조각이 앞쪽(≤ 0)도 받는다
        tr = {g: [(i, round(max(0.0, t - a), 3)) for i, t in v if lo < t <= b] for g, v in u.trans.items()}
        if tt: out.append(Unit(key=u.key, order=u.order + k * 1e-3, path=u.path, src_offset=round((u.src_offset or 0.0) + a, 3), dur=round(b - a, 3), tokens=tt,
                               text=u.text if k == 0 else "", uid=f"{u.uid}~{k}", pool=u.pool, trans=tr, cont=u.cont if k == 0 else True))
    return out or [u]


@dataclass
class SimConfig:
    p_hold: Optional[float] = 0.3                  # 같은 화자 재개 확률(None = 실제 대화 값 — 실제 값은 쉼으로 나뉜 같은 화자 말까지 세서 합성 턴에는 크다)
    bc_scale: float = 1.5                          # 맞장구 빈도 배율(짧은 발화가 드문 풀에서 실제보다 성기다)
    fto: Tuple[float, float] = (-2.0, 3.0)          # 뽑은 교대 간격을 이 범위로 자른다
    hold: Tuple[float, float] = (GAP_S, 3.0)        # 같은 화자 재개 간격
    turn: Tuple[float, float] = (0.8, 15.0)         # 턴 길이 목표
    intra_pause: Tuple[float, float] = (0.1, 0.5)   # 한 턴 안 발화 사이 쉼(< 0.25 s 면 한 episode 로 합쳐진다)
    max_turn_units: int = 6
    lead_s: float = 0.5; tail_s: float = 0.5
    min_start_after_other: float = 0.3             # 교대 턴은 앞 턴 시작 + 이 값 이후에 시작(앞 턴을 통째로 덮지 않게)
    bc_min_host_s: float = 1.5                     # 이보다 긴 턴에만 맞장구
    p_back: float = 0.2                            # 3 명 이상: 교대할 때 바로 전 상대(앞앞 턴 화자)에게 되돌아갈 확률(나머지는 다른 화자 중 균등) — 둘씩 주고받는 경향.
    p_interrupt: float = 0.0                       # 끼어들기(2026-10-26 데이터 확장 D6): 교대 때 이 확률로 교대 간격을 interrupt_fto 균등에서 뽑는다(음수 = 앞 턴과 겹침). 0 이면 난수를 더 뽑지 않는다
    interrupt_fto: Tuple[float, float] = (-2.0, -0.3)
    p_full: float = 0.0                            # 완전 겹침(2026-11-08, 사용자 "완전 겹치는 케이스도 추가하고"): 교대 때 이 확률로 앞 턴 시작 + U(full_start) 에 시작한다
    full_start: Tuple[float, float] = (0.0, 0.5)   # — 두 화자가 거의 함께 시작해 짧은 쪽 턴이 통째로 겹친다(min_start_after_other 를 건너뛴다). 0 이면 난수를 더 뽑지 않는다
                                                   # 0.4 면 90 s 4 인 세션의 28 % 에서 한 화자가 발화 1 개 이하(2026-10-20 모의), 0.2 는 16 %


def synth_session(units: Sequence[Sequence[Unit]], stats: TimingStats, rng: random.Random, target_s: float, span: Callable[[Unit], Tuple[float, float]],
                  conv_id: str, corpus: str, lang: str, cfg: SimConfig = SimConfig(), meta: Optional[dict] = None,
                  expand: Optional[Callable[[Unit], List[Unit]]] = None) -> Dialogue:
    """units: 화자 N(2–4)의 발화(각자 쓸 순서대로). span(u) → 그 발화 오디오의 말소리 구간(s, 발화 오디오 기준). → Dialogue(화자 "a"·"b"[·"c"·"d"]).
    3 명 이상(V2): 교대하면 cfg.p_back 확률로 앞앞 턴 화자, 아니면 다른 화자 중 균등. 맞장구 host 의 상대도 다른 화자 중 균등. 2 명은 예전과 같은 난수열이다.
    expand(u) → 긴 발화를 구 단위로 나눈 조각들(split_points·split_unit) — 첫 조각을 지금 쓰고 나머지는 그 화자 줄 맨 앞에 되돌린다.
    턴을 놓을 화자의 발화가 떨어지면 끝낸다(혼자 말하는 꼬리를 만들지 않는다). 맞장구는 짧은 발화(Unit.short)로만, 쓰면 그 발화는 턴에 다시 안 쓴다."""
    N = len(units); assert 2 <= N <= 4, f"화자 2–4: {N}"
    names = ("a", "b", "c", "d")[:N]; q = [list(u) for u in units]; p_hold = stats.p_hold if cfg.p_hold is None else cfg.p_hold
    utts: List[Utterance] = []; pieces: Dict[str, list] = {x: [] for x in names}; last_end = [-1e9] * N; n = {"turns": 0, "bc": 0, "units": 0, "split": 0}
    trans: Dict[str, Dict[str, list]] = {}
    def take(s: int) -> Optional[Unit]:
        if not q[s]: return None
        u = q[s].pop(0)
        if expand is not None:
            subs = expand(u)
            if len(subs) > 1: n["split"] += 1; q[s][:0] = subs[1:]; u = subs[0]
        return u
    def take_short(s: int) -> Optional[Unit]:                                            # 이어지는 조각은 맞장구로 앞당기지 않는다(번역 순서)
        i = next((i for i, u in enumerate(q[s]) if u.short() and not u.cont), None)
        return None if i is None else q[s].pop(i)
    def place(s: int, u: Unit, at: float) -> float:
        a, b = span(u); d = round(b - a, 3); at = round(at, 3)
        toks = [(tid, round(at + min(max(t - a, 0.0), d), 3)) for tid, t in u.tokens]
        utts.append(Utterance(speaker=names[s], start=at, end=round(at + d, 3), text=u.text, raw=u.text, tokens=toks, utt_id=f"{names[s]}:{u.uid}"))
        if u.trans: trans[utts[-1].utt_id] = {g: [[tid, round(at + min(max(t - a, 0.0), d), 3)] for tid, t in v] for g, v in u.trans.items()}
        pieces[names[s]].append((u.path, at, round((u.src_offset or 0.0) + a, 3), d)); n["units"] += 1
        return at + d
    prev = prev2 = None; p_start = p_end = 0.0
    def other(x, back):                                                                 # x 가 아닌 화자(2 명이면 난수 없이 상대)
        if N == 2: return 1 - x
        if back is not None and back != x and rng.random() < cfg.p_back: return back
        return rng.choice([y for y in range(N) if y != x])
    while True:
        s = rng.randrange(N) if prev is None else (prev if rng.random() < p_hold else other(prev, prev2))
        if prev is None: start = cfg.lead_s
        elif s != prev:
            if cfg.p_full > 0 and rng.random() < cfg.p_full:                             # 완전 겹침: 앞 턴과 거의 함께 시작
                start = p_start + rng.uniform(*cfg.full_start); n["full"] = n.get("full", 0) + 1
            else:
                f = rng.uniform(*cfg.interrupt_fto) if cfg.p_interrupt > 0 and rng.random() < cfg.p_interrupt else stats.pick(rng, stats.fto, *cfg.fto)
                start = max(p_end + f, p_start + cfg.min_start_after_other)
        else: start = last_end[s] + stats.pick(rng, stats.hold, *cfg.hold)
        start = max(start, last_end[s] + GAP_S)
        if start >= target_s: break
        D = stats.pick(rng, stats.turn, *cfg.turn); cur = start; k = 0
        while k < cfg.max_turn_units and (k == 0 or cur - start < D):
            u = take(s)
            if u is None: break
            if k: cur += rng.uniform(*cfg.intra_pause)
            cur = place(s, u, cur); k += 1
        if k == 0: break
        last_end[s] = cur; n["turns"] += 1; o = other(s, None)
        if cur - start >= cfg.bc_min_host_s and rng.random() < 1.0 - math.exp(-cfg.bc_scale * stats.bc_rate * (cur - start)):
            u = take_short(o)
            if u is not None:
                a, b = span(u); at = max(start + min(0.95, max(0.15, stats.pick(rng, stats.bc_pos, 0.0, 1.2))) * (cur - start), last_end[o] + GAP_S)
                if at + (b - a) <= cur + 0.5: last_end[o] = place(o, u, at); n["bc"] += 1
                else: q[o].insert(0, u)
        if prev != s: prev2 = prev
        prev = s; p_start, p_end = start, cur
    end = max((u.end for u in utts), default=0.0) + cfg.tail_s
    utts.sort(key=lambda u: (u.start, u.speaker))
    return Dialogue(conv_id=conv_id, corpus=corpus, lang=lang, split="train", duration_s=round(end, 3), speakers=list(names), utterances=utts,
                    channels={k: ChannelRef(pieces=v) for k, v in pieces.items()},
                    meta=dict(synthetic="sim", mask_eot=True, **n, **(meta or {}), **({"trans": trans} if trans else {})))


def overlap_stats(d: Dialogue) -> dict:
    """합성 결과 점검: 말소리 시간, 두 화자가 함께 말한 시간 비율, 분당 화자 교대(episode 기준), 화자별 자기 겹침 수(0 이어야 한다)."""
    eps = build_episodes(d); by: Dict[str, list] = {}
    for e in eps: by.setdefault(e.speaker, []).append((e.start, e.end))
    sp = list(by); ov = sum(max(0.0, min(b0, b1) - max(a0, a1)) for x in range(len(sp)) for y in range(x + 1, len(sp)) for a0, b0 in by[sp[x]] for a1, b1 in by[sp[y]])   # 화자 쌍마다 함께 말한 시간
    self_ov = 0
    for spk in d.speakers:
        us = sorted((u for u in d.utterances if u.speaker == spk), key=lambda u: u.start)
        self_ov += sum(1 for x, y in zip(us, us[1:]) if x.end > y.start + 1e-6)
    speech = sum(e.end - e.start for e in eps); ch = sum(1 for p, q in zip(eps, eps[1:]) if p.speaker != q.speaker)
    return dict(duration_s=d.duration_s, speech_s=round(speech, 2), overlap_s=round(ov, 2), overlap_ratio=round(ov / max(1e-6, d.duration_s), 4),
                changes_per_min=round(ch / max(1e-6, d.duration_s / 60), 2), self_overlap=self_ov, episodes=len(eps))


def turn_gaps(d: Dialogue) -> dict:
    """턴 타이밍 점검(C2 전화 합성 stats): episode 를 시작 순으로 이은 쌍의 교대 간격(change — 다음 시작 − 앞 끝, 음수 = 겹침)·같은 화자 재개 간격(hold),
    화자마다 이어진 발화 사이 쉼(same — GAP_S 미만이면 한 episode 안의 쉼), episode 수."""
    eps = build_episodes(d); change, hold, same = [], [], []
    for p, q in zip(eps, eps[1:]): (change if q.speaker != p.speaker else hold).append(round(q.start - p.end, 3))
    for spk in d.speakers:
        us = sorted((u for u in d.utterances if u.speaker == spk), key=lambda u: u.start); same += [round(y.start - x.end, 3) for x, y in zip(us, us[1:])]
    return dict(change=change, hold=hold, same=same, episodes=len(eps))

"""Semantic commit(<SEM_END>, 선택적 턴 종료) 평가 지표 — 계획 §24 (raw/inbox/streaming_asr_semantic_commit_plan.md).
턴 종료 토큰: 새 모델은 Phase 2 와 같은 <EOT>(151722, --turn-end 로 학습했을 때만), v0.2 체크포인트는 <TURN_END>(151724) — 평가 행의 event_ids 가 어느 쪽인지 기록한다.

입력: 스트리밍 디코드 방출 [(chunk k, token id)] + words.jsonl 행(참조 단어·종료 시각·태그) + labels.jsonl 행(후보 경계 등급 A/B/N).
시각 규약 — 둘 다 보고한다:
  (a) 직렬화 청크 ref_k = int(word_end/0.08)+δ (학습 시퀀스에서 그 단어의 마지막 토큰과 <SEM_END> 가 놓이는 청크. interleave.build_interleaved 와
      같은 산술(ε 없음), ≥K 는 flush 청크 K 로 자른다). P/R/F1 은 비대칭 창 ref_k−early ≤ hyp_k ≤ ref_k+late (청크 단위) 의 최적 1:1 매칭.
  (b) 음향 완결 = word_end. 지연 latency = hyp_time − word_end, hyp_time = (k+1)·0.08 (청크 k 까지 오디오를 본 시각; flush 라운드는 K·0.08).
텍스트 위치: 가설 단어(이벤트 사이 디코드 토큰) ↔ 참조 단어 편집 정렬(align_pairs: 편집 수 최소 중 일치 최대 DP 하나 — rapidfuzz 유무와 무관) → 가설 <SEM_END> 를 '참조 단어 i 뒤' 로 사상.
  A 경계 = 정답, B = 애매(오류로 세지 않고 따로), N 경계(REVISION·all-WAIT·disfluency)·경계 아닌 단위 내부·단어 조각 사이 = 조기 commit(PCR), 범주별.
  가설이 참조 단어를 빠뜨린(deletion) 자리 바로 뒤 이벤트는, 빠진 단어의 ref_k ≤ 이벤트 청크이면 그 단어 뒤로 본다(이상적 직렬화라면 이미 나왔을 단어).
턴 종료: 참조 k_turn = max(int(t_last/0.08)+δ, int((t_last+hangover)/0.08)) (v0 계약), 지연은 t_last(마지막 단어 끝) 기준. turn_id=None 이면 턴 이벤트를 세지 않는다.
  labels 행에 turn_end 가 없으면 True(turn_flag — SemCommitDataset.build_semcommit_tokens·패딩과 같은 기본).
sem-guard 가 막은 <SEM_END>(guard_blocked, 디코더가 센다)는 text.pcr_raw 에서 조기 commit 으로 센다(pcr 은 실제 방출만).
WER/CER: textnorm score_en/score_ko (이벤트 토큰은 디코드 전에 뺀다 — textnorm._TAG 는 <...> 를 공백으로 바꾸므로 단어 중간 이벤트가 남아 있으면 단어가 쪼개진다).
집계: 스트림별 카운트·목록을 합산(micro)한 뒤 finalize 로 비율·분위수. 순수 python(+numpy; scipy 는 매칭, rapidfuzz 는 WER/CER S/D/I 카운트에만 — 있으면 사용).
"""
import copy, re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np

CHUNK_S = 0.08
SEM_END_ID = 151723                                        # Phase1(151705–151716)+Phase2(151717–151722) 뒤에 add_tokens — 전역 계약
EOT_ID = 151722                                            # 턴 종료 = Phase 2 <EOT>(FROZEN_REGISTRY) — mono 에서는 --turn-end 로 학습했을 때만
LEGACY_TURN_END_ID = 151724                                # v0.2 <TURN_END>(폐기) — 옛 평가 행·체크포인트 채점용
SEM_SPECIALS = ["<SEM_END>"]
EARLY, LATES, HANGOVER_S = 1, (0, 2, 5), 0.48
PCR_CATEGORIES = ("filler", "rep", "unclear", "repair", "revision", "all_wait", "other_n", "other_inside", "mid_word", "no_word")
_EPS = 1e-6

# ───────────────────────────── 시각 ─────────────────────────────
def ref_chunk(t: float, delta: int, K: Optional[int] = None) -> int:
    """단어(종료 t 초)의 직렬화 청크 — build_interleaved 와 같은 int(t/0.08)+δ, K 가 있으면 flush(=K) 로 자름."""
    k = int(t / CHUNK_S) + delta
    return k if K is None else min(K, k)

def emit_time(k: int, K: Optional[int] = None) -> float:
    """청크 k 에서 방출된 이벤트의 시각 = 그 청크 끝 (k+1)·0.08. flush 라운드(k ≥ K)는 오디오 끝 K·0.08."""
    return round(((k if K is None else min(k, K - 1)) + 1) * CHUNK_S, 6)

def turn_ref_chunk(t_last: float, delta: int, hangover_s: float = HANGOVER_S, K: Optional[int] = None) -> int:
    k = max(int(t_last / CHUNK_S) + delta, int((t_last + hangover_s) / CHUNK_S))
    return k if K is None else min(K, k)

def turn_flag(lrow: Optional[dict]) -> bool:
    """labels 행의 turn_end — 없으면(키·행 모두) True: SemCommitDataset.build_semcommit_tokens 가 TURN 을 넣는 기본과 같다(패딩·채점·oracle 공통)."""
    return bool((lrow or {}).get("turn_end", True))

def train_pad_K(K0: int, t_last: Optional[float], delays: Sequence[int], has_turn: bool, hangover_s: float = HANGOVER_S, tail_margin: int = 2, pad_tail: bool = True) -> int:
    """학습(vapasr/data/semcommit_dataset.SemCommitDataset, M=0)과 같은 패딩 청크 수:
    K' = max(K0, max_δ 마지막 방출 청크 + tail_margin), 마지막 방출 청크 = int(t_last/0.08)+δ (TURN 이면 turn_ref_chunk 와 max). pad_tail=False 면 TURN 없는 행은 안 늘린다.
    → TURN 은 k_turn(δ) ≤ K'−tail_margin 이라 실제 무음 청크에서 <NEXT_AUDIO> 와 경쟁하고, 마지막 단어·SEM 도 <EMPTY_AUDIO> flush 로 가지 않는다."""
    if t_last is None or not delays or not (has_turn or pad_tail): return int(K0)
    last = max((turn_ref_chunk(t_last, d, hangover_s) if has_turn else int(t_last / CHUNK_S) + d) for d in delays)
    return max(int(K0), last + tail_margin)

def eval_audio_len(wrow: dict, delays: Sequence[int], has_turn: bool, hangover_s: float = HANGOVER_S, tail_margin: int = 2, pad_tail: bool = True) -> Tuple[float, int]:
    """평가 오디오 (길이 s, 청크 수 K') = 학습과 같은 값: 늘렸으면 (K'·0.08, K'), 아니면 (원래 duration_s, words 행 K) — SemCommitDataset 과 같은 식.
    K 는 오디오 길이에서 다시 반올림하지 않는다(d·12.5 ≈ x.5 이면 round 가 manifest K 와 1 다르다) — 인코더·채점에 이 K 를 그대로 넘긴다."""
    K0 = int(wrow.get("K") or round(float(wrow["duration_s"]) / CHUNK_S))
    t_last = max((float(w["end_time"]) for w in wrow.get("words") or []), default=None)
    K = train_pad_K(K0, t_last, delays, has_turn, hangover_s, tail_margin, pad_tail)
    return (float(wrow["duration_s"]), K0) if K == K0 else (round(K * CHUNK_S, 6), K)

def eval_audio_duration(wrow: dict, delays: Sequence[int], has_turn: bool, hangover_s: float = HANGOVER_S, tail_margin: int = 2, pad_tail: bool = True) -> float:
    return eval_audio_len(wrow, delays, has_turn, hangover_s, tail_margin, pad_tail)[0]

def events_from_emits(emitted: Iterable[Tuple[int, int]], tid: Optional[int], K: Optional[int] = None, times: bool = False) -> list:
    """방출 [(k, tid)] 에서 tid 이벤트의 청크 번호(K 가 있으면 flush 는 K 로 자름) 또는 times=True 면 시각 (k+1)·0.08. tid=None 이면 빈 목록."""
    if tid is None: return []
    ks = [int(k) for k, t in emitted if int(t) == tid]
    if times: return [emit_time(k, K) for k in ks]
    return ks if K is None else [min(k, K) for k in ks]

# ───────────────────────────── 최적 1:1 매칭 ─────────────────────────────
def _valid(h: float, r: float, early: float, late: float) -> bool: return r - early - _EPS <= h <= r + late + _EPS

def _match_dp(hyp: Sequence[float], ref: Sequence[float], early: float, late: float) -> List[Tuple[int, int]]:
    """순서 보존 DP. 창(Δ=h−r ∈ [−early, late])이 구간이면 교차한 두 쌍을 풀어도 둘 다 유효하고 Σ|Δ| 가 늘지 않으므로
    교차 없는 최적해가 존재 → (최대 매칭 수, 그중 최소 Σ|Δ|) 에 대해 정확하다. O(n·m)."""
    hi = sorted(range(len(hyp)), key=lambda i: hyp[i]); ri = sorted(range(len(ref)), key=lambda j: ref[j]); n, m = len(hi), len(ri)
    best = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]; back = [[0] * (m + 1) for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(1, m + 1):
            opts = [(best[a - 1][b], 1), (best[a][b - 1], 2)]
            h, r = hyp[hi[a - 1]], ref[ri[b - 1]]
            if _valid(h, r, early, late):
                p = best[a - 1][b - 1]; opts.append(((p[0] + 1, p[1] - abs(h - r)), 3))
            best[a][b], back[a][b] = max(opts, key=lambda x: x[0])
    pairs, a, b = [], n, m
    while a > 0 and b > 0:
        w = back[a][b]
        if w == 3: pairs.append((hi[a - 1], ri[b - 1])); a -= 1; b -= 1
        elif w == 1: a -= 1
        else: b -= 1
    return sorted(pairs)

def _f1(p, r):
    if p is None or r is None: return None
    return 2 * p * r / (p + r) if p + r > 0 else 0.0

def match_optimal(hyp: Sequence[float], ref: Sequence[float], early: float, late: float, use_scipy: bool = True) -> dict:
    """hyp·ref(시각 또는 청크) 의 1:1 최적 매칭: ref−early ≤ hyp ≤ ref+late 인 쌍만, 매칭 수 최대(동률이면 Σ|hyp−ref| 최소).
    scipy.optimize.linear_sum_assignment 가 있으면 그것(비용 −1+w·|Δ|, 무효 0 — w 는 매칭 수가 항상 우선하도록 작게), 없으면 정확한 DP.
    → dict(hits, pairs[(hyp_idx, ref_idx)], n_hyp, n_ref, precision, recall, f1). 탐욕(lane_metrics.match_events)은 최적이 아니다(테스트 참고)."""
    hyp = [float(h) for h in hyp]; ref = [float(r) for r in ref]; pairs: List[Tuple[int, int]] = []
    if hyp and ref:
        lsa = None
        if use_scipy:
            try: from scipy.optimize import linear_sum_assignment as lsa
            except ImportError: lsa = None
        if lsa is None: pairs = _match_dp(hyp, ref, early, late)
        else:
            d = np.asarray(hyp)[:, None] - np.asarray(ref)[None, :]; ok = (d >= -early - _EPS) & (d <= late + _EPS)
            if ok.any():
                w = 0.5 / (min(len(hyp), len(ref)) + 1) / (float(np.abs(d[ok]).max()) + 1.0)
                rows, cols = lsa(np.where(ok, -1.0 + w * np.abs(d), 0.0))
                pairs = sorted((int(i), int(j)) for i, j in zip(rows, cols) if ok[i, j])
    hits = len(pairs); p = hits / len(hyp) if hyp else None; r = hits / len(ref) if ref else None
    return dict(hits=hits, pairs=pairs, n_hyp=len(hyp), n_ref=len(ref), precision=p, recall=r, f1=_f1(p, r))

# ───────────────────────────── 편집 정렬 ─────────────────────────────
def _align_dp(ref: Sequence, hyp: Sequence) -> List[Tuple[Optional[int], Optional[int]]]:
    """사전식 최적: 편집 수 최소, 그중 정확 일치 수 최대(스칼라 비용 = 편집·W − 일치, W > 최대 일치 수). 동률은 대각 > 삭제 > 삽입 순으로 결정적."""
    n, m = len(ref), len(hyp); W = n + m + 1; D = [[W * j for j in range(m + 1)]] + [[W * i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        Di, Dp, ri = D[i], D[i - 1], ref[i - 1]
        for j in range(1, m + 1): Di[j] = min(Dp[j - 1] + (W if ri != hyp[j - 1] else -1), Dp[j] + W, Di[j - 1] + W)
    out, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (W if ref[i - 1] != hyp[j - 1] else -1): out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and D[i][j] == D[i - 1][j] + W: out.append((i - 1, None)); i -= 1
        else: out.append((None, j - 1)); j -= 1
    return out[::-1]

def align_pairs(ref: Sequence, hyp: Sequence) -> List[Tuple[Optional[int], Optional[int]]]:
    """최소 편집 정렬(그중 일치 최대) → 순서대로 (ref_idx|None, hyp_idx|None): 둘 다 있으면 일치/치환, (i, None) 삭제, (None, j) 삽입.
    정렬기는 이 DP 하나뿐 — 이벤트 사상(map_events)이 rapidfuzz 설치 여부(rack4 는 있음, 로컬은 없음)에 따라 달라지지 않는다. 단어 수준이라 O(n·m) 로 충분."""
    return _align_dp(ref, hyp)

def edit_counts(ref: Sequence, hyp: Sequence, use_rapidfuzz: bool = True) -> dict:
    """S/D/I 카운트. rapidfuzz 가 있으면 Levenshtein.editops(single_turn_eval.score_pair 와 같은 카운트, 긴 CER 열도 빠름), 없으면 align_pairs.
    errors(=편집 거리) 는 두 경로가 항상 같고, 동률 정렬에서 S 와 D+I 의 나눔만 다를 수 있다(이벤트 사상에는 쓰지 않는다)."""
    c = dict(substitutions=0, deletions=0, insertions=0); Lev = None
    if use_rapidfuzz:
        try: from rapidfuzz.distance import Levenshtein as Lev
        except ImportError: Lev = None
    if Lev is not None:
        names = {"replace": "substitutions", "delete": "deletions", "insert": "insertions"}
        for op in Lev.editops(list(ref), list(hyp)): c[names[op.tag]] += 1
    else:
        for i, j in align_pairs(ref, hyp):
            if i is None: c["insertions"] += 1
            elif j is None: c["deletions"] += 1
            elif ref[i] != hyp[j]: c["substitutions"] += 1
    c.update(n_ref=len(ref), n_hyp=len(hyp)); c["errors"] = c["substitutions"] + c["deletions"] + c["insertions"]
    return c

def asr_counts(ref_text: str, hyp_text: str, lang: str) -> dict:
    """single_turn_eval.score_pair 와 같은 변형·카운트(EN wer / KO cer_nospace·cer_space·wer). 가설은 이벤트 토큰을 뺀 디코드 텍스트."""
    from ..data.textnorm import score_en, score_ko
    if lang == "Korean":
        v = {"cer_nospace": (list(score_ko(ref_text, False)), list(score_ko(hyp_text, False))), "cer_space": (list(score_ko(ref_text, True)), list(score_ko(hyp_text, True))),
             "wer": (score_ko(ref_text, True).split(), score_ko(hyp_text, True).split())}
    else: v = {"wer": (score_en(ref_text).split(), score_en(hyp_text).split())}
    return {k: edit_counts(r, h) for k, (r, h) in v.items()}

# ───────────────────────────── 가설 단어·이벤트 ─────────────────────────────
def split_hyp(emitted: Iterable[Tuple[int, int]], *, is_text: Callable[[int], bool], word_start: Callable[[int], bool], decode: Callable[[List[int]], str],
              event_ids: Sequence[Optional[int]] = (SEM_END_ID, EOT_ID)) -> dict:
    """방출 [(k, tid)] → dict(words=[가설 단어], events=[dict(id, k, after, mid_word)]).
    단어 경계: 첫 텍스트 토큰 또는 word_start(tid)(Qwen byte-BPE: 바이트가 0x20 으로 시작 = 'Ġ'). 이벤트·텍스트 아닌 토큰(<NEXT_AUDIO> 등 is_text=False)은 단어를 끊지 않는다.
    after = 이벤트 직전 텍스트 토큰이 속한 가설 단어 번호(−1: 단어 전), mid_word = 이벤트 뒤 첫 텍스트 토큰이 같은 단어를 잇는다(조각 사이 commit)."""
    ev_ids = {int(e) for e in event_ids if e is not None}; words, cur, events, pend = [], [], [], []
    for k, t in emitted:
        k, t = int(k), int(t)
        if t in ev_ids:
            ev = dict(id=t, k=k, after=len(words) - (0 if cur else 1), mid_word=False); events.append(ev); pend.append(ev); continue
        if not is_text(t): continue
        start = not cur or word_start(t)
        for ev in pend: ev["mid_word"] = not start
        pend = []
        if start and cur: words.append(decode(cur)); cur = []
        cur.append(t)
    if cur: words.append(decode(cur))
    return dict(words=[w.strip() for w in words], events=events)

def norm_word(w: str) -> str:
    """정렬용 단어 정규화: 소문자, 단어 문자·아포스트로피만. 비면 원형."""
    s = re.sub(r"[^\w']+", "", w.lower()); return s or w

def map_events(ref_words: Sequence[str], hyp_words: Sequence[str], events: Sequence[dict], ref_k: Optional[Sequence[int]] = None, K: Optional[int] = None,
               norm: Callable[[str], str] = norm_word) -> List[int]:
    """각 이벤트(after=가설 단어 j) → 참조 위치 p ('참조 단어 p 뒤', −1 = 첫 단어 전).
    p = (가설 j 까지 정렬 경로가 소비한 참조 단어 수) − 1; j 바로 뒤에 삭제된 참조 단어들은 ref_k[d] ≤ 이벤트 청크인 동안 소비한 것으로 본다."""
    R = [norm(w) for w in ref_words]; H = [norm(w) for w in hyp_words]
    base: Dict[int, int] = {-1: 0}; trail: Dict[int, List[int]] = {-1: []}; cur, consumed = -1, 0
    for i, j in align_pairs(R, H):
        if i is not None: consumed = i + 1
        if j is not None: cur = j; base[j] = consumed; trail[j] = []
        else: trail[cur].append(i)
    out = []
    for ev in events:
        j = ev["after"]; p = base.get(j, 0) - 1; ke = ev["k"] if K is None else min(ev["k"], K)
        for d in trail.get(j, []):
            if ref_k is not None and ref_k[d] <= ke: p = d
            else: break
        out.append(p)
    return out

# ───────────────────────────── 범주 ─────────────────────────────
def _why_words(why) -> set: return set(re.split(r"[^a-z]+", str(why or "").lower()))

def n_category(cand: dict, tags: Iterable[str] = ()) -> str:
    """N 등급(hard negative) 후보의 범주. 우선순위: filler > rep > unclear > repair > revision > all_wait > other_n
    (단어 태그 → why 문자열 단어 → Stage C relation → Stage B 전원 WAIT)."""
    tags = set(tags or ()); why = _why_words(cand.get("why"))
    if "filler" in tags or "filler" in why: return "filler"
    if "rep" in tags or why & {"rep", "repetition", "repeat", "stutter"}: return "rep"
    if "unclear" in tags or "unclear" in why: return "unclear"
    if why & {"repair", "reparandum", "self"}: return "repair"
    if (cand.get("C") or {}).get("relation") == "REVISION" or "revision" in why: return "revision"
    b = cand.get("B") or {}
    if (b and all(v == "WAIT" for v in b.values())) or "wait" in why: return "all_wait"
    return "other_n"

def inside_category(tags: Iterable[str] = ()) -> str:
    tags = set(tags or ())
    for c in ("filler", "rep", "unclear"):
        if c in tags: return c
    return "other_inside"

# ───────────────────────────── 스트림 지표 ─────────────────────────────
def _grade(c: Optional[dict]) -> Optional[str]:
    if c is None: return None
    g = str(c.get("grade", "N")).upper(); return g if g in ("A", "B") else "N"

def text_position_metrics(words: Sequence[dict], cands: Sequence[dict], hyp_words: Sequence[str], events: Sequence[dict], ref_k: Optional[Sequence[int]] = None,
                          K: Optional[int] = None, norm: Callable[[str], str] = norm_word) -> dict:
    """<SEM_END> 이벤트의 텍스트 위치 지표(합산 가능한 카운트). words 는 i 순서, events 는 SEM 만."""
    cand = {int(c["after_word"]): c for c in cands}; tags = {int(w["i"]): (w.get("tags") or []) for w in words}
    pos = map_events([w["text"] for w in words], hyp_words, events, ref_k, K, norm)
    out = dict(n_hyp=len(events), correct=0, ambiguous=0, duplicate=0, premature=0, cat={c: 0 for c in PCR_CATEGORIES},
               n_A=0, n_B=0, n_N=0, N_hit=0, N_cat={}, N_hit_cat={})
    for p, c in cand.items():
        g = _grade(c); out["n_" + g] += 1
        if g == "N": cat = n_category(c, tags.get(p)); out["N_cat"][cat] = out["N_cat"].get(cat, 0) + 1
    seen = set()
    for ev, p in zip(events, pos):
        if ev.get("mid_word"): cat = "mid_word"
        elif p < 0: cat = "no_word"
        elif p in seen: out["duplicate"] += 1; continue
        else:
            seen.add(p); c = cand.get(p); g = _grade(c)
            if g == "A": out["correct"] += 1; continue
            if g == "B": out["ambiguous"] += 1; continue
            if g == "N": cat = n_category(c, tags.get(p)); out["N_hit"] += 1; out["N_hit_cat"][cat] = out["N_hit_cat"].get(cat, 0) + 1
            else: cat = inside_category(tags.get(p))
        out["premature"] += 1; out["cat"][cat] += 1
    return out

def timing_metrics(sem_k: Sequence[int], K: int, words: Sequence[dict], cands: Sequence[dict], delta: int, early: int = EARLY, lates: Sequence[int] = LATES) -> dict:
    """청크 공간 P/R(ref_k, 비대칭 창) + 매칭 쌍의 음향 지연(hyp_time − word_end)·청크 지연(hyp_k − ref_k). A 에 안 맞은 가설 중 B 에 맞은 수(b_hits) 도 센다."""
    wend = {int(w["i"]): float(w["end_time"]) for w in words}
    A = sorted(wend[int(c["after_word"])] for c in cands if _grade(c) == "A" and int(c["after_word"]) in wend)
    B = sorted(wend[int(c["after_word"])] for c in cands if _grade(c) == "B" and int(c["after_word"]) in wend)
    hk = [min(int(k), K) for k in sem_k]; ht = [emit_time(int(k), K) for k in sem_k]
    ak = [ref_chunk(t, delta, K) for t in A]; bk = [ref_chunk(t, delta, K) for t in B]
    out = dict(ideal_lat=[round(emit_time(k, K) - t, 4) for k, t in zip(ak, A)])
    for L in lates:
        m = match_optimal(hk, ak, early, L); used = {h for h, _ in m["pairs"]}
        mb = match_optimal([hk[i] for i in range(len(hk)) if i not in used], bk, early, L)
        out[f"late{L}"] = dict(hits=m["hits"], n_hyp=len(hk), n_ref=len(ak), b_hits=mb["hits"],
                               lat=[round(ht[h] - A[r], 4) for h, r in m["pairs"]], lag=[hk[h] - ak[r] for h, r in m["pairs"]])
    return out

def turn_metrics(turn_k: Sequence[int], K: int, words: Sequence[dict], turn_end: bool, delta: int, hangover_s: float = HANGOVER_S, early: int = EARLY, lates: Sequence[int] = LATES) -> dict:
    """<TURN_END> vs 참조 k_turn(스트림 끝 1 개, labels.turn_end 일 때만). early_turn = 마지막 단어 끝 이전 방출(발화 중 턴 종료), in_flush = <EMPTY_AUDIO> 라운드 방출."""
    t_last = max((float(w["end_time"]) for w in words), default=None)
    ref = [turn_ref_chunk(t_last, delta, hangover_s, K)] if (turn_end and t_last is not None) else []
    hk = [min(int(k), K) for k in turn_k]; ht = [emit_time(int(k), K) for k in turn_k]
    out = dict(n_ref=len(ref), n_hyp=len(hk), early_turn=(sum(t < t_last - _EPS for t in ht) if t_last is not None else 0), in_flush=sum(int(k) >= K for k in turn_k))
    for L in lates:
        m = match_optimal(hk, ref, early, L)
        out[f"late{L}"] = dict(hits=m["hits"], n_hyp=len(hk), n_ref=len(ref), b_hits=0, lat=[round(ht[h] - t_last, 4) for h, _ in m["pairs"]], lag=[hk[h] - ref[r] for h, r in m["pairs"]])
    return out

def score_stream(wrow: dict, lrow: Optional[dict], hyp: dict, delta: int, K: int, *, early: int = EARLY, lates: Sequence[int] = LATES, eval_turn: bool = False,
                 hangover_s: float = HANGOVER_S, sem_id: int = SEM_END_ID, turn_id: Optional[int] = EOT_ID, norm: Callable[[str], str] = norm_word, guard_blocked: int = 0) -> dict:
    """스트림 하나의 지표(합산 가능한 카운트·목록). hyp = dict(emitted=[[k, tid]], words=[가설 단어], events=split_hyp 이벤트, text=이벤트 뺀 디코드 텍스트).
    guard_blocked = 디코더 sem-guard 가 막은 <SEM_END> 발화 수(방출 안 됨) → events·text 에 기록, finalize 의 pcr_raw 가 조기 commit 으로 센다."""
    words = sorted(wrow["words"], key=lambda w: int(w["i"])); cands = (lrow or {}).get("candidates", []); lang = wrow.get("lang") or (lrow or {}).get("lang")
    em = [(int(k), int(t)) for k, t in hyp["emitted"]]; sem_k = events_from_emits(em, sem_id); turn_k = events_from_emits(em, turn_id)
    ref_k = [ref_chunk(float(w["end_time"]), delta, K) for w in words]
    m = dict(streams=1, audio_s=round(K * CHUNK_S, 3), events=dict(sem=len(sem_k), turn=len(turn_k), sem_in_flush=sum(k >= K for k in sem_k)),
             timing=timing_metrics(sem_k, K, words, cands, delta, early, lates),
             text=dict(text_position_metrics(words, cands, hyp["words"], [e for e in hyp["events"] if int(e["id"]) == sem_id], ref_k, K, norm), guard_blocked=int(guard_blocked)),
             asr=asr_counts(wrow.get("text") or " ".join(w["text"] for w in words), hyp.get("text", ""), lang))
    m["events"]["guard_blocked"] = int(guard_blocked)
    if eval_turn:
        assert turn_id is not None, "eval_turn 인데 턴 종료 토큰이 없다(SEM 만 학습한 모델)"
        m["turn"] = turn_metrics(turn_k, K, words, turn_flag(lrow), delta, hangover_s, early, lates)
    return m

def oracle_hyp(wrow: dict, lrow: Optional[dict], delta: int, K: int, *, eval_turn: bool = False, hangover_s: float = HANGOVER_S,
               sem_id: int = SEM_END_ID, turn_id: Optional[int] = EOT_ID) -> dict:
    """참조 직렬화(v0 계약)를 그대로 낸 가설 — 지표 상한 점검용(정밀도·재현율 1, PCR 0, 지연 = 이상 지연, WER 0).
    토큰은 min(K, int(t/0.08)+δ) 청크(시간순 = 목록순), A 경계의 <SEM_END> 는 그 단어 마지막 토큰 바로 뒤(같은 청크), eval_turn 이면 턴 종료 토큰이 맨 뒤 k_turn."""
    words = sorted(wrow["words"], key=lambda w: int(w["i"])); toks = wrow["tokens"]
    A = {int(c["after_word"]) for c in (lrow or {}).get("candidates", []) if _grade(c) == "A"}
    emitted, events = [], []
    for w in words:
        emitted += [[ref_chunk(float(toks[q][1]), delta, K), int(toks[q][0])] for q in range(int(w["a"]), int(w["b"]))]
        if int(w["i"]) in A:
            k = ref_chunk(float(w["end_time"]), delta, K); emitted.append([k, sem_id]); events.append(dict(id=sem_id, k=k, after=int(w["i"]), mid_word=False))
    if eval_turn and turn_flag(lrow) and words:
        k = turn_ref_chunk(max(float(w["end_time"]) for w in words), delta, hangover_s, K); emitted.append([k, turn_id]); events.append(dict(id=turn_id, k=k, after=int(words[-1]["i"]), mid_word=False))
    return dict(emitted=emitted, words=[w["text"] for w in words], events=events, text=wrow.get("text") or " ".join(w["text"] for w in words))

# ───────────────────────────── 집계 ─────────────────────────────
def merge(a: Optional[dict], b: dict) -> dict:
    """스트림 지표 합산(micro): 수는 더하고 목록은 잇고 dict 는 재귀."""
    if a is None: return copy.deepcopy(b)
    out = dict(a)
    for k, v in b.items():
        if k not in out: out[k] = copy.deepcopy(v)
        elif isinstance(v, dict): out[k] = merge(out[k], v)
        else: out[k] = out[k] + v
    return out

def latency_stats(xs: Sequence[float]) -> dict:
    if not xs: return dict(n=0, mean=None, p50=None, p90=None)
    a = np.asarray(xs, dtype=float)
    return dict(n=int(a.size), mean=round(float(a.mean()), 4), p50=round(float(np.percentile(a, 50)), 4), p90=round(float(np.percentile(a, 90)), 4))

def _rate(a, b): return a / b if b else None

def _finalize_windows(t: dict) -> dict:
    out = {}
    for k, v in t.items():
        if not k.startswith("late"): continue
        p, r = _rate(v["hits"], v["n_hyp"]), _rate(v["hits"], v["n_ref"])
        out[k] = dict(hits=v["hits"], n_hyp=v["n_hyp"], n_ref=v["n_ref"], miss=v["n_ref"] - v["hits"], false=v["n_hyp"] - v["hits"] - v.get("b_hits", 0), b_hits=v.get("b_hits", 0),
                      precision=p, recall=r, f1=_f1(p, r), precision_excl_B=_rate(v["hits"], v["n_hyp"] - v.get("b_hits", 0)),
                      latency_s=latency_stats(v["lat"]), lag_chunks=latency_stats(v["lag"]))
    return out

def finalize(m: dict) -> dict:
    """합산 지표 → 비율·분위수. timing.lateL: 청크 창 P/R/F1(ref_k 기준)·miss·지연(s, word_end 기준)·청크 지연; text: 위치 정밀도/재현율·PCR(범주별)·hard negative commit 률.
    pcr_raw = (premature + guard_blocked)/(n_hyp + guard_blocked): sem-guard 가 막은 발화(단어 전·SEM 연속)를 조기 commit 으로 되돌려 센 PCR(가드 끈 argmax 행동 근사)."""
    x = m["text"]; n = x["n_hyp"]; nb = n - x["ambiguous"]; p, r = _rate(x["correct"], n), _rate(x["correct"], x["n_A"]); gb = int(x.get("guard_blocked", 0))
    out = dict(streams=m["streams"], audio_s=round(m["audio_s"], 2), events=dict(m["events"], sem_per_min=_rate(m["events"]["sem"] * 60.0, m["audio_s"])),
               timing=dict(ideal_latency_s=latency_stats(m["timing"]["ideal_lat"]), **_finalize_windows(m["timing"])),
               text=dict(n_hyp=n, correct=x["correct"], ambiguous=x["ambiguous"], duplicate=x["duplicate"], premature=x["premature"], precision=p, recall=r, f1=_f1(p, r),
                         precision_excl_B=_rate(x["correct"], nb), pcr=_rate(x["premature"], n), pcr_excl_B=_rate(x["premature"], nb),
                         guard_blocked=gb, pcr_raw=_rate(x["premature"] + gb, n + gb),
                         pcr_by_category={c: _rate(v, n) for c, v in x["cat"].items()}, premature_by_category=dict(x["cat"]),
                         n_A=x["n_A"], n_B=x["n_B"], n_N=x["n_N"], B_commit_rate=_rate(x["ambiguous"], x["n_B"]), hardneg_commit_rate=_rate(x["N_hit"], x["n_N"]),
                         hardneg_by_category={c: dict(n=v, committed=x["N_hit_cat"].get(c, 0), rate=_rate(x["N_hit_cat"].get(c, 0), v)) for c, v in sorted(x["N_cat"].items())}),
               asr={k: dict(v, rate=_rate(v["errors"], v["n_ref"])) for k, v in m["asr"].items()})
    if "turn" in m:
        t = m["turn"]; out["turn"] = dict(n_ref=t["n_ref"], n_hyp=t["n_hyp"], early_turn=t["early_turn"], in_flush=t["in_flush"], **_finalize_windows(t))
    return out

def aggregate(records: Iterable[dict], keys: Sequence[str] = ("lang", "set")) -> dict:
    """records: dict(metrics=score_stream(...), lang, set, …) → dict(overall, by_lang{…}, by_set{…}) (finalize 된 요약)."""
    tot, groups = None, {k: {} for k in keys}
    for r in records:
        tot = merge(tot, r["metrics"])
        for k in keys: groups[k][r.get(k)] = merge(groups[k].get(r.get(k)), r["metrics"])
    return dict(overall=(finalize(tot) if tot else None), **{f"by_{k}": {str(g): finalize(v) for g, v in sorted(groups[k].items(), key=lambda kv: str(kv[0]))} for k in keys})

def pr_point(summary: Optional[dict], late: int = 2) -> dict:
    """PR 곡선의 한 점(요약 하나에서): 시간 창(late) P/R/F1·지연 p50 + 텍스트 위치 P/R·PCR."""
    if not summary: return {}
    w = summary["timing"].get(f"late{late}", {}); t = summary["text"]
    return dict(n_hyp=t["n_hyp"], sem_per_min=summary["events"]["sem_per_min"], time_precision=w.get("precision"), time_recall=w.get("recall"), time_f1=w.get("f1"),
                latency_p50_s=(w.get("latency_s") or {}).get("p50"), text_precision=t["precision"], text_recall=t["recall"], text_f1=t["f1"], pcr=t["pcr"],
                pcr_raw=t.get("pcr_raw"), guard_blocked=t.get("guard_blocked", 0), hardneg_commit_rate=t["hardneg_commit_rate"])

# ───────────────────────────── 독립 참조: 구두점 전사(LibriSpeech-PC) ─────────────────────────────
# LLM 라벨(A/B/N)과 무관한 EN 교차 점검. SEM_END 는 구두점 경계가 아니지만(계획 §3) 낭독체에서 문장 끝(. ? !)은 거의 늘 완결·안정 단위라,
# 커밋이 문장 끝에 놓인 비율(P_pc)·문장 끝을 커밋한 비율(R_pc)이 라벨 재현율 한계(예: A 가 문장 끝의 18 %)를 드러낸다.
_PC_TRAIL = re.compile(r"([^\w']+)$")

def pc_punct_after(words: Sequence[dict], pnc_text: str, norm: Callable[[str], str] = norm_word) -> Dict[int, str]:
    """참조 단어 i → 그 뒤 구두점 범주 SENT(. ? !) · CLAUSE(, ; :) · NONE. pnc_text 토큰을 참조 단어에 align_pairs 로 정렬(정렬 안 된 참조 단어는 없음)."""
    ws = sorted(words, key=lambda w: int(w["i"])); toks = pnc_text.split(); out: Dict[int, str] = {}
    for i, j in align_pairs([norm(w["text"]) for w in ws], [norm(t) for t in toks]):
        if i is None or j is None: continue
        m = _PC_TRAIL.search(toks[j]); p = m.group(1) if m else ""
        out[int(ws[i]["i"])] = "SENT" if re.search(r"[.?!]", p) else ("CLAUSE" if re.search(r"[,;:]", p) else "NONE")
    return out

def pc_commit_counts(words: Sequence[dict], pnc_text: str, hyp_words: Sequence[str], events: Sequence[dict], ref_k: Optional[Sequence[int]] = None,
                     K: Optional[int] = None, norm: Callable[[str], str] = norm_word) -> dict:
    """<SEM_END> 이벤트(서로 다른 참조 위치, map_events 사상)의 구두점 범주 카운트 + 문장 끝 수·커밋된 문장 끝 수(합산 가능). 첫 단어 전(−1)은 NONE."""
    pa = pc_punct_after(words, pnc_text, norm); ws = sorted(words, key=lambda w: int(w["i"]))
    pos = set(map_events([w["text"] for w in ws], hyp_words, events, ref_k, K, norm))
    cats = [pa.get(p, "NONE") for p in pos]
    return dict(commits=len(pos), SENT=cats.count("SENT"), CLAUSE=cats.count("CLAUSE"), NONE=cats.count("NONE"),
                n_sent=sum(v == "SENT" for v in pa.values()), sent_hit=sum(1 for p in pos if pa.get(p) == "SENT"))

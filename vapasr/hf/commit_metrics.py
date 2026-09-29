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

지연 지표(창 없음) — 모두 알고리즘 지연(오디오 시각; 계산 시간 제외, 인코더 right context 0 가정):
  시각 emit_time(k,K) = (min(k,K−1)+1)·0.08. flush 라운드(k ≥ K) 방출은 K·0.08 로 잘린 하한이라 in_flush 로 따로 센다.
  유효 commit = <SEM_END> 중 mid_word 가 아니고 map_events 위치 p ≥ 0 인 것(방출 순). p 는 방출 순서에 단조 비감소다.
  commit(A 경계별, commit_latency_metrics): A 위치 a_1<…<a_m 에서 경계 a_r 의 확정 = 위치가 [a_r, a_{r+1}) (마지막은 [a_m, n)) 인 첫 유효 commit.
    lat = emit_time − word_end(a_r), excess = min(k,K−1) − min(ref_k(a_r),K−1) (청크, δ 목표 대비), overshoot = p − a_r (단어). 다음 A 이후의 commit 은 그 A 의 몫이라
    늦은 commit 은 지연 분포가 아니라 coverage 감소로 드러난다 — coverage 와 지연을 같이 읽는다. prefix(보조) = 위치 ≥ a_r 인 첫 유효 commit(다음 A 를 넘어도)까지.
    excess 는 emit_time 과 같은 척도(flush 라운드 = 마지막 오디오 청크 K−1 의 끝 시각): excess·0.08 = lat − 이상 지연(oracle 은 0). timing 창의 lag(min(k,K) − ref_k)는
    flush 라운드를 청크 K 로 세서, ref_k = K−1 인 단어를 flush 에서 확정하면 timing lag 는 1, excess 는 0 이다.
    B/N 경계는 따로 센다(위치 == b 인 첫 유효 commit: B.committed = text.ambiguous, N.committed = text.N_hit). A 없는 스트림은 streams_no_A.
    항등식: commit.exact = text.correct, commit.n_A = text.n_A → coverage·exact_rate = text.recall (score_stream 이 단어 범위 밖 후보를 after_word_range 로 거부하므로 성립).
    창 매칭(timing)과 달리 mid_word commit 은 빼고, 창 밖의 늦은 commit 은 넣는다.
  display(참조 단어별 표시 지연, display_metrics): align_pairs(norm(ref), norm(hyp)) 로 짝지은 가설 단어 마지막 토큰의 emit_time − word_end, lag = min(그 청크,K−1) − min(ref_k,K−1).
    삭제는 n_del(지연에서 뺌), 삽입은 n_ins. 첫 토큰 지연 = 첫 정렬 쌍 (i*, j*) 의 가설 첫 토큰 emit_time − word_end(i*) (i* > 0 = 첫 참조 단어 삭제면 first_skip 으로 세고 뺀다),
    first_any = 첫 가설 토큰 emit_time − word_end(0). 여러 토큰 단어는 첫 토큰이 단어 끝보다 일러 음수일 수 있다. 입력 hyp.word_k(split_hyp)가 없으면 missing.
    lag 는 excess 와 같은 척도(flush 라운드 = 청크 K−1)라 lag·0.08 = lat − 그 단어의 이상 표시 지연.
  after_text(commit-after-text lag, after_text_metrics): 유효 commit 청크 − 바로 앞 가설 단어 마지막 토큰 청크(학습 목표 0). 둘 다 원래 k(flush 라운드 K, K+1, … 도
    자르지 않은 디코드 라운드 번호)로 빼므로 flush 에서도 0 착시가 없다 — SEM 이 flush 라운드인 commit 도 lag 에 넣고 in_flush 로도 센다(스트림 끝에서 늦게 확정).
    lag 에서 빼는 commit 은 이름을 나눠 센다: no_word = 참조 위치 p < 0(유효 commit 아님 — 첫 참조 단어 앞 삽입 단어 뒤도 여기; text.cat.no_word 와 같은 수),
    no_text = 유효 commit(p ≥ 0)이지만 앞에 가설 텍스트가 없음(after < 0 — 빠진 참조 단어 뒤로 사상된 경우). 유효 commit 수 = n + no_text.
    lag_A = A 경계를 확정한 commit 만. lag > 0 은 모델이 음향을 더 보고 정한 암묵적 lookahead 라 정확도와 같이 읽는다.
  임계값: late_rate[τ] = 확정된 A 중 lat > τ 비율, recall_at[τ] = lat ≤ τ 로 확정된 A / 전체 A (기본 τ = LATE_THRESHOLDS_S). δ 이상 지연이 [0.08δ, 0.08(δ+1)] 이라
    δ4 는 oracle 도 거의 전부 > 0.32 s, δ8 은 > 0.64 s — δ 끼리는 excess_chunks·recall_within_chunks[L] = #(excess ≤ L)/n_A (L = timing 창 late 목록)로 비교한다.
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
LATE_THRESHOLDS_S = (0.32, 0.64, 1.0)                      # 창 없는 commit·표시 지연의 늦음 임계값(초) — semcommit_eval --late-thresholds
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
    """방출 [(k, tid)] → dict(words=[가설 단어], events=[dict(id, k, after, mid_word)], word_k=[[k_first, k_last]]).
    단어 경계: 첫 텍스트 토큰 또는 word_start(tid)(Qwen byte-BPE: 바이트가 0x20 으로 시작 = 'Ġ'). 이벤트·텍스트 아닌 토큰(<NEXT_AUDIO> 등 is_text=False)은 단어를 끊지 않는다.
    after = 이벤트 직전 텍스트 토큰이 속한 가설 단어 번호(−1: 단어 전), mid_word = 이벤트 뒤 첫 텍스트 토큰이 같은 단어를 잇는다(조각 사이 commit).
    word_k = 가설 단어별 첫·마지막 텍스트 토큰의 방출 청크(원래 k, flush 도 자르지 않음) — 표시·첫 토큰 지연(display_metrics)의 입력."""
    ev_ids = {int(e) for e in event_ids if e is not None}; words, cur, events, pend, word_k, k0, kl = [], [], [], [], [], 0, 0
    for k, t in emitted:
        k, t = int(k), int(t)
        if t in ev_ids:
            ev = dict(id=t, k=k, after=len(words) - (0 if cur else 1), mid_word=False); events.append(ev); pend.append(ev); continue
        if not is_text(t): continue
        start = not cur or word_start(t)
        for ev in pend: ev["mid_word"] = not start
        pend = []
        if start and cur: words.append(decode(cur)); word_k.append([k0, kl]); cur = []
        if not cur: k0 = k
        cur.append(t); kl = k
    if cur: words.append(decode(cur)); word_k.append([k0, kl])
    return dict(words=[w.strip() for w in words], events=events, word_k=word_k)

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

def after_word_range_errors(words: Sequence[dict], cands: Sequence[dict]) -> List[int]:
    """참조 단어 범위 [0, n) 밖의 후보 after_word(정렬, 중복 없음) — 학습 SemCommitDataset.build_semcommit_tokens 가 after_word_range 로 거부하는 라벨.
    이런 후보가 있으면 text(모든 후보를 셈)와 commit·timing(범위 안만 셈)의 n_A 가 달라져 항등식이 깨지므로 score_stream 이 거부한다."""
    n = len(words); return sorted({int(c["after_word"]) for c in cands if not 0 <= int(c["after_word"]) < n})

def text_position_metrics(words: Sequence[dict], cands: Sequence[dict], hyp_words: Sequence[str], events: Sequence[dict], ref_k: Optional[Sequence[int]] = None,
                          K: Optional[int] = None, norm: Callable[[str], str] = norm_word, pos: Optional[Sequence[int]] = None) -> dict:
    """<SEM_END> 이벤트의 텍스트 위치 지표(합산 가능한 카운트). words 는 i 순서, events 는 SEM 만. pos = 같은 인자로 이미 계산한 map_events 결과(없으면 계산)."""
    cand = {int(c["after_word"]): c for c in cands}; tags = {int(w["i"]): (w.get("tags") or []) for w in words}
    if pos is None: pos = map_events([w["text"] for w in words], hyp_words, events, ref_k, K, norm)
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

# ───────────────────────────── 지연(창 없음) ─────────────────────────────
def _eligible(events: Sequence[dict], pos: Sequence[int]) -> List[Tuple[int, int, int]]:
    """유효 commit [(이벤트 번호 q, 참조 위치 p, 청크 k)] — mid_word 가 아니고 p ≥ 0, 방출 순."""
    return [(q, int(p), int(ev["k"])) for q, (ev, p) in enumerate(zip(events, pos)) if not ev.get("mid_word") and p >= 0]

def commit_latency_metrics(words: Sequence[dict], cands: Sequence[dict], hyp_words: Sequence[str], events: Sequence[dict], ref_k: Sequence[int], K: int, *,
                           norm: Callable[[str], str] = norm_word, pos: Optional[Sequence[int]] = None) -> dict:
    """A 경계별 창 없는 commit 지연(합산 가능한 카운트·목록; 정의는 모듈 docstring '지연 지표'). events 는 SEM 만(방출 순), ref_k 는 참조 단어별 직렬화 청크.
    경계 a_r 의 확정 = 위치가 [a_r, a_{r+1}) 인 첫 유효 commit, prefix = 위치 ≥ a_r 인 첫 유효 commit. B/N = 위치 == b 인 첫 유효 commit(따로 셈).
    등급은 cands(같은 after_word 는 뒤 후보 — text_position_metrics 와 같은 dict 규칙). 범위 밖 후보는 세지 않는다(score_stream 은 미리 거부).
    lat·last·excess·overshoot 는 확정된 A 마다 한 항목인 평행 목록(merge 가 행 순서대로 이어 붙여도 짝이 유지된다). _events = 확정 이벤트 번호(score_stream 이 뺀다).
    pos = 같은 인자로 이미 계산한 map_events 결과(없으면 계산)."""
    ws = sorted(words, key=lambda w: int(w["i"])); n = len(ws); e = [float(w["end_time"]) for w in ws]
    grade = {int(c["after_word"]): _grade(c) for c in cands}
    A = sorted(p for p, g in grade.items() if g == "A" and 0 <= p < n)
    if pos is None: pos = map_events([w["text"] for w in ws], hyp_words, events, ref_k, K, norm)
    el = _eligible(events, pos)
    out = dict(n_A=len(A), n_last=sum(int(a == n - 1) for a in A), committed=0, exact=0, in_flush=0, streams_no_A=int(not A),
               lat=[], last=[], excess=[], overshoot=[], prefix_lat=[], prefix_uncovered=0, _events=[])
    for r, a in enumerate(A):
        hi = A[r + 1] if r + 1 < len(A) else n
        f = next((x for x in el if a <= x[1] < hi), None); g = next((x for x in el if x[1] >= a), None)
        if g is None: out["prefix_uncovered"] += 1
        else: out["prefix_lat"].append(round(emit_time(g[2], K) - e[a], 4))
        if f is None: continue                                                   # 미확정: 다음 A 이후의 commit 은 그 A 몫
        q, p, k = f
        out["committed"] += 1; out["exact"] += int(p == a); out["in_flush"] += int(k >= K); out["_events"].append(q)
        out["lat"].append(round(emit_time(k, K) - e[a], 4)); out["last"].append(int(a == n - 1)); out["excess"].append(min(k, K - 1) - min(int(ref_k[a]), K - 1)); out["overshoot"].append(p - a)
    for gname in ("B", "N"):
        blk = dict(n=0, committed=0, lat=[])
        for b in sorted(p for p, x in grade.items() if x == gname and 0 <= p < n):
            blk["n"] += 1; f = next((x for x in el if x[1] == b), None)
            if f is not None: blk["committed"] += 1; blk["lat"].append(round(emit_time(f[2], K) - e[b], 4))
        out[gname] = blk
    return out

def display_metrics(words: Sequence[dict], hyp_words: Sequence[str], word_k: Sequence[Sequence[int]], K: int, delta: int, norm: Callable[[str], str] = norm_word) -> dict:
    """참조 단어별 표시 지연·첫 토큰 지연(합산 가능; 정의는 모듈 docstring '지연 지표'). word_k = split_hyp 의 가설 단어별 [첫, 마지막] 토큰 청크.
    정렬 = align_pairs(norm(ref), norm(hyp)) — map_events 와 같은 정렬. lat·lag·exact 는 정렬된(일치·치환) 참조 단어마다 한 항목인 평행 목록,
    first_lat·first_any_lat 는 스트림당 최대 1 개(첫 정렬 쌍이 첫 참조 단어가 아니면 first_skip, 가설 텍스트가 없으면 no_text)."""
    if len(word_k) != len(hyp_words): raise ValueError(f"word_k {len(word_k)} 개 ≠ 가설 단어 {len(hyp_words)} 개")
    ws = sorted(words, key=lambda w: int(w["i"])); e = [float(w["end_time"]) for w in ws]
    R = [norm(w["text"]) for w in ws]; H = [norm(h) for h in hyp_words]
    out = dict(streams=1, missing=0, n_ref=len(ws), n_aligned=0, n_exact=0, n_del=0, n_ins=0, in_flush=0, lat=[], lag=[], exact=[],
               first_lat=[], first_skip=0, first_any_lat=[], no_text=int(not hyp_words))
    first = None
    for i, j in align_pairs(R, H):
        if i is None: out["n_ins"] += 1; continue
        if j is None: out["n_del"] += 1; continue                               # 표시되지 않은 단어 — 지연에서 빼고 coverage 로
        kl = int(word_k[j][1]); ex = int(R[i] == H[j])
        out["n_aligned"] += 1; out["n_exact"] += ex; out["in_flush"] += int(kl >= K)
        out["lat"].append(round(emit_time(kl, K) - e[i], 4)); out["lag"].append(min(kl, K - 1) - min(ref_chunk(e[i], delta, K), K - 1)); out["exact"].append(ex)
        if first is None: first = (i, j)
    if first is not None:
        if first[0] > 0: out["first_skip"] += 1                                   # 첫 참조 단어 삭제 — 첫 토큰 지연에서 뺀다
        else: out["first_lat"].append(round(emit_time(int(word_k[first[1]][0]), K) - e[0], 4))
    if hyp_words and ws: out["first_any_lat"].append(round(emit_time(int(word_k[0][0]), K) - e[0], 4))
    return out

def after_text_metrics(events: Sequence[dict], pos: Sequence[int], word_k: Sequence[Sequence[int]], K: int, confirm: Iterable[int] = ()) -> dict:
    """commit-after-text lag(청크, 합산 가능): 유효 commit 의 방출 청크 k − 바로 앞 가설 단어(after = j) 마지막 토큰 청크. 학습 목표 0(SEM 은 그 단어 마지막 토큰 바로 뒤 같은 청크).
    두 청크 모두 원래 k(flush 라운드 K, K+1, … 도 자르지 않은 디코드 라운드 번호 — split_hyp 의 word_k 와 이벤트 k)라 flush 에서도 0 착시가 없다.
    그래서 SEM 이 flush 라운드인 commit(단어는 오디오 안에서 나왔을 수 있다 — 스트림 끝에서 늦게 확정)도 lag 에 넣고, in_flush 로도 센다.
    lag 에서 빼는 것: mid_word, no_word(p < 0 — 유효 commit 아님, text.cat.no_word 와 같은 수), no_text(p ≥ 0 이지만 앞 가설 텍스트 없음, after < 0).
    confirm = commit_latency_metrics 가 A 경계를 확정한 이벤트 번호(_events) → lag_A."""
    cf = set(confirm); out = dict(streams=1, missing=0, n=0, mid_word=0, no_word=0, no_text=0, in_flush=0, lag=[], lag_A=[])
    for q, (ev, p) in enumerate(zip(events, pos)):
        j, k = int(ev["after"]), int(ev["k"])
        if ev.get("mid_word"): out["mid_word"] += 1; continue
        if p < 0: out["no_word"] += 1; continue
        if j < 0: out["no_text"] += 1; continue
        x = k - int(word_k[j][1]); out["n"] += 1; out["in_flush"] += int(k >= K); out["lag"].append(x)
        if q in cf: out["lag_A"].append(x)
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
    """스트림 하나의 지표(합산 가능한 카운트·목록). hyp = dict(emitted=[[k, tid]], words=[가설 단어], events=split_hyp 이벤트, text=이벤트 뺀 디코드 텍스트,
    word_k=split_hyp 의 가설 단어별 [첫, 마지막] 토큰 청크 — 없으면(옛 행) display·after_text 는 dict(streams=0, missing=1)).
    guard_blocked = 디코더 sem-guard 가 막은 <SEM_END> 발화 수(방출 안 됨) → events·text 에 기록, finalize 의 pcr_raw 가 조기 commit 으로 센다.
    commit(창 없는 A 경계별 지연)은 저장된 events 만으로 늘 계산한다.
    단어 범위 밖 후보(after_word ∉ [0, n))가 있으면 ValueError('after_word_range …') — 학습과 같은 사유, 블록 사이 n_A 항등식을 지킨다."""
    words = sorted(wrow["words"], key=lambda w: int(w["i"])); cands = (lrow or {}).get("candidates", []); lang = wrow.get("lang") or (lrow or {}).get("lang")
    bad = after_word_range_errors(words, cands)
    if bad: raise ValueError(f"after_word_range: {wrow.get('id')} 후보 after_word {bad[:5]} ∉ [0, {len(words)}) — labels 가 words 행과 맞지 않는다")
    em = [(int(k), int(t)) for k, t in hyp["emitted"]]; sem_k = events_from_emits(em, sem_id); turn_k = events_from_emits(em, turn_id)
    ref_k = [ref_chunk(float(w["end_time"]), delta, K) for w in words]
    sem_ev = [e for e in hyp["events"] if int(e["id"]) == sem_id]; pos = map_events([w["text"] for w in words], hyp["words"], sem_ev, ref_k, K, norm)
    m = dict(streams=1, audio_s=round(K * CHUNK_S, 3), events=dict(sem=len(sem_k), turn=len(turn_k), sem_in_flush=sum(k >= K for k in sem_k)),
             timing=timing_metrics(sem_k, K, words, cands, delta, early, lates),
             text=dict(text_position_metrics(words, cands, hyp["words"], sem_ev, ref_k, K, norm, pos=pos), guard_blocked=int(guard_blocked)),
             asr=asr_counts(wrow.get("text") or " ".join(w["text"] for w in words), hyp.get("text", ""), lang))
    m["events"]["guard_blocked"] = int(guard_blocked)
    m["commit"] = commit_latency_metrics(words, cands, hyp["words"], sem_ev, ref_k, K, norm=norm, pos=pos); confirm = m["commit"].pop("_events")
    wk = hyp.get("word_k")
    if wk is None: m["display"] = dict(streams=0, missing=1); m["after_text"] = dict(streams=0, missing=1)
    else: m["display"] = display_metrics(words, hyp["words"], wk, K, delta, norm); m["after_text"] = after_text_metrics(sem_ev, pos, wk, K, confirm)
    if eval_turn:
        assert turn_id is not None, "eval_turn 인데 턴 종료 토큰이 없다(SEM 만 학습한 모델)"
        m["turn"] = turn_metrics(turn_k, K, words, turn_flag(lrow), delta, hangover_s, early, lates)
    return m

def oracle_word_k(wrow: dict, delta: int, K: int) -> List[List[int]]:
    """참조 직렬화(oracle_hyp)의 단어별 [첫, 마지막] 토큰 청크 = [ref_chunk(첫 토큰 시각), ref_chunk(마지막 토큰 시각)] (토큰 없는 단어는 단어 끝 시각)."""
    toks = wrow["tokens"]; out = []
    for w in sorted(wrow["words"], key=lambda w: int(w["i"])):
        a, b = int(w["a"]), int(w["b"])
        t0, t1 = (float(toks[a][1]), float(toks[b - 1][1])) if b > a else (float(w["end_time"]),) * 2
        out.append([ref_chunk(t0, delta, K), ref_chunk(t1, delta, K)])
    return out

def oracle_hyp(wrow: dict, lrow: Optional[dict], delta: int, K: int, *, eval_turn: bool = False, hangover_s: float = HANGOVER_S,
               sem_id: int = SEM_END_ID, turn_id: Optional[int] = EOT_ID) -> dict:
    """참조 직렬화(v0 계약)를 그대로 낸 가설 — 지표 상한 점검용(정밀도·재현율 1, PCR 0, 지연 = 이상 지연, WER 0).
    토큰은 min(K, int(t/0.08)+δ) 청크(시간순 = 목록순), A 경계의 <SEM_END> 는 그 단어 마지막 토큰 바로 뒤(같은 청크), eval_turn 이면 턴 종료 토큰이 맨 뒤 k_turn.
    word_k = oracle_word_k(표시 지연 = δ 이상 지연, lag 0)."""
    words = sorted(wrow["words"], key=lambda w: int(w["i"])); toks = wrow["tokens"]
    A = {int(c["after_word"]) for c in (lrow or {}).get("candidates", []) if _grade(c) == "A"}
    emitted, events = [], []
    for w in words:
        emitted += [[ref_chunk(float(toks[q][1]), delta, K), int(toks[q][0])] for q in range(int(w["a"]), int(w["b"]))]
        if int(w["i"]) in A:
            k = ref_chunk(float(w["end_time"]), delta, K); emitted.append([k, sem_id]); events.append(dict(id=sem_id, k=k, after=int(w["i"]), mid_word=False))
    if eval_turn and turn_flag(lrow) and words:
        k = turn_ref_chunk(max(float(w["end_time"]) for w in words), delta, hangover_s, K); emitted.append([k, turn_id]); events.append(dict(id=turn_id, k=k, after=int(words[-1]["i"]), mid_word=False))
    return dict(emitted=emitted, words=[w["text"] for w in words], events=events, text=wrow.get("text") or " ".join(w["text"] for w in words),
                word_k=oracle_word_k(wrow, delta, K))

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
    """n·mean·p50·p90·p95·p99·max(numpy linear 분위수, 소수 4 자리). 빈 목록이면 n=0 에 나머지 None. 키 순서는 옛 n/mean/p50/p90 뒤에 덧붙인 것."""
    if not len(xs): return dict(n=0, mean=None, p50=None, p90=None, p95=None, p99=None, max=None)
    a = np.asarray(xs, dtype=float); q = lambda p: round(float(np.percentile(a, p)), 4)
    return dict(n=int(a.size), mean=round(float(a.mean()), 4), p50=q(50), p90=q(90), p95=q(95), p99=q(99), max=round(float(a.max()), 4))

def _rate(a, b): return a / b if b else None

def _tkey(t: float) -> str: return f"{float(t):g}"                                    # 임계값 키: 0.32 → "0.32", 1.0 → "1"

def _late_rate(xs: Sequence[float], t: float): return _rate(sum(x > t + _EPS for x in xs), len(xs))

def _recall_at(xs: Sequence[float], t: float, n: int): return _rate(sum(x <= t + _EPS for x in xs), n)

def _finalize_commit(c: dict, thresholds: Sequence[float], lates: Sequence[int]) -> dict:
    """commit 블록(합산) → coverage(= 확정 A / 전체 A)·exact_rate(= 정확 위치 / 확정)·지연 분포·임계값별 늦음률·지연 재현율·청크 초과(δ 상대)·중간/끝 A·prefix·B/N."""
    nA, lat, last, ex = c["n_A"], c["lat"], c["last"], c["excess"]; nl = c.get("n_last", 0)
    lm = [x for x, l in zip(lat, last) if not l]; ll = [x for x, l in zip(lat, last) if l]; pl = c["prefix_lat"]
    sub = lambda n, xs: dict(n_A=n, committed=len(xs), coverage=_rate(len(xs), n), latency_s=latency_stats(xs))
    grp = lambda b: dict(n=b["n"], committed=b["committed"], rate=_rate(b["committed"], b["n"]), latency_s=latency_stats(b["lat"]))
    return dict(n_A=nA, committed=c["committed"], coverage=_rate(c["committed"], nA), exact=c["exact"], exact_rate=_rate(c["exact"], c["committed"]),
                in_flush=c["in_flush"], streams_no_A=c["streams_no_A"], latency_s=latency_stats(lat),
                late_rate={_tkey(t): _late_rate(lat, t) for t in thresholds}, recall_at={_tkey(t): _recall_at(lat, t, nA) for t in thresholds},
                excess_chunks=latency_stats(ex), recall_within_chunks={str(L): _rate(sum(x <= L for x in ex), nA) for L in lates},
                overshoot_words=latency_stats(c["overshoot"]), mid=sub(nA - nl, lm), last=sub(nl, ll),
                prefix=dict(covered=len(pl), coverage=_rate(len(pl), nA), latency_s=latency_stats(pl), recall_at={_tkey(t): _recall_at(pl, t, nA) for t in thresholds}),
                B=grp(c["B"]), N=grp(c["N"]))

def _finalize_display(d: dict, thresholds: Sequence[float]) -> dict:
    """display 블록(합산) → coverage(= 정렬된 참조 단어 / 참조 단어)·exact_rate(= 일치 / 정렬)·표시 지연(전체·일치 단어만)·청크 lag·늦음률·첫 토큰 지연. 전부 missing 이면 수만."""
    lat, fl = d.get("lat", []), d.get("first_lat", [])
    return dict(streams=d.get("streams", 0), missing=d.get("missing", 0), n_ref=d.get("n_ref", 0), coverage=_rate(d.get("n_aligned", 0), d.get("n_ref", 0)),
                exact_rate=_rate(d.get("n_exact", 0), d.get("n_aligned", 0)), n_del=d.get("n_del", 0), n_ins=d.get("n_ins", 0), in_flush=d.get("in_flush", 0),
                latency_s=latency_stats(lat), latency_exact_s=latency_stats([x for x, e in zip(lat, d.get("exact", [])) if e]), lag_chunks=latency_stats(d.get("lag", [])),
                late_rate={_tkey(t): _late_rate(lat, t) for t in thresholds}, first_token_s=latency_stats(fl), first_token_skip=d.get("first_skip", 0),
                first_any_s=latency_stats(d.get("first_any_lat", [])), no_text=d.get("no_text", 0))

def _finalize_after_text(a: dict) -> dict:
    """after_text 블록(합산) → 청크 lag 분포·0 비율·2 청크 이상 비율(lag 에 넣은 유효 commit 전체, A 를 확정한 commit). in_flush = 그중 SEM 이 flush 라운드인 수."""
    part = lambda xs: dict(n=len(xs), lag_chunks=latency_stats(xs), zero_rate=_rate(sum(x == 0 for x in xs), len(xs)), ge2_rate=_rate(sum(x >= 2 for x in xs), len(xs)))
    return dict(streams=a.get("streams", 0), missing=a.get("missing", 0), mid_word=a.get("mid_word", 0), no_word=a.get("no_word", 0), no_text=a.get("no_text", 0),
                in_flush=a.get("in_flush", 0), **part(a.get("lag", [])), A=part(a.get("lag_A", [])))

def _finalize_windows(t: dict) -> dict:
    out = {}
    for k, v in t.items():
        if not k.startswith("late"): continue
        p, r = _rate(v["hits"], v["n_hyp"]), _rate(v["hits"], v["n_ref"])
        out[k] = dict(hits=v["hits"], n_hyp=v["n_hyp"], n_ref=v["n_ref"], miss=v["n_ref"] - v["hits"], false=v["n_hyp"] - v["hits"] - v.get("b_hits", 0), b_hits=v.get("b_hits", 0),
                      precision=p, recall=r, f1=_f1(p, r), precision_excl_B=_rate(v["hits"], v["n_hyp"] - v.get("b_hits", 0)),
                      latency_s=latency_stats(v["lat"]), lag_chunks=latency_stats(v["lag"]))
    return out

def finalize(m: dict, late_thresholds: Sequence[float] = LATE_THRESHOLDS_S) -> dict:
    """합산 지표 → 비율·분위수. timing.lateL: 청크 창 P/R/F1(ref_k 기준)·miss·지연(s, word_end 기준)·청크 지연; text: 위치 정밀도/재현율·PCR(범주별)·hard negative commit 률.
    pcr_raw = (premature + guard_blocked)/(n_hyp + guard_blocked): sem-guard 가 막은 발화(단어 전·SEM 연속)를 조기 commit 으로 되돌려 센 PCR(가드 끈 argmax 행동 근사).
    commit·display·after_text(창 없는 지연, 모듈 docstring '지연 지표')는 m 에 그 블록이 있을 때만 — late_thresholds(초)는 늦음률·지연 재현율 임계값,
    recall_within_chunks 의 청크 한도는 timing 의 late 창 목록."""
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
    lates = sorted(int(k[4:]) for k in m["timing"] if k.startswith("late") and k[4:].isdigit()) or list(LATES)
    if "commit" in m: out["commit"] = _finalize_commit(m["commit"], late_thresholds, lates)
    if "display" in m: out["display"] = _finalize_display(m["display"], late_thresholds)
    if "after_text" in m: out["after_text"] = _finalize_after_text(m["after_text"])
    return out

def aggregate(records: Iterable[dict], keys: Sequence[str] = ("lang", "set"), late_thresholds: Sequence[float] = LATE_THRESHOLDS_S) -> dict:
    """records: dict(metrics=score_stream(...), lang, set, …) → dict(overall, by_lang{…}, by_set{…}) (finalize 된 요약)."""
    tot, groups = None, {k: {} for k in keys}
    for r in records:
        tot = merge(tot, r["metrics"])
        for k in keys: groups[k][r.get(k)] = merge(groups[k].get(r.get(k)), r["metrics"])
    fin = lambda x: finalize(x, late_thresholds)
    return dict(overall=(fin(tot) if tot else None), **{f"by_{k}": {str(g): fin(v) for g, v in sorted(groups[k].items(), key=lambda kv: str(kv[0]))} for k in keys})

def pr_point(summary: Optional[dict], late: int = 2, late_thresholds: Optional[Sequence[float]] = None) -> dict:
    """PR 곡선의 한 점(요약 하나에서): 시간 창(late) P/R/F1·지연 p50 + 텍스트 위치 P/R·PCR
    + 창 없는 지연(commit coverage·지연 분위수·임계값별 지연 재현율, 표시·첫 토큰 지연, after-text lag) — 블록이 없는 옛 요약이면 그 키는 None.
    bias/θ 스윕 점을 모으면 지연–정확도 곡선. commit_recall_at 키는 요약의 임계값(없으면 late_thresholds, 기본 LATE_THRESHOLDS_S)."""
    if not summary: return {}
    w = summary["timing"].get(f"late{late}", {}); t = summary["text"]
    out = dict(n_hyp=t["n_hyp"], sem_per_min=summary["events"]["sem_per_min"], time_precision=w.get("precision"), time_recall=w.get("recall"), time_f1=w.get("f1"),
               latency_p50_s=(w.get("latency_s") or {}).get("p50"), text_precision=t["precision"], text_recall=t["recall"], text_f1=t["f1"], pcr=t["pcr"],
               pcr_raw=t.get("pcr_raw"), guard_blocked=t.get("guard_blocked", 0), hardneg_commit_rate=t["hardneg_commit_rate"])
    c = summary.get("commit") or {}; cl = c.get("latency_s") or {}; ra = c.get("recall_at") or {}
    ths = list(ra) if ra else [_tkey(x) for x in (late_thresholds or LATE_THRESHOLDS_S)]
    out.update(commit_coverage=c.get("coverage"), commit_exact_rate=c.get("exact_rate"), **{f"commit_latency_{p}_s": cl.get(p) for p in ("p50", "p90", "p95", "p99")},
               commit_excess_p90_chunks=(c.get("excess_chunks") or {}).get("p90"), prefix_latency_p90_s=((c.get("prefix") or {}).get("latency_s") or {}).get("p90"),
               **{f"commit_recall_at_{x}s": ra.get(x) for x in ths})
    d = summary.get("display") or {}; dl = d.get("latency_s") or {}; ft = d.get("first_token_s") or {}; at = summary.get("after_text") or {}
    out.update(display_coverage=d.get("coverage"), **{f"display_latency_{p}_s": dl.get(p) for p in ("p50", "p90", "p95", "p99")},
               first_token_p50_s=ft.get("p50"), first_token_p90_s=ft.get("p90"),
               after_text_zero_rate=at.get("zero_rate"), after_text_p90_chunks=(at.get("lag_chunks") or {}).get("p90"))
    return out

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

# ───────────────────────────── 골드셋(전수 주석) 대조 ─────────────────────────────
# 골드 행 {id, commit:[i], ambig:[i]} — 스트림의 모든 단어 경계를 판정한 것(목록에 없는 경계 = NO). 교사 라벨(Stage A 후보에 한정)과 달리 후보 누락까지 잡는다.
def gold_sets(g: Optional[dict]) -> Tuple[set, set]:
    g = g or {}
    return {int(i) for i in g.get("commit", [])}, {int(i) for i in g.get("ambig", [])}

def gold_label_counts(gold: Optional[dict], cands: Sequence[dict]) -> dict:
    """교사 라벨(labels.jsonl candidates, 등급 A/B/N) vs 골드 — 합산 가능한 카운트. cand_at_commit = 골드 확정 자리가 Stage A 후보였던 수(후보 재현)."""
    C, Am = gold_sets(gold); grade = {int(c["after_word"]): _grade(c) for c in cands}
    out = dict(gold_commit=len(C), gold_ambig=len(Am), cand=len(grade), cand_at_commit=len(set(grade) & C))
    for g in ("A", "B", "N"):
        P = {p for p, x in grade.items() if x == g}
        out.update({g: len(P), f"{g}_at_commit": len(P & C), f"{g}_at_ambig": len(P & Am), f"{g}_at_no": len(P - C - Am)})
    return out

def finalize_gold_labels(c: dict) -> dict:
    """A 정밀도(골드 AMBIG 자리 제외)·A 재현율·후보 재현율·N 정밀도(= 골드 NO 비율, AMBIG 제외)·B 중 골드 확정 비율."""
    r = lambda a, b: round(a / b, 4) if b else None
    return dict(c, A_precision=r(c["A_at_commit"], c["A"] - c["A_at_ambig"]), A_recall=r(c["A_at_commit"], c["gold_commit"]), cand_recall=r(c["cand_at_commit"], c["gold_commit"]),
                N_precision=r(c["N_at_no"], c["N"] - c["N_at_ambig"]), B_commit_share=r(c["B_at_commit"], c["B"]))

def gold_commit_counts(words: Sequence[dict], gold: Optional[dict], hyp_words: Sequence[str], events: Sequence[dict], ref_k: Optional[Sequence[int]] = None,
                       K: Optional[int] = None, norm: Callable[[str], str] = norm_word) -> dict:
    """모델 <SEM_END>(map_events 로 '참조 단어 p 뒤' 에 사상) vs 골드 — 합산 가능한 카운트. 단어 중간·첫 단어 전·같은 자리 중복은 text_position_metrics 와 같은 규칙.
    hit = 골드 COMMIT 자리, at_ambig = 골드 AMBIG(정밀도에서 뺀다), at_no = 그 밖(조기·오확정). lat = 맞힌 확정의 (방출 시각 − 그 단어 끝)."""
    C, Am = gold_sets(gold); ws = sorted(words, key=lambda w: int(w["i"]))
    pos = map_events([w["text"] for w in ws], hyp_words, events, ref_k, K, norm)
    out = dict(n_hyp=len(events), hit=0, at_ambig=0, at_no=0, dup=0, mid_word=0, no_word=0, gold_commit=len(C), lat=[])
    seen = set()
    for ev, p in zip(events, pos):
        if ev.get("mid_word"): out["mid_word"] += 1; continue
        if p < 0: out["no_word"] += 1; continue
        if p in seen: out["dup"] += 1; continue
        seen.add(p)
        if p in C: out["hit"] += 1; out["lat"].append(round(emit_time(int(ev["k"]), K) - float(ws[p]["end_time"]), 4))
        elif p in Am: out["at_ambig"] += 1
        else: out["at_no"] += 1
    return out

def finalize_gold_commits(c: dict) -> dict:
    """P = hit ÷ (확정 − 골드 AMBIG 자리) (중복·단어 중간·첫 단어 전은 오류로 셈), R = hit ÷ 골드 COMMIT, F1, PCR_gold = (골드 NO + 단어 중간 + 첫 단어 전 + 중복) ÷ 확정."""
    den = c["n_hyp"] - c["at_ambig"]; P = c["hit"] / den if den else None; R = c["hit"] / c["gold_commit"] if c["gold_commit"] else None
    bad = c["at_no"] + c["mid_word"] + c["no_word"] + c["dup"]
    return dict({k: v for k, v in c.items() if k != "lat"}, P=None if P is None else round(P, 4), R=None if R is None else round(R, 4), F1=round(_f1(P, R), 4) if P is not None and R is not None else None,
                PCR_gold=round(bad / c["n_hyp"], 4) if c["n_hyp"] else None, latency_s=latency_stats(c["lat"]))

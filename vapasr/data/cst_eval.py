"""Stage 3 평가 도우미(experiments/cst_stage3_eval.py) — 슬롯 출력 해석, 화자 귀속 오류율, 턴 끝 <EOT> 일치. 순수 함수(모델·tokenizer 없이 테스트).
슬롯 수: 2(V1) 또는 최대 4(V2 — spk_ids 길이, 참조 슬롯 = 창 화자 수). cp 는 가설 슬롯 → 참조 슬롯의 모든 순열 중 최소(4 슬롯이면 24 개), 같으면 그대로 짝(항등)을 고른다.
Stage 4(experiments/cst_stage4_eval.py): parse_emitted4(전사·번역 네 흐름), lag_matched(번역 단어 지연 — 학습 타깃 시각 대비).

출력 해석(parse_emitted): 스트리밍 [(k, tid)] — <SPK_A>/<SPK_B> 태그가 슬롯을 열고, 그 뒤 토큰·<EOT> 는 그 슬롯 것이다. 첫 태그 앞 토큰은 슬롯 A 로 본다
  (태그를 빠뜨려도 받아쓰기 평가는 되게). 방출 시각 = 청크 끝 (k+1)·80 ms (ast_eval.word_emission_times 와 같은 규약).
  E9 슬롯 블록(2026-10-28): 디코더가 슬롯 출력 앞에 그 슬롯 태그를 기록하므로 같은 규칙으로 읽는다. <BEGIN>·<EOT> 는 activity_from_events 로 슬롯 활동이 된다
  (LLM 쪽 화자 분리 — vapasr.hf.diar.der_counts 로 DER).
  활동 추종 진단(follow_stats, 2026-10-30): 단어가 디코드에 준 슬롯 활동(E11 이면 고정 전문가)이 켜진 슬롯에 들어갔나 — 다른 슬롯·무음·겹침, 정답으로 누가 맞았나,
  어긋난 단어를 활동 슬롯으로 옮기면 cp. 행에는 act_runs(활동 구간)로 싣는다.
단위(units): 한국어 = 글자(공백 없는 CER, score_ko), 영어 = 단어(score_en), 독일어 = 단어(score_de).
  영어 정규화(set_en_norm, 2026-10-27): ours = score_en(소문자·문장부호·숫자, 채움말 남김 — 기본, 예전 값) | whisper = Whisper EnglishTextNormalizer
  (whisper_normalizer — 채움말 hmm·mm·mhm·mmm·uh·um 제거, 줄임말·영국 철자·숫자·호칭 표준화; 외부 AMI 수치와 견줄 때). 시각을 지키려고 단어마다 정규화한다
  — 여러 단어 숫자("twenty one")는 단어별로 바뀌지만 참조·가설에 같게 적용된다. 회의 통째 문장 단위 대조는 experiments/meeteval_cpwer.py.
창 점수(score_window):
  slot  : 슬롯을 그대로(A↔A, B↔B) 맞춘 오류 합 — 형식 규칙(먼저 말한 사람 = A)까지 지킨 화자 귀속
  cp    : 두 슬롯 순열 중 작은 쪽(cpWER/cpCER, 2 화자) — 화자 귀속
  agn   : 화자 무시 — 두 슬롯 단어를 시각순으로 이어 붙여 비교(겹침 구간 순서 차이가 조금 섞인다)
  swap  : cp 가 순열을 바꾼 창(슬롯 A/B 를 뒤집어 낸 창)
턴 끝(match_eot): 참조 = 창 안 episode 끝(슬롯·끝 시각·p_end). 양성 = p ≥ 0.8(교대·침묵), 보류 = p ≤ 0.3(본인 재개), 0.5(both)는 정밀도에서만 맞음으로 친다.
  가설 <EOT>(슬롯·방출 시각)는 cp 순열로 참조 슬롯에 옮긴 뒤 같은 슬롯의 참조 끝 e 와 e ≤ t ≤ e + tol 이면 짝(시간순 탐욕 1:1).
  recall = 짝지은 양성 / 양성, precision = (p ≥ 0.5 참조와 짝지은 가설) / 가설, hold_fire = 보류 참조와 짝지은 가설 / 보류 참조, 지연 = t − e(양성 짝).
"""
import functools
from typing import Dict, List, Optional, Sequence, Tuple

CHUNK_S = 0.08
EN_NORM = "ours"
_WN = None


def parse_emitted(emitted: Sequence, spk_ids: Sequence[int], eot_id: int, drop_chunks=(), begin_id: Optional[int] = None,
                  events: Optional[list] = None) -> Tuple[List[List[Tuple[float, int]]], List[Tuple[int, float]]]:
    """→ (슬롯별 [(방출 시각 s, tid)], [(슬롯, 방출 시각 s)] <EOT>). drop_chunks 의 청크에서 나온 토큰·EOT 는 버린다(전사 없는 음성 구간).
    begin_id(E9 <BEGIN>): 주면 그 토큰은 글자가 아니다. events(list 를 주면 채운다 — E9 LLM 쪽 화자 분리): drop 과 무관하게 모든
    ('B' | 'E', 슬롯, 청크 k) — <BEGIN>·<EOT>, 그리고 ('W', 슬롯, k) — 글자 토큰(구간 추정 보조, activity_from_events)."""
    slots: List[List[Tuple[float, int]]] = [[] for _ in spk_ids]; eots: List[Tuple[int, float]] = []; cur = 0; drop = set(drop_chunks)
    for k, t in emitted:
        t = int(t)
        if t in spk_ids: cur = list(spk_ids).index(t); continue
        if events is not None: events.append(("B" if t == begin_id else "E" if t == eot_id else "W", cur, int(k)))
        if t == begin_id or k in drop: continue
        tt = (int(k) + 1) * CHUNK_S
        if t == eot_id: eots.append((cur, tt))
        else: slots[cur].append((tt, t))
    return slots, eots


def activity_from_events(events: Sequence[Tuple[str, int, int]], K: int, n: int, delay: int, begin_lag: int = 4):
    """E9 LLM 쪽 화자 분리: parse_emitted events → 슬롯 활동 (K, n) 0/1 numpy. 학습 규칙의 역: <BEGIN> 청크 − begin_lag = 시작 프레임,
    <EOT> 청크 − δ = 마지막 프레임(cst_dialogue.block_streams). 슬롯마다 시간순으로 <BEGIN> 이 구간을 열고 <EOT> 가 닫는다.
    열린 구간 없이 <EOT> 가 오면 지난 닫힘 뒤 첫 글자(청크 − δ), 글자도 없고 앞 사건도 없으면 창 시작(창 앞에서 시작한 발화)부터, 그 밖에는 그 프레임 하나.
    열린 채 <BEGIN> 이 또 오면 앞 구간은 새 시작 직전에 닫고, 끝까지 열려 있으면 K 까지. 슬롯 n 개 밖은 버린다."""
    import numpy as np
    act = np.zeros((K, n), np.uint8)
    for s in range(n):
        ev = [(k, typ) for typ, sl, k in events if sl == s]; open_at, seen, word = None, False, None
        for k, typ in ev:
            if typ == "W":
                if open_at is None and word is None: word = max(0, k - delay)
                continue
            if typ == "B":
                a = max(0, min(K, k - begin_lag))
                if open_at is not None: act[open_at: max(open_at, a), s] = 1
                open_at, seen, word = a, True, None; continue
            e = max(0, min(K, k - delay + 1))                                                     # <EOT>
            a = open_at if open_at is not None else word if word is not None else (0 if not seen else max(0, e - 1))
            act[min(a, e): e, s] = 1; open_at, seen, word = None, True, None
        if open_at is not None: act[open_at:, s] = 1
    return act


def emitted_from_blocks(ids: Sequence[int], is_input: Sequence[bool], tag_chunk: Sequence[int], slot_at: Sequence[int], spk_ids: Sequence[int],
                        trans_id: Optional[int] = None) -> List[Tuple[int, int]]:
    """E9 학습 시퀀스(정답, cst_dialogue.build_block_sequence) → batch_decode.decode_blocks_batch 와 같은 [(k, tid)](슬롯 출력 앞에 그 슬롯 태그 — 오라클 평가).
    trans_id(E13 번역 모드): <TRANS> 입력 뒤 출력은 그 슬롯 번역 — 앞에 (k, 슬롯 태그), (k, <TRANS>)(BlockState 와 같다, parse_emitted4 가 읽는다)."""
    out: List[Tuple[int, int]] = []; cur = None; tagged = False; kind = 0
    for p, (t, inp) in enumerate(zip(ids, is_input)):
        if inp:
            if tag_chunk[p] >= 0: cur, tagged, kind = (int(tag_chunk[p]), int(slot_at[p])), False, 0
            elif trans_id is not None and int(t) == trans_id and slot_at[p] >= 0: tagged, kind = False, 1
            continue
        if not tagged: out += [(cur[0], int(spk_ids[cur[1]]))] + ([(cur[0], int(trans_id))] if kind else []); tagged = True
        out.append((cur[0], int(t)))
    return out


def parse_emitted4(emitted: Sequence, spk_ids: Sequence[int], trans_id: int, eot_id: int, drop_chunks=(), drop_tr_chunks=(), begin_id: Optional[int] = None):
    """Stage 4 네 흐름 출력 해석 → (슬롯별 전사 [(시각 s, tid)], 슬롯별 번역 [(시각 s, tid)], [(슬롯, 시각 s)] <EOT>).
    <SPK_x> 는 슬롯 x 전사 구간을, <TRANS> 는 지금 슬롯의 번역 구간을 연다(cst_dialogue.read_back4 와 같은 규칙). 첫 태그 앞은 슬롯 A 전사로 본다.
    drop_chunks: 버릴 청크(전사 없는 음성 — 전사·번역·<EOT> 모두), drop_tr_chunks: 번역만 버릴 청크(참조 번역이 없는 행).
    begin_id(E13 슬롯 블록 번역 — batch_decode.BlockState 출력): 그 토큰(<BEGIN>)은 글자가 아니다."""
    src: List[List[Tuple[float, int]]] = [[] for _ in spk_ids]; tr: List[List[Tuple[float, int]]] = [[] for _ in spk_ids]; eots: List[Tuple[int, float]] = []
    cur, kind = 0, 0; drop, dtr = set(drop_chunks), set(drop_tr_chunks); spk = list(spk_ids)
    for k, t in emitted:
        t = int(t)
        if t in spk: cur, kind = spk.index(t), 0; continue
        if t == trans_id: kind = 1; continue
        if k in drop or t == begin_id: continue
        tt = (int(k) + 1) * CHUNK_S
        if t == eot_id: eots.append((cur, tt))
        elif kind == 0: src[cur].append((tt, t))
        elif k not in dtr: tr[cur].append((tt, t))
    return src, tr, eots


def lag_matched(ref: Sequence[Tuple[str, float]], hyp: Sequence[Tuple[str, float]]) -> List[float]:
    """번역 지연: 참조·가설 단어열 [(단어, 시각)]을 같은 비용 정렬(align)로 맞춰 일치한 단어의 (가설 시각 − 참조 시각) 목록.
    참조 시각 = 학습 타깃의 방출 청크(stable prefix 확정 + δ) — 0 이면 타깃만큼 빠르다."""
    al = align([w for w, _ in ref], [w for w, _ in hyp])
    return [round(hyp[j][1] - ref[i][1], 3) for i, j in al if i is not None and j is not None and ref[i][0] == hyp[j][0]]


def emitted_from_sequence(ids: Sequence[int], is_input: Sequence[bool], chunk_of: Sequence[int], next_id: int, empty_id: int) -> List[Tuple[int, int]]:
    """학습 시퀀스(정답) → decode_batch 와 같은 [(k, tid)] — <NEXT_AUDIO> 제외, flush 는 k = K, K+1, …(오라클 평가)."""
    K = max(c for c in chunk_of) + 1; out: List[Tuple[int, int]] = []; k = -1; n_empty = 0
    for t, inp, c in zip(ids, is_input, chunk_of):
        if inp:
            if c >= 0: k = c
            elif t == empty_id: k = K + n_empty; n_empty += 1
            continue
        if t != next_id: out.append((k, int(t)))
    return out


def set_en_norm(name: str) -> None:
    """영어 채점 정규화를 고른다 — ours(기본) | whisper. 모듈 전체(이 프로세스의 채점)에 걸린다."""
    global EN_NORM
    assert name in ("ours", "whisper"), f"영어 정규화 ours|whisper: {name}"
    EN_NORM = name


@functools.lru_cache(maxsize=1 << 20)
def whisper_en(text: str) -> Tuple[str, ...]:
    """Whisper EnglishTextNormalizer 단어열(같은 문자열은 캐시 — 채점은 단어마다 부른다)."""
    global _WN
    if _WN is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _WN = EnglishTextNormalizer()
    return tuple(_WN(text).split())


def units(text: str, lang: str) -> List[str]:
    if lang == "Korean":
        from .textnorm import score_ko
        return list(score_ko(text, False))
    if lang == "German":
        from .textnorm_de import score_de
        return score_de(text).split()
    if EN_NORM == "whisper": return list(whisper_en(text))
    from .textnorm import score_en
    return score_en(text).split()


def _ed_bits(a: Sequence, b: Sequence) -> int:
    """Myers·Hyyrö 비트 병렬 Levenshtein(단위 비용 — ed 와 같은 값). a 의 위치를 Python 정수 비트로 둬 b 한 원소마다 정수 연산 몇 번 — 회의 통째(화자당 수천 단어)."""
    m = len(a)
    if not m: return len(b)
    peq = {}
    for i, x in enumerate(a): peq[x] = peq.get(x, 0) | (1 << i)
    full, top = (1 << m) - 1, 1 << (m - 1); pv, mv, d = full, 0, m
    for y in b:
        eq = peq.get(y, 0); xv = eq | mv
        xh = ((((eq & pv) + pv) & full) ^ pv) | eq
        ph = mv | (~(xh | pv) & full); mh = pv & xh
        if ph & top: d += 1
        elif mh & top: d -= 1
        ph = ((ph << 1) | 1) & full; mh = (mh << 1) & full                                   # 첫 행 D[0][j] = j(전역 거리)
        pv = mh | (~(xv | ph) & full); mv = ph & xv
    return d


def ed(a: Sequence, b: Sequence) -> int:
    if len(a) * len(b) > 250_000: return _ed_bits(a, b)                                     # 긴 열(회의 통째 --window full) — 같은 값, 창 평가는 아래 DP 그대로
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def relabel_by_activity(hyp: Sequence[Sequence[Tuple[str, float]]], act, delta: int, back: int = 3, fwd: int = 1, wide: int = 12, floor: float = 0.2):
    """진단(E7, 2026-10-28 — 회의 통째 귀속 붕괴): 가설 단어 [(단어, 방출 시각)] 를 슬롯과 상관없이 활동으로 다시 나눈다. 단어의 오디오 프레임 ≈ 방출 청크 − δ
    (방출 시각 = (k + 1)·80 ms), 그 근처 [−back, +fwd] 프레임 활동 합이 가장 큰 슬롯으로. 근처 최대 활동이 floor 아래면 ±wide 로 넓힌다.
    act (K, N) 활동 확률 또는 0/1 정답 → 슬롯 N 개 [(단어, 시각)](시각순). LLM 태그가 회의 내내 같은 사람을 가리켰다면 얻었을 귀속을 어림한다."""
    import numpy as np
    A = np.asarray(act, dtype=np.float32); K, N = A.shape; out: List[List[Tuple[str, float]]] = [[] for _ in range(N)]
    for sl in hyp:
        for w, t in sl:
            k = int(round(t / CHUNK_S)) - 1 - int(delta)
            lo, hi = max(0, k - back), min(K, k + fwd + 1)
            if hi <= lo: lo, hi = max(0, min(k, K - 1)), max(0, min(k, K - 1)) + 1
            if A[lo:hi].max() < floor: lo, hi = max(0, k - wide), max(1, min(K, k + wide + 1))
            out[int(A[lo:hi].sum(0).argmax())].append((w, t))
    return [sorted(x, key=lambda z: z[1]) for x in out]


def act_runs(act) -> List[List[List[int]]]:
    """슬롯 활동 (K, N) 0/1 → 슬롯마다 [시작, 끝) 프레임 구간 목록 — 평가 행에 싣는 꼴(act_from_runs 로 되돌린다)."""
    import numpy as np
    A = np.asarray(act) > 0; out = []
    for s in range(A.shape[1]):
        d = np.diff(np.concatenate([[0], A[:, s].astype(np.int8), [0]]))
        out.append([[int(a), int(b)] for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])])
    return out


def act_from_runs(runs, K: int):
    """act_runs 의 역 → (K, 슬롯) uint8."""
    import numpy as np
    A = np.zeros((K, len(runs)), np.uint8)
    for s, rr in enumerate(runs):
        for a, b in rr: A[a:b, s] = 1
    return A


def follow_stats(hyp: Sequence[Sequence[Tuple[str, float]]], act, delta: int, ref_act=None, perm: Optional[Sequence[int]] = None, frame_mask=None,
                 back: int = 3, fwd: int = 1, detail: Optional[List] = None):
    """진단(2026-10-30, 사용자 "LLM 활동 추종 진단 구현해놓자" — wiki output-n3d-branch-20261029 6.5 절): LLM 이 슬롯 활동 입력을 얼마나 따르나.
    hyp: 슬롯별 [(단어, 방출 시각 s)](score_window 의 가설), act (K, N): 디코드가 받은 슬롯 활동 0/1(채널 s = 슬롯 s — E11 이면 고정 전문가, E9 면 헤드).
    단어의 오디오 프레임 f = 방출 청크 − δ(relabel_by_activity 와 같은 규약), 창 [f − back, f + fwd] 에 그 단어 슬롯이 활성이면 on, 아니고 다른 슬롯이 활성이면
    off_other(활동은 다른 사람이라 본다), 아무 슬롯도 아니면 off_none(활동은 무음이라 본다). 창 안 활성 슬롯이 하나면 단독(sg), 둘 이상이면 겹침(ov) 단어다.
    ref_act (K, Nr) 정답 슬롯 활동·perm(score_window — 가설 슬롯 → 참조 슬롯)을 주면 누가 맞았나도 센다: 단어 슬롯의 참조 화자가 창 안에서 말했으면 LLM 맞음,
    off_other 단어의 활동 슬롯(창 활동 합이 가장 큰 슬롯)의 참조 화자가 말했으면 활동 맞음 — oo_l{0,1}e{0,1}(LLM·활동 맞음). on_ok·none_ok 는 LLM 맞음 수.
    frame_mask (K,) bool(False = 전사를 못 믿는 구간 — LLM DER 마스크)이면 그 프레임 단어는 세지 않는다(skipped, 슬롯은 그대로).
    detail(2026-10-30, 활동 추종 학습 설계용 진단): 리스트를 주면 센 단어마다 [범주(0 on·1 off_other·2 off_none), f, 단어 슬롯 s, 활동 슬롯 e, 창 안 활성 슬롯 수,
    s 가 마지막으로 활성이던 뒤 프레임(창 앞, 없으면 −1), e 의 말 차례(활성 구간) 시작 뒤 프레임(창에서 처음 활성인 구간, 무음이면 −1), 그 차례 직전 화자 = s(0/1),
    LLM 맞음, 활동 맞음(ref_act 없으면 −1), e 가 그 차례 전에 쉰 프레임(재등장 간격 — 처음 말하는 차례·무음이면 −1)] 을 붙인다.
    → (셈 dict, moved — off_other 단어만 활동 슬롯으로 옮긴 슬롯별 [(단어, 시각)] 시각순: LLM 이 활동을 따랐다면의 가설, score_window 로 cp 를 잰다)."""
    import numpy as np
    A = np.asarray(act) > 0; K, N = A.shape; R = None if ref_act is None else np.asarray(ref_act) > 0
    M = None if frame_mask is None else np.asarray(frame_mask, dtype=bool)
    if detail is not None:
        idx = [np.nonzero(A[:, x])[0] for x in range(N)]; anyo = [np.nonzero(np.delete(A, x, axis=1).any(1))[0] for x in range(N)]
        def last_before(x, j):                                                                      # 프레임 j 앞(< j)에서 슬롯 x 가 마지막으로 활성인 프레임, 없으면 −1
            k = np.searchsorted(idx[x], j) - 1; return int(idx[x][k]) if k >= 0 else -1
        def turn(e, lo, hi):                                                                        # 창에서 e 가 처음 활성인 프레임이 든 구간의 시작, 그 직전(구간 앞)에 말한 다른 슬롯, e 가 쉰 프레임
            j = lo + int(np.argmax(A[lo:hi, e])); st_ = j
            while st_ > 0 and A[st_ - 1, e]: st_ -= 1
            k = np.searchsorted(anyo[e], st_) - 1
            prev = -1 if k < 0 else int(np.argmax(np.where(np.arange(N) == e, -1, A[anyo[e][k]].astype(int))))
            le = last_before(e, st_)
            return st_, prev, (st_ - le if le >= 0 else -1)
    st = dict.fromkeys(("n", "on", "off_other", "off_none", "sg", "sg_off", "ov", "ov_off", "skipped")
                       + (("on_ok", "none_ok", "oo_l1e1", "oo_l1e0", "oo_l0e1", "oo_l0e0") if R is not None else ()), 0)
    moved: List[List[Tuple[str, float]]] = [[] for _ in range(max(N, len(hyp)))]
    ok = lambda x, lo, hi: x < len(perm) and perm[x] < R.shape[1] and bool(R[lo:hi, perm[x]].any())
    for s, sl in enumerate(hyp):
        for w, t in sl:
            f = min(max(int(round(t / CHUNK_S)) - 1 - int(delta), 0), K - 1)
            if M is not None and not M[f]: st["skipped"] += 1; moved[s].append((w, t)); continue
            lo, hi = max(0, f - back), min(K, f + fwd + 1); a = A[lo:hi]; on = a.any(0); n_on = int(on.sum()); mine = s < N and bool(on[s]); e = s
            st["n"] += 1
            if n_on == 1: st["sg"] += 1; st["sg_off"] += int(not mine)
            elif n_on > 1: st["ov"] += 1; st["ov_off"] += int(not mine)
            if mine: st["on"] += 1
            elif n_on: st["off_other"] += 1; e = int(a.sum(0).argmax())
            else: st["off_none"] += 1
            moved[e].append((w, t))
            llm = act_ok = -1
            if R is not None:
                llm = ok(s, lo, hi)
                if mine: st["on_ok"] += int(llm)
                elif n_on: act_ok = ok(e, lo, hi); st[f"oo_l{int(llm)}e{int(act_ok)}"] += 1
                else: st["none_ok"] += int(llm)
            if detail is not None:
                c = 0 if mine else 1 if n_on else 2
                since = (f - last_before(s, lo)) if (c and s < N and last_before(s, lo) >= 0) else (0 if c == 0 else -1)
                if n_on: st_, prev, gap = turn(e, lo, hi); d0, ps = f - st_, int(c == 1 and prev == s)
                else: d0, ps, gap = -1, 0, -1
                detail.append([c, f, s, e, n_on, since, d0, ps, int(llm), int(act_ok), gap])
    return st, [sorted(x, key=lambda z: z[1]) for x in moved]


def follow_pool(rows: Sequence[Dict]) -> Optional[Dict]:
    """행의 follow(follow_stats 셈)·cp_follow·gated(활동 게이트 디코드가 막은 슬롯 첫 출력) 합 → 비율 dict(%), 그런 행이 없으면 None."""
    ff = [r["follow"] for r in rows if "follow" in r]
    if not ff: return None
    S = {k: sum(x.get(k, 0) for x in ff) for k in ff[0]}; pc = lambda x, d: round(100 * x / max(1, d), 2)
    out = dict(words=S["n"], on=pc(S["on"], S["n"]), off_other=pc(S["off_other"], S["n"]), off_none=pc(S["off_none"], S["n"]),
               sg_off=pc(S["sg_off"], S["sg"]), ov_share=pc(S["ov"], S["n"]), ov_off=pc(S["ov_off"], S["ov"]))
    if "oo_l1e0" in S:
        oo = S["off_other"]
        out.update(oo_llm_only=pc(S["oo_l1e0"], oo), oo_act_only=pc(S["oo_l0e1"], oo), oo_both=pc(S["oo_l1e1"], oo), oo_neither=pc(S["oo_l0e0"], oo),
                   on_ok=pc(S["on_ok"], S["on"]), none_ok=pc(S["none_ok"], S["off_none"]))
    cf = [r for r in rows if "cp_follow" in r]
    if cf: out["err_cp_follow"] = round(100 * sum(r["cp_follow"] for r in cf) / max(1, sum(r["n_ref"] for r in cf)), 2)
    g = [r["gated"] for r in rows if "gated" in r]
    if g: out.update(gated_text=sum(x["text"] for x in g), gated_begin=sum(x["begin"] for x in g))
    return out


def follow_line(f: Dict) -> str:
    """follow_pool → 보고 한 줄에 붙이는 꼬리."""
    s = f" · 활동 추종 on {f['on']:.1f} % · 다른 슬롯 {f['off_other']:.1f} %"
    if "oo_llm_only" in f: s += f"(LLM 만 맞음 {f['oo_llm_only']:.0f} · 활동만 {f['oo_act_only']:.0f} · 둘 다 {f['oo_both']:.0f} · 둘 다 틀림 {f['oo_neither']:.0f})"
    s += f" · 무음 {f['off_none']:.1f} % · 겹침 단어 {f['ov_share']:.1f} %(그중 off {f['ov_off']:.1f})"
    if "err_cp_follow" in f: s += f" · 따르면 cp {f['err_cp_follow']:.2f}"
    if "gated_text" in f: s += f" · 게이트가 막은 첫 출력 글자 {f['gated_text']} · <BEGIN> {f['gated_begin']}"
    return s


def score_window(ref: Sequence[Sequence[Tuple[str, float]]], hyp: Sequence[Sequence[Tuple[str, float]]], lang: str) -> Dict:
    """ref·hyp: 슬롯별 [(단어 문자열, 시각 s)](슬롯 수는 서로 달라도 된다 — 모자란 쪽은 빈 슬롯). → dict(n_ref, slot, cp, agn, swap, perm).
    perm[h] = 가설 슬롯 h 를 짝지은 참조 슬롯(참조 슬롯 수 이상이면 참조에 없는 슬롯 — 그 말은 모두 삽입). 2 슬롯은 예전 정의와 같다."""
    from itertools import permutations
    U = lambda ws: [u for w, _ in ws for u in units(w, lang)]
    n = max(2, len(ref), len(hyp)); Rs = [U(ref[i]) if i < len(ref) else [] for i in range(n)]; Hs = [U(hyp[i]) if i < len(hyp) else [] for i in range(n)]
    cost = [[ed(Rs[r], Hs[h]) for h in range(n)] for r in range(n)]                     # cost[참조][가설]
    same = sum(cost[i][i] for i in range(n)); best, bp = same, tuple(range(n))
    for p in permutations(range(n)):                                                    # p[h] = 가설 h 의 참조 슬롯(항등이 먼저 — 같으면 그대로)
        c = sum(cost[p[h]][h] for h in range(n))
        if c < best: best, bp = c, p
    ra = [u for w, _ in sorted([x for sl in ref for x in sl], key=lambda x: x[1]) for u in units(w, lang)]
    ha = [u for w, _ in sorted([x for sl in hyp for x in sl], key=lambda x: x[1]) for u in units(w, lang)]
    return dict(n_ref=sum(map(len, Rs)), n_hyp=sum(map(len, Hs)), slot=same, cp=best, agn=ed(ra, ha), swap=bool(best < same), perm=list(bp))


def match_eot(ref: Sequence[Tuple[int, float, Optional[float]]], hyp: Sequence[Tuple[int, float]], perm: Sequence[int], tol: float) -> Dict:
    """ref: [(슬롯, 끝 시각, p_end)], hyp: [(가설 슬롯, 방출 시각)], perm: 가설 슬롯 → 참조 슬롯. → 개수·지연 목록."""
    refs = sorted([(s, e, p) for s, e, p in ref if p is not None], key=lambda x: x[1]); used = [False] * len(refs)
    out = dict(n_hyp=len(hyp), n_pos=sum(p >= 0.8 for _, _, p in refs), n_hold=sum(p <= 0.3 for _, _, p in refs), n_both=sum(0.3 < p < 0.8 for _, _, p in refs),
               hit_pos=0, hit_hold=0, hit_both=0, unmatched=0, lat=[])
    for hs, t in sorted(hyp, key=lambda x: x[1]):
        s = perm[hs]; j = next((j for j, (rs, e, _) in enumerate(refs) if not used[j] and rs == s and e - 1e-9 <= t <= e + tol), None)
        if j is None: out["unmatched"] += 1; continue
        used[j] = True; p = refs[j][2]
        if p >= 0.8: out["hit_pos"] += 1; out["lat"].append(round(t - refs[j][1], 3))
        elif p <= 0.3: out["hit_hold"] += 1
        else: out["hit_both"] += 1
    return out


def eot_tol(delay: int) -> float:
    """짝 허용 = 후보 위치(끝 + 0.24 s 뒤 청크, δ 청크 지연) + 0.5 s 여유."""
    return 0.24 + (delay + 1) * CHUNK_S + 0.5


def pool(rows: Sequence[Dict]) -> Dict:
    """창 결과 목록 → 합산 지표(오류율 %, EOT 정밀도·재현율·hold 오방출·지연 중앙값)."""
    import statistics
    n = sum(r["n_ref"] for r in rows) or 1; e = [x for r in rows for x in r["eot"]["lat"]]
    nh = sum(r["eot"]["n_hyp"] for r in rows); npos = sum(r["eot"]["n_pos"] for r in rows); nhold = sum(r["eot"]["n_hold"] for r in rows)
    good = sum(r["eot"]["hit_pos"] + r["eot"]["hit_both"] for r in rows)
    return dict(windows=len(rows), ref_units=n, err_slot=round(100 * sum(r["slot"] for r in rows) / n, 2), err_cp=round(100 * sum(r["cp"] for r in rows) / n, 2),
                err_agn=round(100 * sum(r["agn"] for r in rows) / n, 2), swap_share=round(sum(r["swap"] for r in rows) / max(1, len(rows)), 4),
                eot_hyp=nh, eot_pos=npos, eot_recall=round(sum(r["eot"]["hit_pos"] for r in rows) / max(1, npos), 4), eot_precision=round(good / max(1, nh), 4),
                eot_hold_fire=round(sum(r["eot"]["hit_hold"] for r in rows) / max(1, nhold), 4), eot_lat_median=round(statistics.median(e), 3) if e else None,
                eot_lat_p90=round(sorted(e)[int(0.9 * (len(e) - 1))], 3) if e else None)


def align(a: Sequence, b: Sequence) -> List[Tuple[Optional[int], Optional[int]]]:
    """ed 와 같은 비용의 정렬 역추적 → [(i, j)]: 둘 다 있으면 일치·치환, j None = 삭제, i None = 삽입. 비용이 같으면 대각 > 삭제 > 삽입."""
    n, m = len(a), len(b); D = [list(range(m + 1))] + [[i] + [0] * m for i in range(1, n + 1)]
    for i in range(1, n + 1):
        x, Di, Dp = a[i - 1], D[i], D[i - 1]
        for j in range(1, m + 1): Di[j] = min(Dp[j] + 1, Di[j - 1] + 1, Dp[j - 1] + (x != b[j - 1]))
    out: List[Tuple[Optional[int], Optional[int]]] = []; i, j = n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + (a[i - 1] != b[j - 1]): out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i and D[i][j] == D[i - 1][j] + 1: out.append((i - 1, None)); i -= 1
        else: out.append((None, j - 1)); j -= 1
    return out[::-1]


ATTR_TYPES = ("whole_short", "whole_long", "late", "early", "inner")


def attr_errors(ref: Sequence[Sequence[Tuple[str, float]]], hyp: Sequence[Sequence[Tuple[str, float]]], lang: str, perm: Sequence[int],
                busy: Optional[Sequence[Sequence[Tuple[float, float]]]] = None, short_words: int = 3) -> Dict:
    """화자 귀속 오류 유형. ref·hyp: 슬롯별 [(단어, 시각)], perm: 가설 슬롯 → 참조 슬롯(score_window 의 cp 순열), busy: 참조 슬롯별 말하는 구간(창 기준 s).
    화자 무시 정렬(score_window 의 agn 과 같은 시각순·같은 비용)에서 짝지은(일치·치환) 단위의 슬롯이 참조와 다르면 오배정이다.
    참조 run = 시각순으로 같은 슬롯이 이어진 구간(한 화자의 턴 조각). run 안에서 이어진 오배정 덩어리를 위치로 나눈다:
      whole_short·whole_long : run 전체 — 턴을 통째로 다른 화자에 붙임(short = run 이 short_words 단어 이하, 맞장구·짧은 응답)
      late  : run 첫머리부터 — 교대를 늦게 바꿈(새 턴 첫머리를 직전 화자로)
      early : run 끝까지 — 교대를 일찍 바꿈(턴 끝을 다음 화자로)
      inner : run 가운데 — 턴 중간의 헛교대
    겹침(ov) = 그 단위 시각에 다른 슬롯 화자도 말하고 있음(busy), 아니면 solo.
    통째 오배정 run 은 다시 나눈다: 홀로(앞뒤 run 은 통째로 틀리지 않음 — 턴 하나를 주변 화자에 삼킴) / 이어짐(연속 run 이 통째로 틀림 — 여러 턴의 정체가 뒤바뀜).
    → dict(n_ref, agn, n_pair, n_mis, units{'유형/ov|solo': 단위 수}, runs, runs_short, whole_short_runs, whole_long_runs, whole_iso_units, whole_streak_units,
           whole_streak_edge_units(이어짐 중 창 처음·끝 run 에 닿는 것), ref_changes, hyp_changes, late_words)."""
    rw = sorted([(t, s, k, w) for s in range(len(ref)) for k, (w, t) in enumerate(ref[s])], key=lambda x: x[0])
    R = [(u, s, (s, k), t) for t, s, k, w in rw for u in units(w, lang)]
    hw = sorted([(t, perm[s], w) for s in range(len(hyp)) for w, t in hyp[s]], key=lambda x: x[0])
    H = [(u, s) for _, s, w in hw for u in units(w, lang)]
    pairs = align([x[0] for x in R], [x[0] for x in H])
    lab: List[Optional[int]] = [None] * len(R)
    for i, j in pairs:
        if i is not None and j is not None: lab[i] = H[j][1]
    runs: List[List[int]] = []
    for i, x in enumerate(R):
        if runs and R[runs[-1][-1]][1] == x[1]: runs[-1].append(i)
        else: runs.append([i])
    ov = lambda i: busy is not None and any(a - 1e-9 <= R[i][3] <= b + 1e-9 for o in range(len(busy)) if o != R[i][1] for a, b in busy[o])   # 다른 슬롯 화자 누구든
    cnt: Dict[str, int] = {}; late_words: List[int] = []; whole = {"whole_short": 0, "whole_long": 0}; n_short = 0; whole_n = [0] * len(runs)
    for ri, run in enumerate(runs):
        short = len({R[i][2] for i in run}) <= short_words; n_short += short
        al = [i for i in run if lab[i] is not None]; mis = [lab[i] != R[i][1] for i in al]; k = 0
        while k < len(al):
            if not mis[k]: k += 1; continue
            e = k
            while e + 1 < len(al) and mis[e + 1]: e += 1
            first, last = k == 0, e == len(al) - 1
            typ = ("whole_short" if short else "whole_long") if first and last else "late" if first else "early" if last else "inner"
            if typ in whole: whole[typ] += 1; whole_n[ri] = e - k + 1
            if typ == "late": late_words.append(len({R[al[x]][2] for x in range(k, e + 1)}))
            for x in range(k, e + 1):
                key = f"{typ}/{'ov' if ov(al[x]) else 'solo'}"; cnt[key] = cnt.get(key, 0) + 1
            k = e + 1
    iso = streak = streak_edge = 0; ri = 0                               # 통째 오배정 run 이 홀로(앞뒤 run 은 맞음 = 턴 하나를 삼킴)인지, 이어지는지(여러 턴 정체가 뒤바뀜)
    while ri < len(runs):
        if not whole_n[ri]: ri += 1; continue
        e = ri
        while e + 1 < len(runs) and whole_n[e + 1]: e += 1
        if e == ri: iso += whole_n[ri]
        else:
            streak += sum(whole_n[ri:e + 1])
            if ri == 0 or e == len(runs) - 1: streak_edge += sum(whole_n[ri:e + 1])     # 창 끝까지(또는 처음부터) 이어짐 = 되돌아오지 않음
        ri = e + 1
    return dict(n_ref=len(R), agn=sum(1 for i, j in pairs if i is None or j is None or R[i][0] != H[j][0]), n_pair=sum(x is not None for x in lab),
                n_mis=sum(cnt.values()), units=cnt, runs=len(runs), runs_short=n_short, whole_short_runs=whole["whole_short"], whole_long_runs=whole["whole_long"],
                whole_iso_units=iso, whole_streak_units=streak, whole_streak_edge_units=streak_edge, ref_changes=max(0, len(runs) - 1), hyp_changes=sum(H[j][1] != H[j - 1][1] for j in range(1, len(H))), late_words=late_words)

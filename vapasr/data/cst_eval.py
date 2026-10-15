"""Stage 3 평가 도우미(experiments/cst_stage3_eval.py) — 두 슬롯 출력 해석, 화자 귀속 오류율, 턴 끝 <EOT> 일치. 순수 함수(모델·tokenizer 없이 테스트).

출력 해석(parse_emitted): 스트리밍 [(k, tid)] — <SPK_A>/<SPK_B> 태그가 슬롯을 열고, 그 뒤 토큰·<EOT> 는 그 슬롯 것이다. 첫 태그 앞 토큰은 슬롯 A 로 본다
  (태그를 빠뜨려도 받아쓰기 평가는 되게). 방출 시각 = 청크 끝 (k+1)·80 ms (ast_eval.word_emission_times 와 같은 규약).
단위(units): 한국어 = 글자(공백 없는 CER, score_ko), 영어 = 단어(score_en), 독일어 = 단어(score_de).
창 점수(score_window):
  slot  : 슬롯을 그대로(A↔A, B↔B) 맞춘 오류 합 — 형식 규칙(먼저 말한 사람 = A)까지 지킨 화자 귀속
  cp    : 두 슬롯 순열 중 작은 쪽(cpWER/cpCER, 2 화자) — 화자 귀속
  agn   : 화자 무시 — 두 슬롯 단어를 시각순으로 이어 붙여 비교(겹침 구간 순서 차이가 조금 섞인다)
  swap  : cp 가 순열을 바꾼 창(슬롯 A/B 를 뒤집어 낸 창)
턴 끝(match_eot): 참조 = 창 안 episode 끝(슬롯·끝 시각·p_end). 양성 = p ≥ 0.8(교대·침묵), 보류 = p ≤ 0.3(본인 재개), 0.5(both)는 정밀도에서만 맞음으로 친다.
  가설 <EOT>(슬롯·방출 시각)는 cp 순열로 참조 슬롯에 옮긴 뒤 같은 슬롯의 참조 끝 e 와 e ≤ t ≤ e + tol 이면 짝(시간순 탐욕 1:1).
  recall = 짝지은 양성 / 양성, precision = (p ≥ 0.5 참조와 짝지은 가설) / 가설, hold_fire = 보류 참조와 짝지은 가설 / 보류 참조, 지연 = t − e(양성 짝).
"""
from typing import Dict, List, Optional, Sequence, Tuple

CHUNK_S = 0.08


def parse_emitted(emitted: Sequence, spk_ids: Sequence[int], eot_id: int, drop_chunks=()) -> Tuple[List[List[Tuple[float, int]]], List[Tuple[int, float]]]:
    """→ (슬롯별 [(방출 시각 s, tid)], [(슬롯, 방출 시각 s)] <EOT>). drop_chunks 의 청크에서 나온 토큰·EOT 는 버린다(전사 없는 음성 구간)."""
    slots: List[List[Tuple[float, int]]] = [[], []]; eots: List[Tuple[int, float]] = []; cur = 0; drop = set(drop_chunks)
    for k, t in emitted:
        t = int(t)
        if t in spk_ids: cur = list(spk_ids).index(t); continue
        if k in drop: continue
        tt = (int(k) + 1) * CHUNK_S
        if t == eot_id: eots.append((cur, tt))
        else: slots[cur].append((tt, t))
    return slots, eots


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


def units(text: str, lang: str) -> List[str]:
    if lang == "Korean":
        from .textnorm import score_ko
        return list(score_ko(text, False))
    if lang == "German":
        from .textnorm_de import score_de
        return score_de(text).split()
    from .textnorm import score_en
    return score_en(text).split()


def ed(a: Sequence, b: Sequence) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def score_window(ref: Sequence[Sequence[Tuple[str, float]]], hyp: Sequence[Sequence[Tuple[str, float]]], lang: str) -> Dict:
    """ref·hyp: 슬롯별 [(단어 문자열, 시각 s)]. → dict(n_ref, slot, cp, agn, swap, perm)."""
    U = lambda ws: [u for w, _ in ws for u in units(w, lang)]
    rA, rB, hA, hB = U(ref[0]), U(ref[1]), U(hyp[0]), U(hyp[1])
    same = ed(rA, hA) + ed(rB, hB); cross = ed(rA, hB) + ed(rB, hA)
    ra = [u for w, _ in sorted(list(ref[0]) + list(ref[1]), key=lambda x: x[1]) for u in units(w, lang)]
    ha = [u for w, _ in sorted(list(hyp[0]) + list(hyp[1]), key=lambda x: x[1]) for u in units(w, lang)]
    swap = cross < same
    return dict(n_ref=len(rA) + len(rB), n_hyp=len(hA) + len(hB), slot=same, cp=min(same, cross), agn=ed(ra, ha), swap=bool(swap), perm=[1, 0] if swap else [0, 1])


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

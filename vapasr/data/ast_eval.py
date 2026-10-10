"""ast-v1 평가 도우미(experiments/eval_single_turn_asr.py): 출력 토큰열을 전사·번역 구간으로 나누고, 번역의 품질(BLEU·chrF)과
지연(LAAL·AL, 초)을 잰다. 출력 형식은 vapasr/data/ast_dataset.py — 전사 구간은 <SPK_x>, 번역 구간은 <TRANS> 가 연다.

지연: 번역 단어(공백 분리)의 방출 시각 = 그 단어의 마지막 바이트가 나온 토큰의 청크 끝 시각 (k+1)·80 ms(스트리밍 디코드의 청크 k 는
오디오 k 를 받은 뒤 낸다). LAAL(Papi 외 2022, SimulEval 의 시간 기준·계산 시간 제외) = (1/τ) Σ_{i≤τ} (d_i − (i−1)·|X| / max(|Y|, |Y*|)),
τ = 원문이 끝난(d_i ≥ |X|) 첫 단어(없으면 |Y|), |X| = 원문 오디오 길이(꼬리 무음 제외). AL 은 분모가 |Y| 다.
품질: sacreBLEU — 목표가 한국어면 BLEU 토큰화 'char'(mecab 없음), 그 밖은 '13a'. chrF2 는 공통이며 한국어의 주 지표로 본다."""
import bisect
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

CHUNK_S = 0.08


def split_runs(emitted: Sequence, spk_ids: Iterable[int], trans_id: Optional[int]) -> Tuple[list, list]:
    """emitted: 스트리밍 [(k, tid)] 또는 final [tid] → (전사, 번역) 각각 [(k 또는 None, tid)]. 태그 토큰은 버린다.
    첫 태그 앞의 토큰은 전사로 본다(태그를 빠뜨린 출력도 받아쓰기 평가는 되게)."""
    spk = set(int(s) for s in spk_ids); src, tr, run = [], [], 0
    for e in emitted:
        k, t = (e[0], int(e[1])) if isinstance(e, (tuple, list)) else (None, int(e))
        if t in spk: run = 0; continue
        if trans_id is not None and t == int(trans_id): run = 1; continue
        (tr if run else src).append((k, t))
    return src, tr


def token_bytes(tok, ids: Sequence[int]) -> List[bytes]:
    """byte-level BPE 토큰 → 원 바이트(특수·표 밖 토큰은 b'')."""
    from .semcommit_words import piece_bytes, _pieces
    return [piece_bytes(p) or b"" for p in _pieces(tok, list(ids))]


def word_emission_times(tok, tr: Sequence[Tuple[int, int]]) -> Tuple[str, List[float]]:
    """번역 토큰 [(k, tid)] → (텍스트, 공백 분리 단어마다 방출 시각 s). 단어 끝 바이트를 처음 덮는 토큰의 (k+1)·80 ms."""
    ids = [t for _, t in tr]
    bs = token_bytes(tok, ids); raw = b"".join(bs)
    cum, n = [], 0
    for b in bs:
        n += len(b); cum.append(n)
    times = []
    for m in re.finditer(rb"\S+", raw):
        j = min(bisect.bisect_left(cum, m.end()), len(tr) - 1)
        times.append((tr[j][0] + 1) * CHUNK_S)
    return raw.decode("utf-8", errors="replace").strip(), times


def laal(times: Sequence[float], src_s: float, n_ref: int) -> Optional[float]:
    """Length-adaptive average lagging(초). times = 번역 단어 방출 시각, n_ref = 참조 단어 수."""
    Y = len(times)
    if not Y or src_s <= 0: return None
    L = max(Y, int(n_ref)); tau = next((i + 1 for i, d in enumerate(times) if d >= src_s), Y)
    return sum(times[i] - i * src_s / L for i in range(tau)) / tau


def al(times: Sequence[float], src_s: float) -> Optional[float]:
    return laal(times, src_s, 0)


def corpus_scores(hyps: List[str], refs: List[str], tgt_lang: str) -> Dict:
    """sacreBLEU 코퍼스 BLEU·chrF2(+ 서명)."""
    from sacrebleu.metrics import BLEU, CHRF
    tokenize = "char" if tgt_lang == "Korean" else "13a"
    bleu, chrf = BLEU(tokenize=tokenize), CHRF()
    b, c = bleu.corpus_score(hyps, [refs]), chrf.corpus_score(hyps, [refs])
    return dict(bleu=round(b.score, 2), chrf=round(c.score, 2), n=len(hyps), bleu_signature=str(bleu.get_signature()),
                chrf_signature=str(chrf.get_signature()))


def translation_report(rows: List[dict]) -> Dict[str, dict]:
    """예측 행(translation_reference 가 있는 것) → {'<dataset>/delta-<d>': 품질·지연·빈 번역 수}."""
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        if r.get("translation_reference") is not None:
            groups.setdefault(f"{r['dataset']}/delta-{r['delta']}", []).append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        rep = corpus_scores([r["translation_hypothesis"] for r in rs], [r["translation_reference"] for r in rs], rs[0]["tgt_lang"])
        lat = [r for r in rs if r.get("translation_laal") is not None]
        if lat:
            import statistics
            mean = lambda k: round(sum(r[k] for r in lat) / len(lat), 3)
            med = lambda k: round(statistics.median(r[k] for r in lat), 3)                  # 이상치(음성·문장 불일치 발화 등)에 강한 대표값
            rep.update(laal_s=mean("translation_laal"), laal_median_s=med("translation_laal"), al_s=mean("translation_al"),
                       al_median_s=med("translation_al"), pre_end=round(sum(r["translation_pre_end"] for r in lat) / len(lat), 3), latency_n=len(lat))
        rep["empty_translations"] = sum(not r["translation_hypothesis"] for r in rs)
        out[key] = rep
    return out

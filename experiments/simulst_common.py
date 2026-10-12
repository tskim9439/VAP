"""공개 스트리밍 음성번역 베이스라인 공통부(2026-10-11) — q06-s2-ast 평가(experiments/eval_single_turn_asr.py, vapasr/data/ast_eval.py)와
같은 평가셋·같은 입력·같은 지연 정의로 SeamlessStreaming·IWSLT26 연결형·Hibiki-Zero·SimulStreaming 을 잰다.

- 입력: 평가 manifest 행(번역 행: tgt_lang·translation_reference) 의 음성 + 꼬리 무음 tail_s(기본 1 s, 우리 모델과 같다).
- 지연: 번역 단어(공백 분리)의 방출 시각 = 그 단어의 마지막 글자를 낸 시점까지 시스템에 넣은 오디오 길이(s, 계산 시간 제외).
  LAAL·AL 은 vapasr.data.ast_eval.laal/al(|X| = 원 발화 길이, 꼬리 제외)로 낸다.
- 예측 행(predictions-*.jsonl): system·setting·id·dataset·lang·tgt_lang·audio_s·translation_reference·translation_hypothesis·
  translation_word_times(+ 시스템별 raw). 실행기(simulst_<system>.py)는 각자의 env 에서 돌고, 이 모듈의 행 함수는 표준 라이브러리만 쓴다.
- 채점(score): vapasr env 에서 (system, setting, dataset) 마다 BLEU·chrF·LAAL·AL·빈 번역 수.

usage(vapasr env): python experiments/simulst_common.py score --pred-dir DIR [DIR ...] --out summary.json
"""
import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

TAIL_S = 1.0
SR = 16000
LANG3 = {"English": "eng", "German": "deu", "Korean": "kor"}


def load_rows(manifests: Sequence[str], datasets: Optional[Sequence[str]] = None) -> List[dict]:
    """번역 평가 행(tgt_lang 이 있는 행)만, manifest 순서대로. datasets 가 있으면 그 dataset 만."""
    rows = []
    for m in manifests:
        for line in open(m, encoding="utf-8"):
            r = json.loads(line)
            if r.get("tgt_lang") and (not datasets or r["dataset"] in datasets):
                rows.append(r)
    return rows


def shard(rows: List[dict], rank: int, world: int) -> List[dict]:
    return [r for i, r in enumerate(rows) if i % world == rank]


def words_with_times(increments: Iterable[Tuple[str, float]]) -> Tuple[str, List[float]]:
    """[(조각 텍스트, 방출 시각 s)] → (공백 정리한 텍스트, 공백 분리 단어마다 시각). 단어 시각 = 그 단어의 마지막 글자를 담은 조각의 시각."""
    text, owner = "", []
    for frag, t in increments:
        if not frag: continue
        text += frag; owner += [float(t)] * len(frag)
    times = [owner[m.end() - 1] for m in re.finditer(r"\S+", text)]
    return " ".join(text.split()), times


def pred_row(system: str, setting: str, r: dict, text: str, times: List[float], tail_s: float = TAIL_S, **raw) -> dict:
    assert len(text.split()) == len(times), (len(text.split()), len(times))
    return dict(system=system, setting=setting, id=r["id"], dataset=r["dataset"], lang=r["lang"], tgt_lang=r["tgt_lang"],
                audio_s=float(r["audio_s"]), tail_s=tail_s, translation_reference=r["translation_reference"],
                translation_hypothesis=text, translation_word_times=[round(t, 4) for t in times], **({"raw": raw} if raw else {}))


def done_ids(path: str) -> set:
    return {json.loads(l)["id"] for l in open(path, encoding="utf-8")} if os.path.exists(path) else set()


def read_wav_with_tail(path: str, tail_s: float = TAIL_S):
    """16 kHz mono float32 + 꼬리 무음(우리 eval_single_turn_asr.read_audio 와 같은 입력)."""
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == SR, (path, sr)
    x = x.mean(1)
    return np.concatenate([x, np.zeros(int(round(tail_s * SR)), np.float32)]).astype(np.float32)


# ── 채점(vapasr env)
def score(pred_dirs: Sequence[str]) -> Dict[str, dict]:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import statistics
    from vapasr.data.ast_eval import al, corpus_scores, laal
    groups: Dict[Tuple[str, str, str], List[dict]] = {}
    for d in pred_dirs:
        for f in sorted(glob.glob(os.path.join(d, "predictions-*.jsonl"))):
            for line in open(f, encoding="utf-8"):
                r = json.loads(line)
                groups.setdefault((r["system"], r["setting"], r["dataset"]), []).append(r)
    out = {}
    for (system, setting, ds), rs in sorted(groups.items()):
        ids = [r["id"] for r in rs]
        assert len(ids) == len(set(ids)), f"중복 예측: {system}/{setting}/{ds}"
        rep = corpus_scores([r["translation_hypothesis"] for r in rs], [r["translation_reference"] for r in rs], rs[0]["tgt_lang"])
        lat = []
        for r in rs:
            t = r["translation_word_times"]
            if t:
                n_ref = len(r["translation_reference"].split())
                lat.append((laal(t, r["audio_s"], n_ref), al(t, r["audio_s"]), sum(x < r["audio_s"] for x in t) / len(t)))
        if lat:
            rep.update(laal_s=round(sum(x[0] for x in lat) / len(lat), 3), laal_median_s=round(statistics.median(x[0] for x in lat), 3),
                       al_s=round(sum(x[1] for x in lat) / len(lat), 3), al_median_s=round(statistics.median(x[1] for x in lat), 3),
                       pre_end=round(sum(x[2] for x in lat) / len(lat), 3), latency_n=len(lat))
        rep["empty_translations"] = sum(not r["translation_hypothesis"] for r in rs)
        out[f"{system}/{setting}/{ds}"] = rep
    return out


def from_ours(result_dirs: Sequence[str], manifests: Sequence[str], out_dir: str, system: str = "q06-s2-ast") -> int:
    """우리 모델의 평가 예측(eval_single_turn_asr.py run, predictions-rank*.jsonl)을 같은 행(manifests)만 골라 이 형식으로 옮긴다.
    setting = d4/d6/d8/d10(스트리밍, translation_word_times 그대로)·final(오프라인, 시각 없음)."""
    keep = {r["id"] for r in load_rows(manifests)}
    os.makedirs(out_dir, exist_ok=True); n = 0
    with open(os.path.join(out_dir, f"predictions-{system}.jsonl"), "w", encoding="utf-8") as f:
        for d in result_dirs:
            for p in sorted(glob.glob(os.path.join(d, "predictions-rank*.jsonl"))):
                for line in open(p, encoding="utf-8"):
                    r = json.loads(line)
                    if r["id"] not in keep or r.get("translation_reference") is None: continue
                    times = r.get("translation_word_times") or []
                    text = r["translation_hypothesis"]
                    if times and len(text.split()) != len(times): raise ValueError(f"단어 수와 시각 수가 다르다: {r['id']}")
                    f.write(json.dumps(dict(system=system, setting=("final" if r["delta"] == "final" else f"d{r['delta']}"), id=r["id"],
                                            dataset=r["dataset"], lang=r["lang"], tgt_lang=r["tgt_lang"], audio_s=float(r["audio_s"]),
                                            tail_s=float(r.get("tail_s", TAIL_S)), translation_reference=r["translation_reference"],
                                            translation_hypothesis=text, translation_word_times=times), ensure_ascii=False) + "\n"); n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("from-ours")
    q.add_argument("--results", nargs="+", required=True)
    q.add_argument("--manifest", nargs="+", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--system", default="q06-s2-ast")
    q = sub.add_parser("score")
    q.add_argument("--pred-dir", nargs="+", required=True)
    q.add_argument("--out", required=True)
    a = p.parse_args()
    if a.cmd == "from-ours":
        print(f"{from_ours(a.results, a.manifest, a.out, a.system)} 행 → {a.out}"); return
    rep = score(a.pred_dir)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    for k, v in rep.items():
        print(f"{k:58s} n={v['n']:5d} BLEU {v['bleu']:6.2f} chrF {v['chrf']:6.2f} empty={v['empty_translations']:4d}"
              + (f" LAAL {v['laal_s']:.2f}/{v['laal_median_s']:.2f} AL {v['al_s']:.2f}/{v['al_median_s']:.2f} pre_end {v['pre_end']:.2f}" if "laal_s" in v else ""))


if __name__ == "__main__":
    main()

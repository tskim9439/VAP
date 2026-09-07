"""학습 시퀀스를 시간(80 ms 청크) 단위로 표로 표현 — 청크마다 입력 / 라벨(출력) / 정렬 근거 시각 / 방출 지연.
python experiments/s1_show_sequence.py [--manifest kspon-dev --mode utt --delay 2 --dur 2.6] [--show-ms 4000]
온라인 모드 데이터셋(MonoStreamDataset)을 그대로 써서 실제 학습 시퀀스와 동일하다. 청크당 토큰은 BPE 조각을 합쳐 표시하고 조각 수를 괄호에 적는다.
무방출 청크(라벨이 <NEXT_AUDIO> 뿐)는 구간으로 묶어 한 줄로 줄인다."""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from transformers import AutoTokenizer
from vapasr.uslm.mono_data import MonoStreamDataset

def render(tok, name, mode, delay, pick, show_ms=None):
    ds = MonoStreamDataset([name], tok, mode=mode, delays=(delay,), online=True, qc=True)
    i = pick(ds); f, ids, is_input, chunk_of, st, it, n_rounds, n_flush = ds.sequence(i, delay, audio=False); sp = ds.sp
    K = it["K"]; ref = list(it["tokens"]); ref_i = 0
    per = {}; cur = None; flush_rounds = []                          # k → dict(tokens=[(text, end_time)], next=bool)
    for t, inp, k in zip(ids, is_input, chunk_of):
        if k >= 0: cur = k; per.setdefault(k, dict(tokens=[], next=False)); continue
        if t == sp.empty_audio: cur = ("flush", len(flush_rounds)); flush_rounds.append(dict(tokens=[], next=False)); continue
        if inp: continue                                            # prefix
        slot = per[cur] if isinstance(cur, int) else flush_rounds[cur[1]]
        if t == sp.next_audio: slot["next"] = True
        else:
            e = ref[ref_i][1] if ref_i < len(ref) and ref[ref_i][0] == t else None; ref_i += 1 if e is not None else 0
            slot["tokens"].append((tok.decode([t]), e))
    prefix = tok.decode([t for t, inp, k in zip(ids, is_input, chunk_of) if inp and k < 0 and t != sp.empty_audio]).replace("\n", "⏎")
    print(f"\n### {name} · {it['id']} · {it['duration_s']} s · K={K} 청크(80 ms) · δ={delay}({delay*80} ms) · M={ds.M} · 텍스트 토큰 {len(ref)} · 시퀀스 길이 {len(ids)}")
    print(f"lexical_text: `{it['text']}`")
    print(f"\nprefix(입력만, 라벨 없음): `{prefix}`\n")
    print("| 청크 k | 시간창 (ms) | 입력 | 라벨(모델 출력) | 정렬 근거: 토큰 종료 시각 (ms) | 방출 지연 (ms) |\n|---:|---|---|---|---|---|")
    rows = []; empty_run = []
    def flush_empty():
        if empty_run:
            a, b = empty_run[0], empty_run[-1]; n = b - a + 1
            rows.append(f"| {a}–{b} | {a*80}–{(b+1)*80} | `[AUDIO_{a}]`{' … `[AUDIO_%d]`' % b if n > 1 else ''} | `<NEXT>` × {n} (무방출) | – | – |"); empty_run.clear()
    for k in range(K):
        s = per.get(k, dict(tokens=[], next=False)); win = f"{k*80}–{(k+1)*80}"
        if show_ms is not None and k * 80 >= show_ms and k < K - 3: continue
        if not s["tokens"]: empty_run.append(k); continue
        flush_empty()
        pieces = [tx for tx, _ in s["tokens"]]; joined = "".join(pieces).strip()
        toks = f"`{joined}`" + (f" ({len(pieces)} 조각)" if len(pieces) > 1 else "") + (" `<NEXT>`" if s["next"] else "")
        ev = ", ".join(f"{int(e*1000)}" if e is not None else "?" for _, e in s["tokens"])
        lat = ", ".join(f"+{int((k+1)*80 - e*1000)}" if e is not None else "?" for _, e in s["tokens"])
        rows.append(f"| {k} | {win} | `[AUDIO_{k}]` | {toks} | {ev} | {lat} |")
    flush_empty()
    for j, fr in enumerate(flush_rounds):
        pieces = [tx for tx, _ in fr["tokens"]]; toks = (f"`{''.join(pieces).strip()}` " if pieces else "") + "`<NEXT>`"
        rows.append(f"| flush {j} | 스트림 끝 이후 | `<EMPTY_AUDIO>` | {toks} | {'이월 토큰' if pieces else '남은 것 없음 → 종료'} | – |")
    print("\n".join(rows))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default=None); ap.add_argument("--mode", default="utt"); ap.add_argument("--delay", type=int, default=2)
    ap.add_argument("--dur", type=float, default=2.6, help="이 길이(초)에 가장 가까운 항목을 고른다"); ap.add_argument("--show-ms", type=int, default=None, help="앞 N ms 와 끝부분만 표시"); a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B"))
    if a.manifest: render(tok, a.manifest, a.mode, a.delay, lambda ds: min(range(len(ds)), key=lambda i: (abs(ds.items[i]["duration_s"] - a.dur), i)), show_ms=a.show_ms)
    else:
        render(tok, "kspon-dev", "utt", 2, lambda ds: min(range(len(ds)), key=lambda i: (abs(ds.items[i]["duration_s"] - 2.6), i)))
        render(tok, "librispeech-dev", "stream", 2, lambda ds: min(range(len(ds)), key=lambda i: (abs(ds.items[i]["duration_s"] - 20.5), i)), show_ms=4000)

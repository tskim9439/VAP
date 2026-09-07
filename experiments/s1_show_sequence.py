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
    print(f"\n### {name} · {it['id']} · {it['duration_s']} s · K={K} 청크(80 ms) · δ={delay}({delay*80} ms) · 청크당 토큰 상한 {'없음' if not ds.M else ds.M} · 텍스트 토큰 {len(ref)} · 시퀀스 길이 {len(ids)}")
    print(f"lexical_text: `{it['text']}`")
    print(f"\nprefix(입력만, 라벨 없음): `{prefix}`\n")
    # 디코더 전용 LM 이므로 각 위치의 입력 토큰이 다음 토큰을 예측한다: 오디오 자리 → 첫 텍스트, 텍스트 → 다음 텍스트, 마지막 텍스트 → <NEXT>,
    # <NEXT> 위치의 예측은 다음 오디오 자리(런타임이 넣는 입력)라 손실 없음. 학습은 teacher forcing(정답 텍스트가 다음 입력), 추론은 모델 출력이 다음 입력.
    print("| 청크 k | 시간창 (ms) | 위치별 `입력 → 예측(라벨)` | 정렬 근거: 토큰 종료 시각 (ms) | 방출 지연 (ms) |\n|---:|---|---|---|---|")
    rows = []; empty_run = []
    def pairs(first_in, pieces):
        joined = "".join(pieces).strip(); n = len(pieces)           # 한글 byte-level BPE 조각은 단독 decode 가 깨지므로 '단어⟨i/n⟩' 로 표시
        show = [(f"{joined}⟨{i+1}/{n}⟩" if ("\ufffd" in p or n > 1 and not p.strip()) else (p.strip() or "␣")) for i, p in enumerate(pieces)]
        seq = [first_in] + [f"`{x}`" for x in show]; tg = seq[1:] + ["`<NEXT>`"]
        return " · ".join(f"{a}→{b}" for a, b in zip(seq, tg)) + " · `<NEXT>`→(다음 오디오, 손실 없음)"
    def flush_empty():
        if empty_run:
            a, b = empty_run[0], empty_run[-1]; n = b - a + 1
            rows.append(f"| {a}–{b} | {a*80}–{(b+1)*80} | `[AUDIO_k]`→`<NEXT>` × {n} (무방출) | – | – |"); empty_run.clear()
    for k in range(K):
        s = per.get(k, dict(tokens=[], next=False)); win = f"{k*80}–{(k+1)*80}"
        if show_ms is not None and k * 80 >= show_ms and k < K - 3: continue
        if not s["tokens"]: empty_run.append(k); continue
        flush_empty()
        pieces = [tx for tx, _ in s["tokens"]]
        ev = ", ".join(f"{int(e*1000)}" if e is not None else "?" for _, e in s["tokens"])
        lat = ", ".join(f"+{int((k+1)*80 - e*1000)}" if e is not None else "?" for _, e in s["tokens"])
        rows.append(f"| {k} | {win} | {pairs(f'`[AUDIO_{k}]`', pieces)} | {ev} | {lat} |")
    flush_empty()
    for j, fr in enumerate(flush_rounds):
        pieces = [tx for tx, _ in fr["tokens"]]
        rows.append(f"| flush {j} | 스트림 끝 이후 | {pairs('`<EMPTY_AUDIO>`', pieces)} | {'이월 토큰' if pieces else '남은 것 없음 → 종료'} | – |")
    print("\n".join(rows))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default=None); ap.add_argument("--mode", default="utt"); ap.add_argument("--delay", type=int, default=2)
    ap.add_argument("--dur", type=float, default=2.6, help="이 길이(초)에 가장 가까운 항목을 고른다"); ap.add_argument("--show-ms", type=int, default=None, help="앞 N ms 와 끝부분만 표시"); a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(os.environ.get("MXC_QWEN_ASR_DIR", "Qwen/Qwen3-ASR-0.6B"))
    if a.manifest: render(tok, a.manifest, a.mode, a.delay, lambda ds: min(range(len(ds)), key=lambda i: (abs(ds.items[i]["duration_s"] - a.dur), i)), show_ms=a.show_ms)
    else:
        render(tok, "kspon-dev", "utt", 2, lambda ds: min(range(len(ds)), key=lambda i: (abs(ds.items[i]["duration_s"] - 2.6), i)))
        render(tok, "librispeech-dev", "stream", 2, lambda ds: min(range(len(ds)), key=lambda i: (abs(ds.items[i]["duration_s"] - 20.5), i)), show_ms=4000)

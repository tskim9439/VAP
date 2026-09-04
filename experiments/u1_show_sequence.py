#!/usr/bin/env python
"""U1 학습 시퀀스 실물 덤프 — 실제 창 하나를 토큰 단위로 보여준다.
python experiments/u1_show_sequence.py [--manifest otoSpeech] [--index 0] [--delay 2] [--chunks 40]
"""
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--manifest", default="otoSpeech"); ap.add_argument("--index", type=int, default=0)
ap.add_argument("--delay", type=int, default=2); ap.add_argument("--chunks", type=int, default=40); ap.add_argument("--from-chunk", type=int, default=0); ap.add_argument("--only-emits", action="store_true"); ap.add_argument("--split", default="train"); a = ap.parse_args()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-ASR-0.6B")
from vapasr.uslm.interleave_data import InterleavedWindowDataset, build_sequence, CHUNK_S
from vapasr.uslm.data import LANG
ds = InterleavedWindowDataset([a.manifest], tok, split=a.split, delays=(a.delay,), windows_per_conv=2, max_windows=200, seed=1)
name, cid, t0 = ds.items[a.index]; lang = LANG[name]
feats, streams, chunks, st = ds.window(name, cid, t0, a.delay)
pre = ds.prefix(lang, a.delay); ids, is_audio, chunk_of = build_sequence(chunks, ds.K, pre, ds.audio_pad)
inv = {v: k for k, v in ds.sp_ids.items()}
def show(t):
    if t in inv: return inv[t]
    if t == ds.audio_pad: return "<AUDIO>"
    return repr(tok.decode([t]))
print(f"창: {name} / {cid} / t0={t0:.1f}s / 길이 {ds.window_s:.0f}s = {ds.K} chunk(80 ms) / lang={lang} / δ={a.delay} frame({a.delay*80} ms) / M={ds.M}")
print(f"특징: {feats.shape} (화자 2 × {ds.K} 프레임 × 1024) — Nemotron [56,0] 12.5 Hz, 1 프레임 = 1 chunk")
print(f"토큰: 화자A {len(streams[0])} + 화자B {len(streams[1])} = {st.tokens} 방출, 이월 {st.overflow_tokens}, 시퀀스 길이 {len(ids)}\n")
print("--- prefix (손실 제외) ---")
print("  " + " ".join(show(t) for t in pre) + "\n")
print(f"--- 본문: chunk {a.from_chunk}–{a.from_chunk + a.chunks} ---")
i = len(pre); k_shown = 0
while i < len(ids) and k_shown < a.chunks:
    k = chunk_of[i]; assert is_audio[i]
    j = i + 1; emits = []
    while j < len(ids) and not is_audio[j]: emits.append(ids[j]); j += 1
    if k >= a.from_chunk and not (a.only_emits and len(emits) == 1):
        txt = " ".join(show(t) for t in emits)
        star = "" if len(emits) == 1 else "   ← 방출"
        print(f"  k={k:3d} t={(k+1)*CHUNK_S:6.2f}s | <AUDIO_{k}> {txt}{star}")
        k_shown += 1
    i = j
print("\n--- 라벨(손실) 규칙 ---")
n_lab = sum(1 for t, au in zip(ids, is_audio) if not au) - len(pre)
print(f"  audio 위치({sum(is_audio)}개)와 prefix({len(pre)}개)는 -100. 나머지 {n_lab}개(텍스트 + <SPK_x> + <NEXT_AUDIO>)가 next-token 예측 대상.")
print(f"  → 학습 신호의 {100*sum(1 for t,au in zip(ids,is_audio) if not au and t==ds.sp_ids['<NEXT_AUDIO>'])/max(1,n_lab):.0f} % 가 <NEXT_AUDIO>(무방출 결정), 나머지가 텍스트.")
print("\n--- 정렬 원본(참고: 이 창의 발화) ---")
for u in [u for u in ds.convs[(name, cid)].utts if u["start"] < t0 + ds.window_s and u["end"] > t0][:6]:
    print(f"  spk{u['speaker']} {u['start']-t0:6.2f}–{u['end']-t0:6.2f}s  {u['text'][:60]}")

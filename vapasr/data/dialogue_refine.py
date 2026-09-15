"""라벨 시각 보정 — 화자 채널의 에너지 VAD 와 정렬 토큰 시각으로 발화 경계를 다시 잡는다 (정본 §3.2 "화자별 VAD, gap<0.25 s 병합").

동기: AI Hub 71631 원본 라벨은 발화가 길고(중앙값 7 s, p90 20 s) 끝이 실제 마지막 단어보다 0.9 s(p90 1.25 s) 늦다(2026-09-15 샘플 실측).
라벨 span 을 그대로 쓰면 ONSET·EOT 후보·활동 타깃이 실제 음성과 어긋나고, 긴 발화 안의 pause 가 episode 경계로 잡히지 않는다.

refine_utterances(dlg, vad, gap_s=0.25, pad_s=0.1):
  화자별 VAD 구간(gap_s 미만 병합)과 각 발화의 [start, end] 를 교차해 (1) 발화 끝 = min(라벨 끝, max(VAD 끝, 마지막 토큰 끝)+pad) (2) 발화 시작 = max(라벨 시작, VAD 시작−pad)
  (3) 발화 안에 gap_s 이상 VAD 공백이 있고 양쪽에 토큰이 있으면 그 자리에서 분할. 토큰은 시각으로 배분하고 텍스트는 원문 유지(분할된 조각은 text="" 대신 tokens 만 보유 → 학습은 tokens 만 쓴다).
  VAD 가 발화 안에서 전혀 켜지지 않으면 토큰 시각으로만 끝을 당긴다. 결과는 새 Utterance 목록(원 utt_id 에 "#n" 접미).
channel_vad(dlg, sr): 화자별 채널을 읽어 energy_vad → [(s, e)] (streams.load_utt_audio 규약, 조각은 offset 배치).
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from .dialogue import Dialogue, Utterance, GAP_S
from .vad import energy_vad, frames_to_segments
from .dialogue_mix import load_channel
from .streams import SR

def channel_vad(dlg: Dialogue, sr: int = SR, hop_ms: float = 20.0, gap_s: float = GAP_S) -> Dict[str, List[Tuple[float, float]]]:
    out = {}
    for spk in dlg.speakers:
        ref = dlg.channels.get(spk)
        if ref is None: out[spk] = []; continue
        x = load_channel(ref, dlg.duration_s, sr); segs = frames_to_segments(energy_vad(x, sr, hop_ms), hop_ms / 1000)
        merged = []
        for s, e in segs:
            if merged and s - merged[-1][1] < gap_s: merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else: merged.append((s, e))
        out[spk] = merged
    return out

def refine_utterances(dlg: Dialogue, vad: Dict[str, List[Tuple[float, float]]], gap_s: float = GAP_S, pad_s: float = 0.1, min_dur: float = 0.15) -> Tuple[List[Utterance], Dict]:
    out = []; st = dict(utts=0, split=0, tightened_s=0.0, no_vad=0)
    for u in dlg.utterances:
        st["utts"] += 1; toks = sorted(u.tokens or [], key=lambda t: t[1])
        inside = [(max(s, u.start), min(e, u.end)) for s, e in vad.get(u.speaker, []) if e > u.start and s < u.end]
        if not inside:
            st["no_vad"] += 1; end = min(u.end, toks[-1][1] + pad_s) if toks else u.end
            out.append(Utterance(u.speaker, u.start, max(end, u.start + min_dur), u.text, u.raw, toks or None, u.utt_id, u.word_timing, u.flags)); st["tightened_s"] += u.end - end; continue
        # VAD 공백(gap_s 이상)에서 분할 후보
        pieces = [[inside[0][0], inside[0][1]]]
        for s, e in inside[1:]:
            if s - pieces[-1][1] >= gap_s: pieces.append([s, e])
            else: pieces[-1][1] = max(pieces[-1][1], e)
        # 토큰을 조각에 배분(토큰 끝 시각 기준, 조각 사이 공백의 토큰은 앞 조각으로)
        bounds = [(p[0], (pieces[i + 1][0] if i + 1 < len(pieces) else u.end)) for i, p in enumerate(pieces)]
        groups = [[] for _ in pieces]
        for t in toks:
            j = next((i for i, (b0, b1) in enumerate(bounds) if t[1] < b1), len(pieces) - 1); groups[j].append(t)
        keep = [i for i in range(len(pieces)) if groups[i] or (len(pieces) == 1)]
        if len(keep) > 1: st["split"] += len(keep) - 1
        for n, i in enumerate(keep):
            s0 = max(u.start, pieces[i][0] - pad_s); last_tok = groups[i][-1][1] if groups[i] else pieces[i][1]
            e0 = min(u.end, max(pieces[i][1], last_tok) + pad_s); e0 = max(e0, s0 + min_dur)
            st["tightened_s"] += (u.end - u.start) - (e0 - s0) if len(keep) == 1 else 0.0
            out.append(Utterance(u.speaker, round(s0, 3), round(e0, 3), u.text if len(keep) == 1 else "", u.raw, groups[i] or None, u.utt_id if len(keep) == 1 else f"{u.utt_id}#{n}", u.word_timing if len(keep) == 1 else None, u.flags))
    out.sort(key=lambda x: (x.start, x.speaker)); return out, st

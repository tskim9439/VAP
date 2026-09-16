#!/usr/bin/env python
"""Phase 2 Q0 경량 QC — dialogues.jsonl(+정렬) → 창 통계·정렬 coverage·lane·EOT 결과 분포·밀도 (task-phase2-data-prep 순서 8, 정본 §13 Q0 관문).

  python experiments/p2_qc.py --dialogues <dir>/ami.dialogues.jsonl [--align <dir>/align-asr-tn-v1/ami] [--R 6] [--delay 4] [--limit N] --out <dir>/ami.qc.json
보고: dialogues·hours·speakers 분포, 발화 수·quarantine·미정렬·결손, 창 수·제외 사유, allocator(재배정·부족), EOT outcome 히스토그램,
      청크당 텍스트 토큰 p50/p99·overflow·태그/분, 동시 활동 분포(S), 화자별 시간. GPU 불필요."""
import os, sys, json, argparse, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from vapasr.data.dialogue import Dialogue, build_episodes, CHUNK_S
from vapasr.data.lane_alloc import allocate
from vapasr.data.eot_soft import assign_p_end
from vapasr.data.dialogue_interleave import serialize, lane_activity, LaneSpecials
from vapasr.data.dialogue_dataset import load_align_parts

ap = argparse.ArgumentParser(); ap.add_argument("--dialogues", required=True); ap.add_argument("--align"); ap.add_argument("--R", type=int, default=6); ap.add_argument("--delay", type=int, default=4)
ap.add_argument("--limit", type=int); ap.add_argument("--out"); a = ap.parse_args()
SP = LaneSpecials(next_audio=-1, empty_audio=-2, lanes=[-10 - i for i in range(a.R)], onset=-3, eot=-4)
aligned, failed = load_align_parts(a.align) if a.align else ({}, set())
st = collections.Counter(); hours = 0.0; spk_hist = collections.Counter(); outcomes = collections.Counter(); dens = []; simul = collections.Counter(); spk_time = 0.0; n = 0
for line in open(a.dialogues, encoding="utf-8"):
    d = Dialogue.from_json(line); n += 1
    if a.limit and n > a.limit: break
    toks = aligned.get(d.conv_id, {})
    for u in d.utterances:
        if u.utt_id in toks: u.tokens = toks[u.utt_id]
        st["utts"] += 1; spk_time += max(0.0, u.end - u.start)
        if u.flags: st["quarantined"] += 1
        elif u.tokens: st["aligned"] += 1                       # 보정으로 분할된 조각은 text="" 이지만 tokens 가 있어 정렬됨으로 센다
        elif not u.text: st["empty_text"] += 1
        elif a.align: st["unaligned"] += 1
    st["dialogues"] += 1; hours += d.duration_s / 3600; spk_hist[len(d.speakers)] += 1; st["missing_pieces"] += len(d.meta.get("missing", []))
    eps = build_episodes(d); al = allocate(eps, R=a.R, policy="lazy_free", delay_text=a.delay); st["episodes"] += al.episodes; st["reassigned"] += al.reassigned; st["exhausted"] += al.exhausted
    for k, v in assign_p_end(eps, d.observed()).items(): outcomes[k] += v
    K = int(np.ceil(d.duration_s / CHUNK_S)); act = np.array(lane_activity(eps, K, a.R)) if K else np.zeros((0, a.R))
    for s, c in zip(*np.unique(act.sum(1), return_counts=True)): simul[int(s)] += int(c)
    if a.align or any(u.tokens for u in d.utterances):
        chunks, ss = serialize(eps, d.duration_s, SP, delay_text=a.delay); dens += [k for k, v in ss.per_chunk_hist.items() for _ in range(v)]
        st["tokens"] += ss.text; st["tags"] += ss.tags; st["onset"] += ss.onset; st["eot"] += ss.eot; st["overflow"] += ss.overflow
tot = sum(simul.values()) or 1; speech = sum(v for k, v in simul.items() if k > 0) or 1
rep = dict(dialogues=st["dialogues"], hours=round(hours, 2), speaker_hours=round(spk_time / 3600, 2), speakers_per_dialogue=dict(sorted(spk_hist.items())),
           utts=st["utts"], quarantined=st["quarantined"], empty_text=st["empty_text"], aligned=st["aligned"], unaligned=st["unaligned"], align_failed=len(failed), missing_pieces=st["missing_pieces"],
           episodes=st["episodes"], lane_reassigned=st["reassigned"], lane_exhausted=st["exhausted"], eot_outcomes={k: v for k, v in sorted(outcomes.items())},
           simultaneous_share_of_speech={int(k): round(v / speech, 4) for k, v in sorted(simul.items()) if k > 0}, silence_share=round(simul.get(0, 0) / tot, 4),
           text_tokens=st["tokens"], tags=st["tags"], onset=st["onset"], eot=st["eot"], overflow=st["overflow"],
           tokens_per_chunk=dict(p50=float(np.percentile(dens, 50)) if dens else None, p99=float(np.percentile(dens, 99)) if dens else None, max=int(max(dens)) if dens else None),
           tags_per_min=round(st["tags"] / (hours * 60), 2) if hours and st["tags"] else None, R=a.R, delay=a.delay)
print(json.dumps(rep, ensure_ascii=False, indent=1))
if a.out: json.dump(rep, open(a.out, "w"), ensure_ascii=False, indent=1)

#!/usr/bin/env python
"""WindowDataset end-to-end 점검: 50 Hz / 12.5 Hz 창을 실제 오디오와 함께 로드해 shape·유효 라벨 비율·처리 시간을 본다."""
import sys, os, time, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.data import WindowDataset, collate
M = os.environ["DATA_MANIFEST_DIR"]
roots = dict(aihub="/data3/tskim/corpora/aihub/adult-vs02-wav", otoSpeech="/data3/tskim/corpora/turnbench/otoSpeech")
for fh in (50.0, 12.5):
    ds = WindowDataset([M + "/aihub-vs02", M + "/otoSpeech"], window_s=20.0, hop_s=10.0, frame_hz=fh, audio_roots=roots)
    t = time.time(); items = [ds[i] for i in range(6)]; b = collate(items); dt = (time.time() - t) / 6
    a, v, l = b["audio"], b["vad"], b["vap_label"]
    print(f"frame_hz={fh}: windows={len(ds)}  {dt*1000:.0f} ms/item | audio {tuple(a.shape)} vad {tuple(v.shape)} vap_label {tuple(l.shape)} "
          f"valid {(l >= 0).float().mean():.2f} tau_bin {tuple(b['tau_bin'].shape)} censored {b['censored'].float().mean():.2f} | "
          f"ids {[x[:10] for x in b['id'][:3]]} | events/win {sum(len(e) for e in b['events']) / len(b['events']):.1f}")
    print("   audio rms/ch:", [round(x, 4) for x in a.pow(2).mean(-1).sqrt().mean(0).tolist()], "| speech ratio/ch:", [round(x, 3) for x in v.mean((0, 1)).tolist()],
          "| top labels:", torch.bincount(l[l >= 0]).topk(3).indices.tolist())

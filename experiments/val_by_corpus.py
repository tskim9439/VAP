#!/usr/bin/env python
"""저장된 probe 로 코퍼스별(val 매니페스트별) CE/acc 를 따로 계산 — 언어/도메인 차이 진단."""
import os, sys, json, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vapasr.probe import ProbeWindowDataset, ProbeHead, collate_probe
run = sys.argv[1]; ck = torch.load(os.path.join(run, "probe.pt"), map_location="cuda")
FEAT = os.environ.get("DATA_FEATURE_CACHE_DIR", "/data3/tskim/features"); MAN = os.environ.get("DATA_MANIFEST_DIR", "/data3/tskim/manifests")
m = ProbeHead(ck["d_in"], ck["frame_hz"], ck["d_model"], ck["layers"]).cuda().eval(); m.load_state_dict(ck["state"])
for name in sys.argv[2:] or ["aihub-vs02", "otoSpeech", "turnbench-dev"]:
    ds = ProbeWindowDataset(FEAT, ck["encoder"], [(os.path.join(MAN, name), name)], 20.0, 40.0, ck["frame_hz"], max_windows=400)
    if not len(ds): print(name, "no windows"); continue
    dl = torch.utils.data.DataLoader(ds, 16, collate_fn=collate_probe, num_workers=4); ce = acc = n = 0.0; fstd = 0.0; k = 0
    with torch.inference_mode():
        for b in dl:
            x = b["feats"].cuda(); vl, _ = m(x); y = b["vap_label"].cuda(); mk = y >= 0
            ce += F.cross_entropy(vl[mk], y[mk], reduction="sum").item(); acc += (vl.argmax(-1)[mk] == y[mk]).float().sum().item(); n += mk.sum().item()
            fstd += x.float().std().item(); k += 1
    print(f"{ck['encoder']:16s} {name:14s} ce {ce/n:.3f} acc {acc/n:.3f} | feat std {fstd/k:.3f} | windows {len(ds)}")

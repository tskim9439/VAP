# Parakeet + 저장 확률(--diar-probs) 배선 점검(맥 CPU): 길이 다른 합성 회의 둘을 한 배치로, 화자 분리기 forward 가 불리면 실패.
import argparse, os, sys, types
import numpy as np
sys.path.insert(0, "/Users/taesookim/Desktop/VAPKT/experiments"); sys.path.insert(0, "/Users/taesookim/Desktop/VAPKT")
from baseline_ami_ext import MTParakeet
S = sys.argv[1]
a = argparse.Namespace(asr="/Volumes/Samsung_T5/vapkt-models/multitalker-parakeet-streaming-0.6b-v1/multitalker-parakeet-streaming-0.6b-v1.nemo",
                       diar="/Volumes/Samsung_T5/vapkt-models/Nemotron-3-Diarization/Nemotron-3-Diarization.nemo", diar_kind="n3d", att_context=[70, 13],
                       batch_size=2, device="cpu", wav_dir=os.path.join(S, "replay-check", "wav"), rank=0, diar_probs=os.path.join(S, "replay-check", "probs"))
rng = np.random.RandomState(0); units = []
for key, sec in (("fake:m1@0.0", 24.0), ("fake:m2@0.0", 15.0)):
    n = int(16000 * sec); t = np.arange(n) / 16000.0; x = (rng.randn(n) * 0.02).astype(np.float32)
    x[: n // 2] += 0.3 * np.sin(2 * np.pi * 190 * t[: n // 2]).astype(np.float32); x[n // 2:] += 0.3 * np.sin(2 * np.pi * 300 * t[n // 2:]).astype(np.float32)
    K = int(round(sec / 0.08)); p = np.zeros((K, 8), np.float16); p[: K // 2, 0] = 0.9; p[K // 2:, 1] = 0.9
    np.savez_compressed(os.path.join(a.diar_probs, key.rsplit("@", 1)[0] + "@full.npz"), prob=p); units.append((key, x))
m = MTParakeet(a)
def boom(*args, **kw): raise RuntimeError("화자 분리기가 불렸다 — 저장 확률 경로가 아니다")
m.diar.forward_streaming_step = boom
out = m(units)
for k, (hw, info) in out.items(): print(k, "단어", len(hw), "화자", sorted({s for s, _, _ in hw}), "decode_s", info["decode_s"])
print("info", {k: v for k, v in m.info.items() if k in ("diar_kind", "diar_probs", "chunk_len")}); print("REPLAY_OK")

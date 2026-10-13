# 독일어 정제 대화(kec·ztc·nsc)를 학습 로더 경로(DialogueWindowDataset → mono 캐시 → crop → 시퀀스)로 표본 점검. 읽기 전용(mono 캐시는 이미 만든 것을 mmap).
import sys, os, json, random, time, numpy as np
sys.path.insert(0, "/soundai/users/tskim/VAPKT")
from transformers import AutoTokenizer
from vapasr.data.dialogue_dataset import DialogueWindowDataset
D = "/soundai/users/tskim/VAPKT-data/data/phase2"; tok = AutoTokenizer.from_pretrained("/soundai/Model/Qwen3-ASR-0.6B")
for c in sys.argv[1].split(","):
    p = f"{D}/{c}.refined.dialogues.jsonl"; t0 = time.time()
    ds = DialogueWindowDataset([p], tok, delays=(4,), mono_cache_dir=f"{D}/mono"); print(f"== {c}: {len(ds)} windows, stats {ds.stats} ({time.time()-t0:.0f}s)")
    # 정렬 시각이 발화 구간 안에 있는가(전 발화)
    n = ok = 0; lead = []
    for d in ds.dlgs.values():
        for u in d.utterances:
            if not u.tokens: continue
            n += 1; ts = [t for _, t in u.tokens]; ok += (min(ts) >= u.start - 0.3 and max(ts) <= u.end + 0.3)
    print(f"  token times inside utterance ±0.3 s: {ok}/{n} = {ok/max(1,n):.4f}")
    rng = random.Random(0); bad = []; rms = []; ntext = []
    for i in rng.sample(range(len(ds)), min(40, len(ds))):
        it = ds[i]; L = ds.items[i][2]; w = it["wav"].numpy(); r = 10 * np.log10(np.mean(w ** 2) + 1e-12); rms.append(r); ntext.append(it["n_text"])
        if abs(len(w) - L * 16000) > 1600 or r < -50 or it["n_text"] < 8: bad.append((it["id"], len(w), L, round(r, 1), it["n_text"]))
    print(f"  40 windows: rms dB p10/50/90 {np.percentile(rms, [10, 50, 90]).round(1)}, n_text p10/50/90 {np.percentile(ntext, [10, 50, 90])}, bad {bad[:5]}")
    it = ds[0]; lab = [x for x in it["labels"].tolist() if x >= 0]
    print("  sample", it["id"], "text:", tok.decode(lab)[:300].replace("\n", " "))

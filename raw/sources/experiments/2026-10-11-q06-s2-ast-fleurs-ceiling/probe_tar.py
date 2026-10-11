import tarfile, io, sys, glob
import numpy as np, soundfile as sf
root = glob.glob("/soundai/users/tskim/VAPKT-data/data/corpora/fleurs*")[0]
print("root", root)
want = {"17897122138146023259.wav", "14648010410081654131.wav"}
peaks = []
with tarfile.open(f"{root}/data/en_us/audio/test.tar.gz") as tf:
    for i, m in enumerate(tf):
        if not m.isfile(): continue
        x, sr = sf.read(io.BytesIO(tf.extractfile(m).read()), dtype="float32"); info = sf.info(io.BytesIO(tf.extractfile(m).read()))
        p = float(np.abs(x).max()); peaks.append(p)
        if m.name.split("/")[-1] in want: print(m.name, sr, info.subtype, "peak", round(p, 4))
        if len(peaks) >= 300: break
print("first 300 en_us test peaks: median", np.median(peaks), "p10", np.percentile(peaks, 10), "p90", np.percentile(peaks, 90), "frac<0.02", np.mean(np.array(peaks) < 0.02))

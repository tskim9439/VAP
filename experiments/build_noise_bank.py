#!/usr/bin/env python3
"""학습 증강용 잡음·RIR 뱅크 — /soundai/DB 의 MUSAN · RIRS_NOISES · DEMAND · ETSI 를 읽기만 하고(압축을 풀지 않고 멤버를 바로 읽는다)
16 kHz mono 로 <out> 에 쓴다. vapasr/data/noise_aug.py 가 읽는다.

  잡음(category noise | music): <out>/audio/<source>/<name>.wav (PCM16) + <out>/noise.jsonl {path, source, category, n, dur}
    MUSAN noise(free-sound·sound-bible) = noise, MUSAN music = music. MUSAN speech 는 넣지 않는다(영어 음성이 섞이면 전사 타깃과 어긋난다).
    RIRS_NOISES real_rirs_isotropic_noises 의 *noise* = noise(pointsource_noises 는 MUSAN free-sound 와 같은 파일이라 뺀다). DEMAND 환경마다 ch01 = noise. ETSI(바이노럴) 첫 채널 = noise,
    이름에 Voice/Speech 가 든 파일(음성 방해원)은 뺀다.
  RIR: <out>/rir.npy float16 (N, L) + rir.jsonl {i, source, n}. RIRS_NOISES simulated_rirs(작은·중간·큰 방) 에서 방마다 --sim-per-room 개(sha1 결정적 표본)
    + real_rirs_isotropic_noises 의 RIR 전부(다채널이면 첫 채널). 최대 --rir-max-s 초로 자르고 절대값 최대 1 로 맞춘다.
48 kHz 등은 resample_poly 로 16 kHz. 이미 있는 출력 파일은 건너뛴다(이어 하기). 아무것도 지우지 않는다. summary.json 에 원천별 수·시간.

  python experiments/build_noise_bank.py --db /soundai/DB --out /soundai/users/tskim/VAPKT-data/data/noise-bank-v1
"""
import argparse, glob, hashlib, io, json, os, re, sys, tarfile, zipfile
from collections import Counter, defaultdict
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SR = 16000
SPEECHY = re.compile(r"voice|speech", re.I)


def to16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2: x = x[:, 0]
    if sr != SR:
        g = gcd(int(sr), SR); x = resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
    return x


def read_bytes(b: bytes):
    x, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True); return to16k_mono(x, sr)


class Writer:
    def __init__(self, out: Path):
        self.out = out; self.rows = []; self.stats = defaultdict(lambda: Counter())
    def noise(self, source, category, name, x):
        rel = f"audio/{source}/{name}.wav"; p = self.out / rel; p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            x = x / max(1e-6, float(np.abs(x).max())) * 0.9 if np.abs(x).max() > 1.0 else x
            tmp = p.with_suffix(".tmp.wav"); sf.write(tmp, x, SR, subtype="PCM_16"); os.replace(tmp, p)
            n = len(x)
        else:
            n = sf.info(p).frames
        if n < SR // 2: self.stats[source]["too_short"] += 1; return
        self.rows.append(dict(path=str(p), source=source, category=category, n=int(n), dur=round(n / SR, 3)))
        self.stats[source][category] += 1; self.stats[source]["hours"] += n / SR / 3600


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="/soundai/DB"); ap.add_argument("--out", required=True)
    ap.add_argument("--sim-per-room", type=int, default=2000, help="simulated_rirs 방 종류(small/medium/large)마다 RIR 수")
    ap.add_argument("--rir-max-s", type=float, default=1.0)
    ap.add_argument("--skip", default="", help="건너뛸 원천(쉼표): musan,rirs,demand,etsi")
    a = ap.parse_args(argv)
    db, out = Path(a.db), Path(a.out); out.mkdir(parents=True, exist_ok=True); skip = set(x for x in a.skip.split(",") if x)
    W = Writer(out); rirs, rir_rows = [], []

    if "rirs" not in skip:
        with zipfile.ZipFile(db / "rirs_noises" / "rirs_noises.zip") as z:
            names = sorted(n for n in z.namelist() if n.lower().endswith(".wav"))
            sim = [n for n in names if "/simulated_rirs/" in n]
            sim.sort(key=lambda n: hashlib.sha1(n.encode()).hexdigest()); room_n = Counter()
            for n in sim:
                room = n.split("/simulated_rirs/")[1].split("/")[0]
                if room_n[room] >= a.sim_per_room: continue
                room_n[room] += 1; rirs.append(read_bytes(z.read(n))); rir_rows.append(dict(source=f"sim-{room}", name=n))
            for n in names:
                base = n.rsplit("/", 1)[-1]
                if "/real_rirs_isotropic_noises/" in n:
                    if "noise" in base.lower(): W.noise("rirs-isotropic", "noise", base[:-4], read_bytes(z.read(n)))
                    else: rirs.append(read_bytes(z.read(n))); rir_rows.append(dict(source="real", name=n))
        L = int(a.rir_max_s * SR); bank = np.zeros((len(rirs), L), np.float16)
        for i, (r, row) in enumerate(zip(rirs, rir_rows)):
            r = r[:L]; r = r / max(1e-8, float(np.abs(r).max())); bank[i, :len(r)] = r; row.update(i=i, n=int(len(r)))
        np.save(out / "rir.npy.tmp.npy", bank); os.replace(out / "rir.npy.tmp.npy", out / "rir.npy")
        (out / "rir.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rir_rows))
        W.stats["rir"].update(Counter(r["source"] for r in rir_rows))
        print(f"rir {len(rir_rows)} ({dict(W.stats['rir'])})", flush=True)

    if "demand" not in skip:
        for zp in sorted(glob.glob(str(db / "demand" / "*.zip"))):
            env = Path(zp).stem.split("_")[0]
            with zipfile.ZipFile(zp) as z:
                ch = sorted(n for n in z.namelist() if n.lower().endswith("ch01.wav"))
                if ch: W.noise("demand", "noise", env, read_bytes(z.read(ch[0])))
        print(f"demand {dict(W.stats['demand'])}", flush=True)

    if "etsi" not in skip:
        for p in sorted(glob.glob(str(db / "ETSI" / "**" / "*.wav"), recursive=True)):
            base = Path(p).stem
            if SPEECHY.search(base): W.stats["etsi"]["skipped_speech"] += 1; continue
            x, sr = sf.read(p, dtype="float32", always_2d=True)
            W.noise("etsi", "noise", re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_"), to16k_mono(x, sr))
        print(f"etsi {dict(W.stats['etsi'])}", flush=True)

    if "musan" not in skip:
        with tarfile.open(db / "musan" / "musan.tar.gz") as t:
            for m in t:
                if not m.isfile() or not m.name.lower().endswith(".wav"): continue
                parts = m.name.split("/")
                if len(parts) < 4 or parts[1] not in ("noise", "music"): continue
                W.noise(f"musan-{parts[1]}-{parts[2]}", parts[1], parts[-1][:-4], read_bytes(t.extractfile(m).read()))
        print(f"musan done", flush=True)

    idx = out / "noise.jsonl"; tmp = out / "noise.jsonl.tmp"
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in W.rows)); os.replace(tmp, idx)
    summ = {s: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()} for s, c in W.stats.items()}
    summ["_total"] = dict(noise_files=len(W.rows), noise_hours=round(sum(r["n"] for r in W.rows) / SR / 3600, 2),
                          music_hours=round(sum(r["n"] for r in W.rows if r["category"] == "music") / SR / 3600, 2), rirs=len(rir_rows))
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1))
    print(json.dumps(summ["_total"]), flush=True)


if __name__ == "__main__":
    main()

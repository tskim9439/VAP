#!/usr/bin/env python
"""원거리 버전 대화(2026-10-26, 사용자 "모두 진행하는 데, 모두 진행하고 나서 E5 E6 학습 진행하자" — 데이터 확장 D-1): 근접 마이크로 정렬·정제된 회의
(<data>/<corpus>.refined.dialogues.jsonl)의 학습 회의마다, 같은 회의의 원거리 녹음(장치·채널 하나)을 mono 로 쓰는 복사본을 만든다.
발화·토큰·화자·플래그·채널 참조는 그대로이고 meta.mono_src = {path, channel, offset_s, device} 만 더한다(dialogue_mix.far_mono 가 mono 캐시를 만든다).
  python experiments/p2_build_far.py --corpus notsofar|icsi|ami|chime6 --data <phase2> --split <phase2>/splits/v4.json
      [--icsi-transcripts <ICSI_original_transcripts.zip>] [--ami-root <AMI 원거리 루트>] [--chime6-root <CHiME6 train 오디오>] [--mc-channels 0] [--procs 16] [--limit N]
장치 원천: notsofar = 회의 폴더 devices.json 의 근접 아닌 장치(어레이는 --mc-channels 채널, 단일 마이크는 그 채널), icsi = .mrt <Channels> 중 참가자 채널이 아닌 것
  (PZM c2·저가 u3 탁상 마이크), ami = <ami-root>/<회의>/<회의>.Array1-0{1..8}.wav, chime6 = <chime6-root>/<세션>_U0{1..6}.CH1.wav.
시각 맞춤: 변형마다 근접 mono 캐시(<data>/mono/<CORPUS>/<conv>.npy)와 장치 녹음의 로그 에너지 포락선(10 ms) 상호상관을 발화가 많은 60 s 구간 3 개에서 ±2 s 로 재
  offset_s(장치 시각 − 대화 시각)를 정한다. 세 구간의 지연이 50 ms 안에서 맞고 정규화 상관이 --min-corr 이상일 때만 쓰고, 아니면 버린다(build.json 에 사유).
출력: <data>/<key>-far.refined.dialogues.jsonl(이미 있으면 멈춤 — 덮어쓰지 않는다) + <data>/<key>-far.build.json. key = --key(기본 --corpus) — 입력도 <data>/<key>.refined.dialogues.jsonl.
  conv_id = "<key>-far:<회의>@<장치>", corpus = "<CORPUS>-far". split(held-out)과 split != train 회의는 넣지 않는다. 읽기 전용 입력.
  --key(2026-10-26): 평가 셋과 파일 이름이 겹치는 코퍼스의 학습용 키 — CHiME-6 train 은 chime6tr(phase2/chime6.dialogues.jsonl 은 dev 평가 셋이고, 예전 평가 스크립트가
  <corpus>.refined 를 먼저 찾는다)."""
import argparse, glob, json, multiprocessing as mp, os, re, sys, time, zipfile
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SR, HOP = 16000, 160

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True, choices=["notsofar", "icsi", "ami", "chime6"]); ap.add_argument("--data", required=True); ap.add_argument("--split", required=True)
ap.add_argument("--icsi-transcripts", default="/soundai/DB/raw/icsi/annotations/ICSI_original_transcripts.zip"); ap.add_argument("--ami-root", default=None)
ap.add_argument("--chime6-root", default=None); ap.add_argument("--mc-channels", default="0", help="notsofar 어레이에서 쓸 채널(쉼표)")
ap.add_argument("--key", default=None, help="phase2 파일 키(기본 --corpus) — 예 chime6tr: <data>/chime6tr.refined.dialogues.jsonl → chime6tr-far")
ap.add_argument("--min-corr", type=float, default=0.3); ap.add_argument("--agree-s", type=float, default=0.05); ap.add_argument("--max-lag-s", type=float, default=2.0)
ap.add_argument("--procs", type=int, default=16); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--dry-run", action="store_true", help="파일을 쓰지 않고 지연만 잰다")
a = ap.parse_args() if __name__ == "__main__" else None


def env(x: np.ndarray) -> np.ndarray:
    """10 ms 로그 에너지 포락선(평균 0, 분산 1)."""
    n = len(x) // HOP
    e = np.log(np.sqrt(np.mean(np.asarray(x[: n * HOP], dtype=np.float32).reshape(n, HOP) ** 2, axis=1)) + 1e-5)
    return (e - e.mean()) / (e.std() + 1e-6)


def xcorr_lag(ref: np.ndarray, dev: np.ndarray, max_lag: int):
    """dev[t + L] ≈ ref[t] 인 L(프레임)과 정규화 상관. dev 는 ref 보다 앞뒤로 max_lag 씩 길다(dev[0] = ref 시작 − max_lag)."""
    n = len(ref) + len(dev); c = np.fft.irfft(np.fft.rfft(dev, n) * np.conj(np.fft.rfft(ref, n)), n)
    k = np.arange(0, 2 * max_lag + 1); v = c[k] / len(ref); j = int(np.argmax(v))
    return int(k[j]) - max_lag, float(v[j])


def speech_segments(utts, dur: float, seg_s: float = 60.0, n: int = 3):
    """발화 시간이 가장 많은 seg_s 구간 n 개(서로 겹치지 않게, 앞뒤 5 s 여유)."""
    cand = []
    for t0 in np.arange(5.0, max(5.0, dur - seg_s - 5.0) + 1e-9, 10.0):
        sp = sum(max(0.0, min(e, t0 + seg_s) - max(s, t0)) for s, e in utts); cand.append((sp, float(t0)))
    out = []
    for sp, t0 in sorted(cand, reverse=True):
        if all(abs(t0 - u) >= seg_s for u in out): out.append(t0)
        if len(out) == n: break
    return sorted(out)


def measure(job):
    """job = (row dict, device dict, mono cache path) → (offset_s, corr, 사유 또는 None)."""
    from vapasr.data.streams import load_utt_audio
    row, dev, mono_f = job; ml = int(round(a.max_lag_s * SR / HOP))
    try:
        ref = np.load(mono_f, mmap_mode="r"); utts = [(u["start"], u["end"]) for u in row["utterances"]]
        segs = speech_segments(utts, row["duration_s"]); lags, cors = [], []
        if not segs: return None, None, "발화 구간 없음"
        path = dev["path"] + (f"#ch{dev['channel']}" if dev.get("channel") else "")
        for t0 in segs:
            r = env(ref[int(t0 * SR): int((t0 + 60.0) * SR)])
            s0 = t0 - a.max_lag_s; pad = max(0.0, -s0); x = load_utt_audio(path, max(0.0, s0), 60.0 + 2 * a.max_lag_s - pad)
            if pad: x = np.concatenate([np.zeros(int(round(pad * SR)), dtype=np.float32), x])
            d = env(x); m = min(len(d), len(r) + 2 * ml)
            if m < len(r) + 2 * ml: return None, None, f"장치 녹음이 짧다({t0:.0f} s 구간)"
            L, c = xcorr_lag(r, d[:m], ml); lags.append(L * HOP / SR); cors.append(c)
        if max(lags) - min(lags) > a.agree_s: return float(np.median(lags)), float(min(cors)), f"구간마다 지연이 다르다 {[round(x, 3) for x in lags]}"
        if min(cors) < a.min_corr: return float(np.median(lags)), float(min(cors)), f"상관이 낮다 {[round(x, 2) for x in cors]}"
        return float(np.median(lags)), float(min(cors)), None
    except Exception as e:
        return None, None, f"{type(e).__name__}: {e}"


def devices_of(row):
    """→ [{path, channel, device, mic?}] — 그 회의의 원거리 녹음."""
    c = a.corpus; ch0 = next(iter(row["channels"].values()))["path"]
    if c == "notsofar":
        mdir = os.path.dirname(os.path.dirname(ch0)); devs = json.load(open(os.path.join(mdir, "devices.json"))); out = []
        mc = [int(x) for x in a.mc_channels.split(",") if x != ""]
        for d in devs:
            if d.get("is_close_talk"): continue
            files = [f.strip() for f in d["wav_file_names"].split(",")]
            for k, f in enumerate(files):
                if d.get("is_mc") and k not in mc: continue
                out.append(dict(path=os.path.join(mdir, f), channel=0, device=f.replace("/", ".").replace(".wav", "")))
        return out
    if c == "icsi":
        m = row["conv_id"].split(":", 1)[1]; z = zipfile.ZipFile(a.icsi_transcripts); f = next(n for n in z.namelist() if n.endswith(f"/{m}.mrt"))
        root = ET.fromstring(z.read(f)); part = {p.get("Channel") for p in root.iter("Participant")}
        adir = os.path.dirname(ch0); out = []
        for ch in root.iter("Channel"):
            name = ch.get("Name")
            if not name or name in part: continue
            p = os.path.join(adir, f"{name}.flac")
            if os.path.exists(p): out.append(dict(path=p, channel=0, device=name, mic=ch.get("Mic")))
        return out
    if c == "ami":
        m = row["conv_id"].split(":", 1)[1]
        return [dict(path=p, channel=0, device=f"Array1-0{k}") for k in range(1, 9) for p in [os.path.join(a.ami_root, m, f"{m}.Array1-0{k}.wav")] if os.path.exists(p)]
    if c == "chime6":
        s = row["conv_id"].split(":", 1)[1]
        return [dict(path=p, channel=0, device=f"U0{k}.CH1") for k in range(1, 7) for p in [os.path.join(a.chime6_root, f"{s}_U0{k}.CH1.wav")] if os.path.exists(p)]
    raise ValueError(c)


def main():
    key = a.key or a.corpus; src = os.path.join(a.data, f"{key}.refined.dialogues.jsonl"); out = os.path.join(a.data, f"{key}-far.refined.dialogues.jsonl")
    if not a.dry_run and os.path.exists(out): sys.exit(f"이미 있다(덮어쓰지 않는다): {out}")
    held = set(json.load(open(a.split))["corpora"].get(key, {}).get("heldout", []))
    rows = [json.loads(l) for l in open(src, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["conv_id"] not in held and r.get("split", "train") == "train"]
    if a.limit: rows = rows[: a.limit]
    from vapasr.data.dialogue import Dialogue
    from vapasr.data.dialogue_dataset import mono_cache_path
    jobs = []
    for r in rows:
        mf = mono_cache_path(os.path.join(a.data, "mono"), Dialogue.from_json(json.dumps(r)))
        if not os.path.exists(mf): print(f"!! 근접 mono 캐시 없음 — 건너뜀 {r['conv_id']}", flush=True); continue
        for d in devices_of(r): jobs.append((r, d, mf))
    print(f"{key}: 학습 회의 {len(rows)} · 원거리 변형 {len(jobs)} — 지연 측정", flush=True); t0 = time.time()
    with mp.get_context("fork").Pool(a.procs) as pool: res = pool.map(measure, jobs, chunksize=1)
    keep, drop = [], []
    for (r, d, _), (off, cor, why) in zip(jobs, res):
        rec = dict(conv=r["conv_id"], device=d["device"], offset_s=off, corr=cor, **({"mic": d["mic"]} if "mic" in d else {}))
        (drop if why else keep).append(dict(rec, **({"why": why} if why else {})))
    offs = [k["offset_s"] for k in keep]
    summary = dict(corpus=key, meetings=len(rows), variants=len(jobs), kept=len(keep), dropped=len(drop), sec=round(time.time() - t0, 1),
                   offset_ms=dict(median=round(1000 * float(np.median(offs)), 1), p05=round(1000 * float(np.percentile(offs, 5)), 1), p95=round(1000 * float(np.percentile(offs, 95)), 1)) if offs else None,
                   corr_median=round(float(np.median([k["corr"] for k in keep])), 3) if keep else None, drop_reasons=sorted({re.sub(r"[-\d.,\[\] ]+$", "", k["why"]) for k in drop}))
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    for x_ in drop[:15]: print("  버림", x_, flush=True)                                      # 2026-10-27: 예전엔 k 로 돌아 출력 키를 덮었다(ami-far conv_id·build.json 이름 사고)
    if a.dry_run: return
    up = a.corpus.upper() if a.corpus != "chime6" else "CHiME6"
    with open(out + f".{os.getpid()}.tmp", "w", encoding="utf-8") as fo:
        for (r, d, _), (off, cor, why) in zip(jobs, res):
            if why: continue
            m = r["conv_id"].split(":", 1)[1]; x = dict(r)
            x["conv_id"] = f"{key}-far:{m}@{d['device']}"; x["corpus"] = f"{up}-far"
            x["meta"] = dict(r.get("meta") or {}, mono_src=dict(path=d["path"], channel=int(d.get("channel", 0)), offset_s=round(off, 3), device=d["device"], corr=round(cor, 3),
                                                              **({"mic": d["mic"]} if "mic" in d else {})), far_of=r["conv_id"])
            fo.write(json.dumps(x, ensure_ascii=False) + "\n")
    os.replace(out + f".{os.getpid()}.tmp", out)
    json.dump(dict(summary, kept_rows=keep, dropped_rows=drop), open(os.path.join(a.data, f"{key}-far.build.json"), "w"), ensure_ascii=False, indent=1)
    print(f"FAR_OK {out}", flush=True)


if __name__ == "__main__":
    main()

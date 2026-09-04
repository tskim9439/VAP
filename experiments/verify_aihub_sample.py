#!/usr/bin/env python
"""AI Hub 감정 태깅 자유대화 — 실물 검증 (task-verify-aihub-stereo-and-access).

검사 항목
  1. 채널 수 / 샘플레이트 / 비트 깊이 — 실제 2채널인가
  2. 채널 간 누설(crosstalk): 화자 A 발화 구간에서 B 채널의 에너지 비율, 채널 상관
  3. 발화 타임스탬프 정밀도: JSON 의 start/end 가 몇 ms 단위인가, 에너지 VAD 경계와의 오차 분포
  4. 겹침(overlap) 비율 — VAP 에 필요한 실제 대화 역학이 있는가

실행: python experiments/verify_aihub_sample.py <다운로드 루트> [--n 100]
결과: stdout 요약 + $DATA_LOG_DIR/aihub-verify-<ts>.json
"""
import os, sys, json, glob, time, argparse, re
import numpy as np, soundfile as sf
def _f(x): return float(str(x).replace(',', ''))   # AI Hub 숫자는 천 단위 구분자 포함 ('1,003.24')

ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--wav-root", default=None, help="wav 탐색 루트 (기본: root)")
ap.add_argument("--frame-ms", type=float, default=20.0); ap.add_argument("--vad-db", type=float, default=-40.0)
a = ap.parse_args()

wavs = sorted(glob.glob(os.path.join(a.wav_root or a.root, "**", "*.wav"), recursive=True))
import random; random.seed(0); random.shuffle(wavs); wavs = wavs[: a.n]   # 무작위 표본
print(f"wav {len(wavs)}개 (최대 {a.n}) under {a.root}")
if not wavs: sys.exit("wav 없음 — 압축 해제 여부와 경로 확인")

_JSON_INDEX = {os.path.splitext(os.path.basename(j))[0]: j for j in glob.glob(os.path.join(a.root, "**", "*.json"), recursive=True)}
def find_json(wav):
    return _JSON_INDEX.get(os.path.splitext(os.path.basename(wav))[0])

def energy_vad(x, sr, frame_ms, thr_db):
    n = int(sr * frame_ms / 1000); T = len(x) // n
    fr = x[: T * n].reshape(T, n); db = 10 * np.log10((fr ** 2).mean(1) + 1e-10)
    return db > thr_db, db

def parse_segments(js):
    """실제 스키마: Conversation[{StartTime, EndTime, SpeakerNo}] (초, 소수 2자리)."""
    return [(_f(u["StartTime"]), _f(u["EndTime"]), u.get("SpeakerNo", "?")) for u in js.get("Conversation", [])]

R = dict(files=[], channels={}, srs={}, subtypes={})
lag_ms, leak_db, corr, overlap_ratio, ts_res = [], [], [], [], []
for w in wavs:
    info = sf.info(w); R["channels"][info.channels] = R["channels"].get(info.channels, 0) + 1
    R["srs"][info.samplerate] = R["srs"].get(info.samplerate, 0) + 1; R["subtypes"][info.subtype] = R["subtypes"].get(info.subtype, 0) + 1
    x, sr = sf.read(w, dtype="float32", always_2d=True)
    row = dict(file=os.path.relpath(w, a.root), ch=info.channels, sr=sr, dur=len(x) / sr)
    if info.channels == 2:
        A, B = x[:, 0], x[:, 1]
        vA, dA = energy_vad(A, sr, a.frame_ms, a.vad_db); vB, dB = energy_vad(B, sr, a.frame_ms, a.vad_db)
        onlyA = vA & ~vB
        if onlyA.sum() > 10:
            leak = (dB[onlyA] - dA[onlyA]).mean(); leak_db.append(float(leak)); row["leak_dB_B_during_A"] = float(leak)
        c = float(np.corrcoef(A[: sr * 60], B[: sr * 60])[0, 1]); corr.append(c); row["corr"] = c
        ov = float((vA & vB).sum() / max(1, (vA | vB).sum())); overlap_ratio.append(ov); row["overlap_ratio"] = ov
        row["speech_ratio"] = [float(vA.mean()), float(vB.mean())]
    js = find_json(w)
    if js:
        try:
            segs = parse_segments(json.load(open(js, encoding="utf-8")))
            row["json"] = os.path.relpath(js, a.root); row["n_segments"] = len(segs)
            if segs:
                # 타임스탬프 해상도 추정: 소수점 자릿수
                decs = [len(str(s).split(".")[1]) if "." in str(s) else 0 for s, _, _ in segs[:50]]
                ts_res.append(max(decs)); row["ts_decimals"] = max(decs)
                if info.channels == 2:
                    # 각 세그먼트 시작을 에너지 VAD 온셋과 비교 (어느 채널이든 가장 가까운 온셋)
                    n = int(sr * a.frame_ms / 1000)
                    on = np.flatnonzero(np.diff((vA | vB).astype(int)) == 1) * a.frame_ms / 1000
                    if len(on):
                        d = [float(np.min(np.abs(on - s)) * 1000) for s, _, _ in segs[:200]]
                        row["onset_err_ms_median"] = float(np.median(d)); lag_ms.extend(d)
        except Exception as e: row["json_err"] = str(e)[:80]
    R["files"].append(row)

def q(v): return dict(n=len(v), median=float(np.median(v)), p90=float(np.percentile(v, 90)), mean=float(np.mean(v))) if v else None
R["summary"] = dict(channels=R["channels"], sr=R["srs"], subtype=R["subtypes"],
                    crosstalk_leak_dB=q(leak_db), channel_corr=q(corr), overlap_ratio=q(overlap_ratio),
                    timestamp_decimals=q(ts_res), onset_err_ms=q(lag_ms))
print(json.dumps(R["summary"], indent=1, ensure_ascii=False))
print("\n판정 가이드: 채널 2 & leak < -20 dB & corr < 0.3 → 진짜 분리 stereo. overlap_ratio 가 0 에 가까우면 대화 역학 부족.")
out = os.path.join(os.environ.get("DATA_LOG_DIR", "/tmp"), f"aihub-verify-{time.strftime('%Y%m%d-%H%M')}.json")
json.dump(R, open(out, "w"), indent=1, ensure_ascii=False); print("saved", out)

# AMI 공식 평가 회의(full-corpus ASR 분할 dev 18 · eval 16)의 다화자 평가 데이터(mxc 호스트 CPU) — 사용자 "AMI-IHM 과 AMI-SDM 으로도 평가", "mxc 에 바로 받자."
#   SDM : diarizers-community/ami(sdm, @8cdaae2, CC-BY-4.0 — 회의 통째 Array1-01, validation = 공식 dev, test = 공식 eval)의 parquet 에서 회의 오디오를 꺼낸다.
#         공식 사이트(groups.inf.ed.ac.uk)는 mxc 에서 막혀 있다(https SSL 오류, http 470).
#   IHM-mix : /soundai/DB/raw/ami/<회의>/<회의>.Mix-Headset.wav(공식 헤드셋 합) — 읽기만 한다.
#   참조·시각은 Phase 2 정제 대화(phase2/ami.refined.dialogues.jsonl — 같은 회의의 헤드셋 정렬 토큰)를 그대로 쓴다(AMI 신호는 한 시간축에 맞춰져 있다).
#   mono 캐시 = 그 오디오를 발화 구간 RMS 로 −23 dBFS 에 맞추고 피크 0.98 로 자른 것(학습·평가 mono 와 같은 음량 규약) → <out>/mono/<코퍼스>/<회의>.npy(float16).
#   길이는 대화 길이에 맞춘다(짧으면 0 패딩, 길면 자름). 회의 길이 차이가 1 s 를 넘으면 기록한다(정렬 의심).
#   python ami_bench_build.py --hf <ami-sdm-hf 폴더> --phase2 <phase2> --out <ami-bench-v1> — 출력 폴더가 있으면 멈춘다. 아무것도 지우지 않는다.
import argparse, glob, io, json, math, os, sys
import numpy as np, soundfile as sf

AMI_DEV = ["ES2011a", "ES2011b", "ES2011c", "ES2011d", "IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011",
           "IS1008a", "IS1008b", "IS1008c", "IS1008d", "TS3004a", "TS3004b", "TS3004c", "TS3004d"]
AMI_EVAL = ["EN2002a", "EN2002b", "EN2002c", "EN2002d", "ES2004a", "ES2004b", "ES2004c", "ES2004d",
            "IS1009a", "IS1009b", "IS1009c", "IS1009d", "TS3003a", "TS3003b", "TS3003c", "TS3003d"]
SR = 16000


def norm(x, spans, target_dbfs=-23.0, peak=0.98):
    acc = n = 0
    for a, b in spans:
        seg = x[int(a * SR): int(b * SR)].astype(np.float64); acc += float(np.sum(seg ** 2)); n += len(seg)
    rms = math.sqrt(acc / n) if n else 1e-6; y = x * ((10 ** (target_dbfs / 20)) / max(rms, 1e-6)); pk = float(np.max(np.abs(y))) if len(y) else 0.0
    return (y * (peak / pk) if pk > peak else y).astype(np.float32), round(20 * math.log10(max(rms, 1e-9)), 2)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--hf", required=True); ap.add_argument("--phase2", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    sys.path.insert(0, "/soundai/users/tskim/VAPKT")
    import pyarrow.parquet as pq
    from vapasr.data.dialogue import Dialogue
    from vapasr.data.dialogue_dataset import mono_cache_path
    assert not os.path.exists(a.out), f"{a.out} 가 이미 있다"
    want = set(AMI_DEV + AMI_EVAL); D = {}
    for l in open(os.path.join(a.phase2, "ami.refined.dialogues.jsonl"), encoding="utf-8"):
        d = Dialogue.from_json(l); m = d.conv_id.split(":", 1)[1]
        if m in want: D[m] = d
    assert set(D) == want, f"Phase 2 에 없는 회의: {sorted(want - set(D))}"
    sdm = {}                                                                                    # 회의 → (오디오, parquet 분할)
    for f in sorted(glob.glob(os.path.join(a.hf, "sdm", "*.parquet"))):
        split = os.path.basename(f).split("-")[0]
        for r in pq.read_table(f, columns=["audio"]).to_pylist():
            au = r["audio"]; name = os.path.basename(au.get("path") or ""); m = name.split(".")[0]
            assert m in want, f"{f}: 모르는 회의 이름 {name!r}"
            x, sr = sf.read(io.BytesIO(au["bytes"]), dtype="float32"); x = x if x.ndim == 1 else x[:, 0]
            assert sr == SR, f"{name}: {sr} Hz"; sdm[m] = (x, split, name)
    got = {m: s for m, (_, s, _) in sdm.items()}
    assert set(sdm) == want, f"SDM 에 없는 회의: {sorted(want - set(sdm))}"
    assert {m for m, s in got.items() if s == "validation"} == set(AMI_DEV) and {m for m, s in got.items() if s == "test"} == set(AMI_EVAL), "HF 분할이 공식 dev·eval 과 다르다"
    os.makedirs(a.out); rep = dict(source=dict(sdm="diarizers-community/ami@8cdaae2eaf968f3b000b6eb1204ab9b8db006ed0 (sdm, CC-BY-4.0)", ihm_mix="/soundai/DB/raw/ami/<m>/<m>.Mix-Headset.wav",
                                               reference="phase2/ami.refined.dialogues.jsonl"), dev=AMI_DEV, eval=AMI_EVAL, meetings={})
    for corpus in ("ami-sdm", "ami-ihmmix"):
        with open(os.path.join(a.out, f"{corpus}.refined.dialogues.jsonl"), "w", encoding="utf-8") as fo:
            for m in AMI_DEV + AMI_EVAL:
                d = Dialogue.from_json(D[m].to_json()); T = int(round(d.duration_s * SR))
                if corpus == "ami-sdm": x, _, src = sdm[m]
                else:
                    src = f"/soundai/DB/raw/ami/{m}/{m}.Mix-Headset.wav"; x, sr = sf.read(src, dtype="float32"); x = x if x.ndim == 1 else x[:, 0]; assert sr == SR, f"{src}: {sr} Hz"
                diff = round(len(x) / SR - d.duration_s, 3)
                x = x[:T] if len(x) >= T else np.pad(x, (0, T - len(x)))
                y, lvl = norm(x, [(u.start, u.end) for u in d.utterances])
                d.corpus = corpus; d.conv_id = f"{corpus}:{m}"; d.meta = dict(d.meta or {}, source=src, official="dev" if m in AMI_DEV else "eval", len_diff_s=diff, src_speech_dbfs=lvl)
                f = mono_cache_path(os.path.join(a.out, "mono"), d); os.makedirs(os.path.dirname(f), exist_ok=True); np.save(f, y.astype(np.float16))
                fo.write(d.to_json() + "\n"); rep["meetings"].setdefault(m, {})[corpus] = dict(len_diff_s=diff, speech_dbfs=lvl, src=os.path.basename(src))
                print(f"{corpus} {m} 길이 차 {diff:+.3f} s · 말소리 {lvl} dBFS", flush=True)
    json.dump(rep, open(os.path.join(a.out, "manifest.json"), "w"), ensure_ascii=False, indent=1)
    big = [(m, c, v["len_diff_s"]) for m, x in rep["meetings"].items() for c, v in x.items() if abs(v["len_diff_s"]) > 1.0]
    print("BUILD_OK", len(rep["meetings"]), "회의 · 길이 차 1 s 초과", big)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""체크포인트 가중치 평균(학습 없음 — V3 체크포인트 사이 TAXI 1–2 pt 흔들림을 줄이는 값싼 대조).

  python experiments/avg_checkpoints.py --out <run 폴더>/final [--done] <ck1> <ck2> …

각 체크포인트의 *.safetensors 를 같은 파일·키·shape·dtype 끼리 산술 평균한다. 부동소수 텐서는 float64 로 더해 원래 dtype 으로 되돌리고,
정수·bool 텐서는 모든 체크포인트에서 같아야 한다(다르면 멈춤). 파일 구성·키·shape·dtype 이 하나라도 다르면 멈춘다.
나머지 파일(config·tokenizer·stage3_run 등)은 마지막 체크포인트에서 복사한다. 학습 상태(optimizer·scheduler·rng_state·trainer_state)는 뺀다.
<out> 이 이미 있으면 멈춘다(덮어쓰지 않는다). <out>.tmp 에 쓰고, 다시 읽어 평균과 비트 동일한지 대조한 뒤 <out> 으로 옮긴다.
avg.json 에 출처·파일별 키 수·크기·대조 결과를 남긴다. --done: <out> 의 부모 폴더에 DONE 을 만든다(평가 큐가 대상 final 을 이 표시로 기다린다)."""
import argparse, json, os, shutil, sys

TRAIN_STATE = ("optimizer.pt", "scheduler.pt", "trainer_state.json")


def average(cks, out, done=False):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    if len(cks) < 2: sys.exit("체크포인트를 2 개 이상 준다")
    if os.path.exists(out) or os.path.exists(out + ".tmp"): sys.exit(f"이미 있다: {out}(또는 .tmp) — 덮어쓰지 않는다")
    files = sorted(f for f in os.listdir(cks[-1]) if f.endswith(".safetensors"))
    if not files: sys.exit(f"safetensors 없음: {cks[-1]}")
    for ck in cks:
        got = sorted(f for f in os.listdir(ck) if f.endswith(".safetensors"))
        if got != files: sys.exit(f"safetensors 파일 구성이 다르다: {ck} {got} 대 {files}")
    tmp = out + ".tmp"; os.makedirs(tmp)
    info = dict(sources=[os.path.abspath(c) for c in cks], n=len(cks), files={})
    for f in files:
        acc, dt, meta = None, None, None
        for ck in cks:
            with safe_open(os.path.join(ck, f), "pt") as h: md = h.metadata()
            t = load_file(os.path.join(ck, f))
            if acc is None:
                acc = {k: (v.double() if v.is_floating_point() else v.clone()) for k, v in t.items()}; dt = {k: v.dtype for k, v in t.items()}; meta = md
                continue
            if set(t) != set(acc): sys.exit(f"{f}: 키가 다르다 — {ck}(빠짐 {sorted(set(acc) - set(t))[:5]} · 더함 {sorted(set(t) - set(acc))[:5]})")
            for k, v in t.items():
                if v.shape != acc[k].shape or v.dtype != dt[k]: sys.exit(f"{f}: {k} shape·dtype 이 다르다 — {ck} {tuple(v.shape)} {v.dtype} 대 {tuple(acc[k].shape)} {dt[k]}")
                if v.is_floating_point(): acc[k] += v.double()
                elif not torch.equal(v, acc[k]): sys.exit(f"{f}: 정수 텐서 {k} 가 체크포인트마다 다르다 — {ck}")
            del t
        avg = {k: ((v / len(cks)).to(dt[k]) if dt[k].is_floating_point else v) for k, v in acc.items()}
        save_file(avg, os.path.join(tmp, f), metadata=meta)
        back = load_file(os.path.join(tmp, f)); same = set(back) == set(avg) and all(torch.equal(back[k], avg[k]) for k in avg)
        info["files"][f] = dict(keys=len(avg), floating=sum(1 for k in avg if dt[k].is_floating_point), bytes=os.path.getsize(os.path.join(tmp, f)), parity=same)
        if not same: sys.exit(f"{f}: 다시 읽은 값이 평균과 다르다 — {tmp} 를 확인한다")
        del acc, avg, back
        print(f"  {f}: 키 {info['files'][f]['keys']} · 부동소수 {info['files'][f]['floating']} · 대조 OK", flush=True)
    for f in sorted(os.listdir(cks[-1])):
        if f.endswith(".safetensors") or f in TRAIN_STATE or f.startswith("rng_state"): continue
        src = os.path.join(cks[-1], f); (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, os.path.join(tmp, f))
    json.dump(info, open(os.path.join(tmp, "avg.json"), "w"), indent=1)
    os.rename(tmp, out)
    if done: open(os.path.join(os.path.dirname(os.path.abspath(out)), "DONE"), "w").write("avg_checkpoints\n")
    print(f"평균 {len(cks)} 개 → {out}", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="평균 체크포인트 폴더(없어야 한다)"); ap.add_argument("--done", action="store_true", help="<out> 부모 폴더에 DONE")
    ap.add_argument("checkpoints", nargs="+", help="평균할 체크포인트 폴더(마지막 것에서 config·tokenizer 를 복사)")
    a = ap.parse_args(); average(a.checkpoints, a.out, a.done)


if __name__ == "__main__":
    main()

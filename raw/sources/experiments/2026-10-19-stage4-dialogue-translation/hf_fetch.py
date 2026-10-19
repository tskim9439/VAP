#!/usr/bin/env python3
"""Hugging Face 저장소(모델·데이터셋)를 고정 리비전으로 받는다 — 이어받기(Range), 크기·LFS sha256 대조, 기존 파일은 지우지 않는다.

  python3 raw/sources/experiments/2026-10-19-stage4-dialogue-translation/hf_fetch.py --repo microsoft/VibeVoice-ASR --rev <sha> --out <dir> [--type dataset] [--include deu/,README.md] [--token-stdin]
  python3 raw/sources/experiments/2026-10-19-stage4-dialogue-translation/hf_fetch.py --url <URL> --sha256 <hex> --out <file>          (단일 파일: 소스 zip·휠)

--token-stdin: 표준 입력 첫 줄을 토큰으로 읽는다(게이트 저장소). 토큰은 출력하지 않고 명령줄·환경·디스크에 두지 않는다.
표준 라이브러리만 쓴다(서버 어느 python3 로도 돈다)."""
import argparse, hashlib, json, os, sys, time, urllib.request


def get(url, token, start=0):
    h = {"User-Agent": "hf_fetch/1"}
    if token: h["Authorization"] = f"Bearer {token}"
    if start: h["Range"] = f"bytes={start}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    return h.hexdigest()


def fetch(url, dest, size, token, tries=6):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True); part = dest + ".part"
    if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size): return "있음"
    for t in range(tries):
        try:
            have = os.path.getsize(part) if os.path.exists(part) else 0
            with get(url, token, have) as r, open(part, "ab" if have and r.status == 206 else "wb") as f:
                while True:
                    b = r.read(1 << 22)
                    if not b: break
                    f.write(b)
            if size is None or os.path.getsize(part) == size: os.replace(part, dest); return "받음"
        except Exception as e:
            print(f"  재시도 {t + 1}: {os.path.basename(dest)} ({type(e).__name__})", flush=True); time.sleep(5 * (t + 1))
    raise SystemExit(f"실패: {dest}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repo"); ap.add_argument("--rev"); ap.add_argument("--type", default="model", choices=["model", "dataset"])
    ap.add_argument("--include", default="", help="쉼표 목록 — 이 접두로 시작하는 파일만"); ap.add_argument("--out", required=True)
    ap.add_argument("--url"); ap.add_argument("--sha256"); ap.add_argument("--token-stdin", action="store_true")
    a = ap.parse_args()
    token = sys.stdin.readline().strip() if a.token_stdin else ""
    if a.url:
        st = fetch(a.url, a.out, None, token); ok = sha256(a.out) == a.sha256 if a.sha256 else None
        print(f"{st} {a.out} {os.path.getsize(a.out)} sha256 {'일치' if ok else ('불일치' if ok is False else '미대조')}"); sys.exit(0 if ok is not False else 1)
    base = "https://huggingface.co" + ("/datasets" if a.type == "dataset" else "")
    api = f"https://huggingface.co/api/{a.type}s/{a.repo}/revision/{a.rev}?blobs=true"
    with get(api, token) as r: info = json.load(r)
    inc = [x for x in a.include.split(",") if x]
    files = [x for x in info["siblings"] if not inc or any(x["rfilename"].startswith(p) for p in inc)]
    print(f"{a.repo}@{a.rev[:10]}: 파일 {len(files)} · {sum(x.get('size') or 0 for x in files) / 1e9:.2f} GB → {a.out}", flush=True)
    bad = 0
    for x in files:
        f = x["rfilename"]; dest = os.path.join(a.out, f); size = x.get("size"); lfs = (x.get("lfs") or {}).get("sha256")
        st = fetch(f"{base}/{a.repo}/resolve/{a.rev}/{f}", dest, size, token)
        ok = os.path.getsize(dest) == size and (lfs is None or sha256(dest) == lfs); bad += not ok
        print(f"  {st} {'OK' if ok else 'MISMATCH'} {f} {size}", flush=True)
    with open(os.path.join(a.out, "REVISION"), "w") as fo: fo.write(f"{a.repo} {a.rev}" + (f" (include {a.include})" if inc else "") + "\n")
    print(f"{a.repo}: 불일치 {bad}"); sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

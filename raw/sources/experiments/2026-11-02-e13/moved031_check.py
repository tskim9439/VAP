"""2026-11-02 AI Hub 031 폴더 이름 변경 점검(서버 호스트 CPU, 읽기만 — 공용 DB 는 건드리지 않는다).
manifest(words.jsonl·jsonl 들)의 옛 031 경로가 vapasr.data.archive.resolve_path 로 새 폴더(031.broadcasting_koen)에서 모두 찾아지는지,
압축(또는 파일)마다 항목 하나를 학습 로더 경로(streams.assemble_stream → load_seg_audio → read_member)로 실제로 읽어 본다.
  python moved031_check.py <words.jsonl|*.jsonl glob> ...
"""
import glob, json, re, sys, time
from vapasr.data import archive
from vapasr.data.streams import assemble_stream

PAT = re.compile(r'"path": "(/soundai/DB/raw/aihub/031\.[^"]*)"')
first, n_lines, n_refs = {}, 0, 0
for g in sys.argv[1:]:
    for f in sorted(glob.glob(g)):
        for l in open(f, encoding="utf-8"):
            n_lines += 1; m = PAT.findall(l)
            if not m: continue
            n_refs += len(m)
            for p in m: first.setdefault(p.split("::")[0], (l, p))
print(f"lines {n_lines} · 031 refs {n_refs} · distinct archives/files {len(first)}", flush=True)
miss = [a for a in first if archive.resolve_path(a) == a]
print(f"unresolved {len(miss)}", *[repr(a[-80:]) for a in miss[:5]], flush=True)
t, bad = time.time(), 0
for a, (l, p) in first.items():
    it = json.loads(l)
    try:
        if "segments" in it: y = assemble_stream(dict(id=it["id"], duration_s=it["duration_s"], segments=it["segments"]))
        else: from vapasr.data.streams import load_utt_audio; y = load_utt_audio(p)
        assert len(y) > 0 and float(abs(y).max()) > 0
    except Exception as e:
        bad += 1; print("FAIL", it.get("id"), type(e).__name__, str(e)[:200], flush=True)
print(f"read ok {len(first) - bad} · bad {bad} · {time.time() - t:.0f}s", flush=True)

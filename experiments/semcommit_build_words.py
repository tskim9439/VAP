#!/usr/bin/env python3
"""SEM_END v0 — words.jsonl(단어/어절 index) 빌더. 라이브러리: vapasr/data/semcommit_words.py.

<manifests>/<set>/aligned-manifest.jsonl.gz(정렬 캐시 export, streams 의 순서 보존 부분집합)와 <manifests>/original/<set>/streams.jsonl 을
id 로 병합하며 한 줄씩 읽는다(kspon-full streams 480 MB 를 메모리에 올리지 않는다; 비후보 행은 앞머리 regex 로 id 만 본다).
--id-prefix(·--mode) 에 맞는 후보를 모두 build_stream 으로 만들고(제외 사유 통계 = 모집단 전체), 유효 스트림 중 key = sha1("{seed}:{id}") 가
가장 작은 n 개를 고른다 — 파일 순서와 무관하게 결정적이며 n 을 키우면 이전 표본을 포함한다. 출력은 manifest 순서.
LibriSpeech-PC 태그(--pnc-dir)는 표본에만 붙인다(스트림을 떨어뜨리지 않으므로 표본 선택과 무관).

id 형식(로컬 T5 manifest 실측): ls-train-clean-100-103-1240-00000 · ls-test-clean-1089-134686-00000 · ls-test-clean-utt-1089-134686-0000(mode=utt) ·
  ks-train-01-KsponSpeech_000001 · ks-dev-utt-KsponSpeech_620001 · ks-eval_clean-utt-KsponSpeech_E00001 (eval 은 'ks-eval_clean', 모두 mode=utt).
  LS test/dev set 은 stream 행과 utt 행이 같은 발화를 담아 섞여 있다 — --mode 없이 후보에 두 mode 가 섞이면 종료(exit≠0).
실패(exit≠0): 후보 0(접두사 오타 등; 출력 안 씀) · 유효 스트림 0(stats 는 쓴다) · --mode 없이 mode 혼재.
--check-audio: auto(기본) = 이 set 코퍼스의 root 디렉터리가 이 호스트에 있으면 오디오 존재 확인(없으면 audio_missing 제외), on/off 강제.
  확인하지 않으면 stats.warnings 에 남긴다(rack4 /data5/LibriSpeech 에는 dev-clean/dev-other 가 없다 — librispeech-dev 는 rack4 에서 쓸 수 없음).

예(rack4):
  python experiments/semcommit_build_words.py --manifests /data3/tskim/manifests/forced-align-manifests --set librispeech-960 \\
      --id-prefix ls-train-clean-100 --mode stream --n 400 --seed 0 --tokenizer <Qwen3-ASR dir 또는 tokenizer.json> \\
      --pnc-dir /data5/LibriSpeech/librispeech_pnc --out words-ls100.jsonl --stats words-ls100.stats.json
"""
import argparse, gzip, hashlib, heapq, json, os, re, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vapasr.data.semcommit_words import _load_json, DEFAULT_ROOTS, DEFAULT_PNC_DIR, TAGS, PieceVocab, PncIndex, apply_en_pnc, build_stream

_ID = re.compile(r'"id":\s*"([^"]*)"'); _MODE = re.compile(r'"mode":\s*"([^"]*)"')

def head_field(line: str, rx, key: str):
    """JSON 줄 앞머리(600자)에서 문자열 필드를 regex 로 — 없으면 json.loads."""
    m = rx.search(line, 0, 600)
    return m.group(1) if m else json.loads(line).get(key)

def iter_joined(aligned_path: str, streams_path: str, counts: Counter):
    """aligned(gzip) 행마다 같은 id 의 streams 행을 찾아 (index, id, mode, aligned_line, streams_line). aligned 는 streams 의 순서 보존 부분집합 —
    streams 가 먼저 끝나면(순서 위반·다른 set) RuntimeError."""
    with gzip.open(aligned_path, "rt", encoding="utf-8") as fa, open(streams_path, encoding="utf-8") as fs:
        for la in fa:
            if not la.strip(): continue
            aid = head_field(la, _ID, "id"); counts["aligned_rows"] += 1
            while True:
                ls = fs.readline()
                if not ls: raise RuntimeError(f"aligned id {aid} not found in {streams_path} (order violated or wrong set)")
                if not ls.strip(): continue
                counts["streams_rows"] += 1
                if head_field(ls, _ID, "id") == aid: break
                counts["streams_not_aligned"] += 1                         # QC 로 정렬 캐시에서 빠진 스트림
            yield counts["aligned_rows"] - 1, aid, head_field(la, _MODE, "mode"), la, ls
        for ls in fs:
            if ls.strip(): counts["streams_rows"] += 1; counts["streams_not_aligned"] += 1

def load_tokenizer(spec: str):
    """tokenizer.json / vocab.json 파일 또는 그것을 담은 디렉터리 → PieceVocab(id→piece 만, transformers 불필요). 그 밖(허브 이름 등) → AutoTokenizer."""
    if os.path.isfile(spec) or (os.path.isdir(spec) and any(os.path.isfile(os.path.join(spec, f)) for f in ("tokenizer.json", "vocab.json"))):
        return PieceVocab(spec)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(spec)

def _write_atomic(path: str, write):
    with open(path + ".tmp", "w", encoding="utf-8") as f: write(f)
    os.replace(path + ".tmp", path)

def sample_key(seed: int, sid: str) -> int: return int(hashlib.sha1(f"{seed}:{sid}".encode()).hexdigest(), 16)
def _q(xs, ps=(0.05, 0.5, 0.95)):
    xs = sorted(xs); return {f"p{int(p * 100)}": round(xs[min(len(xs) - 1, int(p * len(xs)))], 3) for p in ps} if xs else {}
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def sample_stats(rows, infos) -> dict:
    tags, seg_assign, notes = Counter(), Counter(), Counter(); ratios = []
    for r, info in zip(rows, infos):
        for w in r["words"]:
            for t in w["tags"]: tags[t] += 1
        seg_assign[info.get("seg_assign")] += 1; notes.update(info.get("tag_notes", [])); ratios += info.get("pnc_ratios", [])
    durs = [r["duration_s"] for r in rows]; nw = [len(r["words"]) for r in rows]
    return dict(streams=len(rows), words=sum(nw), tokens=sum(len(r["tokens"]) for r in rows), hours=round(sum(durs) / 3600, 3),
                duration_s=dict(mean=round(sum(durs) / max(1, len(durs)), 3), **_q(durs)), words_per_stream=_q(nw),
                segments_per_stream=_q([len(r["segments"]) for r in rows]),
                speech_tail_s=_q([r["duration_s"] - r["words"][-1]["end_time"] for r in rows]),   # 스트림 끝 − 마지막 단어 끝 (TURN_END hangover 설계용)
                langs=dict(Counter(r["lang"] for r in rows)), tags={t: tags[t] for t in TAGS}, words_with_any_tag=sum(1 for r in rows for w in r["words"] if w["tags"]),
                seg_assign=dict(seg_assign), tag_notes=dict(notes), pnc_text=sum(1 for r in rows if "pnc_text" in r),
                pnc_ratio=dict(n=len(ratios), **_q(ratios)) if ratios else None)

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifests", required=True, help="original/<set>/streams.jsonl 과 <set>/aligned-manifest.jsonl.gz 를 담은 디렉터리")
    ap.add_argument("--set", required=True, help="librispeech-960 | librispeech-test | librispeech-dev | kspon-full | kspon-dev | kspon-eval")
    ap.add_argument("--id-prefix", default="", help="id 접두사(예: ls-train-clean-100, ks-train-01, ls-test-clean, ks-eval_clean); 빈 값 = 전부")
    ap.add_argument("--mode", choices=["stream", "utt"], default=None, help="행 mode 필터(LS test/dev 는 stream·utt 혼재; kspon-dev/eval 은 utt 뿐)")
    ap.add_argument("--n", type=int, default=0, help="표본 스트림 수(유효 스트림 중); 0 이하 = 전부")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", required=True, help="Qwen3(-ASR) tokenizer: tokenizer.json·vocab.json(+added_tokens.json) 또는 그 디렉터리(권장), 그 밖은 AutoTokenizer 이름")
    ap.add_argument("--ls-root", default=DEFAULT_ROOTS["LibriSpeech"])
    ap.add_argument("--kspon-root", default=DEFAULT_ROOTS["KsponSpeech"])
    ap.add_argument("--pnc-dir", default=None, help=f"LibriSpeech-PC <split>.json 디렉터리(rack4: {DEFAULT_PNC_DIR}); 없으면 EN 태그 없음")
    ap.add_argument("--pnc-min-ratio", type=float, default=0.9)
    ap.add_argument("--check-audio", nargs="?", const="on", default="auto", choices=["auto", "on", "off"],
                    help="remap 한 오디오 경로 존재 확인(없으면 audio_missing 으로 제외). auto(기본) = set 코퍼스 root 가 이 호스트에 있으면 on; 값 없이 쓰면 on")
    ap.add_argument("--out", required=True); ap.add_argument("--stats", required=True)
    a = ap.parse_args(argv)
    aligned = os.path.join(a.manifests, a.set, "aligned-manifest.jsonl.gz"); streams = os.path.join(a.manifests, "original", a.set, "streams.jsonl")
    for p in (aligned, streams):
        if not os.path.isfile(p): raise SystemExit(f"missing {p}")
    for p in (a.out, a.stats): os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)      # 긴 scan 전에 실패
    t0 = time.time(); tok = load_tokenizer(a.tokenizer); roots = {"LibriSpeech": a.ls_root, "KsponSpeech": a.kspon_root}
    corpus_root = roots["KsponSpeech" if a.set.startswith("kspon") else "LibriSpeech"]
    check = a.check_audio == "on" or (a.check_audio == "auto" and os.path.isdir(corpus_root)); warnings = []
    if not check:
        warnings.append(f"audio existence not checked (--check-audio {a.check_audio}; {corpus_root} {'exists' if os.path.isdir(corpus_root) else 'absent on this host'})")
        if a.set == "librispeech-dev" and a.ls_root.rstrip("/") == DEFAULT_ROOTS["LibriSpeech"]:
            warnings.append(f"rack4 {DEFAULT_ROOTS['LibriSpeech']} has no dev-clean/dev-other — segment paths in {a.out} will not resolve there")
    counts, drops = Counter(), Counter(); heap = []; drop_ex, ex_ids, modes = {}, [], set()   # heap: max-heap by key (−key) — 가장 작은 key n 개
    for idx, sid, mode, la, ls in iter_joined(aligned, streams, counts):
        if len(ex_ids) < 3: ex_ids.append(sid)
        if not sid.startswith(a.id_prefix) or (a.mode and mode != a.mode): continue
        if a.mode is None:
            modes.add(mode)
            if len(modes) > 1: raise SystemExit(f"--id-prefix {a.id_prefix!r} matches rows of modes {sorted(map(str, modes))} in {aligned} "
                                                 f"(stream and utt rows share utterances) — pass --mode stream|utt")
        counts["candidates"] += 1; info = {}
        out = build_stream(json.loads(la), json.loads(ls), tok, roots, info=info, set_name=a.set, path_exists=os.path.exists if check else None)
        if isinstance(out, tuple):
            drops[out[1]] += 1; ex = drop_ex.setdefault(out[1], [])
            if len(ex) < 5: ex.append(dict(id=sid, **({"detail": info["drop_detail"]} if "drop_detail" in info else {})))
            continue
        counts["valid"] += 1; item = (-sample_key(a.seed, sid), idx, out, info)
        if a.n <= 0 or len(heap) < a.n: heapq.heappush(heap, item)
        elif item[0] > heap[0][0]: heapq.heapreplace(heap, item)
    if not counts["candidates"]:
        raise SystemExit(f"no rows match --id-prefix {a.id_prefix!r} --mode {a.mode} in {aligned} (aligned_rows={counts['aligned_rows']}; ids look like {ex_ids}) — nothing written")
    sel = sorted(heap, key=lambda x: x[1]); rows = [x[2] for x in sel]; infos = [x[3] for x in sel]
    if a.pnc_dir:
        pnc = PncIndex(a.pnc_dir)
        for r, info in zip(rows, infos):
            if r["lang"] == "English": apply_en_pnc(r, pnc, a.pnc_min_ratio, info)
    _write_atomic(a.out, lambda f: f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tok_file = getattr(tok, "path", None); card = os.path.join(a.manifests, "original", a.set, "dataset.json")
    card_sha = (_load_json(card).get("fingerprint") or {}).get("tokenizer_json_sha256") if os.path.isfile(card) else None
    stats = dict(set=a.set, id_prefix=a.id_prefix, mode=a.mode, n_requested=a.n, seed=a.seed, aligned=aligned, streams=streams, out=a.out,
                 roots=roots, pnc_dir=a.pnc_dir, check_audio=dict(arg=a.check_audio, checked=check),
                 tokenizer=dict(spec=a.tokenizer, file=tok_file, sha256=_sha256(tok_file) if tok_file else None, manifest_tokenizer_json_sha256=card_sha),
                 population=dict(aligned_rows=counts["aligned_rows"], streams_rows=counts["streams_rows"], streams_not_aligned=counts["streams_not_aligned"],
                                 candidates=counts["candidates"], valid=counts["valid"], dropped=dict(drops), drop_examples=drop_ex),
                 shortfall=max(0, a.n - len(rows)) if a.n > 0 else 0, warnings=warnings, sample=sample_stats(rows, infos), seconds=round(time.time() - t0, 1))
    _write_atomic(a.stats, lambda f: json.dump(stats, f, ensure_ascii=False, indent=1))
    print(f"[semcommit_build_words] {a.set} prefix={a.id_prefix!r} mode={a.mode} candidates={counts['candidates']} valid={counts['valid']} "
          f"dropped={dict(drops)} kept={len(rows)} words={stats['sample']['words']} hours={stats['sample']['hours']} → {a.out}", file=sys.stderr)
    for w in warnings: print(f"[semcommit_build_words] WARNING {w}", file=sys.stderr)
    if stats["shortfall"]: print(f"[semcommit_build_words] WARNING shortfall {stats['shortfall']} (valid streams < n)", file=sys.stderr)
    if not rows: raise SystemExit(f"no valid streams among {counts['candidates']} candidates (dropped={dict(drops)}) — see {a.stats}")
    return stats

if __name__ == "__main__":
    main()

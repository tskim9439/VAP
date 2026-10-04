"""Command line interface.

  cstbench taxi-manifest  --root <TAXI root> --out <dir>/manifest.jsonl
  cstbench prepare-audio  --manifest <manifest.jsonl> --root <corpus root> --out <audio dir> [--sr 16000]
  cstbench render         --manifest <manifest.jsonl> --audio-root <audio dir> --config <config.json> --out <dir>
  cstbench verify         --out <rendered dir> --reference <checksums.json>
  cstbench build-taxi     --root <TAXI root> --out <dir> [--config configs/taxi_natural.json ...]
  cstbench eval           --sessions <rendered dir>/sessions.jsonl --hyp <hyp.jsonl> [--out report.json]

`render` writes per session <out>/<session>/{mix.wav, 2ch.wav, timeline.json}, <out>/sessions.jsonl and
<out>/build_info.json (package version, config, per-session timeline digest and audio SHA-256).
`verify` checks timeline digests exactly (they depend only on the inputs, config and seed) and reports
audio hash differences separately (they can differ across numerical library versions).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import __version__
from .audio import read_mono, resample, write_pcm16
from .schema import Session
from .timeline import overlap_ratio, place_turns, render, session_seed, trim_silence


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def cmd_taxi_manifest(a):
    from .corpora.taxi import EXPECTED, iter_sessions, stats
    sessions = list(iter_sessions(a.root))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for s in sessions:
            f.write(s.to_json() + "\n")
    st = stats(sessions)
    print(json.dumps(st, ensure_ascii=False))
    if any(st[k] != v for k, v in EXPECTED.items()):
        print(f"WARNING: counts differ from TAXI 2.5 as distributed by BAS {EXPECTED}; check your download", file=sys.stderr)


def cmd_prepare_audio(a):
    n = 0
    for line in open(a.manifest):
        s = Session.from_json(line)
        for t in s.usable_turns():
            dst = Path(a.out) / t.audio
            if dst.exists():
                continue
            x, sr = read_mono(Path(a.root) / t.audio)
            write_pcm16(dst, resample(x, sr, a.sr), a.sr)
            n += 1
    print(json.dumps({"written": n, "sample_rate": a.sr}))


def cmd_render(a):
    cfg = json.loads(Path(a.config).read_text())
    sr = int(cfg["sample_rate"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs, info, total = [], {}, 0.0
    for line in open(a.manifest):
        s = Session.from_json(line)
        turns = s.usable_turns()
        if not turns:
            continue
        audio, trims = {}, {}
        for t in turns:
            x, rate = read_mono(Path(a.audio_root) / t.audio)
            if rate != sr:
                raise ValueError(f"{t.audio}: sample rate {rate} != config {sr} (run prepare-audio first)")
            tc = cfg.get("trim", {"enabled": False})
            if tc.get("enabled"):
                i, j = trim_silence(x, sr, tc["frame_s"], tc["rel"], tc["margin_s"])
                trims[t.turn_id] = [round(i / sr, 4), round((len(x) - j) / sr, 4)]
                x = x[i:j]
            audio[t.turn_id] = x
            t.duration_s = round(len(x) / sr, 4)
        seed = session_seed(s.session_id, cfg.get("seed", 0))
        placed = place_turns(turns, cfg["gap"], tuple(cfg["same_speaker_pause"]), seed, cfg["lead_s"])
        speakers = list(s.speakers)
        mix, chans = render(placed, audio, speakers, sr, cfg["tail_s"])
        d = out / s.session_id.split("/")[-1]
        write_pcm16(d / "mix.wav", mix, sr)
        write_pcm16(d / "2ch.wav", chans, sr)
        by = {t.turn_id: t for t in turns}
        tl = [dict(turn_id=q.turn_id, speaker=q.speaker, lang=by[q.turn_id].lang, start_s=q.start_s, end_s=q.end_s,
                   gap_before_s=q.gap_before_s, transcript=by[q.turn_id].transcript,
                   translation=by[q.turn_id].translation.text if by[q.turn_id].translation else None,
                   translation_lang=by[q.turn_id].translation.lang if by[q.turn_id].translation else None,
                   trimmed_s=trims.get(q.turn_id)) for q in placed]
        rec = dict(session_id=s.session_id, corpus=s.corpus, redistributable=s.redistributable, speakers=s.speakers,
                   channels=speakers, sample_rate=sr, config=cfg["name"], seed=seed,
                   duration_s=round(len(mix) / sr, 3), overlap_ratio=round(overlap_ratio(placed), 4),
                   skipped=[t.turn_id for t in s.turns if not t.usable],
                   mix=f"{d.name}/mix.wav", two_channel=f"{d.name}/2ch.wav", turns=tl)
        (d / "timeline.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1, sort_keys=True))
        info[s.session_id] = {"timeline": _digest(rec), "mix": _sha256_file(d / "mix.wav"), "2ch": _sha256_file(d / "2ch.wav")}
        recs.append(rec)
        total += rec["duration_s"]
    with (out / "sessions.jsonl").open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    build = dict(cstbench_version=__version__, config=cfg, sessions=len(recs), hours=round(total / 3600, 3), checksums=info)
    (out / "build_info.json").write_text(json.dumps(build, ensure_ascii=False, indent=1, sort_keys=True))
    print(json.dumps({k: build[k] for k in ("sessions", "hours")} | {"config": cfg["name"]}, ensure_ascii=False))


def cmd_verify(a):
    got = json.loads((Path(a.out) / "build_info.json").read_text())["checksums"]
    ref = json.loads(Path(a.reference).read_text())
    ref = ref.get("checksums", ref)
    missing = sorted(set(ref) - set(got))
    tl_bad = sorted(k for k in ref if k in got and got[k]["timeline"] != ref[k]["timeline"])
    au_bad = sorted(k for k in ref if k in got and (got[k]["mix"], got[k]["2ch"]) != (ref[k]["mix"], ref[k]["2ch"]))
    print(json.dumps({"sessions": len(ref), "missing": missing, "timeline_mismatch": tl_bad,
                      "audio_hash_mismatch": len(au_bad)}, ensure_ascii=False))
    if missing or tl_bad:
        sys.exit(1)


def cmd_build_taxi(a):
    out = Path(a.out)
    man = out / "manifest.jsonl"
    cmd_taxi_manifest(argparse.Namespace(root=a.root, out=str(man)))
    audio = out / f"audio{a.sr // 1000}k"
    cmd_prepare_audio(argparse.Namespace(manifest=str(man), root=a.root, out=str(audio), sr=a.sr))
    for c in a.config:
        name = json.loads(Path(c).read_text())["name"]
        cmd_render(argparse.Namespace(manifest=str(man), audio_root=str(audio), config=c, out=str(out / name)))


def cmd_eval(a):
    from .evaluate import evaluate, load_jsonl
    report, segments = evaluate(load_jsonl(a.sessions), load_jsonl(a.hyp), a.time_key)
    report.update(sessions=a.sessions, hyp=a.hyp, time_key=a.time_key, cstbench_version=__version__)
    text = json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True)
    print(text)
    if a.out:
        Path(a.out).write_text(text)
    if a.segments_out:
        with open(a.segments_out, "w", encoding="utf-8") as f:
            for x in segments:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cstbench", description=__doc__.split("\n\n")[0])
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("taxi-manifest"); q.add_argument("--root", required=True); q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_taxi_manifest)
    q = sub.add_parser("prepare-audio"); q.add_argument("--manifest", required=True); q.add_argument("--root", required=True)
    q.add_argument("--out", required=True); q.add_argument("--sr", type=int, default=16000); q.set_defaults(fn=cmd_prepare_audio)
    q = sub.add_parser("render"); q.add_argument("--manifest", required=True); q.add_argument("--audio-root", required=True)
    q.add_argument("--config", required=True); q.add_argument("--out", required=True); q.set_defaults(fn=cmd_render)
    q = sub.add_parser("verify"); q.add_argument("--out", required=True); q.add_argument("--reference", required=True)
    q.set_defaults(fn=cmd_verify)
    q = sub.add_parser("build-taxi"); q.add_argument("--root", required=True); q.add_argument("--out", required=True)
    q.add_argument("--sr", type=int, default=16000)
    q.add_argument("--config", nargs="+", default=[str(Path(__file__).resolve().parents[2] / "configs" / f) for f in ("taxi_natural.json", "taxi_mediated.json")])
    q.set_defaults(fn=cmd_build_taxi)
    q = sub.add_parser("eval"); q.add_argument("--sessions", required=True); q.add_argument("--hyp", required=True)
    q.add_argument("--out"); q.add_argument("--segments-out", help="per-turn src/hyp/ref/latency (e.g. for COMET)")
    q.add_argument("--time-key", default="t", help="hyp field with the emission time (s)"); q.set_defaults(fn=cmd_eval)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

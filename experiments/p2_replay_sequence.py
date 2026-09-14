"""Replay historical P-mode diagnostic records, not the Phase 2 training loader.

Current C-mode/registry rules live in the canonical wiki plan. Never approve EOT.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import wave


def run(source, tokenizer, output):
    from tokenizers import Tokenizer

    if output.exists():
        raise FileExistsError(output)
    meta = json.loads((source / "summary.json").read_text())
    blocks = json.loads((source / "blocks.json").read_text())
    words = json.loads((source / "words.json").read_text())
    tok = Tokenizer.from_file(str(tokenizer))
    assert hashlib.sha256(tokenizer.read_bytes()).hexdigest() == meta["tokenizer_sha256"]
    tok.add_special_tokens(list(meta["provisional_special_ids"]))
    for name, idx in meta["provisional_special_ids"].items():
        assert tok.token_to_id(name) == idx
    for word in words:
        assert tok.encode(" " + word["normalized"], add_special_tokens=False).ids == word["ids"]
    with wave.open(str(source / "mono-crop.wav")) as wav:
        rate = wav.getframerate()
        assert rate == 16000 and wav.getnchannels() == 1 and wav.getsampwidth() == 2
        assert wav.getnframes() == len(blocks) * 1280
        pcm = wav.readframes(wav.getnframes())
    flat, rebuilt, candidates = [], [], []
    selector = None
    next_id = tok.token_to_id("<NEXT_AUDIO>")
    recovered, expected = defaultdict(list), defaultdict(list)
    for block in blocks:
        k = block["chunk"]
        assert round(block["start_s"] * rate) == round(meta["start_s"] * rate) + k * 1280
        flat.append(dict(kind="audio", chunk=k, token_id=None))
        ids, pieces = [], []

        def emit(kind, idx, text):
            ids.append(idx)
            pieces.append(text)
            flat.append(dict(kind=kind, chunk=k, token_id=idx))

        for event in block["events"]:
            slot, kind = event["slot"], event["kind"]
            if selector != slot or kind != "text":
                selector = slot
                tag = f"<SPK_{slot}>"
                emit("speaker", tok.token_to_id(tag), tag)
            for idx in event["ids"]:
                emit(kind, idx, tok.decode([idx]))
            if kind == "text":
                expected[slot].extend(event["ids"])
            if kind == "eot":
                candidates.append(dict(
                    candidate_id=f"ami:ES2002a:{event['agent']}:{event['ref_sample']}",
                    source_speaker=event["agent"], slot=slot, chunk=k,
                    reference_s=event["ref_s"], target_emit_s=block["end_s"],
                    label_observed_until_s=event["label_observed_until_s"],
                    reason=event["reason"], provenance="weak_auto",
                    review_status="unreviewed", human_verdict=None))
        emit("next", next_id, "<NEXT_AUDIO>")
        assert ids == block["ids"], f"round-trip mismatch {k}"
        rebuilt.append(dict(chunk=k, start_s=block["start_s"], end_s=block["end_s"],
                            activity_proxy=block["activity"], token_ids=ids,
                            output=tok.decode(ids, skip_special_tokens=False)))

    # Independently parse reconstructed IDs, including explicit event ownership.
    selector = None
    slot_ids = {idx: int(name[5:-1]) for name, idx in meta["provisional_special_ids"].items()
                if name.startswith("<SPK_")}
    event_ids = {tok.token_to_id("<ONSET>"), tok.token_to_id("<EOT>")}
    for block in rebuilt:
        ids = block["token_ids"]
        assert ids.count(next_id) == 1 and ids[-1] == next_id
        for i, idx in enumerate(ids):
            if idx in slot_ids:
                selector = slot_ids[idx]
                assert i + 1 < len(ids) and ids[i + 1] != next_id
            elif idx in event_ids:
                assert i > 0 and ids[i - 1] in slot_ids
            elif idx != next_id:
                assert selector is not None
                recovered[selector].append(idx)
    assert dict(recovered) == dict(expected)
    for i, position in enumerate(flat):
        nxt = flat[i + 1] if i + 1 < len(flat) else None
        position["diagnostic_next_token_label"] = nxt["token_id"] if nxt and nxt["kind"] != "audio" else -100
    counts = Counter(p["kind"] for p in flat)
    manifest = dict(
        source=str(source.resolve()), audio=str((source / "mono-crop.wav").resolve()),
        source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                       for name in ("summary.json", "blocks.json", "words.json", "mono-crop.wav")},
        crop_pcm_sha256=hashlib.sha256(pcm).hexdigest(), start_s=meta["start_s"],
        end_s=meta["end_s"], block_count=len(blocks), position_count=len(flat),
        token_counts=dict(counts), next_only=sum(len(b["token_ids"]) == 1 for b in rebuilt),
        activity_proxy_counts=dict(Counter(sum(b["activity_proxy"]) for b in rebuilt)),
        eot_candidates=len(candidates), reviewed_eot=0, event_complete=False,
        approved_for_joint_training=False, model_inference=False, encoder_features=False,
        registry_status="provisional", tn_parity="not_verified",
        pending_records=meta["pending_records"],
        checks=["real_PCM_length", "tokenizer_hash", "word_retokenization", "all_block_ID_round_trip",
                "speaker_transcript_round_trip", "explicit_event_selector", "one_NEXT_per_block"],
        note="Diagnostic weak-target sequence only. No human review or new EOT derivation.")
    output.mkdir(parents=True)
    for name, value in (("blocks.json", rebuilt), ("sequence.json", flat),
                        ("eot-review-candidates.json", candidates), ("manifest.json", manifest)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 실제 AMI 시퀀스: 480개 블록", "",
             "원 음성 ES2002a 54.40–92.80초. 자동 EOT는 전부 미검수 후보이며 학습 승인되지 않았다.",
             "activity는 음향 VAD가 아닌 원 발화 주석 대용값. AUDIO는 실제 PCM에 대응하는 soft-token 자리이며 encoder 미실행.",
             "", "| k | 입력 구간(원 녹음 초) | 활동 1/2/3/4 | 출력 후보 |", "|---:|---|---|---|"]
    for b in rebuilt:
        value = b["output"].replace("|", "\\|")
        lines.append(f"| {b['chunk']} | {b['start_s']:.2f}–{b['end_s']:.2f} | "
                     + "".join(map(str, b["activity_proxy"])) + f" | `{value}` |")
    (output / "blocks.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.tokenizer, args.output)

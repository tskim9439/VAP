"""AMI annotation -> auditable Phase 2 target sequence; not a trained decoder.

Historical P-mode probe, NOT the canonical Phase 2 serializer or training loader.
Current Q1 C-mode and registry contract live in the canonical Phase 2 wiki plan.

Reads original ZIP/WAV/tokenizer without modifying them. Writes a fresh probe
directory only. Segment intervals are activity proxies, NOT acoustic VAD gold.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import wave
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

SR = 16000
CHUNK = 1280


def sample(seconds):
    return round(float(seconds) * SR)


def norm_word(text):
    # Explicit restricted TN subset for this inspection, not a TN-v1 claim.
    if re.search(r"[0-9]", text):
        raise ValueError("numeric normalization needs the project TN pipeline")
    return " ".join(re.sub(r"[^a-z']+", " ", text.lower()).split())


def build(args):
    import numpy as np
    from tokenizers import Tokenizer

    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    start, end = sample(args.start), sample(args.end)
    assert start % CHUNK == end % CHUNK == 0 and end > start
    count = (end - start) // CHUNK
    z = zipfile.ZipFile(args.annotations)
    meetings = ET.fromstring(z.read("corpusResources/meetings.xml"))
    meeting = next(m for m in meetings if m.get("observation") == args.meeting)
    channels = {s.get("nxt_agent"): int(s.get("channel")) for s in meeting}
    segments, words, source_hashes = [], [], {}
    for agent in channels:
        for kind in ("words", "segments"):
            member = f"{kind}/{args.meeting}.{agent}.{kind}.xml"
            raw = z.read(member)
            source_hashes[member] = hashlib.sha256(raw).hexdigest()
            root = ET.fromstring(raw)
            for el in root:
                if kind == "segments":
                    segments.append(dict(agent=agent,
                        start=sample(el.get("transcriber_start")),
                        end=sample(el.get("transcriber_end"))))
                elif el.tag == "w" and el.get("punc") != "true" and el.get("starttime") and el.text:
                    words.append(dict(agent=agent, start=sample(el.get("starttime")),
                        end=sample(el.get("endtime")), raw=el.text,
                        word_id=next(v for k, v in el.attrib.items() if k.endswith("}id"))))
    # Merge adjacent annotation segments, never call them waveform-derived VAD.
    merged = []
    for agent in channels:
        items = sorted((s for s in segments if s["agent"] == agent), key=lambda s: s["start"])
        group = []
        for s in items:
            if group and s["start"] - group[-1]["end"] < sample(.25):
                group[-1]["end"] = max(group[-1]["end"], s["end"])
            else:
                group.append(dict(s))
        merged.extend(group)
    segments = sorted(merged, key=lambda s: (s["start"], s["agent"]))
    assert not any(s["start"] < start < s["end"] for s in segments), "crop begins mid-speech"
    selected = [s for s in segments if start <= s["start"] < end]
    first = {}
    for s in selected:
        first.setdefault(s["agent"], s["start"])
    order = sorted(first, key=lambda a: (first[a], a))
    slots = {agent: i + 1 for i, agent in enumerate(order)}
    assert len(slots) <= args.slots

    tok = Tokenizer.from_file(args.tokenizer)
    before_vocab = tok.get_vocab_size()
    specials = [f"<SPK_{i}>" for i in range(1, args.slots + 1)] + ["<ONSET>", "<EOT>"]
    tok.add_special_tokens(specials)  # in-memory probe registry ONLY
    for name in ("<NEXT_AUDIO>", f"<DELAY_{args.delay}>"):
        assert tok.token_to_id(name) is not None, name
    records, excluded, ordinal = [], [], 0

    def add(kind, agent, ref, chunk, text, ids, reason="", observed=None):
        nonlocal ordinal
        records.append(dict(kind=kind, agent=agent, slot=slots[agent], ref_sample=ref,
            ref_s=ref/SR, chunk=chunk, text=text, ids=ids, reason=reason,
            label_observed_until_s=observed/SR if observed is not None else None,
            ordinal=ordinal))
        ordinal += 1

    def due(t, delay):
        return (t-start)//CHUNK + delay

    selected_words = []
    last_end = defaultdict(int)
    for w in sorted(words, key=lambda w: (w["start"], w["end"], w["agent"])):
        if start <= w["start"] and w["end"] <= end and w["agent"] in slots:
            text = norm_word(w["raw"])
            if not text:
                continue
            # Preserve per-speaker sequence if timestamp intervals overlap.
            when = max(w["end"], last_end[w["agent"]])
            last_end[w["agent"]] = when
            encoding = tok.encode(" " + text, add_special_tokens=False)
            selected_words.append(dict(w, normalized=text, ids=encoding.ids))
            add("text", w["agent"], w["end"], due(when,args.delay),
                " " + text, encoding.ids, reason=w["word_id"])

    decisions = []
    for s in selected:
        agent, on, off = s["agent"], s["start"], s["end"]
        add("onset", agent, on, due(on,args.onset_delay), "<ONSET>",
            [tok.token_to_id("<ONSET>")], "annotation_segment_start", on)
        # Full annotation suffix is available for LABELS, not model inputs.
        horizon = off + sample(3)
        later_same = [v["start"] for v in segments if v["agent"] == agent and v["start"] >= off]
        resumed = min(later_same, default=10**18)
        others = [v for v in segments if v["agent"] != agent and v["start"] < horizon and v["end"] > off]
        continuing = [v for v in others if v["start"] <= on and v["end"] > off]
        reason = None
        if off-on <= sample(1) and continuing:
            reason = "short_overlap_continuation_no_eot"  # BC-like timing, no semantic claim
        elif any(v["start"] < off and v["end"] > off for v in others) and resumed > horizon:
            reason = "overlap_end"
        else:
            next_other = min((v["start"] for v in others if v["start"] >= off), default=10**18)
            if next_other < min(resumed, horizon):
                reason = "shift"
            elif resumed < horizon:
                reason = "same_speaker_resume_no_eot"
            elif not others:
                reason = "silence_timeout"
            else:
                reason = "uncertain_no_eot"
        decision = dict(agent=agent, slot=slots[agent], start_s=on/SR, end_s=off/SR,
                        reason=reason, label_observed_until_s=horizon/SR)
        decisions.append(decision)
        if reason not in ("overlap_end", "shift", "silence_timeout"):
            continue
        if reason == "silence_timeout":
            k = math.ceil((horizon-start)/CHUNK)-1
        else:
            k = due(off,args.delay)
        own = [r["chunk"] for r in records if r["kind"]=="text" and r["agent"]==agent
               and on <= r["ref_sample"] <= off]
        k = max([k] + own)
        if resumed < 10**18 and k >= due(resumed,args.onset_delay):
            excluded.append(dict(decision, reason="eot_next_onset_order_conflict"))
            continue
        add("eot", agent, off, k, "<EOT>", [tok.token_to_id("<EOT>")], reason,horizon)

    # Read actual four aligned headset channels; fixed mean mix (no future gain stats).
    audio = []
    wave_info = []
    for agent in order:
        p = Path(args.audio)/args.meeting/f"{args.meeting}.Headset-{channels[agent]}.wav"
        with wave.open(str(p)) as wav:
            assert wav.getframerate()==SR and wav.getnchannels()==1 and wav.getsampwidth()==2
            assert end+sample(3) <= wav.getnframes(), "label suffix exceeds recording"
            wav.setpos(start)
            pcm = wav.readframes(end-start)
            x=np.frombuffer(pcm,dtype="<i2").astype(np.float32)/32768
            assert len(x)==end-start
            audio.append(x)
            wave_info.append(dict(agent=agent, channel=channels[agent], path=str(p),
                crop_pcm_sha256=hashlib.sha256(pcm).hexdigest(), frames=wav.getnframes()))
    mixed=np.mean(audio,axis=0)
    ordered=sorted(records,key=lambda r:(r["chunk"],r["ref_sample"],r["slot"],
                   {"onset":0,"text":1,"eot":2}[r["kind"]],r["ordinal"]))
    pending=[r for r in ordered if r["chunk"]>=count]
    bins=defaultdict(list)
    for r in ordered:
        if 0<=r["chunk"]<count:
            bins[r["chunk"]].append(r)
    blocks, flat, decoded = [], [], defaultdict(list)
    current=None
    for k in range(count):
        lo,hi=start+k*CHUNK,start+(k+1)*CHUNK
        activity=[int(any(s["agent"]==a and s["start"]<hi and s["end"]>lo for s in segments)) for a in order]
        ids, pieces, kinds=[],[],[]
        flat.append(dict(type="audio",chunk=k,id=None))
        for r in bins[k]:
            if current!=r["slot"] or r["kind"]!="text":
                tag=f'<SPK_{r["slot"]}>'
                ids.append(tok.token_to_id(tag)); pieces.append(tag); kinds.append("speaker")
                current=r["slot"]
            ids.extend(r["ids"]); pieces.append(r["text"]); kinds.extend([r["kind"]]*len(r["ids"]))
            if r["kind"]=="text": decoded[current].extend(r["ids"])
        ids.append(tok.token_to_id("<NEXT_AUDIO>")); pieces.append("<NEXT_AUDIO>"); kinds.append("next")
        for tid,kind in zip(ids,kinds): flat.append(dict(type=kind,chunk=k,id=tid))
        rms=float(np.sqrt(np.mean(mixed[k*CHUNK:(k+1)*CHUNK]**2)))
        blocks.append(dict(chunk=k,start_s=lo/SR,end_s=hi/SR,activity=activity,
            audio_rms=rms,output=" ".join(pieces),ids=ids,output_tokens=len(ids),
            events=bins[k]))
    expected=defaultdict(list)
    for r in ordered:
        if r["kind"]=="text" and 0<=r["chunk"]<count: expected[r["slot"]].extend(r["ids"])
    assert dict(decoded)==dict(expected), "speaker token round-trip failed"
    weights={"text":1.,"speaker":1.,"onset":1.,"eot":2.,"next":.3}
    for i,item in enumerate(flat):
        nxt=flat[i+1] if i+1<len(flat) else None
        item["next_token_label"] = nxt["id"] if nxt and nxt["type"]!="audio" else -100
        item["loss_weight"] = weights[nxt["type"]] if nxt and nxt["type"]!="audio" else 0.
    assert all(b["ids"][-1]==tok.token_to_id("<NEXT_AUDIO>") for b in blocks)
    assert all(b["output_tokens"]<=32 for b in blocks), "probe safety cap 32 exceeded"
    assert all(r["chunk"] >= due(r["ref_sample"],args.delay) for r in ordered if r["kind"]=="text")
    summary=dict(meeting=args.meeting,start_s=start/SR,end_s=end/SR,blocks=count,
        slots=slots,channels=channels,delta_text=args.delay,delta_onset=args.onset_delay,
        tokenizer_sha256=hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest(),
        source_xml_sha256=source_hashes,wave_info=wave_info,
        vocabulary_before=before_vocab,provisional_special_ids={s:tok.token_to_id(s) for s in specials},
        activity_proxy="AMI transcriber segments, gaps <250ms merged; NOT acoustic VAD",
        normalization="restricted EN lowercase/punctuation subset; not project TN-v1 parity verified",
        target_only=True,encoder_features_extracted=False,model_inference=False,
        chunk_counts=dict(Counter(sum(b["activity"]) for b in blocks)),
        emitted_kind_counts=dict(Counter(r["kind"] for r in ordered if 0<=r["chunk"]<count)),
        max_output_tokens=max(b["output_tokens"] for b in blocks),
        over_cap8_chunks=sum(b["output_tokens"]>8 for b in blocks),
        pending_records=pending,excluded_events=excluded,
        transcripts={str(k):tok.decode(v) for k,v in decoded.items()},
        checks=["speaker_token_round_trip","one_NEXT_per_block","lexical_delay_bound","cap32","WAV_crop_lengths"],
        limitations=["annotation-defined silence/overlap", "heuristic EOT, no semantic gold",
                     "no acoustic encoder or live decoder", "provisional special IDs",
                     "mid-conversation end: pending targets carry, no fake EOF flush"])
    out.mkdir(parents=True)
    for name,data in (("summary.json",summary),("words.json",selected_words),("events.json",decisions),
                      ("blocks.json",blocks),("training-sequence.json",flat)):
        (out/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    with (out/"blocks.tsv").open("w") as f:
        writer=csv.writer(f,delimiter="\t")
        writer.writerow(["chunk","start_s","end_s","activity_proxy","target_output","token_count","audio_rms"])
        for b in blocks: writer.writerow([b["chunk"],f'{b["start_s"]:.2f}',f'{b["end_s"]:.2f}',
            "".join(map(str,b["activity"])),b["output"],b["output_tokens"],f'{b["audio_rms"]:.6f}'])
    with wave.open(str(out/"mono-crop.wav"),"wb") as wav:
        wav.setnchannels(1);wav.setsampwidth(2);wav.setframerate(SR)
        wav.writeframes(np.clip(np.rint(mixed*32768),-32768,32767).astype("<i2").tobytes())
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--audio",required=True);p.add_argument("--annotations",required=True)
    p.add_argument("--tokenizer",required=True);p.add_argument("--output",required=True)
    p.add_argument("--meeting",default="ES2002a");p.add_argument("--start",default="54.40")
    p.add_argument("--end",default="92.80");p.add_argument("--slots",type=int,default=4)
    p.add_argument("--delay",type=int,default=4);p.add_argument("--onset-delay",type=int,default=2)
    build(p.parse_args())

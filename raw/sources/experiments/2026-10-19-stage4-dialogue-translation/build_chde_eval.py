# CallHome 독일어 held-out 통화(split v3, 12 통화)의 감독 발화 671 개 → 단일 화자 평가 셋(eval_single_turn_asr 형식, 2026-10-20).
# 참조는 두 교사(Qwen3-ASR-1.7B·Whisper large-v3)가 WER < 3 % 로 맞은 의사 전사(asr-tn-de)다 — 은 표준이라 절대값보다 run 간 비교에 쓴다.
# python3 build_chde_eval.py <callhome-de-words-heldout.jsonl> <out dir>   → <out>/audio/*.wav(발화 구간 ±0.15 s, 16 kHz PCM16) + manifest.jsonl. 지우지 않는다.
import json, os, sys
import soundfile as sf
src, out = sys.argv[1], sys.argv[2]; os.makedirs(os.path.join(out, "audio"), exist_ok=True); man = os.path.join(out, "manifest.jsonl")
assert not os.path.exists(man), f"{man} 가 이미 있다"
rows = [json.loads(l) for l in open(src, encoding="utf-8")]; n = 0
with open(man + ".tmp", "w", encoding="utf-8") as f:
    for r in rows:
        s = r["segments"][0]; x, sr = sf.read(s["path"], start=int(round(s["src_offset_s"] * 16000)), frames=int(round(s["dur_s"] * 16000)), dtype="int16"); assert sr == 16000
        p = os.path.join(out, "audio", r["id"] + ".wav"); sf.write(p, x, 16000, subtype="PCM_16")
        f.write(json.dumps(dict(id=f"callhome-de/{r['id']}", dataset="callhome-de-heldout", lang="German", path=p, raw_reference=r["text"], reference=r["text"], sample_rate=16000,
                                num_samples=len(x), audio_s=round(len(x) / 16000, 4), audio_format="wav", conv_id=r["conv_id"]), ensure_ascii=False) + "\n"); n += 1
os.replace(man + ".tmp", man); print(f"{n} 발화 → {man}")

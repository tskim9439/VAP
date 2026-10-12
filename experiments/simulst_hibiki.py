"""Hibiki-Zero(Kyutai, 3B, X→영어) 베이스라인 실행기 — hibiki env(scripts/setup-baseline-envs.sh hibiki; PYTHONPATH 에 code/hibiki-zero).
공통 규약: experiments/simulst_common.py.

공식 `hibiki-zero generate`(hibiki_zero/run.py)와 같은 경로·설정: fp16, seed 42, config 의 lm_gen_config(샘플링 temp 0.8, top_k 250),
max_delay 프레임 warmup 후 Mimi 코드(12.5 Hz)를 한 프레임씩 넣는 배치 생성, 입력 끝 EOS 표시(add_input_eos), 텍스트 EOS 까지.
바꾼 것: 입력은 원 발화 + 꼬리 무음(16 kHz → Mimi 24 kHz 리샘플), 생성 길이는 배치 최장 입력 + --extra-s(공식 120 s 고정 대신).
지연: 생성 단계 s 에서 나온 텍스트 토큰의 방출 시각 = (s + 1) × 80 ms(그때까지 넣은 입력 프레임). 텍스트는 스트리밍 텍스트 출력(inner monologue)이다.

usage: python experiments/simulst_hibiki.py --manifest M [M ...] --out DIR [--batch 32] [--rank r --world n]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulst_common as C  # noqa: E402

SYSTEM = "hibiki-zero"
M = "/soundai/users/tskim/VAPKT-data/vapkt-baselines/models/kyutai__hibiki-zero-3b-pytorch-bf16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="*", default=["fleurs-de-en", "covost2-de-en"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--extra-s", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tail-s", type=float, default=C.TAIL_S)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    a = ap.parse_args()
    setting = f"default-seed{a.seed}"
    os.makedirs(a.out, exist_ok=True)
    dest = os.path.join(a.out, f"predictions-{setting}-r{a.rank}of{a.world}.jsonl")
    rows = C.shard(C.load_rows(a.manifest, a.datasets), a.rank, a.world)
    assert all(r["tgt_lang"] == "English" for r in rows), "Hibiki-Zero 는 영어로만 번역한다"
    done = C.done_ids(dest); todo = sorted((r for r in rows if r["id"] not in done), key=lambda r: (r["audio_s"], r["id"]))
    print(f"START {SYSTEM} {setting} rank={a.rank}/{a.world} rows={len(rows)} todo={len(todo)}", flush=True)
    if not todo: return
    import sphn
    import torch
    from moshi.models import loaders
    from hibiki_zero.client_utils import stack_and_pad_audio
    from hibiki_zero.inference import encode_inputs, get_lmgen, seed_all
    seed_all(a.seed)
    ci = loaders.CheckpointInfo.from_hf_repo("kyutai/hibiki-zero-3b-pytorch-bf16", moshi_weights=f"{M}/hibiki-pytorch-77f82164@110.safetensors",
                                             mimi_weights=f"{M}/mimi-pytorch-e351c8d8@125.safetensors",
                                             tokenizer=f"{M}/tokenizer_spm_48k_multi6_2.model", config_path=f"{M}/config.json")
    mimi = ci.get_mimi(device="cuda"); tok = ci.get_text_tokenizer()
    lm = ci.get_moshi(device="cuda", dtype=torch.float16, fuse_lora=True)
    sr = mimi.sample_rate; fs = 1.0 / mimi.frame_rate; eos, pad = tok.eos_id(), tok.pad_id()
    with open(dest, "a", encoding="utf-8") as f:
        for b0 in range(0, len(todo), a.batch):
            batch = todo[b0:b0 + a.batch]
            wavs = [torch.tensor(sphn.resample(C.read_wav_with_tail(r["path"], a.tail_s), C.SR, sr))[None] for r in batch]
            durs = [w.shape[-1] / sr for w in wavs]
            bw = stack_and_pad_audio(wavs, max_len=int((max(durs) + a.extra_s) * sr))
            lm_gen = get_lmgen(lm, ci, len(batch))
            codes, warm = encode_inputs(bw, mimi, lm_gen, durs)
            steps = []
            with torch.no_grad(), lm_gen.streaming(len(batch)):
                for s in range(warm.shape[-1]):
                    lm_gen.step(warm[:, :, s:s + 1])
                for s in range(codes.shape[-1]):
                    t = lm_gen.step(codes[:, :, s:s + 1])
                    steps.append(None if t is None else t[:, 0, 0].tolist())
            for i, r in enumerate(batch):
                incs = []
                for s, ids in enumerate(steps):
                    if ids is None: continue
                    tid = int(ids[i])
                    if tid == eos: break
                    if tid > pad: incs.append((tok.id_to_piece(tid).replace("▁", " "), (s + 1) * fs))
                text, times = C.words_with_times(incs)
                f.write(json.dumps(C.pred_row(SYSTEM, setting, r, text, times, a.tail_s), ensure_ascii=False) + "\n")
            f.flush()
            print(f"PROGRESS {min(b0 + a.batch, len(todo))}/{len(todo)} max_dur {max(durs):.1f}s | {text[:70]}", flush=True)
    print(f"DONE {SYSTEM} {setting} rank={a.rank}/{a.world} → {dest}", flush=True)


if __name__ == "__main__":
    main()

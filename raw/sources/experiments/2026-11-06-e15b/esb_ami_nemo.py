#!/usr/bin/env python
"""ESB AMI 단일 화자를 Nemotron 3.5 ASR 스트리밍 0.6B(우리 인코더의 원 모델, RNN-T)로 돌린다 — 같은 셋·같은 셈의 기준(2026-11-07, 사용자 "a, b, c 진행해보자" 의 a).
  python esb_ami_nemo.py --manifest <single-turn-esb-ami-v1/manifest.jsonl> --model <.nemo> --out <폴더> [--rc 3 13] [--batch-size 32] [--rank r --world-size w]
att_context_size [56, rc]: rc 3 = 0.32 s 청크(우리 인코더 설정), 13 = 1.12 s(논문 1.04 s 조건에 가깝다). 발화 전체를 넣고 chunked_limited 마스크로 문맥을 묶는다(캐시 스트리밍과 같은 문맥).
target_lang en-US(언어 태그는 떼어 낸다). 산출 predictions-rank<r>.jsonl(id·raw_reference·hypothesis·delta=nemo-rc<N>) — esb_ami_whisper.py 로 채점한다. 이어 하기 됨, 아무것도 지우지 않는다."""
import argparse, json, os, re, time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--rc", type=int, nargs="+", default=[3, 13]); ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--rank", type=int, default=0); ap.add_argument("--world-size", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch, soundfile as sf
    import nemo.collections.asr as nemo_asr
    rows = [json.loads(l) for l in open(a.manifest, encoding="utf-8") if l.strip()]
    rows = rows[a.rank::a.world_size][: a.limit or None]
    os.makedirs(a.out, exist_ok=True); dest = os.path.join(a.out, f"predictions-rank{a.rank}.jsonl")
    done = set()
    if os.path.exists(dest):
        for l in open(dest, encoding="utf-8"):
            r = json.loads(l); done.add((r["id"], r["delta"]))
    m = nemo_asr.models.ASRModel.restore_from(a.model, map_location="cpu").eval().cuda()
    print(f"모델 {type(m).__name__} · 발화 {len(rows)} · 이미 {len(done)}", flush=True)
    tag = re.compile(r"\s*<[a-z]{2}-[A-Z]{2}>\s*"); pid = int(m.cfg.model_defaults.prompt_dictionary["en-US"])
    print(f"en-US 프롬프트 {pid}", flush=True)
    with open(dest, "a", encoding="utf-8") as w:
        for rc in a.rc:
            m.encoder.set_default_att_context_size([56, rc]); dl = f"nemo-rc{rc}"
            todo = [r for r in rows if (r["id"], dl) not in done]; t0 = time.time()
            todo.sort(key=lambda r: r.get("num_samples", 0))                                       # 길이순 배치(채움 줄이기)
            for i in range(0, len(todo), a.batch_size):
                b = todo[i:i + a.batch_size]
                wav = [sf.read(r["path"], dtype="float32")[0] for r in b]                               # 16 kHz mono(매니페스트 prepare 가 확인)
                L = torch.tensor([len(x) for x in wav]); X = torch.zeros(len(b), int(L.max()))
                for k, x in enumerate(wav): X[k, : len(x)] = torch.from_numpy(x)
                with torch.inference_mode():                                                            # Lhotse 로더는 발화마다 언어 필드를 요구한다 → 직접 forward(_transcribe_forward 와 같다)
                    enc, enc_len = m.forward(input_signal=X.cuda(), input_signal_length=L.cuda(), prompt_indices=torch.full((len(b),), pid, dtype=torch.long, device="cuda"))
                    hy = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=False)
                if isinstance(hy, tuple): hy = hy[0]
                for r, h in zip(b, hy):
                    txt = tag.sub(" ", h.text if hasattr(h, "text") else str(h)).strip()
                    w.write(json.dumps(dict(id=r["id"], dataset=r["dataset"], delta=dl, raw_reference=r["raw_reference"], hypothesis=txt), ensure_ascii=False) + "\n")
                w.flush()
                if (i // a.batch_size) % 50 == 0: print(f"rc{rc} {i + len(b)}/{len(todo)} · {time.time() - t0:.0f} s", flush=True)
    print(f"DONE rank={a.rank}", flush=True)


if __name__ == "__main__":
    main()

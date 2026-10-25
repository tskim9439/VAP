#!/usr/bin/env python
"""E1 화자 분리 가지 학습(2026-10-25, 사용자 "계획대로 진행하는 데 1.7b 모델로 진행해" — wiki output-diarization-integration-20261025):
VapAsr 체크포인트(1.7B)의 인코더를 동결(fp32, 평가 경로와 같다)해 여러 층 출력을 뽑고 vapasr.hf.diar.DiarBranch(슬롯 = 도착 순서 활동)를 학습한다.

  torchrun --nproc_per_node 8 experiments/diar_train.py --init <체크포인트> --nemotron-dir <dir> --out-dir <out> \\
      [--data <phase2>] [--split <phase2>/splits/v4.json] [--corpora aihub71631,…,ami,icsi,notsofar] [--sim-data <dir>,<dir>] [--weights '패턴:가중,…'] \\
      [--window 20,90] [--window-draws 4] [--n-spk 4] [--taps odd] [--d 512 --layers 12 --heads 8 --ffn 2048] [--alpha 0.5] \\
      [--batch 8] [--lr 3e-4] [--warmup 1000] [--max-steps 20000] [--save-every 2000] [--noise-bank <dir> --noise-p 0.4 --rir-p 0.5] [--dry-run] [--smoke N]
데이터: 셋마다 CstDialogueDataset 의 창(길이 U(window), draw window-draws 개, 화자 상한 n-spk 을 넘으면 줄임)을 쓴다. 대화 셋은 split 의 held-out 을 뺀다(v4 는 AMI 공식
  dev·eval·NOTSOFAR dev·eval 도 뺀다). 합성 셋(--sim-data)은 폴더의 *.refined.dialogues.jsonl 전부 — 이름은 '<폴더>/<셋>'. 가중 키는 셋 이름, 폴더 이름(그 폴더 셋 전부),
  이름 패턴(fnmatch) 순으로 찾고 없으면 1(제출 래퍼는 '*' 없이 폴더 이름을 쓴다 — sbatch 의 따옴표 없는 확장에서 glob 이 되지 않게).
  라벨 = slot_activity(창 안 먼저 말한 순서, 80 ms), 마스크 = 패딩·전사 없는 음성 구간(_bad_spans). 증강 = 이득 ±6 dB, 잔향(실제 RIR)·잡음(noise_aug).
산출: out/checkpoint-N/{diar.pt, trainer.pt}, out/final/{diar.pt, diar.json}, out/DONE. out/PREEMPT 가 생기면 저장하고 끝낸다(sbatch 가 requeue).
재개: 마지막 완전한 checkpoint(trainer.pt 있음)에서. 설정 지문(out/config.json)이 다르면 멈춘다."""
import argparse, fnmatch, glob, json, math, os, random, sys, time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNK_S, SR = 0.08, 16000

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="인코더를 읽을 VapAsr 체크포인트(토크나이저도)"); ap.add_argument("--nemotron-dir", default=None); ap.add_argument("--out-dir", required=True)
ap.add_argument("--data", default="/soundai/users/tskim/VAPKT-data/data/phase2"); ap.add_argument("--split", default="/soundai/users/tskim/VAPKT-data/data/phase2/splits/v4.json")
ap.add_argument("--corpora", default="aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar")
ap.add_argument("--sim-data", default="", help="합성 대화 폴더(쉼표) — 폴더의 *.refined.dialogues.jsonl 전부"); ap.add_argument("--weights", default="", help="'이름패턴:가중,…'")
ap.add_argument("--window", default="20,90"); ap.add_argument("--window-draws", type=int, default=4); ap.add_argument("--n-spk", type=int, default=4)
ap.add_argument("--taps", default="odd", help="인코더 탭 층('odd'·'all'·'1,3,5') — 마지막 층은 늘 들어간다"); ap.add_argument("--d", type=int, default=512)
ap.add_argument("--layers", type=int, default=12); ap.add_argument("--heads", type=int, default=8); ap.add_argument("--ffn", type=int, default=2048); ap.add_argument("--dropout", type=float, default=0.1)
ap.add_argument("--alpha", type=float, default=0.5, help="손실 = α·도착 순서 BCE + (1 − α)·PIT"); ap.add_argument("--batch", type=int, default=8, help="GPU 당 창 수")
ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=1000); ap.add_argument("--max-steps", type=int, default=20000)
ap.add_argument("--weight-decay", type=float, default=0.01); ap.add_argument("--save-every", type=int, default=2000); ap.add_argument("--log-every", type=int, default=20)
ap.add_argument("--noise-bank", default=None); ap.add_argument("--noise-p", type=float, default=0.4); ap.add_argument("--rir-p", type=float, default=0.5); ap.add_argument("--noise-snr", default="5,30")
ap.add_argument("--gain-db", type=float, default=6.0); ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dry-run", action="store_true", help="데이터만 만들고 표본 몇 개를 점검(GPU 없음)"); ap.add_argument("--smoke", type=int, default=0, help="이 step 만 돌고 저장·재로드 대조")
a = ap.parse_args()


def log(*x):
    if int(os.environ.get("RANK", "0")) == 0: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)


# ── 데이터
class Source:
    def __init__(self, name, path, mono, tok, exclude):
        from vapasr.data.cst_dialogue import CstDialogueDataset
        lo, hi = (float(x) for x in a.window.split(","))
        self.name = name
        self.ds = CstDialogueDataset([path], tok, delays=(4,), mono_cache_dir=mono, seed=a.seed, window_s=(lo, hi), window_draws=a.window_draws, max_speakers=a.n_spk,
                                     shrink_speakers=True, exclude=exclude)
        self.hours = sum(L for _, _, L in self.ds.items) / 3600 / max(1, a.window_draws)


def build_sources(tok):
    split = json.load(open(a.split)) if a.split else {"corpora": {}}; out = []
    for c in [x for x in a.corpora.split(",") if x]:
        held = set(split["corpora"].get(c, {}).get("heldout", []))
        out.append(Source(c, os.path.join(a.data, f"{c}.refined.dialogues.jsonl"), os.path.join(a.data, "mono"), tok, held))
    for d in [x for x in a.sim_data.split(",") if x]:
        for p in sorted(glob.glob(os.path.join(d, "*.refined.dialogues.jsonl"))):
            out.append(Source(f"{os.path.basename(d.rstrip('/'))}/{os.path.basename(p).split('.')[0]}", p, os.path.join(d, "mono"), tok, None))
    W = dict((k, float(v)) for k, v in (x.rsplit(":", 1) for x in a.weights.split(",") if x))
    wts = [W.get(s.name, W.get(s.name.split("/", 1)[0], next((v for k, v in W.items() if fnmatch.fnmatch(s.name, k)), 1.0))) for s in out]   # 이름 > 폴더 이름 > 패턴
    return out, wts


def make_sample(src, i, rng, aug):
    from vapasr.data.cst_dialogue import slot_activity, window_speakers
    ds = src.ds; cid, t0, L = ds.items[i]; d = ds.dlgs[cid]; K = int(round(L / CHUNK_S))
    wav = np.asarray(ds.mono(cid)[int(round(t0 * SR)): int(round((t0 + L) * SR))], dtype=np.float32)
    if aug is not None: wav = aug(wav, rng)
    if a.gain_db > 0: wav = wav * float(10 ** (rng.uniform(-a.gain_db, a.gain_db) / 20))
    sp = window_speakers(ds.eps[cid], t0, L)[: a.n_spk]
    act = np.asarray(slot_activity(ds.eps[cid], sp, t0, L, K, n_slots=a.n_spk), dtype=np.float32).reshape(K, a.n_spk)
    m = np.ones(K, dtype=bool)
    for s, e in ds._bad_spans(d):
        x0, x1 = s - t0, e - t0
        if x1 > 0 and x0 < L: m[max(0, int(x0 / CHUNK_S)): min(K, int(math.ceil(x1 / CHUNK_S)))] = False
    return dict(wav=np.clip(wav, -1, 1), K=K, act=act, mask=m, src=src.name, n_spk=len(sp))


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, sources, weights, aug, tag):
        self.sources, self.weights, self.aug, self.tag = sources, weights, aug, tag

    def __iter__(self):
        wi = torch.utils.data.get_worker_info(); rng = random.Random(f"{self.tag}:{wi.id if wi else 0}")
        while True:
            s = rng.choices(self.sources, weights=self.weights)[0]
            yield make_sample(s, rng.randrange(len(s.ds.items)), rng, self.aug)


def collate(b):
    B, n, Km = len(b), max(len(x["wav"]) for x in b), max(x["K"] for x in b)
    wav = torch.zeros(B, n); act = torch.zeros(B, Km, a.n_spk); mask = torch.zeros(B, Km, dtype=torch.bool)
    for j, x in enumerate(b):
        wav[j, : len(x["wav"])] = torch.from_numpy(x["wav"]); act[j, : x["K"]] = torch.from_numpy(x["act"]); mask[j, : x["K"]] = torch.from_numpy(x["mask"])
    K = torch.tensor([x["K"] for x in b]); wl = torch.tensor([len(x["wav"]) for x in b])
    return dict(wav=wav, wav_len=wl, K=K, act=act, mask=mask, pad=torch.arange(Km)[None] >= K[:, None], src=[x["src"] for x in b], n_spk=[x["n_spk"] for x in b])


def fingerprint():
    keys = ("init", "data", "split", "corpora", "sim_data", "weights", "window", "window_draws", "n_spk", "taps", "d", "layers", "heads", "ffn", "dropout", "alpha", "batch",
            "lr", "warmup", "max_steps", "weight_decay", "noise_bank", "noise_p", "rir_p", "noise_snr", "gain_db", "seed")
    return {k: getattr(a, k) for k in keys}


def last_checkpoint(out):
    cks = sorted((int(p.rsplit("-", 1)[1]), p) for p in glob.glob(os.path.join(out, "checkpoint-*")) if p.rsplit("-", 1)[1].isdigit() and os.path.exists(os.path.join(p, "trainer.pt")))
    return cks[-1] if cks else (0, None)


def main():
    from vapasr.data.noise_aug import from_args as noise_from_args
    from vapasr.hf import load_tokenizer
    tok = load_tokenizer(a.init)
    sources, wts = build_sources(tok)
    for s, w in zip(sources, wts): log(f"  셋 {s.name}: 창 {len(s.ds.items)}(draw {a.window_draws}) · 약 {s.hours:.0f} h · 가중 {w:g}")
    assert sources and all(len(s.ds.items) for s in sources), "빈 셋이 있다"
    aug = noise_from_args(a.noise_bank, a.noise_p, a.noise_snr, a.rir_p) if a.noise_bank else None
    if a.dry_run:
        rng = random.Random(a.seed)
        for s in sources:
            x = make_sample(s, rng.randrange(len(s.ds.items)), rng, aug)
            log(f"  표본 {s.name}: K {x['K']} · 화자 {x['n_spk']} · 활동 몫 {x['act'].mean(0).round(3).tolist()} · 마스크 몫 {1 - x['mask'].mean():.3f} · 음량 {float(np.abs(x['wav']).max()):.3f}")
        b = collate([make_sample(sources[0], i, rng, aug) for i in range(3)]); log(f"  배치 wav {tuple(b['wav'].shape)} act {tuple(b['act'].shape)} pad {int(b['pad'].sum())}")
        print("DRY OK", flush=True); return
    import torch.distributed as dist
    from vapasr.hf.diar import DiarBranch, load_encoder, sort_pit_loss, tap_layers
    ddp = "RANK" in os.environ
    if ddp: dist.init_process_group("nccl"); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1); dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    os.makedirs(a.out_dir, exist_ok=True); fp = os.path.join(a.out_dir, "config.json")
    if rank == 0:
        if os.path.exists(fp): assert json.load(open(fp)) == fingerprint(), f"재개 설정 불일치 — 새 --out-dir 을 쓴다 ({fp})"
        else: json.dump(fingerprint(), open(fp, "w"), indent=1)
    enc, n_keys = load_encoder(a.init, a.nemotron_dir, "cuda")
    taps = tap_layers(a.taps, len(enc.enc.layers)); log(f"인코더 ← {a.init}(키 {n_keys}, 동결 fp32) · 탭 {list(taps)} · world {world}")
    torch.manual_seed(a.seed)
    model = DiarBranch(1 + len(taps), 1024, a.d, a.layers, a.heads, a.ffn, a.n_spk, a.dropout).to(dev)
    log(f"가지 {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M · {model.cfg}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.98))
    total = a.smoke or a.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))))
    step, ck = last_checkpoint(a.out_dir) if not a.smoke else (0, None)
    if ck:
        model.load_state_dict(torch.load(os.path.join(ck, "diar.pt"), map_location=dev)["state"])
        st = torch.load(os.path.join(ck, "trainer.pt"), map_location=dev, weights_only=False); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        log(f"재개 ← {ck}(step {step})")
    net = torch.nn.parallel.DistributedDataParallel(model, device_ids=[dev.index]) if ddp else model
    dl = torch.utils.data.DataLoader(Stream(sources, wts, aug, f"{a.seed}:{rank}:{step}"), batch_size=a.batch, num_workers=a.num_workers, collate_fn=collate,
                                     pin_memory=True, persistent_workers=a.num_workers > 0, prefetch_factor=4 if a.num_workers else None)

    def save(tag):
        if rank != 0: return
        d = os.path.join(a.out_dir, tag); tmp = d + f".tmp{os.getpid()}"; os.makedirs(tmp, exist_ok=True)
        torch.save(dict(state=model.state_dict(), cfg=model.cfg, taps=list(taps), init=os.path.abspath(a.init), step=step), os.path.join(tmp, "diar.pt"))
        if tag != "final": torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), step=step), os.path.join(tmp, "trainer.pt"))
        else: json.dump(dict(cfg=model.cfg, taps=list(taps), init=os.path.abspath(a.init), step=step, args=fingerprint()), open(os.path.join(tmp, "diar.json"), "w"), indent=1)
        if os.path.exists(d): d = f"{d}-dup{step}-{os.getpid()}"; log(f"!! {tag} 가 이미 있다 — 덮어쓰지 않고 {d} 로")
        os.replace(tmp, d); log(f"저장 → {d}")

    preempt = os.path.join(a.out_dir, "PREEMPT"); t0 = time.time(); acc = {}
    net.train()
    for b in dl:
        if step >= total: break
        flag = torch.tensor([1.0 if rank == 0 and os.path.exists(preempt) else 0.0], device=dev)
        if ddp: dist.all_reduce(flag)
        if flag.item() > 0: save(f"checkpoint-{step}"); log("PREEMPT → 저장하고 끝낸다"); break
        wav, wl, K = b["wav"].to(dev, non_blocking=True), b["wav_len"].to(dev), b["K"].to(dev)
        act, mask, pad = b["act"].to(dev), b["mask"].to(dev), b["pad"].to(dev)
        with torch.no_grad(): feats = enc(wav, wl, K, taps=taps)                                      # (B, 1 + 탭, K, D) fp32
        with torch.autocast("cuda", dtype=torch.bfloat16): logits = net(feats, pad)
        loss, srt, pit = sort_pit_loss(logits.float(), act, mask & ~pad, a.alpha)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); step += 1
        with torch.no_grad():
            m = (mask & ~pad).float()[..., None]; pr = (logits.float() > 0).float()
            for k, v in dict(loss=loss.item(), sort=srt.item(), pit=pit.item(), gn=float(gn), frame_acc=float(((pr == act).float() * m).sum() / (m.sum() * a.n_spk).clamp(min=1))).items():
                acc[k] = acc.get(k, 0.0) + v
        if step % a.log_every == 0 or step == total:
            n = a.log_every if step % a.log_every == 0 else (step % a.log_every or a.log_every)
            log(f"step {step}/{total} · " + " · ".join(f"{k} {v / n:.4f}" for k, v in acc.items()) + f" · lr {sched.get_last_lr()[0]:.2e} · {(time.time() - t0) / n:.2f} s/step"); acc = {}; t0 = time.time()
        if step % a.save_every == 0 and step < total: save(f"checkpoint-{step}")
    if step >= total:
        save("final")
        if rank == 0:
            if a.smoke:
                x = torch.load(os.path.join(a.out_dir, "final", "diar.pt"), map_location=dev); m2 = DiarBranch(**x["cfg"]).to(dev); m2.load_state_dict(x["state"])
                diff = max(float((p - q).abs().max()) for p, q in zip(model.state_dict().values(), m2.state_dict().values()))
                log(f"저장·재로드 대조: 최대 차이 {diff:.2e}"); assert diff == 0.0
            open(os.path.join(a.out_dir, "DONE"), "w").write(f"step {step}\n")
    if ddp: dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()

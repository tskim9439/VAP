#!/usr/bin/env python
"""G1 이식 어댑터 학습(2026-10-29, 사용자 "G1 도 구현해놓자" — wiki output-n3d-branch-20261029 4.2 절): 고정 화자 분리 전문가 Nemotron 3 Diarization(N3D)을
우리 인코더 하나로 합친다. 학생 = 우리 ASR 인코더(고정) 탭 → GraftAdapter(학습, vapasr/hf/n3d_graft) → N3D --graft-layer 층 위·머리(고정), 교사 = N3D 그대로.

  torchrun --nproc_per_node 8 experiments/n3d_graft_train.py --init <VapAsr 체크포인트(E9 final)> --n3d <.nemo> --nemotron-dir <dir> --out-dir <out> \\
      [--graft-layer 8] [--taps odd] [--blocks 3 --kernel 5 --ffn 2048] [--kd-weight 1 --hid-weight 0.1 --ref-weight 0] [--ridge-windows 256 --ridge 10] \\
      [데이터·증강 인자는 diar_train.py 와 같다] [--batch 8] [--lr 3e-4] [--warmup 500] [--max-steps 6000] [--save-every 1000] [--dry-run] [--smoke N]
학습(창 통째, 오프라인 — 교사·학생 모두 창 안 전체 어텐션):
  손실 = kd·BCE(학생 10 ms 확률, 교사 10 ms 확률 — 부드러운 타깃) + hid·‖ĥ − h_layer‖² / var(h_layer) + ref·(정답 슬롯 활동 Sort·PIT, 80 ms — 기본 끔).
  교사·학생 모두 같은 증강 오디오를 본다. 학생 m 층 위는 고정이라 gradient 는 어댑터로만 간다(N3D 는 DDP 밖).
  처음(재개 아님)에는 rank 0 이 --ridge-windows 창으로 탭 → h_layer 능형 회귀를 풀어 어댑터 입력 Linear 에 둔다(G0 선형 이식에서 시작 — 인과 블록은 0 으로 시작).
  무음 프로필: 교사 80 ms 확률이 모두 0.1 아래인 유효 프레임의 h_layer 평균(rank 합) — 스트리밍 캐시의 무음 자리(n3d_graft.stream 의 sil)로 graft.pt 에 남는다.
데이터: diar_train.py 와 같은 셋·창·가중·증강(대화 셋은 split 의 held-out 을 뺀다 — v4 는 AMI 공식 dev·eval·NOTSOFAR dev·eval 도 뺀다). 교사가 라벨을 주므로
  정답 활동은 --ref-weight 를 켤 때만 쓴다.
산출: out/checkpoint-N/{graft.pt, trainer.pt}, out/final/{graft.pt, graft.json}, out/parity.json, out/DONE. out/PREEMPT 가 생기면 저장하고 끝낸다(sbatch 가 requeue).
  graft.pt = dict(state, cfg, layer, taps, sil, init, n3d, step). 평가: experiments/diar_eval.py run --graft <out>/final/graft.pt(창·회의, 스트리밍 청크 4).
저장·재로드 대조: 학습 끝에 final 을 새 어댑터로 읽어 마지막 배치 손실을 메모리 모델과 대조한다(가중치 차이 0, 손실 차이 ≤ 1e-4 — 어긋나면 DONE 을 쓰지 않는다).
G1b 스트리밍 상태 학습(2026-10-30 — G1 첫 학생이 창 DER 에서 교사 + 6 이었다, 오프라인만 학습해 스트리밍 캐시가 분포 밖 ĥ 를 담는 것으로 본다):
  --init-graft <graft.pt>(G1 final 에서 — 무음 sil 도 그것을 고정해 쓴다) --stream-chunk N: 교사·학생 모두 청크 N 스트리밍(AOSC)으로 돌려 80 ms 확률 BCE 를 건다
  (n3d.stream_probs 대 n3d_graft.stream(grad) — gradient 는 캐시·FIFO 에 담긴 앞 청크의 ĥ 로도 흐른다). h_layer MSE 는 오프라인 교사 h 로 그대로."""
import argparse, datetime, fnmatch, glob, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNK_S, SR = 0.08, 16000

ap = argparse.ArgumentParser()
ap.add_argument("--init", required=True, help="인코더(고정)·토크나이저를 읽을 VapAsr 체크포인트"); ap.add_argument("--nemotron-dir", default=None); ap.add_argument("--out-dir", required=True)
ap.add_argument("--n3d", required=True, help="Nemotron 3 Diarization .nemo(교사·이식 위층·머리)"); ap.add_argument("--graft-layer", type=int, default=8)
ap.add_argument("--taps", default="odd", help="인코더 탭 층('odd'·'all'·'1,3,5') — 마지막 층은 늘 들어간다(E9 헤드와 같은 odd)")
ap.add_argument("--blocks", type=int, default=3); ap.add_argument("--kernel", type=int, default=5); ap.add_argument("--ffn", type=int, default=2048)
ap.add_argument("--kd-weight", type=float, default=1.0); ap.add_argument("--hid-weight", type=float, default=0.1); ap.add_argument("--ref-weight", type=float, default=0.0)
ap.add_argument("--init-graft", default=None, help="G1b: 시작 어댑터 graft.pt(능형 회귀 대신, 무음 sil 도 그것) — 재개 checkpoint 가 있으면 그것이 우선")
ap.add_argument("--stream-chunk", type=int, default=0, help="G1b: 0 이면 창 통째(오프라인) 학습, N 이면 교사·학생 모두 청크 N 스트리밍(AOSC)으로 증류 — --init-graft 와 함께")
ap.add_argument("--ridge-windows", type=int, default=256, help="처음 시작 때 능형 회귀로 입력 Linear 를 둘 창 수(0 = 기본 초기화)"); ap.add_argument("--ridge", type=float, default=10.0)
ap.add_argument("--data", default="/soundai/users/tskim/VAPKT-data/data/phase2"); ap.add_argument("--split", default="/soundai/users/tskim/VAPKT-data/data/phase2/splits/v4.json")
ap.add_argument("--corpora", default="aihub71631,aihub134-1,aihub134-2,otoSpeech,kec,ztc,nsc,ami,icsi,notsofar")
ap.add_argument("--sim-data", default="", help="합성 대화 폴더(쉼표)"); ap.add_argument("--weights", default="", help="'이름패턴:가중,…'(diar_train.py 와 같다)")
ap.add_argument("--window", default="20,40"); ap.add_argument("--window-draws", type=int, default=2); ap.add_argument("--n-spk", type=int, default=4)
ap.add_argument("--batch", type=int, default=8, help="GPU 당 창 수"); ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--warmup", type=int, default=500)
ap.add_argument("--max-steps", type=int, default=6000); ap.add_argument("--weight-decay", type=float, default=0.01); ap.add_argument("--save-every", type=int, default=1000)
ap.add_argument("--log-every", type=int, default=20)
ap.add_argument("--noise-bank", default=None); ap.add_argument("--noise-p", type=float, default=0.4); ap.add_argument("--rir-p", type=float, default=0.5); ap.add_argument("--noise-snr", default="5,30")
ap.add_argument("--gain-db", type=float, default=6.0); ap.add_argument("--num-workers", type=int, default=4); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--dry-run", action="store_true", help="데이터만 만들고 표본 몇 개를 점검(GPU 없음)"); ap.add_argument("--smoke", type=int, default=0, help="이 step 만 돌고 저장·재로드 대조")
a = ap.parse_args()
assert 0 <= a.graft_layer <= 30, "--graft-layer 0–30(N3D 31 층)"
assert a.kd_weight > 0 or a.hid_weight > 0 or a.ref_weight > 0, "손실 가중이 모두 0 이다"
assert a.stream_chunk >= 0 and (not a.stream_chunk or a.init_graft), "--stream-chunk(G1b)는 --init-graft(G1 final)와 함께 — 무음 sil 을 거기서 고정해 쓴다"


def log(*x):
    if int(os.environ.get("RANK", "0")) == 0: print(f"[{time.strftime('%H:%M:%S')}]", *x, flush=True)


# ── 데이터(diar_train.py 와 같은 규약)
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
    wts = [W.get(s.name, W.get(s.name.split("/", 1)[0], next((v for k, v in W.items() if fnmatch.fnmatch(s.name, k)), 1.0))) for s in out]
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
    return dict(wav=np.clip(wav, -1, 1), K=K, act=act, mask=m, src=src.name)


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
    return dict(wav=wav, wav_len=torch.tensor([len(x["wav"]) for x in b]), K=torch.tensor([x["K"] for x in b]), act=act, mask=mask, src=[x["src"] for x in b])


def fingerprint():
    keys = ("init", "n3d", "graft_layer", "taps", "blocks", "kernel", "ffn", "kd_weight", "hid_weight", "ref_weight", "ridge_windows", "ridge", "data", "split", "corpora",
            "sim_data", "weights", "window", "window_draws", "n_spk", "batch", "lr", "warmup", "max_steps", "weight_decay", "noise_bank", "noise_p", "rir_p", "noise_snr",
            "gain_db", "seed")
    return {**{k: getattr(a, k) for k in keys}, **({k: getattr(a, k) for k in ("init_graft", "stream_chunk")} if a.stream_chunk or a.init_graft else {})}   # 켰을 때만(G1 지문 그대로)


def last_checkpoint(out):
    cks = sorted((int(p.rsplit("-", 1)[1]), p) for p in glob.glob(os.path.join(out, "checkpoint-*")) if p.rsplit("-", 1)[1].isdigit() and os.path.exists(os.path.join(p, "trainer.pt")))
    return cks[-1] if cks else (0, None)


# ── 한 배치의 교사·학생·손실
def batch_terms(b, enc, taps, n3, adapter, dev, sil0=None):
    """→ (손실, 통계 dict, 무음 프로필 (합 (512,), 프레임 수)). 교사(N3D 오프라인)와 학생(탭 → 어댑터 → layer 층 위·머리)이 같은 오디오를 본다.
    --stream-chunk(G1b)면 확률 손실은 교사 스트리밍 대 학생 스트리밍(sil0 — 캐시 무음 자리)의 80 ms BCE 다."""
    from vapasr.hf.diar import sort_pit_loss
    from vapasr.hf.n3d import stream_probs
    from vapasr.hf.n3d_graft import pool80, stream, student_offline, teacher
    wav, wl, K = b["wav"].to(dev, non_blocking=True), b["wav_len"].to(dev), b["K"].to(dev)
    with torch.no_grad():
        f = enc(wav, wl, K, taps=taps)                                                               # (B, 1 + 탭, Kmax, 1024) fp32
        with torch.autocast("cuda", dtype=torch.bfloat16): h, p10, elen = teacher(n3, wav, wl, a.graft_layer)
    T = min(f.shape[2], h.shape[1]); L = torch.minimum(elen.to(dev).long(), K.long()).clamp(max=T)   # 우리 프레임 K 와 N3D 프레임(80 ms 격자 같음) 중 짧은 쪽
    h = h[:, :T].float(); pt = p10[:, : T * 8].float()
    v10 = (torch.arange(T * 8, device=dev)[None] < (L * 8)[:, None]).float()[..., None]               # 10 ms 유효
    v80 = (torch.arange(T, device=dev)[None] < L[:, None])                                           # 80 ms 유효
    if a.stream_chunk:                                                                               # G1b: 교사·학생 모두 스트리밍(AOSC) — 80 ms 확률
        with torch.no_grad(): t80s = stream_probs(n3, wav, wl, a.stream_chunk, 0)[:, :T].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hh = adapter(f[:, :, :T])
            s80s = stream(n3, hh.float(), L, a.graft_layer, sil0, chunk=a.stream_chunk, grad=True)
        hh = hh.float(); s80s = s80s.float(); vs = v80.float()[..., None]
        kd = (F.binary_cross_entropy(s80s.clamp(1e-6, 1 - 1e-6), t80s, reduction="none") * vs).sum() / (vs.sum() * t80s.shape[-1]).clamp(min=1)
        ps = None
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hh = adapter(f[:, :, :T])
            ps = student_offline(n3, hh, L, a.graft_layer)
        ps = ps.float(); hh = hh.float()
        kd = (F.binary_cross_entropy(ps.clamp(1e-6, 1 - 1e-6), pt, reduction="none") * v10).sum() / (v10.sum() * pt.shape[-1]).clamp(min=1)
    vh = v80.float()[..., None]; mu = (h * vh).sum((0, 1)) / vh.sum().clamp(min=1)
    var = (((h - mu) ** 2) * vh).sum() / (vh.sum() * h.shape[-1]).clamp(min=1)
    hid = (((hh - h) ** 2) * vh).sum() / (vh.sum() * h.shape[-1]).clamp(min=1) / var.clamp(min=1e-6)
    loss = a.kd_weight * kd + a.hid_weight * hid
    s80g = s80s if a.stream_chunk else pool80(ps, L); t80 = t80s if a.stream_chunk else pool80(pt, L); s80 = s80g.detach()
    st = dict(kd=float(kd), hid=float(hid), agree=float((((s80 > 0.5) == (t80 > 0.5)).all(-1) & v80).sum() / v80.sum().clamp(min=1)),
              t_act=float(((t80 > 0.5).any(-1) & v80).sum() / v80.sum().clamp(min=1)))
    if a.ref_weight > 0:                                                                             # 정답 슬롯 활동(앞 n-spk 채널, Sort·PIT)
        act = b["act"].to(dev)[:, :T]; mk = b["mask"].to(dev)[:, :T] & v80
        lg = torch.logit(s80g[..., : a.n_spk].clamp(1e-5, 1 - 1e-5)); ref, _, _ = sort_pit_loss(lg, act, mk, 0.5)
        loss = loss + a.ref_weight * ref; st["ref"] = float(ref)
    with torch.no_grad():
        sil = ((t80.max(-1).values < 0.1) & v80)
        prof = ((h * sil[..., None].float()).sum((0, 1)), float(sil.sum()))
    return loss, st, prof


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
            log(f"  표본 {s.name}: K {x['K']} · 활동 몫 {x['act'].mean(0).round(3).tolist()} · 음량 {float(np.abs(x['wav']).max()):.3f}")
        b = collate([make_sample(sources[0], i, rng, aug) for i in range(3)]); log(f"  배치 wav {tuple(b['wav'].shape)} K {b['K'].tolist()}")
        print("DRY OK", flush=True); return
    import torch.distributed as dist
    from vapasr.hf.diar import load_encoder, tap_layers
    from vapasr.hf.n3d import load_n3d
    from vapasr.hf.n3d_graft import GraftAdapter
    ddp = "RANK" in os.environ
    if ddp: dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=60)); torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))   # 능형 회귀(rank 0) 동안 기다린다
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1); dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    os.makedirs(a.out_dir, exist_ok=True); fp = os.path.join(a.out_dir, "config.json")
    if rank == 0:
        if os.path.exists(fp): assert json.load(open(fp)) == fingerprint(), f"재개 설정 불일치 — 새 --out-dir 을 쓴다 ({fp})"
        else: json.dump(fingerprint(), open(fp, "w"), indent=1)
    enc, n_keys = load_encoder(a.init, a.nemotron_dir, "cuda")
    taps = tuple(tap_layers(a.taps, len(enc.enc.layers))); n3, aux = load_n3d(a.n3d, "cuda")
    log(f"인코더 ← {a.init}(키 {n_keys}, 동결 fp32) · 탭 {list(taps)} · N3D ← {a.n3d}(보조 키 {len(aux)} 뺌, SDPA) · 이식 층 {a.graft_layer} · world {world}")
    torch.manual_seed(a.seed)
    adapter = GraftAdapter(1 + len(taps), 1024, 512, a.blocks, a.kernel, a.ffn).to(dev)
    log(f"어댑터 {sum(p.numel() for p in adapter.parameters()) / 1e6:.2f}M · {adapter.cfg}")
    opt = torch.optim.AdamW(adapter.parameters(), lr=a.lr, weight_decay=a.weight_decay, betas=(0.9, 0.98))
    total = a.smoke or a.max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total))))))
    step, ck = last_checkpoint(a.out_dir) if not a.smoke else (0, None)
    sil_sum = torch.zeros(512, device=dev, dtype=torch.float64); sil_n = torch.zeros((), device=dev, dtype=torch.float64); sil0 = None
    if a.init_graft:                                                                                 # G1b: 무음 sil 은 시작 graft.pt 의 값으로 고정(교사 h_layer 의 무음 평균 — 어댑터와 무관)
        xg = torch.load(a.init_graft, map_location=dev); sil0 = xg["sil"].float().to(dev)
        assert xg["layer"] == a.graft_layer and list(xg["taps"]) == list(taps) and xg["cfg"] == adapter.cfg, f"--init-graft 의 층·탭·어댑터 설정이 다르다: {a.init_graft}"
    if ck:
        x = torch.load(os.path.join(ck, "graft.pt"), map_location=dev); adapter.load_state_dict(x["state"])
        st = torch.load(os.path.join(ck, "trainer.pt"), map_location=dev, weights_only=False); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        if rank == 0: sil_sum += st["sil_sum"].to(dev); sil_n += st["sil_n"]                          # 무음 프로필은 rank 0 이 합을 들고 간다(저장 때 모은 값)
        log(f"재개 ← {ck}(step {step})")
    elif a.init_graft:
        adapter.load_state_dict(xg["state"]); log(f"어댑터 ← {a.init_graft}(step {xg.get('step')}) · 스트리밍 청크 {a.stream_chunk or '없음(오프라인)'}")
    elif a.ridge_windows > 0 and rank == 0:                                                          # G0 와 같은 능형 회귀 해에서 시작(블록은 0 — 붙인 직후 = 선형 이식)
        from vapasr.hf.n3d_graft import pool80, teacher
        t0 = time.time(); it = iter(Stream(sources, wts, aug, f"{a.seed}:ridge")); XtX = XtY = Xs = Ys = None; nw = nfr = 0
        with torch.no_grad():
            while nw < a.ridge_windows:
                b = collate([next(it) for _ in range(a.batch)]); nw += a.batch
                wav, wl, K = b["wav"].to(dev), b["wav_len"].to(dev), b["K"].to(dev)
                f = enc(wav, wl, K, taps=taps)
                with torch.autocast("cuda", dtype=torch.bfloat16): h, p10, elen = teacher(n3, wav, wl, a.graft_layer)
                T = min(f.shape[2], h.shape[1]); Lb = torch.minimum(elen.long(), K.long()).clamp(max=T); t80 = pool80(p10[:, : T * 8].float(), Lb)
                for j in range(f.shape[0]):
                    Lj = int(Lb[j]); x = f[j, :, :Lj].permute(1, 0, 2).reshape(Lj, -1).double(); y = h[j, :Lj].double(); nfr += Lj
                    XtX = x.T @ x if XtX is None else XtX + x.T @ x; XtY = x.T @ y if XtY is None else XtY + x.T @ y
                    Xs = x.sum(0) if Xs is None else Xs + x.sum(0); Ys = y.sum(0) if Ys is None else Ys + y.sum(0)
                    sil = t80[j, :Lj].max(-1).values < 0.1; sil_sum += y[sil].sum(0); sil_n += float(sil.sum())
        mx, my = Xs / nfr, Ys / nfr; D = XtX.shape[0]; C = XtX - nfr * torch.outer(mx, mx)
        A = torch.linalg.cholesky(C + a.ridge * nfr / D * torch.eye(D, dtype=C.dtype, device=C.device))
        W = torch.cholesky_solve(XtY - nfr * torch.outer(mx, my), A); adapter.set_linear(W.float(), (my - mx @ W).float())
        log(f"능형 회귀 시작점: 창 {nw} · 프레임 {nfr} · 차원 {D} · 무음 프레임 {int(sil_n)} · {time.time() - t0:.0f}s"); del XtX, XtY, C, A, W
    if ddp: dist.barrier()                                                                           # 다른 rank 는 능형 회귀를 기다린다(DDP 가 rank 0 가중치를 뿌린다)
    net = torch.nn.parallel.DistributedDataParallel(adapter, device_ids=[dev.index]) if ddp else adapter
    dl = torch.utils.data.DataLoader(Stream(sources, wts, aug, f"{a.seed}:{rank}:{step}"), batch_size=a.batch, num_workers=a.num_workers, collate_fn=collate,
                                     pin_memory=True, persistent_workers=a.num_workers > 0, prefetch_factor=4 if a.num_workers else None)

    def save(tag):
        s_, n_ = sil_sum.clone(), sil_n.clone()                                                        # 무음 프로필 = 모든 rank 의 합(모든 rank 가 같은 step 에서 부른다)
        if ddp: dist.all_reduce(s_); dist.all_reduce(n_)
        if rank != 0: return None
        d = os.path.join(a.out_dir, tag); tmp = d + f".tmp{os.getpid()}"; os.makedirs(tmp, exist_ok=True)
        rec = dict(state=adapter.state_dict(), cfg=adapter.cfg, layer=a.graft_layer, taps=list(taps), sil=(s_ / n_.clamp(min=1)).float().cpu(), sil_frames=float(n_),
                   init=os.path.abspath(a.init), n3d=os.path.abspath(a.n3d), step=step)
        torch.save(rec, os.path.join(tmp, "graft.pt"))
        if tag != "final": torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), step=step, sil_sum=s_.cpu(), sil_n=float(n_)), os.path.join(tmp, "trainer.pt"))
        else: json.dump(dict(cfg=adapter.cfg, layer=a.graft_layer, taps=list(taps), sil_frames=float(n_), init=rec["init"], n3d=rec["n3d"], step=step, args=fingerprint()),
                        open(os.path.join(tmp, "graft.json"), "w"), indent=1)
        if os.path.exists(d): d = f"{d}-dup{step}-{os.getpid()}"; log(f"!! {tag} 가 이미 있다 — 덮어쓰지 않고 {d} 로")
        os.replace(tmp, d); log(f"저장 → {d}"); return d

    preempt = os.path.join(a.out_dir, "PREEMPT"); t0 = time.time(); acc = {}; b = None
    adapter.train()
    for b in dl:
        if step >= total: break
        flag = torch.tensor([1.0 if rank == 0 and os.path.exists(preempt) else 0.0], device=dev)
        if ddp: dist.all_reduce(flag)
        if flag.item() > 0: save(f"checkpoint-{step}"); log("PREEMPT → 저장하고 끝낸다"); break
        loss, st, (ps_, pn_) = batch_terms(b, enc, taps, n3, net, dev, sil0)
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0); opt.step(); sched.step(); step += 1
        sil_sum += ps_.double(); sil_n += pn_
        for k, v in dict(loss=float(loss), gn=float(gn), **st).items(): acc[k] = acc.get(k, 0.0) + v
        if step % a.log_every == 0 or step == total:
            n = a.log_every if step % a.log_every == 0 else (step % a.log_every or a.log_every)
            log(f"step {step}/{total} · " + " · ".join(f"{k} {v / n:.4f}" for k, v in acc.items()) + f" · lr {sched.get_last_lr()[0]:.2e} · {(time.time() - t0) / n:.2f} s/step"); acc = {}; t0 = time.time()
        if step % a.save_every == 0 and step < total: save(f"checkpoint-{step}")
    d = save("final") if step >= total else None
    if ddp: dist.barrier(); dist.destroy_process_group()                                           # 대조는 그룹을 닫은 뒤 rank 0 만
    if d and b is not None:
        reload_check(d, b, adapter, enc, taps, n3, dev, sil0)                                       # 어긋나면 멈추고 DONE 을 쓰지 않는다
        open(os.path.join(a.out_dir, "DONE"), "w").write(f"step {step}\n")


def reload_check(final_dir, b, adapter, enc, taps, n3, dev, sil0=None):
    """final/graft.pt 를 새 어댑터로 읽어 마지막 배치의 손실을 메모리 어댑터와 대조한다(가중치 차이 0, 손실 차이 ≤ 1e-4). out/parity.json."""
    from vapasr.hf.n3d_graft import GraftAdapter
    x = torch.load(os.path.join(final_dir, "graft.pt"), map_location=dev); m2 = GraftAdapter(**x["cfg"]).to(dev); m2.load_state_dict(x["state"])
    adapter.eval(); m2.eval()
    with torch.no_grad():
        l1, s1, _ = batch_terms(b, enc, taps, n3, adapter, dev, sil0); l2, s2, _ = batch_terms(b, enc, taps, n3, m2, dev, sil0)
    diff = max(float((p - q).abs().max()) for p, q in zip(adapter.state_dict().values(), m2.state_dict().values()))
    out = dict(final=final_dir, step=x["step"], windows=len(b["K"]), memory=dict(loss=round(float(l1), 6), **s1), reload=dict(loss=round(float(l2), 6), **s2), max_diff=diff,
               layer=x["layer"], sil_frames=x.get("sil_frames"))
    json.dump(out, open(os.path.join(a.out_dir, "parity.json"), "w"), indent=1); log(f"저장·재로드 대조: {out}")
    assert diff == 0.0 and abs(float(l1) - float(l2)) <= 1e-4, f"재로드 불일치: {out}"


if __name__ == "__main__":
    main()

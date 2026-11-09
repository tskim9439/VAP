#!/usr/bin/env python
"""P1 화자 조건 인코더 단독 검증(2026-11-09, 사용자 "먼저 LLM 없이 인코더만으로 화자를 가를 수 있는지 검증해보자"
— wiki output-agn-reduction-20261104 12 절, output-ssa-papers-review-20261106 5 절).

우리 LLM 인코더의 원 모델 Nemotron 3.5 ASR 스트리밍 0.6B(RNN-T, en-US 프롬프트)에 Multitalker Parakeet 식 화자 커널을 붙인다.
  인코더 0 층 입력 x → x + K_t(x ⊙ m_t) + K_b(x ⊙ m_b). m_t = 대상 화자 활동, m_b = 다른 화자 활동 합(0/1, 80 ms 프레임).
  K = Linear-ReLU-Dropout(0.5)-Linear(NeMo SpeakerKernelMixin 'ff' 와 같은 식). 마지막 Linear 를 0 으로 두어 시작 = 원 모델.
  학습 목표: 섞인 소리 + 대상 화자 활동 → 그 화자의 말(RN-T). 대상 화자는 창 안에 말이 있는 화자 중 무작위(NeMo LhotseSpeechToTextSpkBpeDataset 과 같다).

  train  torchrun -m … experiments/ssa_probe.py train --out DIR [--steps 6000] [--rc 3] [--win 10,30] [--sec-per-gpu 360] [--set 이름=jsonl:mono:가중 …]
         창은 셋별 가중으로 뽑는다(기본: 합성 v7·v6·v4-meet + 실제 회의 근·원거리 + swbd, split v4 held-out·AMI 공식 dev·eval 회의는 뺀다).
         활동은 정답 발화 구간(경계 ±0.16 s 흔들기). 대상 글자 = 그 화자 토큰(Qwen, 시각 있음) 중 창 안의 것을 풀어 쓴다. 저장 DIR/step-N.pt·final.pt.
  eval   python experiments/ssa_probe.py eval --model nemotron|probe:<ckpt.pt>|parakeet:<.nemo> --act n3d|oracle --out DIR [--rc 3] [--rank r --world-size w]
         AMI eval(ami-bench-v1, 회의 16 × IHM-mix·SDM)을 30 s 조각(왼쪽 문맥 4 s)으로 자르고, 조각마다 활동 있는 화자를 대상으로 디코드한다.
         활동 n3d = 저장한 N3D 확률(스트리밍 청크 4, > 0.5), oracle = 같은 파일의 정답. 단어 시각 = RN-T 단어 타임스탬프 + 조각 시작.
         산출 predictions-rank*.jsonl 은 ext 행(id·corpus·delta 10·hyp_words) → gap_diag.py --sys 이름=ext:<DIR> 로 같은 셈(겹침별 놓침·귀속).
아무것도 지우지 않는다.
"""
import argparse, glob, json, math, os, random, sys, time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
FR = 0.08                                                                                       # 인코더 프레임(8 배 서브샘플링 × 10 ms)
NEMO35 = "/soundai/Model/nemotron-3.5-asr-streaming-0.6b/nemotron-3.5-asr-streaming-0.6b.nemo"
D = "/soundai/users/tskim/VAPKT-data/data"
TOK = "/soundai/Model/VAPASR/stage3-q17-e13/final"                                            # 대화 토큰(Qwen) 풀어 쓰기·AMI 평가 참조와 같은 토크나이저
N3D_PROBS = "/soundai/users/tskim/VAPKT-data/tmp/e11-check/diar-probs/n3d/n3d/probs"
DEFAULT_SETS = [                                                                                # 이름=dialogues.jsonl:mono 캐시:가중
    f"v7={D}/sim-dialogue-v7-ovl2/sim_en-ovl2.refined.dialogues.jsonl:{D}/sim-dialogue-v7-ovl2/mono:0.35",
    f"v6={D}/sim-dialogue-v6-ovl/sim_en-ovl.refined.dialogues.jsonl:{D}/sim-dialogue-v6-ovl/mono:0.10",
    f"v4meet={D}/sim-dialogue-v4-meet/sim_en-meet.refined.dialogues.jsonl:{D}/sim-dialogue-v4-meet/mono:0.10",
    *[f"{c}={D}/phase2/{c}.refined.dialogues.jsonl:{D}/phase2/mono:{w}" for c, w in
      (("ami", 0.06), ("icsi", 0.05), ("notsofar", 0.04), ("chime6tr", 0.03), ("ami-far", 0.06), ("icsi-far", 0.05), ("notsofar-far", 0.04), ("chime6tr-far", 0.02), ("swbd", 0.10))],
]


# ── 화자 커널 ─────────────────────────────────────────────────────────────────
def fit_mask(m, T, default):
    """NeMo SpeakerKernelMixin.solve_length_mismatch 와 같다: (B, T') → (B, T) — 짧으면 왼쪽을 default 로 채우고, 길면 왼쪽을 잘라 낸다."""
    import torch
    if m.shape[1] < T: m = torch.nn.functional.pad(m, (T - m.shape[1], 0), value=default)
    return m[:, -T:]


def build_kernels(d=1024, p=0.5):
    import torch.nn as nn

    class SpkKernels(nn.Module):
        def __init__(self):
            super().__init__()
            mk = lambda: nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(p), nn.Linear(d, d))
            self.t, self.b = mk(), mk()
            for k in (self.t, self.b): nn.init.zeros_(k[-1].weight); nn.init.zeros_(k[-1].bias)   # 시작 = 원 모델
            self.mt = self.mb = None

        def hook(self, module, args, kwargs):                                                  # encoder.layers[0] forward pre-hook(with_kwargs)
            if self.mt is None: return args, kwargs
            key = "x" if "x" in kwargs else None
            x = kwargs["x"] if key else args[0]
            mt = fit_mask(self.mt.to(x.dtype), x.shape[1], 1.0); mb = fit_mask(self.mb.to(x.dtype), x.shape[1], 0.0)
            x = x + self.t(x * mt[..., None]) + self.b(x * mb[..., None])
            if key: kwargs["x"] = x
            else: args = (x, *args[1:])
            return args, kwargs
    return SpkKernels()


class Asr:
    """kind: nemotron(커널 없음)·probe(Nemotron 3.5 + 우리 커널)·parakeet(Multitalker Parakeet — 자기 커널, set_speaker_targets)."""
    def __init__(self, kind, path=None, rc=3, device="cuda"):
        import torch
        import nemo.collections.asr as nemo_asr
        from omegaconf import open_dict
        self.kind = kind; self.dev = device
        src = path if kind == "parakeet" else NEMO35
        self.m = nemo_asr.models.ASRModel.restore_from(src, map_location="cpu")
        left = 70 if kind == "parakeet" else 56
        self.m.encoder.set_default_att_context_size([left, rc])
        self.pid = None if kind == "parakeet" else int(self.m.cfg.model_defaults.prompt_dictionary["en-US"])
        self.k = None
        if kind in ("probe", "train"):
            self.k = build_kernels(int(self.m.cfg.encoder.d_model))
            self.m.encoder.layers[0].register_forward_pre_hook(self.k.hook, with_kwargs=True)
            if kind == "probe":
                st = torch.load(path, map_location="cpu")
                self.m.load_state_dict(st["model"]); self.k.load_state_dict(st["kernels"])
        with open_dict(self.m.cfg.decoding): self.m.cfg.decoding.compute_timestamps = True
        self.m.change_decoding_strategy(self.m.cfg.decoding, verbose=False)
        self.m.to(device)
        if self.k is not None: self.k.to(device)

    def set_targets(self, mt, mb):
        if self.kind == "parakeet": self.m.set_speaker_targets(mt, mb)
        elif self.k is not None: self.k.mt, self.k.mb = mt, mb

    def encode(self, wav, lens):
        import torch
        if self.pid is None: return self.m.forward(input_signal=wav, input_signal_length=lens)
        return self.m.forward(input_signal=wav, input_signal_length=lens, prompt_indices=torch.full((wav.shape[0],), self.pid, dtype=torch.long, device=wav.device))

    def rnnt_loss(self, wav, lens, toks, tlens, sb=2):
        """RN-T 손실(참조 토큰 합으로 나눈 평균 — NeMo mean_volume 과 같은 척도). 인코더만 bf16 autocast.
        NeMo 의 warprnnt_numba 는 이 환경(numba 0.67)에서 컴파일이 깨져 torchaudio.functional.rnnt_loss 를 쓴다(2026-11-09 스모크).
        joint 출력(B·T·U·13088)이 커서 표본 sb 개씩 나누고 torch.utils.checkpoint 로 역전파 때 다시 계산한다."""
        import torch
        import torchaudio.functional as AF
        from torch.utils.checkpoint import checkpoint
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc, enc_len = self.encode(wav, lens)
        enc = enc.float().transpose(1, 2)                                                        # (B, T, D)
        dec, _, _ = self.m.decoder(targets=toks, target_length=tlens)
        dec = dec.float().transpose(1, 2)                                                        # (B, U + 1, H)
        blank = int(self.m.decoder.blank_idx)
        def part(e, g, t, el, tl_):
            return AF.rnnt_loss(self.m.joint.joint(e, g).float(), t.int().contiguous(), el.int(), tl_.int(), blank=blank, reduction="sum", fused_log_softmax=True)
        tot = enc.new_zeros(())
        for i in range(0, enc.shape[0], sb):
            Tm = int(enc_len[i:i + sb].max()); Um = int(tlens[i:i + sb].max())
            tot = tot + checkpoint(part, enc[i:i + sb, :Tm], dec[i:i + sb, :Um + 1], toks[i:i + sb, :Um], enc_len[i:i + sb], tlens[i:i + sb], use_reentrant=False)
        return tot / tlens.sum().clamp(min=1)

    def decode_words(self, wav, lens):
        """→ 표본마다 [(단어, 시작 s)] (RN-T 단어 타임스탬프, 프레임 × 80 ms)."""
        import torch, re
        with torch.inference_mode():
            enc, enc_len = self.encode(wav, lens)
            hy = self.m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc, encoded_lengths=enc_len, return_hypotheses=True)
        if isinstance(hy, tuple): hy = hy[0]
        tag = re.compile(r"<[a-z]{2}-[A-Z]{2}>")
        out = []
        for h in hy:
            ws = []
            for w in ((h.timestamp or {}).get("word") or []):
                t = w.get("word") or ""; t = tag.sub("", t).strip()
                if t: ws.append((t, float(w.get("start_offset", 0)) * FR))
            if not ws and (h.text or "").strip():                                            # 타임스탬프가 없으면 단어를 고르게 놓는다
                tx = tag.sub("", h.text).split(); ws = [(x, i * FR) for i, x in enumerate(tx)]
            out.append(ws)
        return out


# ── 학습 데이터 ───────────────────────────────────────────────────────────────
class Windows:
    """셋별 가중으로 창을 뽑아 (wav, 대상 활동, 배경 활동, 대상 글자) 를 낸다."""
    def __init__(self, specs, qtok, win=(10.0, 30.0), seed=0, rank=0):
        from vapasr.data.dialogue import Dialogue
        self.qtok = qtok; self.win = win; self.seed_base = seed * 1000 + rank; self.rng = random.Random(self.seed_base)
        split = json.load(open(f"{D}/phase2/splits/v4.json", encoding="utf-8"))
        man = json.load(open(f"{D}/ami-bench-v1/manifest.json")); amis = set(man["dev"]) | set(man["eval"])
        self.sets = []
        for sp in specs:
            name, rest = sp.split("=", 1); f, mono, w = rest.rsplit(":", 2)
            corp = os.path.basename(f).split(".refined")[0]
            held = set(split["corpora"].get(corp, {}).get("heldout", []))
            ds = []
            for l in open(f, encoding="utf-8"):
                d = Dialogue.from_json(l)
                if d.conv_id in held or d.conv_id.split(":")[-1] in amis or d.duration_s < win[0] + 1: continue
                ds.append(d)
            if ds: self.sets.append((name, ds, mono, float(w)))
            print(f"  셋 {name}: 대화 {len(ds)} · 가중 {w}", flush=True)
        self.W = [w for _, _, _, w in self.sets]

    def bad(self, d, t0, t1):
        miss = set(d.meta.get("missing", []))
        return any((u.flags or not u.tokens or i in miss) and u.end > t0 and u.start < t1 for i, u in enumerate(d.utterances, 1))

    def sample(self):
        import numpy as np
        from vapasr.data.dialogue_dataset import build_mono_cache
        for _ in range(50):
            name, ds, mono, _ = self.rng.choices(self.sets, weights=self.W)[0]
            d = self.rng.choice(ds); L = self.rng.uniform(*self.win); L = min(L, d.duration_s - 0.5)
            t0 = self.rng.uniform(0.0, max(0.0, d.duration_s - L)); t1 = t0 + L
            if self.bad(d, t0, t1): continue
            toks = {}
            for u in d.utterances:
                if u.end <= t0 or u.start >= t1: continue
                toks.setdefault(u.speaker, []).extend((t, tid) for tid, t in (u.tokens or []) if t0 <= t < t1)
            spk = [s for s, v in toks.items() if v]
            if not spk: continue
            s = self.rng.choice(spk); text = self.qtok.decode([tid for _, tid in sorted(toks[s])]).strip()
            if not text: continue
            K = int(math.ceil(L / FR)); mt = np.zeros(K, np.float32); mb = np.zeros(K, np.float32)
            for u in d.utterances:
                if u.end <= t0 or u.start >= t1: continue
                j = lambda: self.rng.uniform(-0.16, 0.16)
                a = max(0, int((max(u.start, t0) - t0 + j()) / FR)); b = min(K, int(math.ceil((min(u.end, t1) - t0 + j()) / FR)))
                (mt if u.speaker == s else mb)[a:b] = 1.0
            try:
                x = np.load(build_mono_cache(d, mono), mmap_mode="r")
            except Exception as e:
                print(f"  ! 오디오 {d.conv_id}: {type(e).__name__} {e}", flush=True); continue
            wav = np.asarray(x[int(t0 * 16000): int(t1 * 16000)], dtype=np.float32)
            if len(wav) < 16000: continue
            return dict(wav=wav, mt=mt, mb=mb, text=text, name=name)
        raise RuntimeError("창을 50 번 못 뽑았다")


def win_loader(W, workers):
    """창 뽑기를 DataLoader 작업자로(학습 프로세스가 NFS 읽기·토큰 풀기를 기다리지 않게). 작업자마다 난수 씨앗이 다르다."""
    import torch

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            wi = torch.utils.data.get_worker_info()
            if wi is not None: W.rng = random.Random(W.seed_base * 100 + wi.id + 1)
            while True: yield W.sample()
    if workers <= 0: return iter(Stream())
    return iter(torch.utils.data.DataLoader(Stream(), batch_size=None, num_workers=workers, prefetch_factor=16, persistent_workers=True))


def collate(items, tokenizer, dev):
    import torch
    B = len(items); T = max(len(i["wav"]) for i in items); K = max(len(i["mt"]) for i in items)
    wav = torch.zeros(B, T); lens = torch.zeros(B, dtype=torch.long); mt = torch.zeros(B, K); mb = torch.zeros(B, K)
    ids = [tokenizer.text_to_ids(i["text"]) for i in items]; U = max(1, max(len(x) for x in ids))
    toks = torch.zeros(B, U, dtype=torch.long); tl = torch.zeros(B, dtype=torch.long)
    for b, i in enumerate(items):
        wav[b, : len(i["wav"])] = torch.from_numpy(i["wav"]); lens[b] = len(i["wav"])
        mt[b, : len(i["mt"])] = torch.from_numpy(i["mt"]); mb[b, : len(i["mb"])] = torch.from_numpy(i["mb"])
        toks[b, : len(ids[b])] = torch.tensor(ids[b]); tl[b] = len(ids[b])
    return wav.to(dev), lens.to(dev), mt.to(dev), mb.to(dev), toks.to(dev), tl.to(dev)


def cmd_train(a):
    import torch, torch.distributed as dist
    from transformers import AutoTokenizer
    ddp = "RANK" in os.environ
    if ddp: dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); lr_ = int(os.environ.get("LOCAL_RANK", 0))
    else: rank, world, lr_ = 0, 1, 0
    torch.cuda.set_device(lr_); dev = f"cuda:{lr_}"; torch.manual_seed(a.seed + rank)
    log = lambda *x: print(f"[{time.strftime('%F %T')}] r{rank}", *x, flush=True) if rank == 0 else None
    os.makedirs(a.out, exist_ok=True)
    qtok = AutoTokenizer.from_pretrained(TOK)
    W = Windows(a.set or DEFAULT_SETS, qtok, win=tuple(float(x) for x in a.win.split(",")), seed=a.seed, rank=rank)
    asr = Asr("train", rc=a.rc, device=dev); m, k = asr.m, asr.k; m.train(); k.train()
    start = 0; last = sorted(glob.glob(os.path.join(a.out, "step-*.pt")), key=lambda p: int(p.rsplit("-", 1)[1][:-3]))
    opt_state = None
    if last:
        st = torch.load(last[-1], map_location="cpu"); m.load_state_dict(st["model"]); k.load_state_dict(st["kernels"]); start = st["step"]; opt_state = st.get("opt")
        log(f"이어 감 ← {last[-1]}(step {start})")
    ker = list(k.parameters()); base = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([dict(params=ker, lr=a.lr_kernel), dict(params=base, lr=a.lr)], weight_decay=1e-3)
    if opt_state: opt.load_state_dict(opt_state)
    peak = [a.lr_kernel, a.lr]
    def set_lr(s):
        f = min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
        for g, p in zip(opt.param_groups, peak): g["lr"] = p * f
    wrap = torch.nn.ModuleDict(dict(m=m, k=k))
    model = torch.nn.parallel.DistributedDataParallel(wrap, device_ids=[lr_]) if ddp else wrap
    log(f"학습 {sum(p.numel() for p in base) / 1e6:.0f}M + 커널 {sum(p.numel() for p in ker) / 1e6:.1f}M · rc {a.rc} · 창 {a.win} s · GPU 당 {a.sec_per_gpu} s · {a.steps} step")
    t0 = time.time(); acc = 0.0; n = 0; src = win_loader(W, a.workers); tw = 0.0
    for step in range(start, a.steps):
        items, sec = [], 0.0; t_ = time.time()
        while sec < a.sec_per_gpu:
            it = next(src); it = {k_: (v.numpy() if hasattr(v, "numpy") else v) for k_, v in it.items()}; items.append(it); sec += len(it["wav"]) / 16000
        tw += time.time() - t_
        wav, lens, mt, mb, toks, tl = collate(items, m.tokenizer, dev)
        set_lr(step); asr.set_targets(mt, mb)
        loss = asr.rnnt_loss(wav, lens, toks, tl)
        if ddp:                                                                                # 쓰지 않은 매개변수(프롬프트 등)까지 DDP 가 기다리지 않게 0 × 합을 더한다
            loss = loss + 0.0 * sum(p.sum() for p in model.parameters() if p.requires_grad)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(wrap.parameters()), 1.0); opt.step()
        acc += float(loss); n += 1
        if (step + 1) % 25 == 0:
            log(f"step {step + 1} loss {acc / n:.3f} · lr {opt.param_groups[1]['lr']:.2e}/{opt.param_groups[0]['lr']:.2e} · 창 {len(items)} · {(time.time() - t0) / n:.2f} s/step(데이터 대기 {tw / n:.2f}) · GPU {torch.cuda.max_memory_allocated() / 2**30:.0f} GB"); acc = 0.0; n = 0; t0 = time.time(); tw = 0.0
        if rank == 0 and ((step + 1) % a.save_every == 0 or step + 1 == a.steps):
            p = os.path.join(a.out, f"step-{step + 1}.pt"); tmp = p + ".tmp"
            torch.save(dict(model=m.state_dict(), kernels=k.state_dict(), step=step + 1, opt=opt.state_dict(), args=vars(a), rc=a.rc), tmp); os.replace(tmp, p)
            log(f"저장 {p}")
    if rank == 0:
        torch.save(dict(model=m.state_dict(), kernels=k.state_dict(), step=a.steps, args=vars(a), rc=a.rc), os.path.join(a.out, "final.pt"))
        open(os.path.join(a.out, "DONE"), "w").write(time.strftime("%F %T") + "\n"); log("DONE")
    if ddp: dist.barrier(); dist.destroy_process_group()


# ── 평가 ──────────────────────────────────────────────────────────────────────
def cmd_eval(a):
    import numpy as np, torch
    from transformers import AutoTokenizer
    from cst_stage3_eval import windows_of
    tok = AutoTokenizer.from_pretrained(TOK)
    ns = argparse.Namespace(data=f"{D}/ami-bench-v1", split=None, corpora=a.corpora, deltas=[10], limit_windows=0, backchannel=(1.0, 3), window="full", hop=10.0,
                            max_speakers=4, window_draws=0, declare=4, xl_order="lang", subset=a.subset)
    sets, items = windows_of(ns, tok)
    work = sorted({(c, sets[c].items[i][0]) for c, i in items})[a.rank::a.world_size][: a.limit or None]
    kind, _, path = a.model.partition(":")
    asr = Asr(kind, path or None, rc=a.rc); asr.m.eval()
    if asr.k is not None: asr.k.eval()
    os.makedirs(a.out, exist_ok=True); dest = os.path.join(a.out, f"predictions-rank{a.rank}.jsonl")
    done = {json.loads(l)["id"] for l in open(dest, encoding="utf-8")} if os.path.exists(dest) else set()
    json.dump(dict(vars(a), when=time.strftime("%F %T")), open(os.path.join(a.out, f"config-rank{a.rank}.json"), "w"), ensure_ascii=False, indent=1)
    for c, cid in work:
        rid = f"{cid}@0.0"
        if rid in done: continue
        t0 = time.time(); wav = np.asarray(sets[c].mono(cid), dtype=np.float32); L = len(wav) / 16000
        z = np.load(os.path.join(N3D_PROBS, f"{cid}@full.npz"))
        act = (z["prob"].astype(np.float32) > 0.5) if a.act == "n3d" else z["ref"].astype(bool)
        act = act[:, act.any(0)]; S = act.shape[1]
        jobs = []                                                                                # (조각 시작, 핵심 시작, 끝, 화자)
        for cs in np.arange(0.0, L, a.seg):
            ls = max(0.0, cs - a.ctx); ce = min(L, cs + a.seg)
            fa, fc, fe = int(ls / FR), int(cs / FR), int(math.ceil(ce / FR))
            for s in range(S):
                if act[fc:fe, s].sum() * FR >= a.min_act: jobs.append((ls, cs, ce, s))
        words = []
        for i in range(0, len(jobs), a.batch):
            b = jobs[i:i + a.batch]
            X = [wav[int(ls * 16000): int(ce * 16000)] for ls, cs, ce, s in b]; T = max(len(x) for x in X); K = int(math.ceil(T / 16000 / FR)) + 1
            W_ = torch.zeros(len(b), T); Ln = torch.zeros(len(b), dtype=torch.long); mt = torch.zeros(len(b), K); mb = torch.zeros(len(b), K)
            for j, ((ls, cs, ce, s), x) in enumerate(zip(b, X)):
                W_[j, : len(x)] = torch.from_numpy(x); Ln[j] = len(x); fa = int(ls / FR); seg = act[fa: fa + K]
                mt[j, : len(seg)] = torch.from_numpy(seg[:, s].astype(np.float32)); mb[j, : len(seg)] = torch.from_numpy(np.delete(seg, s, axis=1).any(1).astype(np.float32))
            asr.set_targets(mt.cuda(), mb.cuda())
            with torch.autocast("cuda", dtype=torch.bfloat16):
                hw = asr.decode_words(W_.cuda(), Ln.cuda())
            for (ls, cs, ce, s), ws in zip(b, hw):
                words += [[f"spk{s}", w, round(ls + t, 3)] for w, t in ws if cs - 1e-6 <= ls + t < ce]
        row = dict(id=rid, corpus=c, lang="English", delta=10, L=round(L, 3), model=a.model, act=a.act, rc=a.rc, n_spk=S, jobs=len(jobs), hyp_words=words, dec_s=round(time.time() - t0, 1))
        with open(dest, "a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[{time.strftime('%T')}] {rid}: 화자 {S} · 조각×화자 {len(jobs)} · 단어 {len(words)} · {row['dec_s']} s", flush=True)
    print(f"DONE rank={a.rank}", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True); t.add_argument("--steps", type=int, default=6000); t.add_argument("--rc", type=int, default=3)
    t.add_argument("--win", default="10,30"); t.add_argument("--sec-per-gpu", type=float, default=360.0); t.add_argument("--lr", type=float, default=3e-5)
    t.add_argument("--lr-kernel", type=float, default=5e-4); t.add_argument("--warmup", type=int, default=500); t.add_argument("--save-every", type=int, default=1000)
    t.add_argument("--set", action="append", default=None, help="이름=dialogues.jsonl:mono 캐시:가중(반복; 주지 않으면 기본 셋)"); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--workers", type=int, default=6, help="창 뽑기 DataLoader 작업자(0 = 학습 프로세스에서)")
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True, help="nemotron | probe:<ckpt.pt> | parakeet:<.nemo>"); e.add_argument("--act", choices=["n3d", "oracle"], default="n3d")
    e.add_argument("--out", required=True); e.add_argument("--rc", type=int, default=3); e.add_argument("--seg", type=float, default=30.0); e.add_argument("--ctx", type=float, default=4.0)
    e.add_argument("--min-act", type=float, default=0.3, help="조각 핵심에서 화자 활동이 이 초 이상일 때만 그 화자를 디코드")
    e.add_argument("--batch", type=int, default=16); e.add_argument("--corpora", default="ami-ihmmix,ami-sdm"); e.add_argument("--subset", default="eval")
    e.add_argument("--rank", type=int, default=0); e.add_argument("--world-size", type=int, default=1); e.add_argument("--limit", type=int, default=0, help="스모크: 회의 수")
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_eval(a)


if __name__ == "__main__":
    main()

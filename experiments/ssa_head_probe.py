#!/usr/bin/env python
"""보조 헤드 진단(2026-11-11, 사용자 "보조 헤드 진단부터 구현해줘" — wiki output-cpwer-hypotheses-20261110 6 절).

묻는 것: 지금 인코더가 내는 특징 한 줄(섞인 소리 → 고정 인코더, 한 번 통과)에 겹친 구간의 각 화자 말이 얼마나 남아 있는가.
  인코더는 **우리 모델의 것**(기본 E13 final 의 encoder.enc.* — 앞 단계에서 학습돼 원 Nemotron 3.5 와 636 개 텐서가 모두 다르다, 2026-11-11 확인)을 얹어 **고정**하고
  한 번만 돌린다. 전처리 → 인코더 출력 Z 는 우리 LLM 어댑터가 받는 특징 그대로다(fp32, 언어 프롬프트 융합 전). 그 Z 위에 작은 헤드만 학습한다.
    x0 = Z + K_t(Z ⊙ m_t) + K_b(Z ⊙ m_b)        m_t = 대상 화자 활동, m_b = 다른 화자 활동(80 ms 프레임) — P1 과 같은 커널이지만 인코더 **뒤**에 넣는다
    H  = 헤드 층 N 개(x0)                        자기 주의 창 = 왼쪽 L · 오른쪽 R 프레임(상대 위치 bias), 마지막 투영 0 초기화 → 시작 = 원 모델
    손실 = RN-T(프롬프트 융합(H), 대상 화자의 말)  원 모델의 언어 프롬프트 융합·디코더·joint 를 쓴다(기본은 같이 학습, --freeze-dec 면 고정)
  읽는 법(P1·Parakeet 과 같은 AMI eval 30 s 조각 평가·같은 셈): 둘 겹침 놓침이 E13·P1(약 43 %)보다 뚜렷이 낮으면 정보는 이미 특징 한 줄에 있고 읽는 쪽 문제다.
    비슷하거나 높으면 인코더가 약한 화자를 버린 것이다 → 인코더를 풀고 보조 손실로 학습(다음 단계).

P1(experiments/ssa_probe.py)과 다른 점
  · 인코더를 풀지 않는다(화자마다 인코더를 돌리지 않는다 — 2026-11-10 사용자 방침: 인코더 한 번 + LLM 한 스트림).
  · 여러 GPU 학습의 기울기를 실제로 합친다. ssa_probe.cmd_train 은 DDP 로 감싸기만 하고 forward 를 DDP 로 거치지 않아 rank 마다 따로 학습했다(2026-11-11 확인 —
    P1·P2 는 사실상 GPU 1 장 분량). 여기서는 손실 계산을 DDP 모듈의 forward 로 하고, 저장 때 rank 사이 가중이 같은지 확인한다.
  · 돌고 있는 P2(ssa_probe.py·ssa-probe.sbatch)를 건드리지 않으려고 새 파일로 둔다. 창 뽑기·평가·RN-T 손실은 ssa_probe 것을 그대로 쓴다.

  train  torchrun … experiments/ssa_head_probe.py train --out DIR [--steps 6000] [--head-layers 4] [--head-ctx 64,2] [--lr-head 3e-4] [--lr 3e-5] [--freeze-dec]
         [--enc-from <우리 모델 폴더>|none(원 Nemotron 3.5)]
         [--sets p1|p2] [--cross-p 0] [--win 10,30] [--sec-per-gpu 360] — 저장 DIR/step-N.pt(헤드·디코더·joint, bf16)·opt-last.pt(덮어씀)·final.pt·DONE
  eval   python experiments/ssa_head_probe.py eval --model head:<ckpt.pt> --act n3d|oracle --out DIR [--rank r --world-size w] — ssa_probe eval 과 같은 ext 행
아무것도 지우지 않는다(opt-last.pt 만 제 파일을 덮어쓴다).
"""
import argparse, glob, json, math, os, sys, time

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path[:0] = [REPO, os.path.join(REPO, "experiments")]
import ssa_probe as P                                                                           # noqa: E402  창 뽑기·collate·평가·RN-T 손실

NEG = -1e4                                                                                      # 채움 프레임 마스크(−inf 면 모두 가려진 행이 NaN 이 된다)
ENC_FROM = "/soundai/Model/VAPASR/stage3-q17-e13/final"                                         # 우리 모델 인코더(E16·E17 도 이 인코더를 고정해 쓴다)


def our_encoder_state(model_dir):
    """우리 HF 모델 폴더의 safetensors 에서 인코더 가중(encoder.enc.*)만 → NeMo 인코더 state_dict(접두사 뺌)."""
    from safetensors.torch import load_file
    sd = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "model*.safetensors"))):
        sd.update({k[len("encoder.enc."):]: v for k, v in load_file(f).items() if k.startswith("encoder.enc.")})
    if not sd: raise FileNotFoundError(f"encoder.enc.* 가중이 없다: {model_dir}")
    return sd


Z_SCALE = 0.045                                                                                 # 인코더 출력 Z 의 표준편차(2026-11-11 AMI 창 실측 0.044–0.045) — 헤드는 Z / 이 값(크기 1 근처)에서 일한다


def build_head(d=1024, layers=4, heads=8, ff=2048, left=64, right=2, p=0.1, kp=0.5, scale=Z_SCALE):
    import torch, torch.nn as nn, torch.nn.functional as F

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d); self.n2 = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(p), nn.Linear(ff, d)); self.dp = nn.Dropout(p)
            self.rel = nn.Parameter(torch.zeros(heads, left + right + 1))                        # 상대 위치(j − i ∈ [−left, right]) bias
            for m_ in (self.o, self.ff[-1]): nn.init.zeros_(m_.weight); nn.init.zeros_(m_.bias)  # 시작 = 항등

        def forward(self, x, band, idx, pad):                                                   # x (B, T, d) · band (T, T) 0/−inf · idx (T, T) · pad (B, T) True = 채움
            B, T, _ = x.shape
            q, k, v = self.qkv(self.n1(x)).view(B, T, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
            m = (band + self.rel[:, idx])[None].expand(B, -1, -1, -1).masked_fill(pad[:, None, None, :], NEG).to(q.dtype)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=p if self.training else 0.0)
            x = x + self.dp(self.o(a.transpose(1, 2).reshape(B, T, d)))
            return x + self.dp(self.ff(self.n2(x)))

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            mk = lambda: nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Dropout(kp), nn.Linear(d, d))
            self.kt, self.kb = mk(), mk()
            for k_ in (self.kt, self.kb): nn.init.zeros_(k_[-1].weight); nn.init.zeros_(k_[-1].bias)
            self.layers = nn.ModuleList(Layer() for _ in range(layers)); self.left, self.right = left, right
            self.cfg = dict(d=d, layers=layers, heads=heads, ff=ff, left=left, right=right, p=p, kp=kp, scale=scale)

        def forward(self, z, lens, mt, mb):                                                     # z (B, T, d) 인코더 출력 · lens (B,) · mt·mb (B, K) 활동
            B, T, _ = z.shape
            mt = P.fit_mask(mt.to(z.dtype), T, 1.0); mb = P.fit_mask(mb.to(z.dtype), T, 0.0)
            z = z / scale                                                                        # Z 는 값이 작다(표준편차 ≈ 0.045) — 그대로 두면 헤드의 첫 갱신이 특징을 덮는다(스모크: 4 step 뒤 출력 0 단어)
            x = z + self.kt(z * mt[..., None]) + self.kb(z * mb[..., None])
            i = torch.arange(T, device=z.device); rel = i[None, :] - i[:, None]                  # (질의 i, 키 j) → j − i
            band = torch.zeros(T, T, device=z.device).masked_fill((rel < -self.left) | (rel > self.right), float("-inf"))
            idx = rel.clamp(-self.left, self.right) + self.left; pad = i[None, :] >= lens[:, None]
            for l in self.layers: x = l(x, band, idx, pad)
            return x * scale
    return Head()


class HeadAsr(P.Asr):
    """고정 Nemotron 3.5(커널 없음, 인코더 한 번) + 인코더 뒤 화자 조건 헤드. encode() 가 헤드 출력을 내므로 ssa_probe 의 RN-T 손실·디코드를 그대로 쓴다."""
    def __init__(self, path=None, rc=3, device="cuda", enc_from=ENC_FROM, **cfg):
        import torch
        super().__init__("nemotron", None, rc=rc, device=device)
        self.kind = "head"; st = torch.load(path, map_location="cpu") if path else None
        self.enc_from = st["enc_from"] if st else (None if enc_from in (None, "", "none") else enc_from)
        if self.enc_from: self.m.encoder.load_state_dict(our_encoder_state(self.enc_from))        # strict — 키·모양이 다르면 멈춘다
        self.m.encoder.float()
        self.head = build_head(**(st["cfg"] if st else dict(d=int(self.m.cfg.encoder.d_model), **cfg))).to(device)
        self.mt = self.mb = None
        if st:
            self.head.load_state_dict(st["head"]); self.m.decoder.load_state_dict(st["dec"]); self.m.joint.load_state_dict(st["joint"])
            if st.get("pk") is not None: self.m.prompt_kernel.load_state_dict(st["pk"])
            self.head.eval()

    def tail_modules(self):
        """헤드 뒤에서 글자를 내는 원 모델 부분: 언어 프롬프트 융합(있으면)·디코더·joint."""
        return ([self.m.prompt_kernel] if getattr(self.m, "concat", False) else []) + [self.m.decoder, self.m.joint]

    def set_targets(self, mt, mb): self.mt, self.mb = mt, mb

    def features(self, wav, lens):
        """섞인 소리 → 고정 인코더 한 번 → Z (B, T, D) fp32 — 우리 모델의 NemotronOnline 과 같은 경로(전처리 → 인코더, autocast 밖, 프롬프트 융합 전)."""
        import torch
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            f, fl = self.m.preprocessor(input_signal=wav.float(), length=lens)
            enc, enc_len = self.m.encoder(audio_signal=f, length=fl)
        return enc.transpose(1, 2).float(), enc_len

    def encode(self, wav, lens):
        import torch
        z, enc_len = self.features(wav, lens)
        h = self.head(z, enc_len, self.mt, self.mb)
        if getattr(self.m, "concat", False):                                                     # NeMo forward 와 같은 언어 프롬프트 융합(en-US)을 헤드 **뒤**에
            pr = torch.zeros(h.shape[0], h.shape[1], int(self.m.num_prompts), dtype=h.dtype, device=h.device); pr[:, :, self.pid] = 1.0
            h = self.m.prompt_kernel(torch.cat([h, pr], dim=-1)).to(h.dtype)
        return h.transpose(1, 2), enc_len


def loss_module(asr):
    """DDP 가 감싸는 모듈 — forward 가 손실을 낸다(이래야 rank 사이 기울기가 합쳐진다). 고정 인코더는 등록하지 않는다."""
    import torch

    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.head = asr.head; self.tail = torch.nn.ModuleList(asr.tail_modules())

        def forward(self, wav, lens, mt, mb, toks, tl):
            asr.set_targets(mt, mb); loss = asr.rnnt_loss(wav, lens, toks, tl)
            return loss + 0.0 * sum(p.sum() for p in self.parameters() if p.requires_grad)      # 손실에 안 쓰인 매개변수를 DDP 가 기다리지 않게
    return Loss()


def state(asr, step, a):
    return dict(head=asr.head.state_dict(), dec=P.half(asr.m.decoder.state_dict()), joint=P.half(asr.m.joint.state_dict()),
                pk=P.half(asr.m.prompt_kernel.state_dict()) if getattr(asr.m, "concat", False) else None, enc_from=asr.enc_from, cfg=asr.head.cfg, step=step, args=vars(a), rc=a.rc)


def same_across_ranks(params, dist):
    """모든 rank 의 가중이 같은가(합의 최대 = 최소). DDP 동기가 빠지면 여기서 멈춘다."""
    import torch
    c = torch.stack([p.detach().double().sum() for p in params]).sum(); t = torch.stack([c, -c]); dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t[0]) == -float(t[1]), float(t[0]), -float(t[1])


def cmd_train(a):
    import torch, torch.distributed as dist
    from transformers import AutoTokenizer
    ddp = "RANK" in os.environ
    if ddp: dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size(); lr_ = int(os.environ.get("LOCAL_RANK", 0))
    else: rank, world, lr_ = 0, 1, 0
    torch.cuda.set_device(lr_); dev = f"cuda:{lr_}"; torch.manual_seed(a.seed)                   # 헤드 초기값은 모든 rank 가 같다(dropout 난수는 아래에서 rank 마다)
    log = lambda *x: print(f"[{time.strftime('%F %T')}] r{rank}", *x, flush=True) if rank == 0 else None
    os.makedirs(a.out, exist_ok=True)
    qtok = AutoTokenizer.from_pretrained(P.TOK); L_, R_ = (int(x) for x in a.head_ctx.split(","))
    last = sorted(glob.glob(os.path.join(a.out, "step-*.pt")), key=lambda p: int(p.rsplit("-", 1)[1][:-3]))
    start = int(last[-1].rsplit("-", 1)[1][:-3]) if last else 0
    W = P.Windows(a.set or (P.P2_SETS if a.sets == "p2" else P.DEFAULT_SETS), qtok, win=tuple(float(x) for x in a.win.split(",")), seed=a.seed + start, rank=rank,
                  cross_p=a.cross_p, sir=tuple(float(x) for x in a.sir.split(",")))                 # 씨앗에 이어 가는 step 을 넣는다 — 재시작마다 같은 창을 다시 뽑지 않게
    asr = HeadAsr(None, rc=a.rc, device=dev, enc_from=a.enc_from, layers=a.head_layers, heads=a.head_heads, ff=a.head_ff, left=L_, right=R_)
    m, h = asr.m, asr.head; m.eval()                                                             # 인코더·전처리 고정(SpecAugment·dropout 없음)
    for p in m.parameters(): p.requires_grad_(False)
    dec_ps = []
    if not a.freeze_dec:
        dec_ps = [p for mod in asr.tail_modules() for p in mod.parameters()]
        for p in dec_ps: p.requires_grad_(True)
        for mod in asr.tail_modules(): mod.train()
    h.train(); opt_state = None
    if last:
        st = torch.load(last[-1], map_location="cpu"); h.load_state_dict(st["head"]); m.decoder.load_state_dict(st["dec"]); m.joint.load_state_dict(st["joint"]); assert st["step"] == start
        if st.get("pk") is not None: m.prompt_kernel.load_state_dict(st["pk"])
        assert st.get("enc_from") == asr.enc_from, f"인코더가 다르다: 저장 {st.get('enc_from')} · 지금 {asr.enc_from}"
        op = os.path.join(a.out, "opt-last.pt")
        if os.path.exists(op):
            o = torch.load(op, map_location="cpu"); opt_state = o["opt"] if o.get("step") == start else None
        log(f"이어 감 ← {last[-1]}(step {start}, 옵티마이저 {'이어 감' if opt_state else '새로'})")
    torch.manual_seed(a.seed * 1000 + rank + start)
    groups = [dict(params=list(h.parameters()), lr=a.lr_head)] + ([dict(params=dec_ps, lr=a.lr)] if dec_ps else [])
    opt = torch.optim.AdamW(groups, weight_decay=1e-3); peak = [g["lr"] for g in groups]
    if opt_state: opt.load_state_dict(opt_state)
    def set_lr(s):
        f = min(1.0, (s + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / a.steps))))
        for g, p in zip(opt.param_groups, peak): g["lr"] = p * f
    lm = loss_module(asr); model = torch.nn.parallel.DistributedDataParallel(lm, device_ids=[lr_]) if ddp else lm
    train_ps = [p for p in lm.parameters() if p.requires_grad]
    if ddp:                                                                                      # 시작 가중을 rank 0 것으로 맞추고 확인한다(DDP 는 처음 가중을 뿌리지 않는다)
        for p in train_ps: dist.broadcast(p.data, src=0)
        ok, hi, lo = same_across_ranks(train_ps, dist); assert ok, f"시작 가중이 rank 마다 다르다: {hi} · {lo}"
    log(f"헤드 {sum(p.numel() for p in h.parameters()) / 1e6:.1f}M(층 {a.head_layers} · 창 −{L_}/+{R_} 프레임) + 프롬프트 융합·디코더·joint {sum(p.numel() for p in dec_ps) / 1e6:.1f}M 학습"
        f" · 인코더 고정 rc {a.rc} ← {asr.enc_from or '원 Nemotron 3.5'}"
        f" · 창 {a.win} s · GPU 당 {a.sec_per_gpu} s × {world} · {a.steps} step · 간섭원 덧섞기 {a.cross_p}(SIR {a.sir} dB)")
    t0 = time.time(); acc = 0.0; n = 0; src = P.win_loader(W, a.workers); tw = 0.0
    for step in range(start, a.steps):
        items, sec = [], 0.0; t_ = time.time()
        while sec < a.sec_per_gpu:
            it = next(src); it = {k_: (v.numpy() if hasattr(v, "numpy") else v) for k_, v in it.items()}; items.append(it); sec += len(it["wav"]) / 16000
        tw += time.time() - t_
        wav, lens, mt, mb, toks, tl = P.collate(items, m.tokenizer, dev)
        set_lr(step); loss = model(wav, lens, mt, mb, toks, tl)                                  # DDP forward → 기울기 동기
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(train_ps, 1.0); opt.step()
        acc += float(loss); n += 1
        if (step + 1) % 25 == 0:
            log(f"step {step + 1} loss {acc / n:.3f} · lr {opt.param_groups[0]['lr']:.2e} · 창 {len(items)} · {(time.time() - t0) / n:.2f} s/step(데이터 대기 {tw / n:.2f}) · GPU {torch.cuda.max_memory_allocated() / 2**30:.0f} GB"); acc = 0.0; n = 0; t0 = time.time(); tw = 0.0
        if (step + 1) % a.save_every == 0 or step + 1 == a.steps:
            if ddp:
                ok, hi, lo = same_across_ranks(train_ps, dist)
                assert ok, f"rank 마다 가중이 다르다(step {step + 1}): 최대 {hi} · 최소 {lo} — DDP 동기 실패"
            if rank == 0:
                p = os.path.join(a.out, f"step-{step + 1}.pt"); torch.save(state(asr, step + 1, a), p + ".tmp"); os.replace(p + ".tmp", p)
                op = os.path.join(a.out, "opt-last.pt"); torch.save(dict(opt=opt.state_dict(), step=step + 1), op + ".tmp"); os.replace(op + ".tmp", op)
                log(f"저장 {p}" + (" · rank 사이 가중 같음" if ddp else ""))
    if rank == 0:
        torch.save(state(asr, a.steps, a), os.path.join(a.out, "final.pt"))
        open(os.path.join(a.out, "DONE"), "w").write(time.strftime("%F %T") + "\n"); log("DONE")
    if ddp: dist.barrier(); dist.destroy_process_group()


def cmd_eval(a):
    """ssa_probe.cmd_eval 그대로 — 모델만 head:<ckpt> 를 받는다(다른 종류는 ssa_probe 의 것)."""
    Base = P.Asr
    def make(kind, path=None, rc=3, device="cuda"):
        return HeadAsr(path, rc=rc, device=device) if kind == "head" else Base(kind, path, rc=rc, device=device)
    P.Asr = make
    try: P.cmd_eval(a)
    finally: P.Asr = Base


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--out", required=True); t.add_argument("--steps", type=int, default=6000); t.add_argument("--rc", type=int, default=3)
    t.add_argument("--head-layers", type=int, default=4); t.add_argument("--head-ctx", default="64,2", help="헤드 층마다 자기 주의 창: 왼쪽,오른쪽 프레임(80 ms) — 4 층 × 오른쪽 2 = 앞보기 0.64 s")
    t.add_argument("--head-heads", type=int, default=8); t.add_argument("--head-ff", type=int, default=2048)
    t.add_argument("--lr-head", type=float, default=3e-4); t.add_argument("--lr", type=float, default=3e-5, help="프롬프트 융합·디코더·joint 학습률"); t.add_argument("--freeze-dec", action="store_true")
    t.add_argument("--enc-from", default=ENC_FROM, help="인코더 가중을 가져올 우리 모델 폴더(encoder.enc.*), none = 원 Nemotron 3.5")
    t.add_argument("--win", default="10,30"); t.add_argument("--sec-per-gpu", type=float, default=360.0); t.add_argument("--warmup", type=int, default=300)
    t.add_argument("--save-every", type=int, default=500); t.add_argument("--seed", type=int, default=0); t.add_argument("--workers", type=int, default=6)
    t.add_argument("--set", action="append", default=None, help="이름=dialogues.jsonl:mono 캐시:가중(반복; 주지 않으면 --sets 묶음)")
    t.add_argument("--sets", choices=["p1", "p2"], default="p1"); t.add_argument("--cross-p", type=float, default=0.0); t.add_argument("--sir", default="-6,3")
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True, help="head:<ckpt.pt>(또는 ssa_probe 의 nemotron | probe:<ckpt> | parakeet:<.nemo>)"); e.add_argument("--act", choices=["n3d", "oracle"], default="n3d")
    e.add_argument("--out", required=True); e.add_argument("--rc", type=int, default=3); e.add_argument("--seg", type=float, default=30.0); e.add_argument("--ctx", type=float, default=4.0)
    e.add_argument("--min-act", type=float, default=0.3); e.add_argument("--batch", type=int, default=16); e.add_argument("--corpora", default="ami-ihmmix,ami-sdm")
    e.add_argument("--subset", default="eval"); e.add_argument("--rank", type=int, default=0); e.add_argument("--world-size", type=int, default=1); e.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cmd_train(a) if a.cmd == "train" else cmd_eval(a)


if __name__ == "__main__":
    main()

"""화자 분리 가지(E1, 2026-10-25 — wiki output-diarization-integration-20261025): 인코더 여러 층 출력의 학습 가중합 → Transformer → 슬롯 활동 sigmoid.
Sortformer(arXiv:2409.06656)·Streaming Sortformer(arXiv:2507.18446)와 같은 출력 규약이다. 슬롯 0 = 먼저 말한 화자(도착 순서)이고, 손실은 도착 순서 BCE(Sort loss)와
순열 불변 BCE(PIT)의 가중합이다. 우리 대화 슬롯 규칙(cst_dialogue.window_speakers — 창 안에서 먼저 말한 사람 = A)과 같은 순서라 라벨은 slot_activity 그대로 쓴다.
- DiarBranch: (B, S, K, D) 인코더 스트림(model.encode / NemotronOnline taps) → (B, K, N) logits. 위치 정보는 깊이별 conv(길이와 무관, wav2vec2 식).
- sort_pit_loss: 프레임 마스크(패딩·전사 없는 음성)를 뺀 BCE.
- der_counts: 80 ms 프레임 DER 셈(collar 0) — 도착 순서 그대로의 짝과 혼동이 가장 적은 짝 둘 다.
- load_encoder: VapAsr 체크포인트(조각 저장 포함)에서 인코더(encoder.enc.*)만 읽어 NemotronOnline 을 만든다(동결·fp32).
- AOSC(E7, 2026-10-27 — 회의 통째·실시간 화자 유지): Streaming Sortformer(arXiv:2507.18446)의 도착 순서 화자 캐시. 가지 입력 임베딩(embed)을
  [화자 캐시 · FIFO · 새 청크] 로 이어 인과 가지(head)에 넣고 청크 몫 출력만 쓴다. 캐시는 화자(도착 순서)마다 묶은 대표 프레임이라, 그 화자가 늘 같은 슬롯이 된다.
  aosc_select(캐시 갱신 규칙) · aosc_stream(추론, 모델 예측으로 캐시) · aosc_clip_loss(학습, 정답으로 캐시·캐시 화자 순서 섞기).
"""
import itertools, json, math, os, random
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvPos(nn.Module):
    def __init__(self, d: int, k: int = 31, groups: int = 16, lookahead: Optional[int] = None):
        super().__init__(); self.conv = nn.Conv1d(d, d, k, padding=k // 2, groups=groups)
        m = torch.ones(k) if lookahead is None else (torch.arange(k) - k // 2 <= int(lookahead)).float()   # 인과(lookahead R): 미래 R 프레임보다 먼 탭을 0 으로
        self.register_buffer("tap_mask", m, persistent=False)

    def forward(self, x):                                                                      # (B, K, d)
        w = self.conv.weight * self.tap_mask
        return x + F.gelu(F.conv1d(x.transpose(1, 2), w, self.conv.bias, padding=self.conv.padding, groups=self.conv.groups).transpose(1, 2))


class DiarBranch(nn.Module):
    """lookahead R(2026-10-26, E1c·E6 — 인과): 주면 프레임 t 의 출력이 입력 t + R 프레임까지만 본다 — 앞보기는 위치 conv 에서만(미래 R 프레임보다 먼 탭 0),
    어텐션은 층마다 엄격히 인과(j ≤ i, 층마다 앞보기를 주면 층 수만큼 쌓인다). 파라미터 모양은 같아 양방향(E1) 가중치를 그대로 불러올 수 있다. None 이면 예전(창 전체)."""
    def __init__(self, n_streams: int, d_in: int = 1024, d: int = 512, layers: int = 12, heads: int = 8, ffn: int = 2048, n_spk: int = 4, dropout: float = 0.1,
                 lookahead: Optional[int] = None):
        super().__init__()
        self.cfg = dict(n_streams=n_streams, d_in=d_in, d=d, layers=layers, heads=heads, ffn=ffn, n_spk=n_spk, dropout=dropout, **({"lookahead": int(lookahead)} if lookahead is not None else {}))
        self.lookahead = None if lookahead is None else int(lookahead)
        self.mix = nn.Parameter(torch.zeros(n_streams)); self.ln = nn.LayerNorm(d_in); self.proj = nn.Linear(d_in, d); self.pos = ConvPos(d, lookahead=self.lookahead)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, ffn, dropout, batch_first=True, norm_first=True), layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, n_spk)

    def embed(self, feats):
        """feats (B, S, K, D_in) → 가지 입력 임베딩 (B, K, d)(탭 가중합 → LN → 투영). AOSC 캐시·FIFO 가 이 값을 들고 다닌다."""
        w = torch.softmax(self.mix, 0).to(feats.dtype); return self.proj(self.ln((feats * w[None, :, None, None]).sum(1)))

    def head(self, x, pad_mask: Optional[torch.Tensor] = None):
        """임베딩 (B, K, d) → logits (B, K, N)(위치 conv → Transformer). pad_mask (B, K) True = 패딩."""
        if pad_mask is not None: x = x.masked_fill(pad_mask[..., None], 0.0)                     # conv 전에 — 끝 패딩이 0 채움과 같아진다
        x = self.pos(x); am = None
        if self.lookahead is not None:                                                          # (K, K) True = 가린다: j > i(앞보기는 위치 conv 의 R 프레임만)
            K = x.shape[1]; i = torch.arange(K, device=x.device); am = i[None, :] > i[:, None]
        return self.out(self.norm(self.tf(x, mask=am, src_key_padding_mask=pad_mask)))

    def forward(self, feats, pad_mask: Optional[torch.Tensor] = None, aosc: Optional[dict] = None):
        """feats (B, S, K, D_in) → logits (B, K, N). pad_mask (B, K) True = 패딩 프레임(어텐션에서 뺀다).
        aosc(E7 학습 — DDP forward 를 거치게 여기서): dict(act (B, K, N), mask (B, K), K (B,), cfg AoscConfig, seeds [표본마다], alpha, perm_p, chunk)
        → (손실, 통계 dict) — 표본마다 aosc_clip_loss(스트리밍 흉내)의 프레임 가중 평균."""
        if aosc is None: return self.head(self.embed(feats), pad_mask)
        e = self.embed(feats); tot = e.new_zeros(()); st = dict(frames=0.0, hit=0.0, sort=0.0, pit=0.0, steps=0)
        for j in range(e.shape[0]):
            Kj = int(aosc["K"][j]); l, s = aosc_clip_loss(self, e[j, :Kj], aosc["act"][j, :Kj], aosc["mask"][j, :Kj], aosc["cfg"], random.Random(aosc["seeds"][j]),
                                                         alpha=aosc.get("alpha", 0.5), perm_p=aosc.get("perm_p", 0.5), chunk=aosc.get("chunk", 24))
            tot = tot + l
            for k in st: st[k] += s[k]
        return tot / max(1.0, st["frames"]), st


@dataclass
class AoscConfig:
    """AOSC 설정(프레임 = 80 ms). Streaming Sortformer 기본: 캐시 188, 점수 규칙(임계 0.5 · 새 프레임 +0.05 · 화자마다 상위 33 개 +2 ln 2 · 상위 66 개 +ln 2 · 무음 칸 3).
    FIFO 가 fifo 를 넘으면 가장 오래된 period 프레임을 캐시 후보로 내보내 갱신한다. chunk = 추론 한 step 의 새 프레임(가지가 인과라 출력은 청크 크기와 무관하고 갱신 시점만 바뀐다)."""
    cache: int = 188
    fifo: int = 188
    period: int = 94
    chunk: int = 6
    thr: float = 0.5
    recency: float = 0.05
    k1: int = 33
    d1: float = 2 * math.log(2.0)
    k2: int = 66
    d2: float = math.log(2.0)
    sil: int = 3

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AoscConfig":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict:
        return asdict(self)


def aosc_select(prob, new, cfg: AoscConfig, noise: float = 0.0, rng: Optional[torch.Generator] = None):
    """캐시 갱신 규칙(Streaming Sortformer AOSC). 후보 프레임 확률 prob (n, N)·새 프레임 표시 new (n,) → 고른 칸 (frame (M',) long — 후보 번호, −1 = 무음 칸,
    spk (M',) long). 점수 S_i = log P_i + Σ_{j≠i} log(1 − P_j), P_i < thr 이면 −∞, 새 프레임 +recency, 화자마다 상위 k1·k2 개에 d1·d2 를 더한다.
    후보에 있는(점수가 유한한) 화자마다 무음 칸 sil 개(+∞). 점수 상위 cache 칸을 고른 뒤 (화자, 무음 칸은 그 화자 프레임 뒤, 후보 순서)로 늘어놓는다
    — 화자 번호 = 도착 순서라 캐시 앞쪽 화자가 앞 슬롯이 된다. noise > 0 이면 유한 점수에 Gumbel(0, noise)를 더한다(학습 — 같은 점수의 정답 프레임 고르기를 흩뜨린다)."""
    n, N = prob.shape; dev = prob.device
    if n == 0: return torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev)
    p = prob.float().clamp(1e-6, 1 - 1e-6); l1 = torch.log1p(-p)
    S = torch.log(p) - l1 + l1.sum(-1, keepdim=True)
    S = S.masked_fill(p < cfg.thr, float("-inf")) + cfg.recency * new.float()[:, None]
    fin = torch.isfinite(S)
    if noise > 0 and fin.any():
        u = torch.rand(S.shape, generator=rng, device="cpu").to(dev).clamp(1e-9, 1 - 1e-9); S = torch.where(fin, S - noise * torch.log(-torch.log(u)), S)
    for k, dl in ((cfg.k1, cfg.d1), (cfg.k2, cfg.d2)):
        for i in range(N):
            m = int(fin[:, i].sum())
            if m: idx = torch.topk(S[:, i], min(k, m)).indices; S[idx, i] = S[idx, i] + dl
    present = fin.any(0)
    fi, si = torch.nonzero(fin, as_tuple=True); sc = S[fi, si]
    ps = torch.nonzero(present, as_tuple=True)[0]
    if cfg.sil > 0 and len(ps):                                                                    # 무음 칸(+∞ — 늘 남는다)
        fi = torch.cat([fi, torch.full((len(ps) * cfg.sil,), -1, dtype=torch.long, device=dev)]); si = torch.cat([si, ps.repeat_interleave(cfg.sil)])
        sc = torch.cat([sc, torch.full((len(ps) * cfg.sil,), float("inf"), device=dev)])
    if len(sc) > cfg.cache: keep = torch.topk(sc, cfg.cache).indices; fi, si = fi[keep], si[keep]
    order = sorted(range(len(fi)), key=lambda j: (int(si[j]), int(fi[j]) < 0, int(fi[j])))        # 화자 → 그 화자 프레임(후보 순서) → 무음 칸
    o = torch.tensor(order, dtype=torch.long, device=dev)
    return fi[o], si[o]


@torch.no_grad()
def aosc_stream(branch: "DiarBranch", emb, cfg: AoscConfig) -> torch.Tensor:
    """추론: 한 세션의 가지 입력 임베딩 emb (K, d)(branch.embed) → logits (K, N). 청크마다 [캐시 · FIFO · 청크] 를 head 에 넣고 청크 몫을 쓴다.
    FIFO 가 cfg.fifo 를 넘으면 가장 오래된 cfg.period 프레임과 지금 캐시(무음 칸 빼고)를 후보로, 이번 pass 의 예측 확률로 aosc_select 해 캐시를 바꾼다.
    무음 평균 = 지금까지 내보낸 프레임 중 모든 화자 확률이 thr 미만인 프레임 임베딩의 평균."""
    K, d = emb.shape; N = branch.out.out_features; dev = emb.device; out = emb.new_zeros(K, N, dtype=torch.float32)
    c_emb, f_emb = emb.new_zeros(0, d), emb.new_zeros(0, d); c_sil = torch.zeros(0, dtype=torch.bool, device=dev)
    sil_sum, sil_n = emb.new_zeros(d, dtype=torch.float32), 0
    for k0 in range(0, K, cfg.chunk):
        x = emb[k0: k0 + cfg.chunk]; n = x.shape[0]; Lc = c_emb.shape[0]
        lg = branch.head(torch.cat([c_emb, f_emb, x])[None])[0].float(); out[k0: k0 + n] = lg[-n:]
        pr = torch.sigmoid(lg); f_emb = torch.cat([f_emb, x]); f_pr = pr[Lc:]
        if f_emb.shape[0] <= cfg.fifo: continue
        m = min(cfg.period, f_emb.shape[0]); pe, pp = f_emb[:m], f_pr[:m]; f_emb = f_emb[m:]
        quiet = (pp < cfg.thr).all(-1)
        if quiet.any(): sil_sum += pe[quiet].float().sum(0); sil_n += int(quiet.sum())
        keep = ~c_sil; ce, cp = c_emb[keep], pr[:Lc][keep]
        cand, prob = torch.cat([ce, pe]), torch.cat([cp, pp]); new = torch.cat([torch.zeros(len(ce), dtype=torch.bool, device=dev), torch.ones(m, dtype=torch.bool, device=dev)])
        fi, _ = aosc_select(prob, new, cfg)
        sil = (sil_sum / max(1, sil_n)).to(emb.dtype)
        c_emb = torch.where((fi < 0)[:, None], sil[None], cand[fi.clamp(min=0)]); c_sil = fi < 0
    return out


def aosc_clip_loss(branch: "DiarBranch", emb, target, mask, cfg: AoscConfig, rng: random.Random, alpha: float = 0.5, perm_p: float = 0.5, chunk: int = 24,
                   noise: float = 0.5):
    """학습(E7): 한 클립의 스트리밍 흉내 → (손실 합(프레임 가중), 통계 dict(frames·hit·sort·pit·steps)). emb (K, d)(grad), target (K, N) 클립 도착 순서 정답,
    mask (K,) bool(손실에 넣는 프레임). 청크(chunk 프레임)마다 [캐시 · FIFO · 청크] 를 head 에 넣고 청크 몫에 Sort·PIT 손실을 건다.
    캐시는 정답 확률(0.025/0.975)로 aosc_select(noise 로 같은 점수 흩뜨리기)해 고르고, 확률 perm_p 로 캐시 화자 묶음 순서를 섞는다.
    그 step 의 슬롯 = 캐시 화자 묶음 순서, 그다음 캐시에 없는 화자를 [FIFO · 청크] 안 첫 발화 순으로 — 정답 열도 그 순서로 바꾼다(추론은 모델 예측으로 같은 규칙)."""
    K, N = target.shape; dev = emb.device; g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
    tgt = target.float(); pr_or = 0.025 + 0.95 * tgt
    c_idx = torch.zeros(0, dtype=torch.long, device=dev); c_spk = torch.zeros(0, dtype=torch.long, device=dev); f0 = 0
    sil_sum, sil_n = emb.new_zeros(emb.shape[1], dtype=torch.float32), 0; tot = emb.new_zeros((), dtype=torch.float32)
    st = dict(frames=0.0, hit=0.0, sort=0.0, pit=0.0, steps=0)
    for k0 in range(0, K, chunk):
        k1 = min(K, k0 + chunk); n = k1 - k0
        groups = []                                                                                # 캐시 화자 묶음(캐시 순서) — [(화자, 칸 번호들)]
        for j, s_ in enumerate(c_spk.tolist()):
            if groups and groups[-1][0] == s_: groups[-1][1].append(j)
            else: groups.append((s_, [j]))
        if len(groups) > 1 and rng.random() < perm_p: rng.shuffle(groups)
        order = [j for _, js in groups for j in js]; cached = [s_ for s_, _ in groups]
        sil = (sil_sum / max(1, sil_n)).to(emb.dtype).detach()
        ci = c_idx[order] if order else c_idx
        ce = torch.where((ci < 0)[:, None], sil[None], emb[ci.clamp(min=0)]) if len(ci) else emb.new_zeros(0, emb.shape[1])
        seq = torch.cat([ce, emb[f0:k1]]); lg = branch.head(seq[None])[0][-n:].float()
        act_win = tgt[f0:k1]; first = [(int(torch.nonzero(act_win[:, s_])[0]) if act_win[:, s_].any() else 10 ** 9, s_) for s_ in range(N) if s_ not in cached]
        slots = cached + [s_ for f_, s_ in sorted(first) if f_ < 10 ** 9]
        t_step = torch.zeros(n, N, device=dev)
        for slot, s_ in enumerate(slots[:N]): t_step[:, slot] = tgt[k0:k1, s_]
        m = mask[k0:k1]
        if m.any():
            l, so, pi = sort_pit_loss(lg[None], t_step[None], m[None], alpha); w = float(m.sum())
            tot = tot + l * w; st["frames"] += w; st["sort"] += float(so) * w; st["pit"] += float(pi) * w
            with torch.no_grad(): st["hit"] += float((((lg > 0).float() == t_step).float() * m.float()[:, None]).sum()) / N
        st["steps"] += 1
        if k1 - f0 <= cfg.fifo: continue
        mm = min(cfg.period, k1 - f0); pop = torch.arange(f0, f0 + mm, device=dev); f0 += mm
        quiet = ~tgt[pop].bool().any(-1)
        if quiet.any(): sil_sum = sil_sum + emb[pop[quiet]].detach().float().sum(0); sil_n += int(quiet.sum())
        keep = c_idx >= 0; cand = torch.cat([c_idx[keep], pop])
        prob = pr_or[cand]; new = torch.cat([torch.zeros(int(keep.sum()), dtype=torch.bool, device=dev), torch.ones(mm, dtype=torch.bool, device=dev)])
        fi, si = aosc_select(prob, new, cfg, noise=noise, rng=g)
        c_idx = torch.where(fi < 0, fi, cand[fi.clamp(min=0)]); c_spk = si
    return tot, st


def arrival_sort(target, mask=None):
    """정답 활동 (B, K, N) → 열을 창 안 첫 활동 프레임 순서로(활동 없는 열은 뒤, 원래 순서 유지) — 창 도착 순서 슬롯. E7: 대화 전체 순서 슬롯 창의
    헤드 손실(헤드는 창만 보므로 창 도착 순서를 배운다 — 회의 전체 정체는 AOSC 캐시가 맡는다)."""
    B, K, N = target.shape; act = target > 0
    if mask is not None: act = act & mask.bool()[..., None]
    idx = torch.arange(K, device=target.device)[None, :, None].expand(B, K, N)
    first = torch.where(act, idx, torch.full_like(idx, K)).min(1).values                         # (B, N)
    order = (first * (N + 1) + torch.arange(N, device=target.device)[None]).argsort(-1)
    return torch.gather(target, 2, order[:, None, :].expand(B, K, N))


def align_slots(pred, ref):
    """pred·ref (B, K, N) → pred 의 열을 창마다 ref 와 가장 많이 겹치는 순열 자리로 옮긴 것(N! 순열 — N 4 면 24). E7 학습: 창 도착 순서 헤드 예측을
    대화 전체 순서 슬롯 입력으로 줄 때 — 추론의 AOSC 캐시가 회의 내내 화자 정체를 지키는 것을 흉내 낸다."""
    B, K, N = pred.shape
    O = torch.einsum("bki,bkj->bij", pred.float(), ref.float())                                   # (B, N, N): 예측 열 i 와 정답 열 j 의 겹침
    perms = torch.tensor(list(itertools.permutations(range(N))), device=pred.device)               # (P, N): 예측 i → 자리 perms[p, i]
    best = perms[O[:, torch.arange(N, device=pred.device)[None, :], perms].sum(-1).argmax(-1)]      # (B, N)
    return torch.zeros_like(pred).scatter_(2, best[:, None, :].expand(B, K, N), pred)


def sort_pit_loss(logits, target, mask, alpha: float = 0.5):
    """logits·target (B, K, N), mask (B, K) 1 = 손실에 넣는 프레임 → (loss, sort, pit). 둘 다 (마스크 프레임 × N) 평균 BCE.
    sort = 도착 순서 그대로, pit = 창마다 가장 작은 슬롯 순열. loss = α·sort + (1 − α)·pit."""
    m = mask.float(); n = logits.shape[-1]; den = (m.sum(1) * n).clamp(min=1)                   # (B,)
    pair = F.binary_cross_entropy_with_logits(logits[..., :, None].expand(-1, -1, -1, n), target[..., None, :].expand(-1, -1, n, -1).float(), reduction="none")
    C = (pair * m[..., None, None]).sum(1)                                                     # (B, N 예측, N 정답)
    sort = C.diagonal(dim1=1, dim2=2).sum(-1) / den
    perms = torch.tensor(list(itertools.permutations(range(n))), device=logits.device)           # (P, N)
    pc = C[:, torch.arange(n, device=logits.device)[None, :], perms].sum(-1)                    # (B, P): Σ_i C[i, p(i)]
    pit = pc.min(-1).values / den
    return alpha * sort.mean() + (1 - alpha) * pit.mean(), sort.mean().detach(), pit.mean().detach()


def der_counts(pred, ref, mask=None) -> dict:
    """pred·ref (K, N) 0/1 텐서, mask (K,) 1 = 셈에 넣는 프레임 → dict(ref, miss, fa, conf_arrival, conf_best). 단위는 프레임 × 화자.
    DER = (miss + fa + conf) / ref. arrival = 슬롯 순서 그대로, best = 혼동이 가장 적은 슬롯 순열(miss·fa 는 순열과 무관)."""
    p, r = pred.float(), ref.float()
    if mask is not None: keep = mask.bool(); p, r = p[keep], r[keep]
    n_ref, n_hyp = r.sum(-1), p.sum(-1)
    miss = (n_ref - n_hyp).clamp(min=0).sum(); fa = (n_hyp - n_ref).clamp(min=0).sum(); both = torch.minimum(n_ref, n_hyp).sum()
    N = r.shape[-1]; corr = lambda perm: (r * p[:, list(perm)]).sum()
    best = max(corr(q) for q in itertools.permutations(range(N)))
    return dict(ref=float(n_ref.sum()), miss=float(miss), fa=float(fa), conf_arrival=float(both - corr(range(N))), conf_best=float(both - best))


def load_encoder(ckpt: str, nemotron_dir: Optional[str] = None, device: str = "cuda"):
    """VapAsr 체크포인트 디렉토리(model.safetensors 또는 조각 + index)에서 encoder.enc.* 만 읽어 NemotronOnline 을 만든다(동결·eval). right_context 는 config 의 값."""
    from safetensors import safe_open
    from ..features.online import NemotronOnline
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    idx = os.path.join(ckpt, "model.safetensors.index.json")
    files = sorted(set(json.load(open(idx))["weight_map"].values())) if os.path.exists(idx) else ["model.safetensors"]
    sd = {}
    for f in files:
        with safe_open(os.path.join(ckpt, f), "pt") as h:
            for k in h.keys():
                if k.startswith("encoder.enc."): sd[k[len("encoder.enc."):]] = h.get_tensor(k)
    assert sd, f"인코더 가중치(encoder.enc.*)가 없다: {ckpt}"
    enc = NemotronOnline(right_context=int(cfg.get("encoder_right_context", 0) or 0), path=nemotron_dir)
    missing, unexpected = enc.enc.load_state_dict(sd, strict=False)
    params = {n for n, _ in enc.enc.named_parameters()}
    assert not (set(missing) & params) and not unexpected, f"인코더 키 불일치: 빠짐 {sorted(set(missing) & params)[:5]} · 남음 {unexpected[:5]}"
    enc.set_trainable(False)
    return enc.to(device).eval(), len(sd)


def tap_layers(spec: str, n_layers: int) -> Sequence[int]:
    """'1,3,5' 또는 'all' 또는 'odd' → 층 번호(0 부터, 마지막 층은 스트림 0 으로 늘 들어간다)."""
    if spec == "all": return tuple(range(n_layers - 1))
    if spec == "odd": return tuple(range(1, n_layers - 1, 2))
    t = tuple(int(x) for x in spec.split(",") if x != "")
    assert all(0 <= j < n_layers for j in t), f"탭 층 0–{n_layers - 1}: {t}"
    return t

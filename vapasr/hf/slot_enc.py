"""E15b 슬롯 조건 인코더(2026-11-06, 사용자 "2번부터 진행해." — wiki output-agn-reduction-20261104 7 절: Parakeet + N3D 와의 격차는 대부분 겹친 말 받아쓰기).

Multitalker Parakeet 은 화자 분리 활동을 인코더에 주입(speaker kernel)해 화자마다 조건 지은 특징을 만든다. 여기서는 LLM 디코드는 한 스트림 그대로 두고
인코더 위층만 슬롯마다 갈라 돌린다 — 층 L 의 입력 x 를 (섞인 소리 그대로 1 + 슬롯 N) 벌로 복제하고, 슬롯 s 벌에는 SlotInject 로 그 슬롯 활동을 주입한다.
층 L 부터 끝까지는 (B·(1 + N)) 배치로 돈다. 아래층은 한 번만 돈다. 섞인 소리 벌(0 번)은 주입이 없어 원래 인코더 출력과 같다(E13 오디오 토큰 그대로).
슬롯 벌 출력은 adapter_slotenc 를 거쳐 그 슬롯 태그 자리에 더한다(E10 의 slot_x·tag_embeds X 경로를 그대로 쓴다).

NeMo ConformerEncoder.forward_internal 은 층을 layer(x=, att_mask=, pos_emb=, pad_mask=, cache_last_channel=, cache_last_time=) 키워드로 부른다.
slot_branch 는 층 L 에 forward pre-hook 을 걸어 x 를 복제·주입하고 마스크를 배치에 맞춰 늘리며, 층 L+1.. 에는 늘린 마스크로 바꿔 끼운다(루프 변수는 B 배치 그대로라서).
캐시 스트리밍(cache_last_*)과는 함께 쓰지 않는다(평가는 encode_long 조각 인코딩)."""
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


class SlotInject(nn.Module):
    """x (B, T, d) · 슬롯 활동 a (B, T, N)(0–1) → (B, N, T, d): 슬롯 s 마다 x + W2·gelu(W1·[x ⊙ a_s, x ⊙ a_others]) — a_others = 다른 슬롯 활동 합(최대 1).
    W2 0 초기화 → 붙인 직후 슬롯 벌 = 섞인 소리 벌(출력 그대로). speaker kernel(대상 화자)과 비대상 화자 몫을 함께 준다."""

    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(2 * d, hidden); self.w2 = nn.Linear(hidden, d)
        with torch.no_grad(): self.w2.weight.zero_(); self.w2.bias.zero_()

    def forward(self, x, a):
        a = a.to(x.dtype); at = a.permute(0, 2, 1)[..., None]                                        # (B, N, T, 1)
        ao = (a.sum(-1, keepdim=True) - a).clamp(0, 1).permute(0, 2, 1)[..., None]
        z = torch.cat([x[:, None] * at, x[:, None] * ao], -1)                                        # (B, N, T, 2d)
        return x[:, None] + self.w2(F.gelu(self.w1(z.to(self.w1.weight.dtype)))).to(x.dtype)


def _fit(a, T):
    """(B, T', N) → (B, T, N): 0 채움·자름."""
    return F.pad(a, (0, 0, 0, max(0, T - a.shape[1])))[:, :T]


@contextmanager
def slot_branch(layers, L: int, inject, act):
    """layers(NeMo enc.layers)의 L.. 을 (1 + N) 벌로 돌린다. act (B, ≥T', N) 슬롯 활동(인코더 프레임 = 80 ms, 모자라면 0). yield 하는 dict 에 R(벌 수)이 남는다.
    인코더 출력은 (B·R, D, T') — 벌 r 은 행 b·R + r(b 마다 연속)."""
    assert 0 <= int(L) < len(layers), f"갈라 돌릴 층 {L}: 0–{len(layers) - 1}"
    st = {}
    MK = ("att_mask", "pad_mask")

    def first(m, args, kw):
        assert not args and "x" in kw, "NeMo 층은 키워드(x=…)로 불려야 한다"
        assert kw.get("cache_last_channel") is None, "slot_branch 는 캐시 스트리밍과 함께 쓰지 않는다"
        x = kw["x"]; B, T, d = x.shape
        xs = torch.cat([x[:, None], inject(x, _fit(act.to(x.device), T)).to(x.dtype)], 1)        # (B, R, T, d)
        R = xs.shape[1]; st.update(R=R, B=B)
        for k in MK:
            v = kw.get(k)
            st[k] = v.repeat_interleave(R, 0) if torch.is_tensor(v) and v.dim() and v.shape[0] == B else v
        pe = kw.get("pos_emb"); st["pos_emb"] = pe.repeat_interleave(R, 0) if torch.is_tensor(pe) and pe.dim() and pe.shape[0] == B and B > 1 else pe
        kw = dict(kw, x=xs.reshape(B * R, T, d), **{k: st[k] for k in MK + ("pos_emb",) if k in kw})
        return args, kw

    def rest(m, args, kw):
        return args, dict(kw, **{k: st[k] for k in MK + ("pos_emb",) if k in kw})

    hs = [layers[int(L)].register_forward_pre_hook(first, with_kwargs=True)] + [l.register_forward_pre_hook(rest, with_kwargs=True) for l in list(layers)[int(L) + 1:]]
    try:
        yield st
    finally:
        for h in hs: h.remove()

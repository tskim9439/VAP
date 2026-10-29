"""E10 슬롯 추출기(2026-10-29, 사용자 "E10 구현해놓자" — wiki output-diarization-integration-20261025 10 절): 한 스트림 안에서 슬롯마다 그 화자 쪽 소리를 만든다.

입력: 인코더 마지막 층(내용, 오디오 어댑터와 같은 스트림 0) + 화자 단서 임베딩 + 슬롯 활동 a_s(t)(0–1).
  화자 단서 임베딩은 추출기 자신의 embed(인코더 여러 층 탭의 학습 가중합 → LN → 투영 — DiarBranch.embed 와 같은 꼴, add_slot_audio 가 헤드 값으로 초기화)다.
  헤드와 따로 두어 회의 통째 평가에서 헤드를 바꿔 끼워도(--diar-head-override E7a) 추출기 입력은 학습 때와 같다.
슬롯 조건:
  - 활동 [a_s, others = min(1, Σ_{j≠s} a_j), solo = a_s·(1 − others)] — 지금 그 슬롯이 말하나, 겹침인가.
  - 프로필 p_s(t) = 헤드 임베딩을 그 슬롯 단독 활동 프레임(solo)에서 누적 평균한 것(인과). 회의 통째(AOSC 활동)면 회의 단위 화자 표본이 된다 — 활동이 틀린
    프레임에서도 '그 슬롯의 목소리'를 기준으로 고를 수 있게(SpeakerBeam 의 화자 임베딩 적응과 같은 생각).
  조건 벡터 c_s(t) = Linear([활동 3, 프로필]) → 블록마다 FiLM(LN 뒤 × (1 + γ) + β).
본체: 공유 입력 h(t) = LN·Linear(내용) + LN·Linear(헤드 임베딩) → 슬롯마다(가중치 공유, 슬롯 축을 배치로) 인과 블록 n 개
  [x + DWConv_causal(FiLM(LN x)) → x + FFN(FiLM(LN x))] → LN → Linear(d → LLM 차원, 0 초기화 — 붙인 직후 모델 출력 그대로) → x_s(t).
인과: conv 는 왼쪽만 채운다(수용 영역 = 블록 × (kernel − 1) 프레임), 프로필은 누적 합 — 프레임 t 의 출력은 t 이후를 보지 않는다(스트리밍과 같은 값).
긴 입력(회의 통째)은 seg 프레임 조각으로 돌린다 — 조각마다 수용 영역만큼 앞 프레임을 다시 넣고 프로필 누적 합은 이어 간다(통째와 같은 값, 메모리는 조각 크기).
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class _Block(nn.Module):
    def __init__(self, d: int, kernel: int, ffn: int):
        super().__init__()
        self.k = int(kernel)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.film1, self.film2 = nn.Linear(d, 2 * d), nn.Linear(d, 2 * d)
        self.conv = nn.Conv1d(d, d, self.k, groups=d)
        self.pw = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        for f in (self.film1, self.film2):                                                    # FiLM 0 초기화 → 처음엔 조건 없는 블록
            nn.init.zeros_(f.weight); nn.init.zeros_(f.bias)

    @staticmethod
    def _film(x, gb):
        g, b = gb.chunk(2, -1); return x * (1.0 + g) + b

    def forward(self, x, c):                                                                   # x (M, K, d), c (M, K, d)
        y = self._film(self.ln1(x), self.film1(c))
        y = F.conv1d(F.pad(y.transpose(1, 2), (self.k - 1, 0)), self.conv.weight, self.conv.bias, groups=self.conv.groups).transpose(1, 2)
        x = x + self.pw(F.gelu(y))
        return x + self.ff(self._film(self.ln2(x), self.film2(c)))


class SlotAudio(nn.Module):
    """(내용 (B, K, d_enc), 헤드 임베딩 (B, K, d_emb), 활동 (B, K, N)) → 슬롯 소리 토큰 (B, K, N, d_out). 모듈 설명 참고."""

    def __init__(self, n_streams: int, d_enc: int = 1024, d_emb: int = 512, d: int = 512, blocks: int = 4, kernel: int = 15, ffn: int = 2048, d_out: int = 2048):
        super().__init__()
        self.cfg = dict(n_streams=int(n_streams), d_enc=int(d_enc), d_emb=int(d_emb), d=int(d), blocks=int(blocks), kernel=int(kernel), ffn=int(ffn), d_out=int(d_out))
        self.mix = nn.Parameter(torch.zeros(int(n_streams))); self.ln_t = nn.LayerNorm(d_enc); self.proj_t = nn.Linear(d_enc, d_emb)   # 화자 단서 임베딩(embed)
        self.inp_enc = nn.Sequential(nn.LayerNorm(d_enc), nn.Linear(d_enc, d))
        self.inp_emb = nn.Sequential(nn.LayerNorm(d_emb), nn.Linear(d_emb, d))
        self.cond = nn.Sequential(nn.Linear(3 + d_emb, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([_Block(d, kernel, ffn) for _ in range(int(blocks))])
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, d_out)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def embed(self, feats):
        """feats (B, S, K, d_enc)(인코더 탭 스트림, 0 = 마지막 층 — encode_diar 의 allf) → 화자 단서 임베딩 (B, K, d_emb)."""
        w = torch.softmax(self.mix, 0).to(feats.dtype); return self.proj_t(self.ln_t((feats * w[None, :, None, None]).sum(1)))

    @property
    def receptive(self) -> int:
        return len(self.blocks) * (self.cfg["kernel"] - 1)

    @staticmethod
    def act_features(act):
        """(…, K, N) → (…, K, N, 3) [a_s, others, solo]."""
        a = act.float().clamp(0, 1); oth = (a.sum(-1, keepdim=True) - a).clamp(max=1.0)
        return torch.stack([a, oth, a * (1.0 - oth)], -1)

    def _run(self, h, emb, af, num0, den0):
        """조각 하나: h (B, k, d), emb (B, k, d_emb), af (B, k, N, 3), 누적 상태 num0 (B, N, d_emb)·den0 (B, N) → (출력 (B, k, N, d_out), num, den)."""
        B, k, N, _ = af.shape
        w = af[..., 2].transpose(1, 2)                                                        # (B, N, k) solo
        num = num0[:, :, None] + torch.cumsum(w[..., None] * emb[:, None].float(), 2)       # (B, N, k, d_emb)
        den = den0[:, :, None] + torch.cumsum(w, 2)
        prof = num / den.clamp(min=1.0)[..., None]
        c = self.cond(torch.cat([af.transpose(1, 2), prof.to(af.dtype)], -1))                  # (B, N, k, d)
        x = h[:, None].expand(B, N, k, h.shape[-1]).reshape(B * N, k, -1)
        c = c.reshape(B * N, k, -1).to(x.dtype)
        for blk in self.blocks: x = blk(x, c)
        y = self.out(self.norm(x)).reshape(B, N, k, -1).transpose(1, 2)
        return y, num[:, :, -1], den[:, :, -1]

    def forward(self, enc, emb, act, seg: int = 0):
        """enc (B, K, d_enc), emb (B, K, d_emb), act (B, K, N) → (B, K, N, d_out). seg > 0 이고 K > seg 면 조각으로(통째와 같은 값)."""
        B, K, N = act.shape
        h = self.inp_enc(enc.to(self.inp_enc[1].weight.dtype)) + self.inp_emb(emb.to(self.inp_emb[1].weight.dtype))
        af = self.act_features(act).to(h.dtype)
        num0 = h.new_zeros(B, N, emb.shape[-1], dtype=torch.float32); den0 = h.new_zeros(B, N, dtype=torch.float32)
        if not seg or K <= seg:
            return self._run(h, emb, af, num0, den0)[0]
        R = self.receptive; outs = []
        for a in range(0, K, seg):
            b = min(K, a + seg); l = max(0, a - R)                                             # 앞 R 프레임은 conv 문맥으로만(누적 합은 이어 간다)
            if l < a:                                                                          # 문맥 프레임을 프로필에 두 번 더하지 않게 상태를 문맥 앞에서 다시 잡는다
                w = af[:, l:a, :, 2].transpose(1, 2)
                num_l = num0 - torch.einsum("bnk,bkd->bnd", w.float(), emb[:, l:a].float()); den_l = den0 - w.float().sum(-1)
            else: num_l, den_l = num0, den0
            y, num_b, den_b = self._run(h[:, l:b], emb[:, l:b], af[:, l:b], num_l, den_l)
            outs.append(y[:, a - l:]); num0, den0 = num_b, den_b
        return torch.cat(outs, 1)

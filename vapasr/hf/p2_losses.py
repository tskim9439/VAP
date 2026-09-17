"""Phase 2 손실 — EOT 두 점 soft target CE 와 lane 활동 헤드 BCE (정본 output-phase2-lane-plan §5·§6·§9).

soft_ce: 라벨 위치별 CE. labels_alt[i] != -100 인 위치(EOT 후보)에서는 w·CE(labels[i]) + (1−w)·CE(labels_alt[i]) (w = p_end). 그 외는 CE(labels[i]).
         NEXT_AUDIO 위치 가중 next_weight, EOT 위치 가중 eot_weight(정본 §5.3: EOT=2) 는 그대로 곱한다. 분모는 가중치 합(기존 forward 와 같은 규약).
activity_bce: audio 위치 hidden → head → (n_audio, R) logits, target activity[b, chunk] ∈ {0,1}, mask 로 유효 청크만. 평균 BCE 와 정확도.
모델·토크나이저 없이 단위 테스트할 수 있도록 순수 텐서 함수로 둔다."""
from typing import Optional, Tuple, Dict
import torch, torch.nn.functional as F

def soft_ce(logits: torch.Tensor, tgt: torch.Tensor, alt: Optional[torch.Tensor], w: Optional[torch.Tensor], next_id: int, next_weight: float, eot_id: int = -1, eot_weight: float = 1.0,
            tag_ids: Optional[torch.Tensor] = None, tag_weight: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    """logits (N,V) float, tgt (N,) long(유효 위치만), alt (N,) long(-100 = 없음), w (N,) float. → (loss, 분해 통계).
    tag_ids/tag_weight: lane 태그·<ONSET> 위치 가중(D1b — 시작 검출이 약해 태그 위치 손실을 키운다)."""
    ce = F.cross_entropy(logits, tgt, reduction="none")
    if alt is not None:
        has = alt != -100
        if has.any():
            ce_alt = F.cross_entropy(logits[has], alt[has], reduction="none"); ww = w[has].to(ce.dtype)
            ce = ce.clone(); ce[has] = ww * ce[has] + (1.0 - ww) * ce_alt
    na = tgt == next_id; ne = (tgt == eot_id) if eot_id >= 0 else torch.zeros_like(na)
    nt = torch.isin(tgt, tag_ids.to(tgt.device)) if tag_ids is not None and tag_ids.numel() else torch.zeros_like(na)
    wt = torch.ones_like(ce); wt[na] = next_weight; wt[ne] = eot_weight; wt[nt] = tag_weight
    loss = (ce * wt).sum() / wt.sum().clamp(min=1)
    tx = ~na & ~ne
    with torch.no_grad():
        stats = dict(loss_next=ce[na].mean().item() if na.any() else 0.0, loss_text=ce[tx].mean().item() if tx.any() else 0.0, loss_eot=ce[ne].mean().item() if ne.any() else 0.0,
                     top1_text=((logits.argmax(-1) == tgt) & tx).sum().item() / max(1, int(tx.sum())), n_soft=int((alt != -100).sum()) if alt is not None else 0, n_eot=int(ne.sum()),
                     loss_tag=ce[nt].mean().item() if nt.any() else 0.0, top1_tag=((logits.argmax(-1) == tgt) & nt).sum().item() / max(1, int(nt.sum())), n_tag=int(nt.sum()))
    return loss, stats

def activity_bce(act_logits: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """act_logits (n, R), target (n, R) ∈ {0,1} — 이미 유효 청크만 모은 뒤 호출. → (BCE 평균, {acc, pos_rate})."""
    if act_logits.numel() == 0: return act_logits.sum() * 0.0, dict(act_acc=0.0, act_pos=0.0)
    loss = F.binary_cross_entropy_with_logits(act_logits.float(), target.float())
    with torch.no_grad(): acc = ((act_logits > 0) == (target > 0.5)).float().mean().item(); pos = target.float().mean().item()
    return loss, dict(act_acc=acc, act_pos=pos)

def gather_audio_targets(h: torch.Tensor, is_audio: torch.Tensor, chunk_of: torch.Tensor, activity: torch.Tensor, activity_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """h (B,L,D), is_audio (B,L), chunk_of (B,L), activity (B,K,R), activity_mask (B,K) → (h_audio (n,D), target (n,R)) — 유효(mask=1) 청크의 audio 위치만."""
    b_idx, l_idx = torch.nonzero(is_audio & (chunk_of >= 0), as_tuple=True); k = chunk_of[b_idx, l_idx]
    k = k.clamp(max=activity.shape[1] - 1); valid = activity_mask[b_idx, k] > 0.5
    b_idx, l_idx, k = b_idx[valid], l_idx[valid], k[valid]
    return h[b_idx, l_idx], activity[b_idx, k]

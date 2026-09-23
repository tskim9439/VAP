"""Phase 2 손실 — EOT 두 점 soft target CE 와 lane 활동 헤드 BCE (정본 output-phase2-lane-plan §5·§6·§9).

soft_ce: 라벨 위치별 CE. labels_alt[i] != -100 인 위치(EOT 후보)에서는 w·CE(labels[i]) + (1−w)·CE(labels_alt[i]) (w = p_end). 그 외는 CE(labels[i]).
         NEXT_AUDIO 위치 가중 next_weight, EOT 위치 가중 eot_weight(정본 §5.3: EOT=2) 는 그대로 곱한다. 분모는 가중치 합(기존 forward 와 같은 규약).
         semantic commit: <SEM_END>/<TURN_END> 타깃 가중(evt_weights)과 위치별 덮어쓰기(pos_override, hard negative) — 기본 None 이면 기존과 동일.
activity_bce: audio 위치 hidden → head → (n_audio, R) logits, target activity[b, chunk] ∈ {0,1}, mask 로 유효 청크만. 평균 BCE 와 정확도.
모델·토크나이저 없이 단위 테스트할 수 있도록 순수 텐서 함수로 둔다."""
from typing import Optional, Tuple, Dict
import torch, torch.nn.functional as F

def soft_ce(logits: torch.Tensor, tgt: torch.Tensor, alt: Optional[torch.Tensor], w: Optional[torch.Tensor], next_id: int, next_weight: float, eot_id: int = -1, eot_weight: float = 1.0,
            tag_ids: Optional[torch.Tensor] = None, tag_weight: float = 1.0, evt_weights: Optional[Dict[int, float]] = None, pos_override: Optional[torch.Tensor] = None,
            sem_id: int = -1, turn_id: int = -1) -> Tuple[torch.Tensor, Dict[str, float]]:
    """logits (N,V) float, tgt (N,) long(유효 위치만), alt (N,) long(-100 = 없음), w (N,) float. → (loss, 분해 통계).
    tag_ids/tag_weight: lane 태그·<ONSET> 위치 가중(D1b — 시작 검출이 약해 태그 위치 손실을 키운다).
    semantic commit(plan §13): evt_weights {id: w} — 그 id 가 타깃인 위치 가중(<SEM_END>=sem_weight, <TURN_END>=turn_weight), text 통계(tx)에서 제외.
    pos_override (N,) — 위치별 가중 덮어쓰기(>0 이면 그 값, 0 이면 위 기본 가중; hard negative 결정 위치를 next_weight 대신 1.0 으로). 우선순위: 기본 < NEXT/EOT/tag < evt < pos_override.
    sem_id/turn_id ≥ 0 이면 통계 loss_sem·top1_sem·n_sem·sem_fp(SEM 이 타깃이 아닌 위치 중 argmax=SEM 비율)·loss_turn·top1_turn·n_turn 을 더한다.
    새 인자가 모두 기본값이면 손실·통계가 이전과 같다."""
    ce = F.cross_entropy(logits, tgt, reduction="none")
    if alt is not None:
        has = alt != -100
        if has.any():
            ce_alt = F.cross_entropy(logits[has], alt[has], reduction="none"); ww = w[has].to(ce.dtype)
            ce = ce.clone(); ce[has] = ww * ce[has] + (1.0 - ww) * ce_alt
    na = tgt == next_id; ne = (tgt == eot_id) if eot_id >= 0 else torch.zeros_like(na)
    nt = torch.isin(tgt, tag_ids.to(tgt.device)) if tag_ids is not None and tag_ids.numel() else torch.zeros_like(na)
    wt = torch.ones_like(ce); wt[na] = next_weight; wt[ne] = eot_weight; wt[nt] = tag_weight
    nv = torch.zeros_like(na)                                                                   # 이벤트(SEM/TURN) 타깃 위치
    for i, x in (evt_weights or {}).items(): m = tgt == int(i); wt[m] = float(x); nv |= m
    for i in (sem_id, turn_id):
        if i >= 0: nv |= tgt == i
    if pos_override is not None: po = pos_override.to(device=wt.device, dtype=wt.dtype); wt = torch.where(po > 0, po, wt)
    loss = (ce * wt).sum() / wt.sum().clamp(min=1)
    tx = ~na & ~ne & ~nv
    with torch.no_grad():
        pred = logits.argmax(-1); hit = pred == tgt
        stats = dict(loss_next=ce[na].mean().item() if na.any() else 0.0, loss_text=ce[tx].mean().item() if tx.any() else 0.0, loss_eot=ce[ne].mean().item() if ne.any() else 0.0,
                     top1_text=(hit & tx).sum().item() / max(1, int(tx.sum())), n_soft=int((alt != -100).sum()) if alt is not None else 0, n_eot=int(ne.sum()),
                     loss_tag=ce[nt].mean().item() if nt.any() else 0.0, top1_tag=(hit & nt).sum().item() / max(1, int(nt.sum())), n_tag=int(nt.sum()))
        if sem_id >= 0:
            ns = tgt == sem_id
            stats.update(loss_sem=ce[ns].mean().item() if ns.any() else 0.0, top1_sem=(hit & ns).sum().item() / max(1, int(ns.sum())), n_sem=int(ns.sum()),
                         sem_fp=((pred == sem_id) & ~ns).sum().item() / max(1, int((~ns).sum())))
        if turn_id >= 0:
            nu = tgt == turn_id
            stats.update(loss_turn=ce[nu].mean().item() if nu.any() else 0.0, top1_turn=(hit & nu).sum().item() / max(1, int(nu.sum())), n_turn=int(nu.sum()))
        if pos_override is not None: stats["n_override"] = int((pos_override > 0).sum())
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

"""Stage 3 학습 도우미(experiments/stage3_train.py) — 셋 가중 라운드로빈, 대화/replay 공용 collate, 학습 지문.

WeightedRoundRobinLoader: RoundRobinLoader(언어별 셋, 토큰 예산 배치·packing)와 같은 DataLoader 들을 쓰되, step 마다 셋을 가중치로 고른다
  (smooth weighted round-robin — 결정적이라 모든 rank 가 같은 순서로 같은 셋을 고른다). 파일럿에서 대화·replay·언어 비율을 비교하려고 둔다.
  epoch 길이 = 기준 셋(anchor)의 배치 수 합 / 기준 셋의 가중 비중 — 기준 셋을 한 번 도는 동안 다른 셋은 가중치만큼 돈다(작은 셋은 새 epoch 순열로 반복).
collate_stage3: 배치 = 한 셋. 대화 행(cst_dialogue, labels_alt 가 있다)은 collate_cst, replay 행(AstStreamDataset)은 collate_semcommit(sem_mask).
set_kind: 셋 이름(--weights 키) 분류 — 교차 언어 합성은 sim_<l1>-<l2> 에 접미사(-tel 등, V3-p1 sim_en-de-tel)를 붙일 수 있다.
"""
import hashlib
import json
import os
import re
from typing import Dict, Optional, Sequence

from .data import RoundRobinLoader

XL_KEY = re.compile(r"sim_[a-z]{2}-[a-z]{2}(?:-[a-z0-9]+)?")                               # 교차 언어 합성 셋(예 sim_en-de · sim_ko-en · sim_en-de-tel)


def set_kind(key: str) -> Optional[str]:
    """셋 이름 → 'xl'(교차 언어 합성 sim_<l1>-<l2>[-접미사]) | 'sim'(같은 언어 합성 sim_*) | 'dlg'(실제 대화) | 'rep'(단일 화자 replay) | None."""
    if XL_KEY.fullmatch(key): return "xl"
    return next((k for k in ("sim", "dlg", "rep") if key.startswith(k + "_")), None)


def parse_set_phone(spec: str) -> Dict[str, float]:
    """--set-phone 'KEY:P[,KEY:P]' → {셋: 창 단위 전화 대역 확률}(0 = 그 셋은 끔). 대화·합성 셋(dlg_*·sim_*)만, P ∈ [0, 1], 같은 셋 두 번은 ValueError."""
    out: Dict[str, float] = {}
    for x in (spec or "").split(","):
        if not x.strip(): continue
        k, _, v = x.partition(":"); k = k.strip()
        if set_kind(k) not in ("dlg", "sim", "xl") or not v.strip(): raise ValueError(f"--set-phone 형식 'dlg_*|sim_*:P' — {x!r}")
        p = float(v)
        if not 0.0 <= p <= 1.0 or k in out: raise ValueError(f"--set-phone P ∈ [0, 1], 셋마다 한 번 — {x!r}")
        out[k] = p
    return out


def parse_weights(spec: str) -> Dict[str, float]:
    """'dlg_ko:1,rep_en:0.5' → {'dlg_ko': 1.0, 'rep_en': 0.5}. 음수·형식 오류는 ValueError."""
    out: Dict[str, float] = {}
    for x in (spec or "").split(","):
        if not x.strip(): continue
        k, _, v = x.partition(":")
        if not k.strip() or not v.strip(): raise ValueError(f"가중치 형식 'name:w' — {x!r}")
        w = float(v)
        if w < 0: raise ValueError(f"가중치는 0 이상 — {x!r}")
        out[k.strip()] = w
    return out


def weighted_order(weights: Dict[str, float], n: int, names: Optional[Sequence[str]] = None):
    """smooth weighted round-robin 으로 n 개 이름 순서(가중치 0 은 나오지 않는다). 같은 입력이면 같은 순서."""
    names = [m for m in (names or list(weights)) if weights.get(m, 0) > 0]
    if not names: raise ValueError("가중치 > 0 인 셋이 없다")
    tot = sum(weights[m] for m in names); cur = {m: 0.0 for m in names}; out = []
    for _ in range(n):
        for m in names: cur[m] += weights[m]
        m = max(names, key=lambda x: (cur[x], -names.index(x))); cur[m] -= tot; out.append(m)
    return out


class WeightedRoundRobinLoader(RoundRobinLoader):
    """RoundRobinLoader + 셋 가중치(weights: 셋 이름 → 가중, 없는 셋은 0 = 쓰지 않음). anchor: epoch 길이를 정하는 셋 이름들."""

    def __init__(self, datasets, bs_of, weights: Dict[str, float], anchor: Sequence[str], **kw):
        unknown = sorted(set(weights) - set(datasets))
        if unknown: raise ValueError(f"가중치에 없는 셋 이름: {unknown} (셋: {sorted(datasets)})")
        super().__init__(datasets, bs_of, schedule="balanced", **kw)
        self.weights = {m: float(weights.get(m, 0.0)) for m in self.names}
        act = [m for m in self.names if self.weights[m] > 0]
        anchor = [m for m in anchor if self.weights.get(m, 0) > 0]
        if not act or not anchor: raise ValueError(f"가중치 > 0 인 셋·기준 셋이 필요하다: weights {self.weights}, anchor {anchor}")
        share = sum(self.weights[m] for m in anchor) / sum(self.weights[m] for m in act)
        self._n = max(1, int(round(sum(len(self.samplers[m]) for m in anchor) / share)))
        self.anchor = list(anchor)

    def plan(self, n: Optional[int] = None):
        return weighted_order(self.weights, self._n if n is None else n, self.names)

    def __iter__(self):
        its = {m: self._cycle(m, self.epoch) for m in self.names if self.weights[m] > 0}
        for m in self.plan(): yield next(its[m])

    def describe(self) -> dict:
        plan = self.plan(); n = len(plan)
        return dict(steps_per_epoch=n, weights=self.weights, anchor=self.anchor, share={m: round(plan.count(m) / n, 4) for m in self.names},
                    batches_per_epoch={m: len(s) for m, s in self.samplers.items()})


def collate_stage3(batch):
    """배치 = 한 셋. 대화 행 → collate_cst(labels_alt·soft_w_full·sem_only_mask), replay 행 → collate_semcommit(pos_weight·sem_mask)."""
    if "labels_alt" in batch[0]:
        from ..data.cst_dialogue import collate_cst
        return collate_cst(batch)
    from ..data.semcommit_dataset import collate_semcommit
    return collate_semcommit(batch)


def file_digest(paths: Sequence[str]) -> str:
    """경로·크기 목록의 sha256(내용은 읽지 않는다 — 수천 개 샤드도 빠르게). 없는 파일은 크기 −1."""
    rows = [(p, os.path.getsize(p) if os.path.exists(p) else -1) for p in sorted(paths)]
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()


def fingerprint_diff(ckpt_dir: str, fp: dict, key: str = "stage3") -> list:
    """체크포인트 config.json 의 key 와 지금 지문의 차이(키 목록). 체크포인트에 지문이 없으면 ['<missing>']."""
    p = os.path.join(ckpt_dir, "config.json")
    old = json.load(open(p)).get(key) if os.path.exists(p) else None
    if old is None: return ["<missing>"]
    return sorted(k for k in set(old) | set(fp) if old.get(k) != fp.get(k))

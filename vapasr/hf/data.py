"""Trainer 용 데이터 로더 — manifest(언어)별 MonoStreamDataset 을 각각 길이 버킷 배치(BucketBatchSampler, rank 분할)로 만들고 step 마다 번갈아 낸다.
기존 s1_train_mono 의 `its[step % len(its)]` 교대와 같다(EN/KO 를 optimizer step 마다 교대, 배치 크기는 언어별). epoch 길이 = 모든 셋의 rank 당 배치 수 합."""
import random
from typing import Dict, List
import torch
from torch.utils.data import DataLoader
from ..uslm.mono_data import MonoStreamDataset, BucketBatchSampler, collate_streams

class RoundRobinLoader:
    """여러 (이름 → DataLoader) 를 라운드로빈으로 돈다. 한 셋이 먼저 끝나면 그 셋은 새 epoch 순열로 이어서 낸다(모든 셋이 같은 수의 배치를 내지 않아도 step 수 = 합).
    Trainer 가 요구하는 것: __len__(epoch 당 step 수), __iter__, set_epoch(rank 간 동일한 셔플)."""
    def __init__(self, datasets: Dict[str, MonoStreamDataset], bs_of, seed: int = 0, rank: int = 0, world: int = 1, num_workers: int = 4, drop_last: bool = True):
        self.names = list(datasets); self.samplers = {m: BucketBatchSampler(ds, min(bs_of(m), len(ds)), seed=seed, drop_last=drop_last, rank=rank, world=world) for m, ds in datasets.items()}
        self.loaders = {m: DataLoader(ds, batch_sampler=self.samplers[m], num_workers=num_workers, collate_fn=collate_streams, persistent_workers=False) for m, ds in datasets.items()}
        self.epoch = 0; self._n = sum(len(s) for s in self.samplers.values())
    def set_epoch(self, e: int): self.epoch = e
    def __len__(self): return self._n
    def _cycle(self, m, ep0):
        ep = ep0
        while True:
            self.samplers[m].set_epoch(ep)
            for b in self.loaders[m]: yield b
            ep += 1
    def __iter__(self):
        its = [self._cycle(m, self.epoch) for m in self.names]
        for i in range(self._n): yield next(its[i % len(its)])
    @property
    def batch_size(self): return None                                            # Trainer 의 total_train_batch_size 계산은 args 로 한다
    @property
    def dataset(self): return next(iter(self.loaders.values())).dataset

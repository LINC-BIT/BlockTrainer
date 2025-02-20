# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# domainbed/lib/fast_data_loader.py

import torch
from .datasets.ab_dataset import ABDataset


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers, collate_fn=None):
        super().__init__()

        if weights:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights, replacement=True, num_samples=batch_size
            )
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=True
        )

        if collate_fn is not None:
            self._infinite_iterator = iter(
                torch.utils.data.DataLoader(
                    dataset,
                    num_workers=num_workers,
                    batch_sampler=_InfiniteSampler(batch_sampler),
                    pin_memory=False,
                    collate_fn=collate_fn
                )
            )
        else:
            self._infinite_iterator = iter(
                torch.utils.data.DataLoader(
                    dataset,
                    num_workers=num_workers,
                    batch_sampler=_InfiniteSampler(batch_sampler),
                    pin_memory=False
                )
            )
        self.dataset = dataset

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        raise ValueError


class FastDataLoader:
    """
    DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch.
    """

    def __init__(self, dataset, batch_size, num_workers, shuffle=False, collate_fn=None):
        super().__init__()
        
        self.num_workers = num_workers

        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False,
        )
        if collate_fn is not None:
            self._infinite_iterator = iter(
                torch.utils.data.DataLoader(
                    dataset,
                    num_workers=num_workers,
                    batch_sampler=_InfiniteSampler(batch_sampler),
                    pin_memory=False,
                    collate_fn=collate_fn
                )
            )
        else:
            self._infinite_iterator = iter(
                torch.utils.data.DataLoader(
                    dataset,
                    num_workers=num_workers,
                    batch_sampler=_InfiniteSampler(batch_sampler),
                    pin_memory=False,
                )
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length


def build_dataloader(dataset: ABDataset, batch_size: int, num_workers: int, infinite: bool, shuffle_when_finite: bool, collate_fn=None):
    assert batch_size <= len(dataset), len(dataset)
    if infinite:
        dataloader = InfiniteDataLoader(
            dataset, None, batch_size, num_workers=num_workers, collate_fn=collate_fn)
    else:
        dataloader = FastDataLoader(
            dataset, batch_size, num_workers, shuffle=shuffle_when_finite, collate_fn=collate_fn)

    return dataloader


def get_a_batch_dataloader(dataset: ABDataset, batch_size: int, num_workers: int, infinite: bool, shuffle_when_finite: bool):
    pass
    
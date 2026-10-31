import random
import torch
from torch.utils.data import DataLoader, Sampler
import pytorch_lightning as pl
from unifiedgroupgen.pl_data.dataset import batch_indices, CrystalDataset

def collate_records(records):
    return records

class OrbitBatchSampler(Sampler):
    def __init__(self, records, batch_size, edge_budget, seed, epoch, shuffle):
        self.records, self.batch_size, self.edge_budget = records, batch_size, edge_budget
        self.seed, self.epoch, self.shuffle = seed, epoch, shuffle
        self.cached_epoch, self.batches = None, None

    def _batches(self):
        epoch = self.epoch() if self.shuffle else 0
        if self.cached_epoch != epoch:
            self.batches = batch_indices(self.records, self.batch_size, self.edge_budget, self.shuffle,
                                         rng=random.Random(self.seed+epoch))
            self.cached_epoch = epoch
        return self.batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())

class CrystalDataModule(pl.LightningDataModule):
    def __init__(self, records, validation, runtime, seed=42, test_records=None):
        super().__init__()
        self.records = CrystalDataset(records)
        self.validation = CrystalDataset(validation)
        self.test_records = CrystalDataset(test_records) if test_records is not None else None
        self.runtime, self.seed = runtime, seed
        # Fail before training when even one structure cannot fit its edge cap.
        for rows in (self.records, self.validation):
            batch_indices(rows, runtime['batch_size'], runtime['edge_budget'], shuffle=False)

    def _loader(self, records, shuffle):
        sampler = OrbitBatchSampler(records, self.runtime['batch_size'], self.runtime['edge_budget'],
                                    self.seed, lambda: self.trainer.current_epoch if self.trainer else 0, shuffle)
        return DataLoader(records, batch_sampler=sampler, num_workers=0, collate_fn=collate_records,
                          generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        return self._loader(self.records, True)

    def val_dataloader(self):
        return self._loader(self.validation, False)

    def test_dataloader(self):
        if self.test_records is None:
            raise ValueError('Explicit held-out test_records are required; validation is not the test set')
        return self._loader(self.test_records, False)

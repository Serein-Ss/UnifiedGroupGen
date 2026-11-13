"""CPU target cache and atom-budget batches, including explicit operator records."""
import random
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from unifiedgroupgen.pl_data.joint_path import target_from_record


class JointDataset(Dataset):
    def __init__(self, records, operation_compiler=False):
        self.records = [{**row, '_joint_target': target_from_record(row, operation_compiler=operation_compiler)} for row in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class JointBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, edge_budget, epoch, shuffle):
        if not isinstance(batch_size, int) or batch_size < 1 or not isinstance(edge_budget, int) or edge_budget < 1:
            raise ValueError('Joint batch size and edge budget must be positive integers')
        self.sizes = [row['_joint_target'].state.num_atoms for row in dataset.records]
        if any(n*(n-1) > edge_budget for n in self.sizes):
            raise ValueError('A joint target exceeds the edge budget')
        self.batch_size, self.edge_budget, self.epoch, self.shuffle = batch_size, edge_budget, epoch, shuffle

    def _batches(self):
        order = list(range(len(self.sizes)))
        if self.shuffle:
            random.Random(42+self.epoch()).shuffle(order)
        batches, batch, edges = [], [], 0
        for index in order:
            required = self.sizes[index]*(self.sizes[index]-1)
            if batch and (len(batch) >= self.batch_size or edges+required > self.edge_budget):
                batches.append(batch); batch, edges = [], 0
            batch.append(index); edges += required
        if batch:
            batches.append(batch)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())


def _collate(records):
    return records


class JointDataLoader(DataLoader):
    """Native Lightning stateful-loader protocol; finite batches and no workers."""
    def __iter__(self):
        epoch = self.batch_sampler.epoch()
        saved = getattr(self, '_resume', None)
        start = saved['cursor'] if saved is not None and saved['epoch'] == epoch else 0
        self._resume = None
        self._epoch, self._cursor = epoch, start
        for index, batch in enumerate(super().__iter__()):
            if index < start:
                continue
            self._cursor = index+1
            yield batch

    def state_dict(self):
        return dict(epoch=getattr(self, '_epoch', self.batch_sampler.epoch()),
                    cursor=getattr(self, '_cursor', 0))

    def load_state_dict(self, state):
        self._resume = dict(state)


class JointDataModule(pl.LightningDataModule):
    def __init__(self, records, validation, runtime, operation_compiler=False):
        super().__init__()
        self.records = JointDataset(records, operation_compiler)
        self.validation = JointDataset(validation, operation_compiler)
        self.runtime = runtime

    def _loader(self, dataset, shuffle):
        sampler = JointBatchSampler(dataset, self.runtime['batch_size'], self.runtime['edge_budget'],
                                    lambda: self.trainer.current_epoch if self.trainer else 0, shuffle)
        return JointDataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=_collate,
                               generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return self._loader(self.records, True)

    def val_dataloader(self):
        return self._loader(self.validation, False)

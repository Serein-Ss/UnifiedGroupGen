"""Fixed-size PyG batches with a resumable Lightning data loader."""
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch_geometric.data import Batch
from unifiedgroupgen.pl_data.dataset import CrystalDataset
from unifiedgroupgen.pl_modules.debug import stage


def _collate(graphs):
    with stage('data.collate'):
        batch=Batch.from_data_list(graphs)
    object.__setattr__(batch,'records',[graph.record for graph in graphs])
    return batch


class JointDataLoader(DataLoader):
    """Native Lightning stateful-loader protocol; finite batches and no workers."""
    def __init__(self, dataset, *, epoch, **kwargs):
        super().__init__(dataset, **kwargs)
        self.epoch = epoch

    def __iter__(self):
        epoch = self.epoch()
        if isinstance(self.sampler, RandomSampler):
            self.sampler.generator.manual_seed(42 + epoch)
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
        return dict(epoch=getattr(self, '_epoch', self.epoch()),
                    cursor=getattr(self, '_cursor', 0))

    def load_state_dict(self, state):
        self._resume = dict(state)


class JointDataModule(pl.LightningDataModule):
    def __init__(self, records, validation, runtime, operation_compiler=False):
        super().__init__()
        self.records = CrystalDataset(records, operation_compiler)
        self.validation = CrystalDataset(validation, operation_compiler)
        self.runtime = runtime

    def _loader(self, dataset, shuffle):
        sampler = RandomSampler(dataset, generator=torch.Generator()) if shuffle else SequentialSampler(dataset)
        return JointDataLoader(dataset, sampler=sampler, batch_size=self.runtime['batch_size'],
                               epoch=lambda: self.trainer.current_epoch if self.trainer else 0,
                               num_workers=0, collate_fn=_collate,
                               generator=torch.Generator().manual_seed(42))

    def train_dataloader(self):
        return self._loader(self.records, True)

    def val_dataloader(self):
        return self._loader(self.validation, False)

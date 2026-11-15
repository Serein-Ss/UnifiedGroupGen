"""Opt-in pytest guard for CPU-only refactor verification sessions."""


def pytest_sessionstart(session):
    import pytorch_lightning as pl
    import torch
    def forbidden(*args, **kwargs):
        raise AssertionError('Refactor verification forbids fit and optimizer parameter updates')
    pl.Trainer.fit=forbidden
    for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):
        cls.step=forbidden

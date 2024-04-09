"""Module with PyTorchTrainer class"""

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .utils import cycle


class PyTorchTrainer:
    """Class for training PyTorch models"""

    def __init__(
        self,
        hparams,
        model: nn.Module,
        train_step_fn: callable = None,
        valid_fn: callable = None,
        train_log_fn: callable = None,
        valid_log_fn: callable = None,
    ):
        self.hparams = hparams
        self.model = model
        self.train_step_fn = train_step_fn
        self.valid_fn = valid_fn
        self.train_log_fn = train_log_fn
        self.valid_log_fn = valid_log_fn

        self.writer = SummaryWriter(log_dir=hparams.log_dir)
        self.step = 0
        self.epoch = 0

    def train(self, dataloader: DataLoader):
        """Train the model"""
        dl_iter = cycle(dataloader)
        self.model.train()

        while True:
            self.step += 1
            loss = self.train_step_fn(self.model, dl_iter)
            self.train_log_fn(self.step, loss)

            if self.step % self.hparams.valid_every == 0:
                valid_stats = self.valid_fn(dl_iter)
                self.valid_log_fn(self.step, valid_stats)

    def _load_checkpoint():
        pass

    def _save_checkpoint():
        pass

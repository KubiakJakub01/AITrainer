"""Module with PyTorchTrainer class"""
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .ddp import global_rank
from .hparams import Hparams
from .utils import cycle, log_info


class PyTorchTrainer:
    """Class for training PyTorch models"""

    def __init__(
        self,
        hparams: Hparams,
        model: nn.Module,
        device: torch.device,
        train_step_fn: Callable,
        valid_fn: Callable,
        train_log_fn: Callable,
        valid_log_fn: Callable,
    ):
        self.hparams = hparams
        self.model = model
        self.device = device
        self.train_step_fn = train_step_fn
        self.valid_fn = valid_fn
        self.train_log_fn = train_log_fn
        self.valid_log_fn = valid_log_fn

        if global_rank():
            self.writer = SummaryWriter(log_dir=self.hparams.log_dir)

        self.step = 1
        self.epoch = 1

        if self.hparams.base_checkpoint:
            log_info('Loading checkpoint from %d', self.hparams.base_checkpoint)
            self.step = self.hparams.base_checkpoint
            self._load_checkpoint()
        else:
            log_info('Starting from scratch')

    def train(self, dataloader: DataLoader):
        """Train the model"""
        dl_iter = cycle(dataloader)
        self.model.to(self.device)
        self.model.train()

        while True:
            self.step += 1
            loss = self.train_step_fn(self.model, dl_iter)
            self.train_log_fn(self.step, loss)

            if self.step % self.hparams.steps_per_ckpt == 0:
                valid_stats = self.valid_fn(dl_iter)
                self.valid_log_fn(self.step, valid_stats)

    def validation(self):
        pass

    def _load_checkpoint(self):
        if not self.hparams.base_checkpoint:
            log_info('No checkpoint to load')
            return

        self.model.load_state_dict(torch.load(self.hparams.base_checkpoint, map_location='cpu'))

    def _save_checkpoint(self):
        pass

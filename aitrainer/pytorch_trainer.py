"""Module with PyTorchTrainer class"""
from collections.abc import Callable

import torch
import torch.distributed
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .ddp import global_leader_only, is_global_leader
from .hparams import Hparams
from .protocols import TrainStepFnProtocol, ValidFnProtocol
from .utils import log_info


class PyTorchTrainer:
    """Class for training PyTorch models"""

    def __init__(
        self,
        hparams: Hparams,
        model: nn.Module,
        train_step_fn: TrainStepFnProtocol,
        valid_fn: ValidFnProtocol,
        train_log_fn: Callable,
        valid_log_fn: Callable,
    ):
        self.hparams = hparams
        self.model = model
        self.device = (
            torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')
        )
        self.train_step_fn = train_step_fn
        self.valid_fn = valid_fn
        self.train_log_fn = train_log_fn
        self.valid_log_fn = valid_log_fn

        if is_global_leader():
            self.writer = SummaryWriter(log_dir=self.hparams.log_dir)

        self.step = 1
        self.epoch = 1

        if self.hparams.base_checkpoint:
            log_info('Loading checkpoint from %d', self.hparams.base_checkpoint)
            self.step = self.hparams.base_checkpoint
            self._load_checkpoint()
        else:
            log_info('Starting from scratch')

    def train(self, train_dl: DataLoader, valid_dl: DataLoader):
        """Train the model"""
        dl_iter = self._cycle(train_dl)
        self.model.to(self.device)
        self.model.train()

        while True:
            if self.step > self.hparams.total_steps:
                self._save_checkpoint()
                self.validation(valid_dl)

            stats = self.train_step_fn(
                model=self.model,
                dl_iter=dl_iter,
                device=self.device,
                hparams=self.hparams,
                step=self.step,
            )
            self.train_log_fn(self.step, stats)

            if self.step % self.hparams.steps_per_ckpt == 0:
                self._save_checkpoint()
                self.validation(valid_dl)

            self.step += 1

    @global_leader_only
    def validation(self, valid_dl: DataLoader):
        """Run validation"""
        self.model.eval()
        output_batch, valid_stats = self.valid_fn(
            model=self.model,
            dl=valid_dl,
            device=self.device,
            hparams=self.hparams,
            step=self.step,
        )
        self.valid_log_fn(self.step, valid_stats, output_batch)

        self.model.train()

    def _load_checkpoint(self):
        """Load a checkpoint from base_checkpoint"""
        if not self.hparams.base_checkpoint:
            log_info('No checkpoint to load')
            return

        self.model.load_state_dict(torch.load(self.hparams.base_checkpoint, map_location='cpu'))

    def _save_checkpoint(self):
        """Save a checkpoint"""
        checkpoint_path = self.hparams.checkpoint_dir / f'{self.step}.pt'
        torch.save(self.model.state_dict(), checkpoint_path)
        log_info('Saved checkpoint to %s', checkpoint_path.as_posix())

    def _cycle(self, dl):
        while True:
            yield from dl
            self.epoch += 1
            if is_global_leader():
                log_info('New epoch: %d', self.epoch)

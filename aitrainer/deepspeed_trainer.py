"""Module with PyTorchTrainer class"""

import deepspeed
import torch
import torch.distributed
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .ddp import global_leader_only, is_global_leader
from .hparams import Hparams
from .protocols import LogFnProtocol, TrainStepFnProtocol, ValidFnProtocol
from .utils import log_info


class Trainer:
    """Class for training PyTorch models"""

    def __init__(
        self,
        hparams: Hparams,
        model_dict: dict[str, nn.Module],
        train_step_fn: TrainStepFnProtocol,
        valid_fn: ValidFnProtocol,
        train_log_fn: LogFnProtocol,
        valid_log_fn: LogFnProtocol,
    ):
        """Initialize the trainer

        Args:
            hparams: Hyperparameters for the training run
            model_dict: Dictionary of models to train
            train_step_fn: Function to run a training step
            valid_fn: Function to run validation
            train_log_fn: Function to log training stats
            valid_log_fn: Function to log validation stats"""
        self.hparams = hparams
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

        self.engine_dict = self._init_engines(model_dict)

        if self.hparams.base_checkpoint:
            log_info('Loading checkpoint from %d', self.hparams.base_checkpoint)
            self.step = self.hparams.base_checkpoint
            self._load_checkpoint()
        else:
            log_info('Starting from scratch')

    def train(self, train_dl: DataLoader, valid_dl: DataLoader):
        """Train the model"""
        dl_iter = self._cycle(train_dl)

        for model in self.engine_dict.values():
            model.to(self.device)
            model.train()

        while True:
            if self.step > self.hparams.total_steps:
                log_info('Training complete')
                self._save_checkpoint()
                self.validation(valid_dl)
                break

            stats = self.train_step_fn(
                engine_dict=self.engine_dict,
                dl_iter=dl_iter,
                hparams=self.hparams,
                step=self.step,
            )
            if is_global_leader():
                self.train_log_fn(
                    writer=self.writer, step=self.step, stats=stats, hparams=self.hparams
                )

            if self.step % self.hparams.steps_per_ckpt == 0:
                self._save_checkpoint()
                self.validation(valid_dl)

            self.step += 1

    @global_leader_only
    def validation(self, valid_dl: DataLoader):
        """Run validation"""
        for model in self.engine_dict.values():
            model.eval()

        output_batch, valid_stats = self.valid_fn(
            engine_dict=self.engine_dict,
            dl=valid_dl,
            hparams=self.hparams,
            step=self.step,
        )
        self.valid_log_fn(
            writer=self.writer,
            step=self.step,
            stats=valid_stats,
            output_batch=output_batch,
            hparams=self.hparams,
        )

        for model in self.engine_dict.values():
            model.train()

    def _init_engines(
        self, model_dict: dict[str, nn.Module]
    ) -> dict[str, deepspeed.DeepSpeedEngine]:
        """Initialize DeepSpeed engines from pytorch models"""
        engine_dict = {}
        for model_name, model in model_dict.items():
            engine_dict[model_name], *_ = deepspeed.initialize(
                model=model,
                model_parameters=model.parameters(),
                config_params=self.hparams.deepspeed_config,
            )
        return engine_dict

    def _load_checkpoint(self):
        """Load a checkpoint from base_checkpoint"""
        if not self.hparams.base_checkpoint:
            log_info('No checkpoint to load')
            return

        for engine in self.engine_dict.values():
            engine.load_checkpoint(self.hparams.checkpoint_dir, tag=self.hparams.base_checkpoint)

    def _save_checkpoint(self):
        """Save a checkpoint"""
        for engine in self.engine_dict.values():
            engine.save_checkpoint(self.hparams.checkpoint_dir, tag=self.step)

    def _cycle(self, dl):
        while True:
            yield from dl
            self.epoch += 1
            if is_global_leader():
                log_info('New epoch: %d', self.epoch)

from collections.abc import Iterator
from typing import Any

import torch
from deepspeed import DeepSpeedEngine
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from aitrainer import log_info, to_device

from .hparams import MLPHparams


class MLP(nn.Module):
    def __init__(self, hparams: MLPHparams):
        """Multi-layer perceptron.

        Args:
            hparams: Hyperparameters for the MLP model.

        Returns:
            Initialized MLP."""
        super().__init__()
        self.hparams = hparams
        self.net = nn.Sequential(
            nn.Linear(self.hparams.input_dim, self.hparams.hidden_dim),
            nn.Dropout(self.hparams.dropout),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim * 2),
            nn.Dropout(self.hparams.dropout),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_dim * 2, self.hparams.hidden_dim),
            nn.Dropout(self.hparams.dropout),
            nn.ReLU(),
            nn.Linear(self.hparams.hidden_dim, self.hparams.output_dim),
        )

    def forward(self, x):
        x = self.net(x)
        return x


def train_step_fn(  # pylint: disable=unused-argument
    engine_dict: dict[str, DeepSpeedEngine],
    dl_iter: Iterator,
    hparams: MLPHparams,
    step: int,
) -> tuple[float, dict]:
    """Train step function for the MLP model"""
    model = engine_dict['model']
    batch = to_device(next(dl_iter), model.device)

    # Forward pass
    loss = model(batch['inputs'])

    # Backward pass
    model.backward(loss)

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.gradient_clipping)

    # Weight update
    model.step()

    return loss.item(), {'grad_norm': grad_norm}


def valid_fn(  # pylint: disable=unused-argument
    engine_dict: dict[str, DeepSpeedEngine],
    dl: Iterator,
    hparams: MLPHparams,
    step: int,
) -> tuple[dict, dict]:
    """Validation function for the MLP model"""
    model = engine_dict['model']
    batch = to_device(next(dl), model.device)

    # Forward pass
    output = model(batch['inputs'])

    return batch, {'output': output}


def log_fn(  # pylint: disable=unused-argument
    hparams: MLPHparams,
    writer: SummaryWriter,
    step: int,
    stats: dict,
    output_batch: dict[str, Any] | None = None,
):
    """Log function for the MLP model"""
    log_info('Step: %d, Loss: %.4f', step, stats['loss'])
    if hparams.steps_per_log and step % hparams.steps_per_log == 0:
        writer.add_scalar('Loss', stats['loss'], step)
        writer.add_scalar('Gradient Norm', stats['grad_norm'], step)


def log_valid_fn(  # pylint: disable=unused-argument
    hparams: MLPHparams,
    writer: SummaryWriter,
    step: int,
    stats: dict,
    output_batch: dict[str, Any] | None = None,
):
    """Log function for the MLP model validation"""
    log_info('Validation: Loss: %.4f', stats['loss'])
    writer.add_scalar('Validation Loss', stats['loss'], step)
    if output_batch:
        writer.add_histogram('Output', output_batch['output'], step)
        writer.add_histogram('Target', output_batch['inputs'], step)
        writer.add_histogram('Error', output_batch['output'] - output_batch['inputs'], step)

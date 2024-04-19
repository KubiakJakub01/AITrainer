from collections.abc import Iterator
from typing import Any

import torch
from deepspeed import DeepSpeedEngine
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from aitrainer.hparams import Hparams
from aitrainer.utils import log_info, to_device


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, std):
        """Multi-layer perceptron.

        Args:
            input_dim: Dimensionality of the input.
            hidden_dim: Dimensionality of the hidden layers.
            output_dim: Dimensionality of the output.
            dropout: Dropout probability.
            std: Standard deviation of the weights.

        Returns:
            Initialized MLP."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            std,
        )

    def forward(self, x):
        x = self.net(x)
        return x


def train_step_fn(  # pylint: disable=unused-argument
    engine_dict: dict[str, DeepSpeedEngine],
    dl_iter: Iterator,
    hparams: Hparams,
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
    hparams: Hparams,
    step: int,
) -> tuple[dict, dict]:
    """Validation function for the MLP model"""
    model = engine_dict['model']
    batch = to_device(next(dl), model.device)

    # Forward pass
    output = model(batch['inputs'])

    return batch, {'output': output}


def log_fn(  # pylint: disable=unused-argument
    hparams: Hparams,
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
    hparams: Hparams,
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

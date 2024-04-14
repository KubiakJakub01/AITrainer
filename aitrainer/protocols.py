from collections.abc import Iterator
from typing import Any, Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .hparams import Hparams


class TrainStepFnProtocol(Protocol):
    def __call__(
        self,
        engine_dict: dict[str, nn.Module],
        dl_iter: Iterator,
        device: torch.device,
        hparams: Hparams,
        step: int,
    ) -> dict[str, Any]:
        ...


class ValidFnProtocol(Protocol):
    def __call__(
        self,
        engine_dict: dict[str, nn.Module],
        dl: DataLoader,
        device: torch.device,
        hparams: Hparams,
        step: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        ...


class LogFnProtocol(Protocol):
    def __call__(
        self,
        hparams: Hparams,
        writer: SummaryWriter,
        step: int,
        stats: dict[str, Any],
        output_batch: dict[str, Any] | None = None,
    ):
        ...

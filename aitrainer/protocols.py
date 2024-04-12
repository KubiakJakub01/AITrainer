from typing import Any, Protocol

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .hparams import Hparams


class TrainStepFnProtocol(Protocol):
    def __call__(
        self, model: nn.Module, dl: DataLoader, device: torch.device, hparams: Hparams, step: int
    ) -> dict[str, Any]:
        ...


class ValidFnProtocol(Protocol):
    def __call__(
        self, model: nn.Module, dl: DataLoader, device: torch.device, hparams: Hparams, step: int
    ) -> dict[str, Any]:
        ...

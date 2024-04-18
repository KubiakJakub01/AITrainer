from collections.abc import Iterator
from typing import Any, Protocol

from deepspeed import DeepSpeedEngine
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .hparams import Hparams


class TrainStepFnProtocol(Protocol):
    def __call__(
        self,
        engine_dict: dict[str, DeepSpeedEngine],
        dl_iter: Iterator,
        hparams: Hparams,
        step: int,
    ) -> dict[str, Any]:
        ...


class ValidFnProtocol(Protocol):
    def __call__(
        self,
        engine_dict: dict[str, DeepSpeedEngine],
        dl: DataLoader,
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

"""Utility functions for the project."""
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import coloredlogs
import torch

from .ddp import global_rank, local_rank

# Set up logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(
    coloredlogs.ColoredFormatter(
        fmt=f'%(asctime)s :: %(levelname)s :: GR={global_rank()};LR={local_rank()} :: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def log_debug(*args, **kwargs):
    """Log an debug message."""
    logger.debug(*args, **kwargs)


def log_info(*args, **kwargs):
    """Log an info message."""
    logger.info(*args, **kwargs)


def log_warning(*args, **kwargs):
    """Log a warning message."""
    logger.warning(*args, **kwargs)


def log_error(*args, **kwargs):
    """Log an error message."""
    logger.error(*args, **kwargs)


def tree_map(fn: Callable, x):
    if isinstance(x, list):
        x = [tree_map(fn, xi) for xi in x]
    elif isinstance(x, tuple):
        x = (tree_map(fn, xi) for xi in x)
    elif isinstance(x, dict):
        x = {k: tree_map(fn, v) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        x = fn(x)
    return x


def to_device(x: dict, device: torch.device):
    return tree_map(lambda t: t.to(device), x)


def load_config(path: Path) -> dict[str, Any]:
    """Load hparams from a file"""
    with open(path, encoding='utf-8') as f:
        if path.suffix == '.json':
            hparams_dict = json.load(f)
        else:
            raise ValueError(f'Unknown file extension: {path.suffix}')

    return hparams_dict

import os
from collections.abc import Callable
from functools import wraps

import deepspeed


def init_distritubed():
    deepspeed.init_distributed(
        dist_backend='nccl',
        init_method='env://',
        rank=local_rank(),
        world_size=get_world_size(),
    )


def get_world_size():
    return int(os.environ.get('WORLD_SIZE', 1))


def local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def global_rank():
    return int(os.environ.get('RANK', 0))


def is_local_leader():
    return local_rank() == 0


def is_global_leader():
    return global_rank() == 0


def local_leader_only(fn) -> Callable:
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if is_local_leader():
            return fn(*args, **kwargs)
        return None

    return wrapped


def global_leader_only(fn) -> Callable:
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if is_global_leader():
            return fn(*args, **kwargs)
        return None

    return wrapped

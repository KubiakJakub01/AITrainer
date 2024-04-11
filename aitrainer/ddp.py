import os
from collections.abc import Callable
from functools import wraps


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

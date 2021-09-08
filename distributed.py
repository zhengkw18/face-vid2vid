import functools
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.backends.cudnn as cudnn


def init_seeds(cuda_deterministic=True):
    seed = 1 + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def init_dist(local_rank, world_size, backend="nccl"):
    r"""Initialize distributed training"""
    torch.autograd.set_detect_anomaly(True)
    if dist.is_available():
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend=backend, init_method="env://", world_size=world_size, rank=local_rank)
    print("Rank", get_rank(), "initialized.")


def get_rank():
    r"""Get rank of the thread."""
    rank = 0
    if dist.is_available():
        if dist.is_initialized():
            rank = dist.get_rank()
    return rank


def get_world_size():
    r"""Get world size. How many GPUs are available in this job."""
    world_size = 1
    if dist.is_available():
        if dist.is_initialized():
            world_size = dist.get_world_size()
    return world_size


def master_only(func):
    r"""Apply this function only to the master GPU."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        r"""Simple function wrapper for the master function"""
        if get_rank() == 0:
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def is_master():
    r"""check if current process is the master"""
    return get_rank() == 0


@master_only
def master_only_print(*args, **kwargs):
    r"""master-only print"""
    print(*args, **kwargs)

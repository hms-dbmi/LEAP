import hashlib
import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic_algorithms: bool = True) -> None:
    """Seed Python, NumPy and PyTorch and switch cuDNN into deterministic mode.

    With `deterministic_algorithms` the run additionally sets PYTHONHASHSEED, disables the
    cuDNN autotuner and asks PyTorch for deterministic kernels. That last setting makes cuBLAS
    require CUBLAS_WORKSPACE_CONFIG=:4096:8 in the environment, so pass False for pipelines
    that are not run with it set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    if deterministic_algorithms:
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def stable_slide_seed(slide_id: str, base_seed: int) -> int:
    """Per-slide RNG seed, independent of process, worker and run order.

    Returns a 32-bit integer derived from (base_seed, slide_id), so a slide always draws
    the same patches at inference time.
    """
    digest = hashlib.md5(f"{base_seed}_{slide_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def worker_init_fn(worker_id: int) -> None:
    """Seed a dataloader worker's NumPy and Python RNGs from the loader's base seed."""
    info = torch.utils.data.get_worker_info()
    base = (info.seed if info is not None else torch.initial_seed()) % (2 ** 32)
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)

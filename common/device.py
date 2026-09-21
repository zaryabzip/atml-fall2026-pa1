"""Device helpers that work on Kaggle (2x T4, CUDA) and on Apple Silicon (MPS).

T4 notes: fp16 autocast + GradScaler gives a large speed-up; bf16 is not supported.
MPS notes: runs in fp32; keep batch sizes modest on 16 GB unified memory.
Two GPUs: running one experiment per GPU (CUDA_VISIBLE_DEVICES=0 / =1) is usually faster
than DataParallel for these small models. DataParallel is available via `data_parallel: true`.
"""
import os
from contextlib import nullcontext

import torch
from torch import nn

from common.seed import make_generator, seed_worker


def get_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_backends(device: torch.device) -> None:
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True


def amp_enabled(device: torch.device, requested: bool = True) -> bool:
    return requested and device.type == "cuda"


def autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=enabled)


def default_num_workers(device: torch.device) -> int:
    cpus = os.cpu_count() or 1
    if device.type == "cuda":
        return min(4, cpus)  # Kaggle gives 4 vCPUs
    return min(4, max(cpus // 2, 0))


def make_loader(dataset, batch_size, device, *, shuffle=False, drop_last=False, seed=None, num_workers=None):
    workers = default_num_workers(device) if num_workers is None else num_workers
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker if workers > 0 else None,
        generator=make_generator(seed) if seed is not None else None,
    )


def maybe_data_parallel(model: nn.Module, device: torch.device, enabled: bool) -> nn.Module:
    if enabled and device.type == "cuda" and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model

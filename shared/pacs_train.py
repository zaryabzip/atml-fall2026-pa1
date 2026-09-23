"""One training loop for every PACS method (Tasks 2 and 3).

A method subclasses PACSMethod and implements `compute_loss` (or overrides `train_step`, e.g. SAM).
The loop handles: seeds, data, frozen-BN train mode, AMP, early stopping on mean source-val
macro-F1, checkpointing, and logs. Keeping this shared guarantees identical initialization,
sampling, augmentation, optimizer and budget across methods.
"""
import torch
from torch import nn

from common.config import save_config
from common.device import (
    amp_enabled, autocast, configure_backends, default_num_workers, get_device, grad_scaler, make_loader,
    maybe_data_parallel, unwrap,
)
from common.io import checkpoint_dir, data_root, results_dir, save_json
from common.logger import JsonlLogger
from common.seed import seed_everything
from shared.pacs_eval import evaluate_sources, val_loaders
from shared.pacs_models import PACSResNet18
from shared.pacs_protocol import (
    SOURCES, balanced_source_batches, cycle, eval_transform, load_splits, set_train_mode_frozen_bn,
    source_datasets, target_dataset, train_transform,
)


class PACSMethod:
    name = "base"
    uses_target = False  # only Task 2 adaptation methods may set this

    def __init__(self, cfg: dict, model: nn.Module, device: torch.device):
        self.cfg = cfg
        self.model = model  # may be wrapped in DataParallel
        self.device = device

    def extra_modules(self):
        """Extra trainable module (e.g. a domain discriminator), optimized and saved with the model."""
        return None

    def compute_loss(self, source: dict, target, progress: float):
        """source: {'x','y','domain'} on device. target: {'x'} on device or None.
        progress: training progress p in [0, 1]. Return (loss, {name: detached tensor})."""
        raise NotImplementedError

    def train_step(self, source, target, optimizer, scaler, amp: bool, progress: float) -> dict:
        optimizer.zero_grad(set_to_none=True)
        with autocast(self.device, amp):
            loss, logs = self.compute_loss(source, target, progress)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return {"loss": loss.detach(), **logs}


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def fit(cfg: dict, method_cls, task: str) -> dict:
    run_name = cfg["run_name"]
    train_cfg = cfg["train"]
    seed = cfg["seed"]
    seed_everything(seed)

    device = get_device(cfg.get("device", "auto"))
    configure_backends(device)
    amp = amp_enabled(device, train_cfg.get("amp", True))
    pacs_root = cfg["data"].get("pacs_root") or data_root() / "pacs"

    ckpt_dir = checkpoint_dir(task, run_name)
    out_dir = results_dir(task, run_name)
    save_config(cfg, out_dir / "config.yaml")
    logger = JsonlLogger(out_dir / "train_log.jsonl")

    splits = load_splits()
    n_loaders = len(SOURCES) + (1 if method_cls.uses_target else 0)
    # Split the CPU budget across the 3 source loaders (+1 target loader)
    workers = train_cfg.get("num_workers_per_loader")
    if workers is None:
        workers = max(1, default_num_workers(device) // n_loaders)
    src_train = source_datasets(pacs_root, "train", train_transform(), splits)
    src_iter = balanced_source_batches(src_train, train_cfg["per_domain_batch"], device, seed, workers)
    val = val_loaders(source_datasets(pacs_root, "val", eval_transform(), splits), train_cfg["eval_batch"], device, workers)

    model = PACSResNet18().to(device)
    net = maybe_data_parallel(model, device, train_cfg.get("data_parallel", False))
    method = method_cls(cfg, net, device)

    target_iter = None
    if method.uses_target:
        if task != "task2":
            raise PermissionError("Only Task 2 may load Sketch during training.")
        tgt = target_dataset(pacs_root, train_transform(), purpose="task2_adaptation")
        target_iter = cycle(make_loader(tgt, train_cfg["target_batch"], device, shuffle=True, drop_last=True,
                                        seed=seed + 100, num_workers=workers))

    extra = method.extra_modules()
    if extra is not None:
        extra.to(device)
    params = list(model.parameters()) + (list(extra.parameters()) if extra is not None else [])
    optimizer = torch.optim.AdamW(params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    scaler = grad_scaler(amp)

    steps = train_cfg.get("steps_per_epoch", "auto")
    if steps == "auto":  # one "source epoch" = pooled source training images / total source batch
        steps = sum(len(ds) for ds in src_train.values()) // (train_cfg["per_domain_batch"] * len(SOURCES))
    total_steps = steps * train_cfg["max_epochs"]

    best_score, bad_epochs, global_step, best_record = -1.0, 0, 0, None
    for epoch in range(1, train_cfg["max_epochs"] + 1):
        set_train_mode_frozen_bn(net)
        if extra is not None:
            extra.train()
        sums = {}
        for _ in range(steps):
            source = _to_device(next(src_iter), device)
            target = _to_device({"x": next(target_iter)[0]}, device) if target_iter else None
            logs = method.train_step(source, target, optimizer, scaler, amp, progress=global_step / total_steps)
            global_step += 1
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v.float()
        train_logs = {f"train/{k}": (v / steps).item() for k, v in sums.items()}

        metrics = evaluate_sources(model, val, device, amp)
        record = {"epoch": epoch, "step": global_step, **train_logs,
                  "val/mean_macro_f1": metrics["mean_macro_f1"], "val/mean_accuracy": metrics["mean_accuracy"],
                  **{f"val/{d}/macro_f1": m["macro_f1"] for d, m in metrics["per_domain"].items()}}
        logger.log(record, echo=True)

        if metrics["mean_macro_f1"] > best_score:
            best_score, bad_epochs, best_record = metrics["mean_macro_f1"], 0, {"epoch": epoch, **metrics}
            torch.save({
                "model": unwrap(model).state_dict(),
                "extra": extra.state_dict() if extra is not None else None,
                "epoch": epoch, "config": cfg, "source_val": metrics,
            }, ckpt_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= train_cfg["patience"]:
                break

    save_json(best_record, out_dir / "source_selection.json")
    return best_record

"""Task 4 training (Vanilla, GCSC, PROSER, optional RPL). Uses CIFAR-10 only.

    python -m task4.train --config task4/configs/vanilla.yaml
Kaggle, two runs at once:
    CUDA_VISIBLE_DEVICES=0 python -m task4.train --config task4/configs/vanilla.yaml &
    CUDA_VISIBLE_DEVICES=1 python -m task4.train --config task4/configs/gcsc.yaml &
"""
import torch

from common.config import config_arg_parser, load_config, save_config
from common.device import (
    amp_enabled, autocast, configure_backends, get_device, grad_scaler, make_loader, maybe_data_parallel, unwrap,
)
from common.io import checkpoint_dir, results_dir, save_json
from common.logger import JsonlLogger
from common.seed import seed_everything
from task4.data.cifar import cifar10_train, cifar10_val, train_transform
from task4.methods import METHODS

NUM_KNOWN = 10


@torch.no_grad()
def known_accuracy(model, loader, device, amp) -> float:
    """Accuracy using only the 10 known-class logits (so PROSER's dummies never count)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast(device, amp):
            logits = model(x)
        correct += (logits[:, :NUM_KNOWN].argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    args = config_arg_parser("Task 4 training").parse_args()
    cfg = load_config(args.config, args.overrides)
    tc = cfg["train"]
    seed_everything(cfg["seed"])
    device = get_device(cfg.get("device", "auto"))
    configure_backends(device)
    amp = amp_enabled(device, tc.get("amp", True))

    out_dir = results_dir("task4", cfg["run_name"])
    ckpt_dir = checkpoint_dir("task4", cfg["run_name"])
    save_config(cfg, out_dir / "config.yaml")
    logger = JsonlLogger(out_dir / "train_log.jsonl")

    train_loader = make_loader(cifar10_train(train_transform(tc["randaugment"])), tc["batch_size"], device,
                               shuffle=True, drop_last=tc["drop_last"], seed=cfg["seed"], num_workers=tc["num_workers"])
    val_loader = make_loader(cifar10_val(), tc["eval_batch"], device, num_workers=tc["num_workers"])

    method_cls = METHODS[cfg["method"]["name"]]
    model = method_cls.build_model(cfg, device).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    net = maybe_data_parallel(model, device, tc.get("data_parallel", False))
    method = method_cls(cfg, net, device)

    optimizer = torch.optim.SGD(model.parameters(), lr=tc["lr"], momentum=tc["momentum"], weight_decay=tc["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc["epochs"])
    scaler = grad_scaler(amp)

    # Resume: if a previous run of this config was interrupted, continue after its last finished epoch.
    best_acc, start_epoch = -1.0, 1
    last_path = ckpt_dir / "last.pt"
    if tc.get("resume", True) and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        if state["config"] != cfg:
            raise SystemExit(f"{last_path} was saved with a different config. Delete it to start over, "
                             "or rerun with the original settings to resume.")
        unwrap(model).load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best_acc, start_epoch = state["best_acc"], state["epoch"] + 1
        print(f"Resuming {cfg['run_name']} after epoch {state['epoch']} (best val acc so far {best_acc:.4f})", flush=True)

    max_steps = tc.get("max_steps")  # preflight checks only: stop each epoch after this many steps
    for epoch in range(start_epoch, tc["epochs"] + 1):
        net.train()
        sums, steps = {}, 0
        for x, y in train_loader:
            if max_steps and steps >= max_steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if device.type == "cuda":
                x = x.contiguous(memory_format=torch.channels_last)
            for k, v in method.train_step(x, y, optimizer, scaler, amp).items():
                sums[k] = sums.get(k, 0.0) + v.float()
            steps += 1
        scheduler.step()

        val_acc = known_accuracy(net, val_loader, device, amp)
        logger.log({"epoch": epoch, "lr": scheduler.get_last_lr()[0],
                    **{f"train/{k}": (v / steps).item() for k, v in sums.items()}, "val/accuracy": val_acc}, echo=True)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model": unwrap(model).state_dict(), "epoch": epoch, "val_accuracy": val_acc,
                        "num_outputs": unwrap(model).net.fc.out_features, "config": cfg}, ckpt_dir / "best.pt")
        torch.save({"model": unwrap(model).state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best_acc": best_acc, "config": cfg}, ckpt_dir / "last.pt")

    save_json({"best_val_accuracy": best_acc}, out_dir / "selection.json")


if __name__ == "__main__":
    main()

import torch
from torch import nn

from common.device import autocast
from task4.models.resnet_cifar import CifarResNet18


class CifarMethod:
    name = "base"

    def __init__(self, cfg: dict, model: nn.Module, device: torch.device):
        self.cfg = cfg
        self.model = model  # may be wrapped in DataParallel
        self.device = device

    @classmethod
    def build_model(cls, cfg: dict, device) -> nn.Module:
        return CifarResNet18(num_classes=10)

    def compute_loss(self, x, y):
        raise NotImplementedError

    def train_step(self, x, y, optimizer, scaler, amp: bool) -> dict:
        optimizer.zero_grad(set_to_none=True)
        with autocast(self.device, amp):
            loss, logs = self.compute_loss(x, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return {"loss": loss.detach(), **logs}

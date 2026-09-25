import torch
import torch.nn.functional as F

from common.device import autocast
from shared.pacs_train import PACSMethod


class SAM(PACSMethod):
    """Non-adaptive SAM on the ERM loss, with AdamW as the base optimizer. Two passes per batch."""
    name = "sam"
    uses_target = False

    def compute_loss(self, source, target, progress):
        # Plain cross-entropy on the source batch, identical to ERM. Both SAM passes use it.
        logits = self.model(source["x"])
        loss = F.cross_entropy(logits, source["y"])
        return loss, {"cls_loss": loss.detach()}

    def train_step(self, source, target, optimizer, scaler, amp, progress):
        """One SAM update: (1) gradient g at the current weights, (2) climb to theta + rho * g/||g||,
        (3) gradient there, (4) go back to theta and step with that gradient.

        Loss-scaling note (only matters when AMP is on, i.e. CUDA): pass 1's gradients are unscaled
        by hand rather than with scaler.unscale_. unscale_ would mark the optimizer as already
        unscaled, and scaler.step would then skip unscaling pass 2's gradients, feeding it scaled ones."""
        rho = float(self.cfg["method"]["rho"])  # float(): an override like rho=1e-2 is parsed by YAML as a string
        params = [p for p in self.model.parameters() if p.requires_grad]

        # Pass 1: loss and gradient at the current weights theta.
        optimizer.zero_grad(set_to_none=True)
        with autocast(self.device, amp):
            loss, _ = self.compute_loss(source, target, progress)
        scaler.scale(loss).backward()

        # Turn the scaled gradients into real ones, then take their global L2 norm over all parameters.
        inv_scale = 1.0 / scaler.get_scale()  # exactly 1.0 when the scaler is disabled (MPS / CPU)
        for p in params:
            if p.grad is not None:
                p.grad.mul_(inv_scale)
        grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in params if p.grad is not None]), 2)

        # A non-finite gradient (fp16 overflow, or a diverging run) means this step can't be trusted, so
        # skip it. With an enabled scaler use its own skip path (unscale_ records the inf, step skips the
        # optimizer, update backs the scale off AND resets the growth counter). A disabled scaler would
        # call optimizer.step() no matter what, so there we just drop the gradients. Either way the skip
        # is logged, so a run that keeps skipping is visible in the training log instead of silently frozen.
        if not torch.isfinite(grad_norm):
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                scaler.step(optimizer)
                scaler.update()
            optimizer.zero_grad(set_to_none=True)
            return {"loss": loss.detach(), "cls_loss": loss.detach(), "perturbed_loss": loss.detach(),
                    "skipped_step": torch.ones((), device=loss.device)}

        # Climb: move every parameter by rho * g / ||g||, remembering the original weights exactly.
        originals = [p.detach().clone() for p in params]
        with torch.no_grad():
            step_size = rho / (grad_norm + 1e-12)
            for p in params:
                if p.grad is not None:
                    p.add_(p.grad * step_size)

        # Pass 2: loss and gradient at the perturbed weights theta + e.
        optimizer.zero_grad(set_to_none=True)
        with autocast(self.device, amp):
            perturbed_loss, _ = self.compute_loss(source, target, progress)
        scaler.scale(perturbed_loss).backward()

        # Restore theta, then update it using the gradient computed at theta + e.
        with torch.no_grad():
            for p, original in zip(params, originals):
                p.copy_(original)
        scaler.step(optimizer)
        scaler.update()

        return {"loss": loss.detach(), "cls_loss": loss.detach(), "perturbed_loss": perturbed_loss.detach(),
                "skipped_step": torch.zeros((), device=loss.device)}

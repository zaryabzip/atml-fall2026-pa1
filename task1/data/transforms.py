"""Interventions on common 224x224 tensors in [0, 1], shape (3, H, W) or (N, 3, H, W).
Generate once, save, and feed the SAME images to every model."""
import torch
import torch.nn.functional as F


def to_grayscale(x: torch.Tensor) -> torch.Tensor:
    """Luminance (0.299 R + 0.587 G + 0.114 B) repeated to 3 channels."""
    # Work with a batch dimension throughout, and undo it at the end if we were given a single image.
    is_single_image = x.dim() == 3
    if is_single_image:
        x = x.unsqueeze(0)

    # Weighted sum of the channels gives brightness; repeat it into R, G and B so the shape matches.
    red, green, blue = x[:, 0], x[:, 1], x[:, 2]
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    luminance = luminance.unsqueeze(1)
    gray = luminance.repeat(1, 3, 1, 1)

    if is_single_image:
        gray = gray.squeeze(0)
    return gray


def extra_color(x: torch.Tensor, kind: str = "hue_rotation", **params) -> torch.Tensor:
    """Chosen color change (default: hue rotation). Rotating hue keeps shape, luminance and
    saturation intact and only changes what color each object is painted."""
    if kind != "hue_rotation":  # only one kind is implemented right now
        raise ValueError(f"Unknown extra_color kind {kind!r}; choose 'hue_rotation' (or implement another).")

    # Rotate every pixel's hue by the same amount, around the hue color wheel.
    from torchvision.transforms import functional as TF
    hue_factor = params.get("hue_factor", 0.3)  # in [-0.5, 0.5]; design choice
    x_clamped = x.clamp(0, 1)  # adjust_hue requires values inside [0, 1]
    return TF.adjust_hue(x_clamped, hue_factor)


def translate(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Reflection-pad by |shift| then crop back to 224x224 so the content moves by (dx, dy)."""
    if dx == 0 and dy == 0:
        return x  # no movement requested

    # Work with a batch dimension throughout, and undo it at the end if we were given a single image.
    is_single_image = x.dim() == 3
    if is_single_image:
        x = x.unsqueeze(0)

    # Mirror the border outward by the shift amount, then crop a window back out at an offset that
    # makes the visible content move by exactly (dx, dy).
    height, width = x.shape[2], x.shape[3]
    pad_x, pad_y = abs(dx), abs(dy)
    padded = F.pad(x, (pad_x, pad_x, pad_y, pad_y), mode="reflect")
    top, left = pad_y - dy, pad_x - dx
    cropped = padded[:, :, top:top + height, left:left + width]

    if is_single_image:
        cropped = cropped.squeeze(0)
    return cropped


def patch_shuffle(x: torch.Tensor, grid: int, generator: torch.Generator):
    """Split into grid x grid patches (224 / 4 = 56 px), apply one NON-identity permutation
    per image drawn from `generator` (seed 6304), reassemble. Returns (shuffled, perms) so the
    caller can save the permutations."""
    if grid < 2:
        # With a single patch there is no other order to put it in: torch.randperm(1) can only
        # ever return [0], so the "keep redrawing until non-identity" loop below would spin forever.
        raise ValueError(f"patch_shuffle needs grid >= 2 (grid={grid} has no non-identity permutation).")

    # Work with a batch dimension throughout, and undo it at the end if we were given a single image.
    is_single_image = x.dim() == 3
    if is_single_image:
        x = x.unsqueeze(0)

    n, channels, height, width = x.shape
    if height % grid != 0 or width % grid != 0:
        # patch_h/patch_w below use floor division, so any remainder rows/columns would never be
        # written into `shuffled` (left as uninitialized memory, not zeros) and never be moved.
        raise ValueError(f"grid={grid} must evenly divide the image size ({height}x{width}); "
                          f"{height} % {grid} = {height % grid}, {width} % {grid} = {width % grid}.")
    patch_h, patch_w = height // grid, width // grid
    n_patches = grid * grid

    # Step 1: cut every image into a list of n_patches small tensors, top-left patch first,
    # scanning left-to-right then top-to-bottom (row-major order).
    patches_per_image = []
    for row in range(grid):
        for col in range(grid):
            top, left = row * patch_h, col * patch_w
            patch = x[:, :, top:top + patch_h, left:left + patch_w]  # (N, C, patch_h, patch_w)
            patches_per_image.append(patch)

    # Step 2: draw one random, non-identity patch order per image.
    identity = torch.arange(n_patches)
    perms = torch.empty(n, n_patches, dtype=torch.long)
    for i in range(n):
        perm = torch.randperm(n_patches, generator=generator)
        while torch.equal(perm, identity):  # keep redrawing if we happened to get "no shuffle"
            perm = torch.randperm(n_patches, generator=generator)
        perms[i] = perm

    # Step 3: rebuild each image by placing its patches back down in the shuffled order.
    shuffled = torch.empty_like(x)
    for i in range(n):
        patch_index = 0
        for row in range(grid):
            for col in range(grid):
                top, left = row * patch_h, col * patch_w
                source_patch_index = perms[i, patch_index].item()
                shuffled[i, :, top:top + patch_h, left:left + patch_w] = patches_per_image[source_patch_index][i]
                patch_index += 1

    if is_single_image:
        shuffled, perms = shuffled.squeeze(0), perms.squeeze(0)
    return shuffled, perms

"""PACS image-folder loader.

Expects some folder (possibly nested, as in many Kaggle uploads) that contains one sub-folder per
domain, each with one sub-folder per class:
    <root>/.../{photo,art_painting,cartoon,sketch}/{dog,elephant,giraffe,guitar,horse,house,person}/*.jpg|png
Domain folder names are matched case-insensitively with a few common aliases.
"""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

DOMAINS = ("photo", "art_painting", "cartoon", "sketch")
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")
NUM_CLASSES = len(CLASSES)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

_DOMAIN_ALIASES = {
    "photo": "photo", "photos": "photo",
    "art_painting": "art_painting", "artpainting": "art_painting", "art": "art_painting", "art-painting": "art_painting",
    "cartoon": "cartoon", "cartoons": "cartoon",
    "sketch": "sketch", "sketches": "sketch",
}


def _domain_dirs_in(folder: Path) -> dict:
    found = {}
    for child in folder.iterdir():
        if child.is_dir() and child.name.lower() in _DOMAIN_ALIASES:
            found[_DOMAIN_ALIASES[child.name.lower()]] = child
    return found


def find_pacs_root(root, max_depth: int = 4) -> Path:
    """Return the folder that directly contains all four domain folders."""
    root = Path(root)
    frontier = [root]
    for _ in range(max_depth + 1):
        next_frontier = []
        for folder in frontier:
            if len(_domain_dirs_in(folder)) == len(DOMAINS):
                return folder
            next_frontier.extend(p for p in folder.iterdir() if p.is_dir() and not p.name.startswith("."))
        frontier = next_frontier
    raise FileNotFoundError(f"No folder with all PACS domains {DOMAINS} found under {root}")


def list_domain(pacs_root, domain: str) -> list:
    """Sorted [(relative_path, class_index), ...] for one domain."""
    pacs_root = Path(pacs_root)
    domain_dir = _domain_dirs_in(pacs_root)[domain]
    samples = []
    for class_idx, class_name in enumerate(CLASSES):
        class_dir = domain_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class folder {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((str(path.relative_to(pacs_root)), class_idx))
    return samples


class PACSDataset(Dataset):
    """Returns (image, class_index, domain_index)."""

    def __init__(self, pacs_root, samples, domain: str, transform=None):
        self.pacs_root = Path(pacs_root)
        self.samples = [tuple(s) for s in samples]
        self.domain = domain
        self.domain_index = DOMAINS.index(domain)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        rel_path, label = self.samples[i]
        image = Image.open(self.pacs_root / rel_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label, self.domain_index

"""STL-10 loading. Every intervention is applied to the SAME common 224x224 RGB tensor in [0, 1];
each backbone applies its own normalization afterwards (task1/models/backbones.py).

Data source: Hugging Face mirror `tanganke/stl10` (labeled train 5,000 + test 8,000, same 10 classes).
The official Stanford archive is 2.6 GB (mostly the unused unlabeled split) and downloads very slowly.
Image identifiers saved by make_subset.py are row indices into these parquet files.
"""
import io
import urllib.request

import pyarrow.parquet as pq
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from common.io import PROJECT_ROOT, data_root

STL10_CLASSES = ("airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck")
COMMON_SIZE = 224
SPLIT_FILE = PROJECT_ROOT / "task1" / "data" / "splits" / "stl10_seed6304.json"
_URL = "https://huggingface.co/datasets/tanganke/stl10/resolve/main/data/{split}-00000-of-00001.parquet"


def common_transform():
    return T.Compose([
        T.Resize((COMMON_SIZE, COMMON_SIZE), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.ToTensor(),
    ])


class STL10Parquet(Dataset):
    """Returns (image, label). `labels` is a plain list, like torchvision's STL10."""

    def __init__(self, split: str, transform=None):
        path = data_root() / "stl10" / f"{split}.parquet"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading STL-10 {split} to {path}")
            urllib.request.urlretrieve(_URL.format(split=split), path)
        table = pq.read_table(path).to_pydict()
        self.images = [img["bytes"] for img in table["image"]]
        self.labels = list(table["label"])
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        image = Image.open(io.BytesIO(self.images[i])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[i]


def load_stl10(split: str, transform=None) -> STL10Parquet:
    """split in {'train', 'test'}. Downloads ~230 MB on first use."""
    return STL10Parquet(split, transform)

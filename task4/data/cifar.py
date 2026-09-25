"""CIFAR-10 (known) and the fixed CIFAR-100 test unknowns. CIFAR-100 TRAIN images are never used.

Images come from the Hugging Face copies (uoft-cs/cifar10, uoft-cs/cifar100), because the original
Toronto server is very slow (~80 kB/s on Kaggle). They are cached once as .npz under $PA1_DATA_ROOT/cifar/hf/.
  - CIFAR-10: the HF train set holds the 5 original training batches in the order 4, 3, 2, 5, 1. It is put
    back in the original order, and both CIFAR-10 arrays are checked against the MD5 of the original
    torchvision arrays, so the saved split indices point at exactly the same images.
  - CIFAR-100: class names come from the standard list below, by index (the HF list misspells "crab" as "cra").
"""
import hashlib
import io
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms as T

from common.io import PROJECT_ROOT, data_root, load_json, save_json
from common.metrics import stratified_split
from common.seed import SEED

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
NEAR_UNKNOWN = ("bus", "pickup_truck", "motorcycle", "tractor", "wolf", "fox", "leopard", "camel")
FAR_UNKNOWN = ("bottle", "bowl", "chair", "clock", "keyboard", "mushroom", "sunflower", "wardrobe")
UNKNOWN_GROUPS = {"near": NEAR_UNKNOWN, "far": FAR_UNKNOWN}
VAL_FRACTION = 0.1
SPLIT_FILE = PROJECT_ROOT / "task4" / "data" / "splits" / "cifar10_seed6304.json"


CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
CIFAR100_CLASSES = (
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle", "bowl", "boy",
    "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur", "dolphin", "elephant", "flatfish",
    "forest", "fox", "girl", "hamster", "house", "kangaroo", "keyboard", "lamp", "lawn_mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain", "mouse", "mushroom", "oak_tree", "orange",
    "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal", "shark", "shrew", "skunk",
    "skyscraper", "snail", "snake", "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip", "turtle", "wardrobe", "whale",
    "willow_tree", "wolf", "woman", "worm",
)
_HF = "https://huggingface.co/datasets"
_SOURCES = {  # name -> (parquet URL, label column)
    "cifar10_train": (f"{_HF}/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet", "label"),
    "cifar10_test": (f"{_HF}/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet", "label"),
    "cifar100_test": (f"{_HF}/uoft-cs/cifar100/resolve/main/cifar100/test-00000-of-00001.parquet", "fine_label"),
}
# MD5 of the uint8 image arrays exactly as torchvision's CIFAR10(...).data returns them.
_EXPECTED_MD5 = {"cifar10_train": "17e2cb5a4119d89a4a13e02b91701bff", "cifar10_test": "6d2714505047e13831cd876923568712"}
_HF_TRAIN_BATCH_START = (40000, 20000, 10000, 0, 30000)  # original batch b = HF rows [start, start + 10000)


def _download(url: str, path: Path, tries: int = 5) -> None:
    for attempt in range(1, tries + 1):
        try:
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(path)
            return
        except Exception as err:  # network hiccup: retry a few times before giving up
            if attempt == tries:
                raise RuntimeError(f"could not download {url}") from err
            print(f"download failed ({err}); retry {attempt}/{tries - 1}")
            time.sleep(5 * attempt)


def _arrays(name: str):
    """(images uint8 [N, 32, 32, 3], labels int64 [N]) for one split, downloaded and cached on first use."""
    cache = data_root() / "cifar" / "hf" / f"{name}.npz"
    if not cache.exists():
        import pyarrow.parquet as pq

        cache.parent.mkdir(parents=True, exist_ok=True)
        url, label_col = _SOURCES[name]
        parquet = cache.with_suffix(".parquet")
        if not parquet.exists():
            print(f"downloading {name} from Hugging Face")
            _download(url, parquet)
        table = pq.read_table(parquet).to_pydict()
        images = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in table["img"]])
        labels = np.asarray(table[label_col], dtype=np.int64)
        if name == "cifar10_train":
            order = np.concatenate([np.arange(s, s + 10000) for s in _HF_TRAIN_BATCH_START])
            images, labels = images[order], labels[order]
        if name in _EXPECTED_MD5 and hashlib.md5(images.tobytes()).hexdigest() != _EXPECTED_MD5[name]:
            raise RuntimeError(f"{name}: images differ from the original CIFAR-10 arrays; the split would be wrong")
        np.savez(cache, images=images, labels=labels)
        parquet.unlink()
    data = np.load(cache)
    return data["images"], data["labels"]


class ArrayCIFAR(Dataset):
    """Minimal stand-in for torchvision's CIFAR10/CIFAR100: (PIL image -> transform, int label)."""

    def __init__(self, name: str, classes, transform=None):
        self.data, labels = _arrays(name)
        self.targets = labels.tolist()
        self.classes = list(classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = Image.fromarray(self.data[i])
        return (self.transform(img) if self.transform else img), self.targets[i]


def train_transform(randaugment: bool = False):
    ops = [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip()]
    if randaugment:
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    return T.Compose(ops + [T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def eval_transform():
    return T.Compose([T.ToTensor(), T.Normalize(CIFAR10_MEAN, CIFAR10_STD)])


def make_splits(seed: int = SEED) -> dict:
    targets = ArrayCIFAR("cifar10_train", CIFAR10_CLASSES).targets
    train_idx, val_idx = stratified_split(targets, VAL_FRACTION, seed)
    split = {"seed": seed, "val_fraction": VAL_FRACTION, "train": train_idx, "val": val_idx}
    save_json(split, SPLIT_FILE)
    return split


def load_splits() -> dict:
    return load_json(SPLIT_FILE) if Path(SPLIT_FILE).exists() else make_splits()


def cifar10_train(transform) -> Subset:
    return Subset(ArrayCIFAR("cifar10_train", CIFAR10_CLASSES, transform), load_splits()["train"])


def cifar10_val(transform=None) -> Subset:
    return Subset(ArrayCIFAR("cifar10_train", CIFAR10_CLASSES, transform or eval_transform()), load_splits()["val"])


def cifar10_test(transform=None) -> ArrayCIFAR:
    return ArrayCIFAR("cifar10_test", CIFAR10_CLASSES, transform or eval_transform())


def cifar100_unknowns(group: str, transform=None) -> Subset:
    """Test-split images of the fixed near/far classes. Labels are CIFAR-100 fine-class indices."""
    ds = ArrayCIFAR("cifar100_test", CIFAR100_CLASSES, transform or eval_transform())
    class_ids = {ds.class_to_idx[name] for name in UNKNOWN_GROUPS[group]}
    idx = np.flatnonzero(np.isin(ds.targets, list(class_ids)))
    assert len(idx) == 800, f"expected 800 {group} unknowns, got {len(idx)}"
    return Subset(ds, idx.tolist())

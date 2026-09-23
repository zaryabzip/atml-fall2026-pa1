"""Download PACS (Hugging Face: flwrlabs/pacs, 9,991 images) and unpack it to
$PA1_DATA_ROOT/pacs/<domain>/<class>/<index>.<ext>

    python -m shared.download_pacs
"""
import io
import urllib.request
from collections import Counter

import pyarrow.parquet as pq
from PIL import Image

from common.io import data_root
from shared.pacs import _DOMAIN_ALIASES, CLASSES

URL = "https://huggingface.co/datasets/flwrlabs/pacs/resolve/main/data/train-00000-of-00001.parquet"


def main():
    out = data_root() / "pacs"
    parquet = data_root() / "pacs_raw" / "pacs.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    if not parquet.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, parquet)

    table = pq.read_table(parquet).to_pydict()
    counts = Counter()
    for i, (image, domain, label) in enumerate(zip(table["image"], table["domain"], table["label"])):
        domain = _DOMAIN_ALIASES[domain.lower().replace(" ", "_")]
        data = image["bytes"]
        ext = {"JPEG": "jpg", "PNG": "png"}[Image.open(io.BytesIO(data)).format]
        folder = out / domain / CLASSES[label]
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{i:05d}.{ext}").write_bytes(data)
        counts[domain] += 1
    print(dict(counts), "total", sum(counts.values()), "->", out)


if __name__ == "__main__":
    main()

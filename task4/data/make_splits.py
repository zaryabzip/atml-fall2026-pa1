"""Stratified 90/10 split of CIFAR-10 train (seed 6304).

    python -m task4.data.make_splits
"""
from task4.data.cifar import SPLIT_FILE, make_splits

if __name__ == "__main__":
    split = make_splits()
    print(f"train={len(split['train'])} val={len(split['val'])} -> {SPLIT_FILE}")

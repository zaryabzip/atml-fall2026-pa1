"""Linear heads on frozen features (one per backbone).

train_linear_head(...) returns (head, history):
  head    -- a torch.nn.Linear(feature_dim, 10), already on `device`, holding the weights from
             whichever epoch had the best validation accuracy (not necessarily the last epoch).
             Call head(features_tensor) to get logits; run softmax yourself for probabilities.
  history -- a list of per-epoch dicts: {"epoch": int, "train_loss": float, "val_accuracy": float}.
             Useful for a training curve / for saving into the report JSON.
"""
import copy

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

NUM_CLASSES = 10  # STL-10 has 10 classes


def train_linear_head(train_feats, train_labels, val_feats, val_labels, cfg, device):
    """nn.Linear(feature_dim, 10); AdamW(lr=1e-3, wd=1e-4); <= 50 epochs; early stop after
    5 epochs without better val accuracy; seed 6304. See the module docstring for the return contract."""
    torch.manual_seed(6304)  # reproducible weight initialization and batch shuffling

    # Build the head, and wrap the cached numpy features into DataLoaders.
    feature_dim = train_feats.shape[1]
    head = nn.Linear(feature_dim, NUM_CLASSES).to(device)
    train_dataset = TensorDataset(torch.from_numpy(train_feats).float(), torch.from_numpy(train_labels).long())
    val_dataset = TensorDataset(torch.from_numpy(val_feats).float(), torch.from_numpy(val_labels).long())
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False)

    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    best_val_accuracy = -1.0  # anything real beats this, so epoch 1 always becomes the first "best"
    best_state = None  # snapshot (deep copy) of the best epoch's weights
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, cfg["max_epochs"] + 1):

        # Training pass: one full sweep over the training features.
        head.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = head(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
        train_loss = total_loss / len(train_dataset)

        # Validation pass: check accuracy, no weight updates.
        head.eval()
        correct = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                pred = head(batch_x).argmax(dim=1)
                correct += (pred == batch_y).sum().item()
        val_accuracy = correct / len(val_dataset)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_accuracy": val_accuracy})

        # Early stopping bookkeeping: snapshot the weights whenever val accuracy improves, and stop
        # once it's gone `patience` epochs without a new best.
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(head.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= cfg["patience"]:
            break

    head.load_state_dict(best_state)  # load the best epoch's weights back in before returning
    return head, history

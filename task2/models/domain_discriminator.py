from torch import nn


class DomainDiscriminator(nn.Module):
    """Linear(in_dim, hidden) -> ReLU -> Dropout(dropout) -> Linear(hidden, 2)."""

    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.5):
        super().__init__()
        # A small MLP that guesses "source" or "target" from a feature vector.
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)  # (N, 2) domain logits

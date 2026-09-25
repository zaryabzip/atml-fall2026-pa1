from task4.methods.base import CifarMethod


class RPL(CifarMethod):
    """Optional: Reciprocal Point Learning (Chen et al., 2020). Document the 4 points listed in the PDF."""
    name = "rpl"

    def compute_loss(self, x, y):
        raise NotImplementedError

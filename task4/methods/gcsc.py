from task4.methods.vanilla import Vanilla


class GCSC(Vanilla):
    """Same loss as Vanilla. The only change (RandAugment) lives in configs/gcsc.yaml."""
    name = "gcsc"

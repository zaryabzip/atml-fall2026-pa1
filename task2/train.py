"""Task 2 training.

    python -m task2.train --config task2/configs/dann.yaml
    python -m task2.train --config task2/configs/dann.yaml method.max_grl=0.25 run_name=dann_grl0.25
"""
from common.config import config_arg_parser, load_config
from shared.pacs_train import fit
from task2.methods import METHODS


def main():
    args = config_arg_parser("Task 2: UDA on PACS (Sketch target)").parse_args()
    cfg = load_config(args.config, args.overrides)
    fit(cfg, METHODS[cfg["method"]["name"]], task="task2")


if __name__ == "__main__":
    main()

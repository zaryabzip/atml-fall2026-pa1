"""Task 3 training. Never loads Sketch (the shared loop refuses target access outside Task 2).

    python -m task3.train --config task3/configs/sam.yaml
    python -m task3.train --config task3/configs/sam.yaml method.rho=0.1 run_name=sam_rho0.1
"""
from common.config import config_arg_parser, load_config
from shared.pacs_train import fit
from task3.methods import METHODS


def main():
    args = config_arg_parser("Task 3: DG on PACS (Sketch unseen)").parse_args()
    cfg = load_config(args.config, args.overrides)
    if cfg["method"]["name"] == "erm":
        raise SystemExit("ERM is not retrained: it reuses checkpoints/task2/source_only/best.pt (task3/methods/erm.py).")
    fit(cfg, METHODS[cfg["method"]["name"]], task="task3")


if __name__ == "__main__":
    main()

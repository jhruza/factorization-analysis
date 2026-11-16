"""Train the independent-factor setting; choose an explicit training budget."""

import argparse
from pathlib import Path

from factorization.train import read_config, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.toml"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--until", type=int, help="Stop early at this global update; resume later")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    train(read_config(args.config), args.output, args.device, args.resume, args.until)


if __name__ == "__main__":
    main()

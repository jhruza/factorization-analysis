"""Paired plasticity interventions, preserving parent Adam and scheduler state."""

import argparse
from pathlib import Path

from factorization.checkpoints import load_checkpoint
from factorization.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1000, help="Additional updates in each arm")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted branch suite")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    parent = load_checkpoint(args.checkpoint)
    if args.steps < 1:
        parser.error("--steps must be positive")
    config = dict(parent["config"], name="freeze_branch", steps=parent["step"] + args.steps)
    for freeze in ("none", "attention", "mlp", "embeddings"):
        output = args.output / freeze
        exists = (output / "config.json").exists()
        train(
            config,
            output,
            args.device,
            resume=args.resume and exists,
            parent_checkpoint=None if args.resume and exists else args.checkpoint,
            freeze=freeze,
        )


if __name__ == "__main__":
    main()

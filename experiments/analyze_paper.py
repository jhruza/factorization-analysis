"""Measure paper-style vary-one PCA on saved models; no training occurs."""

import argparse
from pathlib import Path

from factorization.analysis.paper import analyze_paper


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--steps", nargs="+", type=int)
    parser.add_argument("--n-sequences", type=int, default=1024)
    parser.add_argument("--backgrounds", type=int, default=4)
    parser.add_argument("--batch-per-background", type=int, default=128)
    parser.add_argument("--max-components", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--interface", default="L4_mlp")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    analyze_paper(
        args.run,
        args.steps,
        args.n_sequences,
        args.backgrounds,
        args.batch_per_background,
        args.max_components,
        args.batch_size,
        args.device,
        args.interface,
        args.output,
    )


if __name__ == "__main__":
    main()

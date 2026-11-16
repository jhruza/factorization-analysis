"""Measure saved models; no training is performed."""

import argparse
from pathlib import Path

from factorization.analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Fresh analysis directory")
    parser.add_argument("--steps", nargs="+", type=int, help="Saved update numbers; default all")
    parser.add_argument("--interfaces", nargs="+", help="E.g. input L1_attn L1_mlp final")
    parser.add_argument("--n-fit", type=int, default=1024)
    parser.add_argument("--n-test", type=int, default=1024)
    parser.add_argument("--pairs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--scope", choices=["last", "non_bos"], default="last")
    parser.add_argument("--observational-only", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    import torch

    torch.set_num_threads(args.threads)
    analyze(
        args.run,
        args.steps,
        args.n_fit,
        args.n_test,
        args.pairs,
        args.batch_size,
        args.device,
        args.scope,
        args.interfaces,
        not args.observational_only,
        args.output,
    )


if __name__ == "__main__":
    main()

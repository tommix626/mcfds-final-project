from __future__ import annotations

import argparse
import os
from pprint import pprint

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

from spectral_project.experiments import (
    run_adaptive_sigma,
    run_all_experiments,
    run_sigma_eigengap_sweep,
    run_sigma_sweep,
    run_sparsification_tradeoff,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the spectral clustering project experiments.")
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument(
        "--only",
        choices=["sigma-sweep", "adaptive-sigma", "sparsification", "sigma-eigengap"],
        default=None,
        help="Run only one diagnostic experiment instead of regenerating the full project.",
    )
    args = parser.parse_args()
    if args.only == "sigma-sweep":
        outputs = {"sigma_sweep": run_sigma_sweep(random_state=args.random_state)}
    elif args.only == "adaptive-sigma":
        outputs = {"adaptive_sigma": run_adaptive_sigma(random_state=args.random_state)}
    elif args.only == "sparsification":
        outputs = {"sparsification": run_sparsification_tradeoff(random_state=args.random_state)}
    elif args.only == "sigma-eigengap":
        outputs = {"sigma_eigengap": run_sigma_eigengap_sweep(random_state=args.random_state)}
    else:
        outputs = run_all_experiments(random_state=args.random_state)
    for name, df in outputs.items():
        print(f"\n=== {name} ===")
        print(df.head())

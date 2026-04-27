from __future__ import annotations

import argparse
from pprint import pprint

from spectral_project.experiments import run_all_experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the spectral clustering project experiments.")
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    outputs = run_all_experiments(random_state=args.random_state)
    for name, df in outputs.items():
        print(f"\n=== {name} ===")
        print(df.head())

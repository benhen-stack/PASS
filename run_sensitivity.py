#!/usr/bin/env python
"""The empirical check of Appendix C on the Hungarian Chickenpox dataset.

Three panels:

  alpha   with the total training size held fixed, the mixture ratio alpha is
          varied and MMD(p_X, p_mix) is measured. Eq. (mmd_alpha) predicts a
          straight line through the origin.
  gamma   gamma enters only through the distance ||mu_X - mu_gen||, so the
          discrepancy should grow with gamma while alpha stays put.
  J       J enters only through alpha(J) of Eq. (alpha_j), a rectangular
          hyperbola, and the support of p_gen does not depend on J, so the
          curve should saturate.

Usage
    python run_sensitivity.py --county 6 --plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as datasets
from llm import make_llm
from metrics import induced_shift, mmd
from pass_kbs import geometric_score, median_sigma, pass_augment


def fixed_size_sweep(X, X_gen, ratios, total=None, seed=42):
    """Vary alpha at a constant training size, so only the composition changes."""
    rng = np.random.default_rng(seed)
    total = total or len(X)
    rows = []
    for alpha in ratios:
        n_gen = int(round(alpha * total))
        n_orig = total - n_gen
        if n_gen == 0 or n_orig == 0 or n_gen > len(X_gen):
            continue
        mixture = np.vstack([
            X[rng.choice(len(X), n_orig, replace=False)],
            X_gen[rng.choice(len(X_gen), n_gen, replace=n_gen > len(X_gen))],
        ])
        rows.append({"alpha": n_gen / total, "mmd": mmd(X, mixture)})
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--county", type=int, default=6)
    p.add_argument("--backend", default="constant")
    p.add_argument("--q", type=float, default=0.10)
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--gammas", nargs="+", type=float, default=[1, 2, 3, 5, 10, 20, 50])
    p.add_argument("--Js", nargs="+", type=int, default=[1, 2, 3, 5, 8, 12, 20, 30])
    p.add_argument("--base-gamma", type=float, default=3.0)
    p.add_argument("--base-J", type=int, default=1)
    p.add_argument("--sigma-scale", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    split = datasets.chickenpox_split(args.county)
    X, f = split.X_fit, split.f_fit
    sigma = median_sigma(X, scale=args.sigma_scale)
    geo = geometric_score(X, f, sigma)
    llm = make_llm(args.backend)

    def augment(gamma, J):
        return pass_augment(X, f, geo["S_geo"], geo["E"], llm.generate, q=args.q,
                            gamma=gamma, k=args.k, J=J,
                            random_state=args.seed, verbose=False)

    print(f"County {split.meta['county']}, N={len(X)}")

    base = augment(args.base_gamma, args.base_J)
    print(f"|X_high|={len(base['anchors'])}, {len(base['X_aug'])} synthetic samples")

    alpha_table = fixed_size_sweep(
        X, base["X_aug"], np.linspace(0.02, 0.30, 12), seed=args.seed
    )
    if len(alpha_table) > 1:
        slope = np.polyfit(alpha_table["alpha"], alpha_table["mmd"], 1)
        r = np.corrcoef(alpha_table["alpha"], alpha_table["mmd"])[0, 1]
        print(f"\nalpha sweep at fixed size: slope {slope[0]:.4f}, "
              f"intercept {slope[1]:.4f}, r={r:.4f}")
        print(alpha_table.round(4).to_string(index=False))

    gamma_rows = []
    for gamma in args.gammas:
        out = augment(gamma, args.base_J)
        shift = induced_shift(X, out["X_aug"])
        gamma_rows.append({"gamma": gamma, **{k: round(v, 5) for k, v in shift.items()}})
    gamma_table = pd.DataFrame(gamma_rows)
    print("\ngamma sweep (alpha is constant, the mean distance is not):")
    print(gamma_table.to_string(index=False))

    J_rows = []
    for J in args.Js:
        out = augment(args.base_gamma, J)
        shift = induced_shift(X, out["X_aug"])
        distinct = len({(r["anchor"], r["neighbour"]) for r in out["record"]})
        J_rows.append({"J": J, "distinct_samples": distinct,
                       **{k: round(v, 5) for k, v in shift.items()}})
    J_table = pd.DataFrame(J_rows)
    print("\nJ sweep (alpha saturates, the support does not grow past k * |X_high|):")
    print(J_table.to_string(index=False))

    out_dir = Path(args.out) if args.out else Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    alpha_table.to_csv(out_dir / "sensitivity_alpha.csv", index=False)
    gamma_table.to_csv(out_dir / "sensitivity_gamma.csv", index=False)
    J_table.to_csv(out_dir / "sensitivity_J.csv", index=False)
    print(f"\nTables written to {out_dir}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        axes[0].plot(alpha_table["alpha"], alpha_table["mmd"], "o-")
        axes[0].set_xlabel(r"$\alpha$")
        axes[0].set_ylabel("MMD")
        axes[0].set_title("Fixed training size")
        axes[1].plot(gamma_table["gamma"], gamma_table["mmd_mix"], "o-")
        axes[1].set_xlabel(r"$\gamma$")
        axes[1].set_title("Extrapolation parameter")
        axes[2].plot(J_table["J"], J_table["mmd_mix"], "o-")
        axes[2].set_xlabel("$J$")
        axes[2].set_title("Sample count")
        for ax in axes:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / "sensitivity_analysis.png"
        fig.savefig(path, dpi=160)
        print(f"Figure written to {path}")


if __name__ == "__main__":
    main()

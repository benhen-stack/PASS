#!/usr/bin/env python
"""The Hungarian Chickenpox experiment of Section 4, as the manuscript defines it.

Protocol:
  * L = 8, H = 1, first 400 trajectories for training, the rest for test
  * an internal 80/20 validation split inside the training partition
  * standard scaling per county, parameters from the training partition only
  * the downstream forecaster is held fixed across every augmentation method
  * hyperparameters chosen on the validation block, then refitted on the whole
    training partition and reported once on the test block
  * the reported error is the average over the 20 counties

Usage
    python run_chickenpox.py --backend constant               # geometry-only ablation
    python run_chickenpox.py --backend ollama --counties 0 1 2
    python run_chickenpox.py --backend ollama --rag --out results/full.csv
    GEMINI_API_KEY=... python run_chickenpox.py --backend gemini --rag
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as datasets
import rag as rag_module
from baselines import AUGMENTERS
from forecaster import fit_predict
from llm import make_llm
from metrics import evaluate, induced_shift
from profiling import Profiler
from pass_kbs import geometric_score, median_sigma, pass_augment

# The full grid of Section 4.5 of the manuscript.
FULL_GRID = {
    "q": [0.05, 0.10],
    "gamma": [1, 3, 5, 10, 50],
    "k": [1, 2, 3],
    "J": [1, 2, 3, 4, 5],
    "m": [3, 5, 10],
    "sigma_scale": [0.5, 10, 50, 100],
}

# The full grid is 2 x 3 x 5 x 3 x 5 = 450 configurations per county, each one
# a refit of the kernel forecaster, so it is opt-in through --full-grid. The
# default below holds m fixed and thins gamma and J, which keeps a 20-county
# run to minutes while covering the same ranges.
GRID = {
    "q": [0.05, 0.10],
    "gamma": [1, 3, 5],
    "k": [1, 3, 5],
    "J": [1, 3, 5],
    "m": [5],
    "sigma_scale": [0.5, 10, 50, 100],
}

DOMAIN = "infectious disease epidemiology"


def tune_sigma(split, grid=None, mode="vendored", prof=None):
    """Select the bandwidth factor C on the validation block, forecaster only."""
    grid = grid or GRID["sigma_scale"]
    best, best_rmse = grid[0], float("inf")
    for scale in grid:
        sigma = median_sigma(split.X_train, scale=scale)
        try:
            with prof.stage("forecaster_fit", purpose="sigma_tuning"):
                pred, _ = fit_predict(split.X_train, split.f_train, split.X_val,
                                      sigma=sigma, mode=mode)
            prof.count("kernel_fits")
        except Exception as exc:
            print(f"  sigma_scale={scale}: failed ({exc})")
            continue
        rmse = evaluate(split.f_val, pred)["RMSE"]
        print(f"  sigma_scale={scale:>5}: val RMSE {rmse:.4f}")
        if rmse < best_rmse:
            best, best_rmse = scale, rmse
    return best, best_rmse


def pass_grid(split, llm, sigma, insights, grid, n_estimators=500, seed=42,
              verbose=False, mode="vendored", prof=None):
    """Grid search for PASS on the validation block.

    The geometric decomposition and, thanks to the prompt cache, the contextual
    scores are computed once per (q, m) pair and reused across gamma, k and J.
    """
    with prof.stage("geometric_score", partition="train"):
        geo = geometric_score(split.X_train, split.f_train, sigma,
                              n_estimators=n_estimators)
    results = []
    for q, m in itertools.product(grid["q"], grid["m"]):
        for gamma, k, J in itertools.product(grid["gamma"], grid["k"], grid["J"]):
            with prof.stage("pass_augment", partition="train", q=q, m=m):
                out = pass_augment(
                    split.X_train, split.f_train, geo["S_geo"], geo["E"], llm.generate,
                    q=q, gamma=gamma, k=k, J=J, m=m,
                    domain_insights=insights, random_state=seed, verbose=verbose,
                )
            prof.count("pass_configs")
            try:
                with prof.stage("forecaster_fit", purpose="pass_validation"):
                    pred, _ = fit_predict(
                        split.X_train, split.f_train, split.X_val,
                        X_aug=out["X_aug"], f_aug=out["f_aug"], sigma=sigma, mode=mode,
                    )
                prof.count("kernel_fits")
            except Exception as exc:
                print(f"  PASS q={q} m={m} g={gamma} k={k} J={J}: failed ({exc})")
                continue
            scores = evaluate(split.f_val, pred)
            results.append(
                {"q": q, "m": m, "gamma": gamma, "k": k, "J": J,
                 "val_RMSE": scores["RMSE"], "val_MAE": scores["MAE"],
                 "n_anchors": len(out["anchors"]), "alpha": out["alpha"],
                 "n_llm_calls": out["n_llm_calls"]}
            )
    if not results:
        raise RuntimeError("the PASS grid produced no usable configuration")
    table = pd.DataFrame(results).sort_values("val_RMSE").reset_index(drop=True)
    return table, geo


def baseline_grid(split, name, sigma, grid, seed=42, mode="vendored", prof=None):
    """Grid search for a data-level baseline on the validation block."""
    augment = AUGMENTERS[name]
    if name == "none":
        with prof.stage("forecaster_fit", purpose="none_validation"):
            pred, _ = fit_predict(split.X_train, split.f_train, split.X_val,
                                  sigma=sigma, mode=mode)
        prof.count("kernel_fits")
        scores = evaluate(split.f_val, pred)
        return pd.DataFrame([{"J": 0, "k": 0, "val_RMSE": scores["RMSE"],
                              "val_MAE": scores["MAE"]}])
    rows = []
    k_values = grid["k"] if name == "smote" else [0]
    for J, k in itertools.product(grid["J"], k_values):
        with prof.stage(f"augment_{name}", partition="train"):
            X_aug, f_aug = augment(split.X_train, split.f_train, J=J, k=max(k, 1),
                                   random_state=seed)
        try:
            with prof.stage("forecaster_fit", purpose=f"{name}_validation"):
                pred, _ = fit_predict(split.X_train, split.f_train, split.X_val,
                                      X_aug=X_aug, f_aug=f_aug, sigma=sigma, mode=mode)
            prof.count("kernel_fits")
        except Exception as exc:
            print(f"  {name} J={J} k={k}: failed ({exc})")
            continue
        scores = evaluate(split.f_val, pred)
        rows.append({"J": J, "k": k, "val_RMSE": scores["RMSE"],
                     "val_MAE": scores["MAE"]})
    return pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)


def run_county(county, llm, insights, args, prof):
    split = datasets.chickenpox_split(
        county,
        lookback=args.lookback,
        horizon=args.horizon,
        train_size=args.train_size,
        test_size=args.test_size,
    )
    name = split.meta["county"]
    print(f"\n=== {name} ({split.meta['n_trajectories']} trajectories) ===")

    grid = {key: (getattr(args, key) or GRID[key]) for key in
            ("q", "gamma", "k", "J", "m")}

    sigma_scale, _ = tune_sigma(split, args.sigma_scale or GRID["sigma_scale"],
                                mode=args.forecaster, prof=prof)
    sigma = median_sigma(split.X_train, scale=sigma_scale)
    print(f"  selected sigma_scale={sigma_scale}, sigma={sigma:.4f}")

    rows = []

    for baseline in ("none", "smote", "staug"):
        if baseline not in args.methods:
            continue
        table = baseline_grid(split, baseline, sigma, grid, seed=args.seed,
                              mode=args.forecaster, prof=prof)
        best = table.iloc[0].to_dict()
        X_aug, f_aug = (np.empty((0, split.X_fit.shape[1])), np.empty(0))
        if baseline != "none":
            with prof.stage(f"augment_{baseline}", partition="fit"):
                X_aug, f_aug = AUGMENTERS[baseline](
                    split.X_fit, split.f_fit, J=int(best["J"]),
                    k=max(int(best["k"]), 1), random_state=args.seed,
                )
        with prof.stage("forecaster_fit", purpose=f"{baseline}_test"):
            pred, _ = fit_predict(split.X_fit, split.f_fit, split.X_test,
                                  X_aug=X_aug, f_aug=f_aug, sigma=sigma,
                                  mode=args.forecaster)
        prof.count("kernel_fits")
        scores = evaluate(split.f_test, pred, seasonality=args.horizon,
                          scale=split.scaler.sd)
        shift = induced_shift(split.X_fit, X_aug)
        rows.append({"county": name, "method": baseline, **scores,
                     "val_RMSE": best["val_RMSE"], "J": best.get("J"),
                     "k": best.get("k"), "alpha": shift["alpha"],
                     "mmd_mix": shift["mmd_mix"]})
        print(f"  {baseline:<6} test RMSE {scores['RMSE']:.4f}  MAE {scores['MAE']:.4f}"
              f"   |  {scores['RMSE_orig']:7.2f}  {scores['MAE_orig']:7.2f} (orig units)")

    if "pass" in args.methods:
        table, _ = pass_grid(split, llm, sigma, insights, grid,
                             n_estimators=args.n_estimators, seed=args.seed,
                             verbose=args.verbose, mode=args.forecaster, prof=prof)
        best = table.iloc[0].to_dict()
        print(f"  PASS best on validation: {fmt_best(best)}")

        with prof.stage("geometric_score", partition="fit"):
            geo = geometric_score(split.X_fit, split.f_fit, sigma,
                                  n_estimators=args.n_estimators)
        with prof.stage("pass_augment", partition="fit", q=best["q"], m=best["m"]):
            out = pass_augment(
                split.X_fit, split.f_fit, geo["S_geo"], geo["E"], llm.generate,
                q=best["q"], gamma=best["gamma"], k=int(best["k"]), J=int(best["J"]),
                m=int(best["m"]), domain_insights=insights,
                random_state=args.seed, verbose=args.verbose,
            )
        with prof.stage("forecaster_fit", purpose="pass_test"):
            pred, _ = fit_predict(split.X_fit, split.f_fit, split.X_test,
                                  X_aug=out["X_aug"], f_aug=out["f_aug"], sigma=sigma,
                                  mode=args.forecaster)
        prof.count("kernel_fits")
        prof.note_memory()
        scores = evaluate(split.f_test, pred, seasonality=args.horizon,
                          scale=split.scaler.sd)
        shift = induced_shift(split.X_fit, out["X_aug"])
        rows.append({"county": name, "method": "pass", **scores,
                     "val_RMSE": best["val_RMSE"], "q": best["q"],
                     "gamma": best["gamma"], "k": best["k"], "J": best["J"],
                     "m": best["m"], "n_anchors": len(out["anchors"]),
                     "n_llm_calls": out["n_llm_calls"],
                     "llm_failures": out["llm_failures"],
                     "alpha": shift["alpha"], "mmd_mix": shift["mmd_mix"]})
        print(f"  {'pass':<6} test RMSE {scores['RMSE']:.4f}  MAE {scores['MAE']:.4f}"
              f"   |  {scores['RMSE_orig']:7.2f}  {scores['MAE_orig']:7.2f} (orig units)"
              f"  ({out['n_llm_calls']} LLM calls)")

    return rows


def fmt_best(best):
    keys = ("q", "m", "gamma", "k", "J", "val_RMSE")
    return " ".join(f"{key}={best[key]}" for key in keys if key in best)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", default="ollama",
                   choices=["ollama", "gemini", "constant"],
                   help="contextual-score backend. ollama is the default and "
                        "needs no key; gemini is the hosted configuration of "
                        "the paper and reads GEMINI_API_KEY; constant pins the "
                        "severity to 1 and gives the geometry-only ablation")
    p.add_argument("--model", default=None, help="model name for the backend")
    p.add_argument("--rag", action="store_true",
                   help="run Algorithm 3 and inject I_domain into the prompt")
    p.add_argument("--serpapi-key", default=None)
    p.add_argument("--methods", nargs="+", default=["none", "smote", "staug", "pass"])
    p.add_argument("--counties", nargs="+", type=int, default=list(range(20)))
    p.add_argument("--lookback", type=int, default=8)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--train-size", type=int, default=400)
    p.add_argument("--test-size", type=int, default=None,
                   help="defaults to every trajectory after the training partition")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="CSV path for the per-county table")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--forecaster", default="vendored",
                   choices=["vendored", "consistent"],
                   help="vendored calls laplacian_pyramids.LP directly; "
                        "consistent puts the training and test kernels on the "
                        "same coordinates")
    p.add_argument("--full-grid", action="store_true",
                   help="use the complete grid of Section 4.5 instead of the "
                        "thinned default")
    for key in ("q", "gamma", "k", "J", "m", "sigma_scale"):
        p.add_argument(f"--{key}", nargs="+",
                       type=float if key in ("q", "sigma_scale") else int,
                       default=None, help=f"override the {key} grid")
    args = p.parse_args()

    if args.full_grid:
        GRID.update(FULL_GRID)

    kwargs = {"model_name": args.model} if args.model else {}
    llm = make_llm(args.backend, **kwargs)

    prof = Profiler()
    prof.note_memory()

    insights = ""
    if args.rag:
        with prof.stage("rag_domain_insights"):
            insights = rag_module.domain_insights(DOMAIN, llm,
                                                  serpapi_key=args.serpapi_key)

    started = time.time()
    rows = []
    for county in args.counties:
        try:
            with prof.stage("county_total", county=county):
                rows.extend(run_county(county, llm, insights, args, prof))
        except Exception as exc:
            print(f"[run] county {county} failed: {exc}")
        if hasattr(llm, "flush"):
            llm.flush()

    if not rows:
        print("No results.")
        return

    table = pd.DataFrame(rows)
    cols = [c for c in ("RMSE", "MAE", "MASE", "RMSE_orig", "MAE_orig")
            if c in table.columns]
    summary = table.groupby("method")[cols].agg(["mean", "std"]).round(4)

    print("\n" + "=" * 72)
    print(f"Averaged over {table['county'].nunique()} counties, "
          f"{time.time() - started:.0f}s, backend={args.backend}, "
          f"RAG={'on' if insights else 'off'}, "
          f"forecaster={args.forecaster}")
    print("=" * 72)
    print(summary.to_string())

    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "results" / "chickenpox.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    (out.with_suffix(".json")).write_text(json.dumps(vars(args), indent=2, default=str))
    print(f"\nPer-county table: {out}")

    stats = llm.stats() if hasattr(llm, "stats") else {}
    prof.note_memory()
    prof.report(stats)
    prof.save(out.with_name(out.stem + "_profile.json"), stats)


if __name__ == "__main__":
    main()

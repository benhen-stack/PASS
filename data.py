"""Dataset loading, scaling and the sliding-window splits of Section 4.4.

A ``WindowSplit`` holds the trajectory set X with its targets f, split
chronologically, plus the scaler fitted on the training partition only.

Chickenpox, as the manuscript specifies: L = 8, H = 1, first 400 trajectories
for training and the following ones for test, an internal 80/20 validation
split inside the training partition, standard scaling per county, each of the
20 counties forecast independently from its own history.

The series has 522 weekly observations, so with L = 8 and H = 1 there are
522 - 8 - 1 + 1 = 514 trajectories with an observed target, of which 400 are
training. ``test_size=None`` takes every remaining trajectory, which is 114, and
an explicit ``test_size`` larger than that raises rather than letting the
partitions overlap.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Where the benchmark CSVs live. Override with the PASS_DATA_ROOT environment
# variable, or pass path= to any loader. The default is a "datasets" directory
# beside this file.
DATA_ROOT = Path(
    os.environ.get("PASS_DATA_ROOT", Path(__file__).resolve().parent / "datasets")
)

CHICKENPOX_COUNTIES = [
    "BUDAPEST", "BARANYA", "BACS", "BEKES", "BORSOD", "CSONGRAD", "FEJER",
    "GYOR", "HAJDU", "HEVES", "JASZ", "KOMAROM", "NOGRAD", "PEST", "SOMOGY",
    "SZABOLCS", "TOLNA", "VAS", "VESZPREM", "ZALA",
]


@dataclass
class StandardScaler1D:
    """Standard scaling with parameters taken from the training partition only."""

    mu: float = 0.0
    sd: float = 1.0

    def fit(self, values):
        v = np.asarray(values, dtype=float)
        self.mu = float(v.mean())
        self.sd = float(v.std())
        if self.sd < 1e-12:
            self.sd = 1.0
        return self

    def transform(self, values):
        return (np.asarray(values, dtype=float) - self.mu) / self.sd

    def inverse_transform(self, values):
        return np.asarray(values, dtype=float) * self.sd + self.mu


@dataclass
class WindowSplit:
    X_train: np.ndarray
    f_train: np.ndarray
    X_val: np.ndarray
    f_val: np.ndarray
    X_test: np.ndarray
    f_test: np.ndarray
    scaler: StandardScaler1D
    meta: dict = field(default_factory=dict)

    @property
    def X_fit(self):
        """Training partition including the validation block, for the final fit."""
        return np.vstack([self.X_train, self.X_val]) if len(self.X_val) else self.X_train

    @property
    def f_fit(self):
        return np.concatenate([self.f_train, self.f_val]) if len(self.f_val) else self.f_train

    def as_frame(self, X, f):
        """Pack (X, f) into the DataFrame layout the kernel forecaster expects.

        ``LP_Train`` and ``LP`` take a frame whose last column is the target and
        build the kernels on the remaining columns.
        """
        data = np.column_stack([np.asarray(X, dtype=float), np.asarray(f, dtype=float)])
        return pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])


def sliding_windows(series, lookback, horizon):
    """Trajectories x_t = (y(t), ..., y(t+L-1)) with targets f(t) = y(t+L-1+H)."""
    y = np.asarray(series, dtype=float).ravel()
    n = len(y)
    count = n - lookback - horizon + 1
    if count <= 0:
        raise ValueError(
            f"series of length {n} is too short for L={lookback}, H={horizon}"
        )
    X = np.stack([y[t: t + lookback] for t in range(count)])
    f = np.array([y[t + lookback - 1 + horizon] for t in range(count)])
    return X, f


def load_chickenpox(path=None):
    path = Path(path) if path else DATA_ROOT / "HungaryChickenpox" / "hungary_chickenpox.csv"
    df = pd.read_csv(path)
    return df


def chickenpox_split(
    county,
    lookback=8,
    horizon=1,
    train_size=400,
    test_size=None,
    val_fraction=0.2,
    path=None,
):
    """Build the WindowSplit for one county.

    ``county`` is either a column name or an integer index into
    ``CHICKENPOX_COUNTIES``.
    """
    df = load_chickenpox(path)
    name = CHICKENPOX_COUNTIES[county] if isinstance(county, (int, np.integer)) else county
    series = df[name].to_numpy(dtype=float)

    # Scale on the raw observations that fall inside the training partition, so
    # no test observation touches the scaler. The training partition spans the
    # first train_size trajectories, i.e. observations up to train_size+L+H-2.
    n_train_obs = train_size + lookback + horizon - 2
    scaler = StandardScaler1D().fit(series[:n_train_obs])
    X, f = sliding_windows(scaler.transform(series), lookback, horizon)

    available = len(X) - train_size
    if available <= 0:
        raise ValueError(f"{name}: only {len(X)} trajectories, train_size={train_size}")
    if test_size is None:
        test_size = available
    elif test_size > available:
        raise ValueError(
            f"{name}: {test_size} test trajectories requested but only "
            f"{available} exist after {train_size} training trajectories "
            f"(the series yields {len(X)} trajectories at L={lookback}, H={horizon})"
        )

    n_val = int(round(val_fraction * train_size))
    return WindowSplit(
        X_train=X[: train_size - n_val],
        f_train=f[: train_size - n_val],
        X_val=X[train_size - n_val: train_size],
        f_val=f[train_size - n_val: train_size],
        X_test=X[train_size: train_size + test_size],
        f_test=f[train_size: train_size + test_size],
        scaler=scaler,
        meta={
            "dataset": "chickenpox",
            "county": name,
            "lookback": lookback,
            "horizon": horizon,
            "n_trajectories": len(X),
            "train_size": train_size,
            "test_size": test_size,
        },
    )


def load_ili(path=None, target="OT"):
    path = Path(path) if path else DATA_ROOT / "ILI" / "national_illness.csv"
    df = pd.read_csv(path)
    return df


def ili_split(lookback=36, horizon=24, target="OT", ratios=(0.7, 0.1, 0.2), path=None):
    """ILI with the chronological 70/10/20 split of Section 4.4.

    The univariate target column is used, so that the same kernel forecaster
    serves ILI and Chickenpox without a change of representation.
    """
    df = load_ili(path)
    column = target if target in df.columns else df.columns[-1]
    series = df[column].to_numpy(dtype=float)

    n = len(series)
    n_train_obs = int(ratios[0] * n)
    n_val_obs = int((ratios[0] + ratios[1]) * n)

    scaler = StandardScaler1D().fit(series[:n_train_obs])
    X, f = sliding_windows(scaler.transform(series), lookback, horizon)

    # A trajectory belongs to a partition when its target index does.
    target_index = np.arange(len(X)) + lookback - 1 + horizon
    train = target_index < n_train_obs
    val = (target_index >= n_train_obs) & (target_index < n_val_obs)
    test = target_index >= n_val_obs

    return WindowSplit(
        X_train=X[train], f_train=f[train],
        X_val=X[val], f_val=f[val],
        X_test=X[test], f_test=f[test],
        scaler=scaler,
        meta={
            "dataset": "ILI",
            "column": column,
            "lookback": lookback,
            "horizon": horizon,
            "n_trajectories": len(X),
        },
    )

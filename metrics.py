"""Evaluation metrics and the MMD of the distributional-shift appendix.

Metrics are computed on whatever space the arrays are in. Since every series is
standard scaled before the windows are built, that space is standard deviations
by default, which is not the unit anyone reads a forecast in. Pass ``scale`` to
``evaluate`` and it also reports the error in the units of the original series:
RMSE and MAE are positively homogeneous, so multiplying by the standard
deviation used for the scaling inverts it exactly.

    scores = evaluate(split.f_test, predictions, scale=split.scaler.sd)
    scores["RMSE"]        # standard deviations
    scores["RMSE_orig"]   # cases per week, for Hungarian Chickenpox

``scale`` must be the standard deviation of the partition the scaler was fitted
on, which ``WindowSplit.scaler.sd`` holds.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import rbf_kernel


def mase(actual, predicted, seasonality=1):
    """Mean absolute scaled error against the seasonal naive forecast.

    The denominator is the in-sample mean absolute error of the naive forecast
    y_hat(t) = y(t - seasonality), with actual and predicted aligned at the same
    index. MASE is scale free, so it is unaffected by the standard scaling.
    """
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if len(a) <= seasonality:
        return float("nan")
    scale = np.mean(np.abs(a[seasonality:] - a[:-seasonality]))
    if scale < 1e-12:
        return float("nan")
    return float(np.mean(np.abs(a - p)) / scale)


def evaluate(actual, predicted, seasonality=1, scale=None):
    """RMSE, MAE, MSE and MASE, optionally also in the units of the raw series.

    Parameters
    ----------
    actual, predicted : array
        Aligned target and forecast, in the space the model was fitted in.
    seasonality : int
        Lag of the naive forecast used as the MASE denominator.
    scale : float, optional
        The standard deviation the series was divided by. When given, the
        result carries ``RMSE_orig``, ``MAE_orig`` and ``MSE_orig`` as well.
    """
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    mse = float(mean_squared_error(a, p))
    out = {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(a, p)),
        "MSE": mse,
        "MASE": mase(a, p, seasonality=seasonality),
        "n": int(len(a)),
    }
    if scale is not None:
        s = float(scale)
        out["RMSE_orig"] = out["RMSE"] * s
        out["MAE_orig"] = out["MAE"] * s
        out["MSE_orig"] = out["MSE"] * s * s
        out["scale"] = s
    return out


def mmd(X, Y, gamma=None):
    """Unsquared MMD with an RBF kernel."""
    X = np.atleast_2d(np.asarray(X, dtype=float))
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    if len(X) == 0 or len(Y) == 0:
        return float("nan")
    value = (
        rbf_kernel(X, X, gamma).mean()
        + rbf_kernel(Y, Y, gamma).mean()
        - 2 * rbf_kernel(X, Y, gamma).mean()
    )
    return float(np.sqrt(max(value, 0.0)))


def induced_shift(X, X_aug, gamma=None):
    """MMD(p_X, p_mix), with the two channels the shift appendix separates.

    The appendix gives MMD(p_X, p_mix) = alpha * ||mu_X - mu_gen||, so the ratio
    of the measured discrepancy to alpha estimates the mean-embedding distance,
    which depends on the extrapolation parameter but not on the sample count.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    if len(X_aug) == 0:
        return {"alpha": 0.0, "mmd_mix": 0.0, "mu_distance": 0.0, "mmd_gen": 0.0}
    X_aug = np.atleast_2d(np.asarray(X_aug, dtype=float))
    mixture = np.vstack([X, X_aug])
    alpha = len(X_aug) / len(mixture)
    value = mmd(X, mixture, gamma=gamma)
    return {
        "alpha": float(alpha),
        "mmd_mix": value,
        "mu_distance": float(value / alpha) if alpha > 0 else 0.0,
        "mmd_gen": mmd(X, X_aug, gamma=gamma),
    }

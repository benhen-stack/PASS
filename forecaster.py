"""The downstream forecaster of Section 4.4.

Every augmentation method is evaluated on top of the same iterative multi-scale
kernel regression model, extended to a new input window through the kernels of
each pyramid level. It involves no gradient-based training and contributes no
optimisation stochasticity, so any difference between PASS, the unaugmented
reference and the competing augmentations is attributable to the training set
rather than to the model.

Two extension modes are available and are selected with ``--forecaster``.
``vendored`` calls ``laplacian_pyramids.LP`` directly, which builds the pyramid
on the first d-1 look-back coordinates of its training frame and computes the
test distances over all d. ``consistent`` puts both on the same d coordinates.
The vendored module itself is identical in either case. On Hungarian Chickenpox
the two differ by 0.004 to 0.019 RMSE and do not change the ranking of the
augmentation methods.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from laplacian_pyramids import LP, LP_Train
from pass_kbs import median_sigma


def to_frame(X, f=None):
    """Pack trajectories, optionally with a target column, into a frame.

    ``LP_Train`` and ``LP`` address columns positionally and drop the last one
    when they build the kernels, so only the column count matters.
    """
    data = np.asarray(X, dtype=float)
    if f is not None:
        data = np.column_stack([data, np.asarray(f, dtype=float)])
    return pd.DataFrame(data, columns=[str(i) for i in range(data.shape[1])])


def kernel_forecast_vendored(X_train, f_train, X_test, sigma, is_zero=True):
    """Predict through ``laplacian_pyramids.LP``.

    ``dataframe_train`` is the features-only frame and ``dataframe_test`` carries
    a trailing placeholder column that ``LP`` strips.
    """
    train_features = to_frame(X_train)
    test_frame = to_frame(X_test, np.zeros(len(np.atleast_2d(X_test))))
    predictions = LP(
        train_features, np.asarray(f_train, dtype=float), test_frame, sigma, is_zero
    )
    return np.asarray(predictions, dtype=float).ravel()


def kernel_forecast_consistent(X_train, f_train, X_test, sigma, is_zero=True):
    """Same estimator with the training and test kernels on the same coordinates.

    The training frame carries the target as its last column, so
    ``kernel_matrix`` strips exactly the target, the pyramid is built on all d
    look-back coordinates, and the test distances are computed over the same d.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.atleast_2d(np.asarray(X_test, dtype=float))
    f_train = np.asarray(f_train, dtype=float)

    _, _, details, err, sigmas = LP_Train(
        to_frame(X_train, f_train), f_train, sigma, is_zero
    )
    stopping_scale = int(min(range(len(err)), key=err.__getitem__))
    distances = cdist(X_test, X_train, "sqeuclidean")

    predictions = np.zeros(len(X_test), dtype=float)
    for level in range(stopping_scale + 1):
        kernel = np.exp(-distances / (sigmas[level] ** 2))
        row_sums = kernel.sum(axis=1, keepdims=True)
        normalised = np.divide(
            kernel, row_sums, out=np.zeros_like(kernel), where=row_sums > 0
        )
        degenerate = (row_sums <= 0).ravel()
        if degenerate.any():
            normalised[degenerate] = kernel[degenerate]
        source = f_train if level == 0 else np.asarray(details[level], dtype=float)
        predictions += normalised @ source
    return predictions


MODES = {
    "vendored": kernel_forecast_vendored,
    "consistent": kernel_forecast_consistent,
}


def kernel_forecast(X_train, f_train, X_test, sigma, is_zero=True, mode="vendored"):
    try:
        fn = MODES[mode]
    except KeyError:
        raise ValueError(f"mode must be one of {sorted(MODES)}, got {mode!r}") from None
    return fn(X_train, f_train, X_test, sigma, is_zero=is_zero)


def fit_predict(X_train, f_train, X_test, X_aug=None, f_aug=None, sigma=None,
                sigma_scale=10.0, is_zero=True, mode="vendored"):
    """Train on X u X_aug and predict on X_test.

    The bandwidth is computed from the original trajectories only, so that an
    augmentation cannot change the forecaster's own scale and thereby confound
    the comparison.
    """
    if sigma is None:
        sigma = median_sigma(X_train, scale=sigma_scale)
    if X_aug is not None and len(X_aug):
        X_fit = np.vstack([X_train, X_aug])
        f_fit = np.concatenate([f_train, f_aug])
    else:
        X_fit, f_fit = np.asarray(X_train), np.asarray(f_train)
    return kernel_forecast(X_fit, f_fit, X_test, sigma, is_zero=is_zero, mode=mode), sigma

"""Data-level augmentation baselines of Section 4.2.

All three operate on the same sliding-window representation, feed the same
downstream forecaster and are evaluated under identical splits, so that the
contribution of directing the augmentation is separated from the contribution
of merely enlarging the training set.

    none    no augmentation, the reference error level
    smote   interpolation towards a random nearest neighbour, lambda ~ U(0, 1)
    staug   empirical mode recombination followed by a time-domain mixup
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


def augment_none(X, f, **kwargs):
    return np.empty((0, X.shape[1])), np.empty(0)


def augment_smote(X, f, n_samples=None, k=5, J=1, random_state=42, **kwargs):
    """SMOTE for regression: a convex step towards a random nearest neighbour.

    The coefficient is drawn uniformly on [0, 1), so every sample lies strictly
    inside the convex hull of the observed data. The target is interpolated with
    the same coefficient and the same neighbour, which keeps the input-target
    pair consistent.
    """
    X = np.asarray(X, dtype=float)
    f = np.asarray(f, dtype=float)
    rng = np.random.default_rng(random_state)

    if n_samples is None:
        n_samples = J * len(X)
    k_eff = int(min(k, len(X) - 1))
    knn = NearestNeighbors(n_neighbors=k_eff + 1).fit(X)
    _, neighbours = knn.kneighbors(X)

    anchors = rng.integers(0, len(X), size=n_samples)
    X_new = np.empty((n_samples, X.shape[1]))
    f_new = np.empty(n_samples)
    for row, t in enumerate(anchors):
        pool = [j for j in neighbours[t] if j != t][:k_eff]
        s = int(rng.choice(pool))
        lam = float(rng.random())
        X_new[row] = X[t] + lam * (X[s] - X[t])
        f_new[row] = f[t] + lam * (f[s] - f[t])
    return X_new, f_new


def _emd(signal, max_imfs=4):
    """Empirical mode decomposition, PyEMD when available.

    The fallback is an iterative moving-average sifting that returns a
    comparable set of oscillatory components plus a residual trend. It is not
    EMD, and a run that uses it is reported as ``staug-fallback`` so the
    distinction stays in the results.
    """
    try:
        from PyEMD import EMD

        imfs = EMD().emd(np.asarray(signal, dtype=float), max_imf=max_imfs)
        return np.atleast_2d(imfs), True
    except Exception:
        x = np.asarray(signal, dtype=float).copy()
        components, residual = [], x
        for level in range(max_imfs):
            window = 2 ** (level + 1) + 1
            if window >= len(residual):
                break
            kernel = np.ones(window) / window
            trend = np.convolve(residual, kernel, mode="same")
            components.append(residual - trend)
            residual = trend
        components.append(residual)
        return np.atleast_2d(np.asarray(components)), False


def augment_staug(
    X, f, n_samples=None, J=1, alpha_beta=0.5, max_imfs=4, random_state=42, **kwargs
):
    """STAug: spectral recombination of the IMFs, then a Beta mixup in time.

    The decomposition and the mixup are applied to the concatenated
    window-and-target sequence, so the input-target relation is perturbed
    consistently rather than only on the look-back side.
    """
    X = np.asarray(X, dtype=float)
    f = np.asarray(f, dtype=float)
    rng = np.random.default_rng(random_state)
    if n_samples is None:
        n_samples = J * len(X)

    sequences = np.column_stack([X, f])
    used_pyemd = True

    def reweighted(row):
        nonlocal used_pyemd
        imfs, ok = _emd(sequences[row], max_imfs=max_imfs)
        used_pyemd = used_pyemd and ok
        weights = rng.uniform(0.0, 2.0, size=len(imfs))
        return (weights[:, None] * imfs).sum(axis=0)

    out = np.empty((n_samples, sequences.shape[1]))
    for i in range(n_samples):
        a, b = rng.integers(0, len(sequences), size=2)
        lam = float(rng.beta(alpha_beta, alpha_beta))
        out[i] = lam * reweighted(int(a)) + (1 - lam) * reweighted(int(b))

    if not used_pyemd:
        print("[staug] PyEMD unavailable, using the sifting fallback")
    return out[:, :-1], out[:, -1]


AUGMENTERS = {
    "none": augment_none,
    "smote": augment_smote,
    "staug": augment_staug,
}

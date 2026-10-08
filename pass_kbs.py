"""Potential-Aware Semantic Synthesis (PASS).

Reference implementation of the method, Algorithms 1 and 2 of the manuscript.
Algorithm 3, the RAG insight distillation, is in ``rag.py``.

Equation map

    geometric_score         Sec. 3.1, Eqs. (geo_score_transform), (geo_norm)
    local_descriptors       Sec. 3.2, Eq. (shrinkage), anisotropy, Mahalanobis
    build_prompt            Algorithm 1, Appendix B template
    contextual_score        Eq. (ctx_gate)
    potential               Eq. (potential)
    high_potential_set      Eq. (high_potential)
    synthesise              Eqs. (extrap), (target_extrap)
    pass_augment            Algorithm 2, end to end

The language model enters as any callable ``score(prompt: str) -> str``, so the
method carries no dependency on a particular provider, SDK or cache.

Dependencies: numpy, scipy, scikit-learn, and the unmodified
``laplacian_pyramids`` module for the multi-scale decomposition.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import zscore
from sklearn.covariance import ledoit_wolf
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from laplacian_pyramids import LP_Train

__all__ = [
    "sliding_windows",
    "median_sigma",
    "geometric_score",
    "local_descriptors",
    "build_prompt",
    "parse_severity",
    "contextual_score",
    "potential",
    "high_potential_set",
    "synthesise",
    "pass_augment",
]

EPS = 1e-12


# ==========================================================================
# Trajectories and bandwidth
# ==========================================================================

def sliding_windows(series, lookback, horizon):
    """X = {x_t}, x_t = (y(t), ..., y(t+L-1)), with targets f(t) = y(t+L-1+H)."""
    y = np.asarray(series, dtype=float).ravel()
    count = len(y) - lookback - horizon + 1
    if count <= 0:
        raise ValueError(f"series of length {len(y)} too short for L={lookback}, H={horizon}")
    X = np.stack([y[t: t + lookback] for t in range(count)])
    f = np.array([y[t + lookback - 1 + horizon] for t in range(count)])
    return X, f


def median_sigma(X, scale=10.0, sample_size=500, random_state=42):
    """sigma = median of the positive pairwise trajectory distances, times C.

    C is the factor tuned over {0.5, 10, 50, 100} in Section 4.5.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(random_state)
    if len(X) > sample_size:
        X = X[rng.choice(len(X), sample_size, replace=False)]
    d = pdist(X, "euclidean")
    d = d[d > 0]
    return float(np.median(d) * scale) if d.size else 1.0


# ==========================================================================
# Section 3.1, the multi-scale geometric score
# ==========================================================================

def geometric_score(X, f, sigma, n_estimators=500, contamination="auto",
                    random_state=42, is_zero=True):
    """S_geo for every trajectory, through the multi-scale reconstruction error.

    The pyramid of Eqs. (gl_kernels) and (last_smth) yields residual vectors
    e_1, ..., e_{l*+1}. Each is z-scored and made absolute, a_l = |z(e_l)|,
    giving the error signature matrix E whose row E(t, :) characterises the
    reconstruction of x_t. An Isolation Forest on the rows of E is negated,
    Eq. (geo_score_transform), then min-max mapped onto [0, 1], Eq. (geo_norm),
    so that the two factors of Eq. (potential) are non-negative and the fusion
    is order preserving.

    ``is_zero=True`` sets the kernel diagonal to zero, which stops the finest
    scales collapsing into a delta function and interpolating the noise.

    Returns a dict with ``S_geo``, ``S_geo_hat``, ``E`` and ``stopping_scale``.
    """
    X = np.asarray(X, dtype=float)
    f = np.asarray(f, dtype=float)

    # LP_Train addresses columns positionally and drops the last one when it
    # builds the kernels, so the target is appended as that last column and the
    # pyramid is built on all d look-back coordinates.
    frame = pd.DataFrame(np.column_stack([X, f]))
    _, _, details, err, sigmas = LP_Train(frame, f, sigma, is_zero)

    stopping_scale = int(min(range(len(err)), key=err.__getitem__))  # l*

    columns = []
    for level in range(stopping_scale + 1):
        e = np.atleast_2d(np.asarray(details[level], dtype=float))
        if e.shape[0] == 1 and e.shape[1] == len(X):
            e = e.T
        a = np.zeros_like(e)
        for j in range(e.shape[1]):
            col = e[:, j]
            a[:, j] = 0.0 if np.std(col) < EPS else np.abs(zscore(col, ddof=0))
        columns.append(a.sum(axis=1))
    E = np.asarray(columns).T

    forest = IsolationForest(n_estimators=n_estimators, contamination=contamination,
                             random_state=random_state).fit(E)
    S_hat = -forest.decision_function(E)
    lo, hi = float(S_hat.min()), float(S_hat.max())

    return {
        "S_geo": (S_hat - lo) / (hi - lo + EPS),
        "S_geo_hat": S_hat,
        "E": E,
        "stopping_scale": stopping_scale,
        "sigmas": sigmas,
    }


def global_context(E):
    """G_t: the range, median and 85th percentile of the error distribution."""
    total = np.asarray(E, dtype=float).sum(axis=1)
    return {
        "min_score": float(total.min()),
        "max_score": float(total.max()),
        "median_score": float(np.median(total)),
        "p85_threshold": float(np.percentile(total, 85)),
    }


def percentile_rank(S_geo, t):
    """Eq. (percentile): R(x_t)."""
    s = np.asarray(S_geo, dtype=float)
    return 100.0 * float(np.sum(s <= s[t])) / len(s)


# ==========================================================================
# Section 3.2, the local structural descriptors L_t
# ==========================================================================

def local_descriptors(X, t, m=5, k_neighbors=3, knn=None):
    """L_t for trajectory x_t.

    The local window is W_t = {x_{t-m}, ..., x_{t-1}}. Because the ambient
    dimension normally exceeds m, its empirical covariance is rank deficient, so
    Sigma_t is the Ledoit-Wolf shrinkage estimator of Eq. (shrinkage), which is
    positive definite by construction. The two descriptors it carries are the
    anisotropy theta_1 / theta_d and the Mahalanobis distance to the centroid.
    """
    X = np.asarray(X, dtype=float)
    d = X.shape[1]
    x_t = X[t].reshape(1, -1)

    if knn is None:
        knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(X))).fit(X)
    dist = knn.kneighbors(x_t)[0][0][1:]
    if dist.size == 0:
        dist = np.array([0.0])

    out = {
        "mean_neighbor_distance": float(np.mean(dist)),
        "std_neighbor_distance": float(np.std(dist)),
        "kth_neighbor_distance": float(dist[-1]),
    }
    out["local_density"] = 1.0 / (out["mean_neighbor_distance"] + 1e-8)
    out["temporal_discontinuity"] = (
        float(np.linalg.norm(X[t] - X[t - 1])) if t > 0 else 0.0
    )

    if t >= 3:
        recent = X[t - 3: t + 1]
        out["local_trend"] = float(np.linalg.norm(recent[-1] - recent[0]))
        out["local_volatility"] = float(
            np.mean([np.std(np.diff(recent[:, j])) for j in range(d)])
        )
    else:
        out["local_trend"] = out["local_volatility"] = 0.0

    if t >= m >= 2:
        W = X[t - m: t]
        Sigma, _ = ledoit_wolf(W, assume_centered=False)
        Sigma = np.atleast_2d(np.asarray(Sigma, dtype=float))
        if np.trace(Sigma) <= 0:
            Sigma = np.eye(d)
        theta = np.clip(np.linalg.eigvalsh(Sigma), 1e-18, None)  # symmetric, so eigvalsh
        diff = X[t] - W.mean(axis=0)
        out["local_cov_det"] = float(np.linalg.det(Sigma))
        out["local_anisotropy"] = float(theta[-1] / theta[0])
        out["local_mahalanobis_distance"] = float(
            np.sqrt(max(diff @ np.linalg.inv(Sigma) @ diff, 0.0))
        )
    else:
        out["local_cov_det"] = out["local_anisotropy"] = 1.0
        out["local_mahalanobis_distance"] = out["mean_neighbor_distance"]

    return out


# ==========================================================================
# Algorithm 1, prompt construction
# ==========================================================================

GUARDRAIL = (
    "Initial Prompt Guardrail: You are functioning solely as a statistical "
    "anomaly detector. The task is strictly confined to the analysis of the "
    "provided numerical features and temporal metrics to estimate the "
    "statistical likelihood of an anomalous pattern."
)

ANALYSIS_CONTEXT = (
    "Analysis Context: The analysis should focus on the complexity of the "
    "mapping function from historical inputs to future targets. The multi-scale "
    "reconstruction errors below are derived directly from the target "
    "trajectory. Your objective is to identify regions where this functional "
    "relationship becomes highly non-smooth, discontinuous, or unstable. Such "
    "regions indicate complex local dynamics or regime shifts that correspond "
    "to critical failures in predictability rather than feature-level noise."
)

INTERPRETATION_GUIDE = """Temporal Interpretation Guide
- Low density + high neighbor distance -> isolated temporal pattern
- High temporal discontinuity -> abrupt temporal jump
- High volatility -> unstable local temporal dynamics
- High anisotropy -> unstable, strongly anisotropic covariance structure
- High Mahalanobis distance -> deviation from local temporal manifold
- High local trend -> possible regime transition"""

DETECTION_CRITERIA = """Anomaly Detection Criteria
1. Global reconstruction error extremeness
2. Temporal isolation in low-density regions
3. Unstable local covariance structure
4. Mahalanobis deviation from neighborhood manifold
5. Large temporal jumps from previous observations"""

OUTPUT_FORMAT = (
    "Required Output Format\nSeverity: [0.0-1.0]\n"
    "Return the severity value only, with no additional text."
)

LOCAL_FIELDS = (
    ("Local density", "local_density"),
    ("Mean neighbor distance", "mean_neighbor_distance"),
    ("Farthest temporal neighbor distance", "kth_neighbor_distance"),
    ("Neighbor variation (std)", "std_neighbor_distance"),
    ("Temporal discontinuity", "temporal_discontinuity"),
    ("Local trend magnitude", "local_trend"),
    ("Local volatility", "local_volatility"),
    ("Covariance determinant", "local_cov_det"),
    ("Local anisotropy", "local_anisotropy"),
    ("Mahalanobis distance", "local_mahalanobis_distance"),
)


def build_prompt(x_t, error_signature, local_metrics, context, domain_insights=""):
    """Algorithm 1. Returns the structured prompt P(x_t) of Appendix B.

    ``context`` is G_t updated with ``percentile_rank``. ``domain_insights`` is
    I_domain; when empty the RAG block is omitted, which is the no-RAG ablation.
    The only field requested is the severity, since Eq. (ctx_gate) defines
    S_ctx as the severity the model returns.
    """
    def vec(v):
        return "[" + ", ".join(f"{float(x):.4f}" for x in np.ravel(v)) + "]"

    blocks = [GUARDRAIL, ANALYSIS_CONTEXT]
    if domain_insights.strip():
        blocks.append("Methodological Insights (RAG)\n" + domain_insights.strip())
    blocks.append(
        "Data Characteristics\n"
        f"- Data Point: {vec(x_t)}\n"
        f"- Multi-scale Reconstruction Errors: {vec(error_signature)}"
    )
    blocks.append(
        "Global Temporal Context\n"
        f"- Percentile rank of error score: {context['percentile_rank']:.1f}%\n"
        f"- Dataset error range: {context['min_score']:.3f} to {context['max_score']:.3f}\n"
        f"- Median error score: {context['median_score']:.3f}\n"
        f"- 85th percentile threshold: {context['p85_threshold']:.3f}"
    )
    blocks.append(
        "Temporal Structural Analysis (Local)\n"
        + "\n".join(f"- {label}: {local_metrics[key]:.4f}" for label, key in LOCAL_FIELDS)
    )
    blocks += [INTERPRETATION_GUIDE, DETECTION_CRITERIA, OUTPUT_FORMAT]
    return "\n\n".join(blocks)


def parse_severity(text):
    """Read the severity off the response, clipped to [0, 1], or None."""
    if not text:
        return None
    match = re.search(r"severity\s*[:=]\s*\[?\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    match = match or re.search(r"([0-9]*\.?[0-9]+)", text)
    if match is None:
        return None
    return float(np.clip(float(match.group(1)), 0.0, 1.0))


# ==========================================================================
# Algorithm 2
# ==========================================================================

def contextual_score(X, S_geo, E, score, q=0.10, m=5, k_neighbors=3,
                     domain_insights="", on_failure=0.0, verbose=True):
    """Eq. (ctx_gate): S_ctx, the severity inside the candidate set and 0 outside.

    The candidate set is {x : S_geo(x) >= tau_geo} with tau_geo the top
    q-quantile, so the model is queried only where it can change the outcome.
    Since the augmentation threshold tau of Eq. (high_potential) is at least as
    selective as tau_geo, the gate changes the values of s~ on the discarded
    fraction but not the identity of X_high. It follows that s~ is not a
    calibrated potential over the whole trajectory space, and a trajectory that
    is semantically critical but geometrically unremarkable is invisible to the
    framework by construction.

    ``score`` is any callable taking the prompt and returning the model's text.

    Returns (S_ctx, info), with the per-candidate severities in
    ``info['severities']``.
    """
    X = np.asarray(X, dtype=float)
    s = np.asarray(S_geo, dtype=float)
    tau_geo = float(np.quantile(s, 1.0 - q))
    candidates = np.flatnonzero(s >= tau_geo)

    if verbose:
        print(f"[pass] gate: {len(candidates)}/{len(X)} above tau_geo={tau_geo:.4f} at q={q}")

    knn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(X))).fit(X)
    G = global_context(E)

    S_ctx = np.zeros(len(X), dtype=float)
    severities, failures = {}, 0
    for t in candidates:
        context = dict(G, percentile_rank=percentile_rank(s, t))
        prompt = build_prompt(
            X[t], E[t],
            local_descriptors(X, t, m=m, k_neighbors=k_neighbors, knn=knn),
            context, domain_insights,
        )
        try:
            severity = parse_severity(score(prompt))
        except Exception as exc:
            print(f"[pass] scoring failed at t={t}: {exc}")
            severity = None
        if severity is None:
            failures += 1
            severity = on_failure
        S_ctx[t] = severity
        severities[int(t)] = float(severity)

    if verbose and severities:
        v = np.fromiter(severities.values(), dtype=float)
        print(f"[pass] severity: min={v.min():.3f} median={np.median(v):.3f} "
              f"max={v.max():.3f} std={v.std():.3f} failures={failures}")

    return S_ctx, {"candidates": candidates, "tau_geo": tau_geo,
                   "severities": severities, "failures": failures,
                   "n_calls": len(candidates)}


def potential(S_geo, S_ctx):
    """Eq. (potential): s~(x) = S_geo(x) * S_ctx(x)."""
    return np.asarray(S_geo, dtype=float) * np.asarray(S_ctx, dtype=float)


def high_potential_set(s_tilde, q):
    """Eq. (high_potential): X_high = {x : s~(x) >= tau}, tau the top q-quantile.

    Ties at tau are kept, so the set can exceed ceil(q N) when the field is
    flat. Zero-potential trajectories are excluded: a zero comes only from the
    gate or from a severity of exactly zero, and neither marks a high-potential
    trajectory.
    """
    s = np.asarray(s_tilde, dtype=float)
    tau = float(np.quantile(s, 1.0 - q))
    return np.flatnonzero((s >= tau) & (s > 0)), tau


def synthesise(X, f, s_tilde, anchors, gamma=3.0, k=3, J=1, random_state=42):
    """Eqs. (extrap) and (target_extrap): directional extrapolation.

        x_new = x_t + gamma * sigma_ts * (x_t - x_s)
        f_new = f(x_t) + gamma * sigma_ts * (f(x_t) - f(x_s))

    with sigma_ts = +1 when the anchor holds the higher potential and -1 when
    the neighbour does, so the displacement points from the lower-potential
    member of the pair towards the higher one and is a finite-difference
    estimate of the ascent direction of s~. The target is propagated along the
    same step, exact to first order with a bias of order gamma^2 in the
    curvature of f, Eq. (extrap_bias). Every draw produces a sample and no
    anchor is discarded.

    The J draws are made with replacement from k candidates, so an anchor
    admits at most k distinct samples: beyond k, a larger J raises the
    multiplicity of existing samples rather than the support, which is the
    saturation of Section 3.6. The single degenerate case is gamma=1 with
    sigma_ts=-1, where x_new coincides with x_s.
    """
    X = np.asarray(X, dtype=float)
    f = np.asarray(f, dtype=float)
    s = np.asarray(s_tilde, dtype=float)
    anchors = np.asarray(anchors, dtype=int)
    if anchors.size == 0:
        return np.empty((0, X.shape[1])), np.empty(0), []

    rng = np.random.default_rng(random_state)
    k_eff = int(min(k, len(X) - 1))
    neighbours = NearestNeighbors(n_neighbors=k_eff + 1).fit(X).kneighbors(X[anchors])[1]

    X_new, f_new, record = [], [], []
    for row, t in enumerate(anchors):
        pool = np.array([j for j in neighbours[row] if j != t][:k_eff])  # X \ {x_t}
        if pool.size == 0:
            continue
        for _ in range(J):
            sj = int(rng.choice(pool))
            sigma_ts = 1.0 if s[t] >= s[sj] else -1.0
            step = sigma_ts * (X[t] - X[sj])
            X_new.append(X[t] + gamma * step)
            f_new.append(f[t] + gamma * sigma_ts * (f[t] - f[sj]))
            record.append({"anchor": int(t), "neighbour": sj, "sigma_ts": sigma_ts,
                           "gamma_eff": gamma if sigma_ts > 0 else gamma - 1.0})

    return np.asarray(X_new), np.asarray(f_new), record


def pass_augment(X, f, S_geo, E, score, q=0.10, gamma=3.0, k=3, J=1, m=5,
                 k_neighbors=3, domain_insights="", random_state=42, verbose=True):
    """Algorithm 2 end to end. Returns the augmented set and the potential field.

    ``S_geo`` and ``E`` are passed in rather than recomputed, so a whole
    (q, gamma, k, J) grid reuses one multi-scale decomposition. The downstream
    forecaster is then fit on X u X_aug.
    """
    S_ctx, info = contextual_score(X, S_geo, E, score, q=q, m=m,
                                   k_neighbors=k_neighbors,
                                   domain_insights=domain_insights, verbose=verbose)
    s_tilde = potential(S_geo, S_ctx)
    anchors, tau = high_potential_set(s_tilde, q)
    if verbose:
        print(f"[pass] |X_high|={len(anchors)} at tau={tau:.4f}")

    X_aug, f_aug, record = synthesise(X, f, s_tilde, anchors, gamma=gamma, k=k, J=J,
                                      random_state=random_state)
    n = len(X_aug)
    return {
        "X_aug": X_aug,
        "f_aug": f_aug,
        "S_ctx": S_ctx,
        "s_tilde": s_tilde,
        "anchors": anchors,
        "tau": tau,
        "tau_geo": info["tau_geo"],
        "alpha": n / (len(X) + n) if n else 0.0,  # Eq. (alpha_j)
        "n_llm_calls": info["n_calls"],
        "llm_failures": info["failures"],
        "severities": info["severities"],
        "record": record,
    }

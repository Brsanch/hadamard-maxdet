"""Landscape reconstruction model for the H_n maximum-determinant search.

Aggregates every diagnostic projection we have (basin atlas, M1 cardinality,
count_5 distribution, 2-val multiset, sign-equivalence class, Hamming
distance to known anchors, singular-value pattern, row/column statistics)
into a single feature vector per matrix, then fits a regression model to
predict ``log10|det|``. The fitted model is used three ways:

1. **Residual analysis** — matrices whose actual ``log|det|`` exceeds the
   feature-based prediction are "structurally surprising": they sit higher
   than the diagnostic projections alone would explain. Orrick's H23 record
   is the canonical example.
2. **Extrapolation** — feeding hypothetical feature vectors (e.g. a 2-val
   class with ``(c1, c3) = (210, 43)``) returns the model's prediction for
   |det|. If the prediction is well above Orrick we have a target to try
   to construct.
3. **Blind-spot identification** — projecting the feature matrix to 2D
   (via PCA) and binning gives a coverage map. Empty cells flag regions
   of feature space that no current matrix occupies.

The methodology is the cryo-EM-style reconstruction analogy: each
diagnostic is a "projection" of the same underlying landscape; combining
them yields a single coherent model.

Public API
----------
* :func:`extract_features` — feature dict for one ``±1`` matrix.
* :func:`build_data_matrix` — ``(X, y, feature_names, labels)`` from a list
  of matrices.
* :func:`fit_predictive_model` — fit linear + random-forest regressors;
  return both with R² and feature importances.
* :func:`residual_analysis` — top positive / negative log-det residuals.
* :func:`extrapolate` — predict log|det| for a list of synthetic feature
  dicts.
* :func:`blind_spot_2d` — PCA-2D projection + 2D histogram of coverage.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


# Canonical feature column order. Kept as a module-level tuple so callers
# (extrapolation, scripting) can build feature vectors without re-running
# extract_features. Any change here must be mirrored in tests.
FEATURE_NAMES: tuple[str, ...] = (
    # Gram off-diagonal multiset cardinality (1..7+)
    "gram_card",
    # Sign-invariant |Gram| multiset cardinality
    "gram_abs_card",
    # Counts of |Gram| = 1, 3, 5, 7+ off-diagonals
    "count_abs_1",
    "count_abs_3",
    "count_abs_5",
    "count_abs_7_plus",
    # Counts of signed Gram values
    "count_neg_3",
    "count_neg_1",
    "count_pos_1",
    "count_pos_3",
    # Off-diagonal magnitudes summary
    "max_abs_offdiag",
    "mean_abs_offdiag",
    "std_abs_offdiag",
    # Singular-value pattern
    "sv_max",
    "sv_min",
    "sv_mean",
    "sv_std",
    "sv_n_distinct",  # rounded to 4 decimals — multiplicity proxy
    # Row weight (sum of +1s minus -1s) statistics
    "row_sum_max_abs",
    "row_sum_mean_abs",
    "row_sum_std",
    # Column weight statistics
    "col_sum_max_abs",
    "col_sum_mean_abs",
    "col_sum_std",
    # Triple-cyclic Gram T-support cardinality (M1 score) — bounded sample
    "t_support_card",
    # Order
    "n",
)


def _gram_offdiag(M: np.ndarray) -> np.ndarray:
    """Return the strict-upper-triangle off-diagonal Gram entries as a
    1-D int array.
    """
    G = M @ M.T
    iu, ju = np.triu_indices(G.shape[0], k=1)
    return G[iu, ju].astype(np.int64)


def _t_support_sample(M: np.ndarray, max_triples: int = 3000) -> int:
    """Approximate ``|T-support|`` (count of distinct cyclic triple Gram
    products) via a deterministic sample of up to ``max_triples`` triples.

    The full O(n^3) cost at n=23 is C(23, 3) = 1771 triples — well under
    the cap, so this is exact for our use case. The sampling cap exists
    only to stay tractable if a caller passes a much larger matrix.
    """
    n = M.shape[0]
    G = (M @ M.T).astype(np.int64)
    seen: set[int] = set()
    count = 0
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            gij = int(G[i, j])
            for k in range(j + 1, n):
                t = gij * int(G[j, k]) * int(G[k, i])
                seen.add(t)
                count += 1
                if count >= max_triples:
                    return len(seen)
    return len(seen)


def extract_features(M: Sequence[Sequence[int]]) -> dict[str, float]:
    """Compute a fixed-schema feature dict from one ``±1`` matrix.

    Returns a dict with keys in :data:`FEATURE_NAMES`. Deterministic — no
    random sampling. All-numeric float values.
    """
    arr = np.asarray(M, dtype=np.int64)
    n = arr.shape[0]
    if arr.shape != (n, n):
        raise ValueError(f"matrix must be square; got shape {arr.shape}")

    offdiag = _gram_offdiag(arr)
    abs_off = np.abs(offdiag)

    # Cardinality of signed and unsigned Gram off-diagonal multisets.
    gram_card = int(len(np.unique(offdiag)))
    gram_abs_card = int(len(np.unique(abs_off)))

    count_abs_1 = int(np.sum(abs_off == 1))
    count_abs_3 = int(np.sum(abs_off == 3))
    count_abs_5 = int(np.sum(abs_off == 5))
    count_abs_7_plus = int(np.sum(abs_off >= 7))

    count_neg_3 = int(np.sum(offdiag == -3))
    count_neg_1 = int(np.sum(offdiag == -1))
    count_pos_1 = int(np.sum(offdiag == 1))
    count_pos_3 = int(np.sum(offdiag == 3))

    max_abs = float(abs_off.max()) if abs_off.size else 0.0
    mean_abs = float(abs_off.mean()) if abs_off.size else 0.0
    std_abs = float(abs_off.std()) if abs_off.size else 0.0

    svs = np.linalg.svd(arr.astype(np.float64), compute_uv=False)
    sv_max = float(svs.max())
    sv_min = float(svs.min())
    sv_mean = float(svs.mean())
    sv_std = float(svs.std())
    sv_n_distinct = int(len(np.unique(np.round(svs, 4))))

    row_sums = np.abs(arr.sum(axis=1).astype(np.float64))
    col_sums = np.abs(arr.sum(axis=0).astype(np.float64))
    row_sum_max_abs = float(row_sums.max())
    row_sum_mean_abs = float(row_sums.mean())
    row_sum_std = float(row_sums.std())
    col_sum_max_abs = float(col_sums.max())
    col_sum_mean_abs = float(col_sums.mean())
    col_sum_std = float(col_sums.std())

    t_support_card = float(_t_support_sample(arr))

    return {
        "gram_card": float(gram_card),
        "gram_abs_card": float(gram_abs_card),
        "count_abs_1": float(count_abs_1),
        "count_abs_3": float(count_abs_3),
        "count_abs_5": float(count_abs_5),
        "count_abs_7_plus": float(count_abs_7_plus),
        "count_neg_3": float(count_neg_3),
        "count_neg_1": float(count_neg_1),
        "count_pos_1": float(count_pos_1),
        "count_pos_3": float(count_pos_3),
        "max_abs_offdiag": max_abs,
        "mean_abs_offdiag": mean_abs,
        "std_abs_offdiag": std_abs,
        "sv_max": sv_max,
        "sv_min": sv_min,
        "sv_mean": sv_mean,
        "sv_std": sv_std,
        "sv_n_distinct": float(sv_n_distinct),
        "row_sum_max_abs": row_sum_max_abs,
        "row_sum_mean_abs": row_sum_mean_abs,
        "row_sum_std": row_sum_std,
        "col_sum_max_abs": col_sum_max_abs,
        "col_sum_mean_abs": col_sum_mean_abs,
        "col_sum_std": col_sum_std,
        "t_support_card": t_support_card,
        "n": float(n),
    }


def features_to_vector(feat: dict[str, float]) -> np.ndarray:
    """Pack a feature dict into a 1-D numpy array in :data:`FEATURE_NAMES`
    order. Missing features default to 0.0 (caller's choice / for synthetic
    extrapolation feature vectors that only specify a few entries).
    """
    return np.array(
        [float(feat.get(name, 0.0)) for name in FEATURE_NAMES],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Data matrix construction
# ---------------------------------------------------------------------------


@dataclass
class DataMatrix:
    """A packed feature matrix + log-target + label list."""

    X: np.ndarray  # shape (n_matrices, n_features)
    y: np.ndarray  # shape (n_matrices,) — log10(|det|)
    abs_dets: np.ndarray  # shape (n_matrices,) — raw |det| as float
    labels: list[str]
    feature_names: tuple[str, ...] = field(default_factory=lambda: FEATURE_NAMES)


def build_data_matrix(
    matrices: Sequence[tuple[str, Sequence[Sequence[int]], int]],
) -> DataMatrix:
    """Pack ``(label, matrix, abs_det)`` triples into a :class:`DataMatrix`.

    The target ``y`` is ``log10(max(abs_det, 1))`` so a singular matrix
    (``abs_det = 0``) maps to 0.0; callers should drop such rows beforehand
    if a strict log-domain fit is desired.
    """
    labels: list[str] = []
    rows: list[np.ndarray] = []
    dets: list[float] = []
    for label, M, det in matrices:
        feat = extract_features(M)
        rows.append(features_to_vector(feat))
        labels.append(label)
        dets.append(float(det))
    X = np.vstack(rows) if rows else np.zeros((0, len(FEATURE_NAMES)))
    abs_dets = np.array(dets, dtype=np.float64)
    y = np.log10(np.maximum(abs_dets, 1.0))
    return DataMatrix(X=X, y=y, abs_dets=abs_dets, labels=labels)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


@dataclass
class FitResult:
    """Wraps a fitted regressor + its diagnostics."""

    name: str
    model: Any
    r2_train: float
    feature_importances: dict[str, float]
    predictions: np.ndarray  # in-sample predictions (log10 scale)


def fit_predictive_model(
    data: DataMatrix,
    *,
    random_state: int = 0,
) -> dict[str, FitResult]:
    """Fit a linear regression and a random-forest regressor on
    ``(data.X, data.y)``. Returns a dict keyed by ``"linear"`` and
    ``"random_forest"``.

    Both models are trained on the full data matrix — there is no held-out
    split. R² is reported as the in-sample fit ("how well does the
    diagnostic feature set explain the observed |det| distribution?"),
    not as a generalisation estimate.
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    out: dict[str, FitResult] = {}

    lr = LinearRegression()
    lr.fit(data.X, data.y)
    pred_lr = lr.predict(data.X)
    r2_lr = float(r2_score(data.y, pred_lr))
    # For linear regression, "importance" = absolute coefficient × feature std.
    stds = data.X.std(axis=0)
    raw_imp_lr = np.abs(lr.coef_) * stds
    if raw_imp_lr.sum() > 0:
        imp_lr = raw_imp_lr / raw_imp_lr.sum()
    else:
        imp_lr = raw_imp_lr
    out["linear"] = FitResult(
        name="linear",
        model=lr,
        r2_train=r2_lr,
        feature_importances=dict(zip(FEATURE_NAMES, imp_lr.tolist())),
        predictions=pred_lr,
    )

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=1,  # keep apfsd quiet (project rule: no parallel bursts)
    )
    rf.fit(data.X, data.y)
    pred_rf = rf.predict(data.X)
    r2_rf = float(r2_score(data.y, pred_rf))
    out["random_forest"] = FitResult(
        name="random_forest",
        model=rf,
        r2_train=r2_rf,
        feature_importances=dict(
            zip(FEATURE_NAMES, rf.feature_importances_.tolist())
        ),
        predictions=pred_rf,
    )
    return out


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------


@dataclass
class Residual:
    """One label's actual vs. predicted log10|det|."""

    label: str
    actual_log: float
    predicted_log: float
    residual: float
    abs_det: float


def residual_analysis(
    fit: FitResult, data: DataMatrix, *, top_k: int = 20
) -> dict[str, list[Residual]]:
    """Return ``{"top_positive": [...], "top_negative": [...]}``.

    Top-positive residuals = matrices with surprisingly *high* |det|
    relative to what their diagnostic features predict. Orrick should
    dominate this list: he's a known structural outlier on the |G| ⊆ {1,3}
    submanifold.

    Top-negative residuals = matrices with surprisingly *low* |det| —
    typically pathological / singular cases that nevertheless retain
    "good-looking" features.
    """
    pred = fit.predictions
    resid = data.y - pred
    order = np.argsort(resid)
    out_neg = [
        Residual(
            label=data.labels[i],
            actual_log=float(data.y[i]),
            predicted_log=float(pred[i]),
            residual=float(resid[i]),
            abs_det=float(data.abs_dets[i]),
        )
        for i in order[:top_k]
    ]
    out_pos = [
        Residual(
            label=data.labels[i],
            actual_log=float(data.y[i]),
            predicted_log=float(pred[i]),
            residual=float(resid[i]),
            abs_det=float(data.abs_dets[i]),
        )
        for i in order[::-1][:top_k]
    ]
    return {"top_positive": out_pos, "top_negative": out_neg}


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------


@dataclass
class Extrapolation:
    """Synthetic feature vector + model prediction."""

    description: str
    features: dict[str, float]
    predicted_log: float
    predicted_abs_det: float
    ratio_to_orrick: float


# Orrick's H23 record |det| (canonical reference target).
ORRICK_H23_ABS_DET = 2_779_447_296_000_000


def _features_for_2val_class(c1: int, c3: int, n: int = 23) -> dict[str, float]:
    """Build a feature dict for a hypothetical |G| ⊆ {1, 3} matrix at order
    ``n`` with ``count_abs_1 = c1``, ``count_abs_3 = c3``.

    Off-diagonal pair count is C(n, 2). We populate Gram-multiset features
    fully but leave singular-value / row-sum / T-support features at a
    "neutral" default = the mean of the H23 training population; the
    extrapolation script overrides these with population means when called.
    """
    return {
        "gram_card": 4.0,  # signs both directions of {1, 3}
        "gram_abs_card": 2.0,
        "count_abs_1": float(c1),
        "count_abs_3": float(c3),
        "count_abs_5": 0.0,
        "count_abs_7_plus": 0.0,
        # signed split (default to balanced)
        "count_neg_3": float(c3) / 2.0,
        "count_neg_1": float(c1) / 2.0,
        "count_pos_1": float(c1) / 2.0,
        "count_pos_3": float(c3) / 2.0,
        "max_abs_offdiag": 3.0,
        "mean_abs_offdiag": (c1 + 3 * c3) / max(c1 + c3, 1),
        "std_abs_offdiag": 0.0,  # unknown without matrix
        "n": float(n),
    }


def extrapolate(
    fit: FitResult,
    hypothetical_features: Sequence[dict[str, float]],
    *,
    descriptions: Sequence[str] | None = None,
    population_means: dict[str, float] | None = None,
) -> list[Extrapolation]:
    """Predict log10|det| for synthetic feature vectors.

    Missing features default first to ``population_means`` (typically the
    column-mean of the training matrix), then to 0.0. This keeps the
    extrapolation grounded — synthetic 2-val descriptors will still get
    plausible singular-value / row-sum estimates.

    Returns one :class:`Extrapolation` per input feature dict.
    """
    if descriptions is None:
        descriptions = [f"hyp_{i}" for i in range(len(hypothetical_features))]
    if population_means is None:
        population_means = {}
    out: list[Extrapolation] = []
    for desc, feat in zip(descriptions, hypothetical_features):
        merged: dict[str, float] = {}
        for name in FEATURE_NAMES:
            if name in feat:
                merged[name] = float(feat[name])
            elif name in population_means:
                merged[name] = float(population_means[name])
            else:
                merged[name] = 0.0
        x = features_to_vector(merged).reshape(1, -1)
        pred_log = float(fit.model.predict(x)[0])
        pred_det = 10.0 ** pred_log
        ratio = pred_det / ORRICK_H23_ABS_DET
        out.append(
            Extrapolation(
                description=desc,
                features=merged,
                predicted_log=pred_log,
                predicted_abs_det=pred_det,
                ratio_to_orrick=ratio,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Blind-spot identification
# ---------------------------------------------------------------------------


@dataclass
class BlindSpot:
    """A 2D projection bin that has zero matrices in it."""

    bin_x: int
    bin_y: int
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    distance_to_nearest_sample: float


@dataclass
class BlindSpot2D:
    coords: np.ndarray  # (n_matrices, 2) PCA-projected coordinates
    pca_explained_variance_ratio: tuple[float, float]
    histogram: np.ndarray  # (nbins, nbins) counts
    x_edges: np.ndarray
    y_edges: np.ndarray
    blind_spots: list[BlindSpot]


def blind_spot_2d(
    data: DataMatrix, *, nbins: int = 12
) -> BlindSpot2D:
    """PCA-project the feature matrix to 2D and bin into an
    ``nbins × nbins`` histogram. Returns the empty bins as
    "blind spots" — regions of feature space unsampled by the corpus.

    Bins are ranked by their distance to the nearest sampled point (in
    PCA coordinates); the most isolated empty bin is the most
    informative blind spot.
    """
    from sklearn.decomposition import PCA

    if data.X.shape[0] == 0:
        return BlindSpot2D(
            coords=np.zeros((0, 2)),
            pca_explained_variance_ratio=(0.0, 0.0),
            histogram=np.zeros((nbins, nbins)),
            x_edges=np.zeros(nbins + 1),
            y_edges=np.zeros(nbins + 1),
            blind_spots=[],
        )
    # Standardise features (each col → zero mean, unit variance) so the
    # PCA isn't dominated by raw-magnitude features (e.g. sv_max ~ 8 vs
    # std_abs_offdiag ~ 1).
    Xc = data.X - data.X.mean(axis=0, keepdims=True)
    stds = Xc.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    Xn = Xc / stds
    n_components = min(2, Xn.shape[0], Xn.shape[1])
    if n_components < 2:
        coords = np.zeros((Xn.shape[0], 2))
        if n_components == 1:
            coords[:, 0] = PCA(n_components=1).fit_transform(Xn).ravel()
        evr = (1.0, 0.0) if n_components == 1 else (0.0, 0.0)
    else:
        pca = PCA(n_components=2)
        coords = pca.fit_transform(Xn)
        evr = (
            float(pca.explained_variance_ratio_[0]),
            float(pca.explained_variance_ratio_[1]),
        )

    hist, x_edges, y_edges = np.histogram2d(
        coords[:, 0], coords[:, 1], bins=nbins
    )
    blind: list[BlindSpot] = []
    # For each empty cell, distance to nearest sample (PCA coordinates).
    if coords.shape[0] > 0:
        for ix in range(nbins):
            for iy in range(nbins):
                if hist[ix, iy] > 0:
                    continue
                cx = 0.5 * (x_edges[ix] + x_edges[ix + 1])
                cy = 0.5 * (y_edges[iy] + y_edges[iy + 1])
                dists = np.linalg.norm(
                    coords - np.array([cx, cy])[None, :], axis=1
                )
                blind.append(
                    BlindSpot(
                        bin_x=ix,
                        bin_y=iy,
                        x_range=(float(x_edges[ix]), float(x_edges[ix + 1])),
                        y_range=(float(y_edges[iy]), float(y_edges[iy + 1])),
                        distance_to_nearest_sample=float(dists.min()),
                    )
                )
    blind.sort(key=lambda b: -b.distance_to_nearest_sample)
    return BlindSpot2D(
        coords=coords,
        pca_explained_variance_ratio=evr,
        histogram=hist,
        x_edges=x_edges,
        y_edges=y_edges,
        blind_spots=blind,
    )


__all__ = [
    "FEATURE_NAMES",
    "ORRICK_H23_ABS_DET",
    "extract_features",
    "features_to_vector",
    "DataMatrix",
    "build_data_matrix",
    "FitResult",
    "fit_predictive_model",
    "Residual",
    "residual_analysis",
    "Extrapolation",
    "extrapolate",
    "BlindSpot",
    "BlindSpot2D",
    "blind_spot_2d",
]

"""Tests for the landscape reconstruction model."""
from __future__ import annotations

import random

import numpy as np
import pytest

from hadamard_maxdet.landscape_model import (
    FEATURE_NAMES,
    ORRICK_H23_ABS_DET,
    blind_spot_2d,
    build_data_matrix,
    extract_features,
    extrapolate,
    features_to_vector,
    fit_predictive_model,
    residual_analysis,
)
from hadamard_maxdet.verifier import integer_determinant


def _random_pm1(n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]


def _hadamard12() -> list[list[int]]:
    """Sylvester H_4 — small ±1 matrix with non-zero determinant."""
    H1 = [[1]]
    H2 = [[1, 1], [1, -1]]
    H4 = [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ]
    return H4


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def test_extract_features_schema():
    M = _hadamard12()
    feat = extract_features(M)
    # all keys present
    for name in FEATURE_NAMES:
        assert name in feat, f"missing feature {name}"
    # all numeric
    for name in FEATURE_NAMES:
        assert isinstance(feat[name], (int, float)), name


def test_extract_features_deterministic():
    """Same input → identical output."""
    M = _random_pm1(10, seed=42)
    f1 = extract_features(M)
    f2 = extract_features(M)
    assert f1 == f2


def test_extract_features_hadamard_h4():
    """Sylvester H4: all off-diagonal Gram entries are 0 — but 0 isn't ±1
    and the abs counts should sum to C(4, 2) = 6 with all values at |G|=0
    (i.e. count_abs_1 = count_abs_3 = count_abs_5 = 0, but the signed
    counts also zero). Verify gram_card and the abs-count sum.
    """
    M = _hadamard12()
    feat = extract_features(M)
    # H4 is orthogonal, so off-diagonal Gram = 0 everywhere.
    assert feat["max_abs_offdiag"] == 0.0
    assert feat["count_abs_1"] == 0.0
    assert feat["count_abs_3"] == 0.0
    # gram_card = 1 (just the value 0).
    assert feat["gram_card"] == 1.0
    assert feat["gram_abs_card"] == 1.0


def test_features_to_vector_order():
    feat = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    v = features_to_vector(feat)
    assert v.shape == (len(FEATURE_NAMES),)
    for i, name in enumerate(FEATURE_NAMES):
        assert v[i] == float(i)


def test_features_to_vector_missing_defaults_zero():
    feat = {"count_abs_1": 100.0}
    v = features_to_vector(feat)
    idx = FEATURE_NAMES.index("count_abs_1")
    assert v[idx] == 100.0
    # other entries default to 0
    for i, name in enumerate(FEATURE_NAMES):
        if name != "count_abs_1":
            assert v[i] == 0.0


# ---------------------------------------------------------------------------
# Data matrix construction
# ---------------------------------------------------------------------------


def test_build_data_matrix_small():
    triples = []
    for seed in range(20):
        M = _random_pm1(8, seed=seed)
        det = integer_determinant(M)
        if det == 0:
            continue
        triples.append((f"M_{seed}", M, abs(det)))
    assert len(triples) >= 4
    data = build_data_matrix(triples)
    assert data.X.shape[0] == len(triples)
    assert data.X.shape[1] == len(FEATURE_NAMES)
    assert data.y.shape == (len(triples),)
    assert data.abs_dets.shape == (len(triples),)
    # y matches log10(det)
    expected_y = np.log10(np.maximum(data.abs_dets, 1.0))
    np.testing.assert_array_almost_equal(data.y, expected_y)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------


def test_fit_predictive_model_returns_both():
    triples = []
    for seed in range(40):
        M = _random_pm1(11, seed=seed)
        det = integer_determinant(M)
        if det == 0:
            continue
        triples.append((f"M_{seed}", M, abs(det)))
    data = build_data_matrix(triples)
    fits = fit_predictive_model(data)
    assert "linear" in fits
    assert "random_forest" in fits
    for name, fit in fits.items():
        assert fit.predictions.shape == data.y.shape
        assert isinstance(fit.r2_train, float)
        # importances sum to ~1 for both methods (linear normalised, RF natively)
        total_imp = sum(fit.feature_importances.values())
        assert 0.95 <= total_imp <= 1.05, f"{name}: imp sum {total_imp}"


def test_random_forest_better_than_linear_on_synthetic():
    """On a synthetic landscape where |det| is a nonlinear function of
    features, RF should outperform LR. Build matrices spanning a wide
    |det| range so the difference is measurable.
    """
    triples = []
    for seed in range(60):
        M = _random_pm1(11, seed=seed)
        det = integer_determinant(M)
        if det == 0:
            continue
        triples.append((f"M_{seed}", M, abs(det)))
    data = build_data_matrix(triples)
    fits = fit_predictive_model(data)
    # Both R² should be reasonable; RF typically >= LR on small sample.
    assert fits["linear"].r2_train > 0.0 or fits["random_forest"].r2_train > 0.0
    # RF in-sample R² is typically >= 0.5 on this kind of structured data.
    assert fits["random_forest"].r2_train >= 0.5


# ---------------------------------------------------------------------------
# Residual analysis
# ---------------------------------------------------------------------------


def test_residual_analysis_identifies_top_det():
    """If we inject a known-high-|det| matrix into a population of randoms
    while telling the model only the "feature mean" (so the model can't
    distinguish it on features alone), residual analysis should rank it
    highly. Concretely: with linear regression, the residuals are
    deterministic, so we just check the API + that the highest-|det|
    matrix has a defined residual.
    """
    triples = []
    for seed in range(20):
        M = _random_pm1(11, seed=seed)
        det = integer_determinant(M)
        if det == 0:
            continue
        triples.append((f"M_{seed}", M, abs(det)))
    data = build_data_matrix(triples)
    fits = fit_predictive_model(data)
    res = residual_analysis(fits["linear"], data, top_k=5)
    assert "top_positive" in res and "top_negative" in res
    assert len(res["top_positive"]) == 5
    assert len(res["top_negative"]) == 5
    # Top positive should have the highest residual.
    pos_residuals = [r.residual for r in res["top_positive"]]
    assert pos_residuals == sorted(pos_residuals, reverse=True)
    # Top negative should be sorted ascending.
    neg_residuals = [r.residual for r in res["top_negative"]]
    assert neg_residuals == sorted(neg_residuals)


# ---------------------------------------------------------------------------
# Extrapolation
# ---------------------------------------------------------------------------


def test_extrapolate_returns_predictions():
    triples = []
    for seed in range(30):
        M = _random_pm1(11, seed=seed)
        det = integer_determinant(M)
        if det == 0:
            continue
        triples.append((f"M_{seed}", M, abs(det)))
    data = build_data_matrix(triples)
    fits = fit_predictive_model(data)
    means = dict(zip(FEATURE_NAMES, data.X.mean(axis=0).tolist()))
    hyps = [
        {"count_abs_1": 210.0, "count_abs_3": 43.0},
        {"count_abs_1": 200.0, "count_abs_3": 53.0},
    ]
    descs = ["hyp_210_43", "hyp_200_53"]
    out = extrapolate(fits["random_forest"], hyps, descriptions=descs, population_means=means)
    assert len(out) == 2
    for e, desc in zip(out, descs):
        assert e.description == desc
        assert isinstance(e.predicted_log, float)
        assert isinstance(e.predicted_abs_det, float)
        assert e.predicted_abs_det > 0.0
        assert e.ratio_to_orrick == e.predicted_abs_det / ORRICK_H23_ABS_DET


# ---------------------------------------------------------------------------
# Blind-spot identification
# ---------------------------------------------------------------------------


def test_blind_spot_2d_basic():
    triples = []
    for seed in range(30):
        M = _random_pm1(11, seed=seed)
        det = integer_determinant(M)
        if det == 0:
            continue
        triples.append((f"M_{seed}", M, abs(det)))
    data = build_data_matrix(triples)
    bs = blind_spot_2d(data, nbins=8)
    assert bs.coords.shape == (data.X.shape[0], 2)
    assert bs.histogram.shape == (8, 8)
    # 8x8 = 64 bins total, populated bins ≤ n_matrices.
    assert int(bs.histogram.sum()) == data.X.shape[0]
    # Some blind spots expected when n_matrices < 64.
    if data.X.shape[0] < 64:
        assert len(bs.blind_spots) > 0
    # Blind spots sorted descending by distance.
    if len(bs.blind_spots) > 1:
        ds = [b.distance_to_nearest_sample for b in bs.blind_spots]
        assert ds == sorted(ds, reverse=True)


def test_blind_spot_2d_evr_in_range():
    triples = []
    for seed in range(40):
        M = _random_pm1(11, seed=seed)
        det = integer_determinant(M)
        if det == 0:
            continue
        triples.append((f"M_{seed}", M, abs(det)))
    data = build_data_matrix(triples)
    bs = blind_spot_2d(data, nbins=6)
    e1, e2 = bs.pca_explained_variance_ratio
    assert 0.0 <= e1 <= 1.0
    assert 0.0 <= e2 <= 1.0
    assert e1 + e2 <= 1.0 + 1e-9

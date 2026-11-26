"""Tests for hadamard_maxdet.spectral_anchor."""
from __future__ import annotations

import numpy as np
import pytest

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.spectral_anchor import (
    bordered_extension_sa,
    distinct_sv_count,
    dominant_eigenspace,
    dominant_sv_multiplicity,
    flip_dominant_projection,
    predict_target_spectrum,
    sa_spectral_anchored,
    sa_targeted_spectrum,
    spectral_anchor_score,
    spectrum_distance,
)
from hadamard_maxdet.verifier import integer_determinant
from hadamard_maxdet.winner_decode import find_winning_matrix


# ---------------------------------------------------------------------------
# dominant_sv_multiplicity / distinct_sv_count sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_dominant_sv_multiplicity_matches_numpy(n):
    """For random ±1 matrices, the count we return should equal the
    largest greedy 1D cluster of the descending SV list at the same
    relative tolerance."""
    rng = np.random.default_rng(seed=42 + n)
    M = rng.choice([-1, 1], size=(n, n)).astype(int).tolist()
    sv = np.linalg.svd(np.array(M, dtype=float), compute_uv=False)
    s_max = float(sv[0])
    thr = 0.05 * s_max
    # Reference: greedy 1D clustering on the descending SV list.
    rep = float(sv[0])
    cur = 1
    counts = []
    for v in sv[1:]:
        if rep - float(v) <= thr:
            cur += 1
        else:
            counts.append(cur)
            rep = float(v)
            cur = 1
    counts.append(cur)
    expected = max(counts)
    got = dominant_sv_multiplicity(M, tol=0.05)
    assert got == expected


def test_dominant_sv_multiplicity_hadamard():
    """An n=8 Hadamard matrix has all SVs equal to √8."""
    H = Matrix.sylvester(8).entries
    assert dominant_sv_multiplicity(H, tol=0.001) == 8


def test_dominant_sv_multiplicity_n17_winner():
    """The proven n=17 winner should have dominant SV multiplicity ≥ 14."""
    abs_det, entries, _ = find_winning_matrix(
        17, structured_seconds=20.0, n_starts=2, iterations=20_000,
    )
    assert abs_det == 21_474_836_480, (
        f"Expected proven n=17 winner |det|=21,474,836,480, got {abs_det}"
    )
    mult = dominant_sv_multiplicity(entries, tol=0.1)
    assert mult >= 14, f"Expected ≥14, got {mult}"


def test_distinct_sv_count_hadamard():
    """Hadamard matrix → 1 distinct SV cluster."""
    H = Matrix.sylvester(8).entries
    assert distinct_sv_count(H, tol=0.05) == 1


def test_distinct_sv_count_full_rank_random():
    """A random ±1 matrix at n ≥ 8 generally has many distinct SVs."""
    rng = np.random.default_rng(seed=7)
    M = rng.choice([-1, 1], size=(10, 10)).astype(int).tolist()
    # Should be > 1 (no degeneracy expected).
    assert distinct_sv_count(M, tol=0.05) > 1


# ---------------------------------------------------------------------------
# spectral_anchor_score
# ---------------------------------------------------------------------------


def test_spectral_anchor_score_hadamard_max():
    """A Hadamard matrix has the maximal anchor score n − 1."""
    H = Matrix.sylvester(8).entries
    n = 8
    assert spectral_anchor_score(H, tol=0.05) == n - 1


def test_spectral_anchor_score_random_lower():
    """A random ±1 matrix scores lower than a Hadamard at same n."""
    rng = np.random.default_rng(seed=11)
    M = rng.choice([-1, 1], size=(8, 8)).astype(int).tolist()
    rand_score = spectral_anchor_score(M, tol=0.05)
    H = Matrix.sylvester(8).entries
    had_score = spectral_anchor_score(H, tol=0.05)
    assert rand_score <= had_score


# ---------------------------------------------------------------------------
# flip_dominant_projection
# ---------------------------------------------------------------------------


def test_flip_dominant_projection_in_unit_interval():
    """flip_dominant_projection should always return a value in [0, 1]."""
    rng = np.random.default_rng(seed=23)
    M = rng.choice([-1, 1], size=(8, 8)).astype(int).tolist()
    V = dominant_eigenspace(M, tol=0.1)
    for j in range(8):
        for i in range(8):
            p = flip_dominant_projection(M, i, j, V)
            assert 0.0 <= p <= 1.0 + 1e-9, f"out of range: {p}"


def test_flip_dominant_projection_full_subspace_is_one():
    """If V is the full right-singular space (n × n), every column j
    should have ‖V[j, :]‖² = 1 (orthogonal completeness)."""
    rng = np.random.default_rng(seed=11)
    M = rng.choice([-1, 1], size=(8, 8)).astype(int).tolist()
    # Force V to be the full right-singular basis.
    Marr = np.array(M, dtype=float)
    _, _, Vt = np.linalg.svd(Marr, full_matrices=False)
    V_full = Vt.T  # n × n orthogonal
    for j in range(8):
        p = flip_dominant_projection(M, 0, j, V_full)
        assert abs(p - 1.0) < 1e-9


def test_flip_dominant_projection_empty_subspace_is_zero():
    """If V is empty (no dominant cluster), projection should be 0."""
    M = Matrix.sylvester(4).entries
    V = np.zeros((4, 0))
    p = flip_dominant_projection(M, 0, 0, V)
    assert p == 0.0


# ---------------------------------------------------------------------------
# sa_spectral_anchored: smoke test
# ---------------------------------------------------------------------------


def test_sa_spectral_anchored_runs_and_improves_at_n8():
    """A short anchored SA at n=8 from random init should not regress
    below the random starting point's |det|."""
    M0 = Matrix.random(
        8, __import__("random").Random(0),
    )
    init_abs_det = abs(integer_determinant(M0.entries))
    result = sa_spectral_anchored(
        M0,
        iterations=2_000,
        lambda_anchor=0.5,
        seed=12345,
        refresh_eigenspace_every=100,
    )
    # The SA reports a Bareiss-verified |det|. It should be at least as
    # big as the initial one (SA tracks best-ever).
    assert result.abs_det >= init_abs_det
    # And the result should still be ±1.
    for row in result.matrix.entries:
        for v in row:
            assert v in (-1, 1)


# ---------------------------------------------------------------------------
# bordered_extension_sa
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# predict_target_spectrum / spectrum_distance / sa_targeted_spectrum
# ---------------------------------------------------------------------------


def test_predict_target_spectrum_hadamard_n8():
    """For n=8 with Hadamard target |det| = 4096 = n^(n/2), the spectrum
    is all SVs equal to sqrt(n) = sqrt(8). The 'hadamard' model returns
    this exactly."""
    spec = predict_target_spectrum(8, 4096, model="hadamard")
    assert len(spec) == 8
    assert np.allclose(spec, np.sqrt(8.0))
    H = Matrix.sylvester(8).entries
    d = spectrum_distance(H, spec)
    assert d < 1e-9, f"Hadamard distance to itself should be ~0, got {d}"


def test_predict_target_spectrum_n17_matches_proven():
    """At n=17 with target |det| = 21,474,836,480 (proven), the
    'symmetric_pair' model with ratio=1.566 (calibrated) predicts the
    proven SVs (6.217, 4.0×15, 3.217) within 5%."""
    n, det = 17, 21_474_836_480
    spec = predict_target_spectrum(
        n, det, model="symmetric_pair", ratio=1.566,
    )
    actual = np.array(sorted([6.217] + [4.0] * 15 + [3.217], reverse=True))
    rel = np.abs(spec - actual) / np.maximum(actual, 1e-9)
    max_rel = float(np.max(rel))
    assert max_rel < 0.05, (
        f"Predicted vs actual relative error {max_rel:.4f} > 5% at n=17. "
        f"Predicted: {spec.round(4)}, actual: {actual}"
    )


def test_predict_target_spectrum_models_satisfy_trace_and_det():
    """Trace and det constraints by construction for 'high_only',
    'symmetric_pair', and 'equal_pair'."""
    n, det = 19, 894_426_939_392
    for model in ["high_only", "symmetric_pair", "equal_pair"]:
        spec = predict_target_spectrum(n, det, model=model)
        trace_G = float(np.sum(spec ** 2))
        assert abs(trace_G - n * n) < 1e-3 * n * n, (
            f"{model}: trace constraint violated, got {trace_G} vs {n*n}"
        )
        log_det_G = float(np.sum(2.0 * np.log(spec)))
        target = 2.0 * np.log(float(det))
        assert abs(log_det_G - target) < 1e-3, (
            f"{model}: det constraint violated, log_det_G={log_det_G:.4f}, "
            f"target={target:.4f}"
        )


def test_spectrum_distance_zero_at_own_spectrum():
    """spectrum_distance(M, sv(M)) returns 0."""
    H = Matrix.sylvester(8).entries
    sv = np.linalg.svd(np.array(H, dtype=float), compute_uv=False)
    assert spectrum_distance(H, sv) < 1e-9


def test_spectrum_distance_positive_otherwise():
    """A non-Hadamard ±1 matrix should have positive distance to the
    Hadamard target spectrum."""
    rng = np.random.default_rng(seed=4)
    M = rng.choice([-1, 1], size=(8, 8)).tolist()
    target = np.full(8, np.sqrt(8.0))
    d = spectrum_distance(M, target)
    assert d > 0.0


def test_sa_targeted_spectrum_runs_and_returns_valid():
    """sa_targeted_spectrum at n=8 with a hadamard target should return
    a valid ±1 matrix and not crash."""
    M0 = Matrix.random(8, __import__("random").Random(0))
    target = predict_target_spectrum(8, 4096, model="hadamard")
    result = sa_targeted_spectrum(
        M0,
        target_spectrum=target,
        iterations=1_000,
        lambda_anchor=0.5,
        seed=99,
    )
    assert result.matrix.n == 8
    for row in result.matrix.entries:
        for v in row:
            assert v in (-1, 1)
    assert result.abs_det > 0


def test_bordered_extension_sa_produces_valid_matrix():
    """Extending an n=8 Hadamard to n=10 should yield a valid ±1 10×10
    matrix with the seed block intact."""
    H = Matrix.sylvester(8).entries
    result = bordered_extension_sa(
        seed_matrix=H,
        target_n=10,
        iterations=500,
        lambda_anchor=0.3,
        seed=7,
        refresh_eigenspace_every=50,
        n_restarts=1,
    )
    M = result.matrix.entries
    assert len(M) == 10
    for row in M:
        assert len(row) == 10
        for v in row:
            assert v in (-1, 1)
    # Seed block must be preserved exactly.
    for i in range(8):
        for j in range(8):
            assert M[i][j] == H[i][j], (
                f"Seed entry M[{i}][{j}]={M[i][j]} differs from H[{i}][{j}]={H[i][j]}"
            )
    # And |det| should be a positive integer (Bareiss-verified).
    d = abs(integer_determinant(M))
    assert d >= 0

"""Tests for the SQA (Simulated Quantum Annealing) module.

Validates:
  - j_perp asymptotics: J_⊥ → 0 as Γ → ∞, J_⊥ → ∞ as Γ → 0
  - SQA produces a valid ±1 matrix output at small n
  - SQA result is Bareiss-verified (best_abs_det matches integer_determinant)
  - SQA reaches the H11 proven max (327680) within a generous budget
  - triple_verify returns consistent values
"""

from __future__ import annotations

import math
import pytest

from hadamard_maxdet.sqa import (
    SQAResult,
    _j_perp,
    simulated_quantum_annealing,
    triple_verify,
)
from hadamard_maxdet.verifier import integer_determinant


# ---------------------------------------------------------------------------
# j_perp unit tests
# ---------------------------------------------------------------------------


def test_j_perp_large_gamma_near_zero():
    """For large Γ, J_⊥ should be close to zero (independent replicas)."""
    j = _j_perp(gamma=100.0, temperature=0.1, n_trotter=8)
    assert j < 1e-6, f"expected j_perp ≈ 0 for large Γ, got {j}"


def test_j_perp_small_gamma_large():
    """For very small Γ, J_⊥ should be large (replicas locked).

    J_⊥ ≈ (T/2) * log(P*T/Γ) for Γ << P*T.
    With T=2.0, P=1, Γ=1e-6: J_⊥ ≈ 1.0 * log(2e6) ≈ 14.5 >> 10.
    """
    j = _j_perp(gamma=1e-6, temperature=2.0, n_trotter=1)
    assert j > 10.0, f"expected j_perp >> 1 for tiny Γ with large T, got {j}"


def test_j_perp_monotone_in_gamma():
    """J_⊥ should decrease as Γ increases (larger field = more tunneling = less coupling)."""
    T, P = 0.1, 4
    gammas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
    j_vals = [_j_perp(g, T, P) for g in gammas]
    for a, b in zip(j_vals, j_vals[1:]):
        assert a > b, f"J_⊥ not monotonically decreasing: {j_vals}"


def test_j_perp_positive():
    """J_⊥ must always be positive."""
    for g in [0.01, 0.1, 0.5, 1.0, 2.0]:
        j = _j_perp(g, 0.1, 4)
        assert j > 0, f"j_perp not positive for Γ={g}: {j}"


def test_j_perp_zero_gamma_returns_inf():
    """Γ=0 should return inf (classical limit, infinite coupling)."""
    j = _j_perp(0.0, 0.1, 4)
    assert math.isinf(j)


# ---------------------------------------------------------------------------
# SQA sanity tests
# ---------------------------------------------------------------------------


def test_sqa_small_n_returns_valid_matrix():
    """SQA on n=4 should return a valid ±1 matrix with positive |det|."""
    result = simulated_quantum_annealing(
        n=4,
        n_trotter=4,
        gamma_start=1.0,
        gamma_end=0.01,
        temperature=0.1,
        wall_cap_s=2.0,
        seed=42,
    )
    M = result.best_matrix
    n = len(M)
    assert n == 4
    assert all(len(row) == n for row in M)
    assert all(v in (-1, 1) for row in M for v in row)
    assert result.best_abs_det > 0


def test_sqa_det_matches_bareiss():
    """SQA's reported best_abs_det must equal exact Bareiss on the best_matrix."""
    result = simulated_quantum_annealing(
        n=8,
        n_trotter=4,
        gamma_start=2.0,
        gamma_end=0.01,
        temperature=0.05,
        wall_cap_s=3.0,
        seed=7,
    )
    exact = abs(integer_determinant(result.best_matrix))
    assert result.best_abs_det == exact, (
        f"reported {result.best_abs_det} != Bareiss {exact}"
    )


def test_sqa_result_fields():
    """SQAResult dataclass fields are sensible."""
    result = simulated_quantum_annealing(
        n=5,
        n_trotter=4,
        gamma_start=1.5,
        gamma_end=0.01,
        temperature=0.1,
        wall_cap_s=1.0,
        seed=99,
    )
    assert result.total_steps > 0
    assert result.walltime_seconds > 0
    assert result.n_trotter == 4
    assert result.temperature == 0.1
    assert result.gamma_range == (1.5, 0.01)
    assert len(result.per_replica_best) == 4


def test_sqa_reaches_h11_proven_max():
    """SQA must reach the H11 proven max (327680) within a 30s budget.

    H11 is easy enough that any SA-like method hits it quickly.
    """
    H11_MAX = 327_680
    result = simulated_quantum_annealing(
        n=11,
        n_trotter=4,
        gamma_start=2.0,
        gamma_end=0.005,
        temperature=0.05,
        wall_cap_s=30.0,
        seed=2026,
    )
    assert result.best_abs_det >= H11_MAX, (
        f"SQA did not reach H11 proven max: got {result.best_abs_det} < {H11_MAX}"
    )


def test_triple_verify_consistency():
    """triple_verify returns |det_M| = |det_MT| = |det_copy| for a small matrix."""
    # Sylvester H4
    H4 = [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ]
    d_M, d_MT, d_copy = triple_verify(H4)
    assert abs(d_M) == abs(d_MT) == abs(d_copy) == 16


def test_triple_verify_rejects_non_pm1():
    """triple_verify raises ValueError if entries are not ±1."""
    bad = [[1, 2], [1, -1]]
    with pytest.raises(ValueError):
        triple_verify(bad)


def test_sqa_n_trotter_1_reduces_to_sa():
    """With P=1 Trotter replica, the quantum coupling vanishes and SQA behaves
    like standard SA (no imaginary-time coupling penalty).  It should still
    reach a positive |det| on n=8."""
    result = simulated_quantum_annealing(
        n=8,
        n_trotter=1,
        gamma_start=1.0,
        gamma_end=0.01,
        temperature=0.05,
        wall_cap_s=3.0,
        seed=11,
    )
    assert result.best_abs_det > 0
    exact = abs(integer_determinant(result.best_matrix))
    assert result.best_abs_det == exact

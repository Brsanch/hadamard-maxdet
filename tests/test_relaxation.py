"""Tests for the SDP / spectral upper-bound relaxations."""

from __future__ import annotations

import math

import pytest

from hadamard_maxdet.orders import record_pm1
from hadamard_maxdet.relaxation import (
    barba_bound_squared,
    ehlich_block_squared,
    ehlich_two_level_squared,
    hadamard_bound_squared,
    lp_upper_bound,
    sdp_upper_bound,
)
from hadamard_maxdet.verifier import upper_bound


# Hadamard orders where the classical bound n^(n/2) is tight (Hadamard matrix exists).
HADAMARD_TIGHT_ORDERS = (1, 2, 4, 8, 12, 16, 20)


# --------------------------------------------------------------------------- #
# Closed-form value sanity                                                    #
# --------------------------------------------------------------------------- #


def test_hadamard_bound_squared_matches_n_to_n() -> None:
    for n in (1, 2, 4, 5, 8, 12, 16, 23):
        assert hadamard_bound_squared(n) == pytest.approx(float(n) ** n)


def test_barba_bound_squared_n23_matches_classical() -> None:
    # Barba: sqrt((2n-1) (n-1)^(n-1)). Squared = (2n-1)·(n-1)^(n-1).
    n = 23
    expected = (2 * n - 1) * (n - 1) ** (n - 1)
    assert barba_bound_squared(n) == pytest.approx(float(expected))
    # And the square root should match verifier.upper_bound to within 1.
    assert math.isclose(math.sqrt(barba_bound_squared(n)), upper_bound(n),
                        rel_tol=1e-12, abs_tol=1.0)


def test_ehlich_block_squared_matches_classical_for_n2_mod4() -> None:
    for n in (6, 10, 14, 18, 22):
        block = ehlich_block_squared(n)
        # Classical Ehlich bound is sqrt of this.
        assert math.isclose(math.sqrt(block), upper_bound(n),
                            rel_tol=1e-12, abs_tol=1.0)


def test_ehlich_two_level_is_diagnostic_below_truth_at_n6() -> None:
    # Single-level Gram is NOT a valid upper bound; demonstrate at n=6
    # where the two-level value 128 is strictly below the true max 160.
    n = 6
    two_level_abs = math.sqrt(ehlich_two_level_squared(n))
    true_max = record_pm1(n)
    assert two_level_abs < true_max, (
        f"single-level Gram value {two_level_abs} should be strictly below "
        f"true max {true_max} at n={n} — confirming it is NOT a valid bound"
    )


# --------------------------------------------------------------------------- #
# Tight at Hadamard orders                                                    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [4, 8, 12, 16, 20])
def test_spectral_bound_matches_proven_at_hadamard_orders(n: int) -> None:
    """At Hadamard orders, the spectral bound = proven max = n^(n/2)."""
    bound = sdp_upper_bound(n, cuts="spectral")
    proven = record_pm1(n)
    assert math.isclose(bound.abs_det_bound, proven, rel_tol=1e-9, abs_tol=1.0), (
        f"n={n}: spectral bound {bound.abs_det_bound} vs proven {proven}"
    )


@pytest.mark.parametrize("n", [4, 8, 12, 16, 20])
def test_trivial_bound_matches_proven_at_hadamard_orders(n: int) -> None:
    """Trivial (Hadamard) bound is also tight at n ≡ 0 (mod 4) when achievable."""
    bound = sdp_upper_bound(n, cuts="trivial")
    proven = record_pm1(n)
    assert math.isclose(bound.abs_det_bound, proven, rel_tol=1e-9, abs_tol=1.0)


# --------------------------------------------------------------------------- #
# Bounds are upper bounds (≥ proven max)                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                               17, 18, 19, 20, 21])
def test_spectral_is_upper_bound_for_all_proven_orders(n: int) -> None:
    bound = sdp_upper_bound(n, cuts="spectral")
    proven = record_pm1(n)
    # Spectral bound must be ≥ proven max (modulo tiny floating-point slack).
    assert bound.abs_det_bound + 1e-6 >= proven, (
        f"n={n}: spectral bound {bound.abs_det_bound} below proven {proven}!"
    )


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                               17, 18, 19, 20, 21])
def test_trivial_is_upper_bound_for_all_proven_orders(n: int) -> None:
    bound = sdp_upper_bound(n, cuts="trivial")
    proven = record_pm1(n)
    assert bound.abs_det_bound + 1e-6 >= proven


def test_lp_matches_spectral() -> None:
    for n in (4, 5, 6, 7, 8, 9, 11, 13, 22, 23):
        a = lp_upper_bound(n).abs_det_bound
        b = sdp_upper_bound(n, cuts="spectral").abs_det_bound
        assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1.0), (
            f"n={n}: lp={a} vs spectral={b}"
        )


# --------------------------------------------------------------------------- #
# n=5 specific check requested in the brief                                   #
# --------------------------------------------------------------------------- #


def test_sdp_at_n5_at_least_proven_48() -> None:
    """At n=5 the SDP must give an upper bound ≥ 48 (proven max)."""
    bound = sdp_upper_bound(n=5, cuts="spectral")
    assert bound.abs_det_bound >= 48 - 1e-6
    # And report the slack (zero for n=5 since Barba is tight: 48 = √2304).
    slack = bound.abs_det_bound - 48
    assert slack < 1e-6, f"slack at n=5 should be ~0 (Barba tight), got {slack}"


def test_sdp_trivial_at_n5_strictly_loose() -> None:
    """The trivial Hadamard relaxation at n=5 is strictly looser than Barba."""
    trivial = sdp_upper_bound(n=5, cuts="trivial")
    spectral = sdp_upper_bound(n=5, cuts="spectral")
    assert trivial.abs_det_bound > spectral.abs_det_bound, (
        f"Hadamard {trivial.abs_det_bound} should exceed Barba {spectral.abs_det_bound}"
    )
    # 5^(5/2) = 55.9017
    assert math.isclose(trivial.abs_det_bound, math.sqrt(5 ** 5),
                        rel_tol=1e-9, abs_tol=1e-6)


# --------------------------------------------------------------------------- #
# CVXPY numerical sanity (small n only — keeps test suite fast)               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_cvxpy_sdp_with_cap_returns_hadamard(n: int) -> None:
    """Verify the magnitude-cap SDP numerically returns the Hadamard answer.

    With |G_ij| ≤ 1 (or any non-negative cap), G = nI is feasible and is
    the unique maximiser of det(G). Any positive cap therefore yields
    the Hadamard bound n^(n/2). Confirming this numerically with CVXPY
    is the cheapest way to rule out a sharper convex relaxation.
    """
    cvxpy = pytest.importorskip("cvxpy")
    bound = sdp_upper_bound(n, cuts="cap", cap=1.0, use_cvxpy=True)
    expected = math.sqrt(hadamard_bound_squared(n))  # n^(n/2)
    # SCS tolerance is moderate; allow 1% slack on the determinant
    # (i.e. ~0.5% on |det|).
    assert math.isclose(bound.abs_det_bound, expected, rel_tol=5e-3), (
        f"n={n}: CVXPY cap-SDP returned {bound.abs_det_bound}, expected ~{expected}"
    )


# --------------------------------------------------------------------------- #
# Detail: spectral matches verifier.upper_bound for proven orders             #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16,
                               17, 19, 20, 21, 22, 23])
def test_spectral_matches_verifier_upper_bound(n: int) -> None:
    bound = sdp_upper_bound(n, cuts="spectral")
    classical = upper_bound(n)
    assert math.isclose(bound.abs_det_bound, classical, rel_tol=1e-12, abs_tol=1.0)

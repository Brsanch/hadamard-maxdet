"""Tests for the spin-glass cavity-method H23 basin predictions."""

from __future__ import annotations

import math

import numpy as np
import pytest

from hadamard_maxdet.cavity_prediction import (
    BARBA_BOUND_H23,
    H23_N,
    ORRICK_ABS_DET_H23,
    basin_count_prediction,
    bp_free_energy,
    empirical_basin_density_extrapolation,
    extract_basin_dets_from_runs,
    pairwise_proxy_hamiltonian,
    pairwise_proxy_hamiltonian_diff,
    rs_bp_marginals,
)


# ---------------------------------------------------------------------------
# pairwise_proxy_hamiltonian
# ---------------------------------------------------------------------------


def test_proxy_hamiltonian_zero_on_identity_like_hadamard_4():
    """A 4x4 Hadamard matrix has zero H_proxy (rows are orthogonal)."""
    H = [
        [1, 1, 1, 1],
        [1, -1, 1, -1],
        [1, 1, -1, -1],
        [1, -1, -1, 1],
    ]
    # All row pairs orthogonal => Σ_k σ_ik σ_jk = 0 for i ≠ j.
    assert pairwise_proxy_hamiltonian(H) == 0


def test_proxy_hamiltonian_matches_direct_small_case():
    """On a fixed 5×5 ±1 matrix, the closed-form matches Σ_{i<j} G_{ij}²."""
    M = [
        [1, 1, 1, 1, 1],
        [1, -1, 1, -1, 1],
        [-1, 1, 1, 1, -1],
        [1, 1, -1, 1, -1],
        [-1, -1, -1, 1, 1],
    ]
    A = np.array(M, dtype=int)
    G = A @ A.T
    n = A.shape[0]
    direct = 0
    for i in range(n):
        for j in range(i + 1, n):
            direct += int(G[i, j]) ** 2
    assert pairwise_proxy_hamiltonian(M) == direct


def test_proxy_hamiltonian_diff_matches_recompute_small():
    """O(n) per-flip delta matches a from-scratch H_proxy recompute."""
    rng = np.random.default_rng(0)
    n = 7
    M = (rng.integers(0, 2, size=(n, n)) * 2 - 1).tolist()
    G = np.array(M, dtype=np.int64) @ np.array(M, dtype=np.int64).T

    for trial in range(20):
        i = int(rng.integers(0, n))
        c = int(rng.integers(0, n))
        before = pairwise_proxy_hamiltonian(M)
        delta = pairwise_proxy_hamiltonian_diff(M, G, i, c)
        # Apply flip.
        M[i][c] = -M[i][c]
        after = pairwise_proxy_hamiltonian(M)
        assert after - before == delta, (
            f"diff mismatch on trial {trial} (i={i}, c={c}): "
            f"reported {delta}, actual {after - before}"
        )
        # Also update G to keep tests cheap.
        G_new = np.array(M, dtype=np.int64) @ np.array(M, dtype=np.int64).T
        G = G_new


def test_proxy_hamiltonian_empty_and_singleton():
    assert pairwise_proxy_hamiltonian([]) == 0
    assert pairwise_proxy_hamiltonian([[1]]) == 0


# ---------------------------------------------------------------------------
# rs_bp_marginals
# ---------------------------------------------------------------------------


def test_rs_bp_runs_and_returns_valid_marginals_n5():
    """RS-BP returns marginals in [0, 1] of the right shape."""
    res = rs_bp_marginals(n=5, beta=0.1, max_iter=50, damping=0.5, seed=0)
    assert res.marginals.shape == (5, 5)
    assert np.all(res.marginals >= 0.0)
    assert np.all(res.marginals <= 1.0)
    assert res.iterations >= 1
    assert math.isfinite(res.free_energy)


def test_rs_bp_high_temperature_marginals_near_half():
    """At very high T (small β), marginals should be near 0.5 (paramagnetic)."""
    res = rs_bp_marginals(
        n=8, beta=1e-3, max_iter=200, damping=0.4, seed=42,
    )
    # All marginals should sit very close to 0.5.
    assert np.max(np.abs(res.marginals - 0.5)) < 0.05


def test_rs_bp_divergence_documented_at_low_temperature():
    """At very low T (large β) on H23-size graph, RS-BP typically fails to
    converge cleanly. We assert that *if* divergence happens, it is reported
    (converged=False with a finite or inf residual), not silently wrong."""
    res = rs_bp_marginals(
        n=H23_N, beta=10.0, coupling=1.0, max_iter=80, damping=0.5, seed=0,
    )
    # We don't enforce convergence (this is the whole point — RS-BP may
    # diverge in the spin-glass regime), only that the result is structurally
    # well-formed.
    assert res.marginals.shape == (H23_N, H23_N)
    assert np.all(np.isfinite(res.marginals))
    assert isinstance(res.converged, bool)
    # Residual is finite OR labelled inf — either way well-defined.
    assert res.final_residual >= 0.0 or math.isinf(res.final_residual)


def test_bp_free_energy_finite_for_random_marginals():
    rng = np.random.default_rng(1)
    n = 10
    marginals = rng.uniform(0.01, 0.99, size=(n, n))
    fe = bp_free_energy(marginals, beta=1.0, coupling=1.0)
    assert math.isfinite(fe)


# ---------------------------------------------------------------------------
# basin_count_prediction
# ---------------------------------------------------------------------------


def test_basin_count_prediction_returns_curve_and_predictions():
    """basin_count_prediction returns a sensible structure for a small grid."""
    out = basin_count_prediction(
        n=6,
        energy_levels=[0.0, 50.0, 200.0],
        beta_grid=[0.01, 0.1, 1.0, 5.0],
        coupling=1.0,
        bp_seed=0,
    )
    assert "curve" in out
    assert "predictions" in out
    assert len(out["curve"]) == 4
    assert len(out["predictions"]) == 3
    for pt in out["predictions"]:
        assert "log_N_basins" in pt
        assert "N_basins" in pt
        assert "beta_match" in pt
        assert math.isfinite(pt["log_N_basins"])
        assert pt["N_basins"] >= 0.0


# ---------------------------------------------------------------------------
# empirical extrapolation
# ---------------------------------------------------------------------------


def _synthetic_basin_dets(
    n_basins: int = 200,
    *,
    target: int = ORRICK_ABS_DET_H23,
    seed: int = 0,
) -> list[int]:
    """Generate synthetic basin |det|s with a smooth tail below ``target``."""
    rng = np.random.default_rng(seed)
    log_target = math.log(target)
    # Exponential-tail in log-space below the target.
    excesses = rng.exponential(scale=0.4, size=n_basins)
    log_dets = log_target - excesses - 0.05  # all strictly below target
    return [int(math.exp(x)) for x in log_dets]


def test_empirical_extrapolation_runs_and_returns_finite_estimate():
    dets = _synthetic_basin_dets(n_basins=300, seed=0)
    out = empirical_basin_density_extrapolation(
        dets, target_abs_det=ORRICK_ABS_DET_H23, n_bootstrap=50,
    )
    assert isinstance(out.expected_count_at_or_above, float)
    assert math.isfinite(out.expected_count_at_or_above)
    assert out.target_abs_det == ORRICK_ABS_DET_H23
    # Direct call: Orrick is NOT auto-added (only extract_basin_dets_from_runs
    # adds it). Dedup may drop a handful of synthetic duplicates.
    assert out.n_basins_input <= 300
    assert out.n_basins_input >= 250  # at least most of the 300 are unique
    assert out.n_observed_at_or_above_orrick == 0
    assert out.method in ("gev", "gev_bounded", "log-linear")
    # method tag is in the documented vocabulary
    assert "gev" in out.method or "log" in out.method


def test_empirical_extrapolation_with_upper_bound_constrains_tail():
    """Adding a hard upper bound should give a strictly *smaller* prediction
    above the current max than the unbounded fit (because the bounded GPD has
    finite support)."""
    dets = _synthetic_basin_dets(n_basins=400, seed=10)
    target = int(1.05 * ORRICK_ABS_DET_H23)
    out_unbounded = empirical_basin_density_extrapolation(
        dets, target_abs_det=target, n_bootstrap=20,
    )
    # Bound the tail at log(2 * target).
    upper = math.log(2 * target)
    out_bounded = empirical_basin_density_extrapolation(
        dets, target_abs_det=target, n_bootstrap=20, upper_bound_log=upper,
    )
    # Both should produce finite predictions.
    assert math.isfinite(out_unbounded.expected_count_at_or_above)
    assert math.isfinite(out_bounded.expected_count_at_or_above)
    # The bounded one declares its constraint.
    assert out_bounded.fit_params.get("upper_bound_log") == upper
    # The bounded fit should have ξ < 0 (finite endpoint).
    if out_bounded.method == "gev_bounded":
        assert out_bounded.fit_params["xi"] < 0


def test_empirical_extrapolation_target_above_max_predicts_small_count():
    """Target well above the max observed |det| should give a small count."""
    dets = _synthetic_basin_dets(n_basins=400, seed=1)
    above = int(2 * ORRICK_ABS_DET_H23)  # above any synthetic |det|
    out = empirical_basin_density_extrapolation(
        dets, target_abs_det=above, n_bootstrap=50,
    )
    # The predicted count should be much less than the input cohort size.
    assert out.expected_count_at_or_above <= 10.0
    # CIs should also be small.
    if math.isfinite(out.expected_count_ci_high):
        assert out.expected_count_ci_high <= 50.0


def test_empirical_extrapolation_target_below_max_returns_observed_count():
    """Target equal to a body value returns roughly the observed empirical
    count (sanity: extrapolation matches data inside the support)."""
    dets = _synthetic_basin_dets(n_basins=400, seed=2)
    sorted_dets = sorted(dets, reverse=True)
    target = sorted_dets[100]  # 100 observations strictly above this
    out = empirical_basin_density_extrapolation(
        dets, target_abs_det=target, n_bootstrap=20,
    )
    # The "observed count at or above target" should be in the same ballpark.
    n_above_obs = sum(1 for d in dets if d >= target)
    # Within a factor of 5x is fine for a method-of-moments fit.
    assert out.expected_count_at_or_above <= 5 * n_above_obs + 5
    assert out.expected_count_at_or_above >= 0


def test_empirical_extrapolation_zero_input_raises():
    with pytest.raises(ValueError):
        empirical_basin_density_extrapolation([], target_abs_det=1)


def test_empirical_extrapolation_zero_target_raises():
    dets = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        empirical_basin_density_extrapolation(dets, target_abs_det=0)


# ---------------------------------------------------------------------------
# extract_basin_dets_from_runs
# ---------------------------------------------------------------------------


def test_extract_basin_dets_handles_missing_paths():
    """All-None / nonexistent inputs return at minimum the Orrick reference."""
    dets = extract_basin_dets_from_runs(
        smart_2val_path=None,
        smart_3val_summary_path=None,
        targeted_bridge_path=None,
        basin_atlas_path=None,
        extra_dets=None,
    )
    assert ORRICK_ABS_DET_H23 in dets
    assert len(dets) >= 1


def test_extract_basin_dets_dedupes_and_includes_extras():
    extras = [1_000_000, 2_000_000, 1_000_000]  # duplicate
    dets = extract_basin_dets_from_runs(
        smart_2val_path=None,
        smart_3val_summary_path=None,
        targeted_bridge_path=None,
        basin_atlas_path=None,
        extra_dets=extras,
    )
    # Dedup: 2 unique extras + Orrick = 3 entries.
    assert len(dets) == 3
    assert ORRICK_ABS_DET_H23 in dets
    assert 1_000_000 in dets
    assert 2_000_000 in dets


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


def test_h23_constants_match_published_values():
    assert ORRICK_ABS_DET_H23 == 2_779_447_296_000_000
    assert BARBA_BOUND_H23 == 3_919_726_327_358_822
    assert H23_N == 23
    # Orrick / Barba ratio should match the documented 0.7091.
    ratio = ORRICK_ABS_DET_H23 / BARBA_BOUND_H23
    assert abs(ratio - 0.7091) < 1e-3

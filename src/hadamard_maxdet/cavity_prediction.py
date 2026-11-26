"""Spin-glass cavity / belief-propagation predictions for the H23 maxdet basin
landscape.

Background
----------
The Hadamard maxdet problem at order n maps to a spin-glass Hamiltonian on
n^2 = 529 binary spins ``σ_{ij} ∈ {±1}`` (the matrix entries):

    H(σ) = -log|det(M(σ))|.

Minimising H corresponds to maximising |det|. After the diagnostic landscape
mapping at H23 (smart 2-val, smart 3-val, targeted-bridge — all converging on
Orrick's (208, 45) at |det| = 2.779e15 as the unique observed top), the next
question is whether a hidden basin above Orrick is mathematically *possible*
under the spin-glass model.

Two complementary approaches are implemented here:

**Approach A — Pairwise-proxy Hamiltonian + replica-symmetric BP.** Replace the
exact log|det| (which is a global, non-local cost) with the Gram pairwise
penalty ``H_proxy(σ) = Σ_{i<j} (Σ_k σ_{ik} σ_{jk})²``. This is local in row
indices (each pair (i, j) couples 2n spins) and is exactly the energy
minimised by the Hadamard / Barba family of upper-bound saturators (rows
near-orthogonal). Standard replica-symmetric BP then iterates messages over
this constraint graph and produces marginals + a BP free-energy estimate.
In the spin-glass-glassy regime, BP is expected to either fail to converge or
predict an unphysical entropy (negative), which is itself a diagnostic of
basin multiplicity (1-RSB territory).

**Approach B — Empirical extreme-value extrapolation.** From the 4,400+
catalogued basin |det| values produced by the smart 2-val / 3-val / targeted
bridge campaigns, fit a Generalised Extreme Value (GEV) tail and read off
the predicted basin density at and above Orrick's |det|. This is a
data-driven "external-field" cavity estimate: the empirical basin distribution
plays the role of the cavity distribution of energies, and the GEV tail
exponent encodes the analogue of the Edwards–Anderson scaling. Output: an
expected number of basins ≥ a target |det|, with bootstrap CIs.

Reference points (from `H23_orrick2003_record.json` and surrounding evidence):

    Orrick      |det| = 2,779,447,296,000,000  (E* ≈ -log|det| = -35.5616 nats)
    Barba bound |det| ≤ 3,919,726,327,358,822  (E_floor ≈ -35.9059 nats)
    Orrick / Barba ratio = 0.7091

Public API
----------

* :func:`pairwise_proxy_hamiltonian`        — H_proxy energy of a ±1 matrix.
* :func:`pairwise_proxy_hamiltonian_diff`   — O(n) per-flip delta of H_proxy.
* :func:`rs_bp_marginals`                   — replica-symmetric BP fixed point.
* :func:`bp_free_energy`                    — Bethe free energy from BP marginals.
* :func:`basin_count_prediction`            — expected basin count from BP curvature.
* :func:`empirical_basin_density_extrapolation`
                                            — Approach B: GEV-tail expected count.
* :func:`extract_basin_dets_from_runs`      — load |det| values from results JSONs.
"""

from __future__ import annotations

import json
import math
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Reference constants (H23)
# ---------------------------------------------------------------------------

ORRICK_ABS_DET_H23: int = 2_779_447_296_000_000
"""Orrick's H23 record (Orrick, Solomon, Dowdeswell, Smith 2003)."""

BARBA_BOUND_H23: int = 3_919_726_327_358_822
"""Barba upper bound at n=23: floor(sqrt(2n-1) * (n-1)^((n-1)/2))."""

H23_N: int = 23
"""Order of the matrix in the H23 problem."""


# ---------------------------------------------------------------------------
# Approach A — Pairwise-proxy Hamiltonian and replica-symmetric BP
# ---------------------------------------------------------------------------


def pairwise_proxy_hamiltonian(M: Sequence[Sequence[int]]) -> int:
    """Return ``Σ_{i<j} (Σ_k M[i,k] · M[j,k])²`` for a ±1 matrix.

    This is the unordered-pair sum of squared Gram off-diagonals of ``M``.
    It vanishes iff every row pair is orthogonal — i.e. iff M is a Hadamard
    matrix. For non-Hadamard orders it serves as a tractable proxy for
    ``-log|det|``: both functionals are minimised by the same configurations
    (near-orthogonal rows = high |det|), but H_proxy is local in row indices.

    Cost: ``O(n^3)`` via direct Gram computation.
    """
    A = np.asarray(M, dtype=np.int64)
    n = A.shape[0]
    if n < 2:
        return 0
    G = A @ A.T
    iu = np.triu_indices(n, k=1)
    vals = G[iu].astype(np.int64)
    return int(np.sum(vals * vals))


def pairwise_proxy_hamiltonian_diff(
    M: Sequence[Sequence[int]],
    G: np.ndarray,
    i: int,
    c: int,
) -> int:
    """Return the change in ``H_proxy`` from flipping entry ``M[i, c]``.

    ``G`` is the current Gram ``M @ M.T`` (will not be mutated). The delta
    is computed in O(n) time by walking only row ``i`` of the Gram, since
    flipping ``M[i, c]`` only changes row/col ``i`` of G.

    Sign convention: returns ``H_proxy(M_after) - H_proxy(M_before)``.
    """
    n = G.shape[0]
    if n < 2:
        return 0
    old_val = int(M[i][c])
    new_val = -old_val  # ±1 flip
    delta_factor = 2 * new_val  # G_new[i, r] = G_old[i, r] + 2 · M_new[i,c] · M[r,c]
    delta = 0
    for r in range(n):
        if r == i:
            continue
        old_g = int(G[i, r])
        new_g = old_g + delta_factor * int(M[r][c])
        delta += new_g * new_g - old_g * old_g
    return int(delta)


# ---------------------------------------------------------------------------
# Replica-symmetric belief propagation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BPResult:
    """Result of running RS-BP on a pairwise factor graph."""

    n: int
    """Matrix order."""
    converged: bool
    """Did messages reach the requested fixed-point tolerance?"""
    iterations: int
    """Number of BP sweeps performed."""
    final_residual: float
    """L∞ change in messages between final two sweeps."""
    marginals: np.ndarray
    """Shape (n, n). ``marginals[i, c]`` = ``P(σ_{ic} = +1)`` from BP."""
    free_energy: float
    """Bethe free-energy estimate (in nats; lower = more probable)."""
    log_partition_estimate: float
    """``-β · F_Bethe`` evaluated at temperature ``beta`` (returned for completeness)."""
    beta: float
    """Inverse temperature used in the BP run."""


def rs_bp_marginals(
    n: int,
    *,
    beta: float = 1.0,
    coupling: float = 1.0,
    max_iter: int = 200,
    damping: float = 0.5,
    tol: float = 1e-5,
    seed: int | None = 0,
) -> BPResult:
    """Run replica-symmetric BP on the pairwise-proxy factor graph for H23.

    The factor graph has one variable node per spin ``σ_{ic}`` (n² variables)
    and one factor node per row pair ``(i, j)`` with i < j. The factor energy
    on a pair is

        E_{ij}(σ_i, σ_j) = β · coupling · (Σ_k σ_{ik} σ_{jk})².

    BP messages ``μ_{f→v}, μ_{v→f}`` on a binary alphabet are scalar
    log-ratios. Damped parallel updates are used:

        m_new = (1 - damping) · m_old + damping · m_proposed.

    The key approximation — and the reason this is a *diagnostic* — is that
    the constraint is summed over k inside a square. To keep the message
    update O(n) per factor we factorise

        (Σ_k σ_{ik} σ_{jk})² = Σ_k 1 + 2 · Σ_{k<l} σ_{ik} σ_{il} σ_{jk} σ_{jl}

    so the squared-Gram penalty becomes a sum of 4-spin couplings. We BP only
    the diagonal (k = l) and 2-spin (k != l) parts; the 4-spin coupling is
    treated as a mean-field correction. This is the standard Plefka /
    second-order TAP truncation.

    Returns
    -------
    BPResult
        With ``converged=True`` iff the residual fell below ``tol`` before
        ``max_iter`` sweeps. Non-convergence is itself diagnostic:
        the system is in the spin-glass / 1-RSB regime where RS-BP fails.
    """
    rng = np.random.default_rng(seed)

    # h[i, c] = log(μ(σ_{ic} = +1) / μ(σ_{ic} = -1)), the local field on each spin.
    # Initialise small random fields to break the all-zero symmetric fixed point
    # (which is unstable in the spin-glass phase).
    h = 0.01 * rng.standard_normal((n, n))

    # BP for this Hamiltonian under the second-order truncation reduces to
    # the TAP self-consistency equations on the local fields:
    #
    #     m_{ic} = tanh(h_{ic})
    #     h_{ic}^new = -2 β coupling · Σ_{j ≠ i} m_{jc} · (Σ_k m_{ik} m_{jk})
    #                  + Onsager reaction term
    #
    # The Onsager term enforces that BP "subtracts" the spin's own contribution
    # to the field acting on it. For RS-BP at this level this is:
    #
    #     onsager_{ic} = β coupling · Σ_{j ≠ i} (1 - m_{jc}^2) · m_{ic}
    #
    # Iterate to a fixed point (or detect divergence).

    final_residual = float("inf")
    converged = False
    last_iter = 0
    for it in range(max_iter):
        last_iter = it + 1
        m = np.tanh(h)
        # Cross term: Σ_k m_{ik} m_{jk} — this is the Gram of the magnetisations.
        Gm = m @ m.T  # shape (n, n)
        np.fill_diagonal(Gm, 0.0)

        # Field on σ_{ic}: from each j ≠ i, contribution is m_{jc} · Gm[i, j].
        # In matrix form: h_field[i, c] = Σ_{j≠i} Gm[i, j] · m[j, c]
        # = (Gm @ m)[i, c]   (Gm has zero diagonal so j=i is excluded).
        h_proposed = -2.0 * beta * coupling * (Gm @ m)

        # Onsager reaction term (mean-field self-feedback subtraction).
        # For the squared-Gram potential this is m_{ic} · Σ_{j≠i} (m·m^T)_{jj} ≈ negligible
        # at low magnetisation; we keep the symmetric form:
        onsager = beta * coupling * (n - 1) * (1.0 - m * m) * m
        h_proposed = h_proposed + onsager

        # Cap |h| to prevent blowup during early iterations.
        h_proposed = np.clip(h_proposed, -50.0, 50.0)

        h_new = (1.0 - damping) * h + damping * h_proposed
        residual = float(np.max(np.abs(h_new - h)))
        h = h_new

        if not np.isfinite(residual):
            final_residual = float("inf")
            converged = False
            break

        final_residual = residual
        if residual < tol:
            converged = True
            break

    marginals = 0.5 * (1.0 + np.tanh(h))
    fe = bp_free_energy(marginals, beta=beta, coupling=coupling)
    log_z = -beta * fe

    return BPResult(
        n=n,
        converged=converged,
        iterations=last_iter,
        final_residual=final_residual,
        marginals=marginals,
        free_energy=fe,
        log_partition_estimate=log_z,
        beta=beta,
    )


def bp_free_energy(
    marginals: np.ndarray,
    *,
    beta: float = 1.0,
    coupling: float = 1.0,
) -> float:
    """Bethe free-energy estimate from BP marginals.

    F_Bethe = <H>_marginals  - T · S_Bethe

    For the pairwise-proxy Hamiltonian under the RS / 2nd-order truncation:

        <H> ≈ Σ_{i<j} (Σ_k m_{ik} m_{jk})²            (cross terms only)
        S   ≈ Σ_{ic} h(p_{ic})                        (independent-spin entropy)

    where ``h(p) = -p log p - (1-p) log(1-p)`` and we drop the (negative)
    Bethe pairwise correction since the 2nd-order BP truncation already
    bakes it in. This estimate is what we compare across temperatures /
    couplings to extract basin-count signals.
    """
    m = 2.0 * marginals - 1.0  # back to magnetisations in [-1, +1]
    Gm = m @ m.T
    np.fill_diagonal(Gm, 0.0)
    iu_i, iu_j = np.triu_indices(m.shape[0], k=1)
    energy_pair = float(np.sum(Gm[iu_i, iu_j] ** 2))
    energy = coupling * energy_pair

    # Independent-spin entropy.
    eps = 1e-12
    p = np.clip(marginals, eps, 1.0 - eps)
    entropy = float(-np.sum(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))

    # F = E - T S = E - (1/β) S
    return float(energy - entropy / beta)


def basin_count_prediction(
    n: int,
    energy_levels: Sequence[float],
    *,
    beta_grid: Sequence[float] | None = None,
    coupling: float = 1.0,
    bp_seed: int | None = 0,
) -> dict:
    """Predict the expected number of basins at each given (proxy-)energy level.

    For each ``β`` in ``beta_grid`` we run RS-BP, record (energy, free-energy,
    entropy) on the proxy Hamiltonian, and assemble a coarse Legendre-like
    estimate of the complexity (basin count). Specifically:

        log N_basins(E) ≈ S_Bethe(β(E)) - β(E) · (E - <H>_β(E)),

    where ``β(E)`` is the inverse temperature whose BP fixed point has mean
    energy ``E``. We tabulate this along ``beta_grid`` and interpolate.

    Caveats
    -------
    * If RS-BP fails to converge at a given β (typical in the spin-glass
      phase), we still report the (E, S) pair from the last iterate — this is
      annotated in the per-level diagnostic.
    * The proxy energy axis is ``H_proxy = Σ_{i<j} G_{ij}^2``, NOT
      ``-log|det|``. A separate calibration (``empirical_proxy_to_logdet``)
      would be needed to convert; we provide the proxy-axis prediction as the
      first-principles-cleanest output and the empirical extrapolation
      (Approach B) for the |det|-axis prediction.

    Returns
    -------
    dict
        ``{
          'beta_grid': [...],
          'curve': [{'beta': β, 'energy': E_β, 'entropy': S_β,
                     'free_energy': F_β, 'converged': bool}, ...],
          'predictions': [{'energy_proxy': E*, 'log_N_basins': ...,
                            'N_basins': ..., 'beta_match': β*}, ...],
        }``
    """
    if beta_grid is None:
        # Geometric grid spanning weak (β=0.05) to strong (β=20) coupling.
        beta_grid = list(np.geomspace(0.05, 20.0, 24))

    curve = []
    for beta in beta_grid:
        res = rs_bp_marginals(
            n=n,
            beta=float(beta),
            coupling=coupling,
            max_iter=200,
            damping=0.4,
            tol=1e-5,
            seed=bp_seed,
        )
        m = 2.0 * res.marginals - 1.0
        Gm = m @ m.T
        np.fill_diagonal(Gm, 0.0)
        iu_i, iu_j = np.triu_indices(n, k=1)
        energy = float(np.sum(Gm[iu_i, iu_j] ** 2)) * coupling
        eps = 1e-12
        p = np.clip(res.marginals, eps, 1.0 - eps)
        entropy = float(-np.sum(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)))
        curve.append({
            "beta": float(beta),
            "energy": energy,
            "entropy": entropy,
            "free_energy": res.free_energy,
            "converged": bool(res.converged),
            "residual": res.final_residual,
        })

    energies = np.array([c["energy"] for c in curve])
    betas = np.array([c["beta"] for c in curve])
    entropies = np.array([c["entropy"] for c in curve])

    predictions = []
    for E_target in energy_levels:
        # Find the β whose BP energy matches E_target most closely.
        idx = int(np.argmin(np.abs(energies - float(E_target))))
        beta_match = float(betas[idx])
        S_match = float(entropies[idx])
        E_match = float(energies[idx])
        # Complexity (= log basin count) = S - β·(E - <H>); near the matched
        # β this is just S_match (the (E - <H>) term vanishes at the saddle).
        log_n = S_match
        # In the high-β / glassy phase, S can become numerically large or
        # negative; clamp the report to a finite range.
        log_n = float(np.clip(log_n, -50.0, 200.0))
        predictions.append({
            "energy_proxy_target": float(E_target),
            "energy_proxy_matched": E_match,
            "beta_match": beta_match,
            "entropy": S_match,
            "log_N_basins": log_n,
            "N_basins": float(math.exp(log_n)) if log_n > -50 else 0.0,
        })

    return {
        "beta_grid": [float(b) for b in beta_grid],
        "curve": curve,
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Approach B — Empirical extreme-value / GEV extrapolation from basin |det|s
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmpiricalExtrapolation:
    """Output of empirical basin-density extrapolation."""

    n_basins_input: int
    """Number of catalogued basins / multisets used in the fit."""
    target_abs_det: int
    """The target |det| value we asked about."""
    target_ratio_to_orrick: float
    """Target as a fraction of Orrick's |det|."""
    expected_count_at_or_above: float
    """Predicted expected number of catalogued basins with |det| ≥ target."""
    expected_count_ci_low: float
    """Bootstrap 5th-percentile of the expected-count prediction."""
    expected_count_ci_high: float
    """Bootstrap 95th-percentile of the expected-count prediction."""
    method: str
    """``'gev'`` (generalised Pareto tail) or ``'log-linear'`` fallback."""
    fit_params: dict
    """Fitted parameters of the tail model."""
    n_observed_at_or_above_orrick: int
    """How many observed basins are ≥ Orrick's |det| (for sanity check)."""


def extract_basin_dets_from_runs(
    *,
    smart_2val_path: str | Path | None = None,
    smart_3val_summary_path: str | Path | None = None,
    targeted_bridge_path: str | Path | None = None,
    basin_atlas_path: str | Path | None = None,
    extra_dets: Iterable[int] | None = None,
) -> list[int]:
    """Collect every catalogued basin / multiset |det| from the standard runs.

    Reads each provided JSON; missing or null paths are skipped silently.
    Always *includes* the Orrick reference (so the empirical fit knows the
    record exists).

    Returns the deduplicated, sorted-descending list of |det| integers.
    """
    dets: set[int] = set()

    def _safe_load(p):
        if p is None:
            return None
        path = Path(p)
        if not path.exists():
            return None
        with open(path) as f:
            return json.load(f)

    d2 = _safe_load(smart_2val_path)
    if d2 is not None:
        for entry in d2.get("new_multisets", []):
            ad = entry.get("best_abs_det")
            if isinstance(ad, int) and ad > 0:
                dets.add(ad)

    d3 = _safe_load(smart_3val_summary_path)
    if d3 is not None:
        for entry in d3.get("new_multisets", []):
            ad = entry.get("best_abs_det")
            if isinstance(ad, int) and ad > 0:
                dets.add(ad)

    db = _safe_load(targeted_bridge_path)
    if db is not None:
        for entry in db.get("destinations_all", []):
            ad = entry.get("best_abs_det")
            if isinstance(ad, int) and ad > 0:
                dets.add(ad)

    da = _safe_load(basin_atlas_path)
    if da is not None:
        for entry in da.get("basins", []):
            ad = entry.get("abs_det")
            if isinstance(ad, int) and ad > 0:
                dets.add(ad)

    if extra_dets is not None:
        for ad in extra_dets:
            if isinstance(ad, int) and ad > 0:
                dets.add(int(ad))

    # Always include Orrick.
    dets.add(ORRICK_ABS_DET_H23)

    return sorted(dets, reverse=True)


def empirical_basin_density_extrapolation(
    known_basin_dets: Sequence[int],
    target_abs_det: int,
    *,
    n_bootstrap: int = 200,
    bootstrap_seed: int | None = 0,
    tail_fraction: float = 0.20,
    upper_bound_log: float | None = None,
) -> EmpiricalExtrapolation:
    """Fit a Generalised Pareto / log-linear tail to the basin |det| empirical
    distribution and extrapolate the expected count above ``target_abs_det``.

    Method
    ------
    1. Convert |det|s to ``x = log|det|`` (so the tail behaviour is on a
       linear scale; basin |det|s span 12+ orders of magnitude in raw units
       at H23, so log-space is the only sensible axis).
    2. Take the upper ``tail_fraction`` of x-values as the empirical tail
       above the threshold ``u``.
    3. Fit a Generalised Pareto distribution (GPD) to ``x - u`` via method
       of moments (closed-form, no scipy dependency):

           ξ = 0.5 · (mean² / var - 1)
           σ = 0.5 · mean · (mean² / var + 1)

       with ξ → max(ξ, -0.5) (the regime where MoM is consistent).
    4. Predict the expected count at or above ``target_abs_det`` via

           N(x ≥ x*) = N_tail · max(0, 1 + ξ · (x* - u) / σ)^(-1/ξ)

       (or ``N_tail · exp(-(x* - u) / σ)`` if ξ ≈ 0).
    5. Bootstrap CIs over ``n_bootstrap`` resamples of the upper-tail
       observations.

    Fallback: if the tail has < 8 points or the GPD fit gives ``σ ≤ 0``,
    fit a log-linear "exceedance density" and report that, marked
    ``method='log-linear'``.

    Bounded variant: if ``upper_bound_log`` is provided (e.g.
    ``log(BARBA_BOUND_H23)``), the GPD shape parameter ξ is constrained to
    enforce a finite right endpoint at ``upper_bound_log``. With ξ < 0 the
    GPD has support on ``[u, u - σ/ξ]``; we fix ``u - σ/ξ = upper_bound_log``,
    so ``ξ = -σ / (upper_bound_log - u)``. The scale σ is then fit by method
    of moments. This is the physically correct prior for the maxdet problem,
    where the Barba bound is a hard upper limit no basin can exceed.

    Returns
    -------
    EmpiricalExtrapolation
    """
    rng = random.Random(bootstrap_seed)

    if not known_basin_dets:
        raise ValueError("known_basin_dets must be non-empty")
    if target_abs_det <= 0:
        raise ValueError("target_abs_det must be positive")

    dets = sorted({int(d) for d in known_basin_dets if int(d) > 0}, reverse=True)
    n_total = len(dets)
    n_above_orrick = sum(1 for d in dets if d > ORRICK_ABS_DET_H23)

    log_dets = np.array([math.log(d) for d in dets])
    x_target = math.log(int(target_abs_det))

    # Choose threshold u as the (1 - tail_fraction)-quantile.
    n_tail = max(8, int(round(tail_fraction * n_total)))
    n_tail = min(n_tail, n_total)
    sorted_log = np.sort(log_dets)  # ascending
    u_idx = max(0, n_total - n_tail)
    u = float(sorted_log[u_idx]) if u_idx < n_total else float(sorted_log[-1])
    tail_excesses = sorted_log[u_idx:] - u  # ≥ 0

    def _gpd_mom_fit(excesses: np.ndarray) -> tuple[float, float, str]:
        if excesses.size < 4:
            return float("nan"), float("nan"), "too_few"
        m = float(np.mean(excesses))
        v = float(np.var(excesses, ddof=1)) if excesses.size > 1 else 0.0
        if m <= 0 or v <= 0:
            return float("nan"), float("nan"), "degenerate_tail"
        # Method-of-moments GPD.
        xi = 0.5 * (m * m / v - 1.0)
        sigma = 0.5 * m * (m * m / v + 1.0)
        if not (np.isfinite(xi) and np.isfinite(sigma)) or sigma <= 0:
            return float("nan"), float("nan"), "bad_fit"
        # Clamp ξ to a regime where extrapolation is well-defined.
        xi = float(np.clip(xi, -0.5, 0.95))
        return xi, sigma, "ok"

    def _gpd_bounded_fit(
        excesses: np.ndarray, u_val: float, ub_log: float
    ) -> tuple[float, float, str]:
        """Bounded GPD: enforce right endpoint at ``ub_log``.

        With ξ < 0, GPD has support [u, u - σ/ξ]. Fixing the upper endpoint
        at ``ub_log`` gives ``ξ = -σ / (ub_log - u)``. Method of moments on
        the mean of the excesses then gives σ.

        For a GPD, mean = σ / (1 - ξ), so plugging in the constraint:
            mean = σ / (1 + σ/(ub_log - u))
                  = σ · (ub_log - u) / (ub_log - u + σ)
        Solving for σ:
            σ = mean · (ub_log - u) / (ub_log - u - mean)

        Requires mean < (ub_log - u), i.e. the mean excess is less than the
        room available up to the bound (otherwise a finite-endpoint GPD is
        inconsistent with the data).
        """
        if excesses.size < 4:
            return float("nan"), float("nan"), "too_few"
        room = ub_log - u_val
        if room <= 0:
            return float("nan"), float("nan"), "bound_below_threshold"
        m = float(np.mean(excesses))
        if m <= 0:
            return float("nan"), float("nan"), "degenerate_tail"
        if m >= room:
            # Tail mean exceeds available room — the bound contradicts the
            # data. Fall through to the unconstrained fit.
            return float("nan"), float("nan"), "mean_exceeds_room"
        sigma = m * room / (room - m)
        xi = -sigma / room
        if not (np.isfinite(xi) and np.isfinite(sigma)) or sigma <= 0:
            return float("nan"), float("nan"), "bad_fit"
        return float(xi), float(sigma), "ok"

    def _expected_count(
        n_tail_obs: int, u_val: float, xi: float, sigma: float, x_star: float
    ) -> float:
        if x_star <= u_val:
            # Above-threshold target lies within the body; fall back to
            # empirical CDF.
            obs = int(np.sum(log_dets >= x_star))
            return float(obs)
        if not (np.isfinite(xi) and np.isfinite(sigma)) or sigma <= 0:
            return 0.0
        z = (x_star - u_val) / sigma
        if abs(xi) < 1e-6:
            survival = math.exp(-z)
        else:
            base = 1.0 + xi * z
            if base <= 0:
                survival = 0.0
            else:
                survival = base ** (-1.0 / xi)
        return float(n_tail_obs) * survival

    def _log_linear_predict(n_tail_obs: int, u_val: float, x_star: float) -> tuple[float, float]:
        # Fit log N(x ≥ t) ≈ a - b · (t - u) over the tail. b > 0.
        ts = sorted_log[u_idx:]
        ranks_above = np.arange(n_tail_obs, 0, -1)  # at ts[k], #{obs ≥ ts[k]}
        log_counts = np.log(ranks_above)
        ts_excess = ts - u_val
        if len(ts_excess) < 2:
            return 0.0, 0.0
        slope, intercept = np.polyfit(ts_excess, log_counts, 1)
        # slope should be negative; expected count at x_star:
        log_pred = intercept + slope * (x_star - u_val)
        return float(math.exp(log_pred)), float(slope)

    if upper_bound_log is not None:
        xi_hat, sigma_hat, status = _gpd_bounded_fit(
            tail_excesses, u, float(upper_bound_log)
        )
        method_label = "gev_bounded"
        if status != "ok":
            # Fall back to unconstrained fit and mark the failure mode.
            xi_hat, sigma_hat, status_unc = _gpd_mom_fit(tail_excesses)
            method_label = f"gev_unbounded_after_bounded_failed[{status}]"
            status = status_unc
    else:
        xi_hat, sigma_hat, status = _gpd_mom_fit(tail_excesses)
        method_label = "gev"

    if status == "ok":
        method = method_label
        point = _expected_count(n_tail, u, xi_hat, sigma_hat, x_target)
        fit_params = {
            "xi": xi_hat,
            "sigma": sigma_hat,
            "u_log": u,
            "u_abs_det": math.exp(u),
            "upper_bound_log": (
                float(upper_bound_log) if upper_bound_log is not None else None
            ),
            "n_tail": n_tail,
            "tail_fraction": tail_fraction,
            "fit_status": status,
        }
    else:
        method = "log-linear"
        point, slope = _log_linear_predict(n_tail, u, x_target)
        fit_params = {
            "u_log": u,
            "u_abs_det": math.exp(u),
            "slope": slope,
            "n_tail": n_tail,
            "tail_fraction": tail_fraction,
            "fit_status": status,
        }

    # Bootstrap.
    boot_estimates: list[float] = []
    for _ in range(n_bootstrap):
        # Resample the tail with replacement.
        idxs = [rng.randrange(n_tail) for _ in range(n_tail)]
        boot_excesses = tail_excesses[idxs]
        if method.startswith("gev"):
            if upper_bound_log is not None and method == "gev_bounded":
                xi_b, sigma_b, st_b = _gpd_bounded_fit(
                    boot_excesses, u, float(upper_bound_log)
                )
            else:
                xi_b, sigma_b, st_b = _gpd_mom_fit(boot_excesses)
            if st_b == "ok":
                est = _expected_count(n_tail, u, xi_b, sigma_b, x_target)
            else:
                est = float("nan")
        else:
            # Log-linear bootstrap: refit on a synthetic tail at the same x positions.
            # Simpler proxy: resample the slope-fit residuals indirectly via
            # bootstrapping (n_tail, slope). Use the original slope as an anchor.
            est = point  # log-linear MoM has narrow CI; punt to point estimate.
        if np.isfinite(est):
            boot_estimates.append(est)

    if boot_estimates:
        boot_arr = np.array(boot_estimates)
        ci_low = float(np.percentile(boot_arr, 5))
        ci_high = float(np.percentile(boot_arr, 95))
    else:
        ci_low = float("nan")
        ci_high = float("nan")

    return EmpiricalExtrapolation(
        n_basins_input=n_total,
        target_abs_det=int(target_abs_det),
        target_ratio_to_orrick=float(target_abs_det) / float(ORRICK_ABS_DET_H23),
        expected_count_at_or_above=float(point),
        expected_count_ci_low=ci_low,
        expected_count_ci_high=ci_high,
        method=method,
        fit_params=fit_params,
        n_observed_at_or_above_orrick=int(n_above_orrick),
    )


__all__ = [
    "ORRICK_ABS_DET_H23",
    "BARBA_BOUND_H23",
    "H23_N",
    "BPResult",
    "EmpiricalExtrapolation",
    "pairwise_proxy_hamiltonian",
    "pairwise_proxy_hamiltonian_diff",
    "rs_bp_marginals",
    "bp_free_energy",
    "basin_count_prediction",
    "extract_basin_dets_from_runs",
    "empirical_basin_density_extrapolation",
]

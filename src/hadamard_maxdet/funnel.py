"""Roll-listen-filter-refine funnel search for ±1 maxdet matrices.

This module implements a two-phase search architecture, distinct from the
single-mode SA paths in :mod:`search` / :mod:`incremental` / :mod:`gram_energy`.

Phase A — Roll, listen, filter, refine
--------------------------------------
1. Roll: draw a uniform random ±1 matrix at order ``n``.
2. Listen: compute its Gram off-diagonal fingerprint cheaply (O(n²)) — the
   set of distinct ⟨v_i, v_j⟩ values, the size of that set ("support"), and
   the sum (signed total).
3. Filter: only "quiet" candidates pass — those whose Gram off-diagonal
   support size is ≤ ``max_support_size`` AND whose off-diagonal sum is
   biased negative (sum < 0). Random ±1 matrices typically fail one or both.
4. Refine: run a short rank-1-incremental SA on each survivor and record the
   resulting |det| ("basin tag") and the Gram fingerprint of the refined
   matrix.

The basin set (distinct refined fingerprints) and the running max |det|
build up over time. When the rate of new-basin discovery decays below a
threshold, Phase A → Phase B switch fires.

Phase B — Gram-conserving pair flips on the running best
--------------------------------------------------------
The new move family. For row pair ``(i, j)`` and column ``k``:

    M[i][k] → -M[i][k]   AND   M[j][k] → -M[j][k]

This *exactly preserves* ⟨v_i, v_j⟩, because the column-k contribution is
``M[i][k]·M[j][k] = (-M[i][k])·(-M[j][k])``. The move spends Gram budget
*not* on the (i, j) pair (which we picked precisely because it is at the
top of the inner-product ladder, e.g. value +3 or +5) but on the *other*
2(n-2) pairs that involve i or j and column k. So we redistribute
Gram-budget away from already-bad pairs.

Energy is the standard ``-log|det|``; the move is applied as two sequential
single-entry flips (``apply_entry_flip`` × 2) so the rank-1 fast path still
carries the determinant. Mixed with 50% standard ``entry_flip`` for
diversity. Standard Metropolis acceptance.

Public API
----------
* :func:`gram_loudness_filter` — O(n²) accept/reject of a candidate.
* :func:`gram_conserving_pair_flip` — apply the new move; returns new matrix.
* :func:`funnel_search` — top-level driver with auto Phase A → Phase B switch.
* :class:`FunnelDiagnostics` — record of phase timings, basin counts,
  fingerprint history, switch trigger.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field

from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
    sa_incremental,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


# ----------------------------------------------------------------------------
# Phase A: filter
# ----------------------------------------------------------------------------


def _gram_offdiag_stats(M: list[list[int]]) -> tuple[set[int], int, int]:
    """Return (support, total_sum, support_size) for the off-diagonal Gram.

    O(n²) work; computes each ⟨v_i, v_j⟩ for i<j once. The support is the
    set of *distinct* inner-product values; the total_sum is Σ_{i<j} ⟨v_i, v_j⟩.
    """
    n = len(M)
    support: set[int] = set()
    total = 0
    for i in range(n):
        Mi = M[i]
        for j in range(i + 1, n):
            Mj = M[j]
            s = 0
            for k in range(n):
                s += Mi[k] * Mj[k]
            support.add(s)
            total += s
    return support, total, len(support)


def gram_loudness_filter(
    M: list[list[int]] | Matrix,
    *,
    max_support_size: int = 8,
    require_negative_bias: bool = True,
) -> bool:
    """Return True iff ``M`` is a "quiet" candidate worth refining.

    Quiet means: the Gram off-diagonal distribution has narrow support
    (≤ ``max_support_size`` distinct values) AND, if
    ``require_negative_bias`` is set, the off-diagonal sum is negative.

    For n = 23 a uniform random ±1 matrix typically has support ≈ 12
    values clustered around 0 with sum ≈ 0; the BTC seed has support
    {-5, -1, +3} with strongly negative sum.

    Cost: O(n²).
    """
    if isinstance(M, Matrix):
        entries = M.entries
    else:
        entries = M
    _support, total, support_size = _gram_offdiag_stats(entries)
    if support_size > max_support_size:
        return False
    if require_negative_bias and total >= 0:
        return False
    return True


# ----------------------------------------------------------------------------
# Phase B: Gram-conserving pair flip
# ----------------------------------------------------------------------------


def _full_gram_offdiag(M: list[list[int]]) -> dict[tuple[int, int], int]:
    """Cache of inner products for all pairs (i, j) with i < j."""
    n = len(M)
    out: dict[tuple[int, int], int] = {}
    for i in range(n):
        for j in range(i + 1, n):
            s = 0
            Mi = M[i]
            Mj = M[j]
            for k in range(n):
                s += Mi[k] * Mj[k]
            out[(i, j)] = s
    return out


def _worst_pair(M: list[list[int]]) -> tuple[int, int]:
    """Return the (i, j) with i<j whose ⟨v_i, v_j⟩ is largest (most positive)."""
    n = len(M)
    best_val = None
    best_pair = (0, 1)
    for i in range(n):
        Mi = M[i]
        for j in range(i + 1, n):
            Mj = M[j]
            s = 0
            for k in range(n):
                s += Mi[k] * Mj[k]
            if best_val is None or s > best_val:
                best_val = s
                best_pair = (i, j)
    return best_pair


def gram_conserving_pair_flip(
    M: list[list[int]] | Matrix,
    rng: random.Random,
    target_pair: tuple[int, int] | None = None,
) -> list[list[int]]:
    """Apply a Gram-conserving pair flip; return the new matrix.

    Picks a row pair (i, j) — either ``target_pair`` if given, or the pair
    with the most-positive current inner product — and a uniformly random
    column k, then flips both ``M[i][k]`` and ``M[j][k]``. Because the
    column-k contribution to ⟨v_i, v_j⟩ is the product M[i][k]·M[j][k],
    this preserves ⟨v_i, v_j⟩ exactly.

    Does not mutate the input.
    """
    if isinstance(M, Matrix):
        entries = M.entries
    else:
        entries = M
    n = len(entries)
    if n < 2:
        return [row[:] for row in entries]
    if target_pair is None:
        i, j = _worst_pair(entries)
    else:
        i, j = target_pair
        if i == j:
            raise ValueError(f"target_pair must have distinct indices; got ({i}, {j})")
        if i > j:
            i, j = j, i
    k = rng.randrange(n)
    new = [row[:] for row in entries]
    new[i][k] = -new[i][k]
    new[j][k] = -new[j][k]
    return new


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------


@dataclass
class FunnelDiagnostics:
    """Per-run record of the funnel search."""

    n: int
    seed: int | None
    total_budget_seconds: float

    # Phase A counters
    phase_a_seconds: float = 0.0
    candidates_tried: int = 0
    candidates_passed_filter: int = 0
    distinct_basins: int = 0  # distinct refined |det| values
    distinct_fingerprints: int = 0
    phase_a_best_det: int = 0
    # When the switch fired (None if Phase B never triggered).
    switch_fired_at_seconds: float | None = None
    switch_fired_at_candidates: int | None = None
    switch_reason: str = ""

    # Phase B counters
    phase_b_seconds: float = 0.0
    phase_b_iterations: int = 0
    phase_b_accepted: int = 0
    phase_b_pair_flip_proposed: int = 0
    phase_b_pair_flip_accepted: int = 0
    phase_b_entry_flip_proposed: int = 0
    phase_b_entry_flip_accepted: int = 0
    phase_b_best_det: int = 0
    phase_b_improved_over_a: bool = False

    # Final
    final_best_det: int = 0
    final_barba_ratio: float = 0.0

    # History (compact)
    fingerprint_history: list[str] = field(default_factory=list)  # tags seen, in order
    basin_det_histogram: dict[int, int] = field(default_factory=dict)  # |det| -> count

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "seed": self.seed,
            "total_budget_seconds": self.total_budget_seconds,
            "phase_a_seconds": self.phase_a_seconds,
            "candidates_tried": self.candidates_tried,
            "candidates_passed_filter": self.candidates_passed_filter,
            "distinct_basins": self.distinct_basins,
            "distinct_fingerprints": self.distinct_fingerprints,
            "phase_a_best_det": self.phase_a_best_det,
            "switch_fired_at_seconds": self.switch_fired_at_seconds,
            "switch_fired_at_candidates": self.switch_fired_at_candidates,
            "switch_reason": self.switch_reason,
            "phase_b_seconds": self.phase_b_seconds,
            "phase_b_iterations": self.phase_b_iterations,
            "phase_b_accepted": self.phase_b_accepted,
            "phase_b_pair_flip_proposed": self.phase_b_pair_flip_proposed,
            "phase_b_pair_flip_accepted": self.phase_b_pair_flip_accepted,
            "phase_b_entry_flip_proposed": self.phase_b_entry_flip_proposed,
            "phase_b_entry_flip_accepted": self.phase_b_entry_flip_accepted,
            "phase_b_best_det": self.phase_b_best_det,
            "phase_b_improved_over_a": self.phase_b_improved_over_a,
            "final_best_det": self.final_best_det,
            "final_barba_ratio": self.final_barba_ratio,
            # Truncate fingerprint history to first/last 50 if huge.
            "fingerprint_history_count": len(self.fingerprint_history),
            "fingerprint_history_sample": (
                self.fingerprint_history[:25]
                + self.fingerprint_history[-25:]
                if len(self.fingerprint_history) > 50
                else list(self.fingerprint_history)
            ),
            "basin_det_histogram": {
                str(k): v for k, v in self.basin_det_histogram.items()
            },
        }


def _fingerprint_tag(M: list[list[int]]) -> str:
    """Compact, hashable Gram-off-diagonal fingerprint string.

    Form: e.g. ``"-5:1|-1:200|+3:22"`` lists each value:multiplicity sorted.
    Two matrices with the same tag have the same multiset of off-diagonal
    inner products (equivalent under at least row/col negation orbit).
    """
    n = len(M)
    counts: Counter = Counter()
    for i in range(n):
        Mi = M[i]
        for j in range(i + 1, n):
            Mj = M[j]
            s = 0
            for k in range(n):
                s += Mi[k] * Mj[k]
            counts[s] += 1
    items = sorted(counts.items())
    return "|".join(f"{v:+d}:{c}" for v, c in items)


# ----------------------------------------------------------------------------
# Top-level driver
# ----------------------------------------------------------------------------


def _barba(n: int) -> float:
    """Barba's bound for n ≡ 3 (mod 4): sqrt(2n-1) * (n-1)^((n-1)/2)."""
    return math.sqrt(2 * n - 1) * (n - 1) ** ((n - 1) / 2)


def funnel_search(
    n: int,
    *,
    total_budget_seconds: float,
    seed: int | None = None,
    initial_best: Matrix | None = None,
    # Phase A controls
    max_support_size: int = 8,
    require_negative_bias: bool = True,
    refine_iterations: int = 50_000,
    refine_t_start: float = 1.0,
    refine_t_end: float = 0.01,
    phase_a_min_seconds: float = 60.0,
    phase_a_max_seconds: float | None = None,
    no_new_window: int = 50,
    # Phase B controls
    phase_b_pair_fraction: float = 0.5,
    phase_b_t_start: float = 0.5,
    phase_b_t_end: float = 0.005,
    phase_b_no_improve_iters: int = 200_000,
    progress_callback=None,
) -> tuple[Matrix, FunnelDiagnostics]:
    """Run the funnel search and return (best_matrix, diagnostics).

    Parameters
    ----------
    n
        Matrix order.
    total_budget_seconds
        Total wall-clock budget (Phase A + Phase B).
    seed
        PRNG seed for reproducibility.
    initial_best
        Optional starting matrix. If given, used as the Phase B seed and as
        a baseline for the running max during Phase A. Useful to seed from
        BTC.
    max_support_size, require_negative_bias
        Forwarded to :func:`gram_loudness_filter`.
    refine_iterations, refine_t_start, refine_t_end
        SA controls for Phase A refinement.
    phase_a_min_seconds
        Phase A is not allowed to switch before this elapsed time.
    phase_a_max_seconds
        Hard cap on Phase A time. If set and exceeded, force-switch to
        Phase B even if the no-new-window trigger has not fired (useful
        when the loudness filter is strict enough that few/no candidates
        pass and the trigger condition cannot be met). Default ``None``
        means Phase A may consume the entire budget.
    no_new_window
        If the last ``no_new_window`` filtered candidates produced no new
        basin (distinct refined |det|) and no new fingerprint, switch.
    phase_b_pair_fraction
        Fraction of Phase B proposals that are Gram-conserving pair flips.
        The remainder are standard entry flips.
    phase_b_t_start, phase_b_t_end
        Phase B SA temperature schedule.
    phase_b_no_improve_iters
        Stop Phase B early if no new best in this many iterations.
    progress_callback
        Optional ``f(phase: str, diag: FunnelDiagnostics, current_best_det: int)``
        invoked roughly every 30s during both phases.

    Returns
    -------
    (best_matrix, FunnelDiagnostics)
    """
    rng = random.Random(seed)
    barba = _barba(n)
    diag = FunnelDiagnostics(
        n=n, seed=seed, total_budget_seconds=total_budget_seconds
    )

    start = time.monotonic()

    # Running best.
    if initial_best is not None:
        if initial_best.n != n:
            raise ValueError(f"initial_best order {initial_best.n} ≠ n={n}")
        best_entries = [row[:] for row in initial_best.entries]
        best_det = abs(integer_determinant(best_entries))
    else:
        best_entries = None
        best_det = 0

    diag.phase_a_best_det = best_det

    # Phase A state.
    seen_basins: set[int] = set()
    seen_fps: set[str] = set()
    last_progress_log = start
    # Window of "since last new" — count of refined candidates that did not
    # add a new basin or fingerprint.
    no_new_streak = 0

    def _maybe_progress(phase: str, current_best: int) -> None:
        nonlocal last_progress_log
        now = time.monotonic()
        if progress_callback is not None and now - last_progress_log >= 30.0:
            progress_callback(phase, diag, current_best)
            last_progress_log = now

    # ------------------- Phase A -------------------
    phase_a_start = time.monotonic()
    while True:
        elapsed_total = time.monotonic() - start
        if elapsed_total >= total_budget_seconds:
            diag.switch_reason = "budget_exhausted_in_phase_a"
            break

        # Hard cap on Phase A wallclock — checked on every loop iteration
        # so it fires even when the loudness filter is rejecting everything.
        elapsed_a_check = time.monotonic() - phase_a_start
        if (
            phase_a_max_seconds is not None
            and elapsed_a_check >= phase_a_max_seconds
            and best_entries is not None
        ):
            diag.switch_fired_at_seconds = elapsed_a_check
            diag.switch_fired_at_candidates = diag.candidates_tried
            diag.switch_reason = (
                f"phase_a_max_seconds={phase_a_max_seconds:.0f} cap "
                f"reached (filtered={diag.candidates_passed_filter} "
                f"of {diag.candidates_tried} tried)"
            )
            break

        # Roll
        M = Matrix.random(n, rng)
        diag.candidates_tried += 1

        # Listen + filter
        if not gram_loudness_filter(
            M.entries,
            max_support_size=max_support_size,
            require_negative_bias=require_negative_bias,
        ):
            _maybe_progress("phase_a", best_det)
            continue
        diag.candidates_passed_filter += 1

        # Refine
        # Bound refine iterations by remaining budget (heuristic: refine
        # is fast at n=23, ~30k iters/s, so 50k iters ≈ 1.5s).
        remaining = total_budget_seconds - (time.monotonic() - start)
        if remaining <= 0:
            diag.switch_reason = "budget_exhausted_in_phase_a"
            break

        result = sa_incremental(
            n,
            iterations=refine_iterations,
            t_start=refine_t_start,
            t_end=refine_t_end,
            seed=rng.randint(0, 2**31 - 1),
            initial=M,
            refresh_every=0,
        )
        refined_det = result.abs_det
        refined_fp = _fingerprint_tag(result.matrix.entries)

        # Track basins / fingerprints.
        new_basin = refined_det not in seen_basins
        new_fp = refined_fp not in seen_fps
        if new_basin:
            seen_basins.add(refined_det)
        if new_fp:
            seen_fps.add(refined_fp)
            diag.fingerprint_history.append(refined_fp)
        diag.basin_det_histogram[refined_det] = (
            diag.basin_det_histogram.get(refined_det, 0) + 1
        )
        diag.distinct_basins = len(seen_basins)
        diag.distinct_fingerprints = len(seen_fps)

        if refined_det > best_det:
            best_det = refined_det
            best_entries = [row[:] for row in result.matrix.entries]
            diag.phase_a_best_det = best_det

        if new_basin or new_fp:
            no_new_streak = 0
        else:
            no_new_streak += 1

        _maybe_progress("phase_a", best_det)

        # Switch trigger
        elapsed_a = time.monotonic() - phase_a_start
        if (
            elapsed_a >= phase_a_min_seconds
            and no_new_streak >= no_new_window
            and best_entries is not None
        ):
            diag.switch_fired_at_seconds = elapsed_a
            diag.switch_fired_at_candidates = diag.candidates_tried
            diag.switch_reason = (
                f"converged: {no_new_window} consecutive refines "
                f"without new basin or fingerprint"
            )
            break

    diag.phase_a_seconds = time.monotonic() - phase_a_start
    if diag.phase_a_best_det == 0:
        diag.phase_a_best_det = best_det  # keep in sync if seeded

    # If we never found anything (should be impossible given seeded best or
    # at least one filtered candidate at n=23), fall back to a random matrix.
    if best_entries is None:
        M = Matrix.random(n, rng)
        best_entries = [row[:] for row in M.entries]
        best_det = abs(integer_determinant(best_entries))

    # ------------------- Phase B -------------------
    phase_b_start = time.monotonic()
    remaining = total_budget_seconds - (phase_b_start - start)
    if remaining <= 0:
        diag.final_best_det = best_det
        diag.final_barba_ratio = best_det / barba
        return Matrix(n=n, entries=best_entries), diag

    state = IncrementalState.from_matrix(best_entries)
    if state.det == 0:
        # extremely unlikely with a seed; refresh by perturbing
        state = IncrementalState.from_matrix(
            Matrix.random(n, rng).entries
        )
    current_e = -math.log(abs(state.det)) if state.det != 0 else math.inf
    best_e = current_e
    pb_best_entries = [row[:] for row in state.M]
    pb_best_det = abs(state.det)

    log_ratio = math.log(phase_b_t_end / phase_b_t_start) if phase_b_t_start > 0 else 0.0
    # Estimate iteration budget from remaining time × empirical rate (we
    # actually loop on wallclock + no-improve, not a fixed iter count).
    # Cap "expected_iters" for the temperature schedule at a generous value.
    expected_iters = max(100_000, int(remaining * 100_000))

    iters_since_improve = 0
    iter_count = 0

    while True:
        elapsed_total = time.monotonic() - start
        if elapsed_total >= total_budget_seconds:
            break
        if iters_since_improve >= phase_b_no_improve_iters:
            break

        # Temperature: linear interp in log space over (elapsed in phase B / remaining_phase_b_estimate)
        elapsed_b = time.monotonic() - phase_b_start
        frac = min(1.0, elapsed_b / max(remaining, 1e-9))
        t = phase_b_t_start * math.exp(log_ratio * frac)

        if rng.random() < phase_b_pair_fraction:
            # Gram-conserving pair flip on the worst pair.
            i, j = _worst_pair(state.M)
            k = rng.randrange(n)
            # Apply as two rank-1 entry flips.
            # Compute new det via two propose+apply steps.
            d1 = propose_entry_flip(state, i, k)
            if d1 == 0:
                # Skip — would zero out determinant after first half.
                diag.phase_b_pair_flip_proposed += 1
                iter_count += 1
                iters_since_improve += 1
                continue
            # Tentatively apply first flip
            apply_entry_flip(state, i, k)
            d2 = propose_entry_flip(state, j, k)
            if d2 == 0:
                # Roll back the first flip and skip.
                apply_entry_flip(state, i, k)
                diag.phase_b_pair_flip_proposed += 1
                iter_count += 1
                iters_since_improve += 1
                continue
            # Apply second
            apply_entry_flip(state, j, k)
            new_det = state.det
            proposal_e = -math.log(abs(new_det))
            delta = proposal_e - current_e
            diag.phase_b_pair_flip_proposed += 1
            if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
                # Accept
                current_e = proposal_e
                diag.phase_b_pair_flip_accepted += 1
                diag.phase_b_accepted += 1
                if current_e < best_e:
                    best_e = current_e
                    pb_best_entries = [row[:] for row in state.M]
                    pb_best_det = abs(new_det)
                    iters_since_improve = 0
                else:
                    iters_since_improve += 1
            else:
                # Reject: roll back both flips.
                apply_entry_flip(state, j, k)
                apply_entry_flip(state, i, k)
                iters_since_improve += 1
        else:
            # Standard entry flip via fast path.
            i = rng.randrange(n)
            k = rng.randrange(n)
            new_det = propose_entry_flip(state, i, k)
            diag.phase_b_entry_flip_proposed += 1
            if new_det == 0:
                iter_count += 1
                iters_since_improve += 1
                continue
            proposal_e = -math.log(abs(new_det))
            delta = proposal_e - current_e
            if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
                apply_entry_flip(state, i, k)
                current_e = proposal_e
                diag.phase_b_entry_flip_accepted += 1
                diag.phase_b_accepted += 1
                if current_e < best_e:
                    best_e = current_e
                    pb_best_entries = [row[:] for row in state.M]
                    pb_best_det = abs(new_det)
                    iters_since_improve = 0
                else:
                    iters_since_improve += 1
            else:
                iters_since_improve += 1

        iter_count += 1
        diag.phase_b_iterations = iter_count

        # Periodic refresh paranoia — every ~50k iters, reseat det/adj.
        if iter_count % 50_000 == 0:
            old_det = state.det
            refresh(state)
            if state.det != old_det and state.det != 0:
                current_e = -math.log(abs(state.det))

        # Progress
        _maybe_progress("phase_b", max(best_det, pb_best_det))

        # Suppress unused-var warning (expected_iters used in narrative).
        _ = expected_iters

    diag.phase_b_seconds = time.monotonic() - phase_b_start
    diag.phase_b_best_det = pb_best_det

    # Final winner: max of Phase A best and Phase B best.
    if pb_best_det > best_det:
        # Re-verify with exact Bareiss.
        verified = abs(integer_determinant(pb_best_entries))
        if verified == pb_best_det:
            best_det = pb_best_det
            best_entries = pb_best_entries
            diag.phase_b_improved_over_a = True
        else:
            # Fall back to the verified value.
            best_det = verified
            if verified > diag.phase_a_best_det:
                best_entries = pb_best_entries
                diag.phase_b_improved_over_a = True
            diag.phase_b_best_det = verified

    diag.final_best_det = best_det
    diag.final_barba_ratio = best_det / barba

    return Matrix(n=n, entries=best_entries), diag

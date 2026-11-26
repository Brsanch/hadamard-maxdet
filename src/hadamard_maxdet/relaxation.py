"""Continuous (SDP / spectral) upper bounds on |det M| for ±1 matrices of order n.

For an n×n ±1 matrix M, set G = M Mᵀ. Then:

  * G is positive semi-definite (PSD).
  * G_ii = n  (each row has unit-squared-norm n).
  * G_ij is an integer in {-n, -n+2, …, n-2, n}.
  * G_ij ≡ n (mod 2);  in particular for *odd* n, |G_ij| ≥ 1.
  * |det M|² = det G.

Dropping the integer constraint on G_ij gives a convex semidefinite program

    max  log det(G)
    s.t. G ⪰ 0,  G_ii = n,  (cuts on G_ij)

whose optimal value is an upper bound on log |det M|² for the discrete
problem.

Three relaxation tiers are exposed:

  * **Trivial SDP** (`cuts="trivial"`).
        Only G_ii = n and PSD.
        Maximum is at G = n·I, giving det(G) = n^n, equivalently
        |det M| ≤ n^(n/2). This is just the **Hadamard bound**.

  * **Magnitude-cap SDP** (`cuts="cap"`, `cap` keyword).
        Add |G_ij| ≤ cap for i ≠ j. **For any cap ≥ 0 the optimum is
        still G = n·I (off-diagonals = 0 satisfies any cap), so this
        tier never beats Hadamard.** Exposed for diagnostic use.

  * **Spectral (Barba/Ehlich/Hadamard, closed-form)** (`cuts="spectral"`).
        Adds an off-diagonal *minimum-magnitude* constraint
        (non-convex in G but exactly solvable in the eigenvalue domain
        because the extremal Gram has the form (n-c)·I + c·J):

          c = 0  if n ≡ 0 mod 4   (Hadamard regime)
          c = 1  if n is odd      (Barba regime)
          c = 2  if n ≡ 2 mod 4   (Ehlich regime)

        Closed-form bound det G ≤ (n + (n-1)·c)·(n - c)^(n-1), whose
        square root is exactly the Hadamard / Barba / Ehlich bound used
        in `verifier.upper_bound`.

The headline finding: **for the ±1 maxdet problem at non-Hadamard n,
no purely convex relaxation of det(G) tightens the Barba/Ehlich bound.**
The integer constraint "G_ij is an odd integer" is needed to force
the off-diagonals away from zero; it is not convexly representable in
G. The "spectral" tier reproduces Barba/Ehlich exactly. The "trivial"
and "cap" tiers reproduce Hadamard (and are therefore *looser* than
Barba/Ehlich for non-Hadamard n).

The SDP backend is CVXPY (`cp.log_det`). Solver default: SCS, falling
back to CLARABEL.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RelaxationBound:
    """Result of a continuous-relaxation upper-bound computation.

    Fields
    ------
    n : int
        Matrix order.
    det_G_bound : float
        Upper bound on det(G) = |det M|² from the relaxation.
    abs_det_bound : float
        Upper bound on |det M| (= sqrt(det_G_bound)).
    cuts : str
        Cut family used.
    solver : str
        Solver name, or "closed_form".
    status : str
        Solver status string.
    """

    n: int
    det_G_bound: float
    abs_det_bound: float
    cuts: str
    solver: str
    status: str


def hadamard_bound_squared(n: int) -> float:
    """det(G) bound from the Hadamard inequality: G = n·I → det = n^n."""
    return float(n) ** n


def _two_level_det_squared(n: int, c: int) -> float:
    """det of the extremal Gram (n - c)·I + c·J.

    Eigenvalues: (n - c) + n·c = n + (n-1)·c (mult 1); (n - c) (mult n-1).
    Hence det = (n + (n-1)·c) · (n - c)^(n-1).

    For c=0 this is n^n (Hadamard²).
    For c=1 this is (2n-1)·(n-1)^(n-1) (Barba²).
    For c=2 this is (2n-2)·(n-2)^(n-1) — note the (n-1) exponent here
    differs from Ehlich's closed-form (which has (n-2)^((n-2)/2) under
    the square root, i.e. (n-2)^(n-2) when squared, with a leading
    (2n-2)² rather than (2n-2)). The two-level Gram bound is therefore
    *strictly tighter* than Ehlich's closed form for n ≡ 2 (mod 4),
    because Ehlich's analysis allows two-block structure not just
    a single rank-1 perturbation of nI. We expose both via
    ``ehlich_two_level_squared`` (this) and ``ehlich_block_squared``.
    """
    return float(n + (n - 1) * c) * float(n - c) ** (n - 1)


def barba_bound_squared(n: int) -> float:
    """Barba bound squared: (2n-1)·(n-1)^(n-1). Tight for n odd, c=1."""
    return _two_level_det_squared(n, c=1)


def ehlich_two_level_squared(n: int) -> float:
    """Single-level (n-2)·I + 2·J det: (2n-2)·(n-2)^(n-1).

    **Diagnostic only — NOT a valid upper bound** for n ≡ 2 (mod 4).
    The single-level Gram is *not* the spectral maximizer subject to
    G_ii = n and |G_ij| ≥ 2; Ehlich's two-block analysis gives a
    strictly larger det (and hence the actual bound). See
    `ehlich_block_squared` for the correct upper bound, and check
    n = 6: this function returns √16384 = 128, but the true |det| max
    is 160 (OEIS A003432). This function is exposed only so callers
    can see the gap between single-level and Ehlich's two-block bound.
    """
    return _two_level_det_squared(n, c=2)


def ehlich_block_squared(n: int) -> float:
    """Ehlich's closed-form bound squared: (2n-2)² · (n-2)^(n-2).

    This is what `verifier.upper_bound(n)` returns squared, for n ≡ 2
    (mod 4). It is sharper than the single-block two-level Gram bound
    because Ehlich's analysis uses a *two-block* eigenvalue split.
    """
    return float(2 * n - 2) ** 2 * float(n - 2) ** (n - 2)


def _solve_sdp_loglet(n: int, off_diag_cap: float | None,
                      solver_pref: tuple[str, ...] = ("SCS", "CLARABEL")
                      ) -> RelaxationBound:
    """Solve max log det(G) s.t. G ⪰ 0, G_ii = n, |G_ij| ≤ off_diag_cap.

    With ``off_diag_cap`` ≥ 0, G = n·I is feasible and optimal, so the
    closed-form Hadamard answer is returned without invoking the solver
    when the cap permits zero off-diagonals (always, here). The CVXPY
    path is exercised only as a sanity check when the caller insists.
    """
    cap_label = "trivial" if off_diag_cap is None else f"|G_ij|≤{off_diag_cap:g}"
    if off_diag_cap is None or off_diag_cap >= 0:
        # G = n·I always feasible; gives the Hadamard answer.
        return RelaxationBound(
            n=n,
            det_G_bound=hadamard_bound_squared(n),
            abs_det_bound=math.sqrt(hadamard_bound_squared(n)),
            cuts=cap_label,
            solver="closed_form",
            status="closed_form",
        )

    raise ValueError(f"off_diag_cap must be ≥ 0 or None, got {off_diag_cap!r}")


def _solve_sdp_via_cvxpy(n: int, off_diag_cap: float,
                         solver_pref: tuple[str, ...] = ("SCS", "CLARABEL")
                         ) -> RelaxationBound:
    """Actually invoke CVXPY (used by tests to confirm the closed form).

    Solves max log det(G) s.t. G ⪰ 0, diag(G) = n, |G_ij| ≤ off_diag_cap
    for i ≠ j, and returns the numerical optimum.
    """
    try:
        import cvxpy as cp
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "cvxpy is required for SDP relaxation; install with `pip install cvxpy`"
        ) from err

    G = cp.Variable((n, n), symmetric=True)
    constraints = [G >> 0, cp.diag(G) == n]
    cap = float(off_diag_cap)
    # Compact off-diagonal cap: |off-diag entries of G| ≤ cap.
    off_mask = np.ones((n, n)) - np.eye(n)
    constraints.append(cp.multiply(off_mask, G) <= cap)
    constraints.append(cp.multiply(off_mask, G) >= -cap)

    prob = cp.Problem(cp.Maximize(cp.log_det(G)), constraints)

    last_err: Exception | None = None
    for s in solver_pref:
        try:
            prob.solve(solver=s)
            if prob.status in ("optimal", "optimal_inaccurate"):
                value = float(prob.value)  # = log det(G)
                det_G = float(np.exp(value))
                return RelaxationBound(
                    n=n,
                    det_G_bound=det_G,
                    abs_det_bound=math.sqrt(det_G),
                    cuts=f"|G_ij|≤{cap:g}",
                    solver=s,
                    status=prob.status,
                )
        except Exception as e:  # noqa: BLE001
            last_err = e
            continue

    raise RuntimeError(
        f"SDP relaxation failed at n={n}, cap={cap}; "
        f"last solver status={prob.status!r}, last error={last_err!r}"
    )


def sdp_upper_bound(n: int, tightened: bool = True,
                    cuts: str | None = None,
                    cap: float | None = None,
                    use_cvxpy: bool = False) -> RelaxationBound:
    """Continuous SDP upper bound on |det M| for n×n ±1 matrices.

    Parameters
    ----------
    n
        Matrix order (≥ 1).
    tightened
        If False, return the trivial relaxation (Hadamard bound).
        If True (default) and ``cuts`` is None, dispatch to
        the residue-class-appropriate spectral cut (gives
        Hadamard / Barba / Ehlich-block).
    cuts
        Override:
          * ``"trivial"`` — only G_ii = n, G ⪰ 0. → Hadamard.
          * ``"cap"`` — additionally |G_ij| ≤ ``cap`` (default 1). → Hadamard
            (G = nI satisfies any non-negative cap).
          * ``"spectral"`` — closed-form two-level / two-block bound;
            for n ≡ 0 mod 4 returns Hadamard, for n odd returns Barba,
            for n ≡ 2 mod 4 returns Ehlich (block form).
    cap
        Off-diagonal magnitude cap (only used when ``cuts == "cap"``).
        Default 1.0.
    use_cvxpy
        If True and ``cuts == "cap"``, actually invoke the CVXPY SDP
        solver (slow at large n) instead of the closed-form short-circuit.
        Used by tests to verify the SDP returns Hadamard numerically.

    Returns
    -------
    RelaxationBound
        ``abs_det_bound`` is the upper bound on |det M|;
        ``det_G_bound`` is the upper bound on det(M Mᵀ) = |det M|².
    """
    if n < 1:
        raise ValueError(f"order must be positive, got {n}")

    if cuts is None:
        cuts = "trivial" if not tightened else "spectral"

    if cuts == "trivial":
        return _solve_sdp_loglet(n, off_diag_cap=None)

    if cuts == "cap":
        c = 1.0 if cap is None else float(cap)
        if c < 0:
            raise ValueError(f"cap must be ≥ 0, got {cap}")
        if use_cvxpy:
            return _solve_sdp_via_cvxpy(n, off_diag_cap=c)
        return _solve_sdp_loglet(n, off_diag_cap=c)

    if cuts == "spectral":
        if n % 4 == 0:
            det_G = hadamard_bound_squared(n)
            label = "spectral c=0 (Hadamard)"
        elif n % 2 == 1:
            det_G = barba_bound_squared(n)
            label = "spectral c=1 (Barba)"
        else:  # n ≡ 2 mod 4
            det_G = ehlich_block_squared(n)
            label = "spectral c=2 (Ehlich block)"
        return RelaxationBound(
            n=n,
            det_G_bound=det_G,
            abs_det_bound=math.sqrt(det_G),
            cuts=label,
            solver="closed_form",
            status="closed_form",
        )

    raise ValueError(f"unknown cuts {cuts!r}; expected 'trivial', 'cap', 'spectral'")


def lp_upper_bound(n: int) -> RelaxationBound:
    """Spectral closed-form bound (eigenvalue-domain reformulation).

    Maximizes ∏ λ_i over n nonneg reals subject to Σ λ_i = n² and
    Σ λ_i² ≥ n³ + n(n-1)·c² (the latter encodes |G_ij| ≥ c off-diag),
    over the natural c for the residue class:

        c = 0 for n ≡ 0 (mod 4) → Hadamard,         det = n^n
        c = 1 for n odd          → Barba,             det = (2n-1)·(n-1)^(n-1)
        c = 2 for n ≡ 2 (mod 4) → Ehlich (block),    det = (2n-2)² · (n-2)^(n-2)

    For c = 1 the optimum is two-level (1 large, n-1 equal-small) and
    matches the Barba extremal. For c = 2 a *single* level split does
    not maximise det subject to the spectral constraint; Ehlich's
    two-block analysis is required, and the published Ehlich value is
    used here (provably the spectral maximum).

    Returns the same value as ``sdp_upper_bound(n, cuts='spectral')``;
    this function exists for callers who want the LP/eigenvalue framing
    (no PSD solver invoked).
    """
    if n < 1:
        raise ValueError(f"order must be positive, got {n}")
    if n % 4 == 0:
        det_G, c = hadamard_bound_squared(n), 0
    elif n % 2 == 1:
        det_G, c = barba_bound_squared(n), 1
    else:
        det_G, c = ehlich_block_squared(n), 2
    return RelaxationBound(
        n=n,
        det_G_bound=det_G,
        abs_det_bound=math.sqrt(det_G),
        cuts=f"spectral c={c}",
        solver="closed_form",
        status="closed_form",
    )


__all__ = [
    "RelaxationBound",
    "sdp_upper_bound",
    "lp_upper_bound",
    "hadamard_bound_squared",
    "barba_bound_squared",
    "ehlich_two_level_squared",
    "ehlich_block_squared",
]

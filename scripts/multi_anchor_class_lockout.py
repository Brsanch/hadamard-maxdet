"""Multi-anchor SA with cross-chain shared class lockout.

Combines existing repo modules:

  parallel_tempering.py  — replica architecture (we don't use the full PT
                           machinery, but borrow the multi-replica idea)
  class_tabu.py          — class signature (Hadamard-equivalence-invariant
                           Gram off-diag count distribution) and
                           sa_with_class_tabu
  tabu.py                — within-chain Gram-fingerprint tabu
  williamson.py          — Williamson H_{4m} constructions
  kms_construction.py    — KMS Prop 28 H23 construction
  bibd_construction.py   — Paley biplane (23, 11, 5)
  conference_method.py   — conference matrix at related orders
  spectral_anchor.py     — high-multiplicity-SV anchors
  matrix.py              — paley_type_i, paley_type_ii, sylvester
  basin_atlas.py         — known H23 anchors

Architecture
------------

K parallel chains, each anchored at a DIFFERENT construction. All chains
SHARE a global locked_classes set: when any chain reaches a local maximum,
its class signature is added to the global lockout, and all OTHER chains
must thereafter reject moves into that class.

This differs from the existing class_tabu single-chain iterative-basin-
discovery: there, one chain runs to completion before the next starts.
Here, K chains run concurrently, each feeding discoveries to the others
in real time.

Key claim of effectiveness:
- Each construction-anchored chain has a DIFFERENT initial basin to refine.
- Shared lockout means each chain doesn't waste cycles rediscovering basins
  another chain already mapped.
- The combined effect is broader basin coverage per unit CPU than running
  iterative_basin_discovery from a single anchor or from random.

Targeted at H23 (the smallest open case), but parameterized to work at any
n where the repo has multiple anchors available.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np

from hadamard_maxdet.class_tabu import class_signature
from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.tabu import fingerprint_offdiag
from hadamard_maxdet.verifier import integer_determinant


@dataclass
class AnchorChain:
    """One construction-anchored search chain."""

    name: str
    initial: list[list[int]]
    state: IncrementalState = field(init=False)
    best_matrix: list[list[int]] = field(init=False)
    best_abs_det: int = field(init=False)
    history: list[tuple[int, int]] = field(default_factory=list)
    rejected_by_class_lockout: int = 0
    rejected_by_metropolis: int = 0
    accepted: int = 0

    def __post_init__(self) -> None:
        self.state = IncrementalState.from_matrix([row[:] for row in self.initial])
        if self.state.det == 0:
            raise ValueError(f"anchor {self.name} singular — choose nonsingular initial")
        self.best_matrix = [row[:] for row in self.state.M]
        self.best_abs_det = abs(self.state.det)


@dataclass
class MultiAnchorResult:
    best_matrix: list[list[int]]
    best_abs_det: int
    best_chain_name: str
    chains: list[AnchorChain]
    locked_classes_count: int
    walltime_seconds: float
    total_steps: int
    epoch_count: int


def _validate_anchor(M, n: int, name: str) -> list[list[int]] | None:
    """Sanity-check an anchor: must be n×n with entries in {-1, +1} and nonsingular.
    Coerce to list of lists of int. Returns None if invalid."""
    if M is None:
        return None
    if hasattr(M, "entries"):
        M = M.entries
    elif hasattr(M, "tolist"):
        M = M.tolist()
    M = [list(int(v) for v in r) for r in M]
    if len(M) != n or any(len(r) != n for r in M):
        return None
    for r in M:
        for v in r:
            if v not in (-1, 1):
                return None
    # Also require nonsingular (otherwise AnchorChain init fails).
    from hadamard_maxdet.verifier import integer_determinant
    if integer_determinant(M) == 0:
        return None
    return M


def get_h23_anchors(rng: random.Random) -> list[tuple[str, list[list[int]]]]:
    """Return list of (name, initial_matrix) for H23 anchors. Validates each.

    Each anchor is sourced from existing repo modules. Defensive imports +
    validation: anchors that don't produce 23×23 ±1 nonsingular matrices
    are silently skipped.
    """
    anchors: list[tuple[str, list[list[int]]]] = []
    n = 23

    # Random
    rand = _validate_anchor(Matrix.random(n, rng), n, "random")
    if rand is not None:
        anchors.append(("random_1", rand))
    rand2 = _validate_anchor(Matrix.random(n, rng), n, "random_2")
    if rand2 is not None:
        anchors.append(("random_2", rand2))

    # Random circulant
    rc = _validate_anchor(Matrix.random_circulant(n, rng), n, "random_circulant")
    if rc is not None:
        anchors.append(("random_circulant", rc))

    # Paley biplane (23, 11, 5)
    try:
        from hadamard_maxdet.bibd_construction import paley_biplane_23
        try:
            M = _validate_anchor(paley_biplane_23(), n, "paley_biplane_23")
            if M is not None:
                anchors.append(("paley_biplane_23", M))
        except Exception:
            pass
    except ImportError:
        pass

    # Williamson H23 (gs_array): williamson_quadruples_m5 returns FIRST ROWS
    # of length-5 symmetric circulants. We must wrap them via circulant_matrix
    # before passing to gs_array_williamson_h23.
    try:
        from hadamard_maxdet.williamson import (
            gs_array_williamson_h23,
            williamson_quadruples_m5,
            circulant_matrix,
        )
        quads = williamson_quadruples_m5()
        # Try multiple quadruples × multiple e patterns until we find nonsingular Williamson H23.
        best_williamson = None
        best_williamson_det = 0
        from hadamard_maxdet.verifier import integer_determinant as _det
        for q_idx in range(min(20, len(quads))):
            a_row, b_row, c_row, d_row = quads[q_idx]
            A = circulant_matrix(a_row)
            B = circulant_matrix(b_row)
            C = circulant_matrix(c_row)
            D = circulant_matrix(d_row)
            for e in ([1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
                      [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]):
                try:
                    Mres = gs_array_williamson_h23(A, B, C, D, e=e, variant="border3")
                    val = _validate_anchor(Mres, n, "williamson")
                    if val is not None:
                        d = abs(_det(val))
                        if d > best_williamson_det:
                            best_williamson_det = d
                            best_williamson = (f"williamson_q{q_idx}_e{tuple(e)}", val)
                except Exception:
                    continue
        if best_williamson is not None:
            anchors.append(best_williamson)
    except ImportError:
        pass

    # KMS Prop 28: returns the (4q-1) order Hadamard matrix; for q=23 this
    # is order 95 not 23. Skip; not an H23 anchor as constructed.

    # Try Orrick 2003 record matrix if results file exists
    try:
        import json
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        orrick_path = repo / "results" / "H23_orrick2003_record.json"
        if orrick_path.exists():
            data = json.loads(orrick_path.read_text())
            for key in ("matrix", "M", "entries"):
                if key in data:
                    M = _validate_anchor(data[key], n, "orrick_2003")
                    if M is not None:
                        anchors.append(("orrick_2003", M))
                        break
    except Exception:
        pass

    # Bordered Two-Circulant (BTC) anchor — in-repo cycle 24 result.
    # |det| = 2,626,567,700,217,856 = 0.945·Orrick. Distinct basin from
    # Orrick's 0.7092 of Barba — sits in the bordered-two-circulant family
    # which is structurally separate from the symmetric Hadamard family.
    try:
        import json
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        for fn in ["H23_attack_best.json", "H23_funnel_best.json"]:
            p = repo / "results" / fn
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            for key in ("entries", "matrix", "best_matrix"):
                if key in data:
                    M = _validate_anchor(data[key], n, fn)
                    if M is not None:
                        anchors.append(("btc_cycle24", M))
                        break
            if any(name == "btc_cycle24" for name, _ in anchors):
                break
    except Exception:
        pass

    # BTC variant in 0.8-1.4 anchor band (in-repo result, ratio 0.914·Orrick)
    try:
        import json
        from pathlib import Path
        p = Path(__file__).resolve().parents[1] / "results" / "H23_anchor_band_08_14_btc.json"
        if p.exists():
            data = json.loads(p.read_text())
            for key in ("entries", "matrix", "best_matrix"):
                if key in data:
                    M = _validate_anchor(data[key], n, "btc_band")
                    if M is not None:
                        anchors.append(("btc_band_anchor", M))
                        break
    except Exception:
        pass

    # High-quality 2-value abs-Gram anchors from smart_2val_search results.
    # All three high-quality H23 basins (Orrick, BTC, smart_2val_best) sit on
    # the abs-Gram manifold {1, 3}; cycle-27's "cleaner Gram = deeper basin"
    # predictor was confirmed by basin diagnostics 2026-05-05.
    smart_anchors = [
        ("smart_2val_best", "H23_smart_2val_best_new_class.json"),       # 0.9750
        ("unexplored_2val_a203", "H23_unexplored_2val_a203_r0.9539.json"),  # 0.9539
        ("anchor_band_04_08", "H23_anchor_band_04_08_smart2val.json"),    # 0.9493
    ]
    for anchor_name, fn in smart_anchors:
        try:
            import json
            from pathlib import Path
            p = Path(__file__).resolve().parents[1] / "results" / fn
            if not p.exists():
                continue
            data = json.loads(p.read_text())
            for key in ("entries", "matrix", "best_matrix"):
                if key in data:
                    M = _validate_anchor(data[key], n, anchor_name)
                    if M is not None:
                        anchors.append((anchor_name, M))
                        break
        except Exception:
            pass

    return anchors


def sa_step_with_lockout(
    chain: AnchorChain,
    locked_classes: set[tuple],
    *,
    t: float,
    rng: random.Random,
) -> bool:
    """Single SA step on `chain` with global class-lockout enforcement.

    Returns True if a flip was accepted, False otherwise.
    """
    n = chain.state.n
    i = rng.randrange(n)
    j = rng.randrange(n)
    new_det = propose_entry_flip(chain.state, i, j)
    if new_det == 0:
        return False
    cur_e = -math.log(abs(chain.state.det))
    prop_e = -math.log(abs(new_det))
    delta = prop_e - cur_e
    if delta > 0 and rng.random() >= math.exp(-delta / max(t, 1e-12)):
        chain.rejected_by_metropolis += 1
        return False
    # Provisional apply for class-lockout check
    old_val = chain.state.M[i][j]
    apply_entry_flip(chain.state, i, j)
    sig = class_signature(chain.state.M)
    if sig in locked_classes:
        chain.state.M[i][j] = old_val
        refresh(chain.state)
        chain.rejected_by_class_lockout += 1
        return False
    chain.accepted += 1
    if abs(chain.state.det) > chain.best_abs_det:
        chain.best_abs_det = abs(chain.state.det)
        chain.best_matrix = [row[:] for row in chain.state.M]
        chain.history.append((chain.accepted, abs(chain.state.det)))
    return True


def lock_chain_basin(chain: AnchorChain, locked_classes: set[tuple]) -> bool:
    """Add the current state's class signature to locked_classes.

    Returns True if a new class was locked, False if already locked.
    """
    sig = class_signature(chain.state.M)
    if sig in locked_classes:
        return False
    locked_classes.add(sig)
    return True


def crossover_between_chains(
    chains: list[AnchorChain],
    rng: random.Random,
    *,
    n_children: int = 8,
) -> tuple[str, int] | None:
    """Pick two highest-|det| chains with different class signatures and
    cross them. If best child has higher |det| than the lower-ranked
    parent, replace that chain's state with the child.

    Returns (replaced_chain_name, child_abs_det) on a successful improving
    crossover; None otherwise.
    """
    from hadamard_maxdet.crossover import crossover_children
    from hadamard_maxdet.class_tabu import class_signature

    if len(chains) < 2:
        return None
    ranked = sorted(chains, key=lambda c: -c.best_abs_det)
    # Pair top-1 with each lower chain having a DIFFERENT class signature.
    a = ranked[0]
    sig_a = class_signature(a.best_matrix)
    best_partner = None
    for b in ranked[1:]:
        if class_signature(b.best_matrix) != sig_a:
            best_partner = b
            break
    if best_partner is None:
        return None
    children = crossover_children(
        a.best_matrix, best_partner.best_matrix,
        n_children=n_children, seed=rng.randrange(1 << 30),
    )
    # Score children
    from hadamard_maxdet.verifier import integer_determinant
    best_child = None
    best_child_det = 0
    for child in children:
        try:
            d = abs(integer_determinant(child))
        except Exception:
            continue
        if d > best_child_det:
            best_child_det = d
            best_child = child
    if best_child is None:
        return None
    # Replace WORST chain with the new child if the child beats the worst.
    worst = ranked[-1]
    if best_child_det > worst.best_abs_det:
        try:
            new_state = IncrementalState.from_matrix(best_child)
            if new_state.det != 0:
                worst.state = new_state
                if best_child_det > worst.best_abs_det:
                    worst.best_abs_det = best_child_det
                    worst.best_matrix = [row[:] for row in best_child]
                return (worst.name, best_child_det)
        except Exception:
            pass
    return None


def polish_chain_best(chain: AnchorChain) -> int:
    """2-flip-polish a chain's best matrix. Returns the new |det|.
    Updates chain.best_matrix and chain.best_abs_det if polish improves."""
    from hadamard_maxdet.k_flip import steepest_2_flip_polish
    polished, det, _ = steepest_2_flip_polish([row[:] for row in chain.best_matrix], max_passes=200)
    if det > chain.best_abs_det:
        chain.best_abs_det = det
        chain.best_matrix = polished
    return chain.best_abs_det


def run(
    n: int = 23,
    *,
    epochs: int = 50,
    steps_per_epoch: int = 5_000,
    t_start: float = 1.5,
    t_end: float = 0.005,
    seed: int = 42,
    extra_random_chains: int = 0,
    lock_after_each_epoch: bool = True,
    crossover_every: int = 10,
    polish_every: int = 25,
    verbose: bool = True,
) -> MultiAnchorResult:
    """Run multi-anchor SA chains with shared class-lockout.

    Each epoch:
      1. Each chain runs `steps_per_epoch` SA steps at the epoch's temperature.
      2. Optionally lock each chain's current basin (class signature) into
         the shared `locked_classes`. All chains see this lockout in the
         next epoch.

    Temperature schedule: geometric from t_start (epoch 0) to t_end
    (final epoch).
    """
    rng = random.Random(seed)
    if n != 23:
        raise NotImplementedError(
            "anchors are H23-specific for now; extend get_h23_anchors() for other n"
        )

    raw_anchors = get_h23_anchors(rng)
    for _ in range(extra_random_chains):
        rand = Matrix.random(n, rng)
        raw_anchors.append((f"random_extra_{_}", rand.entries))

    chains: list[AnchorChain] = []
    for name, init in raw_anchors:
        try:
            chains.append(AnchorChain(name=name, initial=init))
        except ValueError as e:
            if verbose:
                print(f"  skipping {name}: {e}")
    if not chains:
        raise RuntimeError("no usable anchors found")

    if verbose:
        print(f"Initialized {len(chains)} chains:")
        for c in chains:
            print(f"  {c.name:<25} initial |det|={c.best_abs_det:.4e}")

    locked_classes: set[tuple] = set()
    log_ratio = math.log(t_end / t_start) if t_start > 0 else 0.0

    t_walltime = time.monotonic()
    total_steps = 0

    for epoch in range(epochs):
        frac = epoch / max(1, epochs - 1)
        t = t_start * math.exp(log_ratio * frac)
        for chain in chains:
            for _ in range(steps_per_epoch):
                sa_step_with_lockout(chain, locked_classes, t=t, rng=rng)
                total_steps += 1
        if lock_after_each_epoch:
            new_locks = 0
            for chain in chains:
                if lock_chain_basin(chain, locked_classes):
                    new_locks += 1
            xover_msg = ""
            if crossover_every > 0 and (epoch + 1) % crossover_every == 0:
                xover_result = crossover_between_chains(chains, rng)
                if xover_result is not None:
                    xover_msg = f"  XOVER: replaced {xover_result[0]} with |det|={xover_result[1]:.4e}"
            polish_msg = ""
            if polish_every > 0 and (epoch + 1) % polish_every == 0:
                gains = []
                for chain in chains:
                    pre = chain.best_abs_det
                    post = polish_chain_best(chain)
                    if post > pre:
                        gains.append((chain.name, pre, post))
                if gains:
                    polish_msg = f"  POLISH: {len(gains)} chain(s) improved (best gain: {max(g[2] for g in gains):.4e})"
            if verbose:
                global_best = max(c.best_abs_det for c in chains)
                print(
                    f"  epoch {epoch:3d}: t={t:.4f}  global_best=|det|={global_best:.4e}  "
                    f"locked_total={len(locked_classes)} (+{new_locks})  "
                    f"steps={total_steps:_}{xover_msg}{polish_msg}"
                )

    # Cycle-41-style 2-flip polish on each chain's best matrix.
    # Polish past the 1-local-max via exhaustive rank-2 adjugate ascent.
    if verbose:
        print(f"\n  Applying 2-flip polish to each chain's best...")
    from hadamard_maxdet.k_flip import steepest_2_flip_polish
    for c in chains:
        try:
            polished_entries, polished_det, n_passes = steepest_2_flip_polish(
                [row[:] for row in c.best_matrix], max_passes=200
            )
            if polished_det > c.best_abs_det:
                if verbose:
                    print(f"    {c.name:<35} polish: |det| {c.best_abs_det:.4e} → {polished_det:.4e} ({n_passes} passes)")
                c.best_abs_det = polished_det
                c.best_matrix = polished_entries
            elif verbose:
                pass  # no improvement; quiet
        except Exception as e:
            if verbose:
                print(f"    {c.name}: polish failed: {e}")

    best_chain = max(chains, key=lambda c: c.best_abs_det)
    # Compute per-chain class signatures + Gram fingerprints for diagnostics.
    if verbose:
        print(f"\n  Computing per-chain diagnostics...")
        from hadamard_maxdet.class_tabu import class_signature
        for c in chains:
            try:
                sig = class_signature(c.best_matrix)
                # Compact summary
                support = sorted(set(v for v, _ in sig))
                counts_str = " ".join(f"{v}:{cnt}" for v, cnt in sig)
                print(f"    {c.name:<35} |det|={c.best_abs_det:.4e}  Gram support={support}  ({counts_str})")
            except Exception as e:
                print(f"    {c.name}: signature failed: {e}")

    return MultiAnchorResult(
        best_matrix=best_chain.best_matrix,
        best_abs_det=best_chain.best_abs_det,
        best_chain_name=best_chain.name,
        chains=chains,
        locked_classes_count=len(locked_classes),
        walltime_seconds=time.monotonic() - t_walltime,
        total_steps=total_steps,
        epoch_count=epochs,
    )


def main() -> None:
    print("Multi-anchor SA with shared class-lockout — H23")
    res = run(n=23, epochs=20, steps_per_epoch=10_000, seed=2026)
    print(f"\nResult:")
    print(f"  best |det| = {res.best_abs_det}")
    from hadamard_maxdet.parallel_tempering import ORRICK_ABS_DET_H23
    print(f"  Orrick H23 = {ORRICK_ABS_DET_H23}")
    print(f"  ratio      = {res.best_abs_det / ORRICK_ABS_DET_H23:.4f}")
    print(f"  best chain = {res.best_chain_name}")
    print(f"  locked classes total = {res.locked_classes_count}")
    print(f"  walltime   = {res.walltime_seconds:.1f}s")
    print(f"\nPer-chain best:")
    for c in res.chains:
        print(f"  {c.name:<25} |det|={c.best_abs_det:.4e}  "
              f"acc={c.accepted}  rej_metro={c.rejected_by_metropolis}  "
              f"rej_class={c.rejected_by_class_lockout}")


if __name__ == "__main__":
    main()

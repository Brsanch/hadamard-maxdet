"""H29 multi-anchor with cross-chain class-lockout — same architecture as
multi_anchor_class_lockout.py but parameterized for n=29 anchors.

H29 anchor roster (drawing from existing repo results):
  random / random_circulant
  h29_outlier        — best in-repo, ratio ~0.61 of EW(29) bound
  h29_paley32        — Paley H32 → deletion to H29
  paley_ii_h28_border — our cycle-51 Paley II H28 (q=13) bordered to H29
  sylvester_h32_del  — Sylvester H32 → 3-row+col deletion to H29
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from hadamard_maxdet.class_tabu import class_signature
from hadamard_maxdet.incremental import (
    IncrementalState,
    apply_entry_flip,
    propose_entry_flip,
    refresh,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant, upper_bound

# Reuse infrastructure from the H23 script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from multi_anchor_class_lockout import (
    AnchorChain,
    MultiAnchorResult,
    sa_step_with_lockout,
    lock_chain_basin,
    crossover_between_chains,
    polish_chain_best,
    _validate_anchor,
)


def _border_to_n29(M_28: list[list[int]]) -> list[list[int]] | None:
    """Border a 28x28 ±1 matrix to 29x29 by adding a row/col of +1's
    with corner = +1, then SA-optimise the border for |det|.
    """
    if len(M_28) != 28:
        return None
    M_29 = [[1] * 29 for _ in range(29)]
    for i in range(28):
        for j in range(28):
            M_29[i + 1][j + 1] = M_28[i][j]
    # Greedy 1-flip border optimization
    best = [row[:] for row in M_29]
    best_d = abs(integer_determinant(best))
    border = [(0, j) for j in range(29)] + [(i, 0) for i in range(1, 29)]
    improved = True
    while improved:
        improved = False
        for i, j in border:
            best[i][j] = -best[i][j]
            d = abs(integer_determinant(best))
            if d > best_d:
                best_d = d
                improved = True
            else:
                best[i][j] = -best[i][j]
    return best


def get_h29_anchors(rng: random.Random) -> list[tuple[str, list[list[int]]]]:
    n = 29
    anchors: list[tuple[str, list[list[int]]]] = []
    repo = Path(__file__).resolve().parents[1]

    # Random + circulant
    rand1 = _validate_anchor(Matrix.random(n, rng), n, "random_1")
    if rand1:
        anchors.append(("random_1", rand1))
    rand2 = _validate_anchor(Matrix.random(n, rng), n, "random_2")
    if rand2:
        anchors.append(("random_2", rand2))
    rc = _validate_anchor(Matrix.random_circulant(n, rng), n, "random_circulant")
    if rc:
        anchors.append(("random_circulant", rc))

    # h29_outlier (best in-repo)
    try:
        p = repo / "results" / "h29_outlier_0_70.npy"
        if p.exists():
            M = np.load(p).astype(np.int64).tolist()
            v = _validate_anchor(M, n, "h29_outlier")
            if v:
                anchors.append(("h29_outlier", v))
    except Exception:
        pass

    # h29_paley32_climbed
    try:
        p = repo / "results" / "h29_paley32_climbed.npy"
        if p.exists():
            M = np.load(p).astype(np.int64).tolist()
            v = _validate_anchor(M, n, "h29_paley32")
            if v:
                anchors.append(("h29_paley32", v))
    except Exception:
        pass

    # Paley II H28 (q=13) → bordered to H29
    try:
        H28 = Matrix.paley_type_ii(28).entries
        Bordered = _border_to_n29(H28)
        v = _validate_anchor(Bordered, n, "paley_ii_h28_border")
        if v:
            anchors.append(("paley_ii_h28_border", v))
    except Exception:
        pass

    # Sylvester H32 → 3-row+col deletion to H29
    try:
        H32 = Matrix.sylvester(32).entries
        # Greedy 3-stage deletion (delete row+col with minimal det loss each stage)
        M = [row[:] for row in H32]
        for _ in range(3):
            best_sub = None
            best_d = -1
            for i in range(len(M)):
                for j in range(len(M)):
                    sub = [[M[r][c] for c in range(len(M)) if c != j]
                           for r in range(len(M)) if r != i]
                    d = abs(integer_determinant(sub))
                    if d > best_d:
                        best_d = d
                        best_sub = sub
            M = best_sub
        v = _validate_anchor(M, n, "sylvester_h32_del")
        if v:
            anchors.append(("sylvester_h32_del", v))
    except Exception:
        pass

    return anchors


def run(
    *,
    epochs: int = 200,
    steps_per_epoch: int = 50_000,
    t_start: float = 2.0,
    t_end: float = 0.005,
    seed: int = 42,
    extra_random_chains: int = 3,
    crossover_every: int = 10,
    polish_every: int = 20,
    verbose: bool = True,
) -> MultiAnchorResult:
    rng = random.Random(seed)
    n = 29
    raw_anchors = get_h29_anchors(rng)
    for k in range(extra_random_chains):
        rand = Matrix.random(n, rng)
        v = _validate_anchor(rand, n, f"random_extra_{k}")
        if v:
            raw_anchors.append((f"random_extra_{k}", v))

    chains: list[AnchorChain] = []
    for name, init in raw_anchors:
        try:
            chains.append(AnchorChain(name=name, initial=init))
        except ValueError as e:
            if verbose:
                print(f"  skipping {name}: {e}")
    if not chains:
        raise RuntimeError("no usable H29 anchors")

    if verbose:
        print(f"Initialized {len(chains)} H29 chains:")
        ub = upper_bound(n)
        for c in chains:
            print(f"  {c.name:<30} initial |det|={c.best_abs_det:.4e}  ratio_to_UB={c.best_abs_det/ub:.4f}")

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
        new_locks = 0
        for chain in chains:
            if lock_chain_basin(chain, locked_classes):
                new_locks += 1
        xover_msg = ""
        if crossover_every > 0 and (epoch + 1) % crossover_every == 0:
            xr = crossover_between_chains(chains, rng)
            if xr is not None:
                xover_msg = f"  XOVER: replaced {xr[0]} with |det|={xr[1]:.4e}"
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
            ub = upper_bound(n)
            print(
                f"  epoch {epoch:3d}: t={t:.4f}  global_best={global_best:.4e}  "
                f"ratio_UB={global_best/ub:.4f}  locked={len(locked_classes)} (+{new_locks})"
                f"{xover_msg}{polish_msg}"
            )

    # Final polish
    if verbose:
        print("\n  Final polish...")
    from hadamard_maxdet.k_flip import steepest_2_flip_polish
    for c in chains:
        try:
            entries, det, _ = steepest_2_flip_polish([row[:] for row in c.best_matrix], max_passes=200)
            if det > c.best_abs_det:
                c.best_abs_det = det
                c.best_matrix = entries
        except Exception:
            pass

    if verbose:
        print(f"\n  Per-chain finals:")
        ub = upper_bound(n)
        for c in chains:
            try:
                sig = class_signature(c.best_matrix)
                support = sorted(set(v for v, _ in sig))
                print(f"    {c.name:<30} ratio_UB={c.best_abs_det/ub:.4f}  Gram support={support}")
            except Exception:
                pass

    best_chain = max(chains, key=lambda c: c.best_abs_det)
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
    print("Multi-anchor H29 with shared class-lockout")
    res = run(epochs=150, steps_per_epoch=30_000, seed=2026)
    ub = upper_bound(29)
    print(f"\nResult: best |det|={res.best_abs_det}  ratio_to_UB={res.best_abs_det/ub:.4f}  chain={res.best_chain_name}")
    import json, numpy as np
    np.save("/tmp/h29_run_best.npy", np.array(res.best_matrix, dtype=np.int8))
    out = {
        "best_abs_det": res.best_abs_det,
        "ub": ub,
        "ratio_to_ub": res.best_abs_det / ub,
        "best_chain": res.best_chain_name,
        "classes_locked": res.locked_classes_count,
        "walltime_s": res.walltime_seconds,
        "matrix": res.best_matrix,
        "per_chain": [(c.name, c.best_abs_det) for c in res.chains],
    }
    open("/tmp/h29_run.json", "w").write(json.dumps(out, indent=2))
    print(f"\nSaved /tmp/h29_run.json")


if __name__ == "__main__":
    main()

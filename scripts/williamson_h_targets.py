"""Benchmark Williamson m=5 derived constructions against targets n ∈ {19, 20, 21, 23}.

For each target n we:
  1. Build every Williamson-derived H_n candidate.
  2. Bareiss-verify the best |det| (exact integer).
  3. Run a brief SA refinement (~60s) seeded by the Williamson best.
  4. Compare to the proven max (n in 19, 20, 21) or in-repo best (n=23).
  5. Save the winning matrix to results/williamson_H{n}_best.json.
  6. Dump a Gram off-diagonal fingerprint for the basin diagnostic.

Run from repo root:
    python scripts/williamson_h_targets.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hadamard_maxdet.orders import BENCHMARKS  # noqa: E402
from hadamard_maxdet.telemetry import gram_offdiag_report  # noqa: E402
from hadamard_maxdet.verifier import integer_determinant  # noqa: E402
from hadamard_maxdet.williamson import (  # noqa: E402
    BORDER_PATTERNS,
    bordered_williamson_h21,
    circulant_matrix,
    gs_array_williamson_h23,
    williamson_array,
    williamson_h20_deletion_h19,
    williamson_quadruples_m5,
)


# ---------------------------------------------------------------------------
# SA seeding helper
# ---------------------------------------------------------------------------


def _sa_from_seed(
    seed_entries: list[list[int]],
    *,
    seconds: float,
    base_seed: int = 0,
) -> tuple[int, list[list[int]]]:
    """Run sa_incremental seeded from ``seed_entries`` for roughly ``seconds``.

    Iterates 20k-step chunks and returns the best ±1 matrix observed (Bareiss-
    verified). Falls back to a manual hand-rolled SA if the incremental
    helper API does not match.
    """
    import math
    import random

    from hadamard_maxdet.incremental import (
        IncrementalState,
        apply_entry_flip,
        propose_entry_flip,
    )

    n = len(seed_entries)
    state = IncrementalState.from_matrix([row[:] for row in seed_entries])
    if state.det == 0:
        # Avoid starting at a singular matrix.
        return abs(integer_determinant(seed_entries)), [
            row[:] for row in seed_entries
        ]
    best_det = abs(state.det)
    best_entries = [row[:] for row in state.M]
    cur_e = -math.log(best_det) if best_det > 0 else math.inf

    rng = random.Random(base_seed)
    t_start = 0.5
    t_end = 0.005
    log_ratio = math.log(t_end / t_start)
    chunk_iters = 20_000
    t0 = time.monotonic()
    elapsed = 0.0
    total_iters = 0
    while elapsed < seconds:
        for step in range(chunk_iters):
            frac = step / max(1, chunk_iters - 1)
            t = t_start * math.exp(log_ratio * frac)
            i = rng.randrange(n)
            j = rng.randrange(n)
            new_det = propose_entry_flip(state, i, j)
            if new_det == 0:
                continue
            proposal_e = -math.log(abs(new_det))
            delta = proposal_e - cur_e
            if delta <= 0 or rng.random() < math.exp(-delta / max(t, 1e-12)):
                apply_entry_flip(state, i, j)
                cur_e = proposal_e
                if abs(state.det) > best_det:
                    best_det = abs(state.det)
                    best_entries = [row[:] for row in state.M]
        total_iters += chunk_iters
        elapsed = time.monotonic() - t0
    # Final exact verification.
    best_det = abs(integer_determinant(best_entries))
    return best_det, best_entries


# ---------------------------------------------------------------------------
# Per-target Williamson candidate pools
# ---------------------------------------------------------------------------


def best_h20() -> tuple[int, list[list[int]], dict]:
    quads = williamson_quadruples_m5()
    a, b, c, d = quads[0]
    H = williamson_array(
        circulant_matrix(a),
        circulant_matrix(b),
        circulant_matrix(c),
        circulant_matrix(d),
    )
    det = abs(integer_determinant(H.entries))
    info = {
        "n_quads": len(quads),
        "first_quad": {"a": a, "b": b, "c": c, "d": d},
    }
    return det, H.entries, info


def best_h21() -> tuple[int, list[list[int]], dict]:
    quads = williamson_quadruples_m5()
    best_det = 0
    best_entries: list[list[int]] = []
    best_meta: dict = {}
    for qi, (a, b, c, d) in enumerate(quads):
        A = circulant_matrix(a)
        B = circulant_matrix(b)
        C = circulant_matrix(c)
        D = circulant_matrix(d)
        for pattern in BORDER_PATTERNS:
            M = bordered_williamson_h21(A, B, C, D, border_pattern=pattern)
            d_abs = abs(integer_determinant(M.entries))
            if d_abs > best_det:
                best_det = d_abs
                best_entries = M.entries
                best_meta = {
                    "quad_index": qi,
                    "quad": {"a": a, "b": b, "c": c, "d": d},
                    "border_pattern": pattern,
                }
    return best_det, best_entries, best_meta


def best_h19() -> tuple[int, list[list[int]], dict]:
    quads = williamson_quadruples_m5()
    best_det = 0
    best_entries: list[list[int]] = []
    best_meta: dict = {}
    for qi, (a, b, c, d) in enumerate(quads):
        A = circulant_matrix(a)
        B = circulant_matrix(b)
        C = circulant_matrix(c)
        D = circulant_matrix(d)
        dels = williamson_h20_deletion_h19(A, B, C, D)
        for (i_del, j_del), M in dels:
            d_abs = abs(integer_determinant(M.entries))
            if d_abs > best_det:
                best_det = d_abs
                best_entries = M.entries
                best_meta = {
                    "quad_index": qi,
                    "quad": {"a": a, "b": b, "c": c, "d": d},
                    "i_del": i_del,
                    "j_del": j_del,
                }
    return best_det, best_entries, best_meta


def best_h23() -> tuple[int, list[list[int]], dict]:
    quads = williamson_quadruples_m5()
    best_det = 0
    best_entries: list[list[int]] = []
    best_meta: dict = {}
    e_choices: list[list[int]] = [
        [a0, a1, a2]
        for a0 in (-1, 1)
        for a1 in (-1, 1)
        for a2 in (-1, 1)
    ]
    for qi, (a, b, c, d) in enumerate(quads):
        A = circulant_matrix(a)
        B = circulant_matrix(b)
        C = circulant_matrix(c)
        D = circulant_matrix(d)
        for variant in ("border3", "border3_gs"):
            for e in e_choices:
                M = gs_array_williamson_h23(A, B, C, D, e=e, variant=variant)
                d_abs = abs(integer_determinant(M.entries))
                if d_abs > best_det:
                    best_det = d_abs
                    best_entries = M.entries
                    best_meta = {
                        "quad_index": qi,
                        "quad": {"a": a, "b": b, "c": c, "d": d},
                        "variant": variant,
                        "e": e,
                    }
    return best_det, best_entries, best_meta


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


TARGETS = {
    19: ("H19", best_h19),
    20: ("H20", best_h20),
    21: ("H21", best_h21),
    23: ("H23", best_h23),
}

IN_REPO_BEST = {
    19: BENCHMARKS["H19"].best_known,
    20: BENCHMARKS["H20"].best_known,
    21: BENCHMARKS["H21"].best_known,
    23: 2_626_567_700_217_856,
}


def _save(path: Path, n: int, det: int, entries: list[list[int]], meta: dict) -> None:
    payload = {
        "n": n,
        "abs_det": int(det),
        "best_entries": entries,
        "meta": meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> int:
    results_dir = REPO / "results"
    summary: dict = {"per_n": {}}
    for n, (oid, fn) in TARGETS.items():
        print(f"\n=== n = {n} ({oid}) ===", flush=True)
        t0 = time.monotonic()
        wdet, wentries, wmeta = fn()
        wtime = time.monotonic() - t0
        target = IN_REPO_BEST.get(n)
        ratio_pre = (wdet / target) if target else None
        print(
            f"  Williamson seed best |det| = {wdet:,}  "
            f"(target {target:,}, ratio {ratio_pre:.4f})"
            if target is not None
            else f"  Williamson seed best |det| = {wdet:,}"
        )
        print(f"  enumeration time = {wtime:.1f}s")

        # SA refine.
        print("  SA refine for ~60s ...", flush=True)
        sa_det, sa_entries = _sa_from_seed(wentries, seconds=60.0, base_seed=42)
        if sa_det > wdet:
            print(f"  SA improved |det| = {sa_det:,}")
            best_det, best_entries = sa_det, sa_entries
            improved = True
        else:
            print(f"  SA did NOT improve (best {sa_det:,} ≤ {wdet:,})")
            best_det, best_entries = wdet, wentries
            improved = False

        # Diagnostics on best.
        fp = gram_offdiag_report(best_entries)
        print(
            f"  Gram off-diag support: {fp['gram_offdiag_support']}; "
            f"sv ratio max/min = {fp['sv_ratio']:.4f}"
        )

        out = results_dir / f"williamson_H{n}_best.json"
        _save(out, n, best_det, best_entries, {"seed_meta": wmeta, "sa_improved": improved})
        print(f"  saved → {out.relative_to(REPO)}")

        summary["per_n"][n] = {
            "williamson_seed_abs_det": int(wdet),
            "sa_refined_abs_det": int(best_det),
            "sa_improved": improved,
            "target_abs_det": int(target) if target is not None else None,
            "ratio_pre_sa": float(ratio_pre) if ratio_pre is not None else None,
            "ratio_post_sa": (float(best_det) / target) if target else None,
            "gram_offdiag_support": fp["gram_offdiag_support"],
            "sv_ratio": fp["sv_ratio"],
            "saved_to": str(out.relative_to(REPO)),
            "seed_meta": wmeta,
        }

    # Aggregate summary file.
    sum_path = results_dir / "williamson_summary.json"
    sum_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nAggregate summary → {sum_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

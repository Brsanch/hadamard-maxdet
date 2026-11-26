"""Audit which "high-prediction" (a, b, partition) targets actually have
parity-feasible row-weight assignments.

The generalized satellite view (`generalized_ehlich.py`) ranked targets by
predicted ``|det| = sqrt(det(G_block))`` over (intra, cross, partition)
triples, finding 261 H23 triples above Orrick's 0.7091·Barba record. The
top single triple — (a=+1, b=-1) at (1, 1, 1, 20), predicted 0.994·Barba —
is shown by `feasibility.parity_constraint` to be **infeasible**: no ±1
matrix can have that planted Gram because of a row-weight parity
obstruction.

This script audits every triple in the satellite-view's table:
* Counts how many of the 261 H23 above-Orrick triples survive parity.
* Checks the headline matches at H19 / H21 (the (+1,-1) regime hits the
  proven max within ~5 % at both): are those targets feasible?
* For surviving H23 targets, reports the top by predicted ratio (the
  candidates worth a construction attempt).

Output: ``results/feasibility_audit.json``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path when running directly.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hadamard_maxdet.feasibility import (
    enumerate_feasible_targets,
    parity_constraint,
)
from hadamard_maxdet.generalized_ehlich import (
    DEFAULT_AB_VALUES,
    enumerate_ab_grid,
    predicted_block_det,
)
from hadamard_maxdet.verifier import upper_bound


def audit_at_n(n: int, floor: float) -> dict:
    """Audit all (a, b, partition) triples at ``n`` with predicted ratio > floor."""
    grid = enumerate_ab_grid(n, ab_values=DEFAULT_AB_VALUES, min_block_size=1)
    above_floor = [r for r in grid if r.predicted_ratio > floor]
    feasible_rows = []
    infeasible_rows = []
    for r in above_floor:
        pc = parity_constraint(n, r.partition, r.intra, r.cross)
        rec = {
            "n": r.n,
            "intra": r.intra,
            "cross": r.cross,
            "partition": list(r.partition),
            "predicted_abs_det": r.predicted_abs_det,
            "predicted_ratio": r.predicted_ratio,
            "parity_feasible": pc.feasible,
            "n_assignments": pc.n_assignments,
            "intra_required": pc.intra_required,
            "cross_required": pc.cross_required,
            "block_parity_pattern": (
                list(pc.block_parity_pattern)
                if pc.block_parity_pattern is not None
                else None
            ),
            "reason": pc.reason,
        }
        if pc.feasible:
            feasible_rows.append(rec)
        else:
            infeasible_rows.append(rec)
    return {
        "n": n,
        "floor": floor,
        "n_above_floor": len(above_floor),
        "n_feasible": len(feasible_rows),
        "n_infeasible": len(infeasible_rows),
        "feasible": sorted(feasible_rows, key=lambda d: -d["predicted_ratio"]),
        "infeasible_top10": sorted(infeasible_rows, key=lambda d: -d["predicted_ratio"])[:10],
    }


def named_target_check(n: int, partition, intra: int, cross: int, label: str) -> dict:
    """Check a specific named target by hand."""
    pc = parity_constraint(n, partition, intra, cross)
    pred = predicted_block_det(n, tuple(partition), intra, cross)
    base = float(upper_bound(n))
    ratio = pred / base if base > 0 else 0.0
    return {
        "label": label,
        "n": n,
        "partition": list(partition),
        "intra": intra,
        "cross": cross,
        "predicted_abs_det": pred,
        "predicted_ratio": ratio,
        "parity_feasible": pc.feasible,
        "n_assignments": pc.n_assignments,
        "intra_required": pc.intra_required,
        "cross_required": pc.cross_required,
        "reason": pc.reason,
    }


def main() -> None:
    out_path = ROOT / "results" / "feasibility_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Named targets singled out in the satellite-view writeup.
    named = [
        named_target_check(23, (1, 1, 1, 20), 1, -1, "H23 headline (1,1,1,20)/(+1,-1)"),
        named_target_check(23, (1, 1, 2, 19), 1, -1, "H23 second (1,1,2,19)/(+1,-1)"),
        named_target_check(23, (1, 2, 2, 18), 1, -1, "H23 third (1,2,2,18)/(+1,-1)"),
        named_target_check(23, (2, 2, 2, 17), 1, -1, "H23 (2,2,2,17)/(+1,-1)"),
        named_target_check(23, (5, 6, 6, 6), 3, -1, "H23 canonical Ehlich (5,6,6,6)/(+3,-1)"),
        named_target_check(19, (1, 1, 1, 16), 1, -1, "H19 alleged-match (1,1,1,16)/(+1,-1)"),
        named_target_check(21, (1, 3, 6, 11), 1, -1, "H21 alleged-match (1,3,6,11)/(+1,-1)"),
        named_target_check(21, (2, 2, 5, 12), 1, -1, "H21 alleged-match (2,2,5,12)/(+1,-1)"),
        named_target_check(21, (1, 3, 7, 10), 1, -1, "H21 alleged-match (1,3,7,10)/(+1,-1)"),
        named_target_check(8, (2, 2, 2, 2), 0, 0, "H8 Hadamard sanity (2,2,2,2)/(0,0)"),
    ]

    # 2) Per-n audits at the relevant floor.
    audits = {}
    audits["n=19_above_proven_0.9413"] = audit_at_n(19, 0.9413 - 1e-6)
    audits["n=21_above_proven_0.9058"] = audit_at_n(21, 0.9058 - 1e-6)
    audits["n=23_above_orrick_0.7091"] = audit_at_n(23, 0.7091)

    # 3) Best surviving feasible H23 target above Orrick (the construction
    #    candidate, if any).
    h23_audit = audits["n=23_above_orrick_0.7091"]
    surviving = h23_audit["feasible"]
    survivor_summary = {
        "count": len(surviving),
        "top_5": surviving[:5],
    }

    summary = {
        "named_targets": named,
        "audits": audits,
        "h23_survivors_above_orrick": survivor_summary,
        "verdict": _verdict(named, audits, surviving),
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_path}")
    _print_human_summary(summary)


def _verdict(named, audits, surviving) -> dict:
    headline = next(
        x for x in named if x["label"].startswith("H23 headline")
    )
    h19_alleged = next(
        x for x in named if x["label"].startswith("H19 alleged-match")
    )
    h21_alleged = [x for x in named if x["label"].startswith("H21 alleged-match")]
    return {
        "headline_h23_111_20_plus1_minus1": {
            "feasible": headline["parity_feasible"],
            "predicted_ratio": headline["predicted_ratio"],
        },
        "h19_alleged_match_feasible": h19_alleged["parity_feasible"],
        "h21_alleged_matches_feasible": [
            {"label": x["label"], "feasible": x["parity_feasible"]}
            for x in h21_alleged
        ],
        "h23_above_orrick_feasibility_rate": (
            audits["n=23_above_orrick_0.7091"]["n_feasible"]
            / max(1, audits["n=23_above_orrick_0.7091"]["n_above_floor"])
        ),
        "h23_survivors_count": len(surviving),
        "h23_top_survivor": surviving[0] if surviving else None,
    }


def _print_human_summary(summary: dict) -> None:
    print("\n=== feasibility audit — human summary ===")
    for nt in summary["named_targets"]:
        flag = "FEAS" if nt["parity_feasible"] else "INFEAS"
        print(
            f"  [{flag}] {nt['label']:55s}  ratio={nt['predicted_ratio']:.4f}"
            f"  n_assign={nt['n_assignments']}"
        )
    print()
    for key, a in summary["audits"].items():
        print(
            f"  {key}: {a['n_above_floor']} above-floor, "
            f"{a['n_feasible']} feasible, {a['n_infeasible']} infeasible"
        )
    print()
    sv = summary["h23_survivors_above_orrick"]
    print(f"H23 above-Orrick survivors: {sv['count']}")
    for r in sv["top_5"]:
        print(
            f"    a={r['intra']:+d} b={r['cross']:+d} part={tuple(r['partition'])} "
            f"ratio={r['predicted_ratio']:.4f} pred|det|={r['predicted_abs_det']:.3e}"
        )


if __name__ == "__main__":
    main()

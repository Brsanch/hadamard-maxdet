"""Audit every results JSON for |det| consistency under the fixed verifier.

For every JSON file under ``results/`` we walk the dict tree, find every
embedded square ±1 matrix, and recompute its determinant exactly with
``integer_determinant``. We compare the recomputed |det| to whatever |det|
field the matrix's parent dict claims (``abs_det``, ``det``, ``final_abs_det``,
``overall_best_abs_det``, ``btc_seed_det``).

Output:

* a per-file report on stdout, and
* a structured ``results/det_audit_report.json`` with one record per matrix
  containing ``(file, path, n, claimed, recomputed, match, source_field,
  shape_valid, pm1_valid)``.

This audit was written 2026-04-25 in response to the numpy int64 overflow
bug in ``integer_determinant``; see ``docs/findings.md`` "verifier numpy
int64 overflow".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Iterable

# Make the repo importable when invoked directly.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

from hadamard_maxdet.verifier import integer_determinant  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")
REPORT_PATH = os.path.join(RESULTS_DIR, "det_audit_report.json")

DET_KEYS = (
    "abs_det",
    "det",
    "final_abs_det",
    "overall_best_abs_det",
    "btc_seed_det",
    "best_abs_det",
    "best_feasible_abs_det",
)


def _is_pm1_matrix(obj: Any) -> bool:
    """Return True iff obj is a square nested list of ints in {-1, +1}."""
    if not isinstance(obj, list) or len(obj) < 1 or len(obj) > 200:
        return False
    n = len(obj)
    for row in obj:
        if not isinstance(row, list) or len(row) != n:
            return False
        for v in row:
            if not isinstance(v, int) or v not in (-1, 1):
                return False
    return True


def _is_int_matrix(obj: Any) -> bool:
    """Return True iff obj is a square nested list of ints (any value)."""
    if not isinstance(obj, list) or len(obj) < 1 or len(obj) > 200:
        return False
    n = len(obj)
    for row in obj:
        if not isinstance(row, list) or len(row) != n:
            return False
        for v in row:
            if not isinstance(v, int):
                return False
    return True


def _claimed_det_from_parent(parent: dict | None, key: str | None) -> int | None:
    """Look up a |det| claim associated with this matrix in its parent dict."""
    if parent is None:
        return None
    # Direct sibling fields.
    for k in DET_KEYS:
        if k in parent and isinstance(parent[k], int):
            return parent[k]
    return None


def find_matrices(
    obj: Any,
    path: str = "",
    parent: dict | None = None,
    parent_key: str | None = None,
    depth: int = 0,
) -> Iterable[tuple[str, list[list[int]], dict | None, str | None]]:
    """Yield (path, matrix, parent_dict, parent_key) for every embedded
    square ±1 matrix found in ``obj``."""
    if depth > 12:
        return
    if _is_pm1_matrix(obj):
        yield (path, obj, parent, parent_key)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if _is_pm1_matrix(v):
                yield (f"{path}.{k}" if path else k, v, obj, k)
            else:
                yield from find_matrices(
                    v, f"{path}.{k}" if path else k, obj, k, depth + 1
                )
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from find_matrices(
                v, f"{path}[{i}]", parent, parent_key, depth + 1
            )


def audit_file(path: str) -> list[dict]:
    """Return a list of audit records (one per matrix) for one JSON file."""
    rel = os.path.relpath(path, _ROOT)
    try:
        with open(path) as fh:
            data = json.load(fh)
    except Exception as exc:
        return [
            {
                "file": rel,
                "path": "",
                "error": f"failed to load: {exc}",
            }
        ]

    records: list[dict] = []
    for mpath, matrix, parent, parent_key in find_matrices(data):
        n = len(matrix)
        claimed = _claimed_det_from_parent(parent, parent_key)

        # Find the actual claimed-key name used (for traceability).
        source_field = None
        if parent is not None:
            for k in DET_KEYS:
                if k in parent and isinstance(parent[k], int):
                    source_field = k
                    break

        try:
            d = integer_determinant(matrix)
            recomputed = abs(d)
            err = None
        except Exception as exc:  # pragma: no cover
            recomputed = None
            err = f"{type(exc).__name__}: {exc}"

        match = (claimed is not None and recomputed is not None and claimed == recomputed)

        records.append(
            {
                "file": rel,
                "path": mpath,
                "n": n,
                "claimed": claimed,
                "recomputed": recomputed,
                "match": match if claimed is not None else None,
                "source_field": source_field,
                "error": err,
            }
        )
    return records


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help=f"directory of JSON files to audit (default: {RESULTS_DIR})",
    )
    p.add_argument(
        "--report",
        default=REPORT_PATH,
        help=f"path to write the JSON audit report (default: {REPORT_PATH})",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-file output (still writes the JSON report)",
    )
    args = p.parse_args()

    files = sorted(
        f for f in os.listdir(args.results_dir) if f.endswith(".json")
    )

    all_records: list[dict] = []
    n_files_with_matrices = 0
    n_matrices = 0
    n_with_claim = 0
    n_match = 0
    mismatches: list[dict] = []
    no_claim: list[dict] = []

    for f in files:
        if f == os.path.basename(args.report):
            continue  # skip the report itself
        recs = audit_file(os.path.join(args.results_dir, f))
        if not recs:
            continue
        # Skip files that loaded but had no matrices.
        any_matrix = any(r.get("n") for r in recs)
        if not any_matrix and not any(r.get("error") for r in recs):
            continue
        n_files_with_matrices += 1
        for r in recs:
            if r.get("n"):
                n_matrices += 1
                if r.get("claimed") is not None:
                    n_with_claim += 1
                    if r.get("match"):
                        n_match += 1
                    else:
                        mismatches.append(r)
                else:
                    no_claim.append(r)
            all_records.append(r)

        if not args.quiet:
            interesting = [r for r in recs if r.get("claimed") is not None]
            for r in interesting:
                tag = "OK   " if r["match"] else "MISMATCH"
                print(
                    f"{tag} {r['file']:50s} {r['path']:30s} "
                    f"n={r['n']:3d}  claimed={r['claimed']}  recomputed={r['recomputed']}"
                )

    summary = {
        "files_audited": len(files),
        "files_with_matrices": n_files_with_matrices,
        "matrices_total": n_matrices,
        "matrices_with_claim": n_with_claim,
        "matches": n_match,
        "mismatches": len(mismatches),
        "no_claim": len(no_claim),
    }
    report = {"summary": summary, "records": all_records, "mismatches": mismatches}

    with open(args.report, "w") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)

    print()
    print("=== AUDIT SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if mismatches:
        print()
        print("Mismatches:")
        for r in mismatches:
            print(
                f"  {r['file']} :: {r['path']}  "
                f"claimed={r['claimed']} recomputed={r['recomputed']}"
            )
    print()
    print(f"Wrote {args.report}")

    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())

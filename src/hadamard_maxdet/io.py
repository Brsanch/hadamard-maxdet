"""JSON serialization for Hadamard maximal determinant submissions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


def matrix_to_submission(matrix: Matrix, order_id: str) -> dict[str, Any]:
    """Convert a :class:`Matrix` to the submission schema.

    Returns a dict with order_id, n, entries, and the exact integer |det|
    (for cross-verification).
    """
    det = integer_determinant(matrix.entries)
    return {
        "order_id": order_id,
        "n": matrix.n,
        "entries": [list(row) for row in matrix.entries],
        "det": det,
        "abs_det": abs(det),
    }


def submission_from_dict(data: Mapping[str, Any]) -> tuple[str, int, list[list[int]]]:
    """Parse a submission dict. Returns (order_id, n, entries)."""
    for key in ("order_id", "n", "entries"):
        if key not in data:
            raise ValueError(f"submission missing required key {key!r}")
    n = int(data["n"])
    entries = [[int(v) for v in row] for row in data["entries"]]
    if len(entries) != n:
        raise ValueError(
            f"declared n={n} but entries has {len(entries)} rows"
        )
    return str(data["order_id"]), n, entries


def write_submission(matrix: Matrix, order_id: str, path: str | Path) -> None:
    """Write a submission JSON file to ``path``."""
    payload = matrix_to_submission(matrix, order_id)
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def read_submission(path: str | Path) -> tuple[str, int, list[list[int]]]:
    """Read a submission JSON file and return (order_id, n, entries)."""
    data = json.loads(Path(path).read_text())
    return submission_from_dict(data)

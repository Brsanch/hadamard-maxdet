"""Telemetry / diagnostics for Hadamard maxdet search runs.

The point of this module is to surface signals that vanilla SA throws
away — acceptance by move type, entry-flip frequency maps, Gram
off-diagonal distributions of visited states, per-temperature acceptance
in PT — so that after a run we can *diagnose* why the search got stuck
and which move set / ansatz relaxation would help.

Design:

``RunTelemetry`` is a light dataclass that a search loop updates inline.
It records counters and histograms, not full trajectories (keeps memory
bounded). At the end of a run, :func:`summarise` produces a human-readable
report and a JSON dump suitable for cross-run comparison.

Nothing here is in the hot loop's O(n^2) per step — the telemetry cost
is O(1) per step for counters plus an optional periodic O(n^2) Gram
snapshot gated by a ``snapshot_every`` counter.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunTelemetry:
    """Counters + histograms recorded during a single search run."""

    # Move-kind accounting (entry_flip, row_negate, two_row_xor, ...).
    move_proposed: dict[str, int] = field(default_factory=lambda: Counter())
    move_accepted: dict[str, int] = field(default_factory=lambda: Counter())
    # Per-position flip counts (only the "volatile" positions tell us where
    # the frontier is; stable positions are the structural backbone).
    flip_counts: dict[tuple[int, int], int] = field(default_factory=lambda: Counter())
    # Histogram of accepted-proposal log|det| change; useful for diagnosing
    # whether the ladder / temperature is at the right scale.
    d_logdet_hist: Counter = field(default_factory=Counter)
    # Periodic Gram snapshots: (step, off_diag_counter).
    gram_offdiag_snapshots: list[tuple[int, dict[int, int]]] = field(default_factory=list)
    # Replica-exchange: per-pair swap attempts / accepts.
    swap_attempts: Counter = field(default_factory=Counter)
    swap_accepts: Counter = field(default_factory=Counter)

    def record_proposal(self, kind: str, accepted: bool,
                        pos: tuple[int, int] | None = None,
                        d_logdet: float | None = None) -> None:
        self.move_proposed[kind] += 1
        if accepted:
            self.move_accepted[kind] += 1
            if pos is not None:
                self.flip_counts[pos] += 1
            if d_logdet is not None:
                # Bucket to 0.25 nat resolution.
                b = round(d_logdet * 4) / 4
                self.d_logdet_hist[b] += 1

    def record_swap(self, pair: tuple[int, int], accepted: bool) -> None:
        self.swap_attempts[pair] += 1
        if accepted:
            self.swap_accepts[pair] += 1

    def snapshot_gram(self, step: int, matrix: list[list[int]]) -> None:
        n = len(matrix)
        counts: Counter = Counter()
        for i in range(n):
            for j in range(i + 1, n):
                s = sum(matrix[i][k] * matrix[j][k] for k in range(n))
                counts[s] += 1
        self.gram_offdiag_snapshots.append((step, dict(counts)))

    def summarise(self) -> dict:
        """Return a JSON-serialisable summary. Also printable for humans."""
        out: dict = {}
        # Acceptance rates by kind.
        rates = {}
        for kind, prop in self.move_proposed.items():
            acc = self.move_accepted.get(kind, 0)
            rates[kind] = {
                "proposed": prop,
                "accepted": acc,
                "rate": acc / prop if prop else 0.0,
            }
        out["acceptance_by_move"] = rates
        # Top-K most-flipped positions (frontiers of the search).
        top_flipped = sorted(
            self.flip_counts.items(), key=lambda kv: -kv[1]
        )[:20]
        out["top_flipped_positions"] = [
            {"pos": list(p), "flips": c} for p, c in top_flipped
        ]
        # Stable positions: total flipped zero times (if we know the grid size).
        # (Caller can compute from flip_counts if they know n.)
        out["num_unique_positions_flipped"] = len(self.flip_counts)
        # d_logdet distribution.
        out["d_logdet_hist"] = dict(sorted(self.d_logdet_hist.items()))
        # Swap stats.
        swap = {}
        for pair, att in self.swap_attempts.items():
            acc = self.swap_accepts.get(pair, 0)
            swap[f"{pair[0]}-{pair[1]}"] = {
                "attempts": att, "accepts": acc,
                "rate": acc / att if att else 0.0,
            }
        out["swap_rates"] = swap
        # Gram snapshots (compact form).
        out["gram_offdiag_snapshots"] = [
            {"step": s, "offdiag": g}
            for s, g in self.gram_offdiag_snapshots
        ]
        return out

    def write_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.summarise(), indent=2) + "\n")


def gram_offdiag_report(matrix: list[list[int]]) -> dict:
    """Standalone fingerprint of a single matrix: Gram off-diagonal +
    singular-value multiplicity pattern. Handy for diagnosing a local optimum.
    """
    import numpy as np

    n = len(matrix)
    M = np.array(matrix, dtype=int)
    G = (M @ M.T).astype(int)
    off = [int(G[i, j]) for i in range(n) for j in range(i + 1, n)]
    counts = dict(sorted(Counter(off).items()))
    sv = np.linalg.svd(M.astype(float), compute_uv=False)
    # Bin singular values to 4 decimals and report multiplicities.
    sv_buckets: Counter = Counter()
    for s in sv:
        sv_buckets[round(float(s), 4)] += 1
    multiplicities = sorted(sv_buckets.items(), reverse=True)
    return {
        "n": n,
        "gram_offdiag_counts": counts,
        "gram_offdiag_support": sorted(counts),
        "sv_max": float(sv.max()),
        "sv_min": float(sv.min()),
        "sv_ratio": float(sv.max() / max(sv.min(), 1e-12)),
        "sv_multiplicities": [(s, c) for s, c in multiplicities],
    }

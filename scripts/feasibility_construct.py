"""Construction attempt on the top parity-feasible H23 targets above Orrick.

Reads the audit output, picks targets above Orrick's 0.7091·Barba whose
predicted ratio sits at or below 1.0 (≤ Barba — guaranteed below the
n ≡ 3 (mod 4) achievable ceiling) AND survive parity, and runs the
generalised SA construction.

Reports the best |det| achieved per target, exact-Bareiss verified.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hadamard_maxdet.feasibility import construct_from_partition_general
from hadamard_maxdet.verifier import upper_bound, verify


# Hard-coded short-list. We pull from the audit; for clarity we list the
# targets explicitly here so this script is self-contained.
TARGETS = [
    # (label, n, partition, intra, cross, predicted_ratio, predicted_abs_det)
    ("H23 (2,3,6,12) (-5,-1) — top sub-Barba feasible",
     23, (2, 3, 6, 12), -5, -1),
    ("H23 (2,3,4,14) (-5,+3) — second sub-Barba feasible",
     23, (2, 3, 4, 14), -5, 3),
    ("H23 (5,6,6,6) (+3,-1) — canonical Ehlich",
     23, (5, 6, 6, 6), 3, -1),
    ("H23 (1,7,7,8) (+3,-1) — singleton-block Ehlich",
     23, (1, 7, 7, 8), 3, -1),
]

ORRICK_RATIO = 0.7091
BTC_DET = 2.6266e15  # current in-repo best


def main() -> None:
    base = float(upper_bound(23))
    print(f"H23 Barba bound: {base:.4e}")
    print(f"Orrick (0.7091): {ORRICK_RATIO * base:.4e}")
    print(f"BTC (in-repo):   {BTC_DET:.4e}")
    print()

    results = []
    for label, n, partition, intra, cross, *_ in TARGETS:
        print(f"--- {label} ---")
        t0 = time.monotonic()
        # Modest budget: 60 s per target.
        res = construct_from_partition_general(
            n, partition, intra, cross,
            iterations=2_000_000,
            seed=42,
            t_start=4.0,
            t_end=0.05,
            wall_time_budget=60.0,
        )
        elapsed = time.monotonic() - t0
        # Exact-Bareiss verify the returned matrix.
        verdict = verify(res.matrix.entries)
        assert verdict.valid, "construction returned invalid ±1 matrix"
        ratio = verdict.abs_det / base if base > 0 else 0.0
        beats_btc = verdict.abs_det > BTC_DET
        beats_orrick = ratio > ORRICK_RATIO
        record = {
            "label": label,
            "n": n,
            "partition": list(partition),
            "intra": intra,
            "cross": cross,
            "abs_det": verdict.abs_det,
            "ratio_to_barba": ratio,
            "final_gram_loss": res.final_gram_loss,
            "matched_block_structure": res.matched_block_structure,
            "iterations": res.iterations,
            "walltime_seconds": elapsed,
            "beats_btc": beats_btc,
            "beats_orrick": beats_orrick,
        }
        results.append(record)
        print(
            f"  abs_det={verdict.abs_det}  ratio={ratio:.4f}  "
            f"loss={res.final_gram_loss}  matched={res.matched_block_structure}  "
            f"iters={res.iterations}  t={elapsed:.1f}s"
        )
        if beats_btc:
            print(f"  *** BEATS BTC by {(verdict.abs_det/BTC_DET - 1)*100:.1f}% ***")
        if beats_orrick:
            print(f"  *** BEATS Orrick (0.7091·Barba) ***")
        print()

    out = ROOT / "results" / "feasibility_construction.json"
    with open(out, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

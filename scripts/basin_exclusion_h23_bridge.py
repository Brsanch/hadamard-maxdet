"""Bridge phase only — uses already-saved basin files from
basin_exclusion_h23.py to run unlocked SA on a small selection of basins
and write incremental per-basin results so we don't lose work to a kill.

Usage:
    python scripts/basin_exclusion_h23_bridge.py --top 6 --bridge-iters 1500000
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

from hadamard_maxdet.class_tabu import (
    bridge_from_seed,
    class_signature,
)
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


BTC_H23_SIG: tuple[tuple[int, int], ...] = (
    (-3, 11), (-1, 209), (1, 11), (3, 22),
)
BTC_H23_ABSDET: int = 2_626_567_700_217_856


def load_basins(n: int, results_dir: Path) -> list[dict]:
    files = sorted(glob.glob(str(results_dir / f"H{n}_basin_*.json")))
    out = []
    for f in files:
        d = json.load(open(f))
        out.append(d)
    out.sort(key=lambda d: -d["abs_det"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--top", type=int, default=6,
                    help="bridge from top-K basins by |det|")
    ap.add_argument("--bridge-iters", type=int, default=1_500_000,
                    help="iterations per bridge run (~75s at n=23, 20k it/s)")
    args = ap.parse_args()

    out_dir = Path(args.results_dir)
    basins_23 = load_basins(23, out_dir)
    print(f"Loaded {len(basins_23)} n=23 basins.")

    # Filter out BTC if it accidentally appeared, and take top K.
    non_btc = [b for b in basins_23 if tuple(tuple(p) for p in b["signature"]) != BTC_H23_SIG]
    non_btc = non_btc[:args.top]
    print(f"Bridging from top {len(non_btc)} non-BTC basins, {args.bridge_iters:,} iters each.")

    bridge_results: list[dict] = []
    overall_best = 0
    overall_best_label: str | None = None
    overall_best_matrix: list[list[int]] | None = None

    t_total = time.monotonic()
    for i, basin in enumerate(non_btc):
        seed_M = Matrix(n=basin["n"], entries=basin["matrix_entries"])
        sig = tuple(tuple(p) for p in basin["signature"])
        sig_str = ",".join(f"({v:+d}:{c})" for v, c in sig)
        print(f"\n[{i+1}/{len(non_btc)}] basin |det|={basin['abs_det']:,} sig={sig_str}", flush=True)
        bridge = bridge_from_seed(
            seed_M,
            iterations=args.bridge_iters,
            t_start=0.5,
            t_end=0.002,
            seed=99_991 + i,
        )
        end_sig = bridge.final_signature
        is_btc = end_sig == BTC_H23_SIG
        verdict = (
            "stayed (same class)" if not bridge.drifted
            else ("→ BTC class" if is_btc else "→ third class")
        )
        rec = {
            "seed_basin_index_in_top_K": i,
            "seed_signature": [list(p) for p in sig],
            "seed_abs_det": basin["abs_det"],
            "final_signature": [list(p) for p in end_sig],
            "final_abs_det": bridge.final_abs_det,
            "drifted": bridge.drifted,
            "reached_btc_class": is_btc,
            "exceeded_btc_ceiling": bridge.final_abs_det > BTC_H23_ABSDET,
            "walltime_seconds": bridge.walltime_seconds,
            "verdict": verdict,
        }
        bridge_results.append(rec)
        # Persist per-bridge result so kill is safe.
        per_path = out_dir / f"H23_bridge_{i}.json"
        per_path.write_text(json.dumps(rec, indent=2) + "\n")
        end_str = ",".join(f"({v:+d}:{c})" for v, c in end_sig)
        print(f"    -> |det|={bridge.final_abs_det:,} ({bridge.final_abs_det/BTC_H23_ABSDET:.4f}·BTC) "
              f"in {bridge.walltime_seconds:.1f}s. {verdict}", flush=True)
        print(f"       end sig: {end_str}", flush=True)
        if bridge.final_abs_det > overall_best:
            overall_best = bridge.final_abs_det
            overall_best_label = f"bridge_from_top_{i}"
            overall_best_matrix = bridge.final_matrix.entries

    consolidated = {
        "experiment": "basin_exclusion_h23_bridge",
        "btc_reference": {
            "abs_det": BTC_H23_ABSDET,
            "signature": [list(p) for p in BTC_H23_SIG],
        },
        "bridge_results": bridge_results,
        "overall_best_abs_det": overall_best,
        "overall_best_label": overall_best_label,
        "overall_best_matrix": overall_best_matrix,
        "exceeded_btc_ceiling": overall_best > BTC_H23_ABSDET,
        "n_basins_bridged": len(non_btc),
        "iters_per_bridge": args.bridge_iters,
        "total_walltime_seconds": time.monotonic() - t_total,
    }
    fp = out_dir / "H23_basin_exclusion_best.json"
    fp.write_text(json.dumps(consolidated, indent=2) + "\n")
    print(f"\nWrote {fp}.")
    print(f"Overall best from bridges: {overall_best:,} = {overall_best/BTC_H23_ABSDET:.4f}·BTC")
    print(f"Exceeded BTC ceiling? {overall_best > BTC_H23_ABSDET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

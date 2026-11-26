"""Targeted bridge experiment at H23.

Load near-2-val matrices (count_5 ∈ {1, 2}) from the 3-val smart-search
output, enumerate single-flip and 2-flip moves that snap each one back to
|G| ⊆ {1, 3}, and report:

* Total bridges found.
* Distinct 2-val (count_1, count_3) destination signatures reached.
* Cross-reference: which destinations are NEW vs. {Orrick, BTC, KMS} ∪
  the 104 prior 2-val classes from ``smart_2val_search.json``.
* Best |det| in any new 2-val class. If any > Orrick, triple-verify and
  flag as a record candidate.

Usage
-----
    python scripts/targeted_bridge_h23.py
    python scripts/targeted_bridge_h23.py --max-k 1     # single-flip only (fast)
    python scripts/targeted_bridge_h23.py --count-5-max 1  # only count_5=1 sources
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hadamard_maxdet.smart_2val_search import enumerate_known_2val_classes
from hadamard_maxdet.targeted_bridge import (
    ORRICK_DET,
    cross_reference,
    enumerate_bridges,
    load_known_2val_signatures,
    load_near_2val_matrices_from_3val_run,
    triple_verify_abs_det,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-3val", default="results/smart_3val_search.json",
        help="Path to the 3-val smart-search output (per-job JSON).",
    )
    parser.add_argument(
        "--out", default="results/targeted_bridge_h23.json",
        help="Path to write the bridge results JSON.",
    )
    parser.add_argument(
        "--count-5-max", type=int, default=2,
        help="Upper bound on count_5 of source matrices.",
    )
    parser.add_argument(
        "--max-k", type=int, default=2,
        help="Maximum flip-set size to enumerate per source (1 or 2).",
    )
    parser.add_argument(
        "--n-max-solutions-per-source", type=int, default=200,
        help="Cap on flip-sets returned per source matrix.",
    )
    args = parser.parse_args()

    in_path = Path(args.in_3val)
    if not in_path.exists():
        print(f"FATAL: input file {in_path} not found.")
        return 1

    print(f"Loading near-2-val matrices from {in_path} (count_5 ≤ {args.count_5_max})...")
    t0 = time.monotonic()
    near = load_near_2val_matrices_from_3val_run(in_path, count_5_max=args.count_5_max)
    t_load = time.monotonic() - t0
    print(f"  Loaded {len(near)} near-2-val matrices in {t_load:.1f}s.")
    if not near:
        print("FATAL: no near-2-val matrices found; nothing to bridge.")
        return 1

    by_c5: dict[int, int] = {}
    for rec in near:
        by_c5[rec.count_5] = by_c5.get(rec.count_5, 0) + 1
    print("  count_5 breakdown:", dict(sorted(by_c5.items())))

    distinct_sigs = set(rec.signature for rec in near)
    print(f"  Distinct source signatures: {len(distinct_sigs)}")

    # Bridge enumeration.
    print(f"\nEnumerating bridges (max_k={args.max_k}, "
          f"n_max_solutions_per_source={args.n_max_solutions_per_source})...")
    t1 = time.monotonic()
    bridges, by_dest = enumerate_bridges(
        near,
        max_k=args.max_k,
        n_max_solutions_per_source=args.n_max_solutions_per_source,
        progress_every=10,
        verbose=True,
    )
    t_enum = time.monotonic() - t1
    print(f"\n  Bridge enumeration done in {t_enum:.1f}s.")
    print(f"  Total bridges (source × flip-set): {len(bridges)}")
    print(f"  Distinct 2-val destinations reached: {len(by_dest)}")

    # Sources that produced at least one bridge.
    sources_with_bridge = set()
    for b in bridges:
        sources_with_bridge.add(b.source_job)
    print(f"  Source jobs with ≥1 bridge: {len(sources_with_bridge)} / {len(near)}")

    # Cross-reference.
    known_2val = load_known_2val_signatures(
        results_dir="results",
        smart_2val_path="results/smart_2val_search.json",
    )
    print(f"\nKnown 2-val signatures: {len(known_2val)} "
          "(canonical {Orrick, BTC, KMS} + smart-2val 104 priors)")
    already, new = cross_reference(by_dest, known_2val)
    print(f"  Destinations already known: {len(already)}")
    print(f"  Destinations NEW: {len(new)}")

    # Sort and report top destinations by |det|.
    by_dest_sorted = sorted(
        by_dest.values(), key=lambda agg: -agg.best_abs_det
    )
    print("\nTop 10 destinations by best |det|:")
    canonical_sigs = {kc.abs_class_multiset: kc.name
                      for kc in enumerate_known_2val_classes("results")}
    for agg in by_dest_sorted[:10]:
        sig = agg.destination_signature
        marker = "[KNOWN-canonical]" if sig in canonical_sigs else (
            "[KNOWN-prior]" if sig in known_2val else "[NEW]")
        canon = f" ({canonical_sigs[sig]})" if sig in canonical_sigs else ""
        print(
            f"  {marker:18s}{canon}  sig={list(sig)}  "
            f"best_det={agg.best_abs_det:>20d}  ratio={agg.best_abs_det/ORRICK_DET:.6f}  "
            f"#bridges={agg.n_bridges}"
        )

    # Best NEW destination.
    new_sorted = sorted(new.values(), key=lambda agg: -agg.best_abs_det)
    if new_sorted:
        best_new = new_sorted[0]
        print(f"\nBest NEW 2-val destination:")
        print(f"  signature: {list(best_new.destination_signature)}")
        print(f"  |det|: {best_new.best_abs_det:>20d}")
        print(f"  ratio to Orrick: {best_new.best_abs_det/ORRICK_DET:.6f}")
        print(f"  #bridges to it: {best_new.n_bridges}")
        print(f"  sample source jobs: {best_new.sample_source_jobs}")
    else:
        print("\nNo NEW 2-val destinations found; all bridges land on already-known classes.")

    # Above-Orrick scan.
    above: list[dict] = []
    for agg in by_dest_sorted:
        if agg.best_abs_det <= ORRICK_DET:
            break  # by_dest_sorted is descending
        try:
            verified = triple_verify_abs_det(agg.best_matrix)
        except RuntimeError as e:
            print(f"  [VERIFY-FAIL] sig={list(agg.destination_signature)}: {e}")
            continue
        if verified > ORRICK_DET:
            # Triple-verify a SECOND time for paranoia.
            verified2 = triple_verify_abs_det(agg.best_matrix)
            assert verified == verified2, "second triple-verify disagreed"
            is_known = agg.destination_signature in known_2val
            entry = {
                "signature": [list(p) for p in agg.destination_signature],
                "abs_det": verified,
                "ratio_to_orrick": verified / ORRICK_DET,
                "is_known_2val_class": is_known,
                "n_bridges": agg.n_bridges,
                "sample_source_jobs": agg.sample_source_jobs,
                "matrix_entries": agg.best_matrix,
            }
            above.append(entry)
            print(f"\n  *** ABOVE ORRICK *** sig={list(agg.destination_signature)} "
                  f"|det|={verified} ratio={verified/ORRICK_DET:.6f} "
                  f"is_known={is_known}")

    # Write outputs.
    summary = {
        "n": 23,
        "experiment": "targeted_bridge_h23",
        "args": vars(args),
        "orrick_det": ORRICK_DET,
        "n_near_2val_sources": len(near),
        "near_2val_count_5_breakdown": by_c5,
        "n_distinct_source_signatures": len(distinct_sigs),
        "max_k": args.max_k,
        "n_bridges": len(bridges),
        "n_distinct_destinations": len(by_dest),
        "n_destinations_already_known": len(already),
        "n_destinations_new": len(new),
        "n_sources_with_bridge": len(sources_with_bridge),
        "destinations_all": [
            {
                "signature": [list(p) for p in agg.destination_signature],
                "best_abs_det": agg.best_abs_det,
                "ratio_to_orrick": agg.best_abs_det / ORRICK_DET,
                "n_bridges": agg.n_bridges,
                "is_known": agg.destination_signature in known_2val,
                "is_canonical": agg.destination_signature in canonical_sigs,
                "canonical_name": canonical_sigs.get(agg.destination_signature),
                "sample_source_jobs": agg.sample_source_jobs,
            }
            for agg in by_dest_sorted
        ],
        "destinations_new_all": [
            {
                "signature": [list(p) for p in agg.destination_signature],
                "best_abs_det": agg.best_abs_det,
                "ratio_to_orrick": agg.best_abs_det / ORRICK_DET,
                "n_bridges": agg.n_bridges,
                "sample_source_jobs": agg.sample_source_jobs,
            }
            for agg in new_sorted
        ],
        "candidates_above_orrick": above,
        "walltime_load_seconds": t_load,
        "walltime_enumerate_seconds": t_enum,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")

    if above:
        record_path = Path("results/H23_targeted_bridge_above_orrick.json")
        with record_path.open("w") as f:
            json.dump({
                "n": 23,
                "manifold": "|G_offdiag| in {1, 3}",
                "candidates": above,
                "verification": "triple-verified twice (Bareiss x2 + transpose Bareiss agreed, then re-run)",
            }, f, indent=2)
        print(f"Wrote {record_path}  (RECORD CANDIDATE — please review)")
        print(f"\n*** RECORD CANDIDATE FLAG: {len(above)} new 2-val class(es) above Orrick. ***")
        return 0

    print("\nNo candidates above Orrick.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Build the H23 basin atlas — six diagnostics over all known local-max
H23 ±1 matrices on disk plus the three constructed families.

Outputs:
    results/basin_atlas.json — distance matrices, MDS coords, saddle path,
        partition assignments, walk reachability graph, plus a human
        summary block.
    results/basin_atlas_mds.png — 2D MDS scatter coloured by |det| (if
        matplotlib is importable).

Run:
    python scripts/build_basin_atlas.py
    python scripts/build_basin_atlas.py --quick      # smaller saddle/walk

The defaults complete in ~75 minutes total on an M2 Ultra; ``--quick``
is ~5 minutes for CI / smoke tests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Make sure the package is importable when run from the repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hadamard_maxdet.basin_atlas import (  # noqa: E402
    BasinEntry,
    bounded_walk_reachability,
    distance_from_anchor_profile,
    hamming_distance,
    load_all_h23_basins,
    mds_embed,
    pairwise_distances,
    partition_assignment,
    saddle_search_between,
)
from hadamard_maxdet.orrick_hunt import enumerate_ehlich_partitions  # noqa: E402
from hadamard_maxdet.verifier import upper_bound  # noqa: E402


# Standing BTC determinant for n=23 (rich-ansatz / attack / orrick-hunt all
# converge to this). We identify the BTC-class basin by max |det|.
BTC_ABS_DET = 2_626_567_700_217_856


def _identify_btc(basins: list[BasinEntry]) -> int:
    """Return the index of the BTC basin (highest |det|)."""
    return max(range(len(basins)), key=lambda i: basins[i].abs_det)


def _identify_closest_cousin(
    basins: list[BasinEntry], btc_idx: int, hamming: np.ndarray
) -> int | None:
    """Return the index of the basin closest to BTC by Hamming, excluding BTC.

    "Closest cousin" in the brief is the one with |det| closest to BTC; we
    interpret it as the next-best-|det| basin, which matches the documented
    BTC cousins at 0.916 and 0.943 of BTC.
    """
    candidates = [
        (i, basins[i].abs_det) for i in range(len(basins)) if i != btc_idx
    ]
    if not candidates:
        return None
    # Highest |det| among non-BTC = closest cousin in determinant.
    candidates.sort(key=lambda t: -t[1])
    return candidates[0][0]


def _try_plot_mds(
    coords: np.ndarray, dets: list[int], labels: list[str], path: Path
) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], c=np.array(dets, dtype=float),
        cmap="viridis", s=80, edgecolor="black", linewidth=0.4,
    )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("|det|")
    for i, label in enumerate(labels):
        # Trim long labels.
        short = label[:18]
        ax.annotate(
            short, (coords[i, 0], coords[i, 1]),
            fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points",
        )
    ax.set_xlabel("MDS axis 1")
    ax.set_ylabel("MDS axis 2")
    ax.set_title("H23 basin atlas — MDS embedding from Hamming distance")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir", default=str(REPO_ROOT / "results"),
        help="results/ directory holding H23_*.json files",
    )
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "results" / "basin_atlas.json"),
        help="output JSON path",
    )
    ap.add_argument(
        "--plot", default=str(REPO_ROOT / "results" / "basin_atlas_mds.png"),
        help="output PNG path for MDS scatter",
    )
    ap.add_argument(
        "--saddle-iters", type=int, default=5000,
        help="SA iterations per saddle waypoint (default: 5000)",
    )
    ap.add_argument(
        "--saddle-waypoints", type=int, default=15,
        help="number of interior waypoints in saddle search",
    )
    ap.add_argument(
        "--walk-budget", type=int, default=1000,
        help="2-flip walk steps per basin (default: 1000)",
    )
    ap.add_argument(
        "--walk-floor", type=float, default=0.5,
        help="bounded-walk |det| floor as fraction of seed basin |det|",
    )
    ap.add_argument(
        "--quick", action="store_true",
        help="reduce saddle and walk budgets for a fast smoke run",
    )
    ap.add_argument(
        "--no-include-constructed", action="store_true",
        help="skip Williamson/KMS/Paley-deletion construction",
    )
    args = ap.parse_args()

    if args.quick:
        args.saddle_iters = 500
        args.saddle_waypoints = 5
        args.walk_budget = 100

    print("--- H23 basin atlas ---")
    print(f"results dir: {args.results_dir}")
    print(f"output JSON: {args.out}")
    print(f"saddle: waypoints={args.saddle_waypoints}, iters={args.saddle_iters}")
    print(f"walk: budget={args.walk_budget}, floor={args.walk_floor}")
    print()

    t0 = time.monotonic()
    basins = load_all_h23_basins(
        args.results_dir, n=23,
        include_constructed=not args.no_include_constructed,
    )
    print(f"Loaded {len(basins)} distinct basin matrices "
          f"(after class-signature dedup).")
    barba = upper_bound(23)
    print(f"Barba bound (n=23): {barba:.4e}")
    print(f"Top-5 |det| / Barba:")
    for i, b in enumerate(basins[:5]):
        print(f"  [{i:2d}] {b.label:30s} |det|={b.abs_det:.4e}  "
              f"ratio={b.abs_det/barba:.4f}  source={b.source}")
    print()

    btc_idx = _identify_btc(basins)
    print(f"BTC basin: index {btc_idx}, label='{basins[btc_idx].label}', "
          f"|det|={basins[btc_idx].abs_det:.4e}")

    # ---- Diagnostic 1: pairwise distances ----
    t = time.monotonic()
    dists = pairwise_distances(basins)
    print(f"[1] Pairwise distances: {time.monotonic()-t:.2f}s")
    H = dists["hamming"]
    G = dists["gram_emd"]
    S = dists["spectral"]

    # Find closest pair (off-diagonal).
    N = len(basins)
    H_off = H + np.eye(N, dtype=H.dtype) * 10**9
    flat_idx = int(np.argmin(H_off))
    ci, cj = flat_idx // N, flat_idx % N
    print(f"    Closest pair (Hamming): basins[{ci}] '{basins[ci].label}' "
          f"|det|={basins[ci].abs_det:.4e} <-> basins[{cj}] "
          f"'{basins[cj].label}' |det|={basins[cj].abs_det:.4e}, "
          f"distance={H[ci, cj]} of 529")

    # ---- Diagnostic 2: MDS embedding ----
    t = time.monotonic()
    coords = mds_embed(H.astype(np.float64), k=2)
    print(f"[2] MDS embedding: {time.monotonic()-t:.2f}s; shape={coords.shape}")
    plot_path = Path(args.plot)
    plot_made = _try_plot_mds(
        coords, [b.abs_det for b in basins], [b.label for b in basins], plot_path,
    )
    if plot_made:
        print(f"    MDS scatter saved to {plot_path}")
    else:
        print(f"    matplotlib unavailable; skipping {plot_path}")

    # ---- Diagnostic 3: saddle search BTC <-> closest cousin ----
    t = time.monotonic()
    cousin_idx = _identify_closest_cousin(basins, btc_idx, H)
    if cousin_idx is not None:
        print(f"[3] Saddle search: BTC <-> '{basins[cousin_idx].label}' "
              f"(|det| ratio cousin/BTC = "
              f"{basins[cousin_idx].abs_det/basins[btc_idx].abs_det:.3f})")
        saddle = saddle_search_between(
            basins[btc_idx].matrix, basins[cousin_idx].matrix,
            n_intermediate=args.saddle_waypoints,
            sa_iters=args.saddle_iters, seed=0,
            label_a=basins[btc_idx].label, label_b=basins[cousin_idx].label,
        )
        print(f"    Hamming(BTC, cousin) = {saddle.hamming_a_b}")
        print(f"    Saddle height = {saddle.saddle_height:.4e} "
              f"(ratio to BTC = "
              f"{saddle.saddle_height/saddle.abs_det_a:.3f})")
        print(f"    walltime: {saddle.walltime_seconds:.1f}s "
              f"({time.monotonic()-t:.1f}s wall)")
        saddle_json = saddle.to_json()
    else:
        saddle_json = None
        print(f"[3] Saddle search skipped (no cousin found).")

    # ---- Diagnostic 4: Ehlich-partition coverage ----
    t = time.monotonic()
    target_partitions = [
        (5, 6, 6, 6),
        (5, 5, 6, 7),
        (4, 6, 6, 7),
        (5, 5, 5, 8),
        (4, 5, 7, 7),
    ]
    # Pull predicted bound from enumerate_ehlich_partitions.
    all_parts = enumerate_ehlich_partitions(23, intra_values=(3,), cross_values=(-1,))
    pred_by_size = {tuple(sorted(p.sizes)): p.predicted_abs_det for p in all_parts}
    partition_results = []
    for psize in target_partitions:
        sized = tuple(sorted(psize))
        predicted = pred_by_size.get(sized)
        per_basin: list[dict] = []
        for i, b in enumerate(basins):
            pa = partition_assignment(
                b.matrix, partition_sizes=psize,
                intra_value=3, cross_value=-1, n_restarts=4,
            )
            per_basin.append({
                "basin_idx": i,
                "label": b.label,
                "abs_det": b.abs_det,
                "match_score": pa.gram_match_score,
                "intra_match": pa.intra_match_count,
                "intra_total": pa.intra_total,
                "cross_match": pa.cross_match_count,
                "cross_total": pa.cross_total,
            })
        per_basin.sort(key=lambda r: -r["match_score"])
        partition_results.append({
            "partition": list(psize),
            "predicted_abs_det": predicted,
            "best_basin": per_basin[0],
            "all_basins": per_basin,
        })
        print(f"[4] Partition {psize}: predicted={predicted:.4e}, "
              f"best fit '{per_basin[0]['label']}' "
              f"score={per_basin[0]['match_score']:.3f} "
              f"|det|={per_basin[0]['abs_det']:.4e}")
    print(f"    [4] Ehlich-partition coverage: {time.monotonic()-t:.1f}s")

    # ---- Diagnostic 5: distance-from-BTC profile ----
    t = time.monotonic()
    profile = distance_from_anchor_profile(basins, anchor_idx=btc_idx)
    profile.sort(key=lambda e: e["hamming_to_anchor"])
    # Compute Pearson correlation between Hamming and ratio.
    hams = np.array([e["hamming_to_anchor"] for e in profile], dtype=float)
    rats = np.array([e["ratio_to_bound"] for e in profile], dtype=float)
    corr = float(np.corrcoef(hams, rats)[0, 1]) if len(hams) > 1 else 0.0
    print(f"[5] Distance-from-BTC profile: Pearson corr(Hamming, ratio)="
          f"{corr:+.3f} over {len(hams)} non-BTC basins "
          f"({time.monotonic()-t:.2f}s)")

    # ---- Diagnostic 6: bounded-walk reachability ----
    t = time.monotonic()
    walk = bounded_walk_reachability(
        basins, walk_budget=args.walk_budget,
        det_floor_fraction=args.walk_floor, seed=0,
    )
    n_edges = len(walk["edges"])
    btc_outdeg = sum(1 for e in walk["edges"] if e["from"] == btc_idx)
    btc_indeg = sum(1 for e in walk["edges"] if e["to"] == btc_idx)
    print(f"[6] Bounded-walk reachability: {n_edges} edges; BTC out-degree="
          f"{btc_outdeg}, in-degree={btc_indeg} ({time.monotonic()-t:.1f}s)")

    # ---- Persist ----
    out = {
        "n": 23,
        "barba_upper_bound": barba,
        "btc_idx": btc_idx,
        "btc_abs_det": basins[btc_idx].abs_det,
        "n_basins": N,
        "basins": [b.to_json() for b in basins],
        "distances": {
            "hamming": H.astype(int).tolist(),
            "gram_emd": G.tolist(),
            "spectral": S.tolist(),
        },
        "mds_coords": coords.tolist(),
        "closest_pair": {
            "i": int(ci), "j": int(cj),
            "labels": [basins[ci].label, basins[cj].label],
            "hamming": int(H[ci, cj]),
            "abs_dets": [basins[ci].abs_det, basins[cj].abs_det],
        },
        "saddle": saddle_json,
        "partition_coverage": partition_results,
        "distance_from_btc": {
            "anchor_label": basins[btc_idx].label,
            "anchor_abs_det": basins[btc_idx].abs_det,
            "pearson_corr_hamming_vs_ratio": corr,
            "profile": profile,
        },
        "walk_reachability": walk,
        "config": {
            "saddle_iters": args.saddle_iters,
            "saddle_waypoints": args.saddle_waypoints,
            "walk_budget": args.walk_budget,
            "walk_floor": args.walk_floor,
            "include_constructed": not args.no_include_constructed,
        },
        "walltime_seconds_total": time.monotonic() - t0,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path} ({os.path.getsize(out_path)/1024:.1f} KB)")
    print(f"Total walltime: {out['walltime_seconds_total']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

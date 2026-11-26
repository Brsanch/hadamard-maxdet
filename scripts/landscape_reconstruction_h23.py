#!/usr/bin/env python3
"""Landscape reconstruction for H23.

Loads every H23 ±1 matrix accessible on disk (single-matrix result files
+ best-feasible matrices from the smart 2-val and 3-val campaigns),
extracts diagnostic features, fits a regression model that predicts
log10|det|, and reports:

* Overall R² (linear + random forest).
* Top 5 features by importance.
* Top 20 positive residuals (matrices with surprisingly high |det|).
* Predicted |det| at hypothetical (c1, c3) 2-val classes above Orrick.
* Blind-spot bins (PCA-2D regions of feature space with no samples).

Output: ``results/landscape_model.json``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from hadamard_maxdet.landscape_model import (  # noqa: E402
    FEATURE_NAMES,
    ORRICK_H23_ABS_DET,
    blind_spot_2d,
    build_data_matrix,
    extrapolate,
    fit_predictive_model,
    residual_analysis,
)
from hadamard_maxdet.verifier import integer_determinant  # noqa: E402


N = 23
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------


def _walk_for_matrices(obj, n: int, found: list[tuple[str, list[list[int]]]], path: str = ""):
    """Recursively scan a JSON-decoded object for embedded n×n ±1 matrices."""
    matrix_keys = (
        "matrix",
        "matrix_entries",
        "entries",
        "overall_best_matrix",
        "final_matrix",
        "best_feasible_entries",
        "best_matrix_entries",
    )
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            if k in matrix_keys and isinstance(v, list) and len(v) == n and isinstance(v[0], list):
                rows_ok = (
                    all(isinstance(row, list) and len(row) == n for row in v)
                    and all(all(int(x) in (-1, 1) for x in row) for row in v[:1])
                )
                if rows_ok:
                    try:
                        mat = [[int(x) for x in row] for row in v]
                        found.append((new_path, mat))
                    except (TypeError, ValueError):
                        pass
            else:
                _walk_for_matrices(v, n, found, new_path)
    elif isinstance(obj, list):
        for i, x in enumerate(obj):
            _walk_for_matrices(x, n, found, f"{path}[{i}]")


def load_h23_corpus(verbose: bool = True) -> list[tuple[str, list[list[int]], int]]:
    """Return ``(label, matrix, abs_det)`` triples for every H23 matrix on
    disk. Deduplicates exact-equal matrices (same row order, same entries)
    by hashing.
    """
    triples: list[tuple[str, list[list[int]], int]] = []
    seen_hashes: set[int] = set()
    skipped_singular = 0
    skipped_dup = 0

    files = sorted(RESULTS_DIR.glob("*.json"))
    for fp in files:
        # Skip the huge per-job log; we'll handle it specially.
        if fp.name in ("smart_3val_search.json", "smart_2val_search.json"):
            continue
        try:
            with open(fp) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        found: list[tuple[str, list[list[int]]]] = []
        _walk_for_matrices(d, N, found)
        for path, mat in found:
            h = hash(tuple(tuple(r) for r in mat))
            if h in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(h)
            try:
                det = integer_determinant(mat)
            except Exception:
                continue
            if det == 0:
                skipped_singular += 1
                continue
            label = f"{fp.stem}::{path}"
            triples.append((label, mat, abs(det)))

    # Now stream the two large per-job result files for `best_feasible_entries`.
    for fp in (RESULTS_DIR / "smart_2val_search.json", RESULTS_DIR / "smart_3val_search.json"):
        if not fp.exists():
            continue
        if verbose:
            print(f"loading large file {fp.name} ({fp.stat().st_size / 1024 / 1024:.1f} MB) ...", flush=True)
        try:
            with open(fp) as f:
                d = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        per_job = d.get("per_job") or d.get("per_job_summary") or []
        if isinstance(per_job, dict):
            per_job = list(per_job.values())
        for entry in per_job:
            result = entry.get("result", entry) if isinstance(entry, dict) else None
            if not isinstance(result, dict):
                continue
            mat = result.get("best_feasible_entries")
            if not (isinstance(mat, list) and len(mat) == N and isinstance(mat[0], list)):
                continue
            try:
                mat = [[int(x) for x in row] for row in mat]
            except (TypeError, ValueError):
                continue
            h = hash(tuple(tuple(r) for r in mat))
            if h in seen_hashes:
                skipped_dup += 1
                continue
            seen_hashes.add(h)
            try:
                det = integer_determinant(mat)
            except Exception:
                continue
            if det == 0:
                skipped_singular += 1
                continue
            tag = entry.get("name") or result.get("seed_name") or "job"
            label = f"{fp.stem}::{tag}"
            triples.append((label, mat, abs(det)))

    if verbose:
        print(
            f"loaded {len(triples)} unique H23 matrices "
            f"(skipped {skipped_singular} singular, {skipped_dup} dup)",
            flush=True,
        )
    return triples


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        default=str(RESULTS_DIR / "landscape_model.json"),
        help="Output JSON path",
    )
    ap.add_argument("--top-residuals", type=int, default=20)
    ap.add_argument("--blind-spot-bins", type=int, default=10)
    args = ap.parse_args()

    print("=" * 72)
    print("Landscape reconstruction for H23")
    print("=" * 72)

    t0 = time.monotonic()
    triples = load_h23_corpus(verbose=True)
    if len(triples) < 10:
        print(f"ERROR: only {len(triples)} H23 matrices found; need >= 10.")
        sys.exit(1)

    print(f"\n[1/5] Building feature matrix ...")
    t1 = time.monotonic()
    data = build_data_matrix(triples)
    print(
        f"  X = {data.X.shape}, y range = "
        f"[{data.y.min():.3f}, {data.y.max():.3f}], "
        f"|det| max = {data.abs_dets.max():.3e} (Orrick = {ORRICK_H23_ABS_DET:.3e}); "
        f"feature extraction took {time.monotonic() - t1:.1f}s"
    )

    print(f"\n[2/5] Fitting models ...")
    t2 = time.monotonic()
    fits = fit_predictive_model(data)
    for name, fit in fits.items():
        print(f"  {name:<15} R² = {fit.r2_train:.4f}")
    print(f"  fit took {time.monotonic() - t2:.1f}s")

    # Top features for the better fit (RF usually).
    rf_fit = fits["random_forest"]
    sorted_imp = sorted(
        rf_fit.feature_importances.items(), key=lambda kv: -kv[1]
    )
    print("\n  Top 10 features by importance (random forest):")
    for name, imp in sorted_imp[:10]:
        print(f"    {name:<22} {imp:.4f}")

    print(f"\n[3/5] Residual analysis (top {args.top_residuals}) ...")
    res_rf = residual_analysis(rf_fit, data, top_k=args.top_residuals)
    res_lr = residual_analysis(fits["linear"], data, top_k=args.top_residuals)
    print("\n  Top positive residuals (RF — surprisingly HIGH |det|):")
    for r in res_rf["top_positive"][:10]:
        ratio = r.abs_det / ORRICK_H23_ABS_DET
        print(
            f"    +{r.residual:+.3f}  |det|={r.abs_det:.4e} "
            f"({ratio:.3f}·Orrick)  {r.label}"
        )
    print("\n  Top negative residuals (RF — surprisingly LOW |det|):")
    for r in res_rf["top_negative"][:5]:
        ratio = r.abs_det / ORRICK_H23_ABS_DET
        print(
            f"    {r.residual:+.3f}  |det|={r.abs_det:.4e} "
            f"({ratio:.3f}·Orrick)  {r.label}"
        )

    # Top positive (linear) — useful cross-check.
    print("\n  Top positive residuals (linear cross-check):")
    for r in res_lr["top_positive"][:5]:
        ratio = r.abs_det / ORRICK_H23_ABS_DET
        print(
            f"    +{r.residual:+.3f}  |det|={r.abs_det:.4e} "
            f"({ratio:.3f}·Orrick)  {r.label}"
        )

    print(f"\n[4/5] Extrapolation to hypothetical (c1, c3) 2-val classes ...")
    means = dict(zip(FEATURE_NAMES, data.X.mean(axis=0).tolist()))
    # Population mean is mostly 3-val, so override Gram fields per-hypothesis.
    n_pairs = N * (N - 1) // 2  # = 253 for N=23
    candidates = []
    # Sweep around (208, 45), the Orrick class. Add classes pushing higher c1
    # (more |G|=1 entries → flatter Gram → conjecturally higher |det|).
    for c1, c3 in [
        (208, 45),  # Orrick (sanity)
        (210, 43),
        (212, 41),
        (215, 38),
        (218, 35),
        (220, 33),  # BTC class
        (200, 53),
        (197, 56),
        (195, 58),
        (185, 68),
        (175, 78),
        (165, 88),
        (253, 0),  # extreme: all |G|=1
        (0, 253),  # extreme: all |G|=3
    ]:
        if c1 + c3 != n_pairs:
            continue
        candidates.append((c1, c3))

    hyp_features = []
    descs = []
    for c1, c3 in candidates:
        # Synthesise a feature dict: Gram-multiset features set explicitly,
        # all others default to the population mean.
        feat = {
            "gram_card": 4.0,  # signed: -3, -1, +1, +3
            "gram_abs_card": 2.0,
            "count_abs_1": float(c1),
            "count_abs_3": float(c3),
            "count_abs_5": 0.0,
            "count_abs_7_plus": 0.0,
            "count_neg_3": float(c3) // 2,
            "count_neg_1": float(c1) // 2,
            "count_pos_1": float(c1) - float(c1) // 2,
            "count_pos_3": float(c3) - float(c3) // 2,
            "max_abs_offdiag": 3.0 if c3 > 0 else 1.0,
            "mean_abs_offdiag": (c1 + 3 * c3) / max(c1 + c3, 1),
            "std_abs_offdiag": math_std_2val(c1, c3),
            "n": float(N),
        }
        hyp_features.append(feat)
        descs.append(f"2val_({c1},{c3})")

    extra_rf = extrapolate(rf_fit, hyp_features, descriptions=descs, population_means=means)
    extra_lr = extrapolate(fits["linear"], hyp_features, descriptions=descs, population_means=means)

    print("  Hypothetical 2-val (c1, c3) predictions (RF / Linear):")
    print(f"  {'class':<22} {'RF |det|':>14} {'×Orrick':>9}   {'LR |det|':>14} {'×Orrick':>9}")
    for e_rf, e_lr in zip(extra_rf, extra_lr):
        print(
            f"  {e_rf.description:<22} {e_rf.predicted_abs_det:14.4e} {e_rf.ratio_to_orrick:9.3f}   "
            f"{e_lr.predicted_abs_det:14.4e} {e_lr.ratio_to_orrick:9.3f}"
        )

    # Find any extrapolation predicting > Orrick.
    above_orrick = [e for e in extra_rf if e.ratio_to_orrick > 1.0] + [
        e for e in extra_lr if e.ratio_to_orrick > 1.0
    ]
    if above_orrick:
        print(f"\n  *** {len(above_orrick)} extrapolations predict > Orrick ***")

    print(f"\n[5/5] Blind-spot identification (PCA-2D, {args.blind_spot_bins}x{args.blind_spot_bins} bins) ...")
    bs = blind_spot_2d(data, nbins=args.blind_spot_bins)
    populated = int(np.sum(bs.histogram > 0))
    total_bins = args.blind_spot_bins ** 2
    print(
        f"  PCA explained variance: PC1={bs.pca_explained_variance_ratio[0]:.3f}, "
        f"PC2={bs.pca_explained_variance_ratio[1]:.3f}; "
        f"populated bins: {populated} / {total_bins}; "
        f"blind bins: {len(bs.blind_spots)}"
    )
    print("  Top 10 most-isolated blind spots (PCA-distance to nearest sample):")
    for b in bs.blind_spots[:10]:
        print(
            f"    bin ({b.bin_x:2d}, {b.bin_y:2d})  PC1∈[{b.x_range[0]:+.2f}, {b.x_range[1]:+.2f}]  "
            f"PC2∈[{b.y_range[0]:+.2f}, {b.y_range[1]:+.2f}]  "
            f"d={b.distance_to_nearest_sample:.3f}"
        )

    walltime = time.monotonic() - t0
    print(f"\nTotal walltime: {walltime:.1f}s")

    # Build serialisable output.
    out = {
        "n": N,
        "n_matrices": int(data.X.shape[0]),
        "n_features": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "orrick_abs_det": ORRICK_H23_ABS_DET,
        "max_abs_det_in_corpus": float(data.abs_dets.max()),
        "max_log_det_in_corpus": float(data.y.max()),
        "models": {
            name: {
                "r2_train": fit.r2_train,
                "feature_importances": fit.feature_importances,
            }
            for name, fit in fits.items()
        },
        "top_positive_residuals_rf": [
            {
                "label": r.label,
                "actual_log": r.actual_log,
                "predicted_log": r.predicted_log,
                "residual": r.residual,
                "abs_det": r.abs_det,
                "ratio_to_orrick": r.abs_det / ORRICK_H23_ABS_DET,
            }
            for r in res_rf["top_positive"]
        ],
        "top_negative_residuals_rf": [
            {
                "label": r.label,
                "actual_log": r.actual_log,
                "predicted_log": r.predicted_log,
                "residual": r.residual,
                "abs_det": r.abs_det,
                "ratio_to_orrick": r.abs_det / ORRICK_H23_ABS_DET,
            }
            for r in res_rf["top_negative"]
        ],
        "extrapolations": {
            "random_forest": [
                {
                    "description": e.description,
                    "features": e.features,
                    "predicted_log": e.predicted_log,
                    "predicted_abs_det": e.predicted_abs_det,
                    "ratio_to_orrick": e.ratio_to_orrick,
                }
                for e in extra_rf
            ],
            "linear": [
                {
                    "description": e.description,
                    "features": e.features,
                    "predicted_log": e.predicted_log,
                    "predicted_abs_det": e.predicted_abs_det,
                    "ratio_to_orrick": e.ratio_to_orrick,
                }
                for e in extra_lr
            ],
        },
        "blind_spots": {
            "pca_explained_variance_ratio": list(bs.pca_explained_variance_ratio),
            "n_bins_per_axis": args.blind_spot_bins,
            "n_populated_bins": populated,
            "n_blind_bins": len(bs.blind_spots),
            "top_isolated_bins": [
                {
                    "bin_x": b.bin_x,
                    "bin_y": b.bin_y,
                    "x_range": list(b.x_range),
                    "y_range": list(b.y_range),
                    "distance_to_nearest_sample": b.distance_to_nearest_sample,
                }
                for b in bs.blind_spots[:25]
            ],
        },
        "walltime_seconds": walltime,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


def math_std_2val(c1: int, c3: int) -> float:
    """Std-dev of the |Gram| values for a hypothetical 2-val class."""
    import math
    n = c1 + c3
    if n == 0:
        return 0.0
    mean = (c1 * 1 + c3 * 3) / n
    var = (c1 * (1 - mean) ** 2 + c3 * (3 - mean) ** 2) / n
    return math.sqrt(var)


if __name__ == "__main__":
    main()

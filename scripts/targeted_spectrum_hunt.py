"""Targeted-spectrum SA hunt at n=17 (sanity), n=19, n=23.

Driver for the three experiments described in the targeted-spectrum
finding (2026-04-25):

  1. n=17 reverse-validation: predict the spectrum at the proven |det|,
     compare to the proven winner's actual SVs.
  2. n=19 forward target: predict the spectrum at proven |det|
     894,426,939,392 and run anchored SA from BTC seed + random.
  3. n=23 multi-target sweep: for each target |det| in the candidate
     set, predict spectrum and run anchored SA at H23 from BTC seed.

Usage::

    PYTHONPATH=src python3 scripts/targeted_spectrum_hunt.py

Time budget ≈ 75 minutes default; CLI flags below override.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.rich_ansatz import sa_bordered_two_circulant
from hadamard_maxdet.spectral_anchor import (
    predict_target_spectrum,
    sa_targeted_spectrum,
    spectrum_distance,
)
from hadamard_maxdet.verifier import integer_determinant
from hadamard_maxdet.winner_decode import find_winning_matrix

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _btc_seed(n: int, *, seconds: float, base_seed: int) -> tuple[int, list[list[int]]]:
    """Best bordered-two-circulant SA seed at order ``n`` within
    wall-time ``seconds``. Odd ``n`` only.
    """
    if n % 2 == 0:
        # Fallback: random ±1.
        rng = np.random.default_rng(base_seed)
        ent = rng.choice([-1, 1], size=(n, n)).tolist()
        return abs(integer_determinant(ent)), ent
    t0 = time.monotonic()
    best_det = 0
    best_ent: list[list[int]] = []
    chunks = 0
    while time.monotonic() - t0 < seconds:
        r = sa_bordered_two_circulant(
            n,
            iterations=20_000,
            t_start=1.0,
            t_end=0.005,
            seed=base_seed + chunks,
        )
        if int(r.abs_det) > best_det:
            best_det = int(r.abs_det)
            best_ent = [list(row) for row in r.matrix.entries]
        chunks += 1
    return best_det, best_ent


def _summary_spectrum(M: list[list[int]]) -> list[list[float]]:
    """Cluster the SVs of M and return [(value, mult)] descending."""
    arr = np.array(M, dtype=np.float64)
    sv = np.linalg.svd(arr, compute_uv=False)
    s_max = float(sv[0])
    out: list[list[float]] = []
    if sv.size == 0:
        return out
    rep = float(sv[0])
    cnt = 1
    thr = 0.05 * s_max
    for v in sv[1:]:
        if rep - float(v) <= thr:
            cnt += 1
        else:
            out.append([round(rep, 4), int(cnt)])
            rep = float(v)
            cnt = 1
    out.append([round(rep, 4), int(cnt)])
    return out


def _save_matrix(path: Path, n: int, entries: list[list[int]], extra: dict) -> None:
    payload = {"n": n, "entries": entries, **extra}
    path.write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Experiment 1: n=17 sanity
# ---------------------------------------------------------------------------


def experiment_1_n17_sanity() -> dict[str, Any]:
    """Find a proven n=17 winner; compare its SVs to several predictions."""
    print("\n" + "=" * 70)
    print("Experiment 1: n=17 reverse-validation")
    print("=" * 70)
    n, det = 17, 21_474_836_480
    abs_det, entries, src = find_winning_matrix(
        n,
        target=det,
        n_starts=4,
        iterations=80_000,
        structured_seconds=40.0,
    )
    print(f"  Proven winner found: |det|={abs_det:,} from {src}")
    actual_spec = sorted(
        [float(s) for s in np.linalg.svd(np.array(entries, dtype=float),
                                           compute_uv=False)],
        reverse=True,
    )
    actual_clustered = _summary_spectrum(entries)
    print(f"  Actual SV clusters: {actual_clustered}")

    predictions: dict[str, Any] = {}
    for model in ["high_only", "symmetric_pair", "equal_pair"]:
        for ratio_kw in [{}] if model != "symmetric_pair" else [{"ratio": 1.0}, {"ratio": 1.566}, {"ratio": 2.0}]:
            label = model + ("" if not ratio_kw else f"_r{list(ratio_kw.values())[0]}")
            try:
                pred = predict_target_spectrum(n, det, model=model, **ratio_kw)
                rel_err = float(np.max(
                    np.abs(np.array(actual_spec) - pred) / np.maximum(np.array(actual_spec), 1e-9)
                ))
                pred_summary = sorted(set(round(float(p), 3) for p in pred), reverse=True)
                predictions[label] = {
                    "spec_top5": [round(float(p), 4) for p in pred[:5]],
                    "spec_bot2": [round(float(p), 4) for p in pred[-2:]],
                    "max_relative_error_vs_actual": rel_err,
                    "distinct_predicted_values": pred_summary,
                }
                print(f"    {label}: max_rel_err={rel_err:.4f}, distinct_vals={pred_summary}")
            except Exception as e:
                predictions[label] = {"error": str(e)}
                print(f"    {label}: FAILED ({e})")

    return {
        "n": n,
        "target_det": det,
        "abs_det_found": abs_det,
        "source": src,
        "actual_sv_clusters": actual_clustered,
        "actual_sv_top5": [round(s, 4) for s in actual_spec[:5]],
        "actual_sv_bot2": [round(s, 4) for s in actual_spec[-2:]],
        "predictions": predictions,
    }


# ---------------------------------------------------------------------------
# Experiment 2: n=19 forward
# ---------------------------------------------------------------------------


def _run_anchored(
    seed_ent: list[list[int]],
    target: np.ndarray,
    *,
    n: int,
    sa_iterations: int,
    sa_seconds: float,
    lambda_anchor: float,
    rng_seed: int,
    rescore_every: int = 1,
) -> tuple[Any, float]:
    deadline = time.monotonic() + sa_seconds
    res = sa_targeted_spectrum(
        Matrix(n=n, entries=seed_ent),
        target_spectrum=target,
        iterations=sa_iterations,
        lambda_anchor=lambda_anchor,
        seed=rng_seed,
        deadline=deadline,
        rescore_every=rescore_every,
        t_start=1.0,
        t_end=0.005,
    )
    dist = float(getattr(res, "spectrum_distance", spectrum_distance(res.matrix.entries, target)))
    return res, dist


def experiment_2_n19(
    *,
    sa_iterations: int = 80_000,
    btc_seconds: float = 30.0,
    sa_seconds: float = 180.0,
    lambda_anchor: float = 1.0,
) -> dict[str, Any]:
    """n=19 forward target. Tries multiple spectrum models (since the
    proven n=19 winner's actual spectrum is unknown — extrapolating from
    n=17's calibrated ratio=1.566 may not transfer).
    """
    print("\n" + "=" * 70)
    print("Experiment 2: n=19 forward-target")
    print("=" * 70)
    n, det = 19, 894_426_939_392

    # BTC seed.
    btc_det, btc_ent = _btc_seed(n, seconds=btc_seconds, base_seed=10001)
    print(f"  BTC seed: |det|={btc_det:,} ({btc_det / det:.4f}·proven)")

    rng = np.random.default_rng(2026)
    rand_ent = rng.choice([-1, 1], size=(n, n)).tolist()
    rand_det = abs(integer_determinant(rand_ent))

    # Try several model variants for the target spectrum.
    spectra_to_try: list[tuple[str, np.ndarray]] = []
    for ratio in [1.0, 1.566, 2.0]:
        try:
            spectra_to_try.append((
                f"symmetric_pair_r{ratio}",
                predict_target_spectrum(n, det, model="symmetric_pair", ratio=ratio),
            ))
        except Exception as e:
            print(f"  symmetric_pair r={ratio} failed: {e}")
    for model in ["high_only", "equal_pair"]:
        try:
            spectra_to_try.append((
                model,
                predict_target_spectrum(n, det, model=model),
            ))
        except Exception as e:
            print(f"  {model} failed: {e}")

    # Per-target time budget: divide sa_seconds/2 across spectrum models for
    # the BTC seed and similarly for random.
    budget_per_model = sa_seconds / max(1, len(spectra_to_try))

    runs: list[dict[str, Any]] = []
    overall_best_det = 0
    overall_best_ent: list[list[int]] = []
    overall_best_label = ""
    overall_best_dist = math.inf

    for (label, target) in spectra_to_try:
        print(f"\n  -- target spectrum: {label} (top3 {target[:3].round(3).tolist()}, "
              f"bot {target[-1].round(3)})")
        btc_dist0 = spectrum_distance(btc_ent, target)
        print(f"     BTC seed dist: {btc_dist0:.4f}")
        for lam in [lambda_anchor, lambda_anchor * 0.2]:
            print(f"     λ_anchor={lam}, t≈{budget_per_model/2:.0f}s from BTC...")
            res_btc, dbtc = _run_anchored(
                btc_ent, target, n=n, sa_iterations=sa_iterations,
                sa_seconds=budget_per_model / 2, lambda_anchor=lam,
                rng_seed=2027 + int(lam * 1000),
            )
            print(f"       -> |det|={res_btc.abs_det:,} ({res_btc.abs_det/det:.4f}), "
                  f"dist={dbtc:.4f}, acc={res_btc.accepted}/{res_btc.iterations}")
            runs.append({
                "spectrum_model": label, "lambda": lam, "seed": "btc",
                "abs_det": int(res_btc.abs_det),
                "ratio_to_proven": res_btc.abs_det / det,
                "spectrum_distance": dbtc,
                "actual_sv_clusters": _summary_spectrum(res_btc.matrix.entries),
                "iterations": res_btc.iterations, "accepted": res_btc.accepted,
                "walltime_seconds": res_btc.walltime_seconds,
            })
            if res_btc.abs_det > overall_best_det:
                overall_best_det = int(res_btc.abs_det)
                overall_best_ent = [list(r) for r in res_btc.matrix.entries]
                overall_best_label = f"btc/{label}/lam{lam}"
                overall_best_dist = dbtc

        # Also try one random-seed run per spectrum model at default lambda.
        print(f"     λ_anchor={lambda_anchor}, t≈{budget_per_model/2:.0f}s from random...")
        res_rand, drand = _run_anchored(
            rand_ent, target, n=n, sa_iterations=sa_iterations,
            sa_seconds=budget_per_model / 2, lambda_anchor=lambda_anchor,
            rng_seed=2028 + len(runs),
        )
        print(f"       -> |det|={res_rand.abs_det:,} ({res_rand.abs_det/det:.4f}), "
              f"dist={drand:.4f}, acc={res_rand.accepted}/{res_rand.iterations}")
        runs.append({
            "spectrum_model": label, "lambda": lambda_anchor, "seed": "random",
            "abs_det": int(res_rand.abs_det),
            "ratio_to_proven": res_rand.abs_det / det,
            "spectrum_distance": drand,
            "actual_sv_clusters": _summary_spectrum(res_rand.matrix.entries),
            "iterations": res_rand.iterations, "accepted": res_rand.accepted,
            "walltime_seconds": res_rand.walltime_seconds,
        })
        if res_rand.abs_det > overall_best_det:
            overall_best_det = int(res_rand.abs_det)
            overall_best_ent = [list(r) for r in res_rand.matrix.entries]
            overall_best_label = f"random/{label}/lam{lambda_anchor}"
            overall_best_dist = drand

    if overall_best_ent:
        out_path = RESULTS_DIR / "H19_targeted_spectrum_best.json"
        _save_matrix(out_path, n, overall_best_ent, {
            "abs_det": overall_best_det,
            "ratio_to_proven": overall_best_det / det,
            "spectrum_distance": overall_best_dist,
            "label": overall_best_label,
            "actual_sv_clusters": _summary_spectrum(overall_best_ent),
        })
        print(f"\n  Saved best to {out_path}: {overall_best_det:,} from {overall_best_label}")

    return {
        "n": n,
        "target_det": det,
        "btc_seed_det": btc_det,
        "rand_seed_det": rand_det,
        "spectrum_models_tried": [s[0] for s in spectra_to_try],
        "runs": runs,
        "best_abs_det": overall_best_det,
        "best_label": overall_best_label,
        "best_spectrum_distance": overall_best_dist,
        "best_ratio_to_proven": overall_best_det / det,
    }


# ---------------------------------------------------------------------------
# Experiment 3: n=23 multi-target sweep
# ---------------------------------------------------------------------------


def experiment_3_n23(
    *,
    target_dets: list[float] | None = None,
    sa_iterations: int = 60_000,
    btc_seconds: float = 60.0,
    sa_seconds_per_target: float = 240.0,
    lambda_anchor: float = 1.0,
) -> dict[str, Any]:
    print("\n" + "=" * 70)
    print("Experiment 3: n=23 multi-target sweep")
    print("=" * 70)
    n = 23
    if target_dets is None:
        target_dets = [2.5e15, 2.778e15, 3.0e15, 3.5e15, 3.9e15]

    # One BTC seed for all targets.
    btc_det, btc_ent = _btc_seed(n, seconds=btc_seconds, base_seed=20001)
    print(f"  BTC seed: |det|={btc_det:,}")

    per_target: list[dict[str, Any]] = []
    overall_best_det = 0
    overall_best_ent: list[list[int]] = []
    overall_best_label = ""
    overall_best_target = 0.0
    overall_best_dist = math.inf

    for target_det in target_dets:
        print(f"\n  Target |det| = {target_det:.4e}")
        try:
            target = predict_target_spectrum(
                n, target_det, model="symmetric_pair", ratio=1.566,
            )
        except Exception as e:
            print(f"    spectrum prediction failed: {e}")
            per_target.append({
                "target_det": target_det,
                "prediction_error": str(e),
            })
            continue
        print(f"    Target spec top3: {[round(float(t), 4) for t in target[:3]]}, "
              f"bot2: {[round(float(t), 4) for t in target[-2:]]}")

        seed_dist = spectrum_distance(btc_ent, target)
        # Two lambda values per target; allocate half the time to each.
        sub_runs: list[dict[str, Any]] = []
        for lam in [lambda_anchor, lambda_anchor * 0.2]:
            deadline = time.monotonic() + sa_seconds_per_target / 2
            result = sa_targeted_spectrum(
                Matrix(n=n, entries=btc_ent),
                target_spectrum=target,
                iterations=sa_iterations,
                lambda_anchor=lam,
                seed=(hash((target_det, lam)) & 0x7fffffff),
                deadline=deadline,
                rescore_every=2,
            )
            final_dist = float(getattr(result, "spectrum_distance", seed_dist))
            ratio = result.abs_det / target_det
            print(f"    λ={lam}: |det|={result.abs_det:,} ({ratio:.4f}·target), "
                  f"dist {seed_dist:.4f} -> {final_dist:.4f}, "
                  f"accepted={result.accepted}/{result.iterations} in {result.walltime_seconds:.1f}s")
            sub_runs.append({
                "lambda": lam,
                "abs_det": int(result.abs_det),
                "ratio_to_target": ratio,
                "spectrum_distance": final_dist,
                "actual_sv_clusters": _summary_spectrum(result.matrix.entries),
                "iterations": result.iterations,
                "accepted": result.accepted,
                "walltime_seconds": result.walltime_seconds,
            })
            if result.abs_det > overall_best_det:
                overall_best_det = int(result.abs_det)
                overall_best_ent = [list(r) for r in result.matrix.entries]
                overall_best_label = f"target_{target_det:.3e}/lam{lam}"
                overall_best_target = target_det
                overall_best_dist = final_dist

        per_target.append({
            "target_det": target_det,
            "target_spectrum_top3": [round(float(t), 4) for t in target[:3]],
            "target_spectrum_bot2": [round(float(t), 4) for t in target[-2:]],
            "seed_dist": seed_dist,
            "runs": sub_runs,
            "best_run_abs_det": max(r["abs_det"] for r in sub_runs),
            "best_run_dist": min(r["spectrum_distance"] for r in sub_runs),
        })

    if overall_best_ent:
        out_path = RESULTS_DIR / "H23_targeted_spectrum_best.json"
        _save_matrix(out_path, n, overall_best_ent, {
            "abs_det": overall_best_det,
            "best_target": overall_best_target,
            "spectrum_distance": overall_best_dist,
            "btc_seed_det": btc_det,
            "actual_sv_clusters": _summary_spectrum(overall_best_ent),
            "label": overall_best_label,
        })
        print(f"\n  Saved best to {out_path} ({overall_best_det:,})")

    return {
        "n": n,
        "btc_seed_det": btc_det,
        "targets": per_target,
        "best_abs_det": overall_best_det,
        "best_target_det": overall_best_target,
        "best_spectrum_distance": overall_best_dist,
        "btc_baseline_ratio_orrick": btc_det / 2.778e15,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n19-seconds", type=float, default=180.0,
                        help="Per-seed SA wall budget at n=19")
    parser.add_argument("--n23-seconds-per-target", type=float, default=240.0,
                        help="SA wall budget per target at n=23")
    parser.add_argument("--n23-btc-seconds", type=float, default=60.0,
                        help="BTC seed wall budget at n=23")
    parser.add_argument("--lambda-anchor", type=float, default=1.0,
                        help="Spectrum-distance penalty weight")
    parser.add_argument("--sa-iterations", type=int, default=80_000,
                        help="Per-restart SA iteration cap")
    parser.add_argument("--n23-targets", type=str, default="2.5e15,2.778e15,3.0e15,3.5e15,3.9e15",
                        help="Comma-separated list of target |det| values for H23")
    parser.add_argument("--skip-n17", action="store_true")
    parser.add_argument("--skip-n19", action="store_true")
    parser.add_argument("--skip-n23", action="store_true")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / "targeted_spectrum.json"
    out: dict[str, Any]
    if out_path.exists():
        try:
            out = json.loads(out_path.read_text())
        except Exception:
            out = {}
    else:
        out = {}
    out.setdefault("meta", {})
    out["meta"]["args"] = vars(args)

    t0 = time.monotonic()
    if not args.skip_n17:
        out["experiment_1_n17_sanity"] = experiment_1_n17_sanity()
    if not args.skip_n19:
        out["experiment_2_n19"] = experiment_2_n19(
            sa_iterations=args.sa_iterations,
            sa_seconds=args.n19_seconds,
            lambda_anchor=args.lambda_anchor,
        )
    if not args.skip_n23:
        targets = [float(t) for t in args.n23_targets.split(",") if t.strip()]
        out["experiment_3_n23"] = experiment_3_n23(
            target_dets=targets,
            sa_iterations=args.sa_iterations,
            btc_seconds=args.n23_btc_seconds,
            sa_seconds_per_target=args.n23_seconds_per_target,
            lambda_anchor=args.lambda_anchor,
        )
    out["meta"]["total_walltime_seconds"] = time.monotonic() - t0

    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n=== Saved diagnostic to {out_path} ===")
    print(f"Total walltime: {out['meta']['total_walltime_seconds']:.1f} s")


if __name__ == "__main__":
    main()

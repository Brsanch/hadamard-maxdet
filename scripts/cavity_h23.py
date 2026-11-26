"""Spin-glass cavity-method predictions for the H23 maxdet basin landscape.

Runs:
  - Approach B (empirical extreme-value extrapolation) on every catalogued
    basin |det| from the smart-2val + smart-3val + targeted-bridge campaigns
    + the basin atlas.
  - Approach A (replica-symmetric BP on the pairwise-proxy Hamiltonian) over
    a temperature grid, recording convergence behaviour.

Outputs:
  - results/cavity_h23.json — full numerical predictions + curves
  - results/cavity_h23.png  — basin-density curve (matplotlib if available)

Usage:
  python scripts/cavity_h23.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from hadamard_maxdet.cavity_prediction import (
    BARBA_BOUND_H23,
    H23_N,
    ORRICK_ABS_DET_H23,
    basin_count_prediction,
    empirical_basin_density_extrapolation,
    extract_basin_dets_from_runs,
    rs_bp_marginals,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"


def _summarise_dets(dets: list[int]) -> dict:
    arr = np.array(dets, dtype=np.float64)
    return {
        "n": int(len(arr)),
        "max": int(arr.max()),
        "min": int(arr.min()),
        "median": float(np.median(arr)),
        "mean": float(np.mean(arr)),
        "log_mean": float(np.mean(np.log(arr))),
        "log_std": float(np.std(np.log(arr), ddof=1)) if len(arr) > 1 else 0.0,
        "n_at_or_above_orrick": int(np.sum(arr >= ORRICK_ABS_DET_H23)),
        "max_ratio_to_orrick": float(arr.max() / ORRICK_ABS_DET_H23),
    }


def main() -> None:
    print(f"Loading catalogued H23 basins from {RESULTS} ...")
    dets = extract_basin_dets_from_runs(
        smart_2val_path=RESULTS / "smart_2val_search.json",
        smart_3val_summary_path=RESULTS / "smart_3val_search_summary.json",
        targeted_bridge_path=RESULTS / "targeted_bridge_h23.json",
        basin_atlas_path=RESULTS / "basin_atlas.json",
    )
    summary = _summarise_dets(dets)
    print(f"  loaded n={summary['n']} catalogued |det|s")
    print(f"  max = {summary['max']:,}  ({summary['max_ratio_to_orrick']:.4f} · Orrick)")
    print(f"  observed at-or-above Orrick: {summary['n_at_or_above_orrick']}")
    print(f"  log|det| mean = {summary['log_mean']:.4f}  std = {summary['log_std']:.4f}")

    # ------------------------------------------------------------------
    # Approach B — empirical extrapolation at multiple Barba-relative levels
    # ------------------------------------------------------------------
    barba_ratios = [0.70, 0.7091, 0.72, 0.75, 0.80, 0.90]
    targets_named: list[tuple[str, int, float]] = []
    for r in barba_ratios:
        target = int(r * BARBA_BOUND_H23)
        ratio_orrick = target / ORRICK_ABS_DET_H23
        name = f"{r:.4f}·Barba (= {ratio_orrick:.4f}·Orrick)"
        targets_named.append((name, target, ratio_orrick))

    # Also probe the 1.05·Orrick level explicitly (5% above record).
    above_orrick = [
        ("1.00·Orrick (= record)", ORRICK_ABS_DET_H23, 1.0),
        ("1.01·Orrick", int(1.01 * ORRICK_ABS_DET_H23), 1.01),
        ("1.05·Orrick", int(1.05 * ORRICK_ABS_DET_H23), 1.05),
        ("1.10·Orrick", int(1.10 * ORRICK_ABS_DET_H23), 1.10),
        ("Barba bound", BARBA_BOUND_H23, BARBA_BOUND_H23 / ORRICK_ABS_DET_H23),
    ]
    all_targets = targets_named + above_orrick

    print(
        "\n=== Approach B — empirical GEV / log-linear tail extrapolation ===\n"
    )
    print("(Bounded GEV: hard right endpoint enforced at log(Barba bound).)")
    print(
        f"  {'level':<40s} {'target |det|':>22s} {'E[# basins ≥]':>14s} "
        f"{'CI95':>22s} {'method':>16s}"
    )
    print("  " + "-" * 116)
    extrap_results = []
    upper_bound_log = math.log(BARBA_BOUND_H23)
    for name, target, ratio_orrick in all_targets:
        out = empirical_basin_density_extrapolation(
            dets,
            target_abs_det=target,
            n_bootstrap=400,
            bootstrap_seed=0,
            tail_fraction=0.20,
            upper_bound_log=upper_bound_log,
        )
        ci_str = f"[{out.expected_count_ci_low:.2g}, {out.expected_count_ci_high:.2g}]"
        print(
            f"  {name:<40s} {target:>22,d} {out.expected_count_at_or_above:>14.4g} "
            f"{ci_str:>22s} {out.method:>16s}"
        )
        extrap_results.append({
            "level_name": name,
            "target_abs_det": int(target),
            "target_ratio_to_orrick": float(ratio_orrick),
            "expected_count_at_or_above": out.expected_count_at_or_above,
            "expected_count_ci_low": out.expected_count_ci_low,
            "expected_count_ci_high": out.expected_count_ci_high,
            "method": out.method,
            "fit_params": out.fit_params,
            "n_basins_input": out.n_basins_input,
            "n_observed_at_or_above_orrick": out.n_observed_at_or_above_orrick,
        })

    # Also collect the unbounded (heavy-tail) prediction for the same targets
    # — useful as a sanity contrast: an unbounded GPD on this dataset gives a
    # heavy-tail estimate which is the upper-end "what if Barba is loose"
    # prediction.
    print("\n  (For contrast: same fits with NO upper-bound constraint.)")
    print(
        f"  {'level':<40s} {'target |det|':>22s} {'E[# basins ≥]':>14s} "
        f"{'CI95':>22s} {'method':>16s}"
    )
    print("  " + "-" * 116)
    extrap_unbounded = []
    for name, target, ratio_orrick in all_targets:
        out = empirical_basin_density_extrapolation(
            dets,
            target_abs_det=target,
            n_bootstrap=400,
            bootstrap_seed=0,
            tail_fraction=0.20,
        )
        ci_str = f"[{out.expected_count_ci_low:.2g}, {out.expected_count_ci_high:.2g}]"
        print(
            f"  {name:<40s} {target:>22,d} {out.expected_count_at_or_above:>14.4g} "
            f"{ci_str:>22s} {out.method:>16s}"
        )
        extrap_unbounded.append({
            "level_name": name,
            "target_abs_det": int(target),
            "target_ratio_to_orrick": float(ratio_orrick),
            "expected_count_at_or_above": out.expected_count_at_or_above,
            "expected_count_ci_low": out.expected_count_ci_low,
            "expected_count_ci_high": out.expected_count_ci_high,
            "method": out.method,
            "fit_params": out.fit_params,
        })

    # ------------------------------------------------------------------
    # Approach A — RS-BP on pairwise-proxy Hamiltonian over a β grid
    # ------------------------------------------------------------------
    print("\n=== Approach A — RS-BP on pairwise-proxy Hamiltonian (n=23) ===\n")

    bp_betas = list(np.geomspace(0.005, 5.0, 16))
    print(f"  β grid: {len(bp_betas)} values from {bp_betas[0]:.4g} to {bp_betas[-1]:.4g}")
    bp_curve = []
    for beta in bp_betas:
        res = rs_bp_marginals(
            n=H23_N,
            beta=float(beta),
            coupling=1.0,
            max_iter=120,
            damping=0.3,
            tol=1e-5,
            seed=0,
        )
        # Diagnostic: variance of marginals from 0.5 = magnetisation magnitude.
        m_dev = float(np.mean(np.abs(res.marginals - 0.5)))
        bp_curve.append({
            "beta": float(beta),
            "converged": bool(res.converged),
            "iterations": int(res.iterations),
            "residual": float(res.final_residual) if math.isfinite(res.final_residual) else None,
            "free_energy": float(res.free_energy),
            "marginal_dev_from_half": m_dev,
        })

    n_converged = sum(1 for c in bp_curve if c["converged"])
    print(f"  RS-BP converged at {n_converged} / {len(bp_curve)} β values")
    print(
        f"  {'β':>10s} {'converged':>10s} {'iters':>6s} {'residual':>14s} "
        f"{'F_Bethe':>14s} {'<|m-1/2|>':>10s}"
    )
    for c in bp_curve:
        res_str = f"{c['residual']:.3g}" if c["residual"] is not None else "inf"
        print(
            f"  {c['beta']:>10.4g} {str(c['converged']):>10s} {c['iterations']:>6d} "
            f"{res_str:>14s} {c['free_energy']:>14.4g} "
            f"{c['marginal_dev_from_half']:>10.4g}"
        )

    bp_predict_proxy = basin_count_prediction(
        n=H23_N,
        # Proxy energies: H_proxy values that bracket "near the floor" to
        # "fully disordered". For n=23, n*(n-1)/2 = 253 row pairs; each
        # squared Gram entry ≤ n² = 529; an ABSOLUTE upper bound on H_proxy
        # is 253 * 529 ≈ 1.34e5. The Barba/Orrick neighbourhood corresponds
        # to small but positive H_proxy.
        energy_levels=[0.0, 5.0, 50.0, 500.0, 5000.0, 50000.0],
        beta_grid=bp_betas,
        coupling=1.0,
        bp_seed=0,
    )

    # ------------------------------------------------------------------
    # Persist + plot.
    # ------------------------------------------------------------------
    out_path = RESULTS / "cavity_h23.json"
    out_data = {
        "n": H23_N,
        "orrick_abs_det": ORRICK_ABS_DET_H23,
        "barba_bound": BARBA_BOUND_H23,
        "input_basin_summary": summary,
        "approach_B_empirical_extrapolation": extrap_results,
        "approach_B_empirical_extrapolation_unbounded": extrap_unbounded,
        "approach_A_rs_bp_curve": bp_curve,
        "approach_A_bp_proxy_predictions": bp_predict_proxy["predictions"],
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, default=float)
    print(f"\nWrote {out_path}")

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: empirical |det| histogram (log space) with extrapolation curve.
        ax = axes[0]
        log_dets = np.log(np.array(dets, dtype=np.float64))
        ax.hist(log_dets, bins=50, alpha=0.6, color="steelblue", label="catalogued basins")
        ax.axvline(math.log(ORRICK_ABS_DET_H23), color="red", linestyle="--",
                   label=f"Orrick (log = {math.log(ORRICK_ABS_DET_H23):.3f})")
        ax.axvline(math.log(BARBA_BOUND_H23), color="black", linestyle=":",
                   label=f"Barba bound")
        ax.set_xlabel("log |det|")
        ax.set_ylabel("# basins")
        ax.set_title(f"H23 catalogued basin density (n={summary['n']})")
        ax.legend(loc="upper left", fontsize=8)

        # Right: extrapolated expected basin count vs ratio-to-Orrick.
        ax = axes[1]
        ratios = [e["target_ratio_to_orrick"] for e in extrap_results]
        counts = [e["expected_count_at_or_above"] for e in extrap_results]
        ci_lo = [e["expected_count_ci_low"] for e in extrap_results]
        ci_hi = [e["expected_count_ci_high"] for e in extrap_results]
        order = np.argsort(ratios)
        ratios = np.array(ratios)[order]
        counts = np.array(counts)[order]
        ci_lo = np.array(ci_lo)[order]
        ci_hi = np.array(ci_hi)[order]
        ax.errorbar(
            ratios, np.maximum(counts, 1e-4),
            yerr=[
                np.maximum(counts - ci_lo, 0),
                np.maximum(ci_hi - counts, 0),
            ],
            fmt="o-", capsize=3, color="darkorange",
            label="GEV / log-linear extrapolation",
        )
        ax.axvline(1.0, color="red", linestyle="--", alpha=0.7,
                   label="1.0 · Orrick (record)")
        ax.axhline(1.0, color="grey", linestyle=":", alpha=0.7,
                   label="1 basin")
        ax.set_yscale("log")
        ax.set_xlabel("|det| / Orrick |det|")
        ax.set_ylabel("E[ # basins with |det| ≥ target ]")
        ax.set_title("Cavity / GEV-tail prediction at H23")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        png_path = RESULTS / "cavity_h23.png"
        plt.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"Wrote {png_path}")
    except Exception as exc:
        print(f"(plot skipped: {exc!r})")

    # Headline take-aways for the operator.
    print("\n=== Headline ===")
    extrap_by_ratio = {round(e["target_ratio_to_orrick"], 4): e for e in extrap_results}

    # Predicted at Orrick (= 1.0).
    e_at = extrap_by_ratio.get(1.0)
    if e_at:
        print(
            f"  Predicted basins at |det| ≥ Orrick: "
            f"{e_at['expected_count_at_or_above']:.4g}  "
            f"CI95 [{e_at['expected_count_ci_low']:.3g}, "
            f"{e_at['expected_count_ci_high']:.3g}]"
        )
    e_above = extrap_by_ratio.get(1.05)
    if e_above:
        print(
            f"  Predicted basins at |det| ≥ 1.05·Orrick: "
            f"{e_above['expected_count_at_or_above']:.4g}  "
            f"CI95 [{e_above['expected_count_ci_low']:.3g}, "
            f"{e_above['expected_count_ci_high']:.3g}]"
        )
    e_at_b = extrap_by_ratio.get(round(BARBA_BOUND_H23 / ORRICK_ABS_DET_H23, 4))
    if e_at_b:
        print(
            f"  Predicted basins at |det| ≥ Barba bound: "
            f"{e_at_b['expected_count_at_or_above']:.4g}  "
            f"CI95 [{e_at_b['expected_count_ci_low']:.3g}, "
            f"{e_at_b['expected_count_ci_high']:.3g}]"
        )
    print(
        f"  RS-BP convergence: {n_converged}/{len(bp_curve)} β values "
        f"({'mostly converged' if n_converged > len(bp_curve)/2 else 'mostly DIVERGED — glassy/RSB regime'})."
    )


if __name__ == "__main__":
    main()

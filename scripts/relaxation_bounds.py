"""Compare continuous-relaxation upper bounds against classical bounds and
in-repo best lower bounds, for a range of orders.

Prints a table of:
    n, residue (mod 4), Barba/EW/Ehlich, trivial-SDP (= Hadamard),
    spectral-SDP (= Barba/Ehlich), proven-max (when known), in-repo best,
    ratio (in-repo best / spectral-SDP).

The headline question: at n=23 (smallest open case), how much room is left
between our local-search lower bound (0.6704·Barba ≈ 2.626e15) and the
sharpest convex upper bound we can compute?

Run:
    python scripts/relaxation_bounds.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# Make the package importable when run from the repo root.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

from hadamard_maxdet.orders import BENCHMARKS, record_pm1
from hadamard_maxdet.relaxation import sdp_upper_bound
from hadamard_maxdet.verifier import upper_bound


# In-repo best |det| values (from results/, OEIS, or earlier searches).
# For proven orders we use record_pm1 directly (it IS the best).
# For open orders we record the best |det| found by any in-repo search.
IN_REPO_BEST: dict[int, int] = {
    23: 2_626_567_700_217_856,  # BTC bordered-two-circulant SA, branch class-tabu.
    27: 0,  # not yet attempted in-repo
    29: 0,
    33: 0,
    37: 0,
}


def best_known(n: int) -> int | None:
    """Return the largest |det| we know about for order n.

    Uses ``record_pm1(n)`` (proven max from OEIS A003432·doubling) for n ≤ 21,
    falls back to ``IN_REPO_BEST[n]`` for the open orders.
    """
    if n <= 22:
        try:
            return record_pm1(n)
        except ValueError:
            return None
    return IN_REPO_BEST.get(n)


def proven_max(n: int) -> int | None:
    """Return the proven max for n if known (n ≤ 22), else None."""
    if n <= 22:
        try:
            return record_pm1(n)
        except ValueError:
            return None
    return None


def fmt_big(x: float | int | None) -> str:
    if x is None:
        return "—"
    if isinstance(x, int) and x < 10 ** 9:
        return f"{x:,}"
    return f"{float(x):.4e}"


def main() -> None:
    orders = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
              19, 20, 21, 22, 23, 27, 29, 33, 37]

    rows: list[dict] = []
    for n in orders:
        residue = n % 4
        classical = upper_bound(n)
        trivial = sdp_upper_bound(n, cuts="trivial").abs_det_bound
        spectral = sdp_upper_bound(n, cuts="spectral").abs_det_bound
        pm = proven_max(n)
        bk = best_known(n)
        ratio_to_spectral = (bk / spectral) if (bk and spectral) else None

        rows.append({
            "n": n,
            "residue": residue,
            "classical": classical,
            "trivial_sdp": trivial,
            "spectral_sdp": spectral,
            "proven_max": pm,
            "best_known": bk,
            "ratio_best_over_spectral": ratio_to_spectral,
        })

    # Print a wide formatted table.
    print()
    print("Continuous relaxation bounds for the Hadamard ±1 maxdet problem")
    print("=" * 132)
    header = (
        f"{'n':>3}  {'r':>1}  {'classical (Barba/EW/Ehlich)':>28}  "
        f"{'trivial-SDP (Hadamard)':>22}  {'spectral-SDP':>14}  "
        f"{'proven-max':>14}  {'best-known':>14}  {'best/spec':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['n']:>3}  {row['residue']:>1}  "
            f"{fmt_big(row['classical']):>28}  "
            f"{fmt_big(row['trivial_sdp']):>22}  "
            f"{fmt_big(row['spectral_sdp']):>14}  "
            f"{fmt_big(row['proven_max']):>14}  "
            f"{fmt_big(row['best_known']):>14}  "
            f"{(f'{row['ratio_best_over_spectral']:.4f}' if row['ratio_best_over_spectral'] else '—'):>9}"
        )
    print("=" * 132)
    print()

    # Headline: focus on n=23.
    h23 = next(r for r in rows if r["n"] == 23)
    print("HEADLINE — n = 23 (smallest open case):")
    print(f"  Barba upper bound:                {h23['classical']:,}  (sqrt 2n-1 · (n-1)^((n-1)/2))")
    print(f"  Trivial-SDP (= Hadamard):         {h23['trivial_sdp']:.6e}  ({h23['trivial_sdp']/h23['classical']:.4f}·Barba)")
    print(f"  Spectral-SDP (= Barba):           {h23['spectral_sdp']:.6e}  ({h23['spectral_sdp']/h23['classical']:.4f}·Barba)")
    print(f"  Orrick lower bound (literature):  ~2.778e15                ({2.778e15/h23['classical']:.4f}·Barba)")
    print(f"  In-repo best (BTC, branch class-tabu): {h23['best_known']:,}  "
          f"({h23['best_known']/h23['classical']:.4f}·Barba)")
    print()
    print("Interpretation:")
    print("  - The trivial Hadamard relaxation is *looser* than Barba (1.166·Barba).")
    print("  - The spectral relaxation reproduces Barba exactly; **no purely convex SDP")
    print("    we tried tightens below Barba** (because the integer constraint")
    print("    'G_ij ≡ n (mod 2), |G_ij| ≥ 1' is non-convex).")
    print("  - Therefore, our 0.6704·Barba in-repo result is also 0.6704 of the")
    print("    sharpest convex upper bound. The relaxation gap is NOT informative.")
    print("  - Closing the [Orrick lower 0.7091, Barba 1.0] interval requires either")
    print("    (a) a stronger non-convex bound (e.g. integer / parity LP), or")
    print("    (b) a better lower-bound construction matching Orrick's 0.7091.")
    print()

    # Persist the table as JSON for downstream use.
    out = HERE.parent / "results" / "relaxation_bounds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    serialisable = []
    for row in rows:
        serialisable.append({
            "n": row["n"],
            "residue_mod_4": row["residue"],
            "classical_upper_bound": int(row["classical"]),
            "trivial_sdp_upper_bound": float(row["trivial_sdp"]),
            "spectral_sdp_upper_bound": float(row["spectral_sdp"]),
            "proven_max": (int(row["proven_max"]) if row["proven_max"] is not None else None),
            "best_known_in_repo": (int(row["best_known"]) if row["best_known"] else None),
            "ratio_best_over_spectral": row["ratio_best_over_spectral"],
        })
    with open(out, "w") as f:
        json.dump({
            "experiment": "relaxation_bounds",
            "method": "convex SDP relaxation: max log det(G) s.t. G PSD, G_ii=n, "
                      "with optional |G_ij| caps (trivial / spectral)",
            "rows": serialisable,
            "headline_n23": {
                "barba_upper_bound": int(h23["classical"]),
                "trivial_hadamard": float(h23["trivial_sdp"]),
                "spectral_barba": float(h23["spectral_sdp"]),
                "in_repo_best": int(h23["best_known"]),
                "in_repo_best_over_spectral": h23["ratio_best_over_spectral"],
                "in_repo_best_over_barba": h23["best_known"] / h23["classical"],
            },
            "verdict": (
                "Convex relaxation does NOT tighten below Barba/Ehlich. The "
                "integer constraint G_ij ∈ {odd integers in [-n,n]} is "
                "non-convex; the magnitude-cap SDP returns Hadamard (looser "
                "than Barba), and the spectral relaxation reproduces Barba "
                "exactly. No new framing of the in-repo 0.6704·Barba result."
            ),
        }, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

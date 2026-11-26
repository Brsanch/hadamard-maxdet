"""Generate publication-grade vector figures for the Hadamard maximal determinant paper.

Outputs:
  paper/figures/fig1_r2_unconditional_bounds.pdf
  paper/figures/fig2_algebraic_feasibility_cascade.pdf
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from hadamard_maxdet.feasibility import prove_r2_exceptional_unconditional_ub

OUT = REPO / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9.5,
    "axes.titlesize": 10.2,
    "axes.labelsize": 9.2,
    "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def make_fig1() -> None:
    orders = [22, 34, 58, 70, 78, 94]
    ew_bound = [2 * n - 2 for n in orders]
    ub_2mom = []
    for n in orders:
        cert = prove_r2_exceptional_unconditional_ub(n)
        assert cert.verified
        ub_2mom.append(cert.threshold_mult)

    gaps_2mom = [e - u for e, u in zip(ew_bound, ub_2mom)]
    # Full integer-step improvement at n=34 via 15-dual Cauchy-interlaced Seidel B&B
    gap_34_bb = 66.0 - 65.0

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.8, 4.15), dpi=300,
        gridspec_kw={"width_ratios": [1.0, 1.12]},
    )

    # Panel (a): Margin strictly below Ehlich–Wojtas bound (2n - 2)
    x = np.arange(len(orders))
    w = 0.54
    ax1.bar(
        x, gaps_2mom, width=w, color="#2B5C8F", edgecolor="#16324F", linewidth=0.8,
        label="Seidel-gauge + PSD trace (Theorem 2)",
    )
    ax1.bar(
        [1], [gap_34_bb - gaps_2mom[1]], bottom=[gaps_2mom[1]], width=w,
        color="#D95F02", edgecolor="#7F3600", linewidth=0.8,
        label="15-dual Cauchy-interlaced B&B (Theorem 3)",
    )
    ax1.axhline(1.0, color="#7F3600", linestyle="--", linewidth=0.95, alpha=0.8)
    ax1.text(
        5.35, 1.03, "1 full integer step",
        ha="right", va="bottom", fontsize=7.8, color="#7F3600",
    )

    for i, g in enumerate(gaps_2mom):
        if i == 1:
            ax1.text(
                i, gap_34_bb + 0.035, "1.000\n($<65$)",
                ha="center", va="bottom", fontsize=7.9, fontweight="bold", color="#9E3B00",
            )
        else:
            ax1.text(
                i, g + 0.025, f"{g:.3f}",
                ha="center", va="bottom", fontsize=7.8, color="#16324F",
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"${n}$\n$({e})$" for n, e in zip(orders, ew_bound)])
    ax1.set_xlabel("Order $n$ (Ehlich\u2013Wojtas bound $2n - 2$)")
    ax1.set_ylabel("Certified drop below $2n - 2$")
    ax1.set_ylim(0, 1.54)
    ax1.set_xlim(-0.55, 5.55)
    ax1.set_title("(a) Drop below Ehlich\u2013Wojtas bound ($n \\equiv 2\\;(\\mathrm{mod}\\;4)$)")
    ax1.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel (b): Complete n=34 elimination cascade relative to 65 * 32^16
    labels = [
        "Ehlich\u2013Wojtas $(17,17)\\,\\emptyset\\oplus\\emptyset$\n(Hasse $p=3,11$)",
        "$(17,17)\\,K_2\\oplus K_2$\n(Thm. 1: 0 col. sols.)",
        "$(13,21)\\,\\emptyset\\oplus K_2$\n(Col. eq.: $G_{14,15}=34$)",
        "$(17,17)\\,P_3\\oplus P_3$\n(Col. eq.: $G_{2,3}+G_{19,20}=0$)",
        "$(17,17)\\,2K_2\\oplus 2K_2$\n(Hasse $p=7,47$)",
        "$k=2$ cross $(+4,-4)$\n(Col. eq.: $G_{1,18}=-34$)",
        "Partition $s_1 \\leq 11$\n(2-moment spectral)",
        "Cross $k \\geq 3$\n(PSD trace Schur)",
        "Large entry $m_6 = 2$ ($P_3$)\n(Seidel-gauge capped)",
        "Cross $|C_{ij}| \\geq 8$\n(PSD trace Schur)",
    ]
    vals = [66.000, 65.531, 65.250, 65.125, 65.061, 65.031, 64.900, 64.747, 64.680, 64.091]
    colors = ["#C0392B"] * 6 + ["#2B5C8F"] * 4
    y = np.arange(len(labels))[::-1]
    ax2.barh(
        y, [v - 63.5 for v in vals], left=63.5, height=0.66, color=colors,
        edgecolor="#222222", linewidth=0.6, alpha=0.90,
    )
    ax2.axvline(65.0, color="#1B7837", linestyle="--", linewidth=1.3, label="Target bound $65 \\cdot 32^{16}$")
    for yi, v in zip(y, vals):
        if v >= 65.0:
            ax2.text(
                v + 0.045, yi, f"{v:.3f}", va="center", ha="left", fontsize=7.5,
                color="#7B1D1D", fontweight="bold",
            )
        else:
            ax2.text(
                v - 0.045, yi, f"{v:.3f}", va="center", ha="right", fontsize=7.5,
                color="white", fontweight="bold",
            )
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels, fontsize=7.3)
    ax2.set_xlim(63.5, 66.65)
    ax2.set_xlabel("Determinant multiplier $c = |\\det M| / 32^{16}$")
    ax2.set_title("(b) Complete $n = 34$ closure below $65 \\cdot 32^{16}$")
    ax2.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
    ax2.grid(axis="x", linestyle=":", alpha=0.45)

    plt.tight_layout(w_pad=1.8)
    fig.savefig(OUT / "fig1_r2_unconditional_bounds.pdf", bbox_inches="tight")
    plt.close(fig)


def make_fig2() -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 3.85), dpi=300)

    # Panel (a): n = 29 exact elimination funnel
    stages_29 = [
        "Low-trace\nsignatures",
        "Square-det\nplacements",
        "In gap\n$[320,329)$",
        "After\nHasse\u2013Mink.",
        "After\n$x^\\top G^{-1}x=1$",
        "After int.\nFarkas",
    ]
    counts_29 = [175, 44, 11, 7, 2, 0]
    cols_29 = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3", "#1B7837"]
    x1 = np.arange(len(stages_29))
    ax1.bar(x1, counts_29, width=0.56, color=cols_29, edgecolor="#222222", linewidth=0.7)
    for xi, c in zip(x1, counts_29):
        if c > 0:
            ax1.text(xi, c + 4.0, str(c), ha="center", va="bottom", fontsize=8.8, fontweight="bold")
        else:
            ax1.text(
                xi, 6.0, "0\nsurvivors", ha="center", va="bottom",
                fontsize=8.0, fontweight="bold", color="#1B7837",
                bbox=dict(boxstyle="round,pad=0.22", facecolor="#EDF7ED", edgecolor="#1B7837", linewidth=0.8),
            )
    ax1.set_xticks(x1)
    ax1.set_xticklabels(stages_29, fontsize=7.6)
    ax1.set_ylabel("Surviving candidates")
    ax1.set_ylim(0, 202)
    ax1.set_xlim(-0.55, 5.60)
    ax1.set_title("(a) $n = 29$ spread-placement elimination")
    ax1.grid(axis="y", linestyle=":", alpha=0.45)

    # Panel (b): n = 33 clique-Gram exclusion funnel (honest log scale, NO fake bar for 0)
    stages_33 = [
        "Configs\n$\\geq 441\\cdot 2^{74}$",
        "Square\n$\\det G$",
        "After\nHasse\u2013Mink.",
        "After\n$x^\\top G^{-1}x=1$",
        "After moment\nsystem",
    ]
    counts_33 = [9220, 24, 7, 2, 0]
    cols_33 = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
    x2 = np.arange(len(stages_33))
    # Plot ONLY strictly positive counts as bars on the log axis
    ax2.bar(x2[:4], counts_33[:4], width=0.54, color=cols_33, edgecolor="#222222", linewidth=0.7)
    ax2.set_yscale("log")
    ax2.set_ylim(0.5, 5.5e4)
    ax2.set_xlim(-0.55, 4.60)
    ax2.set_yticks([1, 10, 100, 1000, 10000])

    for xi, c in zip(x2[:4], counts_33[:4]):
        ax2.text(xi, c * 1.32, f"{c:,}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # Annotate exact zero honestly at the axis baseline with NO misleading bar
    ax2.text(
        4.0, 0.78, "0\nsurvivors",
        ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#1B7837",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#EDF7ED", edgecolor="#1B7837", linewidth=0.8),
    )

    ax2.set_xticks(x2)
    ax2.set_xticklabels(stages_33, fontsize=7.6)
    ax2.set_ylabel("Surviving configurations (log scale)")
    ax2.set_title("(b) $n = 33$ clique-Gram exclusion ($|\\det M| \\geq 441 \\cdot 2^{74}$)")
    ax2.grid(axis="y", linestyle=":", alpha=0.45)

    plt.tight_layout(w_pad=1.8)
    fig.savefig(OUT / "fig2_algebraic_feasibility_cascade.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    make_fig1()
    make_fig2()
    print("Wrote fig1_r2_unconditional_bounds.pdf and fig2_algebraic_feasibility_cascade.pdf")

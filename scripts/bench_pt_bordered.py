"""Head-to-head bench: plain bordered SA vs replica-exchange-on-bordered.

Round 1 found two best ingredients on H19:

  * bordered-circulant ansatz collapses 2^(n^2) → 2^(n-1)
    (262,144 cores at n=19) and beats unstructured SA by +44%.
  * 30-replica parallel tempering on the unstructured space beats
    naive multistart by +39%.

This script composes them: 30-replica PT directly over the (n-1)-bit
bordered core. For n=19 the ansatz only has 262,144 distinct cores, so
ladder-mediated information-sharing has very little room to add value
beyond what cold-replica SA already does — but we measure rather than
guess.

Both methods get the same wall-clock budget. Each method is restarted
with a fresh seed whenever its inner call returns, so the reported
``starts × iters`` reflects how many independent attempts the budget
afforded. Best-ever |det| across all attempts is reported.

All |det| values are exact integers via Bareiss. No float det is used.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hadamard_maxdet import BENCHMARKS, verify
from hadamard_maxdet.parallel import replica_exchange_bordered_circulant
from hadamard_maxdet.structured import sa_bordered_circulant


def run_plain_bordered(
    n: int,
    budget_s: float,
    *,
    base_seed: int = 0,
    iters_per_start: int = 20_000,
) -> dict:
    """Repeatedly launch single-start bordered-circulant SA until budget elapses."""
    start = time.monotonic()
    best_abs = 0
    best_entries: list[list[int]] | None = None
    starts_done = 0
    while time.monotonic() - start < budget_s:
        r = sa_bordered_circulant(
            n,
            iterations=iters_per_start,
            seed=base_seed + starts_done,
        )
        starts_done += 1
        if r.abs_det > best_abs:
            best_abs = r.abs_det
            best_entries = [row[:] for row in r.matrix.entries]
    elapsed = time.monotonic() - start
    return {
        "method": "plain_bordered_sa",
        "abs_det": best_abs,
        "elapsed_s": elapsed,
        "starts_done": starts_done,
        "iters_per_start": iters_per_start,
        "total_proposals": starts_done * iters_per_start,
        "entries": best_entries,
    }


def run_pt_bordered(
    n: int,
    budget_s: float,
    *,
    base_seed: int = 0,
    n_replicas: int = 30,
    iters_per_replica_per_round: int = 600,
    exchange_every: int = 100,
    t_cold: float = 0.005,
    t_hot: float = 5.0,
) -> dict:
    """Repeatedly launch a 30-replica PT-on-cores run until budget elapses.

    Each round runs ``iters_per_replica_per_round`` Metropolis steps per
    replica = ``n_replicas × iters`` proposals. With ``n=19`` and the
    default 30 replicas × 600 steps that's 18,000 proposals per round
    (the 262,144-core ansatz is sweep-sized after a handful of rounds).
    """
    start = time.monotonic()
    best_abs = 0
    best_entries: list[list[int]] | None = None
    rounds_done = 0
    last_swap_rates: list[float] | None = None
    last_move_rates: list[float] | None = None
    while time.monotonic() - start < budget_s:
        r = replica_exchange_bordered_circulant(
            n,
            n_replicas=n_replicas,
            iterations=iters_per_replica_per_round,
            exchange_every=exchange_every,
            seed=base_seed + rounds_done,
            t_cold=t_cold,
            t_hot=t_hot,
        )
        rounds_done += 1
        if r.abs_det > best_abs:
            best_abs = r.abs_det
            best_entries = [row[:] for row in r.matrix.entries]
        last_swap_rates = r.swap_rates()
        last_move_rates = [
            (a / max(t, 1)) for a, t in zip(r.move_accepts, r.move_attempts)
        ]
    elapsed = time.monotonic() - start
    return {
        "method": "pt_bordered",
        "abs_det": best_abs,
        "elapsed_s": elapsed,
        "rounds_done": rounds_done,
        "n_replicas": n_replicas,
        "iters_per_replica_per_round": iters_per_replica_per_round,
        "exchange_every": exchange_every,
        "total_proposals": rounds_done * n_replicas * iters_per_replica_per_round,
        "last_swap_rates": last_swap_rates,
        "last_move_rates": last_move_rates,
        "entries": best_entries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", default="H19", choices=list(BENCHMARKS))
    parser.add_argument("--budget", type=float, default=45.0,
                        help="wall-clock budget per method, in seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-replicas", type=int, default=30)
    parser.add_argument("--iters-per-replica", type=int, default=600)
    parser.add_argument("--exchange-every", type=int, default=100)
    parser.add_argument("--iters-per-start", type=int, default=20_000)
    parser.add_argument("--t-cold", type=float, default=0.005)
    parser.add_argument("--t-hot", type=float, default=5.0)
    parser.add_argument("--output", type=Path,
                        default=Path("results/bench_pt_bordered.json"))
    args = parser.parse_args()

    order = BENCHMARKS[args.order]
    n = order.n
    print(f"Bench: plain bordered SA vs PT-on-bordered-core, {order.id} (n={n}).")
    print(f"Per-method wall-clock budget: {args.budget}s.\n")

    print("=== plain bordered-circulant SA (multi-start) ===")
    res_a = run_plain_bordered(
        n,
        args.budget,
        base_seed=args.seed,
        iters_per_start=args.iters_per_start,
    )
    print(f"  best |det| = {res_a['abs_det']}")
    print(f"  starts = {res_a['starts_done']}, iters/start = "
          f"{res_a['iters_per_start']}, total proposals = "
          f"{res_a['total_proposals']:,}")
    print(f"  elapsed = {res_a['elapsed_s']:.2f}s")
    if res_a["entries"] is not None:
        v = verify(res_a["entries"])
        assert v.valid and v.abs_det == res_a["abs_det"]

    print("\n=== PT-bordered-circulant (30-replica) ===")
    res_b = run_pt_bordered(
        n,
        args.budget,
        base_seed=args.seed + 100_000,
        n_replicas=args.n_replicas,
        iters_per_replica_per_round=args.iters_per_replica,
        exchange_every=args.exchange_every,
        t_cold=args.t_cold,
        t_hot=args.t_hot,
    )
    print(f"  best |det| = {res_b['abs_det']}")
    print(f"  rounds = {res_b['rounds_done']}, replicas = "
          f"{res_b['n_replicas']}, iters/replica/round = "
          f"{res_b['iters_per_replica_per_round']}, total proposals = "
          f"{res_b['total_proposals']:,}")
    print(f"  exchange_every = {res_b['exchange_every']}, "
          f"t_cold = {args.t_cold}, t_hot = {args.t_hot}")
    if res_b["last_swap_rates"]:
        rates = res_b["last_swap_rates"]
        print(f"  last-round swap acceptance: min={min(rates):.2f}, "
              f"max={max(rates):.2f}, mean={sum(rates) / len(rates):.2f}")
    if res_b["last_move_rates"]:
        m = res_b["last_move_rates"]
        print(f"  last-round move acceptance (cold→hot): "
              f"{m[0]:.2f} ... {m[-1]:.2f}")
    print(f"  elapsed = {res_b['elapsed_s']:.2f}s")
    if res_b["entries"] is not None:
        v = verify(res_b["entries"])
        assert v.valid and v.abs_det == res_b["abs_det"]

    print("\n=== Summary ===")
    print(f"  plain_bordered_sa  |det| = {res_a['abs_det']}")
    print(f"  pt_bordered        |det| = {res_b['abs_det']}")
    if res_a["abs_det"] > 0:
        ratio = res_b["abs_det"] / res_a["abs_det"]
        print(f"  PT / plain         = {ratio:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "order": order.id,
        "n": n,
        "budget_s": args.budget,
        "seed": args.seed,
        "results": [
            {k: v for k, v in r.items() if k != "entries"}
            for r in (res_a, res_b)
        ],
    }
    best = max((res_a, res_b), key=lambda r: r["abs_det"])
    summary["best_method"] = best["method"]
    summary["best_abs_det"] = best["abs_det"]
    summary["best_entries"] = best["entries"]
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

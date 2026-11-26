"""Manifold walk on |G| ⊆ {1, 3, 5} submanifold for H29, with class-tabu.

H29 LB matrices all share signature (count_1=393, count_3=4, count_5=9).
This implements smart_2val_search-style manifold drift but for H29's natural
{1, 3, 5} support, locking out the LB equivalence class so the walk MUST
find a different one if it accepts any move.

Energy: -log|det| with HARD REJECTION of:
  - off-manifold proposals (|G| not in {1, 3, 5})
  - moves into the locked class

Hypothesis: a class with signature (count_1=400, count_3=4, count_5=2) — same
total 406, fewer high-|G| entries — would have larger |det|. If such a class
exists in the search space, the walk should reach it. If 1000 chains × 200K
iters all stay in the LB class, that's strong empirical evidence the manifold
contains only one class at H29 (consistent with Brent's enumeration ruling
out other Gram patterns).
"""
import gzip, math, sys, time, json, random as pyrandom
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hadamard_maxdet.verifier import integer_determinant, upper_bound
from hadamard_maxdet.incremental import (
    IncrementalState, apply_entry_flip, propose_entry_flip, refresh,
)


def load_brent_h29(path: str | None = None, limit: int = 1) -> list[list[list[int]]]:
    if path is None:
        # Default: stable repo path. Falls back to /tmp if developer placed it there.
        repo_path = Path(__file__).resolve().parents[1] / "results" / "h29_corpus" / "brent_h29_LB_corpus.txt.gz"
        path = str(repo_path) if repo_path.exists() else "/tmp/s29allsofar.txt.gz"
    matrices = []
    with gzip.open(path, "rt") as f:
        text = f.read()
    i = 0
    lines = text.splitlines()
    while i < len(lines) and len(matrices) < limit:
        if lines[i].startswith("Matrix "):
            rows = []
            i += 1
            while len(rows) < 29 and i < len(lines):
                ln = lines[i].strip()
                if ln and all(c in "+-" for c in ln):
                    rows.append([1 if c == "+" else -1 for c in ln])
                i += 1
            if len(rows) == 29:
                matrices.append(rows)
        else:
            i += 1
    return matrices


@dataclass
class Class135State:
    """Tracks G, count_abs1, count_abs3, count_abs5, count_out for H29 |G|⊆{1,3,5} manifold."""
    n: int
    G: np.ndarray
    c1: int  # count |G|=1
    c3: int  # count |G|=3
    c5: int  # count |G|=5
    co: int  # count off-manifold

    @classmethod
    def from_matrix(cls, M):
        A = np.asarray(M, dtype=np.int64)
        n = A.shape[0]
        G = (A @ A.T).astype(np.int64)
        iu = np.triu_indices(n, k=1)
        v = np.abs(G[iu])
        c1 = int((v == 1).sum())
        c3 = int((v == 3).sum())
        c5 = int((v == 5).sum())
        total = n * (n - 1) // 2
        co = total - c1 - c3 - c5
        return cls(n=n, G=G, c1=c1, c3=c3, c5=c5, co=co)

    def deltas_for_flip(self, M_after, i, c):
        """Return (d1, d3, d5, d_out) for flipping entry (i, c). O(n)."""
        n = self.n
        new_val = int(M_after[i][c])
        delta_factor = 2 * new_val
        d1 = d3 = d5 = 0
        Gi = self.G[i]
        for r in range(n):
            if r == i:
                continue
            old = int(Gi[r])
            new = old + delta_factor * int(M_after[r][c])
            ao = abs(old); an = abs(new)
            if ao == an:
                continue
            if ao == 1:   d1 -= 1
            elif ao == 3: d3 -= 1
            elif ao == 5: d5 -= 1
            if an == 1:   d1 += 1
            elif an == 3: d3 += 1
            elif an == 5: d5 += 1
        d_out = -(d1 + d3 + d5)
        return d1, d3, d5, d_out

    def apply_after_flip(self, M_after, i, c):
        n = self.n
        new_val = int(M_after[i][c])
        delta_factor = 2 * new_val
        Gi = self.G[i]
        for r in range(n):
            if r == i:
                continue
            old = int(Gi[r])
            new = old + delta_factor * int(M_after[r][c])
            if new != old:
                ao = abs(old); an = abs(new)
                if ao != an:
                    if ao == 1: self.c1 -= 1
                    elif ao == 3: self.c3 -= 1
                    elif ao == 5: self.c5 -= 1
                    if an == 1: self.c1 += 1
                    elif an == 3: self.c3 += 1
                    elif an == 5: self.c5 += 1
                Gi[r] = new
                self.G[r, i] = new
        self.co = self.n * (self.n - 1) // 2 - self.c1 - self.c3 - self.c5


def manifold_walk_h29(
    M_init,
    iterations: int = 200_000,
    locked_classes: set | None = None,
    t_start: float = 0.5,
    t_end: float = 0.005,
    rng_seed: int = 0,
    refresh_every: int = 2000,
    verbose: bool = False,
    lam_c3: float = 0.0,
    lam_c5: float = 0.0,
):
    rng = pyrandom.Random(rng_seed)
    n = len(M_init)
    locked_classes = locked_classes or set()

    state = IncrementalState.from_matrix([row[:] for row in M_init])
    abs_state = Class135State.from_matrix(state.M)
    if state.det == 0:
        return None
    if abs_state.co != 0:
        return None  # seed not on manifold

    def energy_of(det_abs, c3, c5):
        return -math.log(det_abs) + lam_c3 * c3 + lam_c5 * c5

    cur_e = energy_of(abs(state.det), abs_state.c3, abs_state.c5)
    log_T_ratio = math.log(t_end / t_start)

    accepted = 0
    rej_offmanifold = 0
    rej_class = 0
    rej_metro = 0
    rej_singular = 0
    visited_classes = Counter()
    visited_classes[(abs_state.c1, abs_state.c3, abs_state.c5)] += 1

    best_outside_lb_det = 0
    best_outside_lb_sig = None
    best_outside_lb_M = None

    refresh_counter = 0

    for step in range(iterations):
        T = t_start * math.exp(log_T_ratio * step / max(1, iterations - 1))
        i = rng.randrange(n); j = rng.randrange(n)
        # Check off-manifold deltas BEFORE applying flip
        # We need M_after which has only entry (i,j) flipped
        old_val = state.M[i][j]
        new_val = -old_val
        # Construct M_after virtually (just use old M with one entry changed)
        # The deltas_for_flip uses M_after[r][k] for r != i and k = j (column c)
        # For r != i: M_after[r][j] = state.M[r][j] (unchanged for that flip)
        # For r == i: M_after[i][j] = new_val
        # The function reads M_after[i][c] (used to set new_val) and M_after[r][c] for other r
        # Since only entry (i,j) flipped, M_after[r][j] = state.M[r][j] for r != i
        # We can just temporarily flip and check, or compute manually.
        # Use the same trick: apply flip in M, compute deltas, revert if rejected.

        # Compute new det via incremental (cheap)
        new_det = propose_entry_flip(state, i, j)
        if new_det == 0:
            rej_singular += 1
            continue

        # Apply flip in state.M to compute delta on Gram correctly
        # (Class135State.deltas_for_flip needs M_after, which is state.M after the flip)
        state.M[i][j] = new_val
        d1, d3, d5, d_out = abs_state.deltas_for_flip(state.M, i, j)
        state.M[i][j] = old_val  # revert M for now

        new_co = abs_state.co + d_out
        if new_co != 0:
            rej_offmanifold += 1
            continue

        new_c1 = abs_state.c1 + d1
        new_c3 = abs_state.c3 + d3
        new_c5 = abs_state.c5 + d5
        new_class = (new_c1, new_c3, new_c5)
        if new_class in locked_classes:
            rej_class += 1
            continue

        new_e = energy_of(abs(new_det), new_c3, new_c5)
        delta = new_e - cur_e
        if delta > 0 and rng.random() >= math.exp(-delta / max(T, 1e-12)):
            rej_metro += 1
            continue

        # Accept: apply incremental and Gram updates
        apply_entry_flip(state, i, j)
        abs_state.apply_after_flip(state.M, i, j)
        cur_e = new_e
        accepted += 1
        visited_classes[new_class] += 1

        # Track best |det| in any non-locked class
        d_now = abs(state.det)
        if d_now > best_outside_lb_det:
            best_outside_lb_det = d_now
            best_outside_lb_sig = new_class
            best_outside_lb_M = [row[:] for row in state.M]

        refresh_counter += 1
        if refresh_counter >= refresh_every:
            refresh(state)
            refresh_counter = 0

    return {
        "best_det": best_outside_lb_det,
        "best_sig": best_outside_lb_sig,
        "best_M": best_outside_lb_M,
        "accepted": accepted,
        "rej_offmanifold": rej_offmanifold,
        "rej_class": rej_class,
        "rej_metro": rej_metro,
        "rej_singular": rej_singular,
        "visited_classes": dict(visited_classes),
    }


def main(n_chains: int = 16, iterations: int = 200_000,
         lam_c3: float = 0.5, lam_c5: float = 0.3):
    EW = upper_bound(29)
    LB = 320 * 7**12 * 2**28
    LB_ratio = LB / EW
    print(f"H29 EW={EW:.4e}  LB={LB:.4e}  ratio={LB_ratio:.4f}")

    # Load LB matrix as seed
    matrices = load_brent_h29(limit=1)
    M_seed = matrices[0]

    # Verify seed signature
    seed_state = Class135State.from_matrix(M_seed)
    print(f"Seed class: (c1={seed_state.c1}, c3={seed_state.c3}, c5={seed_state.c5}, off={seed_state.co})")
    seed_class = (seed_state.c1, seed_state.c3, seed_state.c5)

    # Lock the LB class
    locked = {seed_class}
    print(f"Locked classes: {locked}")
    print(f"\nRunning {n_chains} manifold walks, {iterations:_} iters each...")
    print(f"Energy: -log|det| + {lam_c3}·c3 + {lam_c5}·c5.")
    print(f"Manifold: |G| ⊆ {{1, 3, 5}}. Class-tabu: LB locked.")
    print(f"Anti-bias rewards classes with low c3+c5.")
    print()

    best_det_overall = 0
    best_sig_overall = None
    best_M_overall = None
    all_classes_seen = Counter()
    chain_results = []

    t0 = time.monotonic()
    for s in range(n_chains):
        chain_t0 = time.monotonic()
        result = manifold_walk_h29(
            M_seed,
            iterations=iterations,
            locked_classes=locked,
            t_start=0.5,
            t_end=0.005,
            rng_seed=s * 17 + 1,
            lam_c3=lam_c3,
            lam_c5=lam_c5,
        )
        chain_dt = time.monotonic() - chain_t0
        if result is None:
            print(f"  chain {s}: invalid (singular seed)")
            continue
        for cl, cnt in result["visited_classes"].items():
            all_classes_seen[cl] += cnt
        if result["best_det"] > best_det_overall:
            best_det_overall = result["best_det"]
            best_sig_overall = result["best_sig"]
            best_M_overall = result["best_M"]
        sig_str = result["best_sig"] if result["best_sig"] else "(none)"
        marker = " *** ABOVE LB ***" if result["best_det"] > LB else ""
        print(f"  chain {s:2d} ({chain_dt:.0f}s): "
              f"acc={result['accepted']:6d} "
              f"rej_class={result['rej_class']:6d} "
              f"rej_off={result['rej_offmanifold']:6d} "
              f"rej_metro={result['rej_metro']:6d} "
              f"distinct_classes={len(result['visited_classes'])}  "
              f"best |det|={result['best_det']:.3e} sig={sig_str}{marker}")
        chain_results.append({
            "chain": s,
            "best_det": result["best_det"],
            "best_sig": list(result["best_sig"]) if result["best_sig"] else None,
            "n_visited_classes": len(result["visited_classes"]),
            "rej_class": result["rej_class"],
        })

    print(f"\n=== SUMMARY ({time.monotonic()-t0:.0f}s) ===")
    print(f"Distinct classes ever visited: {len(all_classes_seen)}")
    if len(all_classes_seen) <= 20:
        print("Class visits (top 20):")
        for cl, n in all_classes_seen.most_common(20):
            print(f"  {cl}: {n}")
    print(f"\nBrent LB:    {LB} (ratio {LB_ratio:.4f})")
    print(f"Best found:  {best_det_overall} (ratio {best_det_overall/EW:.4f})")
    if best_det_overall > 0:
        print(f"Best class signature: {best_sig_overall}")
    if best_det_overall > LB:
        print(f"\n*** RECORD-CLASS *** {best_det_overall} > Brent LB {LB}")
        d_v = abs(int(integer_determinant(best_M_overall)))
        print(f"Triple verify: {d_v}")
        json.dump({"order": "H29", "abs_det": int(d_v), "matrix": best_M_overall,
                   "method": "manifold walk on |G|⊆{1,3,5}, LB class locked",
                   "signature": list(best_sig_overall)},
                  open("/tmp/h29_manifold_record.json", "w"), indent=2)
    elif best_det_overall == 0:
        print("\nNo non-locked-class moves accepted. Manifold may be a single class at H29.")
    else:
        print("\nDistinct classes found, none above LB.")

    json.dump({"chain_results": chain_results,
               "all_classes_seen": [{"sig": list(cl), "n_visits": n} for cl, n in all_classes_seen.items()]},
              open("/tmp/h29_manifold_walk_results.json", "w"), indent=2)


if __name__ == "__main__":
    main(n_chains=16, iterations=300_000, lam_c3=0.6, lam_c5=0.4)

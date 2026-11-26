"""Structural diagnostics for distinguishing proven-winner ±1 matrices from
near-winner SA bests.

Three hypotheses are tested via three measurable scalar diagnostics:

* **H1 — Best-clustering Gram score** (:func:`gram_clustering_score`).
  Spectral-cluster the rows on the Gram matrix at several ``k``; compute
  ``within_cluster_var(off_diag) - between_cluster_var(off_diag)``. A
  *lower* (more negative) value means the matrix admits a clean row
  partition where intra-cluster Gram values are tighter than inter-cluster.
  Returns the best (lowest) score across ``k`` and the optimal ``k``.

* **H2 — Gram-graph modularity** (:func:`gram_graph_modularity`).
  Build a weighted graph: nodes = rows, edge weight = ``|<v_i, v_j>|``
  (off-diag Gram magnitude). Run ``networkx.community.greedy_modularity``
  and report Newman's modularity for the recovered partition. Higher =
  stronger community structure.

* **H3 — Approximate symmetry score** (:func:`approx_symmetry_group_score`).
  Counts pairs of rows ``(i, j)`` whose Hamming distance from ``v_i`` to
  ``-v_j`` is exactly ``n/2 + offset`` for a small offset window (i.e.
  rows that look like balanced negations of each other modulo a fixed
  off-balance). Such pairs are footprints of dihedral-style symmetry.
  Returns the count of such pairs *plus* the orbit size of row 0 under
  the relation "another row is a balanced match of me", normalised by
  the total ``C(n, 2)`` pair count.

All three diagnostics are pure functions of the matrix; they do not
depend on the search history that produced it.

A summary helper :func:`compare_matrices` prints a labelled comparison
table across multiple matrices.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Optional

import math
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_array(M: Sequence[Sequence[int]]) -> np.ndarray:
    """Coerce to ``np.ndarray[float]`` of shape ``(n, n)``; sanity-check ±1."""
    arr = np.asarray(M, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"matrix must be square 2D; got shape {arr.shape}")
    if not np.all(np.isin(arr, (-1.0, 1.0))):
        raise ValueError("matrix entries must all be in {-1, +1}")
    return arr


def _offdiag_values(G: np.ndarray) -> np.ndarray:
    """Return the strict-upper-triangle off-diagonal entries of ``G``."""
    n = G.shape[0]
    iu = np.triu_indices(n, k=1)
    return G[iu]


# ---------------------------------------------------------------------------
# H1 — Best-clustering Gram score
# ---------------------------------------------------------------------------


def _spectral_partition(G_abs: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """Spectral clustering on a symmetric similarity matrix into ``k`` parts.

    Uses the normalised Laplacian's bottom-``k`` eigenvectors as features
    and applies KMeans (sklearn). Returns an integer label vector of
    length ``n``.
    """
    from sklearn.cluster import KMeans

    n = G_abs.shape[0]
    # Build a similarity matrix with non-negative entries on the off-diag,
    # zero diagonal. Use ``|G_ij|`` (so identical-mod-sign rows pull
    # together and orthogonal rows push apart).
    W = G_abs.copy().astype(float)
    np.fill_diagonal(W, 0.0)
    # Avoid all-zero similarity matrices (e.g. true Hadamard at n=8)
    # where every off-diag is 0; fall back to uniform W to avoid NaNs.
    if W.sum() <= 1e-12:
        W = np.ones_like(W) - np.eye(n)
    d = W.sum(axis=1)
    d_inv_sqrt = np.zeros_like(d)
    nz = d > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(d[nz])
    L_sym = np.eye(n) - (W * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    # Symmetrise for numerical safety.
    L_sym = 0.5 * (L_sym + L_sym.T)
    eigvals, eigvecs = np.linalg.eigh(L_sym)
    # Bottom k eigenvectors as features.
    feats = eigvecs[:, :k]
    # Row-normalise (Ng-Jordan-Weiss).
    row_norms = np.linalg.norm(feats, axis=1, keepdims=True)
    row_norms[row_norms < 1e-12] = 1.0
    feats = feats / row_norms
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    labels = km.fit_predict(feats)
    return labels


def gram_clustering_score(
    M: Sequence[Sequence[int]],
    k_range: Sequence[int] = (2, 3, 4, 5, 6),
    seed: int = 0,
) -> dict:
    """H1 — find the row-partition that gives the cleanest Gram block-structure.

    For each ``k`` in ``k_range`` we spectral-cluster the rows on the
    absolute Gram matrix; we compute the variance of off-diag Gram values
    *within* clusters and *between* clusters and report the best (lowest)
    ``within - between`` score.

    Returns a dict with:
      - ``best_score``: lowest ``within_var - between_var`` across ``k``.
      - ``best_k``: the ``k`` that achieved it.
      - ``per_k``: dict of ``k -> score`` for inspection.

    A *more negative* ``best_score`` indicates strong block structure;
    near zero indicates noisy / uniform Gram.
    """
    A = _to_array(M)
    n = A.shape[0]
    G = A @ A.T
    G_abs = np.abs(G)

    per_k: dict[int, float] = {}

    for k in k_range:
        if k < 2 or k > n - 1:
            continue
        labels = _spectral_partition(G_abs, k=k, seed=seed)
        # Compute within-cluster and between-cluster off-diag Gram variances.
        iu = np.triu_indices(n, k=1)
        i_idx, j_idx = iu
        same = labels[i_idx] == labels[j_idx]
        vals = G[iu].astype(float)
        within_vals = vals[same]
        between_vals = vals[~same]
        if within_vals.size < 2 or between_vals.size < 2:
            # Degenerate split — assign worst possible score (0).
            score = 0.0
        else:
            score = float(np.var(within_vals) - np.var(between_vals))
        per_k[k] = score

    if not per_k:
        return {"best_score": float("nan"), "best_k": None, "per_k": {}}

    best_k = min(per_k, key=lambda k: per_k[k])
    return {
        "best_score": per_k[best_k],
        "best_k": int(best_k),
        "per_k": {int(k): float(v) for k, v in per_k.items()},
    }


# ---------------------------------------------------------------------------
# H2 — Gram-graph modularity
# ---------------------------------------------------------------------------


def gram_graph_modularity(M: Sequence[Sequence[int]]) -> dict:
    """H2 — Newman's modularity of the |Gram|-weighted row-row graph.

    Higher modularity = stronger latent community structure among the
    rows. Returns a dict with ``modularity``, ``n_communities`` and
    ``community_sizes``.
    """
    import networkx as nx
    from networkx.algorithms.community import greedy_modularity_communities, modularity

    A = _to_array(M)
    n = A.shape[0]
    G = A @ A.T

    g = nx.Graph()
    g.add_nodes_from(range(n))

    iu = np.triu_indices(n, k=1)
    weights = np.abs(G[iu]).astype(float)
    # Edges with weight 0 carry no signal; skip them. (For a true Hadamard
    # all off-diags are 0 → graph is edgeless → modularity = 0.)
    for (i, j, w) in zip(iu[0], iu[1], weights):
        if w > 0:
            g.add_edge(int(i), int(j), weight=float(w))

    if g.number_of_edges() == 0:
        return {"modularity": 0.0, "n_communities": n,
                "community_sizes": [1] * n}

    try:
        comms = list(greedy_modularity_communities(g, weight="weight"))
        q = float(modularity(g, comms, weight="weight"))
        sizes = sorted([len(c) for c in comms], reverse=True)
        return {"modularity": q,
                "n_communities": len(comms),
                "community_sizes": sizes}
    except Exception as e:
        return {"modularity": float("nan"), "n_communities": -1,
                "community_sizes": [], "error": str(e)}


# ---------------------------------------------------------------------------
# H3 — Approximate symmetry group score
# ---------------------------------------------------------------------------


def approx_symmetry_group_score(
    M: Sequence[Sequence[int]],
    offset: int = 1,
) -> dict:
    """H3 — count balanced-negation row pairs as a symmetry footprint.

    For ±1 vectors ``u, v`` of length ``n``, the Hamming distance
    ``ham(u, -v) = n - ham(u, v) = (n - <u,v>) / 2 + ... `` actually
    simplifies: ``<u, v> = n - 2 * ham(u, v)``, so ``ham(u, v)`` is
    determined by ``<u, v>``. We say ``v_i, v_j`` are a *balanced
    pair* iff ``|<v_i, v_j>| <= 2 * offset`` (i.e., they differ in
    approximately ``n/2`` positions, the dihedral / sign-symmetry
    signature). A high count of such pairs indicates many local
    sign-flip symmetries.

    We also count, for each row ``i``, the number of balanced partners
    it has, and report the *orbit size* of row 0 under the equivalence
    "transitive closure of the balanced-pair relation".

    Returns a dict with:
      - ``balanced_pair_count``: integer
      - ``balanced_pair_density``: count / C(n, 2) in [0, 1]
      - ``row0_orbit_size``: size of row 0's connected component
      - ``row0_orbit_fraction``: orbit size / n
      - ``score``: density + orbit_fraction (combined scalar in [0, 2])
    """
    A = _to_array(M)
    n = A.shape[0]
    G = (A @ A.T).astype(int)
    iu = np.triu_indices(n, k=1)
    offdiag = G[iu]

    # A row pair is "balanced" iff |<v_i, v_j>| <= 2*offset.
    threshold = 2 * offset
    balanced = np.abs(offdiag) <= threshold
    pair_count = int(balanced.sum())
    total_pairs = n * (n - 1) // 2
    density = pair_count / total_pairs if total_pairs else 0.0

    # Build a graph of balanced pairs and find row 0's connected component.
    adj: list[set[int]] = [set() for _ in range(n)]
    for k_idx, is_balanced in enumerate(balanced):
        if is_balanced:
            i = int(iu[0][k_idx])
            j = int(iu[1][k_idx])
            adj[i].add(j)
            adj[j].add(i)
    seen = {0}
    stack = [0]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    orbit_size = len(seen)
    orbit_frac = orbit_size / n

    return {
        "balanced_pair_count": pair_count,
        "balanced_pair_density": float(density),
        "row0_orbit_size": int(orbit_size),
        "row0_orbit_fraction": float(orbit_frac),
        "score": float(density + orbit_frac),
        "offset": int(offset),
    }


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------


def compare_matrices(
    matrices: Sequence[Sequence[Sequence[int]]],
    labels: Sequence[str],
    *,
    seed: int = 0,
    offset: int = 1,
    print_table: bool = True,
) -> list[dict]:
    """Compute all three diagnostics for each matrix and (optionally) print
    a comparison table. Returns a list of per-matrix dicts.
    """
    if len(matrices) != len(labels):
        raise ValueError("matrices and labels must have the same length")

    rows = []
    for M, lab in zip(matrices, labels):
        h1 = gram_clustering_score(M, seed=seed)
        h2 = gram_graph_modularity(M)
        h3 = approx_symmetry_group_score(M, offset=offset)
        rows.append({
            "label": lab,
            "n": len(M),
            "h1_cluster_score": h1["best_score"],
            "h1_best_k": h1["best_k"],
            "h2_modularity": h2["modularity"],
            "h2_n_communities": h2["n_communities"],
            "h3_score": h3["score"],
            "h3_balanced_pair_density": h3["balanced_pair_density"],
            "h3_row0_orbit_fraction": h3["row0_orbit_fraction"],
        })

    if print_table:
        _print_table(rows)
    return rows


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'matrix':28s} | {'n':3s} | "
        f"{'H1 score (k)':18s} | {'H2 modularity':16s} | "
        f"{'H3 score':10s} | {'pair-dens':10s} | {'orbit-frac':10s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        h1s = r["h1_cluster_score"]
        h1k = r["h1_best_k"]
        h1str = (
            f"{h1s:+9.3f} (k={h1k})" if h1k is not None
            else f"{h1s:+9.3f}      "
        )
        print(
            f"{r['label']:28s} | {r['n']:3d} | "
            f"{h1str:18s} | {r['h2_modularity']:16.4f} | "
            f"{r['h3_score']:10.4f} | "
            f"{r['h3_balanced_pair_density']:10.4f} | "
            f"{r['h3_row0_orbit_fraction']:10.4f}"
        )


# ---------------------------------------------------------------------------
# M1 — Third-order cyclic Gram product distribution
# ---------------------------------------------------------------------------


def triple_cyclic_product_stats(M: Sequence[Sequence[int]]) -> dict:
    """M1 — distribution of cyclic triple Gram products.

    For every ordered triple ``(i, j, k)`` with ``i < j < k``, compute

        T(i, j, k) = <v_i, v_j> * <v_j, v_k> * <v_k, v_i>

    This product is invariant under simultaneous sign-flip of any single
    row ``v_i`` (each row appears in exactly two of the three factors,
    so the sign change cancels), so it is a true *triangle invariant* of
    the row Gram matrix beyond pairwise inner products.

    Returns a dict with:
      - ``support``: sorted list of distinct ``T`` values (ints).
      - ``counts``: ``{value: count}`` multiset.
      - ``cardinality``: number of distinct values.
      - ``mean``, ``std``, ``max_abs``: scalar moments.
      - ``frac_zero``: fraction of triples with ``T == 0``.
      - ``entropy``: Shannon entropy of the value distribution (nats).
      - ``n_triples``: ``C(n, 3)``.
    """
    A = _to_array(M).astype(int)
    n = A.shape[0]
    G = (A @ A.T).astype(int)

    # Vectorised C(n,3) enumeration: build index arrays.
    if n < 3:
        return {
            "support": [],
            "counts": {},
            "cardinality": 0,
            "mean": 0.0,
            "std": 0.0,
            "max_abs": 0,
            "frac_zero": 0.0,
            "entropy": 0.0,
            "n_triples": 0,
        }

    # Build i < j < k explicitly.
    triples = np.array(
        [(i, j, k) for i in range(n - 2)
         for j in range(i + 1, n - 1) for k in range(j + 1, n)],
        dtype=int,
    )
    if triples.size == 0:
        return {
            "support": [],
            "counts": {},
            "cardinality": 0,
            "mean": 0.0,
            "std": 0.0,
            "max_abs": 0,
            "frac_zero": 0.0,
            "entropy": 0.0,
            "n_triples": 0,
        }
    i_idx = triples[:, 0]
    j_idx = triples[:, 1]
    k_idx = triples[:, 2]

    Gij = G[i_idx, j_idx]
    Gjk = G[j_idx, k_idx]
    Gki = G[k_idx, i_idx]
    T = (Gij * Gjk * Gki).astype(np.int64)

    n_triples = int(T.size)
    # Counts (use dict for serialisation friendliness).
    vals, counts = np.unique(T, return_counts=True)
    counts_dict = {int(v): int(c) for v, c in zip(vals, counts)}
    support = sorted(counts_dict.keys())
    cardinality = len(support)

    mean = float(T.mean())
    std = float(T.std())
    max_abs = int(np.max(np.abs(T)))
    frac_zero = float((T == 0).sum()) / n_triples if n_triples else 0.0

    # Shannon entropy of the value distribution.
    probs = counts.astype(float) / n_triples
    nz = probs > 0
    entropy = float(-(probs[nz] * np.log(probs[nz])).sum())

    return {
        "support": support,
        "counts": counts_dict,
        "cardinality": int(cardinality),
        "mean": mean,
        "std": std,
        "max_abs": max_abs,
        "frac_zero": frac_zero,
        "entropy": entropy,
        "n_triples": n_triples,
    }


# ---------------------------------------------------------------------------
# M2 — Column Gram (M^T M) vs row Gram (M M^T)
# ---------------------------------------------------------------------------


def column_gram_stats(M: Sequence[Sequence[int]]) -> dict:
    """M2 — column Gram (``M^T M``) versus row Gram (``M M^T``).

    The eigenvalue spectra agree (same singular values squared), but the
    *off-diagonal multisets* generally differ for ±1 matrices. Returns a
    dict with the column-Gram off-diag support / counts / cardinality
    plus the same for the row Gram, and a small ``comparison`` block
    quantifying any difference.
    """
    A = _to_array(M).astype(int)
    n = A.shape[0]

    G_row = (A @ A.T).astype(int)
    G_col = (A.T @ A).astype(int)

    iu = np.triu_indices(n, k=1)

    def _stats(G: np.ndarray) -> dict:
        off = G[iu]
        vals, counts = np.unique(off, return_counts=True)
        cd = {int(v): int(c) for v, c in zip(vals, counts)}
        return {
            "support": sorted(cd.keys()),
            "counts": cd,
            "cardinality": int(len(cd)),
            "abs_max": int(np.max(np.abs(off))) if off.size else 0,
            "mean": float(off.mean()) if off.size else 0.0,
            "std": float(off.std()) if off.size else 0.0,
        }

    row = _stats(G_row)
    col = _stats(G_col)

    # Comparison: do row & column Gram have identical multisets?
    same_support = row["support"] == col["support"]
    same_counts = row["counts"] == col["counts"]

    # Cardinality delta (col - row) and abs-max delta.
    card_delta = col["cardinality"] - row["cardinality"]
    abs_max_delta = col["abs_max"] - row["abs_max"]

    return {
        "row_gram": row,
        "column_gram": col,
        "comparison": {
            "same_support": bool(same_support),
            "same_multiset": bool(same_counts),
            "cardinality_delta_col_minus_row": int(card_delta),
            "abs_max_delta_col_minus_row": int(abs_max_delta),
        },
    }


# ---------------------------------------------------------------------------
# M3 — Singular vector localisation (IPR)
# ---------------------------------------------------------------------------


def singular_vector_ipr(M: Sequence[Sequence[int]]) -> dict:
    """M3 — Inverse Participation Ratio of singular vectors.

    For each left singular vector ``u_i`` of ``M`` (rows of ``U`` from
    the SVD), the IPR is

        IPR(u_i) = sum_k u_i[k]**4 / (sum_k u_i[k]**2)**2

    Range ``[1/n, 1]``: ``1/n`` ⇔ uniformly delocalised (e.g. Hadamard
    rows), ``1`` ⇔ localised on one coordinate.

    Returns a dict with:
      - ``ipr_all``: list of IPR values, ordered by descending singular value.
      - ``top_3``, ``bottom_3``: short lists.
      - ``mean_ipr``, ``min_ipr``, ``max_ipr``.
      - ``leading_ipr``: IPR of the dominant singular vector.
      - ``uniform_ipr``: theoretical lower bound ``1/n`` for reference.
    """
    A = _to_array(M)
    n = A.shape[0]

    # Full SVD; we need both left and right vectors but use left (rows).
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # Order by descending singular value (svd already returns in that order).
    order = np.argsort(-S)
    U = U[:, order]
    S = S[order]

    iprs = []
    for i in range(U.shape[1]):
        u = U[:, i]
        denom = float((u * u).sum()) ** 2
        if denom <= 0:
            iprs.append(float("nan"))
            continue
        num = float((u ** 4).sum())
        iprs.append(num / denom)

    arr = np.array(iprs, dtype=float)
    finite = arr[np.isfinite(arr)]
    return {
        "ipr_all": [float(x) for x in iprs],
        "top_3": [float(x) for x in iprs[:3]],
        "bottom_3": [float(x) for x in iprs[-3:]],
        "mean_ipr": float(finite.mean()) if finite.size else float("nan"),
        "min_ipr": float(finite.min()) if finite.size else float("nan"),
        "max_ipr": float(finite.max()) if finite.size else float("nan"),
        "leading_ipr": float(iprs[0]) if iprs else float("nan"),
        "uniform_ipr": 1.0 / n,
        "singular_values": [float(s) for s in S],
    }


# ---------------------------------------------------------------------------
# Comparison driver v2 — for the three new (higher-order) metrics
# ---------------------------------------------------------------------------


def compare_matrices_v2(
    matrices: Sequence[Sequence[Sequence[int]]],
    labels: Sequence[str],
    *,
    print_table: bool = True,
) -> list[dict]:
    """Compute all three *higher-order* diagnostics (M1/M2/M3) for each
    matrix and (optionally) print three comparison tables. Returns a
    list of per-matrix dicts (full payload).
    """
    if len(matrices) != len(labels):
        raise ValueError("matrices and labels must have the same length")

    rows: list[dict] = []
    for M, lab in zip(matrices, labels):
        m1 = triple_cyclic_product_stats(M)
        m2 = column_gram_stats(M)
        m3 = singular_vector_ipr(M)
        rows.append({
            "label": lab,
            "n": len(M),
            "m1": m1,
            "m2": m2,
            "m3": m3,
        })

    if print_table:
        _print_table_v2(rows)
    return rows


def _print_table_v2(rows: list[dict]) -> None:
    # ---- M1 table ----
    print()
    print("M1 — Triple cyclic Gram product (T = <vi,vj><vj,vk><vk,vi>)")
    header = (
        f"{'matrix':28s} | {'n':3s} | {'card':4s} | "
        f"{'mean':10s} | {'std':10s} | {'max|T|':9s} | "
        f"{'frac0':7s} | {'H(p)':7s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        m1 = r["m1"]
        print(
            f"{r['label']:28s} | {r['n']:3d} | {m1['cardinality']:4d} | "
            f"{m1['mean']:+10.3f} | {m1['std']:10.3f} | {m1['max_abs']:9d} | "
            f"{m1['frac_zero']:7.4f} | {m1['entropy']:7.4f}"
        )

    # ---- M2 table ----
    print()
    print("M2 — Row Gram vs Column Gram off-diagonal multisets")
    header = (
        f"{'matrix':28s} | {'n':3s} | "
        f"{'row card':9s} | {'col card':9s} | "
        f"{'same set':9s} | {'same mult':10s} | "
        f"{'row absmax':11s} | {'col absmax':11s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        m2 = r["m2"]
        rg = m2["row_gram"]
        cg = m2["column_gram"]
        cmp = m2["comparison"]
        print(
            f"{r['label']:28s} | {r['n']:3d} | "
            f"{rg['cardinality']:9d} | {cg['cardinality']:9d} | "
            f"{str(cmp['same_support']):9s} | {str(cmp['same_multiset']):10s} | "
            f"{rg['abs_max']:11d} | {cg['abs_max']:11d}"
        )

    # ---- M3 table ----
    print()
    print("M3 — Singular-vector IPR (1/n = uniform, 1 = localised)")
    header = (
        f"{'matrix':28s} | {'n':3s} | {'1/n':7s} | "
        f"{'leading':9s} | {'mean':9s} | "
        f"{'top3':28s} | {'bot3':28s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        m3 = r["m3"]
        top3 = ",".join(f"{x:.3f}" for x in m3["top_3"])
        bot3 = ",".join(f"{x:.3f}" for x in m3["bottom_3"])
        print(
            f"{r['label']:28s} | {r['n']:3d} | {m3['uniform_ipr']:7.4f} | "
            f"{m3['leading_ipr']:9.4f} | {m3['mean_ipr']:9.4f} | "
            f"{top3:28s} | {bot3:28s}"
        )


# ---------------------------------------------------------------------------
# Column-gram-deep diagnostics (transpose symmetry & col T-cardinality)
# ---------------------------------------------------------------------------


def transpose_symmetry_score(M: Sequence[Sequence[int]]) -> dict:
    """Quantify the row-Gram vs column-Gram off-diagonal multiset agreement.

    For a square ±1 matrix M, the row Gram ``G_row = M Mᵀ`` and column
    Gram ``G_col = Mᵀ M`` always have identical *spectra* (singular values
    squared), but their off-diagonal *multisets* generally differ.

    This function computes:

      - ``score``: 1.0 if the off-diagonal count distribution of the row
        Gram equals that of the column Gram exactly; otherwise
        ``1 - L1_distance / (2 * total_pairs)``. The L1 normalisation
        ensures ``score ∈ [0, 1]`` (since the multisets each have
        ``C(n, 2)`` entries, max possible L1 distance over their joint
        support is ``2 * C(n, 2)``).

      - ``l1_distance``: integer L1 distance between the two count
        dicts (over the union of supports).

      - ``same_support``: do the row & column Gram have the same set
        of off-diag values?

      - ``same_multiset``: do the count dicts agree exactly?

      - ``row_counts`` / ``col_counts``: the underlying multiset dicts.
    """
    A = _to_array(M).astype(int)
    n = A.shape[0]
    G_row = (A @ A.T).astype(int)
    G_col = (A.T @ A).astype(int)

    iu = np.triu_indices(n, k=1)
    row_off = G_row[iu]
    col_off = G_col[iu]

    rv, rc = np.unique(row_off, return_counts=True)
    cv, cc = np.unique(col_off, return_counts=True)
    row_counts = {int(v): int(c) for v, c in zip(rv, rc)}
    col_counts = {int(v): int(c) for v, c in zip(cv, cc)}

    keys = set(row_counts) | set(col_counts)
    l1 = sum(abs(row_counts.get(k, 0) - col_counts.get(k, 0)) for k in keys)

    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        score = 1.0
    else:
        # Each multiset has total_pairs entries; max possible L1 over
        # the union is 2 * total_pairs (completely disjoint supports).
        score = 1.0 - (l1 / (2.0 * total_pairs))

    return {
        "score": float(score),
        "l1_distance": int(l1),
        "same_support": sorted(row_counts.keys()) == sorted(col_counts.keys()),
        "same_multiset": row_counts == col_counts,
        "row_counts": row_counts,
        "col_counts": col_counts,
        "total_pairs": int(total_pairs),
    }


def col_gram_t_cardinality(M: Sequence[Sequence[int]]) -> dict:
    """Triple-cyclic-product cardinality on the column Gram (Mᵀ M).

    Mirrors :func:`triple_cyclic_product_stats` (which is row-Gram-based)
    but operates on the column Gram. For column index triples
    ``(i < j < k)``, computes

        T_col(i, j, k) = G_col[i,j] * G_col[j,k] * G_col[k,i]

    where ``G_col = Mᵀ M``. Returns the same keys as
    :func:`triple_cyclic_product_stats` plus a small ``row_vs_col``
    block comparing row- and column-side T cardinalities.
    """
    A = _to_array(M).astype(int)
    n = A.shape[0]

    G_col = (A.T @ A).astype(int)

    if n < 3:
        empty = {
            "support": [],
            "counts": {},
            "cardinality": 0,
            "mean": 0.0,
            "std": 0.0,
            "max_abs": 0,
            "frac_zero": 0.0,
            "entropy": 0.0,
            "n_triples": 0,
        }
        return {**empty, "row_vs_col": {
            "row_cardinality": 0, "col_cardinality": 0,
            "card_delta_col_minus_row": 0, "same_card": True,
        }}

    triples = np.array(
        [(i, j, k) for i in range(n - 2)
         for j in range(i + 1, n - 1) for k in range(j + 1, n)],
        dtype=int,
    )
    i_idx = triples[:, 0]; j_idx = triples[:, 1]; k_idx = triples[:, 2]

    Gij = G_col[i_idx, j_idx]
    Gjk = G_col[j_idx, k_idx]
    Gki = G_col[k_idx, i_idx]
    T = (Gij * Gjk * Gki).astype(np.int64)

    n_triples = int(T.size)
    vals, counts = np.unique(T, return_counts=True)
    counts_dict = {int(v): int(c) for v, c in zip(vals, counts)}
    support = sorted(counts_dict.keys())
    cardinality = len(support)
    mean = float(T.mean()); std = float(T.std())
    max_abs = int(np.max(np.abs(T)))
    frac_zero = float((T == 0).sum()) / n_triples if n_triples else 0.0
    probs = counts.astype(float) / n_triples
    nz = probs > 0
    entropy = float(-(probs[nz] * np.log(probs[nz])).sum())

    # Compute row-T cardinality for comparison (cheap via existing helper).
    row_stats = triple_cyclic_product_stats(M)
    row_card = int(row_stats["cardinality"])
    delta = cardinality - row_card

    return {
        "support": support,
        "counts": counts_dict,
        "cardinality": int(cardinality),
        "mean": mean,
        "std": std,
        "max_abs": max_abs,
        "frac_zero": frac_zero,
        "entropy": entropy,
        "n_triples": n_triples,
        "row_vs_col": {
            "row_cardinality": row_card,
            "col_cardinality": int(cardinality),
            "card_delta_col_minus_row": int(delta),
            "same_card": bool(row_card == cardinality),
        },
    }


def compare_proven_vs_near(
    many_provens: Sequence[Sequence[Sequence[int]]],
    many_nears: Sequence[Sequence[Sequence[int]]],
    *,
    print_report: bool = True,
) -> dict:
    """Classify proven-winner vs near-winner matrices on the column-Gram
    discriminators.

    For each input matrix compute:
      - ``transpose_symmetry_score`` (higher = row & col Gram agree).
      - ``col_gram_t_cardinality`` (cardinality + row-vs-col delta).

    A simple linear classifier (threshold on transpose_symmetry_score
    midway between the proven-mean and near-mean) is applied for
    illustration, and per-class summary statistics are reported.

    Returns a dict with per-class arrays, the threshold, and the
    classification accuracy on the union sample.
    """
    def _scores(mats):
        ts = []; col_cards = []; row_cards = []; deltas = []
        for M in mats:
            t = transpose_symmetry_score(M)
            c = col_gram_t_cardinality(M)
            ts.append(float(t["score"]))
            col_cards.append(int(c["cardinality"]))
            row_cards.append(int(c["row_vs_col"]["row_cardinality"]))
            deltas.append(int(c["row_vs_col"]["card_delta_col_minus_row"]))
        return ts, col_cards, row_cards, deltas

    p_ts, p_col, p_row, p_delta = _scores(many_provens)
    n_ts, n_col, n_row, n_delta = _scores(many_nears)

    def _summ(arr):
        a = np.array(arr, dtype=float)
        if a.size == 0:
            return {"n": 0, "mean": float("nan"), "std": float("nan"),
                    "min": float("nan"), "max": float("nan")}
        return {"n": int(a.size), "mean": float(a.mean()),
                "std": float(a.std()), "min": float(a.min()),
                "max": float(a.max())}

    # Threshold midway between class means (or fallback midpoint of ranges).
    p_mean = float(np.mean(p_ts)) if p_ts else float("nan")
    n_mean = float(np.mean(n_ts)) if n_ts else float("nan")
    if math.isfinite(p_mean) and math.isfinite(n_mean):
        threshold = 0.5 * (p_mean + n_mean)
    else:
        threshold = float("nan")

    # If proven > near (expected), classify s >= threshold => proven.
    proven_high = p_mean > n_mean if math.isfinite(p_mean) and math.isfinite(n_mean) else True

    correct = 0; total = 0
    for s in p_ts:
        total += 1
        pred_proven = (s >= threshold) if proven_high else (s <= threshold)
        if pred_proven:
            correct += 1
    for s in n_ts:
        total += 1
        pred_proven = (s >= threshold) if proven_high else (s <= threshold)
        if not pred_proven:
            correct += 1
    accuracy = correct / total if total else float("nan")

    # Cardinality-delta-based classifier: |delta| == 0 => proven.
    delta_acc_correct = 0
    for d in p_delta:
        if d == 0:
            delta_acc_correct += 1
    for d in n_delta:
        if d != 0:
            delta_acc_correct += 1
    delta_accuracy = delta_acc_correct / total if total else float("nan")

    report = {
        "proven": {
            "n": len(many_provens),
            "transpose_symmetry_score": _summ(p_ts),
            "col_T_cardinality": _summ(p_col),
            "row_T_cardinality": _summ(p_row),
            "col_minus_row_T_card": _summ(p_delta),
            "score_values": p_ts,
            "col_card_values": p_col,
            "row_card_values": p_row,
            "delta_values": p_delta,
        },
        "near": {
            "n": len(many_nears),
            "transpose_symmetry_score": _summ(n_ts),
            "col_T_cardinality": _summ(n_col),
            "row_T_cardinality": _summ(n_row),
            "col_minus_row_T_card": _summ(n_delta),
            "score_values": n_ts,
            "col_card_values": n_col,
            "row_card_values": n_row,
            "delta_values": n_delta,
        },
        "classifier": {
            "threshold": threshold,
            "proven_high": bool(proven_high),
            "linear_threshold_accuracy": float(accuracy),
            "delta_zero_accuracy": float(delta_accuracy),
        },
    }

    if print_report:
        print("\nProven-winner class:")
        print(f"  n={report['proven']['n']}")
        print(f"  transpose_symmetry_score: {report['proven']['transpose_symmetry_score']}")
        print(f"  col_T_cardinality:        {report['proven']['col_T_cardinality']}")
        print(f"  row_T_cardinality:        {report['proven']['row_T_cardinality']}")
        print(f"  col-minus-row T card:     {report['proven']['col_minus_row_T_card']}")
        print("\nNear-winner class:")
        print(f"  n={report['near']['n']}")
        print(f"  transpose_symmetry_score: {report['near']['transpose_symmetry_score']}")
        print(f"  col_T_cardinality:        {report['near']['col_T_cardinality']}")
        print(f"  row_T_cardinality:        {report['near']['row_T_cardinality']}")
        print(f"  col-minus-row T card:     {report['near']['col_minus_row_T_card']}")
        print("\nClassifier:")
        print(f"  threshold (midpoint of class means): {report['classifier']['threshold']}")
        print(f"  proven_high={report['classifier']['proven_high']}")
        print(f"  linear-threshold accuracy: {report['classifier']['linear_threshold_accuracy']:.4f}")
        print(f"  delta==0 accuracy:         {report['classifier']['delta_zero_accuracy']:.4f}")

    return report


__all__ = [
    "gram_clustering_score",
    "gram_graph_modularity",
    "approx_symmetry_group_score",
    "compare_matrices",
    "triple_cyclic_product_stats",
    "column_gram_stats",
    "singular_vector_ipr",
    "compare_matrices_v2",
    "transpose_symmetry_score",
    "col_gram_t_cardinality",
    "compare_proven_vs_near",
]

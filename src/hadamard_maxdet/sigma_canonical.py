"""σ-canonical-form relabeling for SBIBD-derived ±1 matrices.

For an SBIBD with parameters (v, k, λ), the {0, 1}-incidence matrix N has
v blocks (columns) each containing k of the v points (rows). After fixing
B1 = lex-min block to {0..k-1}, the second-lex-min block B2 has form
(B2 ∩ B1, B2 \\ B1) for some 6-subset of B1 and 10-subset of complement.
By S_k × S_{v-k} freedom we can permute within B1 and within complement
to make:

    B2 ∩ B1 = {0, 1, ..., λ - 1}            (lex-min λ-subset of B1)
    B2 \\ B1 = {k, k+1, ..., 2k - λ - 1}     (lex-min (k-λ)-subset of compl)

So canonical B1 = {0..k-1}, B2 = {0..λ-1} ∪ {k..2k-λ-1}. This is well-
defined regardless of which "second block" we picked (we permute to make
it canonical).

Verified against Koukouvinos's (41, 16, 6) design at
http://www.math.ntua.gr/~ckoukouv/41_16_6.htm:
    sigma_relabel(...) maps Koukouvinos's lex-2nd block to canonical
    form {0..5, 16..25}, and the relabeled lex-asc sorted-block list
    starts with the canonical (B1, B2) followed by 39 blocks all
    lex > B2 (as my cell-decomposition argument requires).

Use cases:
  - Stronger fingerprint for class_tabu when the candidate is an
    SBIBD-derived matrix (e.g. M = 2N - J at EW-tight n).
  - Canonicalize SBIBD candidates produced by different searches so
    they can be deduplicated globally.
  - Verify that a candidate (B1, B2) extends to a valid SBIBD via
    lex-min DFS — if yes, the design exists with this canonical form.

Limitations:
  - Specific to SBIBD-derived matrices. Does NOT apply to general ±1
    matrices at, e.g., H23 (where there's no underlying SBIBD).
  - Per-call cost is O(v · k) for the relabeling.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def extract_blocks_from_incidence(N: np.ndarray) -> list[list[int]]:
    """Each column of N is a block; return as sorted point lists."""
    v = N.shape[0]
    return sorted(
        [sorted(int(i) for i in range(v) if N[i, j] == 1) for j in range(v)]
    )


def sigma_relabeling(
    blocks: list[list[int]],
    *,
    v: int,
    k: int,
    lam: int,
) -> dict[int, int]:
    """Compute σ : {0..v-1} → {0..v-1} relabeling that canonicalizes the
    lex-min and lex-2nd-min blocks of `blocks`.

    Returns σ as a dict (point → new label).

    Pre-conditions:
        blocks is sorted lex-asc (caller's responsibility).
        blocks[0] is the lex-min block.
        blocks[1] is the lex-2nd-min block.
    """
    if len(blocks) < 2:
        raise ValueError("need at least 2 blocks")
    B1 = blocks[0]
    B2 = blocks[1]
    if len(B1) != k or len(B2) != k:
        raise ValueError(f"blocks must have size {k}")
    if len(set(B1) & set(B2)) != lam:
        raise ValueError(f"|B1 ∩ B2| = {len(set(B1) & set(B2))}, expected {lam}")

    B2_in_B1 = sorted(set(B2) & set(B1))           # λ elements
    B1_only = sorted(set(B1) - set(B2))            # k - λ elements
    B2_only = sorted(set(B2) - set(B1))            # k - λ elements
    outside = sorted((set(range(v)) - set(B1)) - set(B2))  # v - 2k + λ elements

    sigma: dict[int, int] = {}
    for k_, p in enumerate(B2_in_B1):
        sigma[p] = k_                       # → {0..λ-1}
    for k_, p in enumerate(B1_only):
        sigma[p] = lam + k_                 # → {λ..k-1}
    for k_, p in enumerate(B2_only):
        sigma[p] = k + k_                   # → {k..2k-λ-1}
    for k_, p in enumerate(outside):
        sigma[p] = (2 * k - lam) + k_       # → {2k-λ..v-1}
    return sigma


def apply_sigma(blocks: list[list[int]], sigma: dict[int, int]) -> list[list[int]]:
    """Return new sorted-lex-asc block list with sigma applied to each block."""
    return sorted(sorted(sigma[p] for p in B) for B in blocks)


def canonical_block_signature(
    blocks: list[list[int]],
    *,
    v: int,
    k: int,
    lam: int,
) -> tuple[tuple[int, ...], ...]:
    """Return a hashable signature of an SBIBD by its σ-canonical block list.

    Two SBIBDs with the same signature are equivalent under the canonical
    point relabeling (= same up to relabeling within the residual stabilizer
    S_λ × S_{k-λ} × S_{k-λ} × S_{v-2k+λ}, which is itself a small group
    we don't break). Different signatures imply inequivalent SBIBDs (under
    this canonical form's quotient).

    Use as a stronger fingerprint than Gram-off-diagonal count for
    SBIBD-derived matrices.
    """
    blocks_sorted = sorted(blocks)
    sigma = sigma_relabeling(blocks_sorted, v=v, k=k, lam=lam)
    canonical = apply_sigma(blocks_sorted, sigma)
    return tuple(tuple(B) for B in canonical)


def verify_canonical_form(
    blocks: list[list[int]],
    *,
    v: int,
    k: int,
    lam: int,
) -> bool:
    """Check that blocks (after sorting + σ-relabeling) starts with the
    canonical (B1 = {0..k-1}, B2 = {0..λ-1, k..2k-λ-1}).

    For SBIBDs admitting this canonical form (i.e., where the lex-min and
    lex-2nd blocks have the right intersection structure), returns True.
    Returns False if the input blocks don't conform.
    """
    canonical = canonical_block_signature(blocks, v=v, k=k, lam=lam)
    expected_B1 = tuple(range(k))
    expected_B2 = tuple(range(lam)) + tuple(range(k, 2 * k - lam))
    return canonical[0] == expected_B1 and canonical[1] == expected_B2

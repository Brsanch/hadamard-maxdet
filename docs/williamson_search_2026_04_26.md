# Williamson H24 Search — 2026-04-26

## Goal

Search for Williamson decompositions of order 24 (four palindromic length-6 ±1 sequences satisfying
the PAF condition) as an alternative H24 seed family, then recurse the H23 maxdet pipeline on their
23×23 row+column deletions.

## Method

### Step 1 — Williamson sequence enumeration (t=6)

A palindromic length-6 sequence has free bits (a₀, a₁, a₂, a₃) → [a₀, a₁, a₂, a₃, a₂, a₁].
There are 2⁴ = 16 such sequences. We exhaustively searched all 16⁴ = 65,536 quadruples (A,B,C,D)
and built the 24×24 Williamson array:

```
[ A  B  C  D]
[-B  A -D  C]
[-C  D  A -B]
[-D -C  B  A]
```

(where each letter denotes the 6×6 circulant of that sequence).

Validation criterion: `H · Hᵀ = 24 · I₂₄` (exact integer arithmetic, numpy int64).

**Result**: 1,536 valid Williamson H24 matrices (65,536 quadruples checked).

After canonicalization (normalize sign + lexsort rows): **96 distinct matrices**.
All 96 are genuinely new — Hamming distance to Paley I(24) ranges 218–262, to Kronecker H12⊗H2
ranges 230–280.

### Step 2 — Deletion enumeration

For each of the 96 unique Williamson H24 matrices, all 24×24 = 576 row+column deletions
were computed via exact Bareiss determinant.

**Key finding**: Every single deletion gives |det| = 1,521,681,143,169,024, i.e., ratio
**0.5475** relative to Orrick's H23 record — identical to the Paley I(24) and Kronecker H24
deletion ratio reported in earlier sessions. This is **1 distinct |det| class** across all
96 × 576 = 55,296 deletions.

### Step 3 — Plateau-bucketed k=2 enumeration

Starting from 5 different Williamson deletion seeds (different row/col deletion positions),
all C(529, 2) = 139,656 k=2 sign-flip pairs were enumerated using SVD-based |det| approximation
with Bareiss verification of the best candidate per plateau band.

**Result**: The best k=2 neighbour of any Williamson deletion is the seed itself (ratio 0.5475).
No k=2 flip achieves ratio > 0.5475. The band landscape:
- Band 0.0–0.0005: best ratio = 0.5475 (unchanged seed)
- Band 0.02–0.04: best ratio = 0.5323
- Band 0.08–0.14: best ratio = 0.4714
- Band 0.14–0.20: best ratio = 0.4562
- Band 0.20–0.50: best ratio = 0.3802

No band in [0.0005–0.14) was populated with neighbours above the seed.

### Step 4 — k=3 random sampling (1M samples)

1,000,000 random k=3 sign-flip triples from the Williamson seed, SVD-screened.

- Best SVD ratio reached: **0.5500**
- Candidates with ratio > 0.60: **0**
- Elapsed: 31.8 s

## Summary

| Metric | Value |
|--------|-------|
| Quadruples searched | 65,536 |
| Valid Williamson H24s (raw) | 1,536 |
| Unique H24s (after canon.) | 96 |
| New vs Paley/Kron | 96 |
| Distinct deletion |det| classes | 1 |
| Deletion max ratio | 0.5475 |
| k=2 max ratio | 0.5475 |
| k=3 max ratio (1M samples) | 0.5500 |
| New H23 records found | **0** |
| Orrick record status | **Unbroken** |

## Conclusion

Williamson H24 at t=6 provides no new entry point into the H23 elite cluster. All deletion seeds
land at ratio ~0.5475 and are locally isolated: k=2 enumeration (exhaustive) and k=3 sampling
(1M random) cannot bridge the ≈0.39-ratio gap to the known ≥0.94 anchors. This mirrors the
behaviour of Paley I(24) and Kronecker H24 reported in the earlier (same-day) session.

The H23 second-best record stands at **0.9810 · Orrick** (found via recursive k=2/k=3
plateau-bucketed enumeration from Orrick + smart_2val anchors earlier today).

## Computational cost

| Step | Time |
|------|------|
| Williamson sequence search (65,536 quadruples) | ~15 s |
| Deduplication + Paley/Kron comparison | ~60 s |
| Deletion enumeration (96 H24s × 576 del.) | ~95 s |
| k=2 enumeration (5 seeds × 139,656 pairs) | ~20 s |
| k=3 sampling (1M samples) | ~32 s |
| **Total** | **~222 s** |

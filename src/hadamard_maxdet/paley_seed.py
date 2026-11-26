"""Paley-deletion seeds for the (n+1) → n maximal-determinant search.

For an order ``n`` with ``n+1`` admitting a Paley-type-I Hadamard
construction (i.e. ``n+1 = q + 1`` with ``q`` a prime ≡ 3 (mod 4)), every
``n × n`` submatrix obtained by deleting one row and one column from
``H_{n+1}`` is a strong starting point for ±1-determinant search at
order ``n``: the parent has |det| = (n+1)^((n+1)/2) (the Hadamard
bound), so its (n × n) cofactors are large by construction.

This module produces the 576 (= 24·24) deletions for ``n=23`` from
``H_{24}`` and the analogous lists for the small ``n=7``, ``n=11``
sanity-check cases. The top-K by ``|det|`` are returned as starting
seeds for parallel tempering.
"""

from __future__ import annotations

from collections.abc import Iterator

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant


def _delete_row_col(
    entries: list[list[int]], i: int, j: int
) -> list[list[int]]:
    """Return the (n-1)×(n-1) submatrix obtained by deleting row ``i`` and col ``j``."""
    return [
        [entries[r][c] for c in range(len(entries)) if c != j]
        for r in range(len(entries))
        if r != i
    ]


def paley_deletions(n: int) -> Iterator[tuple[int, int, Matrix, int]]:
    """Yield every (row_i, col_j, sub_matrix, signed_det) deletion of H_{n+1}.

    ``H_{n+1}`` is built via :meth:`Matrix.paley_type_i` and therefore
    requires ``n + 1 = q + 1`` with ``q`` a prime ≡ 3 (mod 4).

    The yielded ``Matrix`` has order ``n``; its ``signed_det`` is the
    exact integer determinant computed via :func:`integer_determinant`.
    """
    parent = Matrix.paley_type_i(n + 1)
    parent_entries = parent.entries
    for i in range(n + 1):
        for j in range(n + 1):
            sub = _delete_row_col(parent_entries, i, j)
            d = integer_determinant(sub)
            yield i, j, Matrix(n=n, entries=sub), d


def top_paley_seeds(n: int, k: int = 10) -> list[tuple[int, int, Matrix, int]]:
    """Return the top-``k`` (by |det|) Paley-deletion seeds at order ``n``.

    Sorted in *descending* |det| order. Ties are broken by ``(i, j)``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1; got {k}")
    deletions = list(paley_deletions(n))
    deletions.sort(key=lambda t: (-abs(t[3]), t[0], t[1]))
    return deletions[:k]

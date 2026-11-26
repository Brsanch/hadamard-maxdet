"""Matrix data structure and structured initialisations.

A :class:`Matrix` is a square {-1, +1} matrix carried around with its order
and an efficient copy routine. Structured constructors provided:

  random
      Uniform random ±1 entries.
  sylvester
      Sylvester–Hadamard construction; requires n a power of 2.
  paley_type_i
      Paley type-I Hadamard matrix; requires n = q + 1 where q ≡ 3 (mod 4)
      is a prime power.
  circulant
      Circulant ±1 matrix from a length-n generator row.
  jacobsthal_type_i
      Jacobsthal-like construction for n ≡ 3 (mod 4) (produces a valid
      ±1 matrix even when Paley's prime-power condition fails — it will
      not be Hadamard, but is a strong starting point for search).
"""

from __future__ import annotations

import random as _random
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class Matrix:
    """A square {-1, +1} matrix."""

    n: int
    entries: list[list[int]] = field(default_factory=list)

    @classmethod
    def random(cls, n: int, rng: _random.Random) -> "Matrix":
        """Return a uniform random ±1 matrix of order n."""
        entries = [[1 if rng.random() < 0.5 else -1 for _ in range(n)] for _ in range(n)]
        return cls(n=n, entries=entries)

    @classmethod
    def sylvester(cls, n: int) -> "Matrix":
        """Return the Sylvester Hadamard matrix. Requires n a power of 2."""
        if n < 1 or (n & (n - 1)) != 0:
            raise ValueError(f"Sylvester construction requires n a power of 2, got {n}")
        H: list[list[int]] = [[1]]
        while len(H) < n:
            H = [row + row for row in H] + [row + [-x for x in row] for row in H]
        return cls(n=n, entries=H)

    @classmethod
    def paley_type_i(cls, n: int) -> "Matrix":
        """Paley type-I Hadamard matrix of order n = q + 1 (q ≡ 3 mod 4, q prime).

        Raises ValueError if n does not fit the construction. (Prime-power q
        is possible in general, but we restrict to primes here for clarity.)
        """
        q = n - 1
        if q < 3 or q % 4 != 3 or not _is_prime(q):
            raise ValueError(
                f"Paley I requires n = q + 1 with q ≡ 3 (mod 4) prime; got n={n}"
            )

        # Quadratic-residue indicator chi on F_q; chi(0)=0, chi(x)=1 if QR else -1.
        qr = {pow(x, 2, q) for x in range(1, q)}
        chi = [0] * q
        for x in range(1, q):
            chi[x] = 1 if x in qr else -1

        # Jacobsthal matrix Q of size q × q, with Q_{i,j} = chi(i - j).
        Q = [[chi[(i - j) % q] for j in range(q)] for i in range(q)]

        # Paley I: H has first row (1, 1, ..., 1), first col (1, -1, -1, ..., -1),
        # and H[i+1][j+1] = Q[i][j] + delta_{ij} (so diagonal of the block is +1).
        H = [[1] * n for _ in range(n)]
        for i in range(1, n):
            H[i][0] = -1
        for i in range(q):
            for j in range(q):
                H[i + 1][j + 1] = Q[i][j] if i != j else 1
        return cls(n=n, entries=H)

    @classmethod
    def paley_type_ii(cls, n: int) -> "Matrix":
        """Paley type-II Hadamard matrix of order n = 2(q + 1).

        Requires ``q = n/2 - 1`` to be a prime ≡ 1 (mod 4). The Jacobsthal
        matrix Q is then symmetric (since χ(−1) = +1 for q ≡ 1 mod 4), and
        the bordered S = [[0, 1...], [1, Q]] is a symmetric ±1-with-zero-
        diagonal matrix of order q + 1. We then substitute each entry by
        a 2×2 block:

          0  →  [[ 1, −1], [−1, −1]]   (only on the diagonal)
          +1 →  [[ 1,  1], [ 1, −1]]
          −1 →  [[−1, −1], [−1,  1]]

        Equivalently H = S ⊗ B + I_{q+1} ⊗ C with B = [[1,1],[1,−1]] and
        C = [[1,−1],[−1,−1]]. The result is a symmetric Hadamard matrix
        of order 2(q + 1).
        """
        if n < 4 or n % 4 != 0:
            raise ValueError(
                f"Paley II requires n divisible by 4; got n={n}"
            )
        q = n // 2 - 1
        if q < 5 or q % 4 != 1 or not _is_prime(q):
            raise ValueError(
                f"Paley II requires q = n/2 - 1 ≡ 1 (mod 4) prime; got q={q}"
            )

        qr = {pow(x, 2, q) for x in range(1, q)}
        chi = [0] * q
        for x in range(1, q):
            chi[x] = 1 if x in qr else -1

        Q = [[chi[(i - j) % q] for j in range(q)] for i in range(q)]

        m = q + 1
        S = [[0] * m for _ in range(m)]
        for k in range(1, m):
            S[0][k] = 1
            S[k][0] = 1
        for i in range(q):
            for j in range(q):
                S[i + 1][j + 1] = Q[i][j]

        block_off_pos = [[1, 1], [1, -1]]
        block_off_neg = [[-1, -1], [-1, 1]]
        block_diag = [[1, -1], [-1, -1]]

        H = [[0] * n for _ in range(n)]
        for i in range(m):
            for j in range(m):
                if i == j:
                    block = block_diag
                elif S[i][j] == 1:
                    block = block_off_pos
                else:
                    block = block_off_neg
                H[2 * i][2 * j] = block[0][0]
                H[2 * i][2 * j + 1] = block[0][1]
                H[2 * i + 1][2 * j] = block[1][0]
                H[2 * i + 1][2 * j + 1] = block[1][1]
        return cls(n=n, entries=H)

    @classmethod
    def circulant(cls, first_row: Sequence[int]) -> "Matrix":
        """Return the circulant ±1 matrix whose first row is ``first_row``."""
        n = len(first_row)
        row = list(first_row)
        for v in row:
            if v not in (-1, 1):
                raise ValueError(f"circulant generator must be ±1; got {v}")
        entries = [[row[(j - i) % n] for j in range(n)] for i in range(n)]
        return cls(n=n, entries=entries)

    @classmethod
    def random_circulant(cls, n: int, rng: _random.Random) -> "Matrix":
        """Random ±1 circulant of order n."""
        first = [1 if rng.random() < 0.5 else -1 for _ in range(n)]
        return cls.circulant(first)

    def copy(self) -> "Matrix":
        return Matrix(n=self.n, entries=[list(r) for r in self.entries])

    def to_list(self) -> list[list[int]]:
        return [list(r) for r in self.entries]

    def __iter__(self):
        return iter(self.entries)

    def __len__(self) -> int:
        return self.n


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    r = 3
    while r * r <= n:
        if n % r == 0:
            return False
        r += 2
    return True

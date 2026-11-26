"""Hadamard maximal determinant problem: search for {-1, +1} matrices of
maximum determinant.

For each order n, the question is: what is the largest possible |det M|
over all n × n matrices with entries in {-1, +1}? Hadamard's bound
n^(n/2) is attained only when n = 1, 2 or n ≡ 0 (mod 4) and a Hadamard
matrix of that order exists. For other residue classes sharper bounds
(Barba, Ehlich, Ehlich–Wojtas) apply.
"""

from hadamard_maxdet.io import matrix_to_submission, submission_from_dict
from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.orders import BENCHMARKS, OPEN_IDS, PROVEN_IDS, Order
from hadamard_maxdet.structured import (
    bordered_circulant,
    sa_bordered_circulant,
    sa_skew_circulant,
    sa_two_circulant,
    skew_circulant,
    two_circulant,
)
from hadamard_maxdet.rich_ansatz import (
    bordered_two_circulant,
    double_bordered_circulant,
    paley_deletion_seeds,
    sa_bordered_two_circulant,
    sa_double_bordered_circulant,
    sa_from_paley_deletion,
    sa_from_paley_deletion_budgeted,
)
from hadamard_maxdet.verifier import Verdict, integer_determinant, upper_bound, verify

__all__ = [
    "Order",
    "BENCHMARKS",
    "PROVEN_IDS",
    "OPEN_IDS",
    "Verdict",
    "verify",
    "integer_determinant",
    "upper_bound",
    "Matrix",
    "matrix_to_submission",
    "submission_from_dict",
    # Structured ansätze.
    "two_circulant",
    "bordered_circulant",
    "skew_circulant",
    "sa_two_circulant",
    "sa_bordered_circulant",
    "sa_skew_circulant",
    # Richer ansätze for H23.
    "double_bordered_circulant",
    "sa_double_bordered_circulant",
    "bordered_two_circulant",
    "sa_bordered_two_circulant",
    "paley_deletion_seeds",
    "sa_from_paley_deletion",
    "sa_from_paley_deletion_budgeted",
]

__version__ = "0.1.0"

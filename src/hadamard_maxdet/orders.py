"""Benchmark orders for the Hadamard maximal determinant problem.

All record values for n = 4 through n = 21 are derived programmatically from
OEIS [A003432](https://oeis.org/A003432) via the classical doubling identity

    D(±1, n) = 2^(n-1) · D({0,1}, n-1)

which follows from the bijection ``±1 ↔ {0,1}`` that subtracts the first
row from every other row and drops the first column. The resulting
values match (and cross-reference) the Orrick–Solomon / Brent–Osborn /
Wikipedia record tables.

For n = 4 through 22 these values are the *proven* maxima, established
by exhaustive equivalence-class enumerations culminating in the Brent,
Osborn, Orrick, Zimmermann 2012 paper (orders 19, 37) and earlier work
for smaller orders.

**n = 23 is the smallest actually-open case** per the current Wikipedia
article. H23's `best_known` is left as `None` and is the real target for
any search effort in this repo.

Sequence class notation:
  r0  n ≡ 0 (mod 4)   Hadamard bound n^(n/2); attained iff a Hadamard matrix exists.
  r1  n ≡ 1 (mod 4)   Barba bound sqrt(2n-1)*(n-1)^((n-1)/2); tight only if 2n-1 is a square.
  r2  n ≡ 2 (mod 4)   Ehlich–Wojtas bound (2n-2)*(n-2)^((n-2)/2); tight only if 2n-2 is a sum of two squares.
  r3  n ≡ 3 (mod 4)   Barba (1933) / Ehlich (1964b) bounds.
"""

from __future__ import annotations

from dataclasses import dataclass

# OEIS A003432: max |det| of an n×n {0,1} matrix, indexed by n.
# a(0)..a(21). Values beyond a(21) intentionally omitted pending verification.
A003432: tuple[int, ...] = (
    1,           # a(0)
    1,           # a(1)
    1,           # a(2)
    2,           # a(3)
    3,           # a(4)
    5,           # a(5)
    9,           # a(6)
    32,          # a(7)
    56,          # a(8)
    144,         # a(9)
    320,         # a(10)
    1458,        # a(11)
    3645,        # a(12)
    9477,        # a(13)
    25515,       # a(14)
    131072,      # a(15)
    327680,      # a(16)
    1114112,     # a(17)
    3411968,     # a(18)
    19531250,    # a(19)  — via Wikipedia; n=20 Hadamard so = 20^10 / 2^19
    56640625,    # a(20)  — via Wikipedia record table
    195312500,   # a(21)  — via Wikipedia record table
)


def record_pm1(n: int) -> int:
    """D(±1, n) = 2^(n-1) · A003432[n-1], for n in [1, len(A003432)]."""
    if n < 1 or n > len(A003432):
        raise ValueError(f"A003432 is only tabulated for n-1 in [0, {len(A003432)-1}]")
    return (1 << (n - 1)) * A003432[n - 1]


@dataclass(frozen=True)
class Order:
    """A benchmark order n with its best-known maximal |det|."""

    id: str
    n: int
    residue: int            # n mod 4
    best_known: int | None  # None if unknown to this scaffold
    proven_optimal: bool
    note: str


def _proven(n: int, note: str) -> Order:
    return Order(f"H{n}", n, n % 4, record_pm1(n), True, note)


BENCHMARKS: dict[str, Order] = {
    # Proven-exact small cases (OEIS A003432 · doubling identity).
    "H4":  _proven(4,  "Hadamard bound tight (Sylvester)"),
    "H5":  _proven(5,  "Barba tight (2n-1=9=3^2)"),
    "H6":  _proven(6,  "Ehlich–Wojtas tight (2n-2=10=3^2+1^2)"),
    "H7":  _proven(7,  "Below Barba; exhaustive"),
    "H8":  _proven(8,  "Hadamard bound tight"),
    "H9":  _proven(9,  "OEIS A003432"),
    "H10": _proven(10, "Ehlich–Wojtas tight (2n-2=18=3^2+3^2)"),
    "H11": _proven(11, "OEIS A003432"),
    "H12": _proven(12, "Hadamard bound tight (12^6)"),
    "H13": _proven(13, "Barba tight (2n-1=25=5^2)"),
    "H14": _proven(14, "Ehlich–Wojtas tight (2n-2=26=5^2+1^2)"),
    "H15": _proven(15, "OEIS A003432"),
    "H16": _proven(16, "Hadamard bound tight (16^8)"),
    "H17": _proven(17, "OEIS A003432"),
    "H18": _proven(18, "Ehlich–Wojtas tight (2n-2=34=5^2+3^2)"),
    "H19": _proven(19, "Brent–Orrick–Osborn–Zimmermann 2012"),
    "H20": _proven(20, "Hadamard bound tight (20^10)"),
    "H21": _proven(21, "Proven; ≈90.58% of Barba bound"),
    "H22": _proven(22, "Proven; ≈95.24% of Ehlich–Wojtas bound (42 is not sum of 2 squares)"),

    # *** The real frontier. H23 is the smallest unresolved order. ***
    # Barba upper bound:                              ≈ 3.919e15     (= sqrt(45) · 22^11)
    # Orrick's best-known lower bound (0.7091·Barba): ≈ 2.778e15     (per Browne et al. 2021 Table 1)
    # Koukouvinos–Mitrouli–Seberry / Prop 28 (Browne et al. 2021):
    #     formula prediction:  ≈ 2.654e15     (0.6771·Barba; would-be value if q=5
    #                                          were skew-Paley, which it is NOT)
    #     standard Paley core: = 32,768,000,000,000     (0.0084·Barba; symmetric-Paley
    #                                          regime, proof's Q+Qᵀ=0 step fails)
    #     Browne et al. Table 1 lists 0.7060 in the Prop 28 column at n=23,
    #     which must use a skew-Hadamard substitute for the absent order-6
    #     Hadamard. Not implemented (no order-6 Hadamard exists).
    # In-repo bordered two-circulant SA (2026-04-24): = 2,626,567,700,217,856    (0.6704·Barba)
    # Gap to Orrick: we are ~5.7% below the published lower bound.
    # Smallest open case (no proof of optimality). Orrick et al. 2003 record
    # matrix transcribed from arXiv:math/0304410 and Bareiss-verified at
    # `results/H23_orrick2003_record.json`:
    #     |det| = 2,779,447,296,000,000 = 2^22·3·5^6·67·211 = 0.7092·Barba.
    # `best_known` left None to mark H23 as still proof-open; the record
    # value is the in-tree comment + the verified JSON file.
    "H23": Order("H23", 23, 3, None, False,
                 "Smallest open case. Orrick 2003 record |det| = "
                 "2,779,447,296,000,000 = 0.7092·Barba (exact-verified at "
                 "results/H23_orrick2003_record.json)."),

    # Larger open orders (left for later; fill best-known from Orrick when cross-checked).
    "H27": Order("H27", 27, 3, None, False, "Open; large gap to bound"),
    "H29": Order("H29", 29, 1, None, False, "Open"),
    "H33": Order("H33", 33, 1, None, False, "Open"),
    "H37": Order("H37", 37, 1, None, False, "Brent–Osborn–Orrick–Zimmermann target"),
}

PROVEN_IDS: tuple[str, ...] = tuple(o.id for o in BENCHMARKS.values() if o.proven_optimal)
OPEN_IDS:   tuple[str, ...] = tuple(o.id for o in BENCHMARKS.values() if not o.proven_optimal)
SUBMITTABLE_IDS: tuple[str, ...] = OPEN_IDS


def get(order_id: str) -> Order:
    """Return the Order with the given id, or raise KeyError."""
    try:
        return BENCHMARKS[order_id]
    except KeyError as err:
        known = ", ".join(sorted(BENCHMARKS))
        raise KeyError(
            f"unknown order_id {order_id!r}; known ids: {known}"
        ) from err

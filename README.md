# hadamard-maxdet

Companion code, exact-arithmetic verification routines, and manuscript source for the Hadamard maximal determinant problem: for each positive integer $n$, determine
$$D(n) = \max \bigl\{ |\det M| : M \in \{-1, +1\}^{n \times n} \bigr\}.$$

Hadamard (1893) proved $D(n) \le n^{n/2}$, with equality if and only if $M M^\top = n I_n$ ($n \in \{1, 2\}$ or $n \equiv 0 \pmod 4$). For the remaining congruence classes, classical closed-form upper bounds are:

- **$n \equiv 1 \pmod 4$** — Barba (1933): $D(n) \le \sqrt{2n - 1}\,(n - 1)^{(n-1)/2}$, tight only if $2n - 1$ is a perfect square.
- **$n \equiv 2 \pmod 4$** — Ehlich (1964a) and Wojtas (1964): $D(n) \le (2n - 2)(n - 2)^{(n-2)/2}$, tight only if $2n - 2$ is a sum of two integer squares.
- **$n \equiv 3 \pmod 4$** — Barba (1933) proved the general odd-order bound above, which is never attained for $n \ge 7$; Ehlich (1964b) established a sharper block-matrix bound.

Record values and lower-bound constructions are maintained in the Orrick–Solomon database (`http://www.indiana.edu/~maxdet/`) and Brent's online resource (`https://maths-people.anu.edu.au/~brent/maxdet/`). Proven maxima for $n \le 22$ are tabulated via the doubling identity $D(n) = 2^{n-1} D_{\{0,1\}}(n - 1)$ at [OEIS A003432](https://oeis.org/A003432).

## Manuscript

- **Paper:** [`paper/hadamard_maxdet_algebraic_spectral.pdf`](paper/hadamard_maxdet_algebraic_spectral.pdf) ([LaTeX source](paper/hadamard_maxdet_algebraic_spectral.tex)) — *Algebraic Gram Feasibility and Spectral Branch-and-Bound for the Hadamard Maximal Determinant Problem* (Bryan Sanchez, ORCID [`0009-0000-6080-5407`](https://orcid.org/0009-0000-6080-5407)).
- **Pinned release:** [`v1.0.0`](https://github.com/Brsanch/hadamard-maxdet/releases/tag/v1.0.0) (with archived execution logs in [`docs/verification_logs/`](docs/verification_logs/)).

## Main Verified Results

1. **Infinite-family obstruction (Theorem 1):** For every integer $k \ge 2$ ($n = 4k + 2 \ge 10$, $h = 2k + 1$), the balanced two-block Gram matrix $(h, h)\, K_2 \oplus K_2$ has no realization $G = M M^\top$ over $\{-1, +1\}^{n \times n}$.
2. **Upper bounds below Ehlich–Wojtas for all exceptional $n \equiv 2 \pmod 4$, $n \le 100$ (Theorem 2):**
   $D(22) < 41.386 \cdot 20^{10}$, $D(34) < 65.453 \cdot 32^{16}$, $D(58) < 113.604 \cdot 56^{28}$, $D(70) < 137.663 \cdot 68^{34}$, $D(78) < 153.697 \cdot 76^{38}$, and $D(94) < 185.748 \cdot 92^{46}$, proved via Seidel-gauge monotonicity, capped-spectrum KKT bounds, exact PSD trace Schur bounds, and exact-integer polar-factor Weyl / Davis–Kahan spectral certification.
3. **Full integer-step bound at $n = 34$ (Theorem 3):**
   $$D(34) < 65 \cdot 32^{16} = 65 \cdot 2^{80}$$
   below the Ehlich–Wojtas bound $66 \cdot 32^{16}$, established over all $34 \times 34$ $\{\pm 1\}$-matrices via a 15-dual Cauchy-interlaced Lagrange-dual Seidel branch-and-bound with explicit verified switching-equivalence witnesses, Hasse–Minkowski invariants, exact column-norm equations $x^\top G^{-1} x = 1$, and certified PSD trace bounds.
4. **Exact algebraic closures at $n = 29$ and $n = 33$ (Theorem 4):**
   - At $n = 29$, among all $175$ non-Barba $\{1, 3, 5\}$-supported signatures with $\operatorname{tr}(G^2) < 25697$, all $11$ canonical spread-placement candidates in $[320, 329) \cdot 7^{12} \cdot 2^{28}$ are eliminated in exact arithmetic, including compact integer Farkas certificates for both candidates at $322 \cdot 7^{12} \cdot 2^{28}$.
   - At $n = 33$, every Gram matrix $G = 32 I_{33} + J_{33} + 4E$ whose shifted matrix $G - J_{33}$ is block-diagonal (uniform cliques with off-diagonal entries in $\{-7, -3, 1, 5, 9\}$) attaining $|\det M| \ge 441 \cdot 2^{74}$ (Solomon's lower bound) is excluded in exact integer/rational arithmetic ($9{,}220$ complete $G - J_{33} \succ 0$ configurations above threshold, $3{,}764$ for clique weights $w_i \in \{\pm 1\}$, enumerated across $659{,}380$ visited search-tree nodes).

## Reproduction

Requires Python 3.10+ with `numpy`, `sympy`, `pytest`, and (for figures) `matplotlib`.

```bash
pip install -e .[dev]

# 1. Exact algebraic feasibility tests and Theorems 1–2
pytest tests/test_feasibility.py

# 2. End-to-end exact verification of Theorem 4 (n = 29 and n = 33; ~20 s)
python scripts/verify_r1_closures.py

# 3. End-to-end verification of Theorem 3 (D(34) < 65 * 32^16; ~9 min)
python scripts/prove_h34_below_65.py

# 4. Regenerate vector figures (Figures 1 and 2)
python paper/scripts/make_hadamard_figures.py
```

Archived execution outputs are stored in [`docs/verification_logs/`](docs/verification_logs/):
- [`docs/verification_logs/pytest_test_feasibility.log`](docs/verification_logs/pytest_test_feasibility.log)
- [`docs/verification_logs/verify_r1_closures.log`](docs/verification_logs/verify_r1_closures.log)
- [`docs/verification_logs/prove_h34_below_65.log`](docs/verification_logs/prove_h34_below_65.log)

## Repository Layout

```text
.
├── paper/
│   ├── hadamard_maxdet_algebraic_spectral.tex   Manuscript LaTeX source
│   ├── hadamard_maxdet_algebraic_spectral.pdf   Compiled manuscript PDF
│   ├── figures/                                 Vector PDFs for Figures 1–2
│   └── scripts/make_hadamard_figures.py         Figure-generation script
├── src/hadamard_maxdet/
│   ├── feasibility.py                           Hasse–Minkowski, column-norm, block-Gram DFS, and r = 2 bounds
│   ├── verifier.py                              Exact integer Bareiss determinant verifier
│   ├── orders.py                                Small-order reference table (n <= 22 from OEIS A003432)
│   └── ...                                      Search, switching, and construction utilities
├── scripts/
│   ├── prove_h34_below_65.py                    End-to-end proof script for Theorem 3 (n = 34)
│   └── verify_r1_closures.py                    End-to-end proof script for Theorem 4 (n = 29, 33)
├── tests/
│   └── test_feasibility.py                      Unit tests and Theorems 1–2 verification
└── docs/verification_logs/                      Archived execution logs
```

## Reference Values ($n \le 22$)

Proven maximal determinants for $4 \le n \le 22$ (derived from [OEIS A003432](https://oeis.org/A003432) via $D(n) = 2^{n-1} D_{\{0,1\}}(n - 1)$) together with the smallest open orders:

| ID | $n$ | $n \bmod 4$ | Proven $D(n)$ | Notes |
|----|----:|:-----------:|--------------:|-------|
| H4  |  4 | 0 | 16 | Hadamard |
| H5  |  5 | 1 | 48 | Barba tight ($2n - 1 = 3^2$) |
| H6  |  6 | 2 | 160 | Ehlich–Wojtas tight ($10 = 3^2 + 1^2$) |
| H7  |  7 | 3 | 576 | Exhaustive |
| H8  |  8 | 0 | 4,096 | Hadamard |
| H9  |  9 | 1 | 14,336 | Exhaustive |
| H10 | 10 | 2 | 73,728 | Ehlich–Wojtas tight ($18 = 3^2 + 3^2$) |
| H11 | 11 | 3 | 327,680 | Exhaustive |
| H12 | 12 | 0 | 2,985,984 | Hadamard ($12^6$) |
| H13 | 13 | 1 | 14,929,920 | Barba tight ($2n - 1 = 5^2$) |
| H14 | 14 | 2 | 77,635,584 | Ehlich–Wojtas tight ($26 = 5^2 + 1^2$) |
| H15 | 15 | 3 | 418,037,760 | Orrick (2005) |
| H16 | 16 | 0 | 4,294,967,296 | Hadamard ($16^8$) |
| H17 | 17 | 1 | 21,474,836,480 | Moyssiadis & Kounias (1982) |
| H18 | 18 | 2 | 146,028,888,064 | Ehlich–Wojtas tight ($34 = 5^2 + 3^2$) |
| H19 | 19 | 3 | 894,426,939,392 | Brent, Orrick, Osborn & Zimmermann (2012) |
| H20 | 20 | 0 | 10,240,000,000,000 | Hadamard ($20^{10}$) |
| H21 | 21 | 1 | 59,388,536,406,016 | Chadjipantelis, Kounias & Moyssiadis (1987) |
| H22 | 22 | 2 | 409,600,000,000,000 | Chasiotis, Kounias & Farmakis (2018, 2019); $42 \ne a^2 + b^2$ |
| H23 | 23 | 3 | open | Smallest open order |
| H29 | 29 | 1 | open | $320 \cdot 7^{12} 2^{28} \le D(29) < 329 \cdot 7^{12} 2^{28}$ |
| H33 | 33 | 1 | open | Lower bound $441 \cdot 2^{74}$ (Solomon) |
| H34 | 34 | 2 | open | $D(34) < 65 \cdot 32^{16}$ (Theorem 3) |

## References

- Barba, G. (1933). Intorno al teorema di Hadamard sui determinanti a valore massimo. *Giorn. Mat. Battaglini* 71, 70–86.
- Brent, R. P., Orrick, W. P., Osborn, J.-A. H., & Zimmermann, P. (2012). Maximal determinants and saturated $D$-optimal designs of orders 19 and 37. arXiv:1112.4160.
- Brent, R. P., & Osborn, J.-A. H. (2013). General lower bounds on maximal determinants of binary matrices. *Electron. J. Combin.* 20(2), #P15.
- Brent, R. P., Osborn, J.-A. H., & Smith, W. D. (2016). Probabilistic lower bounds on maximal determinants of binary matrices. *Australas. J. Combin.* 66, 350–364.
- Browne, P., Egan, R., Hegarty, F., & Ó Catháin, P. (2021). A survey of the Hadamard maximal determinant problem. *Electron. J. Combin.* 28(4), #P4.41.
- Chadjipantelis, T., Kounias, S., & Moyssiadis, C. (1987). The maximum determinant of $21 \times 21$ $(+1, -1)$-matrices and $D$-optimal designs. *J. Statist. Plann. Inference* 16, 167–178.
- Chasiotis, V., Kounias, S., & Farmakis, N. (2018, 2019). The $D$-optimal saturated designs of order 22. *Discrete Math.* 341(2), 380–387; Corrigendum, 342(7), 2161.
- Ehlich, H. (1964a). Determinantenabschätzungen für binäre Matrizen. *Math. Z.* 83, 123–132.
- Ehlich, H. (1964b). Determinantenabschätzungen für binäre Matrizen mit $N \equiv 3 \bmod 4$. *Math. Z.* 84, 438–447.
- Hadamard, J. (1893). Résolution d'une question relative aux déterminants. *Bull. Sci. Math.* 17, 240–246.
- Moyssiadis, C., & Kounias, S. (1982). The exact $D$-optimal first order saturated design with 17 observations. *J. Statist. Plann. Inference* 7(1), 13–27.
- Orrick, W. P. (2005). The maximal $\{-1, 1\}$-determinant of order 15. *Metrika* 62(2), 195–219.
- Orrick, W. P., Solomon, B., Dowdeswell, R., & Smith, W. D. (2003). New lower bounds for the maximal determinant problem. arXiv:math/0304410.
- Orrick, W. P., & Solomon, B. (2007). Large-determinant sign matrices of order $4k+1$. *Discrete Math.* 307(2), 226–236.
- Wojtas, M. (1964). On Hadamard's inequality for the determinants of order non-divisible by 4. *Colloq. Math.* 12, 73–83.

## License

Apache-2.0. See [LICENSE](LICENSE).

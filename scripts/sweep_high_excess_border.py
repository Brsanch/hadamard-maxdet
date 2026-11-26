"""Sweep open n ≡ 1 mod 4 cases (H29, H33, H45, H49, H53, H57) with
high-excess Hadamard bordering.

For each n, find a Hadamard predecessor of order n−1 from available
sources (Paley I, Sylvester, Kronecker products), SA-optimize sign-flip
excess, border, compute |det|, save.
"""
import math, sys, json
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from hadamard_maxdet.matrix import Matrix
from hadamard_maxdet.verifier import integer_determinant, upper_bound
from h29_high_excess_border import maximize_excess_sa, border_to_n_plus_1


def get_h_predecessors(n: int) -> list[tuple[str, np.ndarray]]:
    """Return list of (label, Hadamard matrix of order n−1) sources."""
    sources = []
    pred = n - 1
    # Paley I if pred = q+1, q ≡ 3 mod 4 prime
    q = pred - 1
    try:
        from hadamard_maxdet.matrix import _is_prime
        if q >= 3 and q % 4 == 3 and _is_prime(q):
            H = np.array(Matrix.paley_type_i(pred).entries, dtype=np.int64)
            assert (H @ H.T == pred * np.eye(pred, dtype=np.int64)).all()
            sources.append((f"Paley I (q={q})", H))
    except Exception:
        pass
    # Paley II if pred = 2(q+1), q ≡ 1 mod 4 prime
    try:
        if pred % 2 == 0:
            q2 = pred // 2 - 1
            if q2 >= 5 and q2 % 4 == 1 and _is_prime(q2):
                H = np.array(Matrix.paley_type_ii(pred).entries, dtype=np.int64)
                assert (H @ H.T == pred * np.eye(pred, dtype=np.int64)).all()
                sources.append((f"Paley II (q={q2})", H))
    except Exception:
        pass
    # Sylvester if pred is power of 2
    if pred > 0 and (pred & (pred - 1)) == 0:
        try:
            H = np.array(Matrix.sylvester(pred).entries, dtype=np.int64)
            sources.append(("Sylvester", H))
        except Exception:
            pass
    # Kronecker products: H_a ⊗ H_b where a*b = pred
    # Common factorizations:
    factor_options = [(2, pred // 2), (4, pred // 4), (8, pred // 8)]
    for a, b in factor_options:
        if a * b != pred or a < 2 or b < 2:
            continue
        Ha = None
        Hb = None
        # Try to build Ha
        if (a & (a - 1)) == 0:
            Ha = np.array(Matrix.sylvester(a).entries, dtype=np.int64)
        # Try Hb
        try:
            qb = b - 1
            if qb >= 3 and qb % 4 == 3 and _is_prime(qb):
                Hb = np.array(Matrix.paley_type_i(b).entries, dtype=np.int64)
            elif (b & (b - 1)) == 0:
                Hb = np.array(Matrix.sylvester(b).entries, dtype=np.int64)
            elif b % 2 == 0 and _is_prime(b // 2 - 1) and (b // 2 - 1) % 4 == 1:
                Hb = np.array(Matrix.paley_type_ii(b).entries, dtype=np.int64)
        except Exception:
            pass
        if Ha is not None and Hb is not None:
            HK = np.kron(Ha, Hb)
            if (HK @ HK.T == pred * np.eye(pred, dtype=np.int64)).all():
                sources.append((f"H{a} ⊗ H{b}", HK))
    return sources


def attack_n(n: int, *, iters: int = 500_000, n_trials: int = 8) -> dict:
    print(f"\n{'='*60}")
    print(f"H{n}: high-excess H{n-1} bordering")
    print('='*60)
    sources = get_h_predecessors(n)
    if not sources:
        print(f"  NO Hadamard predecessor available for H{n-1}!")
        return {}
    ub = upper_bound(n)
    best = {'abs_det': 0}
    for label, H in sources:
        try:
            print(f"\n  source: {label}, initial excess = {int(H.sum())}")
            e_best, D1, D2 = maximize_excess_sa(H, iters=iters, n_trials=n_trials)
            theo_max = int(round((n - 1) * math.sqrt(n - 1)))
            print(f"  SA best excess: {e_best} (theoretical max ≈ {theo_max})")
            H_opt = D1[:, None] * H * D2[None, :]
            assert int(H_opt.sum()) == e_best
            assert (H_opt @ H_opt.T == (n - 1) * np.eye(n - 1, dtype=np.int64)).all()
            M = border_to_n_plus_1(H_opt)
            det = abs(integer_determinant(M.tolist()))
            ratio = det / ub
            print(f"  |det| = {det}  ratio_UB = {ratio:.4f}")
            if det > best['abs_det']:
                best = {
                    'source': label,
                    'h_excess': e_best,
                    'h_excess_theo_max': theo_max,
                    'abs_det': det,
                    'ratio_to_upper_bound': ratio,
                    'matrix': M.tolist(),
                }
        except Exception as e:
            print(f"  source {label} failed: {e}")
    if best['abs_det'] == 0:
        print(f"  NO viable sources!")
        return {}
    out = {
        'order_id': f'H{n}', 'n': n,
        'source': best['source'] + ' + SA-optimized sign-flip excess + Proposition 13 bordering',
        'h_excess': best['h_excess'],
        'h_excess_theo_max': best['h_excess_theo_max'],
        'abs_det': best['abs_det'],
        'ratio_to_upper_bound': best['ratio_to_upper_bound'],
        'verified_by_bareiss': True,
        'entries': best['matrix'],
    }
    out_path = REPO / "results" / f"H{n}_high_excess_border.json"
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n  >>> H{n} best: {best['source']}, ratio_UB={best['ratio_to_upper_bound']:.4f}")
    return out


if __name__ == "__main__":
    summary = []
    for n in [33, 45, 49, 53, 57]:
        result = attack_n(n)
        if result:
            summary.append((n, result['source'], result['ratio_to_upper_bound']))
    print(f"\n{'='*60}\nSUMMARY (open n ≡ 1 mod 4, high-excess border attack):")
    for n, src, r in summary:
        print(f"  H{n}: ratio_UB = {r:.4f}  via {src}")

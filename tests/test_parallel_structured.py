"""Tests for replica exchange over structured ansatz cores.

These tests verify that ``replica_exchange_bordered_circulant`` reaches
the proven bordered-circulant ansatz ceiling at small orders.

Ansatz ceilings (from exhaustive enumeration over 2^(n-1) cores in the
canonical bordered form, see tests/test_structured.py):

  n =  8: |det| = 4096   (Sylvester-equivalent, Hadamard tight)
  n = 11: |det| = 281600 (NB: the *unrestricted* H11 record is 327,680,
          which is NOT attainable by a bordered-circulant matrix in this
          canonical form — the ansatz ceiling is 281,600.)

Acceptance-rate diagnostic: for a well-tuned ladder, average swap
acceptance should sit roughly in the [0.2, 0.8] band. The exact value
depends on ladder geometry; we use a loose check that *some* swaps fire
and that move acceptance is bounded between 0 and 1 (sanity, not tuning).
"""

from __future__ import annotations

import pytest

from hadamard_maxdet import verify
from hadamard_maxdet.parallel import (
    replica_exchange_bordered_circulant,
    replica_exchange_skew_circulant,
)


def test_pt_bordered_h8_reaches_ansatz_ceiling():
    """n=8: bordered-circulant ceiling = 4096 (Sylvester)."""
    r = replica_exchange_bordered_circulant(
        8,
        n_replicas=8,
        iterations=400,
        exchange_every=50,
        seed=0,
    )
    assert r.abs_det == 4096
    v = verify(r.matrix.entries)
    assert v.valid and v.abs_det == 4096


def test_pt_bordered_h11_reaches_ansatz_ceiling():
    """n=11: bordered-circulant ceiling = 281600 (exhaustive)."""
    r = replica_exchange_bordered_circulant(
        11,
        n_replicas=16,
        iterations=600,
        exchange_every=100,
        seed=1,
    )
    assert r.abs_det == 281600
    v = verify(r.matrix.entries)
    assert v.valid and v.abs_det == 281600


def test_pt_bordered_acceptance_diagnostics():
    """Acceptance-rate diagnostic sanity (run on H8, cheap)."""
    r = replica_exchange_bordered_circulant(
        8,
        n_replicas=6,
        iterations=200,
        exchange_every=40,
        seed=2,
    )
    # Per-replica move acceptance must be a valid rate.
    assert len(r.move_attempts) == 6
    assert len(r.move_accepts) == 6
    for atts, accs in zip(r.move_attempts, r.move_accepts):
        assert atts > 0
        assert 0 <= accs <= atts

    # Hot replicas should accept at least as often as cold ones (loose
    # monotonicity diagnostic). Replicas are stored cold→hot, so
    # move_accepts[hot] >= move_accepts[cold] up to noise.
    cold_rate = r.move_accepts[0] / max(r.move_attempts[0], 1)
    hot_rate = r.move_accepts[-1] / max(r.move_attempts[-1], 1)
    assert hot_rate >= cold_rate - 0.2  # generous tolerance

    # Some swap attempts should fire over the run.
    assert sum(r.swap_attempts) > 0
    # And swap acceptance per pair is a valid rate.
    rates = r.swap_rates()
    for rate in rates:
        assert 0.0 <= rate <= 1.0


def test_pt_bordered_returns_valid_matrix():
    """Output matrix is a verified ±1 matrix at the declared order."""
    r = replica_exchange_bordered_circulant(
        8,
        n_replicas=4,
        iterations=80,
        exchange_every=20,
        seed=3,
    )
    assert r.matrix.n == 8
    v = verify(r.matrix.entries)
    assert v.valid
    assert v.abs_det == r.abs_det


def test_pt_bordered_initial_cores_honoured():
    """Custom ``initial_cores`` is plumbed through to all replicas."""
    n = 8
    n_bits = n - 1
    cores = [[1] * n_bits for _ in range(4)]
    r = replica_exchange_bordered_circulant(
        n,
        n_replicas=4,
        iterations=10,
        exchange_every=5,
        seed=4,
        initial_cores=cores,
    )
    # All-ones core makes the matrix singular (all rows equal under the
    # canonical border), but the run must complete and return a valid
    # ±1 matrix; it just may report a trivially small |det| if the
    # short run never escapes.
    v = verify(r.matrix.entries)
    assert v.valid


def test_pt_bordered_rejects_bad_n():
    with pytest.raises(ValueError):
        replica_exchange_bordered_circulant(1, n_replicas=4, iterations=10)


def test_pt_skew_h7_runs():
    """Skew-circulant PT runs at small odd n. n=7: (n-1)/2 = 3 free bits."""
    r = replica_exchange_skew_circulant(
        7,
        n_replicas=4,
        iterations=80,
        exchange_every=20,
        seed=0,
    )
    # 2^3 = 8 skew generators total; PT should sweep them easily.
    v = verify(r.matrix.entries)
    assert v.valid
    assert r.abs_det > 0


def test_pt_skew_rejects_even_n():
    with pytest.raises(ValueError):
        replica_exchange_skew_circulant(8, n_replicas=4, iterations=10)

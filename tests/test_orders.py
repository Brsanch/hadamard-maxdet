from hadamard_maxdet import BENCHMARKS, OPEN_IDS, PROVEN_IDS
from hadamard_maxdet.orders import get


def test_proven_and_open_partition():
    assert set(PROVEN_IDS).isdisjoint(OPEN_IDS)
    assert set(PROVEN_IDS) | set(OPEN_IDS) == set(BENCHMARKS)


def test_residue_matches_n():
    for o in BENCHMARKS.values():
        assert o.residue == o.n % 4


def test_get_known_and_unknown():
    import pytest

    assert get("H4").n == 4
    with pytest.raises(KeyError):
        get("H999")


def test_proven_orders_have_best_known():
    for oid in PROVEN_IDS:
        o = BENCHMARKS[oid]
        assert o.best_known is not None
        assert o.proven_optimal is True


def test_open_orders_have_no_best_known_inlined():
    # Intentional: we refuse to ship unverified numbers.
    for oid in OPEN_IDS:
        assert BENCHMARKS[oid].best_known is None

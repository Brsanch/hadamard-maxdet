from hadamard_maxdet.search import multi_start, simulated_annealing


def test_sa_reaches_h4_hadamard_bound():
    # n=4 is tiny; SA should reach |det|=16 fast.
    r = simulated_annealing(4, iterations=5_000, seed=0)
    assert r.abs_det == 16


def test_sa_matches_h5_max_48():
    # n=5 max is 48 (Ehlich-Wojtas tight); easy for SA.
    r = simulated_annealing(5, iterations=20_000, seed=0)
    assert r.abs_det == 48


def test_multi_start_non_decreasing():
    r = multi_start(4, n_starts=3, iterations=2_000, base_seed=0)
    assert r.abs_det == 16

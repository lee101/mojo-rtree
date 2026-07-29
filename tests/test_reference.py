"""Independent brute-force checks for the spatial predicates."""

import numpy as np

from mojo_rtree import Index
from mojo_rtree.index import _PARALLEL_BUILD_THRESHOLD


def test_fuzz_intersection_against_brute_force():
    rng = np.random.default_rng(91)
    lower = rng.normal(size=(2000, 2))
    upper = lower + rng.uniform(0, 0.5, size=(2000, 2))
    tree = Index((i, (*lower[i], *upper[i]), None) for i in range(len(lower)))
    for _ in range(50):
        qlo = rng.normal(size=2)
        qhi = qlo + rng.uniform(0, 1, size=2)
        mask = (
            (lower[:, 0] <= qhi[0])
            & (upper[:, 0] >= qlo[0])
            & (lower[:, 1] <= qhi[1])
            & (upper[:, 1] >= qlo[1])
        )
        assert sorted(tree.intersection((*qlo, *qhi))) == np.flatnonzero(mask).tolist()


def test_fuzz_nearest_against_brute_force():
    rng = np.random.default_rng(123)
    points = rng.normal(size=(1500, 2))
    tree = Index((i, (*point, *point), None) for i, point in enumerate(points))
    for query in rng.normal(size=(25, 2)):
        expected = np.argsort(np.sum((points - query) ** 2, axis=1), kind="stable")[:13]
        assert list(tree.nearest(query, 13)) == expected.tolist()


def test_nearest_simd_tail():
    points = np.arange(34, dtype=np.float64).reshape(17, 2)
    tree = Index((i, (*point, *point), None) for i, point in enumerate(points))
    query = np.array([7.25, 8.5])
    expected = np.argsort(np.sum((points - query) ** 2, axis=1), kind="stable")[:5]
    assert list(tree.nearest(query, 5)) == expected.tolist()


def test_parallel_build_threshold_edges():
    for n in (_PARALLEL_BUILD_THRESHOLD - 1, _PARALLEL_BUILD_THRESHOLD + 1):
        x = np.arange(n, dtype=np.float64)
        tree = Index(
            (i, (x[i], x[i] % 97, x[i] + 0.5, x[i] % 97 + 0.5), None)
            for i in range(n)
        )
        assert tree.count((100.25, -1.0, 200.25, 98.0)) == 101
        assert list(tree.nearest((100.1, 3.1), 3)) == [100, 99, 101]
        assert len(tree) == n

"""Behavioral parity with rtree 1.4's in-memory two-dimensional Index."""

from __future__ import annotations

import numpy as np
import pytest

from mojo_rtree import core, index as mojo_index
from rtree import index as upstream_index


def records(n=500, seed=7):
    rng = np.random.default_rng(seed)
    lower = rng.uniform(-100, 100, size=(n, 2))
    upper = lower + rng.uniform(0.01, 8, size=(n, 2))
    ids = rng.permutation(n) + 1000
    return [
        (int(ids[i]), (*lower[i], *upper[i]), {"row": i})
        for i in range(n)
    ]


@pytest.fixture
def pair():
    data = records()
    return mojo_index.Index(iter(data)), upstream_index.Index(iter(data))


def test_random_intersection_parity(pair):
    ours, theirs = pair
    rng = np.random.default_rng(3)
    for _ in range(100):
        lower = rng.uniform(-110, 100, size=2)
        upper = lower + rng.uniform(0, 30, size=2)
        query = (*lower, *upper)
        assert sorted(ours.intersection(query)) == sorted(theirs.intersection(query))


def test_point_intersection_is_inclusive(pair):
    ours, theirs = pair
    for point in [(0.0, 0.0), (-50.0, 72.0), (99.0, -20.0)]:
        assert sorted(ours.intersection(point)) == sorted(theirs.intersection(point))
        assert ours.count(point) == theirs.count(point)


def test_nearest_points_parity():
    rng = np.random.default_rng(11)
    points = rng.normal(size=(300, 2))
    data = [(i, (*point, *point), None) for i, point in enumerate(points)]
    ours = mojo_index.Index(iter(data))
    theirs = upstream_index.Index(iter(data))
    for query in rng.normal(size=(40, 2)):
        assert list(ours.nearest(query, 8)) == list(theirs.nearest(query, 8))


def test_nearest_rectangles_match_distances(pair):
    ours, theirs = pair
    rng = np.random.default_rng(19)
    id_to_bounds = {identifier: bounds for identifier, bounds, _ in records()}

    def distance2(identifier, point):
        xmin, ymin, xmax, ymax = id_to_bounds[identifier]
        dx = max(xmin - point[0], 0.0, point[0] - xmax)
        dy = max(ymin - point[1], 0.0, point[1] - ymax)
        return dx * dx + dy * dy

    for query in rng.uniform(-120, 120, size=(30, 2)):
        a = list(ours.nearest(query, 9))
        b = list(theirs.nearest(query, 9))
        assert sorted(distance2(i, query) for i in a) == pytest.approx(
            sorted(distance2(i, query) for i in b)
        )


def test_objects_and_raw_mode():
    data = [(4, (0, 0, 2, 2), {"name": "square"}), (8, (5, 5, 6, 6), None)]
    ours = mojo_index.Index(iter(data))
    item = next(ours.intersection((1, 1), objects=True))
    assert isinstance(item, mojo_index.Item)
    assert (item.id, item.bbox, item.bounds, item.object) == (
        4,
        [0.0, 0.0, 2.0, 2.0],
        [0.0, 2.0, 0.0, 2.0],
        {"name": "square"},
    )
    assert list(ours.intersection((-1, -1, 10, 10), objects="raw")) == [
        {"name": "square"},
        None,
    ]
    nearest_item = next(ours.nearest((1, 1), objects=True))
    assert (nearest_item.id, nearest_item.object) == (4, {"name": "square"})
    assert list(ours.nearest((1, 1), objects="raw")) == [{"name": "square"}]


def test_noninterleaved_coordinates_match_upstream():
    data = [
        (1, (0, 2, 10, 12), "a"),
        (2, (5, 7, 8, 9), "b"),
        (3, (-4, -1, 3, 6), "c"),
    ]
    ours = mojo_index.Index(iter(data), interleaved=False)
    theirs = upstream_index.Index(iter(data), interleaved=False)
    query = (-2, 6, 4, 11)
    assert sorted(ours.intersection(query)) == sorted(theirs.intersection(query))
    assert ours.bounds == theirs.bounds
    assert ours.get_bounds(True) == theirs.get_bounds(True)


def test_insert_add_delete_and_duplicate_ids():
    ours = mojo_index.Index()
    theirs = upstream_index.Index()
    operations = [
        (3, (0, 0, 1, 1), "first"),
        (3, (5, 5, 6, 6), "second"),
        (9, (2, 2, 4, 4), "third"),
    ]
    for i, (identifier, bounds, obj) in enumerate(operations):
        (ours.add if i == 1 else ours.insert)(identifier, bounds, obj)
        theirs.insert(identifier, bounds, obj)
    ours.delete(3, operations[0][1])
    theirs.delete(3, operations[0][1])
    ours.delete(999, (0, 0, 1, 1))
    theirs.delete(999, (0, 0, 1, 1))
    assert len(ours) == len(theirs) == 2
    assert sorted(ours.intersection((-10, -10, 10, 10))) == sorted(
        theirs.intersection((-10, -10, 10, 10))
    )


def test_mutation_rebuilds_tree():
    tree = mojo_index.Index()
    tree.insert(1, (0, 0))
    assert list(tree.nearest((10, 10))) == [1]
    tree.insert(2, (10, 10))
    assert list(tree.nearest((10, 10))) == [2]
    tree.delete(2, (10, 10))
    assert list(tree.nearest((10, 10))) == [1]


def test_contains_parity(pair):
    ours, theirs = pair
    for query in [(-20, -20, 20, 20), (-200, -200, 200, 200), (1, 2, 3, 4)]:
        assert sorted(ours.contains(query)) == sorted(theirs.contains(query))


def test_vector_intersection_parity(pair):
    ours, theirs = pair
    mins = np.array([[-100, -100], [-10, -5], [50, 50], [0, 0]], dtype=float)
    maxs = np.array([[-50, -50], [20, 25], [90, 90], [1, 1]], dtype=float)
    a_ids, a_counts = ours.intersection_v(mins, maxs)
    b_ids, b_counts = theirs.intersection_v(mins, maxs)
    assert np.array_equal(a_counts, b_counts)
    a_groups = np.split(a_ids, np.cumsum(a_counts)[:-1])
    b_groups = np.split(b_ids, np.cumsum(b_counts)[:-1])
    assert all(sorted(a) == sorted(b) for a, b in zip(a_groups, b_groups))


def test_vector_nearest_parity():
    rng = np.random.default_rng(55)
    points = rng.normal(size=(50, 2))
    data = [
        (i, (point[0], point[1], point[0], point[1]), None)
        for i, point in enumerate(points)
    ]
    ours = mojo_index.Index(iter(data))
    theirs = upstream_index.Index(iter(data))
    mins = rng.normal(size=(3, 2))
    a_ids, a_counts = ours.nearest_v(mins, mins, num_results=4)
    b_ids, b_counts = theirs.nearest_v(mins, mins, num_results=4)
    assert np.array_equal(a_counts, b_counts)
    assert np.array_equal(a_ids, b_ids)


def test_vector_nearest_rectangles_parity(pair):
    ours, theirs = pair
    mins = np.array([[-90, -70], [-5, 10], [60, 40]], dtype=np.float64)
    maxs = mins + np.array([[3, 8], [10, 2], [4, 12]], dtype=np.float64)
    a_ids, a_counts = ours.nearest_v(mins, maxs, num_results=6)
    b_ids, b_counts = theirs.nearest_v(mins, maxs, num_results=6)
    assert np.array_equal(a_counts, b_counts)
    assert np.array_equal(a_ids, b_ids)


def test_result_limit_and_offset(pair):
    ours, _ = pair
    all_ids = list(ours.intersection((-200, -200, 200, 200)))
    ours.result_offset = 7
    ours.result_limit = 11
    assert list(ours.intersection((-200, -200, 200, 200))) == all_ids[7:18]
    assert ours.get_result_offset() == 7
    assert ours.get_result_limit() == 11
    ours.result_offset = 1
    ours.result_limit = 2
    assert len(list(ours.nearest((0, 0), 5))) == 2
    assert len(list(ours.contains((-200, -200, 200, 200)))) == 2


def test_bounds_and_empty_index():
    ours = mojo_index.Index()
    theirs = upstream_index.Index()
    assert ours.bounds == theirs.bounds
    assert len(ours) == ours.get_size() == 0
    assert list(ours.intersection((0, 0, 1, 1))) == []
    assert list(ours.nearest((0, 0), 3)) == []
    ours.insert(1, (-4, 3, 8, 10))
    ours.insert(2, (0, -7, 2, 5))
    assert ours.bounds == [-4.0, -7.0, 8.0, 10.0]
    assert ours.get_bounds(False) == [-4.0, 8.0, -7.0, 10.0]


def test_leaves_are_capacity_bounded_and_cover_every_entry():
    props = mojo_index.Property(index_capacity=8, leaf_capacity=8)
    tree = mojo_index.Index(iter(records(257)), properties=props)
    leaves = tree.leaves()
    ids = [identifier for _, children, _ in leaves for identifier in children]
    assert sorted(ids) == sorted(identifier for identifier, _, _ in records(257))
    assert all(1 <= len(children) <= 8 for _, children, _ in leaves)
    assert len(leaves) > 1


@pytest.mark.parametrize(
    "coordinates",
    [(0, 0, -1, 2), (0, 3, 2, 1), (0, 1, 2), (0, np.nan, 1, 2)],
)
def test_invalid_coordinates_raise(coordinates):
    with pytest.raises(core.RTreeError):
        mojo_index.Index().insert(1, coordinates)


def test_property_api_and_unsupported_modes():
    props = mojo_index.Property(dimension=2, index_capacity=7)
    assert props.get_dimension() == 2
    props.set_leaf_capacity(9)
    assert props.leaf_capacity == 9
    assert props.as_dict()["index_capacity"] == 7
    with pytest.raises(NotImplementedError):
        mojo_index.Index(properties=mojo_index.Property(dimension=3))
    with pytest.raises(NotImplementedError):
        mojo_index.Index("disk-index")
    assert (
        mojo_index.RT_RTree,
        mojo_index.RT_Linear,
        mojo_index.RT_Quadratic,
        mojo_index.RT_Star,
        mojo_index.RT_Memory,
        mojo_index.RT_Disk,
    ) == (0, 0, 1, 2, 0, 1)


def test_interleave_helpers_match_upstream():
    values = [0, 1, 10, 11, 20, 21]
    assert mojo_index.Index.interleave(values) == upstream_index.Index.interleave(values)
    interleaved = [0, 10, 20, 1, 11, 21]
    assert mojo_index.Index.deinterleave(interleaved) == upstream_index.Index.deinterleave(
        interleaved
    )


def test_close_and_valid():
    tree = mojo_index.Index()
    assert tree.valid()
    tree.close()
    assert not tree.valid()
    with pytest.raises(core.RTreeError):
        tree.insert(1, (0, 0))


def test_flush_builds_and_clear_buffer_alias_works():
    tree = mojo_index.Index()
    tree.insert(1, (0, 0))
    tree.flush()
    assert not tree._dirty
    tree.insert(2, (2, 2))
    tree.clearBuffer()
    assert list(tree.intersection((-1, -1, 3, 3))) == [1, 2]


def test_ids_must_fit_ffi_int64_without_lazy_narrowing():
    tree = mojo_index.Index()
    with pytest.raises(OverflowError):
        tree.insert(2**63, (0, 0))
    with pytest.raises(OverflowError):
        tree.insert(-(2**63) - 1, (0, 0))
    tree.insert(-(2**63), (0, 0))
    tree.insert(2**63 - 1, (1, 1))
    assert sorted(tree.intersection((-1, -1, 2, 2))) == [-(2**63), 2**63 - 1]


@pytest.mark.parametrize(
    ("mins", "maxs"),
    [
        ([[0.0, np.nan]], [[1.0, 1.0]]),
        ([[0.0, 0.0]], [[np.inf, 1.0]]),
        ([[2.0, 0.0]], [[1.0, 1.0]]),
    ],
)
def test_vector_queries_reject_invalid_coordinates(mins, maxs):
    tree = mojo_index.Index([(1, (0, 0), None)])
    with pytest.raises(core.RTreeError):
        tree.intersection_v(mins, maxs)
    with pytest.raises(core.RTreeError):
        tree.nearest_v(mins, maxs)


def test_disk_storage_property_is_rejected():
    with pytest.raises(NotImplementedError):
        mojo_index.Index(properties=mojo_index.Property(storage=mojo_index.RT_Disk))

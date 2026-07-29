"""In-memory API compatible with the commonly used subset of ``rtree.index``."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
import math
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np

from ._lib import addr, ensure_parallel_runtime, lib
from .core import RTreeError

RT_RTree = 0
RT_Linear = 0
RT_Quadratic = 1
RT_Star = 2
RT_Memory = 0
RT_Disk = 1
_PARALLEL_BUILD_THRESHOLD = 16384
_I64 = np.dtype(np.int64)
_F64 = np.dtype(np.float64)
_I64_MIN = np.iinfo(np.int64).min
_I64_MAX = np.iinfo(np.int64).max


def _f64_addr(array: np.ndarray) -> int:
    return addr(array, _F64)


def _i64_addr(array: np.ndarray) -> int:
    return addr(array, _I64)


class Property:
    _defaults = {
        "dimension": 2,
        "index_capacity": 100,
        "leaf_capacity": 100,
        "fill_factor": 0.7,
        "variant": RT_Star,
        "type": RT_RTree,
        "storage": RT_Memory,
        "overwrite": False,
        "pagesize": 4096,
        "filename": "",
        "dat_extension": "dat",
        "idx_extension": "idx",
    }

    def __init__(self, handle=None, owned: bool = True, **kwargs: Any) -> None:
        del handle, owned
        for name, value in self._defaults.items():
            setattr(self, name, kwargs.pop(name, value))
        for name, value in kwargs.items():
            setattr(self, name, value)

    def as_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._defaults}

    def initialize_from_dict(self, values: dict[str, Any]) -> None:
        for name, value in values.items():
            setattr(self, name, value)

    def __getattr__(self, name: str):
        if name.startswith("get_"):
            field = name[4:]
            if field in self._defaults:
                return lambda: getattr(self, field)
        if name.startswith("set_"):
            field = name[4:]
            if field in self._defaults:
                return lambda value: setattr(self, field, value)
        raise AttributeError(name)


@dataclass(slots=True)
class Item:
    id: int
    object: object
    bounds: list[float]

    @property
    def bbox(self) -> list[float]:
        return Index.interleave(self.bounds)

    def __lt__(self, other: "Item") -> bool:
        return self.id < other.id

    def __gt__(self, other: "Item") -> bool:
        return self.id > other.id


class Index:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.properties = kwargs.pop("properties", Property())
        self.interleaved = bool(kwargs.pop("interleaved", True))
        if kwargs:
            unknown = next(iter(kwargs))
            raise TypeError(f"unexpected keyword argument {unknown!r}")
        if int(self.properties.dimension) != 2:
            raise NotImplementedError("mojo-rtree currently supports two dimensions")
        if int(self.properties.type) != RT_RTree:
            raise NotImplementedError("only static two-dimensional R-trees are supported")
        if int(self.properties.storage) != RT_Memory:
            raise NotImplementedError("disk-backed indexes are not supported")
        capacity = min(
            int(self.properties.index_capacity), int(self.properties.leaf_capacity)
        )
        if capacity < 2:
            raise RTreeError("index_capacity and leaf_capacity must be at least 2")
        self._capacity = capacity
        self._ids: list[int] = []
        self._bounds_list: list[tuple[float, float, float, float]] = []
        self._objects: list[object] = []
        self._dirty = True
        self._closed = False
        self._root = -1
        self._result_limit = 0
        self._result_offset = 0
        self._bounds_array = np.empty((0, 4), dtype=np.float64)
        self._ids_array = np.empty(0, dtype=np.int64)
        self._node_bounds = np.empty((0, 4), dtype=np.float64)
        self._node_start = np.empty(0, dtype=np.int64)
        self._node_size = np.empty(0, dtype=np.int64)
        self._node_leaf = np.empty(0, dtype=np.int64)
        self._children = np.empty(0, dtype=np.int64)
        self._stack = np.empty(0, dtype=np.int64)
        self._queue_distances = np.empty(0, dtype=np.float64)
        self._query = np.empty(4, dtype=np.float64)
        if args:
            if len(args) != 1:
                raise TypeError("Index accepts one in-memory item stream")
            if isinstance(args[0], (str, bytes)):
                raise NotImplementedError("disk-backed indexes are not supported")
            for identifier, coordinates, obj in args[0]:
                self.insert(identifier, coordinates, obj)

    @staticmethod
    def interleave(deinterleaved: Sequence[float]) -> list[float]:
        values = list(deinterleaved)
        half = len(values) // 2
        return values[::2] + values[1::2] if half and len(values) == half * 2 else values

    @staticmethod
    def deinterleave(interleaved: Sequence[object]) -> list[object]:
        values = list(interleaved)
        half = len(values) // 2
        result: list[object] = []
        for i in range(half):
            result.extend((values[i], values[i + half]))
        return result

    def _coordinates(self, coordinates: Any) -> tuple[float, float, float, float]:
        try:
            size = len(coordinates)
        except TypeError:
            size = -1
        if size == 2:
            minx = float(coordinates[0])
            miny = float(coordinates[1])
            result = (minx, miny, minx, miny)
        elif size == 4:
            if self.interleaved:
                result = (
                    float(coordinates[0]),
                    float(coordinates[1]),
                    float(coordinates[2]),
                    float(coordinates[3]),
                )
            else:
                result = (
                    float(coordinates[0]),
                    float(coordinates[2]),
                    float(coordinates[1]),
                    float(coordinates[3]),
                )
        else:
            values = np.asarray(coordinates, dtype=np.float64).reshape(-1)
            if values.size not in (2, 4):
                raise RTreeError("Coordinates must have dimension 2 or 4")
            return self._coordinates(values)
        if not (
            math.isfinite(result[0])
            and math.isfinite(result[1])
            and math.isfinite(result[2])
            and math.isfinite(result[3])
        ):
            raise RTreeError("Coordinates must be finite")
        if result[0] > result[2] or result[1] > result[3]:
            raise RTreeError("Coordinates must not have minimums more than maximums")
        return result

    def _check_open(self) -> None:
        if self._closed:
            raise RTreeError("index is closed")

    def insert(self, id: int, coordinates: Any, obj: object = None) -> None:
        self._check_open()
        identifier = int(id)
        if identifier < _I64_MIN or identifier > _I64_MAX:
            raise OverflowError("id must fit in a signed 64-bit integer")
        self._ids.append(identifier)
        self._bounds_list.append(self._coordinates(coordinates))
        self._objects.append(obj)
        self._dirty = True

    add = insert

    def delete(self, id: int, coordinates: Any) -> None:
        self._check_open()
        target = self._coordinates(coordinates)
        identifier = int(id)
        for i, (stored_id, bounds) in enumerate(zip(self._ids, self._bounds_list)):
            if stored_id == identifier and bounds == target:
                del self._ids[i]
                del self._bounds_list[i]
                del self._objects[i]
                self._dirty = True
                return

    def _rebuild(self) -> None:
        if not self._dirty:
            return
        n = len(self._ids)
        if n == 0:
            self._bounds_array = np.empty((0, 4), dtype=np.float64)
            self._ids_array = np.empty(0, dtype=np.int64)
            self._node_bounds = np.empty((0, 4), dtype=np.float64)
            self._stack = np.empty(0, dtype=np.int64)
            self._queue_distances = np.empty(0, dtype=np.float64)
            self._root = -1
            self._dirty = False
            return
        self._bounds_array = np.ascontiguousarray(self._bounds_list, dtype=np.float64)
        self._ids_array = np.asarray(self._ids, dtype=np.int64)
        level_count = (n + self._capacity - 1) // self._capacity
        max_nodes = level_count
        max_edges = n
        while level_count > 1:
            max_edges += level_count
            level_count = (level_count + self._capacity - 1) // self._capacity
            max_nodes += level_count
        self._node_bounds = np.empty((max_nodes, 4), dtype=np.float64)
        self._node_start = np.empty(max_nodes, dtype=np.int64)
        self._node_size = np.empty(max_nodes, dtype=np.int64)
        self._node_leaf = np.empty(max_nodes, dtype=np.int64)
        self._children = np.empty(max_edges, dtype=np.int64)
        order = np.empty(n, dtype=np.int64)
        if n >= _PARALLEL_BUILD_THRESHOLD:
            ensure_parallel_runtime()
        self._root = int(
            lib().mrt_build(
                _f64_addr(self._bounds_array),
                n,
                self._capacity,
                _f64_addr(self._node_bounds),
                _i64_addr(self._node_start),
                _i64_addr(self._node_size),
                _i64_addr(self._node_leaf),
                _i64_addr(self._children),
                _i64_addr(order),
            )
        )
        if self._root < 0 or self._root >= max_nodes:
            raise RTreeError("Mojo tree builder returned an invalid root")
        self._stack = np.empty(self._root + 1, dtype=np.int64)
        self._queue_distances = np.empty(self._root + 1, dtype=np.float64)
        self._dirty = False

    def _intersection_positions(self, coordinates: Any) -> np.ndarray:
        self._check_open()
        self._query[:] = self._coordinates(coordinates)
        self._rebuild()
        n = len(self._ids)
        if n == 0:
            return np.empty(0, dtype=np.int64)
        result = np.empty(n, dtype=np.int64)
        count = int(
            lib().mrt_intersection(
                _f64_addr(self._bounds_array),
                _f64_addr(self._node_bounds),
                _i64_addr(self._node_start),
                _i64_addr(self._node_size),
                _i64_addr(self._node_leaf),
                _i64_addr(self._children),
                self._root,
                _f64_addr(self._query),
                _i64_addr(result),
                _i64_addr(self._stack),
            )
        )
        if count < 0 or count > n:
            raise RTreeError("Mojo intersection kernel returned an invalid result count")
        if count and ((result[:count] < 0).any() or (result[:count] >= n).any()):
            raise RTreeError("Mojo intersection kernel returned an invalid item index")
        return result[:count]

    def _intersection_count(self, coordinates: Any) -> int:
        self._check_open()
        self._query[:] = self._coordinates(coordinates)
        self._rebuild()
        if not self._ids:
            return 0
        count = int(
            lib().mrt_intersection_count(
                _f64_addr(self._bounds_array),
                _f64_addr(self._node_bounds),
                _i64_addr(self._node_start),
                _i64_addr(self._node_size),
                _i64_addr(self._node_leaf),
                _i64_addr(self._children),
                self._root,
                _f64_addr(self._query),
                _i64_addr(self._stack),
            )
        )
        if count < 0 or count > len(self._ids):
            raise RTreeError("Mojo intersection kernel returned an invalid result count")
        return count

    def _nearest_positions(self, coordinates: Any, count: int) -> tuple[np.ndarray, np.ndarray]:
        self._check_open()
        self._query[:] = self._coordinates(coordinates)
        self._rebuild()
        k = min(max(int(count), 0), len(self._ids))
        if k == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
        result = np.empty(k, dtype=np.int64)
        distances = np.empty(k, dtype=np.float64)
        found = int(
            lib().mrt_nearest(
                _f64_addr(self._bounds_array),
                _f64_addr(self._node_bounds),
                _i64_addr(self._node_start),
                _i64_addr(self._node_size),
                _i64_addr(self._node_leaf),
                _i64_addr(self._children),
                self._root,
                _f64_addr(self._query),
                k,
                _i64_addr(result),
                _f64_addr(distances),
                _i64_addr(self._stack),
                _f64_addr(self._queue_distances),
            )
        )
        if found < 0 or found > k:
            raise RTreeError("Mojo nearest kernel returned an invalid result count")
        if found and ((result[:found] < 0).any() or (result[:found] >= len(self._ids)).any()):
            raise RTreeError("Mojo nearest kernel returned an invalid item index")
        return result[:found], distances[:found]

    def _window(self, positions: Iterable[int]) -> Iterable[int]:
        offset = max(self._result_offset, 0)
        stop = None
        if self._result_limit > 0:
            stop = offset + self._result_limit
        if isinstance(positions, np.ndarray):
            return positions[offset:stop]
        return islice(positions, offset, stop)

    def _items(
        self, positions: Iterable[int], objects: bool | Literal["raw"]
    ) -> Iterator[Item | int | object]:
        for pos in self._window(positions):
            if objects == "raw":
                yield self._objects[pos]
            elif objects:
                bounds = self._bounds_list[pos]
                yield Item(
                    self._ids[pos],
                    self._objects[pos],
                    [bounds[0], bounds[2], bounds[1], bounds[3]],
                )
            else:
                yield self._ids[pos]

    def intersection(
        self, coordinates: Any, objects: bool | Literal["raw"] = False
    ) -> Iterator[Item | int | object]:
        return self._items(self._intersection_positions(coordinates), objects)

    def count(self, coordinates: Any) -> int:
        return self._intersection_count(coordinates)

    def nearest(
        self,
        coordinates: Any,
        num_results: int = 1,
        objects: bool | Literal["raw"] = False,
    ) -> Iterator[Item | int | object]:
        positions, _ = self._nearest_positions(coordinates, num_results)
        return self._items(positions, objects)

    def contains(
        self, coordinates: Any, objects: bool | Literal["raw"] = False
    ) -> Iterator[Item | int | object]:
        query = self._coordinates(coordinates)
        positions = self._intersection_positions(query)
        contained = (
            pos
            for pos in positions
            if self._bounds_list[int(pos)][0] >= query[0]
            and self._bounds_list[int(pos)][1] >= query[1]
            and self._bounds_list[int(pos)][2] <= query[2]
            and self._bounds_list[int(pos)][3] <= query[3]
        )
        return self._items(contained, objects)

    def intersection_v(self, mins, maxs):
        lo = np.asarray(mins, dtype=np.float64)
        hi = np.asarray(maxs, dtype=np.float64)
        if lo.shape != hi.shape or lo.ndim != 2 or lo.shape[1] != 2:
            raise ValueError("mins and maxs must both have shape (n, 2)")
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise RTreeError("Coordinates must be finite")
        if np.any(lo > hi):
            raise RTreeError("Coordinates must not have minimums more than maximums")
        if len(lo) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint64)
        self._check_open()
        self._rebuild()
        if not self._ids:
            return np.empty(0, dtype=np.int64), np.zeros(len(lo), dtype=np.uint64)
        queries = np.ascontiguousarray(np.column_stack((lo, hi)), dtype=np.float64)
        counts_i64 = np.empty(len(lo), dtype=np.int64)
        lib().mrt_intersection_counts(
            _f64_addr(self._bounds_array),
            _f64_addr(self._node_bounds),
            _i64_addr(self._node_start),
            _i64_addr(self._node_size),
            _i64_addr(self._node_leaf),
            _i64_addr(self._children),
            self._root,
            _f64_addr(queries),
            len(queries),
            _i64_addr(counts_i64),
            _i64_addr(self._stack),
        )
        if (counts_i64 < 0).any() or (counts_i64 > len(self._ids)).any():
            raise RTreeError("Mojo vector kernel returned an invalid result count")
        offsets = np.empty(len(lo), dtype=np.int64)
        offsets[0] = 0
        if len(lo) > 1:
            np.cumsum(counts_i64[:-1], out=offsets[1:])
        total = int(counts_i64.sum())
        if total:
            positions = np.empty(total, dtype=np.int64)
            lib().mrt_intersection_fill(
                _f64_addr(self._bounds_array),
                _f64_addr(self._node_bounds),
                _i64_addr(self._node_start),
                _i64_addr(self._node_size),
                _i64_addr(self._node_leaf),
                _i64_addr(self._children),
                self._root,
                _f64_addr(queries),
                len(queries),
                _i64_addr(offsets),
                _i64_addr(positions),
                _i64_addr(self._stack),
            )
            if (positions < 0).any() or (positions >= len(self._ids)).any():
                raise RTreeError("Mojo vector kernel returned an invalid item index")
            ids = self._ids_array[positions]
        else:
            ids = np.empty(0, dtype=np.int64)
        return ids, counts_i64.astype(np.uint64)

    def nearest_v(
        self,
        mins,
        maxs,
        *,
        num_results=1,
        max_dists=None,
        strict=False,
        return_max_dists=False,
    ):
        if max_dists is not None or strict:
            raise NotImplementedError("max_dists and strict vector nearest are not supported")
        lo = np.asarray(mins, dtype=np.float64)
        hi = np.asarray(maxs, dtype=np.float64)
        if lo.shape != hi.shape or lo.ndim != 2 or lo.shape[1] != 2:
            raise ValueError("mins and maxs must both have shape (n, 2)")
        if not np.isfinite(lo).all() or not np.isfinite(hi).all():
            raise RTreeError("Coordinates must be finite")
        if np.any(lo > hi):
            raise RTreeError("Coordinates must not have minimums more than maximums")
        answers = [
            self._nearest_positions((a[0], a[1], b[0], b[1]), num_results)
            for a, b in zip(lo, hi)
        ]
        counts = np.asarray([len(pos) for pos, _ in answers], dtype=np.uint64)
        ids = np.fromiter(
            (self._ids[int(pos)] for positions, _ in answers for pos in positions),
            dtype=np.int64,
            count=int(counts.sum()),
        )
        if return_max_dists:
            radii = np.asarray(
                [math.sqrt(dist[-1]) if len(dist) else math.inf for _, dist in answers]
            )
            return ids, counts, radii
        return ids, counts

    @property
    def bounds(self) -> list[float]:
        return self.get_bounds(self.interleaved)

    def get_bounds(self, coordinate_interleaved=None):
        if not self._ids:
            limit = np.finfo(np.float64).max
            values = [limit, limit, -limit, -limit]
        else:
            data = np.asarray(self._bounds_list)
            values = [
                float(data[:, 0].min()),
                float(data[:, 1].min()),
                float(data[:, 2].max()),
                float(data[:, 3].max()),
            ]
        use_interleaved = (
            self.interleaved if coordinate_interleaved is None else coordinate_interleaved
        )
        return values if use_interleaved else self.deinterleave(values)

    def leaves(self):
        self._rebuild()
        result = []
        for node in range(self._root + 1):
            if self._node_leaf[node] == 0:
                continue
            start = int(self._node_start[node])
            size = int(self._node_size[node])
            positions = self._children[start : start + size]
            ids = [self._ids[int(pos)] for pos in positions]
            bounds = self._node_bounds[node].tolist()
            if not self.interleaved:
                bounds = self.deinterleave(bounds)
            result.append((node, ids, bounds))
        return result

    def get_size(self) -> int:
        return len(self._ids)

    def __len__(self) -> int:
        return self.get_size()

    def valid(self) -> bool:
        return not self._closed

    def close(self) -> None:
        self._closed = True

    def flush(self) -> None:
        self._rebuild()

    clearBuffer = flush

    def set_result_limit(self, value) -> None:
        self._result_limit = int(value)

    def get_result_limit(self):
        return self._result_limit

    def set_result_offset(self, value) -> None:
        self._result_offset = int(value)

    def get_result_offset(self):
        return self._result_offset

    result_limit = property(get_result_limit, set_result_limit)
    result_offset = property(get_result_offset, set_result_offset)

    def __repr__(self) -> str:
        return f"mojo_rtree.index.Index(bounds={self.bounds}, size={len(self)})"

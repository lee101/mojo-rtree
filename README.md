# mojo-rtree

`mojo-rtree` is a standalone, open-source, in-memory R-tree spatial index. Its
tree construction and spatial query kernels are written in Mojo, with a Python
API modeled on the widely used [`rtree`](https://pypi.org/project/rtree/)
package.

The implementation is a real STR-packed R-tree, not a linear-scan placeholder.
It is intended for workloads that build an index from a set of two-dimensional
points or rectangles and then perform intersection, containment, or
nearest-neighbor queries.

## Install

The repository uses Pixi and pins the Mojo nightly version whose syntax it was
developed against.

```bash
pixi install
pixi run build
pixi run test
```

`pixi install` installs Mojo, NumPy, pytest, and upstream `rtree` for the parity
suite. `pixi run build` produces `dist/libmojo-rtree.so`.

## Usage

```python
from mojo_rtree import index

idx = index.Index()
idx.insert(10, (0.0, 0.0, 2.0, 2.0), obj={"name": "park"})
idx.insert(20, (5.0, 4.0, 7.0, 8.0), obj={"name": "lake"})

assert list(idx.intersection((1.0, 1.0))) == [10]
assert list(idx.nearest((6.0, 6.0), 1)) == [20]

hit = next(idx.intersection((0.0, 0.0, 3.0, 3.0), objects=True))
assert hit.id == 10
assert hit.bbox == [0.0, 0.0, 2.0, 2.0]
assert hit.object == {"name": "park"}
```

Run the example after `pixi run build` with:

```bash
pixi run python your_script.py
```

## API coverage

The covered subset models these parts of `rtree.index`:

- `Index` construction from an item stream, plus `insert`/`add` and `delete`
- `intersection`, `count`, `contains`, and `nearest`
- `objects=False`, `objects=True`, and `objects="raw"` result modes
- `intersection_v` and the basic `nearest_v(..., num_results=...)` form
- `bounds`, `get_bounds`, `get_size`, `len`, `leaves`, `valid`, `flush`, and
  `close`
- result limits and offsets
- interleaved and deinterleaved coordinate layouts
- `Property`, `Item`, `RTreeError`, and the common R-tree constants

The implementation currently supports two-dimensional, in-memory, static
R-trees. Inserts and deletes are supported by rebuilding the packed tree lazily
at the next query. Disk indexes, custom storage, TPR trees, multidimensional
indexes, custom object serialization, and `nearest_v`'s `strict`/`max_dists`
options are not covered. Upstream may return more than `num_results` when
nearest-neighbor distances tie; `mojo-rtree` returns exactly the requested
number (or the index size if smaller).

The tests compare results with upstream `rtree` on randomized data, coordinate
layout and object-return behavior, mutations, vector queries, and edge cases.
They also compare intersection and nearest results with an independent
NumPy brute-force reference.

## Benchmarks

Measured by running `pixi run bench` on this checkout on an Intel Xeon
E5-2697 v4 at 2.30 GHz, Linux x86-64, Python 3.13.14. Values are the best
repeat on the same generated data. The ratio is upstream time divided by Mojo
time, so ratios below one mean Mojo is slower.

| workload | mojo-rtree | rtree 1.4.1 | upstream / Mojo |
|---|---:|---:|---:|
| STR bulk build + first query (200k boxes) | 511.7 ms | 990.2 ms | 1.93x faster |
| 2,000 intersection counts | 84.0 ms | 43.0 ms | 0.51x slower |
| 1,000 nearest-10 queries | 85.9 ms | 63.5 ms | 0.74x slower |
| intersection_v (1,000 boxes) | 4.3 ms | 16.6 ms | 3.90x faster |

This run shows faster bulk construction and batched intersection, but slower
scalar intersection counts and nearest queries. Large STR builds sort
independent y slices in parallel after the x sort, while smaller builds stay
serial. Scalar counts use a count-only kernel. Nearest queries use a
distance-ordered node queue. Batched intersection keeps all query boxes inside
compiled count-and-fill passes, avoiding per-query Python and FFI overhead.

No GPU path is included or benchmarked.

## How it works

Construction uses sort-tile-recursive packing. Entries are sorted by rectangle
center along x, divided into slices, sorted by y inside each slice, and packed
into bounded leaf nodes. Large independent slice sorts are parallelized behind
a size threshold. Bounds copies and extensions use native-width float64 SIMD
with scalar remainder handling. The same process packs successive internal
levels until one root remains. Intersection traversal prunes non-overlapping
node bounding boxes. Nearest traversal visits nodes in minimum-distance order
and stops when the closest remaining node exceeds the current kth-best result.

Python owns every allocation. Item bounds are a contiguous `float64[n, 4]`
array in `[xmin, ymin, xmax, ymax]` order. Nodes use structure-of-arrays
buffers for bounds, child offsets, child counts, and leaf flags, while child
references occupy one contiguous `int64` array. No Mojo allocation crosses the
ABI.

The C ABI exports non-parametric functions from one Mojo compilation unit.
Validated, contiguous NumPy buffer addresses cross the private `ctypes` ABI and
are reconstructed as mutable typed pointers inside Mojo. Python retains strong
references to every buffer for the duration of each synchronous call. Scalar
queries make one FFI call; `intersection_v` uses one batched count call followed
by one packed-result fill call.

Index instances are not safe for concurrent queries because scratch buffers are
reused.

## Development

```bash
pixi run build
pixi run test
pixi run bench
```

The benchmark task holds a machine-wide lock so concurrent jobs do not distort
the published timings.

"""Honest in-memory spatial-index benchmarks against libspatialindex-backed rtree."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from mojo_rtree import Index as MojoIndex  # noqa: E402
from rtree.index import Index as UpstreamIndex  # noqa: E402


def best_time(fn, repeat=3):
    best = math.inf
    value = None
    for _ in range(repeat):
        start = time.perf_counter()
        value = fn()
        best = min(best, time.perf_counter() - start)
    return best, value


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def format_time(seconds):
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def main():
    rng = np.random.default_rng(2026)
    n = 200_000
    lower = rng.uniform(-10_000, 10_000, size=(n, 2))
    upper = lower + rng.uniform(0.01, 25, size=(n, 2))
    data = [
        (i, (lower[i, 0], lower[i, 1], upper[i, 0], upper[i, 1]), None)
        for i in range(n)
    ]

    query_lower = rng.uniform(-10_000, 9_900, size=(2_000, 2))
    query_upper = query_lower + rng.uniform(10, 100, size=(2_000, 2))
    queries = [(*a, *b) for a, b in zip(query_lower, query_upper)]
    points = rng.uniform(-10_000, 10_000, size=(1_000, 2))

    mojo = MojoIndex(iter(data))
    upstream = UpstreamIndex(iter(data))
    assert mojo.count((-20_000, -20_000, 20_000, 20_000)) == n

    cases = []

    def build_mojo():
        tree = MojoIndex(iter(data))
        return tree.count((-1, -1, 1, 1))

    def build_upstream():
        tree = UpstreamIndex(iter(data))
        return tree.count((-1, -1, 1, 1))

    cases.append(("STR bulk build + first query (200k boxes)", build_mojo, build_upstream, 3))

    def intersections_mojo():
        return sum(mojo.count(query) for query in queries)

    def intersections_upstream():
        return sum(upstream.count(query) for query in queries)

    cases.append(("2,000 intersection counts", intersections_mojo, intersections_upstream, 5))

    def nearest_mojo():
        return sum(sum(1 for _ in mojo.nearest(point, 10)) for point in points)

    def nearest_upstream():
        return sum(sum(1 for _ in upstream.nearest(point, 10)) for point in points)

    cases.append(("1,000 nearest-10 queries", nearest_mojo, nearest_upstream, 5))

    vector_mins = query_lower[:1_000]
    vector_maxs = query_upper[:1_000]

    def vector_mojo():
        return int(mojo.intersection_v(vector_mins, vector_maxs)[1].sum())

    def vector_upstream():
        return int(upstream.intersection_v(vector_mins, vector_maxs)[1].sum())

    cases.append(("intersection_v (1,000 boxes)", vector_mojo, vector_upstream, 5))

    rows = []
    for name, ours_fn, theirs_fn, repeat in cases:
        ours_fn()
        theirs_fn()
        ours_time, ours_value = best_time(ours_fn, repeat)
        theirs_time, theirs_value = best_time(theirs_fn, repeat)
        assert ours_value == theirs_value, (name, ours_value, theirs_value)
        ratio = theirs_time / ours_time
        result = "faster" if ratio >= 1 else "slower"
        rows.append((name, ours_time, theirs_time, ratio, result))

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.machine()}; Python {platform.python_version()}")
    print()
    print("| workload | mojo-rtree | rtree 1.4.1 | upstream / Mojo |")
    print("|---|---:|---:|---:|")
    for name, ours_time, theirs_time, ratio, result in rows:
        print(
            f"| {name} | {format_time(ours_time)} | {format_time(theirs_time)} "
            f"| {ratio:.2f}x {result} |"
        )


if __name__ == "__main__":
    main()

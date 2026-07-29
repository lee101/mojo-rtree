"""Flat, caller-allocated, two-dimensional STR-packed R-tree."""

from std.algorithm import parallelize
from std.math import sqrt
from std.sys.info import simd_width_of as simdwidthof

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime INF = 1.7976931348623157e308
comptime PARALLEL_BUILD_THRESHOLD = 16384


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def center(bounds: FPtr, item: Int, axis: Int) -> Float64:
    return bounds[item * 4 + axis] + bounds[item * 4 + axis + 2]


def greater(bounds: FPtr, a: Int64, b: Int64, axis: Int) -> Bool:
    var ai = Int(a)
    var bi = Int(b)
    var av = center(bounds, ai, axis)
    var bv = center(bounds, bi, axis)
    return av > bv or (av == bv and ai > bi)


def sift_down(
    order: IPtr,
    lo: Int,
    count: Int,
    root: Int,
    bounds: FPtr,
    axis: Int,
):
    var r = root
    while r * 2 + 1 < count:
        var child = r * 2 + 1
        if child + 1 < count and greater(
            bounds, order[lo + child + 1], order[lo + child], axis
        ):
            child += 1
        if not greater(bounds, order[lo + child], order[lo + r], axis):
            return
        var tmp = order[lo + r]
        order[lo + r] = order[lo + child]
        order[lo + child] = tmp
        r = child


def heap_sort(order: IPtr, lo: Int, hi: Int, bounds: FPtr, axis: Int):
    var count = hi - lo
    if count < 2:
        return
    var start = (count - 2) // 2
    while start >= 0:
        sift_down(order, lo, count, start, bounds, axis)
        start -= 1
    var end = count - 1
    while end > 0:
        var tmp = order[lo]
        order[lo] = order[lo + end]
        order[lo + end] = tmp
        sift_down(order, lo, end, 0, bounds, axis)
        end -= 1


def parallel_sort_slices(
    order_addr: Int,
    count: Int,
    bounds_addr: Int,
    slice_capacity: Int,
    slice_count: Int,
):
    @parameter
    def sort_slice(slice_id: Int):
        var lo = slice_id * slice_capacity
        var hi = min(lo + slice_capacity, count)
        heap_sort(ip(order_addr), lo, hi, fp(bounds_addr), 1)

    parallelize[sort_slice](slice_count)


def str_order(order: IPtr, count: Int, bounds: FPtr, capacity: Int):
    if count < 2:
        return
    var groups = (count + capacity - 1) // capacity
    var slices = Int(sqrt(Float64(groups)))
    if slices < 1:
        slices = 1
    while slices * slices < groups:
        slices += 1
    var groups_per_slice = (groups + slices - 1) // slices
    var slice_capacity = groups_per_slice * capacity
    heap_sort(order, 0, count, bounds, 0)
    var slice_count = (count + slice_capacity - 1) // slice_capacity

    @parameter
    def sort_slice(slice_id: Int):
        var lo = slice_id * slice_capacity
        var hi = min(lo + slice_capacity, count)
        heap_sort(order, lo, hi, bounds, 1)

    if count >= PARALLEL_BUILD_THRESHOLD and slice_count > 1:
        parallel_sort_slices(
            Int(order), count, Int(bounds), slice_capacity, slice_count
        )
    else:
        for slice_id in range(slice_count):
            sort_slice(slice_id)


def copy_bounds(src: FPtr, src_id: Int, dst: FPtr, dst_id: Int):
    comptime W = simdwidthof[DType.float64]()
    var sb = src_id * 4
    var db = dst_id * 4
    var d = 0
    while d + W <= 4:
        dst.store(db + d, src.load[width=W](sb + d))
        d += W
    while d < 4:
        dst[db + d] = src[sb + d]
        d += 1


def extend_bounds(dst: FPtr, dst_id: Int, src: FPtr, src_id: Int):
    var db = dst_id * 4
    var sb = src_id * 4
    comptime W = simdwidthof[DType.float64]()
    comptime if W == 4:
        var current = dst.load[width=W](db)
        var incoming = src.load[width=W](sb)
        var merged = min(current, incoming)
        var upper = max(current, incoming)
        merged[2] = upper[2]
        merged[3] = upper[3]
        dst.store(db, merged)
    else:
        dst[db] = min(dst[db], src[sb])
        dst[db + 1] = min(dst[db + 1], src[sb + 1])
        dst[db + 2] = max(dst[db + 2], src[sb + 2])
        dst[db + 3] = max(dst[db + 3], src[sb + 3])


def build_tree(
    item_bounds: FPtr,
    n: Int,
    capacity: Int,
    node_bounds: FPtr,
    node_start: IPtr,
    node_size: IPtr,
    node_leaf: IPtr,
    children: IPtr,
    order: IPtr,
) -> Int:
    if n <= 0:
        return -1
    for i in range(n):
        order[i] = Int64(i)
    str_order(order, n, item_bounds, capacity)

    var leaf_count = (n + capacity - 1) // capacity

    @parameter
    def build_leaf(node: Int):
        var pos = node * capacity
        var take = min(capacity, n - pos)
        node_start[node] = Int64(pos)
        node_size[node] = Int64(take)
        node_leaf[node] = 1
        var first = Int(order[pos])
        copy_bounds(item_bounds, first, node_bounds, node)
        for j in range(take):
            var item = Int(order[pos + j])
            children[pos + j] = Int64(item)
            if j > 0:
                extend_bounds(node_bounds, node, item_bounds, item)

    for node in range(leaf_count):
        build_leaf(node)

    var node_cursor = leaf_count
    var edge_cursor = n
    var level_start = 0
    var level_count = node_cursor
    while level_count > 1:
        for i in range(level_count):
            order[i] = Int64(level_start + i)
        str_order(order, level_count, node_bounds, capacity)
        var parent_start = node_cursor
        var pos = 0
        while pos < level_count:
            var take = min(capacity, level_count - pos)
            node_start[node_cursor] = Int64(edge_cursor)
            node_size[node_cursor] = Int64(take)
            node_leaf[node_cursor] = 0
            var first = Int(order[pos])
            copy_bounds(node_bounds, first, node_bounds, node_cursor)
            for j in range(take):
                var child = Int(order[pos + j])
                children[edge_cursor] = Int64(child)
                edge_cursor += 1
                if j > 0:
                    extend_bounds(node_bounds, node_cursor, node_bounds, child)
            node_cursor += 1
            pos += take
        level_start = parent_start
        level_count = node_cursor - parent_start
    return node_cursor - 1


def overlaps(a: FPtr, ai: Int, b: FPtr, bi: Int) -> Bool:
    var aa = ai * 4
    var bb = bi * 4
    comptime W = simdwidthof[DType.float64]()
    comptime if W == 4:
        var av = a.load[width=W](aa)
        var bv = b.load[width=W](bb)
        var separated = av.gt(bv.shuffle[2, 3, 0, 1]()) | bv.gt(
            av.shuffle[2, 3, 0, 1]()
        )
        return not (separated[0] or separated[1])
    else:
        return (
            a[aa] <= b[bb + 2]
            and a[aa + 2] >= b[bb]
            and a[aa + 1] <= b[bb + 3]
            and a[aa + 3] >= b[bb + 1]
        )


def intersection_query(
    item_bounds: FPtr,
    node_bounds: FPtr,
    node_start: IPtr,
    node_size: IPtr,
    node_leaf: IPtr,
    children: IPtr,
    root: Int,
    query: FPtr,
    result: IPtr,
    stack: IPtr,
) -> Int:
    if root < 0:
        return 0
    var top = 1
    stack[0] = Int64(root)
    var found = 0
    while top > 0:
        top -= 1
        var node = Int(stack[top])
        if not overlaps(node_bounds, node, query, 0):
            continue
        var start = Int(node_start[node])
        var count = Int(node_size[node])
        if node_leaf[node] != 0:
            for j in range(count):
                var item = Int(children[start + j])
                if overlaps(item_bounds, item, query, 0):
                    result[found] = Int64(item)
                    found += 1
        else:
            for j in range(count):
                stack[top] = children[start + j]
                top += 1
    return found


def intersection_count(
    item_bounds: FPtr,
    node_bounds: FPtr,
    node_start: IPtr,
    node_size: IPtr,
    node_leaf: IPtr,
    children: IPtr,
    root: Int,
    query: FPtr,
    stack: IPtr,
) -> Int:
    if root < 0:
        return 0
    var top = 1
    stack[0] = Int64(root)
    var found = 0
    while top > 0:
        top -= 1
        var node = Int(stack[top])
        if not overlaps(node_bounds, node, query, 0):
            continue
        var start = Int(node_start[node])
        var count = Int(node_size[node])
        if node_leaf[node] != 0:
            for j in range(count):
                var item = Int(children[start + j])
                if overlaps(item_bounds, item, query, 0):
                    found += 1
        else:
            for j in range(count):
                stack[top] = children[start + j]
                top += 1
    return found


def rectangle_distance2(a: FPtr, ai: Int, b: FPtr, bi: Int) -> Float64:
    var aa = ai * 4
    var bb = bi * 4
    var dx = 0.0
    var dy = 0.0
    if a[aa + 2] < b[bb]:
        dx = b[bb] - a[aa + 2]
    elif b[bb + 2] < a[aa]:
        dx = a[aa] - b[bb + 2]
    if a[aa + 3] < b[bb + 1]:
        dy = b[bb + 1] - a[aa + 3]
    elif b[bb + 3] < a[aa + 1]:
        dy = a[aa + 1] - b[bb + 3]
    return dx * dx + dy * dy


def heap_less(a_dist: Float64, a_node: Int64, b_dist: Float64, b_node: Int64) -> Bool:
    return a_dist < b_dist or (a_dist == b_dist and a_node < b_node)


def heap_push(
    nodes: IPtr,
    node_distances: FPtr,
    size: Int,
    node: Int64,
    distance: Float64,
):
    var pos = size
    while pos > 0:
        var parent = (pos - 1) // 2
        if not heap_less(distance, node, node_distances[parent], nodes[parent]):
            break
        nodes[pos] = nodes[parent]
        node_distances[pos] = node_distances[parent]
        pos = parent
    nodes[pos] = node
    node_distances[pos] = distance


def heap_pop(nodes: IPtr, node_distances: FPtr, size: Int):
    if size <= 1:
        return
    var node = nodes[size - 1]
    var distance = node_distances[size - 1]
    var pos = 0
    while pos * 2 + 1 < size - 1:
        var child = pos * 2 + 1
        if child + 1 < size - 1 and heap_less(
            node_distances[child + 1],
            nodes[child + 1],
            node_distances[child],
            nodes[child],
        ):
            child += 1
        if not heap_less(
            node_distances[child], nodes[child], distance, node
        ):
            break
        nodes[pos] = nodes[child]
        node_distances[pos] = node_distances[child]
        pos = child
    nodes[pos] = node
    node_distances[pos] = distance


def nearest_query(
    item_bounds: FPtr,
    node_bounds: FPtr,
    node_start: IPtr,
    node_size: IPtr,
    node_leaf: IPtr,
    children: IPtr,
    root: Int,
    query: FPtr,
    k: Int,
    result: IPtr,
    distances: FPtr,
    queue: IPtr,
    queue_distances: FPtr,
) -> Int:
    if root < 0 or k <= 0:
        return 0
    comptime W = simdwidthof[DType.float64]()
    var i = 0
    while i + W <= k:
        result.store(i, SIMD[DType.int64, W](-1))
        distances.store(i, SIMD[DType.float64, W](INF))
        i += W
    while i < k:
        result[i] = -1
        distances[i] = INF
        i += 1
    var queue_size = 1
    queue[0] = Int64(root)
    queue_distances[0] = rectangle_distance2(node_bounds, root, query, 0)
    var found = 0
    while queue_size > 0:
        var node = Int(queue[0])
        var node_dist = queue_distances[0]
        heap_pop(queue, queue_distances, queue_size)
        queue_size -= 1
        if found == k and node_dist > distances[k - 1]:
            break
        var start = Int(node_start[node])
        var count = Int(node_size[node])
        if node_leaf[node] != 0:
            for j in range(count):
                var item = Int(children[start + j])
                var dist = rectangle_distance2(item_bounds, item, query, 0)
                if found == k and dist > distances[k - 1]:
                    continue
                var used = min(found, k - 1)
                while used > 0 and (
                    dist < distances[used - 1]
                    or (
                        dist == distances[used - 1]
                        and item < Int(result[used - 1])
                    )
                ):
                    if used < k:
                        distances[used] = distances[used - 1]
                        result[used] = result[used - 1]
                    used -= 1
                distances[used] = dist
                result[used] = Int64(item)
                if found < k:
                    found += 1
        else:
            for j in range(count):
                var child = Int(children[start + j])
                var distance = rectangle_distance2(
                    node_bounds, child, query, 0
                )
                if found < k or distance <= distances[k - 1]:
                    heap_push(
                        queue,
                        queue_distances,
                        queue_size,
                        Int64(child),
                        distance,
                    )
                    queue_size += 1
    return found


@export("mrt_build")
def mrt_build(
    item_bounds_addr: Int,
    n: Int,
    capacity: Int,
    node_bounds_addr: Int,
    node_start_addr: Int,
    node_size_addr: Int,
    node_leaf_addr: Int,
    children_addr: Int,
    order_addr: Int,
) abi("C") -> Int:
    return build_tree(
        fp(item_bounds_addr),
        n,
        capacity,
        fp(node_bounds_addr),
        ip(node_start_addr),
        ip(node_size_addr),
        ip(node_leaf_addr),
        ip(children_addr),
        ip(order_addr),
    )


@export("mrt_intersection")
def mrt_intersection(
    item_bounds_addr: Int,
    node_bounds_addr: Int,
    node_start_addr: Int,
    node_size_addr: Int,
    node_leaf_addr: Int,
    children_addr: Int,
    root: Int,
    query_addr: Int,
    result_addr: Int,
    stack_addr: Int,
) abi("C") -> Int:
    return intersection_query(
        fp(item_bounds_addr),
        fp(node_bounds_addr),
        ip(node_start_addr),
        ip(node_size_addr),
        ip(node_leaf_addr),
        ip(children_addr),
        root,
        fp(query_addr),
        ip(result_addr),
        ip(stack_addr),
    )


@export("mrt_intersection_counts")
def mrt_intersection_counts(
    item_bounds_addr: Int,
    node_bounds_addr: Int,
    node_start_addr: Int,
    node_size_addr: Int,
    node_leaf_addr: Int,
    children_addr: Int,
    root: Int,
    queries_addr: Int,
    m: Int,
    counts_addr: Int,
    stack_addr: Int,
) abi("C"):
    var item_bounds = fp(item_bounds_addr)
    var node_bounds = fp(node_bounds_addr)
    var node_start = ip(node_start_addr)
    var node_size = ip(node_size_addr)
    var node_leaf = ip(node_leaf_addr)
    var children = ip(children_addr)
    var queries = fp(queries_addr)
    var counts = ip(counts_addr)
    var stack = ip(stack_addr)
    for q in range(m):
        counts[q] = Int64(
            intersection_count(
                item_bounds,
                node_bounds,
                node_start,
                node_size,
                node_leaf,
                children,
                root,
                queries + q * 4,
                stack,
            )
        )


@export("mrt_intersection_count")
def mrt_intersection_count(
    item_bounds_addr: Int,
    node_bounds_addr: Int,
    node_start_addr: Int,
    node_size_addr: Int,
    node_leaf_addr: Int,
    children_addr: Int,
    root: Int,
    query_addr: Int,
    stack_addr: Int,
) abi("C") -> Int:
    return intersection_count(
        fp(item_bounds_addr),
        fp(node_bounds_addr),
        ip(node_start_addr),
        ip(node_size_addr),
        ip(node_leaf_addr),
        ip(children_addr),
        root,
        fp(query_addr),
        ip(stack_addr),
    )


@export("mrt_intersection_fill")
def mrt_intersection_fill(
    item_bounds_addr: Int,
    node_bounds_addr: Int,
    node_start_addr: Int,
    node_size_addr: Int,
    node_leaf_addr: Int,
    children_addr: Int,
    root: Int,
    queries_addr: Int,
    m: Int,
    offsets_addr: Int,
    result_addr: Int,
    stack_addr: Int,
) abi("C"):
    var item_bounds = fp(item_bounds_addr)
    var node_bounds = fp(node_bounds_addr)
    var node_start = ip(node_start_addr)
    var node_size = ip(node_size_addr)
    var node_leaf = ip(node_leaf_addr)
    var children = ip(children_addr)
    var queries = fp(queries_addr)
    var offsets = ip(offsets_addr)
    var result = ip(result_addr)
    var stack = ip(stack_addr)
    for q in range(m):
        _ = intersection_query(
            item_bounds,
            node_bounds,
            node_start,
            node_size,
            node_leaf,
            children,
            root,
            queries + q * 4,
            result + Int(offsets[q]),
            stack,
        )


@export("mrt_nearest")
def mrt_nearest(
    item_bounds_addr: Int,
    node_bounds_addr: Int,
    node_start_addr: Int,
    node_size_addr: Int,
    node_leaf_addr: Int,
    children_addr: Int,
    root: Int,
    query_addr: Int,
    k: Int,
    result_addr: Int,
    distances_addr: Int,
    queue_addr: Int,
    queue_distances_addr: Int,
) abi("C") -> Int:
    return nearest_query(
        fp(item_bounds_addr),
        fp(node_bounds_addr),
        ip(node_start_addr),
        ip(node_size_addr),
        ip(node_leaf_addr),
        ip(children_addr),
        root,
        fp(query_addr),
        k,
        ip(result_addr),
        fp(distances_addr),
        ip(queue_addr),
        fp(queue_distances_addr),
    )

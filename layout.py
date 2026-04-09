from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClusterLayout:
    cluster_count: int
    selected_rows: int
    dim_rows: int
    intervals: tuple[tuple[int, int], ...]
    range_sizes: tuple[int, ...]
    gap_sizes: tuple[int, ...]


def split_evenly(total: int, parts: int) -> tuple[int, ...]:
    if parts <= 0:
        return tuple()
    base, remainder = divmod(total, parts)
    return tuple(base + (1 if idx < remainder else 0) for idx in range(parts))


def build_cluster_layout(
    dim_rows: int,
    build_size: int,
    cluster_count: int,
) -> ClusterLayout:
    if cluster_count < 2:
        raise ValueError("cluster count must be at least 2")
    if build_size > dim_rows:
        raise ValueError("build size cannot exceed dimension rows")
    if cluster_count > build_size:
        raise ValueError("cluster count cannot exceed build size")

    range_sizes = split_evenly(build_size, cluster_count)
    excluded_rows = dim_rows - build_size
    gap_sizes = split_evenly(excluded_rows, cluster_count - 1)

    intervals: list[tuple[int, int]] = []
    start = 1
    for idx, range_size in enumerate(range_sizes):
        end = start + range_size - 1
        intervals.append((start, end))
        if idx < len(gap_sizes):
            start = end + 1 + gap_sizes[idx]

    if intervals[0][0] != 1:
        raise AssertionError("first interval must start at key 1")
    if intervals[-1][1] != dim_rows:
        raise AssertionError("last interval must end at the largest key")

    return ClusterLayout(
        cluster_count=cluster_count,
        selected_rows=build_size,
        dim_rows=dim_rows,
        intervals=tuple(intervals),
        range_sizes=tuple(range_sizes),
        gap_sizes=tuple(gap_sizes),
    )


def mask_column_name(cluster_count: int) -> str:
    return f"mask_c{cluster_count:05d}"


from __future__ import annotations

from dataclasses import dataclass
from math import gcd


@dataclass(frozen=True)
class IntervalLayout:
    predicate_type: str
    build_size: int
    selected_rows: int
    intervals: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class ScatteredLayout:
    predicate_type: str
    build_size: int
    selected_rows: int
    stride: int
    offset: int


PredicateLayout = IntervalLayout | ScatteredLayout


def split_evenly(total: int, parts: int) -> tuple[int, ...]:
    if parts <= 0:
        return tuple()
    base, remainder = divmod(total, parts)
    return tuple(base + (1 if index < remainder else 0) for index in range(parts))


def build_range_layout(build_size: int, selected_rows: int) -> IntervalLayout:
    start = ((build_size - selected_rows) // 2) + 1
    end = start + selected_rows - 1
    return IntervalLayout(
        predicate_type="range",
        build_size=build_size,
        selected_rows=selected_rows,
        intervals=((start, end),),
    )


def build_shoulder_layout(build_size: int, selected_rows: int) -> IntervalLayout:
    left_size = (selected_rows + 1) // 2
    right_size = selected_rows // 2
    intervals: list[tuple[int, int]] = [(1, left_size)]
    if right_size > 0:
        intervals.append((build_size - right_size + 1, build_size))
    return IntervalLayout(
        predicate_type="shoulder",
        build_size=build_size,
        selected_rows=selected_rows,
        intervals=tuple(intervals),
    )


def build_seasonal_layout(build_size: int, selected_rows: int) -> IntervalLayout:
    range_sizes = split_evenly(selected_rows, 10)
    gap_sizes = split_evenly(build_size - selected_rows, 9)
    intervals: list[tuple[int, int]] = []
    start = 1
    for index, range_size in enumerate(range_sizes):
        end = start + range_size - 1
        intervals.append((start, end))
        if index < len(gap_sizes):
            start = end + 1 + gap_sizes[index]

    if intervals[0][0] != 1:
        raise AssertionError("seasonal layout must start at key 1")
    if intervals[-1][1] != build_size:
        raise AssertionError("seasonal layout must end at the last key")

    return IntervalLayout(
        predicate_type="seasonal",
        build_size=build_size,
        selected_rows=selected_rows,
        intervals=tuple(intervals),
    )


def choose_coprime_stride(modulus: int, seed: int) -> int:
    candidate = abs(seed) % modulus
    if candidate == 0:
        candidate = 1
    while gcd(candidate, modulus) != 1:
        candidate += 1
        if candidate >= modulus:
            candidate = 1
    return candidate


def build_scattered_layout(build_size: int, selected_rows: int, seed: int) -> ScatteredLayout:
    stride_seed = seed * 2 + 1
    offset_seed = seed * 7 + 3
    return ScatteredLayout(
        predicate_type="scattered",
        build_size=build_size,
        selected_rows=selected_rows,
        stride=choose_coprime_stride(build_size, stride_seed),
        offset=(offset_seed % build_size),
    )


def build_predicate_layout(
    predicate_type: str,
    build_size: int,
    selected_rows: int,
    seed: int,
) -> PredicateLayout:
    if predicate_type == "range":
        return build_range_layout(build_size, selected_rows)
    if predicate_type == "shoulder":
        return build_shoulder_layout(build_size, selected_rows)
    if predicate_type == "seasonal":
        return build_seasonal_layout(build_size, selected_rows)
    if predicate_type == "scattered":
        return build_scattered_layout(build_size, selected_rows, seed)
    raise ValueError(f"unsupported predicate type: {predicate_type}")


def predicate_layout_metadata(layout: PredicateLayout) -> dict:
    if isinstance(layout, IntervalLayout):
        sizes = [end - start + 1 for start, end in layout.intervals]
        return {
            "predicate_type": layout.predicate_type,
            "build_size": layout.build_size,
            "selected_rows": layout.selected_rows,
            "interval_count": len(layout.intervals),
            "intervals": [list(interval) for interval in layout.intervals],
            "range_size_min": min(sizes),
            "range_size_max": max(sizes),
        }
    return {
        "predicate_type": layout.predicate_type,
        "build_size": layout.build_size,
        "selected_rows": layout.selected_rows,
        "stride": layout.stride,
        "offset": layout.offset,
    }


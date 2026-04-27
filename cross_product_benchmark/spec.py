from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_BUILD_SIZES = (100_000, 100_000_000)
DEFAULT_SELECTIVITIES = (0.005, 0.05)
DEFAULT_PREDICATE_TYPES = ("range", "shoulder", "seasonal", "scattered")
DEFAULT_PROBE_CLUSTERINGS = ("ordered", "uniform")


def build_size_label(build_size: int) -> str:
    if build_size % 1_000_000 == 0:
        return f"{build_size // 1_000_000}m"
    if build_size % 1_000 == 0:
        return f"{build_size // 1_000}k"
    return str(build_size)


def selectivity_label(selectivity: float) -> str:
    scaled = int(round(selectivity * 1_000))
    return f"s{scaled:04d}"


def mask_column_name(selectivity: float, predicate_type: str) -> str:
    return f"mask_{selectivity_label(selectivity)}_{predicate_type}"


def dimension_table_name(build_size: int) -> str:
    return f"dim_{build_size_label(build_size)}"


def fact_fk_column_name(build_size: int, probe_clustering: str) -> str:
    return f"fk_{build_size_label(build_size)}_{probe_clustering}"


@dataclass(frozen=True)
class CrossProductBenchmarkSpec:
    db_path: Path
    queries_dir: Path
    build_sizes: tuple[int, ...] = DEFAULT_BUILD_SIZES
    selectivities: tuple[float, ...] = DEFAULT_SELECTIVITIES
    predicate_types: tuple[str, ...] = DEFAULT_PREDICATE_TYPES
    probe_clusterings: tuple[str, ...] = DEFAULT_PROBE_CLUSTERINGS
    fact_rows: int = 100_000_000
    dummy_string_cols: int = 2
    dummy_string_len: int = 24
    seed: int = 0
    force: bool = False

    def selected_rows(self, build_size: int, selectivity: float) -> int:
        return int(round(build_size * selectivity))

    def validate(self) -> None:
        if not self.build_sizes:
            raise ValueError("at least one build size is required")
        if len(set(self.build_sizes)) != len(self.build_sizes):
            raise ValueError("build sizes must be distinct")
        for build_size in self.build_sizes:
            if build_size <= 0:
                raise ValueError("build sizes must be positive")

        if not self.selectivities:
            raise ValueError("at least one selectivity is required")
        if len(set(self.selectivities)) != len(self.selectivities):
            raise ValueError("selectivities must be distinct")
        for selectivity in self.selectivities:
            if not 0 < selectivity <= 1:
                raise ValueError("selectivities must be in the interval (0, 1]")

        if not self.predicate_types:
            raise ValueError("at least one predicate type is required")
        if len(set(self.predicate_types)) != len(self.predicate_types):
            raise ValueError("predicate types must be distinct")
        allowed_predicate_types = {"range", "shoulder", "seasonal", "scattered"}
        for predicate_type in self.predicate_types:
            if predicate_type not in allowed_predicate_types:
                raise ValueError(f"unsupported predicate type: {predicate_type}")

        if not self.probe_clusterings:
            raise ValueError("at least one probe clustering is required")
        if len(set(self.probe_clusterings)) != len(self.probe_clusterings):
            raise ValueError("probe clusterings must be distinct")
        allowed_probe_clusterings = {"ordered", "uniform"}
        for probe_clustering in self.probe_clusterings:
            if probe_clustering not in allowed_probe_clusterings:
                raise ValueError(f"unsupported probe clustering: {probe_clustering}")

        if self.fact_rows <= 0:
            raise ValueError("--fact-rows must be positive")
        if self.dummy_string_cols < 0:
            raise ValueError("--dummy-string-cols cannot be negative")
        if self.dummy_string_len <= 0:
            raise ValueError("--dummy-string-len must be positive")

        for build_size in self.build_sizes:
            for selectivity in self.selectivities:
                selected_rows = self.selected_rows(build_size, selectivity)
                if selected_rows <= 0:
                    raise ValueError(
                        f"selectivity {selectivity} yields zero selected rows "
                        f"for build size {build_size}"
                    )
                if "seasonal" in self.predicate_types and selected_rows < 10:
                    raise ValueError(
                        f"seasonal predicate needs at least 10 selected rows, "
                        f"got {selected_rows} for build size {build_size}"
                    )

from __future__ import annotations

from dataclasses import dataclass


def rgb(hex_value: str) -> tuple[float, float, float]:
    value = hex_value.lstrip("#")
    return tuple(int(value[index:index + 2], 16) / 255.0 for index in (0, 2, 4))


@dataclass(frozen=True)
class FigureSize:
    width: float
    height: float


PAGE_SIZES = {
    "one-column": FigureSize(width=252.0, height=210.0),
    "two-column": FigureSize(width=516.0, height=255.0),
}

COLORS = {
    "background": rgb("#fffdf8"),
    "panel": rgb("#fffdf8"),
    "text": rgb("#111827"),
    "muted_text": rgb("#4b5563"),
    "grid": rgb("#d1d5db"),
    "reference": rgb("#9ca3af"),
    "no prefilter": rgb("#9ca3af"),
    "sip": rgb("#2563eb"),
    "sip + min/max": rgb("#d97706"),
    "prf with fallback": rgb("#0f766e"),
}

TITLE_SIZE = 13.0
LABEL_SIZE = 9.5
TICK_SIZE = 8.0
LEGEND_SIZE = 8.5

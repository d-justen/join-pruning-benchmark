from __future__ import annotations

import math
from html import escape


PALETTE = [
    "#0f766e",
    "#b45309",
    "#1d4ed8",
    "#b91c1c",
    "#6d28d9",
    "#047857",
]


def _fmt(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def svg_document(width: int, height: int, elements: list[str]) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="#fcfcf7"/>\n'
        + "\n".join(elements)
        + "\n</svg>\n"
    )


def text(x: float, y: float, value: str, *, size: int = 12, anchor: str = "start", weight: str = "400") -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Helvetica, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#111827">{escape(value)}</text>'
    )


def line(x1: float, y1: float, x2: float, y2: float, *, stroke: str = "#6b7280", width: float = 1, dash: str | None = None) -> str:
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'


def rect(x: float, y: float, width: float, height: float, *, fill: str, stroke: str = "none") -> str:
    return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}" stroke="{stroke}"/>'


def polyline(points: list[tuple[float, float]], *, stroke: str, width: float = 2, dash: str | None = None) -> str:
    dash_attr = "" if dash is None else f' stroke-dasharray="{dash}"'
    serialized = " ".join(f"{x},{y}" for x, y in points)
    return (
        f'<polyline fill="none" stroke="{stroke}" stroke-width="{width}" '
        f'points="{serialized}"{dash_attr}/>'
    )


def circle(cx: float, cy: float, r: float, *, fill: str) -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>'


def _scale_linear(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if hi <= lo:
        return (out_lo + out_hi) / 2
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def _scale_log10(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    if lo <= 0 or hi <= 0:
        return _scale_linear(value, lo, hi, out_lo, out_hi)
    l_lo = math.log10(lo)
    l_hi = math.log10(hi)
    l_value = math.log10(value)
    return _scale_linear(l_value, l_lo, l_hi, out_lo, out_hi)


def line_chart(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: list[dict],
    x_values: list[float],
    log_x: bool = False,
) -> list[str]:
    elements: list[str] = [
        rect(x, y, width, height, fill="#ffffff", stroke="#d1d5db"),
        text(x + 12, y + 20, title, size=13, weight="600"),
    ]
    plot_x = x + 56
    plot_y = y + 32
    plot_w = width - 74
    plot_h = height - 70
    elements.append(line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#9ca3af"))
    elements.append(line(plot_x, plot_y, plot_x, plot_y + plot_h, stroke="#9ca3af"))
    y_values = [point[1] for item in series for point in item["points"] if point[1] is not None]
    y_min = 0.0
    y_max = max(y_values) if y_values else 1.0
    if y_max <= y_min:
        y_max = y_min + 1.0
    for tick_index in range(5):
        tick_value = y_min + (y_max - y_min) * tick_index / 4
        tick_y = _scale_linear(tick_value, y_min, y_max, plot_y + plot_h, plot_y)
        elements.append(line(plot_x, tick_y, plot_x + plot_w, tick_y, stroke="#e5e7eb"))
        elements.append(text(plot_x - 8, tick_y + 4, _fmt(tick_value), size=10, anchor="end"))
    if x_values:
        x_sorted = sorted(x_values)
        for value in x_sorted:
            tick_x = (
                _scale_log10(value, x_sorted[0], x_sorted[-1], plot_x, plot_x + plot_w)
                if log_x
                else _scale_linear(value, x_sorted[0], x_sorted[-1], plot_x, plot_x + plot_w)
            )
            elements.append(line(tick_x, plot_y + plot_h, tick_x, plot_y + plot_h + 4, stroke="#9ca3af"))
            elements.append(text(tick_x, plot_y + plot_h + 18, str(int(value) if value == int(value) else value), size=10, anchor="middle"))
    elements.append(text(plot_x + plot_w / 2, y + height - 10, x_label, size=11, anchor="middle"))
    elements.append(text(x + 12, plot_y + plot_h / 2, y_label, size=11))
    for index, item in enumerate(series):
        points = []
        x_sorted = sorted(x_values)
        for x_value, y_value in item["points"]:
            px = (
                _scale_log10(x_value, x_sorted[0], x_sorted[-1], plot_x, plot_x + plot_w)
                if log_x
                else _scale_linear(x_value, x_sorted[0], x_sorted[-1], plot_x, plot_x + plot_w)
            )
            py = _scale_linear(y_value, y_min, y_max, plot_y + plot_h, plot_y)
            points.append((px, py))
        color = item.get("color", PALETTE[index % len(PALETTE)])
        dash = item.get("dash")
        elements.append(polyline(points, stroke=color, dash=dash))
        for px, py in points:
            elements.append(circle(px, py, 3, fill=color))
    legend_x = x + width - 180
    legend_y = y + 18
    for index, item in enumerate(series):
        ly = legend_y + index * 16
        color = item.get("color", PALETTE[index % len(PALETTE)])
        elements.append(line(legend_x, ly, legend_x + 18, ly, stroke=color, width=2, dash=item.get("dash")))
        elements.append(text(legend_x + 24, ly + 4, item["label"], size=10))
    return elements


def grouped_bar_chart(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    title: str,
    categories: list[str],
    series: list[dict],
    x_label: str,
) -> list[str]:
    elements: list[str] = [
        rect(x, y, width, height, fill="#ffffff", stroke="#d1d5db"),
        text(x + 12, y + 20, title, size=13, weight="600"),
    ]
    longest_category = max((len(category) for category in categories), default=0)
    label_width = min(260, max(150, longest_category * 7))
    plot_x = x + label_width
    plot_y = y + 32
    plot_w = width - label_width - 24
    plot_h = height - 54
    elements.append(line(plot_x, plot_y + plot_h, plot_x + plot_w, plot_y + plot_h, stroke="#9ca3af"))
    max_value = max((value for item in series for value in item["values"] if value is not None), default=1.0)
    if max_value <= 0:
        max_value = 1.0
    group_height = plot_h / max(1, len(categories))
    bar_height = max(8.0, min(14.0, (group_height - 8) / max(1, len(series))))
    for cat_index, category in enumerate(categories):
        base_y = plot_y + cat_index * group_height + 4
        elements.append(text(x + 8, base_y + group_height / 2 + 4, category, size=10))
        for series_index, item in enumerate(series):
            value = item["values"][cat_index]
            if value is None:
                continue
            bar_y = base_y + series_index * bar_height
            bar_w = _scale_linear(value, 0, max_value, 0, plot_w)
            color = item.get("color", PALETTE[series_index % len(PALETTE)])
            elements.append(rect(plot_x, bar_y, bar_w, bar_height - 1, fill=color))
            label = _fmt(value)
            if bar_w >= 46:
                elements.append(
                    f'<text x="{plot_x + bar_w - 6}" y="{bar_y + bar_height - 3}" '
                    f'font-family="Helvetica, Arial, sans-serif" font-size="9" '
                    f'font-weight="600" text-anchor="end" fill="#ffffff">{escape(label)}</text>'
                )
            else:
                elements.append(text(plot_x + bar_w + 4, bar_y + bar_height - 2, label, size=9))
    for tick_index in range(5):
        tick_value = max_value * tick_index / 4
        tick_x = _scale_linear(tick_value, 0, max_value, plot_x, plot_x + plot_w)
        elements.append(line(tick_x, plot_y, tick_x, plot_y + plot_h, stroke="#f3f4f6"))
        elements.append(text(tick_x, y + height - 10, _fmt(tick_value), size=10, anchor="middle"))
    elements.append(text(plot_x + plot_w / 2, y + height - 26, x_label, size=11, anchor="middle"))
    legend_x = x + width - 180
    legend_y = y + 18
    for index, item in enumerate(series):
        ly = legend_y + index * 16
        color = item.get("color", PALETTE[index % len(PALETTE)])
        elements.append(rect(legend_x, ly - 8, 14, 10, fill=color))
        elements.append(text(legend_x + 22, ly, item["label"], size=10))
    return elements

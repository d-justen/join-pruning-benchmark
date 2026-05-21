from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _escape_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


@dataclass(frozen=True)
class PdfPage:
    width: float
    height: float


class PdfCanvas:
    def __init__(self, width: float, height: float) -> None:
        self.page = PdfPage(width=width, height=height)
        self._ops: list[str] = []

    def set_fill_rgb(self, r: float, g: float, b: float) -> None:
        self._ops.append(f"{r:.4f} {g:.4f} {b:.4f} rg")

    def set_stroke_rgb(self, r: float, g: float, b: float) -> None:
        self._ops.append(f"{r:.4f} {g:.4f} {b:.4f} RG")

    def set_line_width(self, width: float) -> None:
        self._ops.append(f"{width:.3f} w")

    def set_dash(self, on: float, off: float) -> None:
        self._ops.append(f"[{on:.3f} {off:.3f}] 0 d")

    def clear_dash(self) -> None:
        self._ops.append("[] 0 d")

    def line(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self._ops.append(f"{x1:.3f} {y1:.3f} m {x2:.3f} {y2:.3f} l S")

    def rect(self, x: float, y: float, width: float, height: float, *, fill: bool = True, stroke: bool = False) -> None:
        op = "B" if fill and stroke else "f" if fill else "S"
        self._ops.append(f"{x:.3f} {y:.3f} {width:.3f} {height:.3f} re {op}")

    def text(self, x: float, y: float, value: str, *, size: float = 10, font: str = "F1") -> None:
        self._ops.append(
            f"BT /{font} {size:.3f} Tf 1 0 0 1 {x:.3f} {y:.3f} Tm ({_escape_text(value)}) Tj ET"
        )

    def save(self, path: Path) -> None:
        content = "\n".join(self._ops).encode("latin-1", errors="replace")
        objects: list[bytes] = []

        def add_object(payload: bytes) -> int:
            objects.append(payload)
            return len(objects)

        font_regular = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        font_bold = add_object(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        content_id = add_object(
            f"<< /Length {len(content)} >>\nstream\n".encode("latin-1") + content + b"\nendstream"
        )
        page_id = add_object(
            (
                f"<< /Type /Page /Parent 5 0 R /MediaBox [0 0 {self.page.width:.3f} {self.page.height:.3f}] "
                f"/Resources << /Font << /F1 {font_regular} 0 R /F2 {font_bold} 0 R >> >> "
                f"/Contents {content_id} 0 R >>"
            ).encode("latin-1")
        )
        pages_id = add_object(f"<< /Type /Pages /Kids [{page_id} 0 R] /Count 1 >>".encode("latin-1"))
        catalog_id = add_object(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode("latin-1"))

        chunks = [b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"]
        offsets = [0]
        for index, payload in enumerate(objects, start=1):
            offsets.append(sum(len(chunk) for chunk in chunks))
            chunks.append(f"{index} 0 obj\n".encode("latin-1"))
            chunks.append(payload)
            chunks.append(b"\nendobj\n")
        xref_offset = sum(len(chunk) for chunk in chunks)
        chunks.append(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
        chunks.append(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            chunks.append(f"{offset:010d} 00000 n \n".encode("latin-1"))
        chunks.append(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("latin-1")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(chunks))

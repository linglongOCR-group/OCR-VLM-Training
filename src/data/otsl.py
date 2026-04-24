"""Bidirectional conversion between OTSL (Object-oriented Table Structure Label) and HTML tables.

OTSL tokens:
  <fcel> - first cell (cell with content)
  <ecel> - empty cell
  <lcel> - left-spanning cell (horizontal merge continuation)
  <ucel> - up-spanning cell (vertical merge continuation)
  <xcel> - cross-spanning cell (2D merge continuation)
  <nl>   - new line (row separator)
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

# ---------------------------------------------------------------------------
# OTSL token constants
# ---------------------------------------------------------------------------

NL = "<nl>"
FCEL = "<fcel>"
ECEL = "<ecel>"
LCEL = "<lcel>"
UCEL = "<ucel>"
XCEL = "<xcel>"

_TOKEN_RE = re.compile(r"(<fcel>|<ecel>|<lcel>|<ucel>|<xcel>|<nl>)")

# ---------------------------------------------------------------------------
# Data structures used internally
# ---------------------------------------------------------------------------


@dataclass
class _Cell:
    """Represents a single logical table cell."""

    text: str = ""
    colspan: int = 1
    rowspan: int = 1


@dataclass
class _Table:
    """Intermediate table representation as a list of rows of _Cell."""

    rows: list[list[_Cell]] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def n_cols(self) -> int:
        if not self.rows:
            return 0
        return max(len(r) for r in self.rows)


# ---------------------------------------------------------------------------
# otsl_to_html
# ---------------------------------------------------------------------------


def otsl_to_html(otsl: str) -> str:
    """Convert an OTSL string to an HTML <table> string.

    If the input already looks like an HTML table (starts with ``<table`` and
    ends with ``</table>``), it is returned unchanged.
    """
    stripped = otsl.strip()
    if stripped.startswith("<table") and stripped.endswith("</table>"):
        return stripped

    # Split into rows by <nl>
    raw_rows = stripped.split(NL)
    # Filter out empty trailing rows (trailing <nl> produces an empty string)
    raw_rows = [r for r in raw_rows if r]

    if not raw_rows:
        return "<table></table>"

    # Parse each row into a list of (token, text) pairs
    parsed_rows: list[list[tuple[str, str]]] = []
    for raw_row in raw_rows:
        cells = _parse_otsl_row(raw_row)
        parsed_rows.append(cells)

    # Build an intermediate table grid to resolve spans
    table = _build_table(parsed_rows)

    # Render to HTML
    return _render_html(table)


def _parse_otsl_row(raw_row: str) -> list[tuple[str, str]]:
    """Parse a single OTSL row string into a list of (token, text) pairs.

    Each cell token (<fcel>, <ecel>, <lcel>, <ucel>, <xcel>) may be followed
    by text content (only meaningful for <fcel>).
    """
    cells: list[tuple[str, str]] = []
    # Split by token markers
    parts = _TOKEN_RE.split(raw_row)
    current_token = None
    current_text = ""

    for part in parts:
        if part in (FCEL, ECEL, LCEL, UCEL, XCEL):
            # Flush previous cell
            if current_token is not None:
                cells.append((current_token, current_text))
            current_token = part
            current_text = ""
        elif part == NL:
            # Shouldn't happen since we already split by <nl>, but handle gracefully
            if current_token is not None:
                cells.append((current_token, current_text))
                current_token = None
                current_text = ""
        else:
            # Text content
            current_text += part

    # Flush last cell
    if current_token is not None:
        cells.append((current_token, current_text))

    return cells


def _build_table(parsed_rows: list[list[tuple[str, str]]]) -> _Table:
    """Build a _Table from parsed OTSL rows, resolving spans.

    Each OTSL token occupies exactly one grid column.  We lay all tokens out
    in a 2D grid and then resolve spans:

    - <fcel>  -> origin cell; count trailing <lcel> for colspan, count
                 <ucel>/<xcel> directly below for rowspan.
    - <ecel>  -> standalone empty cell.
    - <lcel>  -> horizontal span continuation (consumed by preceding <fcel>).
    - <ucel>  -> vertical span continuation (adds to rowspan of origin above).
    - <xcel>  -> 2D span continuation (consumed, adds to rowspan).
    """
    if not parsed_rows:
        return _Table()

    n_rows = len(parsed_rows)
    n_cols = max(len(row) for row in parsed_rows)

    # Pad rows so every row has exactly n_cols entries
    grid: list[list[tuple[str, str]]] = []
    for row in parsed_rows:
        padded = list(row) + [(ECEL, "")] * (n_cols - len(row))
        grid.append(padded)

    # Track which grid positions are consumed by a span (no HTML <td> output).
    consumed: set[tuple[int, int]] = set()
    # Origin cells: (r, c) -> _Cell
    cells: dict[tuple[int, int], _Cell] = {}

    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) in consumed:
                continue
            token, text = grid[r][c]

            if token == FCEL:
                # --- colspan from trailing <lcel> ---
                colspan = 1
                cc = c + 1
                while cc < n_cols and grid[r][cc][0] == LCEL:
                    colspan += 1
                    consumed.add((r, cc))
                    cc += 1

                # --- rowspan from <ucel>/<xcel> below ---
                rowspan = 1
                rr = r + 1
                while rr < n_rows and grid[rr][c][0] in (UCEL, XCEL):
                    rowspan += 1
                    consumed.add((rr, c))
                    # For <xcel>, also consume the horizontal span positions
                    if grid[rr][c][0] == XCEL:
                        for dc in range(1, colspan):
                            if c + dc < n_cols:
                                consumed.add((rr, c + dc))
                    rr += 1

                cells[(r, c)] = _Cell(text=text, colspan=colspan, rowspan=rowspan)

            elif token == ECEL:
                cells[(r, c)] = _Cell(text="", colspan=1, rowspan=1)

            # <ucel>, <lcel>, <xcel> that weren't consumed by a preceding <fcel>
            # are treated as empty cells (edge case, shouldn't occur in valid OTSL).

    # Build the table: emit cells that are not consumed
    table = _Table()
    for r in range(n_rows):
        row_cells: list[_Cell] = []
        for c in range(n_cols):
            if (r, c) in consumed:
                continue
            if (r, c) in cells:
                row_cells.append(cells[(r, c)])
        table.rows.append(row_cells)

    return table


def _render_html(table: _Table) -> str:
    """Render a _Table to an HTML string."""
    parts = ["<table>"]
    for row in table.rows:
        parts.append("<tr>")
        for cell in row:
            attrs = ""
            if cell.colspan > 1:
                attrs += f' colspan="{cell.colspan}"'
            if cell.rowspan > 1:
                attrs += f' rowspan="{cell.rowspan}"'
            escaped_text = html.escape(cell.text)
            parts.append(f"<td{attrs}>{escaped_text}</td>")
        parts.append("</tr>")
    parts.append("</table>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# html_to_otsl
# ---------------------------------------------------------------------------


class _TableParser(HTMLParser):
    """Parse an HTML table and produce an OTSL string."""

    def __init__(self) -> None:
        super().__init__()
        self._in_table = False
        self._in_cell = False
        self._cell_text = ""
        self._cell_colspan = 1
        self._cell_rowspan = 1

        # Accumulate parsed cells per row: list of (text, colspan, rowspan)
        self._current_row: list[tuple[str, int, int]] = []
        # All rows collected
        self._rows: list[list[tuple[str, int, int]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == "table":
            self._in_table = True
        elif tag_lower in ("td", "th") and self._in_table:
            self._in_cell = True
            self._cell_text = ""
            self._cell_colspan = 1
            self._cell_rowspan = 1
            for attr_name, attr_value in attrs:
                if attr_name.lower() == "colspan" and attr_value:
                    try:
                        self._cell_colspan = int(attr_value)
                    except ValueError:
                        pass
                elif attr_name.lower() == "rowspan" and attr_value:
                    try:
                        self._cell_rowspan = int(attr_value)
                    except ValueError:
                        pass
        elif tag_lower == "tr" and self._in_table:
            self._current_row = []

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "table":
            self._in_table = False
        elif tag_lower in ("td", "th") and self._in_cell:
            self._in_cell = False
            self._current_row.append(
                (self._cell_text.strip(), self._cell_colspan, self._cell_rowspan)
            )
        elif tag_lower == "tr" and self._in_table:
            if self._current_row:
                self._rows.append(self._current_row)
            self._current_row = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_text += data

    def get_otsl(self) -> str:
        """Build the OTSL string from the parsed rows."""
        if not self._rows:
            return ""

        # Determine grid dimensions
        n_cols = 0
        for row in self._rows:
            row_width = sum(cspan for _, cspan, _ in row)
            n_cols = max(n_cols, row_width)
        n_rows = len(self._rows)

        # Shadow grid: each position stores:
        #   None = unoccupied
        #   (origin_row, origin_col) = reference to origin cell
        # We also need to know if the position is an origin cell itself.
        grid: list[list[tuple[int, int] | None]] = [
            [None] * n_cols for _ in range(n_rows)
        ]

        # Place cells into the grid
        for r, row in enumerate(self._rows):
            c = 0
            for text, colspan, rowspan in row:
                # Find next free column in this row
                while c < n_cols and grid[r][c] is not None:
                    c += 1
                if c >= n_cols:
                    break
                # Place origin marker
                grid[r][c] = (r, c)
                # Mark spanned positions
                for dr in range(rowspan):
                    for dc in range(colspan):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if rr < n_rows and cc < n_cols:
                            grid[rr][cc] = (r, c)
                c += colspan

        # Build OTSL from the grid
        # We need to track for each cell: its text, colspan, rowspan
        # Build a lookup for origin cells
        cell_info: dict[tuple[int, int], tuple[str, int, int]] = {}
        for r, row in enumerate(self._rows):
            c = 0
            for text, colspan, rowspan in row:
                while c < n_cols and grid[r][c] is not None and grid[r][c] != (r, c):
                    c += 1
                if c >= n_cols:
                    # This can happen if the grid is already filled by spans
                    # Try to find the next free spot
                    found = False
                    for cc in range(c, n_cols):
                        if grid[r][cc] is None or grid[r][cc] == (r, cc):
                            c = cc
                            found = True
                            break
                    if not found:
                        continue
                cell_info[(r, c)] = (text, colspan, rowspan)
                c += colspan

        parts: list[str] = []
        for r in range(n_rows):
            for c in range(n_cols):
                ref = grid[r][c]
                if ref is None:
                    # Unoccupied cell -> empty
                    parts.append(ECEL)
                elif ref == (r, c):
                    # Origin cell
                    text, colspan, rowspan = cell_info.get(
                        (r, c), ("", 1, 1)
                    )
                    if text:
                        parts.append(f"{FCEL}{text}")
                    else:
                        parts.append(ECEL)
                else:
                    # Span continuation
                    origin_r, origin_c = ref
                    info = cell_info.get((origin_r, origin_c), ("", 1, 1))
                    _, origin_colspan, origin_rowspan = info
                    covered_h = (c - origin_c) < origin_colspan and c > origin_c
                    covered_v = (r - origin_r) < origin_rowspan and r > origin_r

                    if covered_h and covered_v:
                        parts.append(XCEL)
                    elif covered_v:
                        parts.append(UCEL)
                    elif covered_h:
                        parts.append(LCEL)
                    else:
                        parts.append(ECEL)
            parts.append(NL)

        return "".join(parts)


def html_to_otsl(html_str: str) -> str:
    """Convert an HTML <table> string to OTSL format."""
    parser = _TableParser()
    parser.feed(html_str)
    return parser.get_otsl()

"""Kiểm tra định dạng bảng Markdown trong reports/: số cột của dòng tiêu đề phải bằng dòng phân cách.

GitHub chỉ hiển thị thành bảng khi hai số này khớp; nếu lệch, bảng hiện thành văn bản thô.
"""

import re
from pathlib import Path
from typing import List, Tuple

import pytest

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
DELIMITER_ROW = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")


def split_cells(line: str) -> List[str]:
    """Tách các ô của một dòng bảng; dấu | trong code span (`...`) hoặc đã escape (\\|) không phải ranh giới ô."""
    text = line.strip()
    text = text[1:] if text.startswith("|") else text
    text = text[:-1] if text.endswith("|") and not text.endswith("\\|") else text
    cells, current, in_code, escaped = [], "", False, False
    for char in text:
        if escaped:
            current, escaped = current + char, False
            continue
        if char == "\\":
            current, escaped = current + char, True
            continue
        if char == "`":
            in_code = not in_code
        if char == "|" and not in_code:
            cells.append(current)
            current = ""
        else:
            current += char
    cells.append(current)
    return cells


def mismatched_tables(markdown: str) -> List[Tuple[int, int, int]]:
    """(số dòng của tiêu đề, số ô tiêu đề, số ô dòng phân cách) cho các bảng lệch cột; bỏ qua khối code."""
    lines, in_fence, problems = markdown.splitlines(), False, []
    for i in range(len(lines) - 1):
        if lines[i].lstrip().startswith("```"):
            in_fence = not in_fence
        if in_fence or not lines[i].lstrip().startswith("|") or not DELIMITER_ROW.match(lines[i + 1]):
            continue
        header_cells, delimiter_cells = len(split_cells(lines[i])), len(split_cells(lines[i + 1]))
        if header_cells != delimiter_cells:
            problems.append((i + 1, header_cells, delimiter_cells))
    return problems


def test_detects_header_and_delimiter_mismatch():
    bad = "| a | b |\n|---|---|---|\n| 1 | 2 |\n"
    good = "| a | b |\n|:---|---:|\n| 1 | 2 |\n"
    assert mismatched_tables(bad) == [(1, 2, 3)]
    assert mismatched_tables(good) == []


def test_ignores_pipes_in_code_spans_and_fences():
    in_code_span = "| lệnh | ý nghĩa |\n|---|---|\n| `a | b` | c |\n"
    in_fence = "```\n| a | b |\n|---|---|---|\n```\n"
    assert mismatched_tables(in_code_span) == []
    assert mismatched_tables(in_fence) == []


@pytest.mark.parametrize("path", sorted(REPORTS_DIR.rglob("*.md")), ids=lambda p: str(p.relative_to(REPORTS_DIR)))
def test_report_tables_have_matching_columns(path):
    assert mismatched_tables(path.read_text(encoding="utf-8")) == [], f"{path.name}: bảng lệch số cột"

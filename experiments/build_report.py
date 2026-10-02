"""Build docs/report.html from docs/report_template.html and the tables of docs/results.md.

    python -m experiments.build_report
Every {{T:key}} placeholder of the template is replaced by the <!-- TABLE:key --> block of
results.md, converted from Markdown to HTML.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


def md_table_to_html(md):
    rows = [r.strip() for r in md.strip().splitlines() if r.strip().startswith("|")]
    if len(rows) < 2:
        return f"<p class=\"pending\">{html.escape(md.strip())}</p>"

    def cells(row):
        parts = re.split(r"(?<!\\)\|", row.strip()[1:-1])
        return [p.strip().replace("\\|", "|") for p in parts]

    def fmt(c):
        c = html.escape(c)
        return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", c)

    head = "".join(f"<th>{fmt(c)}</th>" for c in cells(rows[0]))
    body = "".join("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in cells(r)) + "</tr>" for r in rows[2:])
    return f'<div class="table-wrap"><table class="data"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def main():
    tables = dict(re.findall(r"<!-- TABLE:(\w+) -->\n(.*?)<!-- /TABLE -->", (DOCS / "results.md").read_text(), re.S))
    out = re.sub(r"\{\{T:(\w+)\}\}", lambda m: md_table_to_html(tables[m.group(1)]),
                 (DOCS / "report_template.html").read_text())
    missing = re.findall(r"\{\{T:(\w+)\}\}", out)
    assert not missing, missing
    (DOCS / "report.html").write_text(out)
    print(f"docs/report.html: {len(out):,} bytes")


if __name__ == "__main__":
    main()

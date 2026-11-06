"""Minimal CSV read/write mirroring the Node pipeline's lib/csv.js.

parse_csv -> (header, rows-as-dicts). Extra fields beyond the header are
dropped, missing fields become '' (matches the Node Object.fromEntries
behavior). Quoted fields, embedded commas/quotes/newlines and CRLF handled by
Python's stdlib csv module (superset of the hand parser).
"""
import csv
import io


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def parse_csv(text: str):
    text = text.lstrip("\ufeff")
    reader = csv.reader(io.StringIO(text, newline=""))
    raw = list(reader)
    while raw and raw[-1] == [""]:
        raw.pop()
    if not raw:
        return ([], [])
    header = raw[0]
    rows = []
    for r in raw[1:]:
        rows.append({h: (r[j] if j < len(r) else "") for j, h in enumerate(header)})
    return (header, rows)


def _esc(v) -> str:
    s = "" if v is None else str(v)
    if any(c in s for c in '",\n\r'):
        return '"' + s.replace('"', '""') + '"'
    return s


def to_csv(header, rows) -> str:
    lines = [",".join(_esc(h) for h in header)]
    for r in rows:
        lines.append(",".join(_esc(r.get(h, "")) for h in header))
    return "\n".join(lines) + "\n"

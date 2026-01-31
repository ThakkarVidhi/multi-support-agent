"""Load CSV into SQLite: one table support_tickets. Run once."""
import csv
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "raw"
DB_PATH = PROJECT_ROOT / "customer_support.db"

# Try these CSV filenames in order
CSV_CANDIDATES = [
    "customer_support_tickets.csv",
    "synthetic_support_tickets.csv",
]


def _normalize_col(name: str) -> str:
    """Convert column name to valid SQL identifier (snake_case)."""
    s = re.sub(r"[^\w\s]", "", name)
    s = re.sub(r"\s+", "_", s.strip()).lower()
    return s or "col"


def _infer_type(values: list[str]) -> str:
    """Infer SQLite type from non-empty values."""
    for v in values:
        if v is None or v == "":
            continue
        try:
            int(v)
            return "INTEGER"
        except ValueError:
            pass
    return "TEXT"


def main() -> None:
    csv_path = None
    for name in CSV_CANDIDATES:
        p = RAW_DIR / name
        if p.exists():
            csv_path = p
            break
    if not csv_path:
        print(f"No CSV found in {RAW_DIR}. Tried: {CSV_CANDIDATES}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        raw_headers = reader.fieldnames or []
        headers = [_normalize_col(h) for h in raw_headers]
        for row in reader:
            rows.append({headers[i]: row.get(h, "") for i, h in enumerate(raw_headers)})

    if not rows:
        print("No data rows in CSV.", file=sys.stderr)
        sys.exit(1)

    # Infer types per column
    col_types = {}
    for col in headers:
        vals = [r.get(col, "") for r in rows]
        col_types[col] = _infer_type(vals)

    # Create table
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DROP TABLE IF EXISTS support_tickets")
    cols_sql = ", ".join(f'"{c}" {col_types[c]}' for c in headers)
    conn.execute(f'CREATE TABLE support_tickets ({cols_sql})')
    placeholders = ", ".join("?" for _ in headers)
    for row in rows:
        conn.execute(
            f'INSERT INTO support_tickets ({", ".join(chr(34) + c + chr(34) for c in headers)}) VALUES ({placeholders})',
            [row.get(c, "") for c in headers],
        )
    conn.commit()

    # Verify at least one "Ema" for demo (customer_name or similar column)
    cur = conn.execute("PRAGMA table_info(support_tickets)")
    col_names = [row[1] for row in cur.fetchall()]
    name_col = None
    for c in ("customer_name", "customer name", "name"):
        if c in col_names or c.replace(" ", "_") in col_names:
            name_col = c if c in col_names else c.replace(" ", "_")
            break
    conn.close()

    if name_col:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute(f'SELECT COUNT(*) FROM support_tickets WHERE LOWER("{name_col}") LIKE "%ema%"')
        count = cur.fetchone()[0]
        if count == 0:
            # Insert one Ema row: same columns, fill name_col with Ema
            placeholders = ", ".join("?" for _ in col_names)
            vals = ["" for _ in col_names]
            idx = col_names.index(name_col)
            vals[idx] = "Ema Demo"
            tidx = None
            for n in ("ticket_id", "ticket id"):
                n2 = n.replace(" ", "_")
                if n2 in col_names:
                    tidx = col_names.index(n2)
                    break
            if tidx is not None:
                vals[tidx] = 999
            conn.execute(
                f'INSERT INTO support_tickets ({", ".join(chr(34) + c + chr(34) for c in col_names)}) VALUES ({placeholders})',
                vals,
            )
            conn.commit()
            print("Added seed row for Ema (no Ema in CSV).")
        conn.close()

    print(f"Seeded {len(rows)} rows from {csv_path.name} -> {DB_PATH}")


if __name__ == "__main__":
    main()

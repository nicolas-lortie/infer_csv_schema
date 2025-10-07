#!/usr/bin/env python3
import csv, math, re, os, sys
from datetime import datetime
from collections import defaultdict, namedtuple

# --- Config ---
NULL_STRINGS = {"", "null", "none", "na", "n/a", "nan"}
SAMPLE_LIMIT = None  # set to an int to speed up inference on huge files
DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y",
    "%Y/%m/%d", "%d-%m-%Y", "%m-%d-%Y"
]
TS_FORMATS = [
    "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M",
    "%d/%m/%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
    "%Y/%m/%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"
]
BOOL_TRUE = {"true","t","yes","y","1"}
BOOL_FALSE = {"false","f","no","n","0"}

# Dialect-specific SQL type names (you can add your own)
DIALECTS = {
    "postgres": {
        "INTEGER": "INTEGER",
        "BIGINT": "BIGINT",
        "DECIMAL": "DECIMAL",   # DECIMAL(p,s)
        "FLOAT": "DOUBLE PRECISION",
        "BOOLEAN": "BOOLEAN",
        "DATE": "DATE",
        "TIMESTAMP": "TIMESTAMP",
        "VARCHAR": "VARCHAR",   # VARCHAR(n)
    },
    "mysql": {
        "INTEGER": "INT",
        "BIGINT": "BIGINT",
        "DECIMAL": "DECIMAL",
        "FLOAT": "DOUBLE",
        "BOOLEAN": "TINYINT(1)",
        "DATE": "DATE",
        "TIMESTAMP": "DATETIME",
        "VARCHAR": "VARCHAR",
    },
    "sqlite": {
        # SQLite is dynamic typed; we still emit familiar names
        "INTEGER": "INTEGER",
        "BIGINT": "INTEGER",
        "DECIMAL": "NUMERIC",
        "FLOAT": "REAL",
        "BOOLEAN": "INTEGER",
        "DATE": "TEXT",
        "TIMESTAMP": "TEXT",
        "VARCHAR": "TEXT",
    },
    "ansi": {  # generic ANSI-ish
        "INTEGER": "INTEGER",
        "BIGINT": "BIGINT",
        "DECIMAL": "DECIMAL",
        "FLOAT": "FLOAT",
        "BOOLEAN": "BOOLEAN",
        "DATE": "DATE",
        "TIMESTAMP": "TIMESTAMP",
        "VARCHAR": "VARCHAR",
    },
}

Identifier = namedtuple("Identifier", "original safe")

def safe_identifier(name):
    # Make a simple SQL-safe identifier: letters, digits, underscore; no leading digit
    safe = re.sub(r"\W+", "_", name.strip())
    if re.match(r"^\d", safe):
        safe = "_" + safe
    if not safe:
        safe = "col"
    return Identifier(original=name, safe=safe)

def is_null(s):
    return s is None or s.strip().lower() in NULL_STRINGS

def is_bool(s):
    v = s.strip().lower()
    return v in BOOL_TRUE or v in BOOL_FALSE

def is_int(s):
    try:
        # reject floats that look like '1.0'
        if re.search(r"[.\,eE]", s):
            return False
        int(s.strip())
        return True
    except:
        return False

def is_decimal(s):
    # Accept 123.45 or -0.001 etc. Use '.' as decimal separator; adapt if needed
    try:
        float(s.strip())
        return True
    except:
        return False

def decimal_precision_scale(s):
    # Compute (precision, scale) ignoring leading '-' and decimal point
    s = s.strip()
    if s.startswith("-") or s.startswith("+"):
        s = s[1:]
    if "." in s:
        i, f = s.split(".", 1)
        # remove leading zeros in integer part for precision calc but keep length if zero
        i_digits = len(re.sub(r"^0+", "", i)) or (1 if set(i) == {"0"} else 0)
        p = i_digits + len(f)
        s_ = len(f)
    else:
        i = re.sub(r"^[-+]", "", s)
        i_digits = len(re.sub(r"^0+", "", i)) or (1 if set(i) == {"0"} else 0)
        p = i_digits
        s_ = 0
    return max(p,1), max(s_,0)

def is_date(s):
    t = s.strip()
    for fmt in DATE_FORMATS:
        try:
            datetime.strptime(t, fmt)
            return True
        except:
            pass
    return False

def is_timestamp(s):
    t = s.strip()
    for fmt in TS_FORMATS:
        try:
            datetime.strptime(t, fmt)
            return True
        except:
            pass
    # Allow ISO-like with timezone 'Z' or +/-hh:mm by a loose regex; map to TIMESTAMP w/o TZ for DDL
    if re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+\-]\d{2}:\d{2})$", t):
        return True
    return False

def update_stats(stats, value):
    # Track what each column could be, max varchar length, decimal precision/scale
    s = value.strip()
    stats["seen"] += 1
    if is_null(s):
        stats["nulls"] += 1
        return

    stats["max_len"] = max(stats["max_len"], len(s))

    if is_bool(s):
        stats["can_bool"] += 1
    if is_int(s):
        stats["can_int"] += 1
        # Track range to decide INT vs BIGINT
        try:
            n = int(s)
            stats["int_abs_max"] = max(stats["int_abs_max"], abs(n))
        except:
            pass
    elif is_decimal(s):
        stats["can_decimal"] += 1
        p, sc = decimal_precision_scale(s)
        stats["dec_prec"] = max(stats["dec_prec"], p)
        stats["dec_scale"] = max(stats["dec_scale"], sc)

    if is_date(s):
        stats["can_date"] += 1
    if is_timestamp(s):
        stats["can_ts"] += 1

def decide_type(stats, dialect):
    dmap = DIALECTS[dialect]
    non_null = stats["seen"] - stats["nulls"]

    # If every non-null value fits a more specific type, prefer that
    if non_null > 0 and stats["can_bool"] == non_null:
        return dmap["BOOLEAN"]

    if non_null > 0 and stats["can_int"] == non_null:
        # pick BIGINT if needed
        if stats["int_abs_max"] >= 2**31:  # conservative threshold
            return dmap["BIGINT"]
        return dmap["INTEGER"]

    if non_null > 0 and (stats["can_decimal"] + stats["can_int"]) == non_null:
        # Numbers with decimals: choose DECIMAL(p,s) if we have a scale, else FLOAT
        if stats["dec_scale"] > 0 and dmap["DECIMAL"]:
            p = min(max(stats["dec_prec"], stats["dec_scale"] + 1), 38)  # keep sane bound
            s = min(stats["dec_scale"], 18)
            return f'{dmap["DECIMAL"]}({p},{s})'
        else:
            return dmap["FLOAT"]

    if non_null > 0 and stats["can_date"] == non_null:
        return dmap["DATE"]

    if non_null > 0 and stats["can_ts"] == non_null:
        return dmap["TIMESTAMP"]

    # Fallback to VARCHAR(max_len)
    maxlen = max(1, stats["max_len"])
    # Cap to a reasonable size; you can adjust
    cap = 65535 if dialect in ("mysql",) else 4000
    n = min(maxlen, cap)
    return f'{dmap["VARCHAR"]}({n})'

def infer_schema(csv_path, dialect="ansi", header=True, delimiter=None, encoding="utf-8"):
    if dialect not in DIALECTS:
        raise ValueError(f"Unknown dialect '{dialect}'. Choose from: {', '.join(DIALECTS)}")

    with open(csv_path, "r", encoding=encoding, newline="") as f:
        # sniff delimiter if not given
        if delimiter is None:
            sample = f.read(1024*64)
            f.seek(0)
            try:
                dialect_sniff = csv.Sniffer().sniff(sample)
                delim = dialect_sniff.delimiter
            except:
                delim = ","
        else:
            delim = delimiter

        reader = csv.reader(f, delimiter=delim)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError("CSV is empty.")

        if header:
            headers = [c if c is not None else "" for c in first_row]
        else:
            # treat first row as data; synthesize headers
            headers = [f"col_{i+1}" for i in range(len(first_row))]
            # roll back to start and re-read from top
            f.seek(0)
            reader = csv.reader(f, delimiter=delim)

        col_ids = [safe_identifier(h) for h in headers]
        stats = []
        for _ in col_ids:
            stats.append({
                "seen": 0, "nulls": 0, "max_len": 0,
                "can_bool": 0, "can_int": 0, "int_abs_max": 0,
                "can_decimal": 0, "dec_prec": 0, "dec_scale": 0,
                "can_date": 0, "can_ts": 0,
            })

        row_count = 0
        for row in reader:
            # row length may vary; pad
            if len(row) < len(stats):
                row = row + [""] * (len(stats) - len(row))
            for i, val in enumerate(row[:len(stats)]):
                update_stats(stats[i], val if val is not None else "")
            row_count += 1
            if SAMPLE_LIMIT and row_count >= SAMPLE_LIMIT:
                break

        columns = []
        for col_id, st in zip(col_ids, stats):
            col_type = decide_type(st, dialect)
            not_null = " NOT NULL" if st["nulls"] == 0 and st["seen"] > 0 else ""
            columns.append((col_id, col_type, not_null))

        return {
            "delimiter": delim,
            "header": header,
            "columns": columns,
            "rows_scanned": row_count,
        }

def render_create(table_name, schema, dialect="ansi", quote_identifiers=True):
    q = '"' if quote_identifiers else ''
    lines = []
    for ident, coltype, notnull in schema["columns"]:
        # keep the original name if safe==original & quoting, otherwise use safe
        name = ident.safe
        lines.append(f"  {q}{name}{q} {coltype}{notnull}")
    body = ",\n".join(lines)
    qtn = f'{q}{safe_identifier(table_name).safe}{q}'
    return f"CREATE TABLE {qtn} (\n{body}\n);"

def main():
    import argparse
    p = argparse.ArgumentParser(description="Infer SQL CREATE TABLE from CSV")
    p.add_argument("csvs", nargs="+", help="Path(s) to CSV file(s)")
    p.add_argument("--table", help="Table name (default: derived from file name)")
    p.add_argument("--dialect", default="ansi", choices=DIALECTS.keys(), help="SQL dialect")
    p.add_argument("--no-header", action="store_true", help="Treat first row as data")
    p.add_argument("--delimiter", help="CSV delimiter (auto-detect if omitted)")
    p.add_argument("--encoding", default="utf-8")
    p.add_argument("--no-quote", action="store_true", help="Do not quote identifiers")
    args = p.parse_args()

    for path in args.csvs:
        schema = infer_schema(
            path,
            dialect=args.dialect,
            header=not args.no_header,
            delimiter=args.delimiter,
            encoding=args.encoding
        )
        tname = args.table or os.path.splitext(os.path.basename(path))[0]
        sql = render_create(tname, schema, dialect=args.dialect, quote_identifiers=not args.no_quote)
        print(f"-- Inferred from {path} (rows scanned: {schema['rows_scanned']}, delimiter: '{schema['delimiter']}')\n{sql}\n")

if __name__ == "__main__":
    # If you import this module, you can call infer_schema()/render_create() programmatically.
    if len(sys.argv) > 1:
        main()

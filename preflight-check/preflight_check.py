"""
preflight_check.py — a script to check which files in a bulk-load fileset will break the load.

A deliberately small script (~fifteen lines of real logic). It does one honest thing:
given a folder of delimited files and an expected schema (a baseline), it tells you
which files CONFORM and which are BROKEN — by name, with the exact reason.

Design stance (kept even in this tiny version):
  * DuckDB does the file reading + type inference; Python only orchestrates.
  * It reports STRUCTURE (names, types, counts), never DATA QUALITY (values).
  * It never guesses. A rename is shown as evidence, not asserted as a conclusion.
  * Every limit and every skip is a reported finding, never a silent drop.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import duckdb


# ---- coarse type vocabulary -------------------------------------------------
# We deliberately work in coarse classes, not precise SQL types. Structural
# equivalence cares whether a column is "int-ish" or "text-ish", not DECIMAL(10,2).
_ALIASES = {
    "bigint": "int", "int64": "int", "integer": "int", "int": "int",
    "smallint": "int", "hugeint": "int", "tinyint": "int", "long": "int",
    "double": "decimal", "float": "decimal", "real": "decimal",
    "numeric": "decimal", "decimal": "decimal", "number": "decimal", "money": "decimal",
    "varchar": "text", "text": "text", "string": "text", "char": "text",
    "boolean": "bool", "bool": "bool",
    "date": "date",
    "timestamp": "timestamp", "datetime": "timestamp",
}

def _coarse(duck_type: str) -> str:
    base = duck_type.lower().split("(")[0].strip()
    return _ALIASES.get(base, "text")   # unknown -> text (conservative)


# ---- typed result objects (the contract) ------------------------------------
@dataclass
class Finding:
    reason_code: str
    severity: str          # "ERROR" | "WARN"
    locus: str             # column name or position
    expected: str
    found: str
    explanation: str

@dataclass
class FileVerdict:
    path: str
    ok: bool
    findings: list[Finding] = field(default_factory=list)

@dataclass
class Report:
    golden: list[str]
    broken: list[FileVerdict]
    skipped: list[tuple[str, str]]   # (path, reason)

    def summary(self) -> str:
        return (f"{len(self.golden) + len(self.broken) + len(self.skipped)} files: "
                f"{len(self.golden)} conform, {len(self.broken)} broken, "
                f"{len(self.skipped)} unreadable")


# ---- baseline ---------------------------------------------------------------
def load_baseline(path: str) -> list[tuple[str, str]]:
    """A dumb, ordered `name,type` list. One column per line. That's it."""
    cols = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, typ = line.partition(",")
        cols.append((name.strip(), _coarse(typ.strip() or "text")))
    return cols


# ---- fingerprint one file (DuckDB does the work) ----------------------------
def fingerprint(con: duckdb.DuckDBPyConnection, path: str) -> list[tuple[str, str]]:
    """
    Return [(column_name, coarse_type), ...] as DuckDB's sniffer sees the file.
    We pass the path as a *parameter*, never string-interpolated into SQL.
    """
    rows = con.execute(
        "SELECT column_name, column_type "
        "FROM (DESCRIBE SELECT * FROM read_csv(?, sample_size=1000))",
        [path],
    ).fetchall()
    return [(name, _coarse(typ)) for name, typ in rows]


# ---- the check: name mode ---------------------------------------------------
def check_name_mode(observed, expected) -> list[Finding]:
    """
    Name-based bulk loads (Spark name-union, ADF byName, pandas concat) bind on
    NAME. So the drift that breaks them is name-level: missing / extra / retyped.
    """
    findings: list[Finding] = []
    obs = {n.lower(): t for n, t in observed}
    exp = {n.lower(): t for n, t in expected}

    for name, etype in expected:
        key = name.lower()
        if key not in obs:
            findings.append(Finding(
                "MISSING_COLUMN", "ERROR", name, etype, "(absent)",
                f"expected column '{name}' is missing"))
        elif obs[key] != etype:
            # widening int->decimal is compatible; anything else is breaking
            widening = (etype, obs[key]) == ("int", "decimal")
            findings.append(Finding(
                "TYPE_CHANGE", "WARN" if widening else "ERROR",
                name, etype, obs[key],
                f"column '{name}': expected {etype}, found {obs[key]}"
                + (" (widening, compatible)" if widening else "")))

    for name, otype in observed:
        if name.lower() not in exp:
            findings.append(Finding(
                "EXTRA_COLUMN", "ERROR", name, "(not in baseline)", otype,
                f"unexpected column '{name}' (baseline doesn't declare it)"))
    return findings


# ---- the check: position mode ----------------------------------------------
def check_position_mode(observed, expected) -> list[Finding]:
    """
    Positional bulk loads (fixed-width, ordinal COPY INTO, headerless) bind on
    POSITION. Deterministically catch count + type-at-position drift. If the file
    HAS names and the baseline HAS names, a positional name mismatch is a WARN
    (possible swap) — surfaced as evidence, never asserted, because the positional
    load itself won't fail on it. Same-type swaps with no names are physics-blind.
    """
    findings: list[Finding] = []
    if len(observed) != len(expected):
        findings.append(Finding(
            "COUNT_MISMATCH", "ERROR", "-", str(len(expected)), str(len(observed)),
            f"expected {len(expected)} columns, found {len(observed)}"))

    for i, ((oname, otype), (ename, etype)) in enumerate(zip(observed, expected)):
        if otype != etype:
            widening = (etype, otype) == ("int", "decimal")
            findings.append(Finding(
                "TYPE_AT_POSITION", "WARN" if widening else "ERROR",
                f"pos {i}", etype, otype,
                f"position {i}: expected {etype}, found {otype}"
                + (" (widening)" if widening else "")))
        elif ename and oname and oname.lower() != ename.lower():
            findings.append(Finding(
                "POSITIONAL_NAME_MISMATCH", "WARN", f"pos {i}", ename, oname,
                f"position {i}: header says '{oname}', baseline expects '{ename}' "
                f"(positional load unaffected; possible swap — verify)"))
    return findings


# ---- the partition ----------------------------------------------------------
def check_fileset(folder: str, baseline_path: str, align: str = "name") -> Report:
    expected = load_baseline(baseline_path)
    con = duckdb.connect()
    golden, broken, skipped = [], [], []

    # DuckDB expands the glob to a file list; we inspect each file independently
    # rather than letting a reader union them (that's the LOAD's job, and the
    # thing that silently misbehaves).
    files = [r[0] for r in con.execute(
        "SELECT file FROM glob(?)", [str(Path(folder) / "*.csv")]).fetchall()]

    checker = check_name_mode if align == "name" else check_position_mode

    for path in sorted(files):
        try:
            observed = fingerprint(con, path)
        except Exception as e:
            skipped.append((Path(path).name, str(e).splitlines()[0][:80]))
            continue
        findings = checker(observed, expected)
        errors = [f for f in findings if f.severity == "ERROR"]
        if errors or any(f.severity == "WARN" for f in findings):
            broken.append(FileVerdict(Path(path).name, ok=not errors, findings=findings))
        else:
            golden.append(Path(path).name)
    return Report(golden, broken, skipped)


# ---- a tiny renderer (the human view over the typed result) -----------------
def render(report: Report) -> str:
    out = [report.summary(), ""]
    if report.golden:
        out.append(f"GOLDEN ({len(report.golden)}) — safe to load:")
        for name in report.golden:
            out.append(f"  \u2713 {name}")
        out.append("")
    if report.broken:
        out.append(f"BROKEN ({len(report.broken)}) — fix or quarantine:")
        for v in report.broken:
            mark = "\u2717" if not v.ok else "\u26a0"
            out.append(f"  {mark} {v.path}")
            for f in v.findings:
                out.append(f"      [{f.severity}] {f.explanation}")
        out.append("")
    if report.skipped:
        out.append(f"UNREADABLE ({len(report.skipped)}):")
        for name, why in report.skipped:
            out.append(f"  ? {name} — {why}")
    return "\n".join(out)


if __name__ == "__main__":
    import sys
    folder, baseline = sys.argv[1], sys.argv[2]
    align = sys.argv[3] if len(sys.argv) > 3 else "name"
    print(render(check_fileset(folder, baseline, align)))

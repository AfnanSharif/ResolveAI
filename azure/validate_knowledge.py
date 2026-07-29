from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def validate(source: str | Path) -> dict[str, object]:
    path = Path(source)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "category", "title", "body", "keywords"}
        missing = required - set(reader.fieldnames or [])
        rows = list(reader)
    errors = []
    if missing: errors.append("missing columns: " + ", ".join(sorted(missing)))
    identifiers = [row.get("id", "").strip() for row in rows]
    if any(not identifier for identifier in identifiers): errors.append("article IDs cannot be empty")
    if len(identifiers) != len(set(identifiers)): errors.append("article IDs must be unique")
    if any(not row.get("body", "").strip() for row in rows): errors.append("article bodies cannot be empty")
    return {"valid": not errors, "articles": len(rows), "categories": sorted({row.get("category", "") for row in rows}), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--knowledge", required=True); parser.add_argument("--report", required=True)
    args = parser.parse_args(); result = validate(args.knowledge)
    destination = Path(args.report); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not result["valid"]: raise SystemExit(1)


if __name__ == "__main__": main()

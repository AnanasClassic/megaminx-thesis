import argparse
import csv
import json
from pathlib import Path

from .io import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", default=[])
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    paths = [Path(p) for p in args.paths] or sorted((ROOT / "logs").glob("search_*.json"))
    rows = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        row = {
            "file": str(path),
            "kind": data.get("kind"),
            "metric": data.get("metric"),
            "beam_width": data.get("beam_width"),
        }
        row.update(data["summary"])
        rows.append(row)
    out = Path(args.out) if args.out else ROOT / "logs" / "search_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["file"])
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()

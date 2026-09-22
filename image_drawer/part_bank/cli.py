"""Part Bank CLI; the original M0 CLI contract remains unchanged."""

from __future__ import annotations

import argparse
import json
import sqlite3
from typing import Sequence

from image_drawer.part_bank.extractor import WholeImageExtractor
from image_drawer.part_bank.ingest import SPLITS, ingest_directory


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="image-drawer-part-bank")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="ingest a directory of local images")
    ingest.add_argument("source")
    ingest.add_argument("--bank", required=True)
    ingest.add_argument("--dataset", required=True)
    ingest.add_argument("--split", choices=SPLITS)
    ingest.add_argument("--category", default="generic")
    args = parser.parse_args(argv)
    try:
        if not args.category.strip():
            raise ValueError("category must be non-empty")
        report = ingest_directory(
            args.source, args.bank, dataset=args.dataset, split=args.split,
            extractor=WholeImageExtractor(category=args.category),
        )
    except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
        import sys
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr)
        return 2
    print(report.to_json())
    return 1 if report.errors else 0

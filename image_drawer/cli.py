"""Command-line entrypoint for the Image Drawer plumbing layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def build_payload(input_value: str) -> dict[str, str]:
    """Return the deterministic M0 input/output contract payload."""
    return {"input": input_value, "status": "ok"}


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-drawer",
        description="Image Drawer command-line entrypoint.",
    )
    parser.add_argument(
        "--input",
        default="",
        help="Deterministic plumbing input used by the M0 smoke contract.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the JSON response instead of stdout.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    rendered = json.dumps(build_payload(args.input), sort_keys=True)

    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")

    return 0

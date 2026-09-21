"""GUI entrypoint.

M0 intentionally keeps this headless. The real workbench arrives in M6.
"""

from __future__ import annotations

import argparse
import json
from typing import Sequence


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-drawer-gui",
        description="Image Drawer GUI entrypoint placeholder.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a headless entrypoint check suitable for CI.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.check:
        print(json.dumps({"component": "gui", "status": "ok"}, sort_keys=True))
    else:
        print("Image Drawer GUI placeholder: workbench implementation is planned for M6.")
    return 0

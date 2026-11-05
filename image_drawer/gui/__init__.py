"""GUI entrypoint。

`--check`はheadlessのまま保つ。画面(tkinter)は引数なしで起動したときだけ読み込む。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-drawer-gui",
        description="Image Drawer GUI（Replay表示・生成・Dataset登録）。",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a headless entrypoint check suitable for CI.",
    )
    parser.add_argument("--open", metavar="EPISODE", help="起動時に開くepisode.json/draft.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.check:
        print(json.dumps({"component": "gui", "status": "ok"}, sort_keys=True))
        return 0
    try:
        from image_drawer.gui import app
    except ImportError as error:
        print(f"GUIを起動できません（tkinterが必要です）: {error}", file=sys.stderr)
        return 2
    try:
        code = app.run(args.open)
    except Exception as error:  # 画面の想定外の例外を終了コードに変える
        print(f"GUIが異常終了しました: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return code if isinstance(code, int) else 0

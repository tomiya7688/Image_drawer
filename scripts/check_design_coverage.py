"""設計書が名指しするテスト名が、tests/ に実在するかを照合する(設計→実装の漏れ検出)。

使い方:
  python scripts/check_design_coverage.py                 # 実装済みの設計書だけ(IMPLEMENTED)
  python scripts/check_design_coverage.py docs/developers/design_trainer.md
終了コード: 欠落があれば1。設計書の `test_xxx` 参照を、tests/ の def / クラス内メソッドと突き合わせる。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "developers"
# 実装済みとして照合を強制する設計書。実装PRで追加する。
IMPLEMENTED = [
    "design_stage_controller.md",
    "design_episode_replay.md",
    "design_data_pipeline_integration.md",
]
REF = re.compile(r"`(test_\w+)`|\b(test_\w+)\b")
DEF = re.compile(r"^\s*(?:async\s+)?def\s+(test_\w+)", re.M)


def defined_tests() -> set[str]:
    names: set[str] = set()
    for path in (ROOT / "tests").rglob("*.py"):
        names.update(DEF.findall(path.read_text(encoding="utf-8", errors="replace")))
    return names


def referenced(doc: Path) -> set[str]:
    text = doc.read_text(encoding="utf-8")
    out = set()
    for a, b in REF.findall(text):
        name = a or b
        if name.endswith("_") or name in {"test_xxx"}:
            continue
        out.add(name)
    return out


def main(argv: list[str]) -> int:
    docs = [Path(a) for a in argv] or [DOCS / n for n in IMPLEMENTED]
    have = defined_tests()
    # ファイル名(test_foo.py)への参照は照合対象外にする
    files = {p.stem for p in (ROOT / "tests").rglob("test_*.py")}
    failed = False
    for doc in docs:
        missing = sorted(n for n in referenced(doc) - have if n not in files)
        status = "OK" if not missing else f"欠落 {len(missing)} 件"
        print(f"{doc.name}: {status}")
        for n in missing:
            print(f"  - {n}")
        failed |= bool(missing)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

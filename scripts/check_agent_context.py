"""共有エージェント文書の容量と参照先を確認する。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


# 行数だけを詰めて肥大化を隠さないよう、UTF-8容量も同時に制限する。
SKILL = ".agents/skills/image-drawer-context"
BUDGETS = {
    "AGENTS.md": (24, 2400),
    "CLAUDE.md": (4, 100),
    "docs/kadoka-codex/coding_conventions.md": (140, 14000),
    ".github/copilot-instructions.md": (6, 400),
    f"{SKILL}/SKILL.md": (40, 3000),
    f"{SKILL}/references/routing.md": (40, 5000),
    f"{SKILL}/references/operations.md": (55, 3500),
    f"{SKILL}/references/integration.md": (35, 3000),
    f"{SKILL}/references/handoff.md": (28, 2000),
}


# {
#   責務: [
#     inspect_document: 一つの共有文書の容量とローカルリンクを確認する
#   ]
#   処理: [
#     1: UTF-8の文書を読み、行数と容量を上限と比較する
#     2: MarkdownリンクとClaudeの共有入口が実在するか確認する
#   ]
#   引数: [
#     root: リポジトリの基準ディレクトリ
#     relative: 対象文書の相対パス
#     limits: 行数とUTF-8バイト数の上限
#   ]
#   戻り値: [
#     report: 容量の短い表示
#     errors: 容量超過または参照先不足の一覧
#   ]
# }
def inspect_document(root: Path, relative: str, limits: tuple[int, int]) -> tuple[str, list[str]]:
    path = root / relative
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        return relative, [f"{relative}: 読み込めません: {exc}"]
    lines, size = len(text.splitlines()), len(raw)
    errors = []
    if lines > limits[0] or size > limits[1]:
        errors.append(f"{relative}: 上限超過 ({lines}/{limits[0]}行, {size}/{limits[1]}bytes)")

    # 外部URLは取得せず、常用する短い文書のローカル参照だけを検査する。
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text):
        parsed = urlsplit(target.strip("<>"))
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        destination = path.parent / unquote(parsed.path)
        if not destination.exists():
            errors.append(f"{relative}: 参照先がありません: {target}")
    if relative == "CLAUDE.md" and text.strip() != "@AGENTS.md":
        errors.append("CLAUDE.md: 共通入口へのimportだけを保持してください")
    return f"{relative}: {lines}/{limits[0]}行, {size}/{limits[1]}bytes", errors


# {
#   責務: [
#     main: 対象文書の確認結果を短く表示し、不備があれば失敗を返す
#   ]
#   処理: [
#     1: 指定されたルートの共通入口とスキル資料を確認する
#     2: 管理対象に未登録のスキル文書を検出する
#     3: 容量と参照先の確認結果を表示する
#   ]
#   引数: [
#     argv: CLI引数。省略時はプロセスの引数
#   ]
#   戻り値: [
#     exit_code: 不備がなければ0、不備があれば1
#   ]
# }
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args(argv).root.resolve()
    errors = []
    for relative, limits in BUDGETS.items():
        report, problems = inspect_document(root, relative, limits)
        print(report)
        errors.extend(problems)

    # 資料を増やして上限確認から漏れることを防ぎ、追加時の見直しを促す。
    for path in (root / SKILL).rglob("*.md"):
        relative = path.relative_to(root).as_posix()
        if relative not in BUDGETS:
            errors.append(f"{relative}: 文書追加時は容量と読む条件を登録してください")
    for error in errors:
        print(f"ERROR: {error}")
    print(f"共有文書{len(BUDGETS)}件: {'不備あり' if errors else '容量・ローカル参照に不備なし'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

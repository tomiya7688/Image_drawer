"""共有エージェント文書の容量と参照先を確認する。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


# 行数だけを詰めて肥大化を隠さないよう、UTF-8容量も同時に制限する。
SKILL = ".agents/skills/image-drawer-context"
REVIEW_START = "<!-- shared-review-operations:"
REVIEW_END = "<!-- /shared-review-operations -->"
REVIEW_BUDGET = (12, 2600)
REVIEW_DOCUMENTS = {"AGENTS.md", "CLAUDE.md"}
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
#     split_review_preamble: 先頭の共有規則とプロジェクト本文を区別する
#   ]
#   処理: [
#     1: 規則のない通常文書はそのまま返す
#     2: v2の区切り、先頭位置、一つだけのブロックを確認する
#   ]
#   引数: [
#     text: 入口文書の全文
#   ]
#   戻り値: [
#     body: プロジェクト固有の本文
#     review: 共有規則のブロック
#     errors: 区切りや配置の不備
#   ]
# }
def split_review_preamble(text: str) -> tuple[str, str, list[str]]:
    if REVIEW_START not in text and REVIEW_END not in text:
        return text, "", []
    # 終端行の改行だけを共有規則に含め、後続の空行も本文として数える。
    match = re.match(
        r"\A(<!-- shared-review-operations:v2:\d{4}-\d{2}-\d{2} -->"
        r"\r?\n.*?<!-- /shared-review-operations -->(?:\r?\n)?)",
        text, re.DOTALL,
    )
    if match is None:
        return text, "", ["共有レビュー規則の区切り・形式・先頭位置が不正です"]
    review, body = match.group(1), text[match.end():]
    if review.count(REVIEW_START) != 1 or REVIEW_START in body or REVIEW_END in body:
        return text, "", ["共有レビュー規則は先頭の一つだけにしてください"]
    return body, review, []


# {
#   責務: [
#     inspect_document: 一つの共有文書の容量とローカルリンクを確認する
#   ]
#   処理: [
#     1: UTF-8の文書を読み、本文と共有規則の行数・容量を個別に比較する
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
    body, review, errors = text, "", []
    if relative in REVIEW_DOCUMENTS:
        body, review, problems = split_review_preamble(text)
        errors.extend(f"{relative}: {problem}" for problem in problems)
    lines, size = len(body.splitlines()), len(body.encode("utf-8"))
    if lines > limits[0] or size > limits[1]:
        errors.append(f"{relative}: 本文の上限超過 ({lines}/{limits[0]}行, {size}/{limits[1]}bytes)")
    report = f"{relative}: 本文 {lines}/{limits[0]}行, {size}/{limits[1]}bytes"
    if review:
        review_lines, review_size = len(review.splitlines()), len(review.encode("utf-8"))
        report += f"; 共有規則 {review_lines}/{REVIEW_BUDGET[0]}行, {review_size}/{REVIEW_BUDGET[1]}bytes"
        if review_lines > REVIEW_BUDGET[0] or review_size > REVIEW_BUDGET[1]:
            errors.append(f"{relative}: 共有レビュー規則の上限超過")

    # 外部URLは取得せず、常用する短い文書のローカル参照だけを検査する。
    for target in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text):
        parsed = urlsplit(target.strip("<>"))
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        destination = path.parent / unquote(parsed.path)
        if not destination.exists():
            errors.append(f"{relative}: 参照先がありません: {target}")
    if relative == "CLAUDE.md" and body.strip() != "@AGENTS.md":
        errors.append("CLAUDE.md: 共通入口へのimportだけを保持してください")
    return report, errors


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

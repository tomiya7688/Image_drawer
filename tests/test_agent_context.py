"""共有レビュー規則を追加しても文書の容量制限が維持されることを確認する。"""

from pathlib import Path

import pytest

from scripts.check_agent_context import BUDGETS, REVIEW_BUDGET, inspect_document


REVIEW_START = "<!-- shared-review-operations:v2:2026-10-24 -->"
REVIEW_END = "<!-- /shared-review-operations -->"
REVIEW_HEADER = f"{REVIEW_START}\n共有レビュー規則\n{REVIEW_END}"


# {
#   責務: [
#     test_valid_entrypoints: 共有規則の有無と改行形式によらず正常な入口を受け入れる
#   ]
#   処理: [
#     1: 改行と共有規則を組み合わせた入口を作成し、検証する
#   ]
#   引数: [
#     tmp_path: 一時保存先
#     filename: 入口文書名
#     newline: 改行文字
#     prefix: 先頭に加える共有規則
#   ]
#   戻り値: []
# }
@pytest.mark.parametrize("filename", ["AGENTS.md", "CLAUDE.md"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("prefix", ["", REVIEW_HEADER + "\n\n"])
def test_valid_entrypoints(tmp_path: Path, filename: str, newline: str, prefix: str):
    body = "@AGENTS.md\n" if filename == "CLAUDE.md" else "# 共有入口\n必要な資料だけ参照します。\n"
    (tmp_path / filename).write_text((prefix + body).replace("\n", newline), encoding="utf-8", newline="")
    _, errors = inspect_document(tmp_path, filename, BUDGETS[filename])
    assert errors == []


# {
#   責務: [
#     test_reject_malformed_review_blocks: 区切り欠落、重複、埋め込み、不明な形式を拒否する
#   ]
#   処理: [
#     1: 本文容量を十分に確保し、構造だけが不正な入口を検証する
#   ]
#   引数: [
#     tmp_path: 一時保存先
#     text: 不正な配置または区切りを持つ文書
#   ]
#   戻り値: []
# }
@pytest.mark.parametrize("text", [
    REVIEW_START + "\n終了なし", REVIEW_END,
    "# 本文\n" + REVIEW_HEADER,
    REVIEW_HEADER + "\n" + REVIEW_HEADER,
    REVIEW_START + "\n" + REVIEW_HEADER + "\n" + REVIEW_END,
    REVIEW_HEADER + "\n" + REVIEW_END,
    REVIEW_HEADER.replace(":v2:", ":v1:"),
    REVIEW_START + "\n不正な終了\n<!-- /shared-review-operations ->",
])
def test_reject_malformed_review_blocks(tmp_path: Path, text: str):
    (tmp_path / "AGENTS.md").write_text(text, encoding="utf-8")
    _, errors = inspect_document(tmp_path, "AGENTS.md", (1000, 100000))
    assert errors


# {
#   責務: [
#     test_project_body_budget_remains_enforced: 共有規則で本文の肥大化を隠せないことを確認する
#   ]
#   処理: [
#     1: 行数、UTF-8容量、通常コメントで本文を上限超過させる
#   ]
#   引数: [
#     tmp_path: 一時保存先
#     body: 上限を超える本文
#   ]
#   戻り値: []
# }
@pytest.mark.parametrize("body", ["x\n" * 25, "あ" * 801, "<!-- " + "x" * 2401 + " -->", "\n" * 25])
def test_project_body_budget_remains_enforced(tmp_path: Path, body: str):
    (tmp_path / "AGENTS.md").write_text(REVIEW_HEADER + "\n\n" + body, encoding="utf-8")
    _, errors = inspect_document(tmp_path, "AGENTS.md", BUDGETS["AGENTS.md"])
    assert len(errors) == 1 and "本文" in errors[0]


# {
#   責務: [
#     test_review_budget_is_independent: 本文が小さくても共有規則の容量超過を拒否する
#   ]
#   処理: [
#     1: 共有規則の行数またはUTF-8容量を超過させる
#   ]
#   引数: [
#     tmp_path: 一時保存先
#     content: 共有規則内の過大な内容
#   ]
#   戻り値: []
# }
@pytest.mark.parametrize("content", ["x\n" * REVIEW_BUDGET[0], "あ" * REVIEW_BUDGET[1]])
def test_review_budget_is_independent(tmp_path: Path, content: str):
    text = f"{REVIEW_START}\n{content}\n{REVIEW_END}\n\n@AGENTS.md\n"
    (tmp_path / "CLAUDE.md").write_text(text, encoding="utf-8")
    _, errors = inspect_document(tmp_path, "CLAUDE.md", BUDGETS["CLAUDE.md"])
    assert len(errors) == 1 and "共有レビュー規則" in errors[0]


# {
#   責務: [
#     test_claude_still_requires_shared_import: Claude本文へ別の規則を追記できないことを確認する
#   ]
#   処理: [
#     1: 容量内だが共有import以外を含む本文を検証する
#   ]
#   引数: [
#     tmp_path: 一時保存先
#     prefix: 共有規則の有無
#   ]
#   戻り値: []
# }
@pytest.mark.parametrize("prefix", ["", REVIEW_HEADER + "\n\n"])
def test_claude_still_requires_shared_import(tmp_path: Path, prefix: str):
    (tmp_path / "CLAUDE.md").write_text(prefix + "@AGENTS.md\n追加規則", encoding="utf-8")
    _, errors = inspect_document(tmp_path, "CLAUDE.md", BUDGETS["CLAUDE.md"])
    assert len(errors) == 1


# {
#   責務: [
#     test_links_in_both_sections_are_checked: 共有規則内のリンクも本文と同様に確認する
#   ]
#   処理: [
#     1: 共有規則と本文のいずれかに存在しない参照先を置く
#   ]
#   引数: [
#     tmp_path: 一時保存先
#     in_review: 共有規則に不正リンクを置くか
#   ]
#   戻り値: []
# }
@pytest.mark.parametrize("in_review", [True, False])
def test_links_in_both_sections_are_checked(tmp_path: Path, in_review: bool):
    link = "[不明](missing.md)"
    header = REVIEW_HEADER.replace("共有レビュー規則", link) if in_review else REVIEW_HEADER
    (tmp_path / "AGENTS.md").write_text(header + "\n\n" + ("# 本文" if in_review else link), encoding="utf-8")
    _, errors = inspect_document(tmp_path, "AGENTS.md", BUDGETS["AGENTS.md"])
    assert len(errors) == 1 and "missing.md" in errors[0]


# {
#   責務: [
#     test_other_documents_cannot_hide_review_blocks: 入口以外の文書に特例を広げない
#   ]
#   処理: [
#     1: 少ない容量上限を持つ通常文書へ共有規則を含める
#   ]
#   引数: [
#     tmp_path: 一時保存先
#   ]
#   戻り値: []
# }
def test_other_documents_cannot_hide_review_blocks(tmp_path: Path):
    (tmp_path / "other.md").write_text(REVIEW_HEADER, encoding="utf-8")
    _, errors = inspect_document(tmp_path, "other.md", (1, 1))
    assert len(errors) == 1 and "本文" in errors[0]

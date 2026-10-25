"""作業開始前に、先に処理するPRと優先度の付いていないIssueを一覧する。"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from typing import Any

EXIT_OK = 0
EXIT_UNAVAILABLE = 2
EXIT_ATTENTION = 3

PRIORITY_ORDER = ("P0", "P1", "P2", "P3")
LABEL_PATTERN = re.compile(r"^priority:\s*(P[0-3])$", re.IGNORECASE)
TITLE_PATTERN = re.compile(r"^\s*\[(P[0-3])\]", re.IGNORECASE)
BODY_PATTERN = re.compile(r"###\s*優先度\s*\r?\n+\s*(P[0-3])\b", re.IGNORECASE)


# {
#   責務: [
#     priority_of: Issueに付けられた優先度をラベル・タイトル・本文の順に読み取る
#   ]
#   処理: [
#     1: ラベル priority: P0〜P3 を探す
#     2: なければタイトル先頭の [P0]〜[P3] を探す
#     3: なければ起票フォームの「優先度」欄を探す
#   ]
#   引数: [
#     issue: gh issue list --json の1件分
#   ]
#   戻り値: [
#     priority: "P0"〜"P3"。付いていなければ None
#   ]
# }
def priority_of(issue: dict[str, Any]) -> str | None:
    for label in issue.get("labels", []):
        matched = LABEL_PATTERN.match(label.get("name", ""))
        if matched:
            return matched.group(1).upper()
    matched = TITLE_PATTERN.match(issue.get("title", ""))
    if matched:
        return matched.group(1).upper()
    matched = BODY_PATTERN.search(issue.get("body") or "")
    return matched.group(1).upper() if matched else None


# {
#   責務: [
#     run_gh_json: GitHub CLIの一覧をJSONとして取得する
#   ]
#   処理: [
#     1: shellを使わず固定のargvでghを実行する
#     2: 起動不能・失敗・不正なJSONは呼び出し元が判断できる例外にする
#   ]
#   引数: [
#     arguments: gh に渡す引数
#   ]
#   戻り値: [
#     items: JSONの配列
#   ]
#   エラー: [
#     RuntimeError: ghが使えない、またはJSONを読めない
#   ]
# }
def run_gh_json(arguments: list[str]) -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["gh", *arguments], capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"ghを実行できません: {exc}") from exc
    if result.returncode != 0:
        raise RuntimeError(f"gh {' '.join(arguments[:2])} が失敗しました: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"ghの出力を読めません: {exc}") from exc


# {
#   責務: [
#     order_issues: 優先度の高い順、同じ優先度は番号の小さい順にIssueを並べる
#   ]
#   引数: [
#     issues: gh issue list --json の一覧
#   ]
#   戻り値: [
#     ordered: (優先度, Issue)の一覧。優先度なしは末尾
#   ]
# }
def order_issues(issues: list[dict[str, Any]]) -> list[tuple[str | None, dict[str, Any]]]:
    pairs = [(priority_of(issue), issue) for issue in issues]
    rank = {name: index for index, name in enumerate(PRIORITY_ORDER)}
    return sorted(pairs, key=lambda pair: (rank.get(pair[0], len(rank)), pair[1]["number"]))


# {
#   責務: [
#     main: 先に処理するPR、優先度順のIssue、優先度なしのIssueを表示する
#   ]
#   処理: [
#     1: オープンPRを表示する。あれば新しい作業より先に処理する
#     2: オープンIssueを PR → P0 → P1 → P2 → P3 の順で表示する
#     3: 優先度の付いていないIssueを、付与コマンド付きで表示する
#   ]
#   戻り値: [
#     exit_code: 問題なしは0、ghが使えなければ2、対応が必要なPRまたは優先度なしがあれば3
#   ]
# }
def main() -> int:
    try:
        pull_requests = run_gh_json(["pr", "list", "--state", "open", "--json", "number,title"])
        issues = run_gh_json(
            [
                "issue",
                "list",
                "--state",
                "open",
                "--limit",
                "200",
                "--json",
                "number,title,labels,body",
            ]
        )
    except RuntimeError as exc:
        print(f"優先度の確認をできませんでした: {exc}")
        return EXIT_UNAVAILABLE

    ordered = order_issues(issues)
    missing = [issue for priority, issue in ordered if priority is None]

    print(
        f"オープンPR: {len(pull_requests)}件"
        + ("(新しい作業より先に処理)" if pull_requests else "")
    )
    for pull_request in pull_requests:
        print(f"  PR #{pull_request['number']} {pull_request['title']}")
    print(f"オープンIssue: {len(issues)}件(PR → P0 → P1 → P2 → P3 の順)")
    for priority, issue in ordered:
        print(f"  {priority or '未設定'} #{issue['number']} {issue['title']}")
    if missing:
        print("優先度が未設定です。作業側で割り振ってください(P0〜P3):")
        for issue in missing:
            print(f'  gh issue edit {issue["number"]} --add-label "priority: P2"')
    return EXIT_ATTENTION if pull_requests or missing else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())

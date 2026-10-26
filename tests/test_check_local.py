"""ローカル検査の対象漏れと失敗終了を確認する。"""

import re
import subprocess
from pathlib import Path

import pytest

from scripts import check_local


EXPECTED_FAILURE_LABELS = {
    "format": "format",
    "lint": "lint",
    "test": "tests",
    "no_tests": "tests",
    "syntax": "discovery/syntax",
}


# {
#   責務: [
#     git_repository: 実Gitを使う独立した検査環境を用意する
#   ]
#   処理: [
#     最小の正常なリポジトリと比較元を作成する
#   ]
#   引数: [
#     tmp_path: 一時保存先
#   ]
#   戻り値: [
#     root: テスト用Gitルート
#   ]
# }
@pytest.fixture
def git_repository(tmp_path: Path) -> Path:
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.name", "Local check test"],
        ["git", "config", "user.email", "test@example.invalid"],
    ):
        subprocess.run(command, cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/check_agent_context.py").write_text("raise SystemExit(0)\n")
    (tmp_path / "test_example.py").write_text("def test_example():\n    assert True\n")
    (tmp_path / "sample.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"], cwd=tmp_path, check=True
    )
    return tmp_path


# {
#   責務: [
#     test_discover_all_change_states: commit後とstage・未stage・未追跡を取りこぼさない
#   ]
#   処理: [
#     空白入りの名前、削除、原文fixtureも含めて検査する
#   ]
#   引数: [
#     git_repository: 独立したGitルート
#   ]
#   戻り値: [
#     なし
#   ]
# }
def test_discover_all_change_states(git_repository: Path) -> None:
    root = git_repository
    (root / "committed.py").write_text("value = 1\n")
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "change"], cwd=root, check=True)
    (root / "staged.py").write_text("value = 2\n")
    subprocess.run(["git", "add", "staged.py"], cwd=root, check=True)
    (root / "sample.py").write_text("value = 3\n")
    (root / "new name.py").write_text("value = 4\n")
    (root / "tests/fixtures").mkdir(parents=True)
    (root / "tests/fixtures/raw.py").write_text("invalid original sample\n")
    (root / "test_example.py").unlink()
    assert check_local.discover_python_changes(root, "origin/main") == [
        "committed.py",
        "new name.py",
        "sample.py",
        "staged.py",
    ]


# {
#   責務: [
#     test_standard_entrypoint: 実際の構文・整形・pytest失敗を一つの入口で検出する
#   ]
#   処理: [
#     正常、構文不正、テスト不正、整形漏れ、対象なしを実行する
#   ]
#   引数: [
#     git_repository: 一時Git, monkeypatch: ルート差替え, capsys: 表示確認, case: ケース
#   ]
#   戻り値: [
#     なし
#   ]
# }
@pytest.mark.parametrize("case", ["valid", "syntax", "test", "format", "empty", "lint", "no_tests"])
def test_standard_entrypoint(git_repository, monkeypatch, capsys, case):
    root = git_repository
    monkeypatch.setattr(check_local, "ROOT", root)
    # 作業順の通信確認は専用テストに任せ、失敗ラベルの検証を外部GitHubへ依存させない。
    monkeypatch.setattr(
        check_local,
        "inspect_triage",
        lambda checker: checker.results.append(
            ("triage", "SKIP", "単体ケースでは外部通信を行わない")
        ),
    )
    if case != "empty":
        (root / "sample.py").write_text(
            {
                "syntax": "def broken(\n",
                "format": "value=2\n",
                "lint": "value = missing_name\n",
            }.get(case, "value = 2\n")
        )
    if case == "test":
        (root / "test_example.py").write_text("def test_example():\n    assert False\n")
    if case == "no_tests":
        (root / "test_example.py").unlink()
    result = check_local.main([])
    output = capsys.readouterr().out
    assert result == (0 if case in {"valid", "empty"} else 1)
    # 無関係な検査の失敗だけで意図的な失敗ケースが通ることを防ぐ。
    if case in EXPECTED_FAILURE_LABELS:
        assert f"FAIL: {EXPECTED_FAILURE_LABELS[case]}:" in output
    if case != "syntax":
        assert "SKIP: triage: 単体ケースでは外部通信を行わない" in output
    assert ("SKIP: syntax" if case == "empty" else "ローカル検査") in output
    assert "再実行: python scripts/check_local.py" in output
    if case == "syntax":
        for label in ("format", "lint", "context", "tests"):
            assert f"SKIP: {label}: 対象探索または構文の失敗により未実行" in output


# {
#   責務: [
#     test_runner_reports_unavailable_command: 起動不能を成功と区別する
#   ]
#   処理: [
#     存在しない実行ファイルを呼び、未検証かつ失敗終了を確認する
#   ]
#   引数: [
#     tmp_path: 実行場所
#   ]
#   戻り値: [
#     なし
#   ]
# }
def test_runner_reports_unavailable_command(tmp_path: Path) -> None:
    checker = check_local.LocalCheck(tmp_path)
    checker.run("missing", [str(tmp_path / "missing-command")])
    assert checker.results[0][1] == "UNVERIFIED"
    assert checker.summarize(0) == 1


# {
#   責務: [
#     test_unrelated_failure_cannot_satisfy_case: 別検査の失敗を対象ケースの成功根拠にできないことを確認する
#   ]
#   処理: [
#     文書だけが失敗する実行結果を渡し、各失敗ケースの検証が拒否することを確認する
#   ]
#   引数: [
#     git_repository: 独立したGit, monkeypatch: 入口の差替え, capsys: 出力確認, case: 対象ケース
#   ]
#   戻り値: [
#     なし
#   ]
# }
@pytest.mark.parametrize("case", list(EXPECTED_FAILURE_LABELS))
def test_unrelated_failure_cannot_satisfy_case(git_repository, monkeypatch, capsys, case):
    # 実行ツールを増やさず、偽陽性が成立する出力だけを再現する。
    # {
    #   責務: [
    #     fail_context_only: 対象外の検査だけが失敗した結果を再現する
    #   ]
    #   処理: [
    #     検査ラベルと共通表示を出力して失敗終了値を返す
    #   ]
    #   引数: [
    #     argv: 入口の呼出し形式との互換用
    #   ]
    #   戻り値: [
    #     exit_code: 文書失敗を表す1
    #   ]
    # }
    def fail_context_only(argv):
        print("FAIL: context: exit=1")
        print("ローカル検査; 再実行: python scripts/check_local.py")
        print("SKIP: triage: 単体ケースでは外部通信を行わない")
        for label in ("format", "lint", "context", "tests"):
            print(f"SKIP: {label}: 対象探索または構文の失敗により未実行")
        return 1

    monkeypatch.setattr(check_local, "main", fail_context_only)
    with pytest.raises(AssertionError, match=re.escape(f"FAIL: {EXPECTED_FAILURE_LABELS[case]}:")):
        test_standard_entrypoint(git_repository, monkeypatch, capsys, case)

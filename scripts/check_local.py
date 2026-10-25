"""PR前の軽い検査を一つの入口から実行する。"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUFF_VERSION = "0.15.22"


# {
#   責務: [
#     discover_python_changes: 比較元からの変更と未追跡のPythonを列挙する
#   ]
#   処理: [
#     GitのNUL区切り出力を読み、削除と原文保持のfixtureを除く
#   ]
#   引数: [
#     root: Gitルート, base: 比較元コミット
#   ]
#   戻り値: [
#     paths: 重複のない相対パス
#   ]
# }
def discover_python_changes(root: Path, base: str) -> list[str]:
    # 作業ツリーと比較するため、commit済み・stage済み・未stageを同時に含める。
    outputs = [
        subprocess.check_output(
            ["git", "diff", "--name-only", "--diff-filter=ACMR", "-z", base, "--"],
            cwd=root,
            timeout=30,
        ),
        subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=root, timeout=30
        ),
    ]
    paths = sorted(
        {os.fsdecode(name) for output in outputs for name in output.split(b"\0") if name}
    )
    selected = []
    for name in paths:
        path = root / name
        if path.suffix != ".py" or name.startswith("tests/fixtures/"):
            continue
        # 差分にあるリンクを経由してGitルート外を読み込まない。
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Pythonの検査対象が安全な通常ファイルではありません: {name}")
        if not path.is_file():
            raise ValueError(f"Pythonの検査対象を読めません: {name}")
        selected.append(name)
    return selected


# {
#   責務: [
#     inspect_python_syntax: bytecodeを生成せずPythonの構文を検査する
#   ]
#   処理: [
#     変更ファイルをPythonのエンコーディング規則で解析する
#   ]
#   引数: [
#     root: ルート, paths: 相対パス
#   ]
#   戻り値: [
#     なし。構文不正は例外
#   ]
# }
def inspect_python_syntax(root: Path, paths: list[str]) -> None:
    for name in paths:
        ast.parse((root / name).read_bytes(), filename=name)


# {
#   責務: [
#     LocalCheck: 固定コマンドの実行結果をまとめる
#   ]
#   フィールド: [
#     root: ルート, results: 検査結果
#   ]
#   処理: [
#     必須の失敗と実行不能を失敗終了へ集約する
#   ]
# }
class LocalCheck:
    # {
    #   責務: [
    #     __init__: 一回の検査状態を用意する
    #   ]
    #   処理: [
    #     検査ルートと空の結果を保持する
    #   ]
    #   引数: [
    #     root: 検査ルート
    #   ]
    #   戻り値: [
    #     なし
    #   ]
    # }
    def __init__(self, root: Path) -> None:
        self.root = root
        self.results: list[tuple[str, str, str]] = []

    # {
    #   責務: [
    #     run: 一つの検査を実行し、起動不能とタイムアウトも記録する
    #   ]
    #   処理: [
    #     shellを使わず固定argvを実行し、終了値を保存する
    #   ]
    #   引数: [
    #     label: 検査名, command: 実行argv
    #   ]
    #   戻り値: [
    #     なし
    #   ]
    # }
    def run(self, label: str, command: list[str]) -> None:
        print(f"\n--- {label} ---", flush=True)
        try:
            # 一つが失敗しても後続の検査を実行し、原因をまとめて表示する。
            result = subprocess.run(command, cwd=self.root, timeout=120, check=False)
            status = "PASS" if result.returncode == 0 else "FAIL"
            detail = f"exit={result.returncode}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            status, detail = "UNVERIFIED", str(exc)
        self.results.append((label, status, detail))

    # {
    #   責務: [
    #     summarize: 全結果と所要時間を表示して終了値を返す
    #   ]
    #   処理: [
    #     未検証を成功に含めず、再実行方法を案内する
    #   ]
    #   引数: [
    #     started: 開始時刻
    #   ]
    #   戻り値: [
    #     exit_code: 必須失敗または実行不能なら1
    #   ]
    # }
    def summarize(self, started: float) -> int:
        print("\n=== ローカル検査 ===")
        for label, status, detail in self.results:
            print(f"{status}: {label}: {detail}")
        print(f"時間: {time.monotonic() - started:.2f}s; 再実行: python scripts/check_local.py")
        print("SKIP: 配布・画像素材全件・GPU・実GUIはCIまたは個別の実環境で確認してください。")
        return int(any(status in {"FAIL", "UNVERIFIED"} for _, status, _ in self.results))


# {
#   責務: [
#     inspect_python_tools: 変更Pythonへ固定設定の整形確認とlintを適用する
#   ]
#   処理: [
#     対象なしをSKIPとして表示し、担当外のファイルを書き換えない
#   ]
#   引数: [
#     checker: 結果の所有者, paths: 変更Python
#   ]
#   戻り値: [
#     なし
#   ]
# }
def inspect_python_tools(checker: LocalCheck, paths: list[str]) -> None:
    if not paths:
        checker.results.extend((name, "SKIP", "変更Pythonなし") for name in ("format", "lint"))
        return
    checker.run(
        "format",
        [sys.executable, "-m", "ruff", "format", "--check", "--force-exclude", "--", *paths],
    )
    checker.run("lint", [sys.executable, "-m", "ruff", "check", "--force-exclude", "--", *paths])


# {
#   責務: [
#     main: 変更を調べ、整形確認・構文・lint・既存検査を実行する
#   ]
#   処理: [
#     比較元を解決し、結果を集約する。依存の導入は行わない
#   ]
#   引数: [
#     argv: CI用比較元または担当者の明示整形指定
#   ]
#   戻り値: [
#     exit_code: 検査の終了値
#   ]
# }
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", help="CIの比較元SHA。通常はorigin/mainとのmerge-base")
    parser.add_argument("--format", action="store_true", help="担当するPythonだけを明示整形する")
    parser.add_argument("files", nargs="*", help="--format時の担当ファイル（必須）")
    args = parser.parse_args(argv)
    checker, started = LocalCheck(ROOT), time.monotonic()
    # 整形も検査も同じ固定バージョンを使い、暗黙の依存導入は行わない。
    try:
        from importlib.metadata import version

        if version("ruff") != RUFF_VERSION:
            raise ValueError(f"ruff=={RUFF_VERSION}が必要です。開発用依存を導入してください")
    except (ImportError, ValueError) as exc:
        checker.results.append(("formatter", "UNVERIFIED", str(exc)))
        return checker.summarize(started)
    if args.format:
        # 整形は担当者が列挙したファイルだけに限定し、dirty全体へ広げない。
        if not args.files or any(
            Path(name).is_absolute()
            or not (ROOT / name).resolve().is_relative_to(ROOT)
            or (ROOT / name).is_symlink()
            or not name.endswith(".py")
            for name in args.files
        ):
            parser.error("--formatにはルート内の担当Pythonファイルを指定してください")
        checker.run(
            "format", [sys.executable, "-m", "ruff", "format", "--force-exclude", "--", *args.files]
        )
        return checker.summarize(started)
    if args.files:
        parser.error("ファイル指定は担当者の--format実行だけに利用できます")
    try:
        # ローカルmainでもHEADだけへ縮めず、変更済みの作業ツリーを含める。
        base = (
            args.base
            or subprocess.check_output(
                ["git", "merge-base", "HEAD", "origin/main"], cwd=ROOT, text=True, timeout=30
            ).strip()
        )
        paths = discover_python_changes(ROOT, base)
        checker.run("diff", ["git", "diff", "--check", base, "--"])
        inspect_python_syntax(ROOT, paths)
        checker.results.append(
            ("syntax", "PASS" if paths else "SKIP", f"{len(paths)} Python files")
        )
    except (OSError, ValueError, SyntaxError, subprocess.SubprocessError) as exc:
        checker.results.append(("discovery/syntax", "FAIL", str(exc)))
        checker.results.extend(
            (label, "SKIP", "対象探索または構文の失敗により未実行")
            for label in ("format", "lint", "context", "tests")
        )
        return checker.summarize(started)
    inspect_python_tools(checker, paths)
    checker.run("context", [sys.executable, "scripts/check_agent_context.py"])
    # 固有の一時パスを使い、他のpytest実行の所有権や既存ファイルに依存しない。
    temporary_root = ROOT / ".tmp"
    if temporary_root.is_symlink():
        checker.results.append(("tests", "FAIL", ".tmpのsymlinkは利用しません"))
        return checker.summarize(started)
    try:
        temporary_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="local-check-", dir=temporary_root) as directory:
            checker.run(
                "tests",
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--basetemp",
                    str(Path(directory) / "pytest"),
                ],
            )
    except OSError as exc:
        checker.results.append(("tests", "UNVERIFIED", str(exc)))
    return checker.summarize(started)


if __name__ == "__main__":
    raise SystemExit(main())

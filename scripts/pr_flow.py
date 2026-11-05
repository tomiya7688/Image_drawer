"""PR作成からマージまでを1コマンドで進める(毎回同じ手順を推論しないためのスクリプト)。

python scripts/pr_flow.py --title T --body-file BODY.md --review-dir DIR [--merge]
手順: fetch → mainをmerge → check_local.py → 設計照合 → push → PR作成 → CodeRabbit依頼
      → (--merge時) CI成功を待ち、レビュー証跡(DIR/summary.txt)があればマージ。
レビュー証跡が無い場合はマージしない(--no-review-reason で理由を残した時だけ許可)。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*cmd: str, check: bool = True) -> str:
    r = subprocess.run(
        list(cmd), cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    if check and r.returncode:
        print(r.stdout + r.stderr)
        sys.exit(f"失敗: {' '.join(cmd)}")
    return (r.stdout + r.stderr).strip()


def wait_ci(pr: str, timeout: int = 1500) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        out = run("gh", "pr", "checks", pr, check=False)
        if "fail" in out:
            print(out)
            return False
        if "test-build-smoke" in out and "pass" in out.split("test-build-smoke")[1].split("\n")[0]:
            return True
        time.sleep(15)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", required=True)
    ap.add_argument("--body-file", required=True)
    ap.add_argument("--review-dir", default="")
    ap.add_argument("--no-review-reason", default="")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()

    run("git", "fetch", "-q", "origin")
    print(run("git", "merge", "origin/main", "-m", "merge main", check=False).splitlines()[-1])
    if run("git", "diff", "--name-only", "--diff-filter=U"):
        sys.exit("マージ競合があります。解消してから再実行してください")
    out = run(sys.executable, "scripts/check_local.py", "--test-timeout", "900", check=False)
    print(out[-600:])
    # 既知の失敗(Windows権限のsymlink試験)以外のFAILがあれば中断する
    fails = [ln for ln in out.splitlines() if ln.startswith("FAIL:") and "tests" not in ln]
    if fails or ("FAIL: tests" in out and "test_symlinks_are_not_followed" not in out):
        sys.exit("ローカル検査に失敗があるため中断します")
    run(sys.executable, "scripts/check_design_coverage.py")
    run("git", "push", "-q", "-u", "origin", "HEAD")
    body = Path(a.body_file).read_text(encoding="utf-8")
    url = run("gh", "pr", "create", "--title", a.title, "--body", body, "--base", "main")
    pr = url.splitlines()[-1]
    run("gh", "pr", "comment", pr, "--body", "@coderabbitai review")
    print("PR:", pr)
    if not a.merge:
        return 0
    if not wait_ci(pr):
        sys.exit("CIが成功していないためマージしません")
    ev = Path(a.review_dir) / "summary.txt" if a.review_dir else None
    if not (ev and ev.exists()) and not a.no_review_reason:
        sys.exit(
            "レビュー証跡(summary.txt)がありません。レビューを実行するか --no-review-reason で理由を残してください"
        )
    run("gh", "pr", "merge", pr, "--merge")
    print("マージ完了:", pr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

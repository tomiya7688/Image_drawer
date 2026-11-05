"""PRの差分から、外部レビュー用の資料(packet)を作る。

使い方: python scripts/review_packet.py <base> <head> <出力ディレクトリ> [--paths P ...]
- 差分をファイル単位で詰め、1資料を12000バイト(UTF-8)以下に分割する(packet-01.txt, ...)。
- テスト・生成物・fixtureは除外する(--include-tests で含める)。
- 1ファイルが上限を超える場合は、hunk単位で分割する。途中で切らず、必ず次の資料へ続ける。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

LIMIT = 12000
HEADER = "コードレビュー資料(レビューのみ。編集・実行・他エージェント起動は禁止)。{i}/{n}。正しさ・競合・境界値・例外処理の欠陥だけを、行・発生条件・修正案つきで指摘してください。\n\n"
SKIP = re.compile(r"(^tests/|fixtures/|\.json$|\.png$|\.lock$|^docs/.*\.md$)")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout


def _split_hunks(chunk: str, room: int) -> list[str]:
    parts = re.split(r"(?m)^(?=@@ )", chunk)
    out: list[str] = []
    cur = ""
    for p in parts:
        while len(p.encode()) > room:
            lines = p.splitlines(keepends=True)
            take = ""
            for ln in lines:
                if len((take + ln).encode()) > room:
                    break
                take += ln
            if not take:
                take = lines[0][: room // 4]
            out.append(cur + take if cur else take)
            cur = ""
            p = p[len(take) :]
        if len((cur + p).encode()) > room:
            out.append(cur)
            cur = p
        else:
            cur += p
    if cur:
        out.append(cur)
    return [x for x in out if x]


def build(base: str, head: str, include_tests: bool, paths: list[str]) -> list[str]:
    names = _git(
        "diff", "--name-only", f"{base}...{head}", *(["--", *paths] if paths else [])
    ).split()
    units: list[str] = []
    room = LIMIT - len(HEADER.format(i=99, n=99).encode()) - 200
    for name in names:
        if not include_tests and SKIP.search(name):
            continue
        diff = _git("diff", "-U2", f"{base}...{head}", "--", name)
        if len(diff.encode()) <= room:
            units.append(diff)
        else:
            units.extend(_split_hunks(diff, room))
    packets: list[str] = []
    cur = ""
    for u in units:
        if len((cur + u).encode()) > room and cur:
            packets.append(cur)
            cur = ""
        cur += u
    if cur:
        packets.append(cur)
    return packets


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("head")
    ap.add_argument("out")
    ap.add_argument("--include-tests", action="store_true")
    ap.add_argument("--paths", nargs="*", default=[])
    a = ap.parse_args()
    packets = build(a.base, a.head, a.include_tests, a.paths)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for i, body in enumerate(packets, 1):
        text = HEADER.format(i=i, n=len(packets)) + body
        assert len(text.encode()) <= LIMIT, (i, len(text.encode()))
        (out / f"packet-{i:02d}.txt").write_text(text, encoding="utf-8", newline="\n")
    print(f"{len(packets)} packets -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

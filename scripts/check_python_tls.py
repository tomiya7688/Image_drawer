"""Python環境のpip・certifi・uv・sigstoreが証明書インシデント対応の下限を満たすか検査する。"""

from __future__ import annotations

import sys
from importlib import metadata

# {
#   責務: [
#     MINIMUM_VERSIONS: 確認対象パッケージと、必須の最小バージョンを保持する
#   ]
# }
MINIMUM_VERSIONS = {"pip": "26.2.2"}

# {
#   責務: [
#     OPTIONAL_MINIMUM_VERSIONS: 導入されている場合だけ最小バージョンを要求するパッケージを保持する
#     certifi: pipは独自のCA bundleを同梱するため、導入されていない環境では要求しない。CIは明示的に導入して確認する
#   ]
# }
OPTIONAL_MINIMUM_VERSIONS = {
    "certifi": "2026.10.19",
    "uv": "0.13.4",
    "sigstore": "4.5.1",
}


# {
#   責務: [
#     parse_version: 数値で始まるバージョン文字列を数値のタプルへ変換する
#   ]
#   処理: [
#     1: ドットで分割する
#     2: 各要素の先頭の数字部分だけを整数にする
#   ]
#   引数: [
#     text: 例 "26.2.2" のバージョン文字列
#   ]
#   戻り値: [
#     parts: 比較用の整数のタプル
#   ]
# }
def parse_version(text: str) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in text.split("."):
        # "0rc1" のような接尾辞は無視し、先頭の数字だけで下限を比較する。
        digits = ""
        for character in piece:
            if not character.isdigit():
                break
            digits += character
        parts.append(int(digits or 0))
    return tuple(parts)


# {
#   責務: [
#     find_violations: 導入済みパッケージを下限と照合し、不足を報告用の文字列で返す
#   ]
#   処理: [
#     1: 必須パッケージが未導入または下限未満ならエラーにする
#     2: 任意パッケージは導入されている場合だけ下限を確認する
#     3: 確認したバージョンを標準出力へ記録する
#   ]
#   戻り値: [
#     violations: 下限を満たさないパッケージの説明のリスト
#   ]
# }
def find_violations() -> list[str]:
    violations: list[str] = []
    print(f"python {sys.version.split()[0]} ({sys.executable})")

    for name, minimum in {**MINIMUM_VERSIONS, **OPTIONAL_MINIMUM_VERSIONS}.items():
        required = name in MINIMUM_VERSIONS
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            print(f"{name}: not installed")
            if required:
                violations.append(f"{name} is not installed (requires >= {minimum})")
            continue

        print(f"{name}: {installed}")
        if parse_version(installed) < parse_version(minimum):
            violations.append(f"{name} {installed} is older than {minimum}")

    return violations


# {
#   責務: [
#     report_ca_bundle: 実際に使われるCA bundleのパスを記録する
#   ]
#   処理: [
#     1: certifiが導入されていれば certifi.where() の実ファイルを出力する
#   ]
# }
def report_ca_bundle() -> None:
    try:
        import certifi
    except ImportError:
        return
    print(f"certifi.where(): {certifi.where()}")


# {
#   責務: [
#     main: 検査結果を表示し、下限未満があれば終了コード1で終了する
#   ]
#   戻り値: [
#     exit_code: 問題がなければ0、あれば1
#   ]
# }
def main() -> int:
    violations = find_violations()
    report_ca_bundle()
    for violation in violations:
        print(f"ERROR: {violation}", file=sys.stderr)
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())

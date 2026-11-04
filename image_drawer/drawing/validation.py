"""描画Engineの入力検証に使う小さな共通関数。"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


# {
#   責務: [
#     DrawingError: 描画Engineが拒否した入力を表す例外
#   ]
#   処理: [
#     ValueErrorを継承し、呼び出し側が不正入力を一括で捕捉できるようにする
#   ]
# }
class DrawingError(ValueError):
    """不正なAction・状態・設定を表す。"""


IDENTIFIER_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")


# {
#   責務: [
#     require_int: 真偽値を除く整数だけを範囲付きで受理する
#   ]
#   処理: [
#     1: 型が厳密にintであることを確認する
#     2: 範囲外なら例外にする
#   ]
#   引数: [
#     value: 検査する値, name: エラー表示名, minimum: 下限, maximum: 上限
#   ]
#   戻り値: [
#     value: 検査済みの整数
#   ]
#   エラー: [
#     DrawingError: 整数でない、または範囲外
#   ]
# }
def require_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    # boolはintの派生なので、type()で厳密に除外する。5.0なども拒否される。
    if type(value) is not int:
        raise DrawingError(f"{name} must be an integer, got {value!r}")
    if not minimum <= value <= maximum:
        raise DrawingError(f"{name} must be in {minimum}..{maximum}, got {value}")
    return value


# {
#   責務: [
#     require_identifier: レイヤーIDなどの識別子形式を検査する
#   ]
#   引数: [
#     value: 検査する値, name: エラー表示名
#   ]
#   戻り値: [
#     value: 検査済みの文字列
#   ]
#   エラー: [
#     DrawingError: 文字列でない、または形式違反
#   ]
# }
def require_identifier(value: Any, name: str) -> str:
    if type(value) is not str or IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise DrawingError(f"{name} must match [A-Za-z][A-Za-z0-9_-]{{0,63}}, got {value!r}")
    return value


# {
#   責務: [
#     require_exact_keys: 辞書のキー集合が仕様と完全一致することを検査する
#   ]
#   処理: [
#     不足キーと未知キーをどちらも拒否し、黙った補完をしない
#   ]
#   引数: [
#     payload: 検査する値, keys: 許可する必須キー, name: エラー表示名
#   ]
#   戻り値: [
#     payload: 検査済みの辞書
#   ]
#   エラー: [
#     DrawingError: 辞書でない、またはキー集合が不一致
#   ]
# }
def require_exact_keys(payload: Any, keys: set[str], name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise DrawingError(f"{name} must be an object")
    actual = set(payload.keys())
    if actual != keys:
        missing, unknown = sorted(keys - actual), sorted(actual - keys)
        raise DrawingError(f"{name} keys mismatch: missing={missing} unknown={unknown}")
    return payload


# {
#   責務: [
#     round_half_up: 非負の分子と正の分母の比を四捨五入（0.5は上）して整数にする
#   ]
#   引数: [
#     numerator: 分子, denominator: 分母
#   ]
#   戻り値: [
#     quotient: floor(numerator/denominator + 1/2)
#   ]
# }
def round_half_up(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)

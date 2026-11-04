"""Controllerの予算（受理するAction数と試行数の上限）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from .errors import StageConfigError

MAX_ACTIONS_LIMIT: Final[int] = 19_900  # Engineの履歴20,000件からsystem用の余裕100件を引く
DEFAULT_ATTEMPT_FLOOR: Final[int] = 100
DEFAULT_ATTEMPT_FACTOR: Final[int] = 5


# {
#   責務: [ Budget: 受理するactor=modelのAction数と、受理・拒否を通した試行数の上限 ]
#   フィールド: [
#     max_actions: 受理できるモデルAction数の上限(1..MAX_ACTIONS_LIMIT),
#     max_attempts: 試行数の上限。省略時はmax(100, 5*max_actions)へ解決され、生成後は常にint
#   ]
#   エラー: [ StageConfigError: 値が整数でない・範囲外、またはmax_attempts < max_actions ]
# }
@dataclass(frozen=True)
class Budget:
    max_actions: int
    max_attempts: int | None = None

    def __post_init__(self) -> None:
        # boolやfloatを数値として黙って受け入れない。
        if type(self.max_actions) is not int or not 1 <= self.max_actions <= MAX_ACTIONS_LIMIT:
            raise StageConfigError(f"max_actions must be an integer in 1..{MAX_ACTIONS_LIMIT}")
        if self.max_attempts is None:
            default = max(DEFAULT_ATTEMPT_FLOOR, DEFAULT_ATTEMPT_FACTOR * self.max_actions)
            object.__setattr__(self, "max_attempts", default)
        elif type(self.max_attempts) is not int or self.max_attempts < self.max_actions:
            raise StageConfigError("max_attempts must be an integer >= max_actions")

"""工程Controllerの例外型。すべて`DrawingError`を直接継承し、例外同士は継承しない。"""

from __future__ import annotations

from ..validation import DrawingError


# {
#   責務: [ StageConfigError: Budget・StyleIntent・Engine・画布など構築時の設定が不正 ]
# }
class StageConfigError(DrawingError):
    """Controllerを構築できない設定。"""


# {
#   責務: [ StageTableError: Stage JSONの内容がEngineの定義や固定順と食い違う ]
# }
class StageTableError(DrawingError):
    """Stage JSONの不整合。"""


# {
#   責務: [ StageClosedError: 閉じたControllerへproposeした ]
# }
class StageClosedError(DrawingError):
    """これ以上提案を受け付けない。"""


# {
#   責務: [ StageInvariantError: 審査済みの遷移グループの適用に失敗した（実装の不具合） ]
# }
class StageInvariantError(DrawingError):
    """不変条件違反。Engineはrollback済みで、Controllerは閉じる。"""

"""promptだけから決まるStyleIntent（版付きの保守的な規則判定器）。目標画像は受け取らない。"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final

from .errors import StageConfigError

STYLE_INTENT_VERSION: Final[str] = "style-intent-rules-v1"
IGNORED_PUNCTUATION: Final[str] = "、。,.!！?？・「」『』()（）"
MONOCHROME_POSITIVE: Final[tuple[str, ...]] = (
    "白黒の線画", "モノクロの線画", "モノクロ線画", "白黒線画", "線画のみ", "線画だけ",
    "blackandwhitelineart", "monochromelineart", "lineartonly",
)  # fmt: skip
MONOCHROME_NEGATIVE: Final[tuple[str, ...]] = (
    "カラー", "着色", "彩色", "色付き", "フルカラー", "水彩", "fullcolor", "colorful", "colored",
)  # fmt: skip
FLAT_POSITIVE: Final[tuple[str, ...]] = (
    "影なし", "影無し", "影をつけない", "影を付けない", "影を入れない", "陰影なし",
    "陰影をつけない", "陰影を付けない", "ノーシェーディング", "ベタ塗り", "フラットカラー",
    "フラットな塗り", "noshading", "withoutshading", "flatcolor",
)  # fmt: skip
FLAT_NEGATIVE: Final[tuple[str, ...]] = (
    "立体感", "グラデーション", "ハイライト", "厚塗り", "リアル",
)  # fmt: skip


# {
#   責務: [
#     StyleIntent: promptから読んだ作品指定。レビュー退出（基本色・陰影）の裏付けにだけ使う
#   ]
#   フィールド: [
#     prompt: 判定に使った原文, version: 規則表の版,
#     monochrome_line_art_only: 白黒の線画だけを求めると明示されているか,
#     flat_style_no_shading: 影なしのフラットな塗りと明示されているか
#   ]
# }
@dataclass(frozen=True)
class StyleIntent:
    prompt: str
    version: str
    monochrome_line_art_only: bool
    flat_style_no_shading: bool


# {
#   責務: [ _normalize: NFKC・casefold・空白と句読点の除去を行う（規則表の語も同じ関数で正規化する） ]
#   引数: [ text: 正規化する文字列 ]
#   戻り値: [ normalized: 部分文字列判定用の文字列 ]
# }
def _normalize(text: str) -> str:
    folded = unicodedata.normalize("NFKC", text).casefold()
    return "".join(
        char for char in folded if not char.isspace() and char not in IGNORED_PUNCTUATION
    )


# {
#   責務: [ _matches: 肯定表現が1つ以上あり、否定手がかりが1つもないときだけTrueを返す ]
# }
def _matches(normalized: str, positive: tuple[str, ...], negative: tuple[str, ...]) -> bool:
    if any(_normalize(cue) in normalized for cue in negative):
        return False
    return any(_normalize(phrase) in normalized for phrase in positive)


# {
#   責務: [ judge_style_intent: promptだけからStyleIntentを判定する（曖昧ならFalse） ]
#   処理: [ 1: promptを正規化 2: フラグごとに肯定表現と否定手がかりを部分文字列で判定 ]
#   引数: [ prompt: 描画の指示文（目標画像は渡さない） ]
#   戻り値: [ intent: 版付きのStyleIntent ]
#   エラー: [ StageConfigError: promptが文字列でない ]
# }
def judge_style_intent(prompt: str) -> StyleIntent:
    if type(prompt) is not str:
        raise StageConfigError("prompt must be a string")
    normalized = _normalize(prompt)
    return StyleIntent(
        prompt=prompt,
        version=STYLE_INTENT_VERSION,
        monochrome_line_art_only=_matches(normalized, MONOCHROME_POSITIVE, MONOCHROME_NEGATIVE),
        flat_style_no_shading=_matches(normalized, FLAT_POSITIVE, FLAT_NEGATIVE),
    )

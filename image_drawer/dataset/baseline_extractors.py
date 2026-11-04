"""モデルを使わず、画素だけから決定的にタグと説明文を作る基本の抽出器。

高精度なタグ付けモデルの代わりではない。モデル不要で動く既定値であり、
利用者はより高性能な抽出器を同じregistryに登録して差し替える。
"""

from __future__ import annotations

from typing import Any

from image_drawer.dataset.models import DISABLED_TAGGER_ID, ExtractorInfo

# 色名と代表RGB。距離が最小の名前を、その画素の色名とする。
_COLOR_NAMES = {
    "黒": (0, 0, 0),
    "白": (255, 255, 255),
    "灰色": (128, 128, 128),
    "赤": (220, 40, 40),
    "橙": (240, 140, 30),
    "黄": (240, 220, 40),
    "緑": (50, 170, 70),
    "青緑": (40, 170, 170),
    "青": (50, 80, 220),
    "紫": (140, 60, 190),
    "桃": (240, 130, 190),
    "茶": (120, 75, 40),
}


# {
#   責務: [
#     nearest_color_name: RGBに最も近い色名を返す
#   ]
#   引数: [
#     rgb: (赤, 緑, 青)の0〜255の整数
#   ]
#   戻り値: [
#     name: 色名。距離が同じ場合は名前の辞書順で最初のもの
#   ]
# }
def nearest_color_name(rgb: tuple[int, int, int]) -> str:
    def distance(name: str) -> tuple[int, str]:
        target = _COLOR_NAMES[name]
        return sum((a - b) ** 2 for a, b in zip(rgb, target)), name

    return min(_COLOR_NAMES, key=distance)


# {
#   責務: [
#     color_shares: 画像を縮小して色名ごとの面積割合を数える
#   ]
#   処理: [
#     1: 16x16に縮小して画素数を固定し、処理時間を画像の大きさに依存させない
#     2: 各画素を最も近い色名に分類し、割合の降順で返す
#   ]
#   引数: [
#     image: RGBのPillow画像
#   ]
#   戻り値: [
#     shares: (色名, 割合)の一覧。割合の降順、同率は名前順
#   ]
# }
def color_shares(image: Any) -> list[tuple[str, float]]:
    from PIL import Image

    with image.resize((16, 16), Image.Resampling.BOX) as small:
        data = small.tobytes()
    pixels = [tuple(data[i : i + 3]) for i in range(0, len(data), 3)]
    counts: dict[str, int] = {}
    for pixel in pixels:
        name = nearest_color_name(pixel[:3])
        counts[name] = counts.get(name, 0) + 1
    return sorted(
        ((name, count / len(pixels)) for name, count in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )


# {
#   責務: [
#     ColorLayoutTagExtractor: 主な色・明るさ・縦横の向きをタグにする基本の抽出器
#   ]
#   フィールド: [
#     info: id "color-layout-tags"、モデルなし(model_hash none)
#   ]
#   処理: [
#     面積が5%以上の色、明るさ、向きをconfidence付きのタグにする
#   ]
# }
class ColorLayoutTagExtractor:
    info = ExtractorInfo(
        id="color-layout-tags",
        version="v1",
        capabilities=["tag"],
        config_hash="share>=0.05;grid=16",
        languages=["ja"],
    )

    # {
    #   責務: [
    #     extract: 画像からタグを作る
    #   ]
    #   引数: [
    #     image: RGBのPillow画像
    #   ]
    #   戻り値: [
    #     tags: label と confidence を持つ辞書の一覧
    #   ]
    # }
    def extract(self, image: Any) -> list[dict[str, Any]]:
        tags = [
            {"label": name, "confidence": round(share, 4)}
            for name, share in color_shares(image)
            if share >= 0.05
        ]
        width, height = image.size
        orientation = "横長" if width > height else "縦長" if height > width else "正方形"
        tags.append({"label": orientation, "confidence": 1.0})
        gray = image.convert("L").resize((1, 1))
        brightness = gray.getpixel((0, 0)) / 255
        tags.append(
            {
                "label": "明るい"
                if brightness >= 0.6
                else "暗い"
                if brightness <= 0.3
                else "中間の明るさ",
                "confidence": 1.0,
            }
        )
        return tags


# {
#   責務: [
#     ColorLayoutCaptionExtractor: 主な色と向きから短い説明文を作る基本の抽出器
#   ]
#   フィールド: [
#     info: id "color-layout-caption"、モデルなし(model_hash none)
#   ]
# }
class ColorLayoutCaptionExtractor:
    info = ExtractorInfo(
        id="color-layout-caption",
        version="v1",
        capabilities=["caption"],
        config_hash="top=2;grid=16",
        languages=["ja"],
    )

    # {
    #   責務: [
    #     extract: 画像から短い説明文を作る
    #   ]
    #   引数: [
    #     image: RGBのPillow画像
    #   ]
    #   戻り値: [
    #     caption: 例「赤と白が中心の横長の画像」
    #   ]
    # }
    def extract(self, image: Any) -> str:
        top = [name for name, _ in color_shares(image)[:2]]
        width, height = image.size
        orientation = "横長" if width > height else "縦長" if height > width else "正方形"
        return f"{'と'.join(top)}が中心の{orientation}の画像"


# {
#   責務: [
#     NoTagExtractor: タグ抽出を無効にするための抽出器。何も出力せず、品質警告も出さない
#   ]
#   フィールド: [
#     info: id "none"。TagExtractorは利用者が無効にできる(説明文だけで学習用promptを作る)
#   ]
# }
class NoTagExtractor:
    info = ExtractorInfo(id=DISABLED_TAGGER_ID, version="v1", capabilities=["tag"])

    # {
    #   責務: [
    #     extract: 常に空のタグを返す
    #   ]
    #   戻り値: [
    #     tags: 空の一覧
    #   ]
    # }
    def extract(self, image: Any) -> list[dict[str, Any]]:
        return []

"""画像の読み込み・正規化・近重複検出。処理はversion付きで、教師画像の解像度を変えない。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

NORMALIZATION_VERSION = "normalize-v1"
EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"})
MAX_PIXELS = 40_000_000
MAX_FILE_BYTES = 256 * 1024 * 1024
NEAR_DUPLICATE_DISTANCE = 4
NEAR_DUPLICATE_COLOR = 12


# {
#   責務: [
#     hash_file: ファイルの内容を一定量ずつ読み、SHA-256を返す
#   ]
#   処理: [
#     読み込み量が上限を超えたら、画像として開く前に拒否する
#   ]
#   引数: [
#     path: 対象ファイル
#     max_bytes: 許容する最大バイト数
#   ]
#   戻り値: [
#     digest: 16進のSHA-256
#   ]
#   エラー: [
#     ValueError: ファイルが上限を超える
#   ]
# }
def hash_file(path: Path, max_bytes: int = MAX_FILE_BYTES) -> str:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"image exceeds max_file_bytes={max_bytes}")
            digest.update(chunk)
    return digest.hexdigest()


# {
#   責務: [
#     load_normalized: 静止画像を開き、向きと色空間を統一したRGB画像にする
#   ]
#   処理: [
#     1: 形式を確認し、アニメーションや壊れたファイルを拒否する(1フレームへ黙って変換しない)
#     2: 画素数の上限を、画素を展開する前に確認する
#     3: EXIFの回転を適用し、透明部分は白背景に合成してRGBにする
#     4: 解像度は変更しない
#   ]
#   引数: [
#     path: 画像ファイル
#     max_pixels: 許容する最大画素数
#   ]
#   戻り値: [
#     image: 正規化済みのRGB画像(呼び出し側が閉じる)
#   ]
#   エラー: [
#     ValueError: 形式が未対応、アニメーション、画素数の超過、壊れたファイル
#   ]
# }
def load_normalized(path: Path, max_pixels: int = MAX_PIXELS) -> Any:
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        with Image.open(path) as source:
            width, height = source.size
            if width * height > max_pixels:
                raise ValueError(f"image exceeds max_pixels={max_pixels}")
            if getattr(source, "n_frames", 1) > 1:
                raise ValueError("animated or multi-frame images are not supported")
            source.load()
            oriented = ImageOps.exif_transpose(source)
            if oriented.mode in ("RGBA", "LA", "P"):
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                background.alpha_composite(rgba)
                return background.convert("RGB")
            return oriented.convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"not a supported image: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"cannot read image: {exc}") from exc


# {
#   責務: [
#     ImageSignature: 近重複の判定に使う、構造と色の署名を保持する
#   ]
#   フィールド: [
#     structure: 9x8のグレースケールから作る64ビットの差分ハッシュ
#     colors: 4x4に縮小した各画素のRGB(48個の整数)
#   ]
# }
@dataclass(frozen=True, slots=True)
class ImageSignature:
    structure: int
    colors: tuple[int, ...]


# {
#   責務: [
#     image_signature: 差分ハッシュと色の署名を計算する
#   ]
#   処理: [
#     1: 9x8のグレースケールに縮小し、隣り合う画素の明暗を比べて差分ハッシュを作る
#     2: 4x4のRGBに縮小して色の署名を作る。単色や滑らかなグラデーションは差分ハッシュが同じになるため、色で区別する
#   ]
#   引数: [
#     image: RGB画像
#   ]
#   戻り値: [
#     signature: ImageSignature
#   ]
# }
def image_signature(image: Any) -> ImageSignature:
    from PIL import Image

    with image.convert("L").resize((9, 8), Image.Resampling.BOX) as small:
        pixels = small.tobytes()
    structure = 0
    for row in range(8):
        for column in range(8):
            structure = (structure << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    with image.resize((4, 4), Image.Resampling.BOX) as small:
        colors = tuple(small.tobytes())
    return ImageSignature(structure, colors)


# {
#   責務: [
#     is_near_duplicate: 2つの署名が、同一作品の派生(わずかな明るさ変更や再保存)とみなせるか判定する
#   ]
#   処理: [
#     差分ハッシュのハミング距離と、色の平均絶対差の両方が閾値以下のときだけ近重複とする
#   ]
#   戻り値: [
#     result: 近重複ならTrue
#   ]
# }
def is_near_duplicate(left: ImageSignature, right: ImageSignature) -> bool:
    if bin(left.structure ^ right.structure).count("1") > NEAR_DUPLICATE_DISTANCE:
        return False
    difference = sum(abs(a - b) for a, b in zip(left.colors, right.colors)) / len(left.colors)
    return difference <= NEAR_DUPLICATE_COLOR


# {
#   責務: [
#     group_near_duplicates: 署名が近い画像を同じグループにまとめる
#   ]
#   処理: [
#     1: 全組を調べ、近重複を連結する(union-find)
#     2: グループIDは、メンバーのSHA-256のうち辞書順で最小のものから作る
#   ]
#   引数: [
#     signatures: image_sha256から署名への対応
#   ]
#   戻り値: [
#     groups: image_sha256からgroup_idへの対応
#   ]
# }
def group_near_duplicates(signatures: dict[str, ImageSignature]) -> dict[str, str]:
    keys = sorted(signatures)
    parent = {key: key for key in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for index, left in enumerate(keys):
        for right in keys[index + 1 :]:
            if is_near_duplicate(signatures[left], signatures[right]):
                parent[find(right)] = find(left)
    roots: dict[str, str] = {}
    for key in keys:
        roots.setdefault(find(key), key)
    return {key: f"group-{roots[find(key)][:12]}" for key in keys}

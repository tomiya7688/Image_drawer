"""正規化済みの画像を、メタデータを含まない固定設定のPNGにする(外部へ送る前の最小化)。"""

from __future__ import annotations

import io
from typing import Any

from image_drawer.prompting.core import PromptExtractionError

PNG_MAX_BYTES = 8 * 1024 * 1024


# {
#   責務: [
#     encode_png: 正規化済みのRGB画像を、メタデータを含まない固定設定のPNGのbytesにする
#   ]
#   処理: [
#     画素だけを新しい画像へ写し(ICC・EXIF・dpiなどを持ち越さない)、固定設定で符号化する。
#     上限を超えても縮小せず失敗にする(品質を勝手に下げた画像を送らない)
#   ]
#   引数: [
#     image: PIL画像
#     max_bytes: 許容するPNGの最大バイト数。省略時はPNG_MAX_BYTES(8MiB)
#   ]
#   戻り値: [
#     data: PNGのbytes
#   ]
#   エラー: [
#     PromptExtractionError: PNGがmax_bytesを超えた(kind=image_too_large)
#   ]
# }
def encode_png(image: Any, max_bytes: int | None = None) -> bytes:
    from PIL import Image

    limit = PNG_MAX_BYTES if max_bytes is None else max_bytes
    rgb = image.convert("RGB")
    clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG", compress_level=6)
    data = buffer.getvalue()
    if len(data) > limit:
        raise PromptExtractionError("image_too_large", f"PNGが上限{limit}バイトを超えた")
    return data

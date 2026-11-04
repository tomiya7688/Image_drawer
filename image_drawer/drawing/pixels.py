"""RGBA8の整数合成と、レイヤー1枚分の画素バッファ。"""

from __future__ import annotations

from .validation import round_half_up


# {
#   責務: [
#     PixelBuffer: キャンバスと同じ大きさのRGBA8画素を1枚分保持する
#   ]
#   フィールド: [
#     width: 幅, height: 高さ, data: 行優先のRGBAバイト列（初期は全て0）
#   ]
#   処理: [
#     1: 画素の読み書きと、整数source-over・destination-outを提供する
#   ]
# }
class PixelBuffer:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.data = bytearray(width * height * 4)

    # {
    #   責務: [ get: 画素1個のRGBAを返す ]
    #   引数: [ x: 列, y: 行 ]
    #   戻り値: [ pixel: [r, g, b, a] ]
    # }
    def get(self, x: int, y: int) -> list[int]:
        offset = (y * self.width + x) * 4
        return list(self.data[offset : offset + 4])

    # {
    #   責務: [ paint_over: 画素1個へ(色, alpha)を整数source-overで1回合成する ]
    #   引数: [ x: 列, y: 行, rgb: 色, alpha: 0..255 ]
    # }
    def paint_over(self, x: int, y: int, rgb: tuple[int, int, int], alpha: int) -> None:
        offset = (y * self.width + x) * 4
        out = over_rgba((rgb[0], rgb[1], rgb[2], alpha), self.data[offset : offset + 4])
        self.data[offset : offset + 4] = bytes(out)

    # {
    #   責務: [ clear_pixel: 画素1個を完全な透明 [0,0,0,0] へ置換する ]
    #   引数: [ x: 列, y: 行 ]
    # }
    def clear_pixel(self, x: int, y: int) -> None:
        offset = (y * self.width + x) * 4
        self.data[offset : offset + 4] = b"\x00\x00\x00\x00"

    # {
    #   責務: [ erase_alpha: 画素のalphaをdestination-outで減らす ]
    #   処理: [ new = round_half_up(Ad*(255-As)/255)。0になったら [0,0,0,0] にする ]
    #   引数: [ x: 列, y: 行, alpha: 消去の強さ0..255 ]
    # }
    def erase_alpha(self, x: int, y: int, alpha: int) -> None:
        offset = (y * self.width + x) * 4
        destination_alpha = self.data[offset + 3]
        remaining = round_half_up(destination_alpha * (255 - alpha), 255)
        if remaining == 0:
            self.data[offset : offset + 4] = b"\x00\x00\x00\x00"
        else:
            self.data[offset + 3] = remaining

    # {
    #   責務: [ rows: 画素を [行][列][r,g,b,a] の入れ子リストで返す ]
    #   戻り値: [ rows: テストと保存用の複製 ]
    # }
    def rows(self) -> list[list[list[int]]]:
        return [[self.get(x, y) for x in range(self.width)] for y in range(self.height)]


# {
#   責務: [
#     over_rgba: RGBA8の整数source-overを1画素に適用する
#   ]
#   処理: [
#     D = 255*As + Ad*(255-As)、Aout = round(D/255)、
#     Cout = round((Cs*As*255 + Cd*Ad*(255-As))/D)。D==0なら全て0
#   ]
#   引数: [
#     source: 上の画素(r,g,b,a), destination: 下の画素(r,g,b,a)
#   ]
#   戻り値: [
#     pixel: 合成後の[r,g,b,a]
#   ]
# }
def over_rgba(
    source: tuple[int, ...] | list[int], destination: bytes | bytearray | list[int]
) -> list[int]:
    source_alpha, dest_alpha = source[3], destination[3]
    denominator = 255 * source_alpha + dest_alpha * (255 - source_alpha)
    if denominator == 0:
        return [0, 0, 0, 0]
    channels = [
        round_half_up(
            source[c] * source_alpha * 255 + destination[c] * dest_alpha * (255 - source_alpha),
            denominator,
        )
        for c in range(3)
    ]
    return [*channels, round_half_up(denominator, 255)]


# {
#   責務: [
#     composite_on_white: 下から上へ並べたレイヤーを白背景へ合成する
#   ]
#   処理: [
#     alpha 0の画素は変化しないため飛ばし、他は整数source-overを順に適用する
#   ]
#   引数: [
#     buffers: 下から上の順の画素バッファ, width: 幅, height: 高さ
#   ]
#   戻り値: [
#     rgba: 行優先のRGBAバイト列（alphaは常に255）
#   ]
# }
def composite_on_white(buffers: list[PixelBuffer], width: int, height: int) -> bytes:
    output = bytearray(b"\xff" * (width * height * 4))
    for buffer in buffers:
        source = buffer.data
        for offset in range(0, width * height * 4, 4):
            if source[offset + 3] == 0:
                continue
            output[offset : offset + 4] = bytes(
                over_rgba(source[offset : offset + 4], output[offset : offset + 4])
            )
    return bytes(output)

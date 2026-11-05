"""標準ライブラリだけのRGBA8 PNG符号化・復号。同じ画素から同じバイト列を得る。"""

from __future__ import annotations

import struct
import zlib

SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_PIXELS = 1024 * 1024 * 4


# {
#   責務: [ PngError: PNGを読めない・未対応の形式 ]
# }
class PngError(ValueError):
    pass


def _chunk(kind: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)


# {
#   責務: [ encode_png_rgba: RGBA8をPNGにする。filter 0・圧縮レベル固定でメタデータを持たない ]
# }
def encode_png_rgba(width: int, height: int, rgba: bytes) -> bytes:
    if width < 1 or height < 1 or len(rgba) != width * height * 4:
        raise PngError("画素数が寸法と一致しません")
    stride = width * 4
    raw = b"".join(b"\x00" + rgba[row * stride : (row + 1) * stride] for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        SIGNATURE
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw, 6))
        + _chunk(b"IEND", b"")
    )


def _paeth(left: int, up: int, corner: int) -> int:
    estimate = left + up - corner
    near_left, near_up, near_corner = (abs(estimate - v) for v in (left, up, corner))
    if near_left <= near_up and near_left <= near_corner:
        return left
    return up if near_up <= near_corner else corner


def _unfilter(kind: int, line: bytearray, prior: bytes, step: int) -> None:
    for i in range(len(line)):
        left = line[i - step] if i >= step else 0
        up = prior[i]
        corner = prior[i - step] if i >= step else 0
        base = (0, left, up, (left + up) // 2, _paeth(left, up, corner))[kind]
        line[i] = (line[i] + base) & 0xFF


# {
#   責務: [ decode_png_rgba: 非interlaceの8bit RGB/RGBA PNGを(幅, 高さ, RGBA)にする ]
#   エラー: [ PngError: 署名・chunk・形式・サイズの不正 ]
# }
def decode_png_rgba(data: bytes) -> tuple[int, int, bytes]:
    if not data.startswith(SIGNATURE):
        raise PngError("PNGの署名がありません")
    position, header, idat = len(SIGNATURE), None, bytearray()
    while position + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[position : position + 8])
        body = data[position + 8 : position + 8 + length]
        if len(body) != length:
            raise PngError("chunkが途中で切れています")
        if kind == b"IHDR":
            header = struct.unpack(">IIBBBBB", body)
        elif kind == b"IDAT":
            idat += body
        elif kind == b"IEND":
            break
        position += 12 + length
    if header is None:
        raise PngError("IHDRがありません")
    width, height, depth, color, _compression, _filter, interlace = header
    if depth != 8 or color not in (2, 6) or interlace != 0:
        raise PngError("8bitのRGB/RGBA・非interlaceだけに対応します")
    if width < 1 or height < 1 or width * height > MAX_PIXELS:
        raise PngError("寸法が範囲外です")
    channels = 4 if color == 6 else 3
    stride = width * channels
    try:
        raw = zlib.decompress(bytes(idat), 15, (stride + 1) * height + 1)
    except zlib.error as error:
        raise PngError(f"画素データを展開できません: {error}") from error
    if len(raw) != (stride + 1) * height:
        raise PngError("画素データの長さが寸法と一致しません")
    out = bytearray()
    prior = bytes(stride)
    for row in range(height):
        start = row * (stride + 1)
        kind = raw[start]
        if kind > 4:
            raise PngError("未知のfilterです")
        line = bytearray(raw[start + 1 : start + 1 + stride])
        _unfilter(kind, line, prior, channels)
        prior = bytes(line)
        if channels == 4:
            out += line
        else:
            for i in range(0, stride, 3):
                out += line[i : i + 3] + b"\xff"
    return width, height, bytes(out)

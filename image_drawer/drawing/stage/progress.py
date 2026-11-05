"""工程内の有効筆跡の計数と、画素差分・残存画素の判定。"""

from __future__ import annotations

from dataclasses import dataclass

from ..models import DrawingState, Layer


# {
#   責務: [ StageProgress: 現在の工程で、画素を実際に変えた筆跡/消去の数。工程入場時に0へ戻す ]
#   フィールド: [ effective_strokes: 有効なstroke_v2の数, effective_erases: 有効なerase_v2の数 ]
# }
@dataclass
class StageProgress:
    effective_strokes: int = 0
    effective_erases: int = 0

    # {
    #   責務: [ total: 有効な筆跡と消去の合計 ]
    # }
    @property
    def total(self) -> int:
        return self.effective_strokes + self.effective_erases

    # {
    #   責務: [ reset: 計数を0へ戻す（工程入場時） ]
    # }
    def reset(self) -> None:
        self.effective_strokes = 0
        self.effective_erases = 0

    # {
    #   責務: [ record: 画素を変えた描画を1件数える ]
    #   引数: [ action_type: "stroke_v2"または"erase_v2" ]
    # }
    def record(self, action_type: str) -> None:
        if action_type == "stroke_v2":
            self.effective_strokes += 1
        else:
            self.effective_erases += 1


# {
#   責務: [ snapshot_pixels: レイヤーの画素をバイト列として複製する（描画前後の比較用） ]
# }
def snapshot_pixels(layer: Layer) -> bytes:
    return bytes(layer.pixels.data)


# {
#   責務: [ pixels_changed: 描画前の複製と現在の画素がバイト単位で異なるか（近似を使わない） ]
# }
def pixels_changed(before: bytes, layer: Layer) -> bool:
    return before != layer.pixels.data


# {
#   責務: [ has_visible_paint: 指定roleの可視レイヤーにalpha>0の画素が残っているか ]
#   引数: [ state: 状態, roles: 対象のrole ]
# }
def has_visible_paint(state: DrawingState, roles: tuple[str, ...]) -> bool:
    return any(
        any(layer.pixels.data[3::4])
        for layer in state.layers
        if layer.visible and layer.role in roles
    )

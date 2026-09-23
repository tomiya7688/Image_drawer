"""交換可能でmodel-freeなregion extraction contract。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Protocol

from image_drawer.core import SourceImage

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True, slots=True)
class Region:
    """stored-pixel coordinateで表したcrop: x, y, width, height。"""

    bbox: tuple[int, int, int, int]
    category: str = "generic"


class PartExtractor(Protocol):
    """extraction挙動を変更した実装はversionを更新しなければならない。"""

    method: str
    version: str

    def extract(self, source: SourceImage, image: Image) -> Iterable[Region]: ...


@dataclass(frozen=True, slots=True)
class WholeImageExtractor:
    """development baselineとして画像全体を1 regionにする。semantic segmentationではない。"""

    category: str = "generic"
    method: str = "whole_image"
    version: str = "1"

    def extract(self, source: SourceImage, image: Image) -> Iterable[Region]:
        return [Region((0, 0, source.width, source.height), self.category)]

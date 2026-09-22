"""Replaceable, model-free region extraction contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Protocol

from image_drawer.core import SourceImage

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True, slots=True)
class Region:
    """A crop in stored-pixel coordinates: x, y, width, height."""

    bbox: tuple[int, int, int, int]
    category: str = "generic"


class PartExtractor(Protocol):
    """Implementations must bump version when extraction behavior changes."""

    method: str
    version: str

    def extract(self, source: SourceImage, image: Image) -> Iterable[Region]: ...


@dataclass(frozen=True, slots=True)
class WholeImageExtractor:
    """Development baseline: one full-image region, not semantic segmentation."""

    category: str = "generic"
    method: str = "whole_image"
    version: str = "1"

    def extract(self, source: SourceImage, image: Image) -> Iterable[Region]:
        return [Region((0, 0, source.width, source.height), self.category)]

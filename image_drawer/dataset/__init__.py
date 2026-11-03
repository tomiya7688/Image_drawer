"""画像だけで学ぶためのDataset登録、抽出器registry、Policy入力と目標の分離。"""

from image_drawer.dataset.baseline_extractors import (
    ColorLayoutCaptionExtractor,
    ColorLayoutTagExtractor,
)
from image_drawer.dataset.ingest import IngestReport, build_prompt, register_images
from image_drawer.dataset.manifest import DatasetManifest, iter_samples
from image_drawer.dataset.models import (
    ExtractorInfo,
    PolicyInput,
    QualityIssue,
    RewardTarget,
    TrainingImage,
)
from image_drawer.dataset.registry import ExtractorRegistry


# {
#   責務: [
#     create_default_registry: モデル不要の基本の抽出器だけを登録したregistryを作る
#   ]
#   戻り値: [
#     registry: 既定のExtractorRegistry。高性能な抽出器は利用者が同じregistryへ登録する
#   ]
# }
def create_default_registry() -> ExtractorRegistry:
    registry = ExtractorRegistry()
    registry.register_tagger(ColorLayoutTagExtractor())
    registry.register_captioner(ColorLayoutCaptionExtractor())
    return registry


__all__ = [
    "ColorLayoutCaptionExtractor",
    "ColorLayoutTagExtractor",
    "DatasetManifest",
    "ExtractorInfo",
    "ExtractorRegistry",
    "IngestReport",
    "PolicyInput",
    "QualityIssue",
    "RewardTarget",
    "TrainingImage",
    "build_prompt",
    "create_default_registry",
    "iter_samples",
    "register_images",
]

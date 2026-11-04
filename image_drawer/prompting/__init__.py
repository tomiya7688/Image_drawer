"""学習前の画像→Caption/タグ抽出、分類・整形とモデル登録。"""

from .core import (
    ExtractionAttempt,
    ExtractionResult,
    ImagePromptPipeline,
    PromptExtractionError,
    SceneDescription,
    VisionResponse,
    normalize_prompt,
    parse_scene,
)
from .openai_compatible import OpenAICompatibleVisionProvider
from .registry import (
    ImageTrainingPreprocessor,
    PreprocessorRegistry,
    TrainingPreprocessingResult,
)
from .tagger_http import HTTPTagProvider
from .tagging import (
    PreparedTags,
    ScoredTag,
    TagExtractionResult,
    TagFormatter,
    TagNormalizer,
    TagSet,
    TaggingPipeline,
)

__all__ = [
    "ExtractionAttempt",
    "ExtractionResult",
    "HTTPTagProvider",
    "ImagePromptPipeline",
    "ImageTrainingPreprocessor",
    "OpenAICompatibleVisionProvider",
    "PreparedTags",
    "PreprocessorRegistry",
    "PromptExtractionError",
    "SceneDescription",
    "ScoredTag",
    "TagExtractionResult",
    "TagFormatter",
    "TagNormalizer",
    "TagSet",
    "TaggingPipeline",
    "TrainingPreprocessingResult",
    "VisionResponse",
    "normalize_prompt",
    "parse_scene",
]

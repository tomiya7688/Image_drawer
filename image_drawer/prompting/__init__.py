"""画像だけからの自動Caption生成とVLMフォールバック。"""

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

__all__ = [
    "ExtractionAttempt",
    "ExtractionResult",
    "ImagePromptPipeline",
    "OpenAICompatibleVisionProvider",
    "PromptExtractionError",
    "SceneDescription",
    "VisionResponse",
    "normalize_prompt",
    "parse_scene",
]

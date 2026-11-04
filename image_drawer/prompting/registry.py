"""ユーザー登録可能な画像前処理モデル・タグ分類・整形器のレジストリ。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .core import (
    ExtractionResult,
    ImagePromptPipeline,
    PromptExtractionError,
    SceneDescription,
    VisionProvider,
    _detect_mime,
)
from .openai_compatible import OpenAICompatibleVisionProvider
from .tagger_http import HTTPTagProvider
from .tagging import (
    PreparedTags,
    TagExtractionResult,
    TagFormatter,
    TagNormalizer,
    TagProvider,
    TaggingPipeline,
)

PromptFormatter = Callable[[SceneDescription, PreparedTags, int], str]
ModelFactory = Callable[[dict], object]


@dataclass(frozen=True)
class TrainingPreprocessingResult:
    """CaptionとTagを別々に抽出した結果。ユーザーが用意するのは画像のみ。"""

    caption: ExtractionResult
    tags: TagExtractionResult
    image_sha256: str
    prompt_formatter: PromptFormatter | None = None

    def training_prompt(self, seed: int) -> str:
        if self.prompt_formatter is None:
            return self.caption.training_prompt(seed)
        result = self.prompt_formatter(self.caption.scene, self.tags.prepared, seed)
        if not isinstance(result, str) or not result.strip():
            raise PromptExtractionError("empty", "ユーザー登録のPromptFormatterが空を返した")
        return result.strip()

    @property
    def formatted_tags(self) -> tuple[str, ...]:
        return self.tags.prepared.positive_tags

    @property
    def tag_categories(self) -> dict[str, tuple[str, ...]]:
        return self.tags.prepared.category_tags


class ImageTrainingPreprocessor:
    """同じ画像をcaption専用とTagger専用へ独立して渡す。"""

    def __init__(
        self,
        caption_pipeline: ImagePromptPipeline,
        tag_pipeline: TaggingPipeline,
        *,
        prompt_formatter: PromptFormatter | None = None,
    ) -> None:
        self.caption_pipeline = caption_pipeline
        self.tag_pipeline = tag_pipeline
        self.prompt_formatter = prompt_formatter

    def preprocess(self, image: bytes, mime_type: str) -> TrainingPreprocessingResult:
        if not image or mime_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("PNG/JPEG/WebP画像のみ対応")
        # どちらか失敗しても他方の実行を試み、どの工程が失敗したか保存する。
        caption: ExtractionResult | None = None
        tags: TagExtractionResult | None = None
        failed: list[str] = []
        try:
            caption = self.caption_pipeline.extract(image, mime_type)
        except PromptExtractionError as exc:
            failed.append(f"caption:{exc.kind}:{exc}")
        try:
            tags = self.tag_pipeline.extract(image, mime_type)
        except PromptExtractionError as exc:
            failed.append(f"tags:{exc.kind}:{exc}")
        if failed:
            raise PromptExtractionError("preprocessing_failed", "; ".join(failed))
        assert caption is not None and tags is not None
        return TrainingPreprocessingResult(
            caption=caption,
            tags=tags,
            image_sha256=hashlib.sha256(image).hexdigest(),
            prompt_formatter=self.prompt_formatter,
        )

    def preprocess_file(self, path: str | Path) -> TrainingPreprocessingResult:
        """追加のTXT等を要求しない、画像1ファイルだけの入口。"""
        file = Path(path)
        if file.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("画像ファイルが32MiBを超えています")
        image = file.read_bytes()
        return self.preprocess(image, _detect_mime(image))


class PreprocessorRegistry:
    """任意個のCaption/Tagger/Verifier/整形器/アダプタを登録する。

    アプリ設定JSONからはエンドポイントを指定する。独自Pythonアダプタは
    信頼できるプラグイン側がregister_adapterで明示登録する。
    ユーザー設定ファイルをimport/evalすることはしない。
    """

    def __init__(self) -> None:
        self.caption_models: dict[str, VisionProvider] = {}
        self.tag_models: dict[str, TagProvider] = {}
        self.verifiers: dict[str, VisionProvider] = {}
        self.normalizers: dict[str, TagNormalizer] = {"default": TagNormalizer()}
        self.formatters: dict[str, TagFormatter] = {"default": TagFormatter()}
        self.prompt_formatters: dict[str, PromptFormatter] = {}
        self.caption_chain: tuple[str, ...] = ()
        self.tag_chain: tuple[str, ...] = ()
        self.verifier_id: str | None = None
        self.normalizer_id = "default"
        self.formatter_id = "default"
        self.prompt_formatter_id: str | None = None
        self.adapters: dict[str, tuple[str, ModelFactory]] = {}
        self.register_adapter("openai_compatible", "caption", self._vision_adapter)
        self.register_adapter("tagger_http", "tagger", self._tagger_adapter)

    @staticmethod
    def _ensure_new(target: dict, name: str) -> None:
        if not isinstance(name, str) or not name.strip() or name in target:
            raise ValueError("登録IDが空か重複しています")

    def register_caption(self, name: str, model: VisionProvider) -> None:
        self._ensure_new(self.caption_models, name)
        self.caption_models[name] = model

    def register_tagger(self, name: str, model: TagProvider) -> None:
        self._ensure_new(self.tag_models, name)
        self.tag_models[name] = model

    def register_verifier(self, name: str, model: VisionProvider) -> None:
        self._ensure_new(self.verifiers, name)
        self.verifiers[name] = model

    def register_normalizer(self, name: str, normalizer: TagNormalizer) -> None:
        self._ensure_new(self.normalizers, name)
        self.normalizers[name] = normalizer

    def register_tag_formatter(self, name: str, formatter: TagFormatter) -> None:
        self._ensure_new(self.formatters, name)
        self.formatters[name] = formatter

    def register_prompt_formatter(self, name: str, formatter: PromptFormatter) -> None:
        self._ensure_new(self.prompt_formatters, name)
        self.prompt_formatters[name] = formatter

    def register_adapter(self, adapter: str, kind: str, factory: ModelFactory) -> None:
        if kind not in {"caption", "tagger", "verifier"}:
            raise ValueError("adapter kindはcaption/tagger/verifier")
        self._ensure_new(self.adapters, adapter)
        self.adapters[adapter] = (kind, factory)

    @staticmethod
    def _vision_adapter(entry: dict) -> OpenAICompatibleVisionProvider:
        env_name = entry.get("api_key_env", "")
        if not isinstance(env_name, str):
            raise ValueError("api_key_envは環境変数名を指定する")
        return OpenAICompatibleVisionProvider(
            provider_id=entry["id"],
            model_id=entry["model_id"],
            endpoint=entry["endpoint"],
            api_key=os.environ.get(env_name, "") if env_name else "",
            model_revision=entry.get("model_revision", ""),
            timeout_seconds=entry.get("timeout_seconds", 120),
            json_mode=entry.get("json_mode", False),
        )

    @staticmethod
    def _tagger_adapter(entry: dict) -> HTTPTagProvider:
        return HTTPTagProvider(
            provider_id=entry["id"],
            model_id=entry["model_id"],
            endpoint=entry["endpoint"],
            backend=entry.get("backend", "tagger_service_http"),
            threshold=entry.get("threshold", 0.35),
            character_threshold=entry.get("character_threshold", 0.85),
            timeout_seconds=entry.get("timeout_seconds", 120),
        )

    def load_settings(self, settings: Mapping) -> None:
        """ユーザー設定。画像ごとの補助ファイルではない。"""
        if settings.get("schema_version") != "idraw-preprocessor-registry/v1":
            raise ValueError("未知のユーザー前処理設定バージョン")
        providers = settings.get("providers", [])
        if not isinstance(providers, list) or len(providers) > 256:
            raise ValueError("providersは256件以内の配列")
        for entry in providers:
            if not isinstance(entry, dict):
                raise ValueError("provider設定はobject")
            if "api_key" in entry:
                raise ValueError("APIキーの直接記載は禁止。api_key_envを使用してください")
            adapter = entry.get("adapter")
            if adapter not in self.adapters:
                raise ValueError(f"未登録のadapter: {adapter}")
            kind, factory = self.adapters[adapter]
            configured_kind = entry.get("kind", kind)
            if configured_kind != kind:
                raise ValueError("adapterとkindが一致しない")
            provider = factory(entry)
            if kind == "caption":
                self.register_caption(entry["id"], provider)
            elif kind == "tagger":
                self.register_tagger(entry["id"], provider)
            else:
                self.register_verifier(entry["id"], provider)

        normalizer = settings.get("tag_normalizer", {})
        if not isinstance(normalizer, dict):
            raise ValueError("tag_normalizer設定が不正")
        self.register_normalizer(
            "user",
            TagNormalizer(
                min_confidence=normalizer.get("min_confidence", 0.35),
                character_confidence=normalizer.get("character_confidence", 0.85),
                category_overrides=normalizer.get("category_overrides", {}),
            ),
        )
        self.normalizer_id = "user"
        formatter = settings.get("tag_formatter", {})
        if not isinstance(formatter, dict):
            raise ValueError("tag_formatter設定が不正")
        self.register_tag_formatter(
            "user",
            TagFormatter(
                include_style=formatter.get("include_style", False),
                include_copyright=formatter.get("include_copyright", False),
                include_rating=formatter.get("include_rating", False),
                exclude_quality_fillers=formatter.get("exclude_quality_fillers", True),
            ),
        )
        self.formatter_id = "user"
        self.configure(
            caption_chain=settings.get("caption_chain", []),
            tag_chain=settings.get("tag_chain", []),
            verifier_id=settings.get("verifier_id"),
        )

    def configure(
        self,
        *,
        caption_chain: list[str] | tuple[str, ...],
        tag_chain: list[str] | tuple[str, ...],
        verifier_id: str | None = None,
        normalizer_id: str | None = None,
        tag_formatter_id: str | None = None,
        prompt_formatter_id: str | None = None,
    ) -> None:
        for chain, available, label in (
            (caption_chain, self.caption_models, "caption"),
            (tag_chain, self.tag_models, "tagger"),
        ):
            if not chain or len(chain) > 256 or len(set(chain)) != len(chain):
                raise ValueError(f"{label} chainは1〜256件の重複なしの順序付きID")
            if any(name not in available for name in chain):
                raise ValueError(f"{label} chainに未登録のモデルが含まれる")
        if verifier_id is not None and verifier_id not in self.verifiers:
            raise ValueError("Verifierが未登録")
        if normalizer_id is not None:
            if normalizer_id not in self.normalizers:
                raise ValueError("normalizerが未登録")
            self.normalizer_id = normalizer_id
        if tag_formatter_id is not None:
            if tag_formatter_id not in self.formatters:
                raise ValueError("tag formatterが未登録")
            self.formatter_id = tag_formatter_id
        if prompt_formatter_id is not None:
            if prompt_formatter_id not in self.prompt_formatters:
                raise ValueError("prompt formatterが未登録")
            self.prompt_formatter_id = prompt_formatter_id
        self.caption_chain = tuple(caption_chain)
        self.tag_chain = tuple(tag_chain)
        self.verifier_id = verifier_id

    def build(self) -> ImageTrainingPreprocessor:
        if not self.caption_chain or not self.tag_chain:
            raise ValueError("CaptionとTaggerの両方にモデルchainが必要")
        caption = ImagePromptPipeline(
            providers=[self.caption_models[name] for name in self.caption_chain],
            verifier=self.verifiers.get(self.verifier_id) if self.verifier_id else None,
        )
        tagger = TaggingPipeline(
            [self.tag_models[name] for name in self.tag_chain],
            normalizer=self.normalizers[self.normalizer_id],
            formatter=self.formatters[self.formatter_id],
        )
        prompt_formatter = (
            self.prompt_formatters[self.prompt_formatter_id]
            if self.prompt_formatter_id is not None
            else None
        )
        return ImageTrainingPreprocessor(caption, tagger, prompt_formatter=prompt_formatter)

    @classmethod
    def from_settings_file(cls, path: str | Path) -> PreprocessorRegistry:
        """固定モデルを焼き込まず、任意のユーザーモデル設定を読み込む。"""
        registry = cls()
        settings = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(settings, dict):
            raise ValueError("設定JSONはobject")
        registry.load_settings(settings)
        return registry

"""ComfyUI Support Toolsの区分を参考にした独立Tag抽出・分類・整形。

人間が用意するのは画像のみ。Rating・Copyright・Styleを説明文に勝手に混ぜない。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from .core import ExtractionAttempt, PromptExtractionError, _detect_mime


@dataclass(frozen=True)
class ScoredTag:
    label: str
    confidence: float
    category: str
    original_category: str


@dataclass(frozen=True)
class TagSet:
    tags: tuple[ScoredTag, ...]
    caption: str = ""
    rating: str = ""

    def grouped(self) -> dict[str, tuple[ScoredTag, ...]]:
        groups: dict[str, list[ScoredTag]] = {}
        for tag in self.tags:
            groups.setdefault(tag.category, []).append(tag)
        return {key: tuple(values) for key, values in groups.items()}


@dataclass(frozen=True)
class PreparedTags:
    """安全に分離・整形したtag結果。文章のcaptionとは別成果物。"""

    normalized: TagSet
    positive_tags: tuple[str, ...]
    category_tags: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class TagExtractionResult:
    prepared: PreparedTags
    model_id: str
    attempts: tuple[ExtractionAttempt, ...]
    fallback_used: bool
    cache_key: str


class TagProvider(Protocol):
    provider_id: str
    model_id: str

    def tag(self, image: bytes, mime_type: str) -> object:
        """Tagger独立の推論入口。VLM captionとは別に必ず実行する。"""


def _label(value: object) -> str:
    if not isinstance(value, str):
        raise PromptExtractionError("invalid_format", "タグは文字列")
    text = re.sub(r"\s+", " ", value.replace("_", " ")).strip()
    if not text or len(text) > 256 or any(ord(c) < 32 for c in text):
        raise PromptExtractionError("invalid_format", "タグが空または長すぎる")
    return text


def _score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PromptExtractionError("invalid_format", "タグconfidenceは数値")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise PromptExtractionError("invalid_format", "タグconfidenceは0..1")
    return result


def _entries(value: object) -> list[tuple[str, float]]:
    if value is None:
        return []
    if isinstance(value, str):
        raw: list[tuple[object, object]] = [
            (s, 1.0) for s in value.split(",") if s.strip()
        ]
    elif isinstance(value, dict):
        raw = list(value.items())
    elif isinstance(value, list):
        raw = []
        for item in value:
            if isinstance(item, str):
                raw.append((item, 1.0))
            elif isinstance(item, dict):
                raw.append(
                    (
                        item.get("name", item.get("tag")),
                        item.get("confidence", item.get("score", 1.0)),
                    )
                )
            else:
                raise PromptExtractionError("invalid_format", "タグ項目が不正")
    else:
        raise PromptExtractionError("invalid_format", "Tagger応答の型が不正")
    if len(raw) > 2000:
        raise PromptExtractionError("invalid_format", "タグ上限2000件")
    deduped: dict[str, tuple[str, float]] = {}
    for name, confidence in raw:
        text = _label(name)
        score = _score(confidence)
        key = text.casefold()
        if key not in deduped or score > deduped[key][1]:
            deduped[key] = (text, score)
    return list(deduped.values())


# 用語の完全一致を基本とする。単純な文字列部分一致による誤分類を避ける。
_SEMANTIC_HINTS = {
    "clothes": ("dress", "shirt", "skirt", "pants", "uniform", "hoodie", "coat",
                "socks", "shoes", "swimsuit", "bra", "underwear", "ジャケット", "服", "シャツ"),
    "pose": ("standing", "sitting", "lying", "kneeling", "walking", "running",
             "full body", "upper body", "cowboy shot", "arms raised", "立ち", "座り"),
    "expression": ("smile", "laugh", "crying", "angry", "sad", "blush", "wink",
                   "closed eyes", "open mouth", "笑顔", "笑い", "泣き"),
    "background": ("sky", "forest", "beach", "room", "street", "background",
                   "mountain", "city", "bedroom", "背景", "空", "森"),
    "situation": ("night", "sunset", "sunrise", "rain", "snow", "festival",
                  "lighting", "nighttime", "夕焼け", "夜", "雨", "雪"),
    "style": ("anime style", "watercolor", "oil painting", "lineart",
              "sketch", "photorealistic", "pixel art", "水彩", "線画"),
}
_ALIAS = {
    "general_tags": "general",
    "content": "general",
    "content_tags": "general",
    "characters": "character",
    "character_tags": "character",
    "copyrights": "copyright",
    "copyright_tags": "copyright",
    "ratings": "rating",
    "styles": "style",
    "image_style": "style",
    "poses": "pose",
    "clothing": "clothes",
    "expressions": "expression",
    "backgrounds": "background",
    "situations": "situation",
}
_SUPPORTED = {
    "general", "character", "copyright", "rating", "style", "pose",
    "clothes", "expression", "background", "situation",
}
_QUALITY_FILLERS = {
    "masterpiece", "best quality", "high quality", "worst quality",
    "low quality", "8k", "4k", "rating", "quality",
}


class TagNormalizer:
    """ユーザーがcategory overrides/信頼度を登録できる決定的な分類器。"""

    def __init__(
        self,
        *,
        min_confidence: float = 0.35,
        character_confidence: float = 0.85,
        category_overrides: Mapping[str, str] | None = None,
    ) -> None:
        self.min_confidence = _score(min_confidence)
        self.character_confidence = _score(character_confidence)
        self.overrides = {
            _label(key).casefold(): _category(value)
            for key, value in (category_overrides or {}).items()
        }

    @property
    def config_fingerprint(self) -> str:
        data = json.dumps(
            {
                "min": self.min_confidence,
                "character": self.character_confidence,
                "overrides": self.overrides,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
        return hashlib.sha256(data).hexdigest()

    def classify(self, label: str, origin: str) -> str:
        name = label.casefold()
        if name in self.overrides:
            return self.overrides[name]
        if origin != "general":
            return origin
        for category, hints in _SEMANTIC_HINTS.items():
            if any(
                name == hint or name.endswith(" " + hint) or name.startswith(hint + " ")
                for hint in hints
            ):
                return category
        return "general"

    def normalize(self, payload: object) -> TagSet:
        for _ in range(4):
            if isinstance(payload, dict) and len(payload) == 1:
                if "result" in payload:
                    payload = payload["result"]
                    continue
                if "data" in payload:
                    payload = payload["data"]
                    continue
            break

        caption = ""
        if isinstance(payload, dict):
            if payload.get("error"):
                raise PromptExtractionError("refused", "Taggerがerrorを返した")
            raw_caption = payload.get("caption", "")
            if not isinstance(raw_caption, str) or len(raw_caption) > 8192:
                raise PromptExtractionError("invalid_format", "Tagger captionが不正")
            caption = raw_caption.strip()
            raw = payload
            for field in ("tags", "tag"):
                if field in payload and isinstance(payload[field], dict):
                    nested = payload[field]
                    if any(_category(k, strict=False) in _SUPPORTED for k in nested):
                        raw = nested
                        break
        else:
            raw = payload

        if isinstance(raw, dict):
            nodes: dict[str, object] = {}
            has_explicit = False
            for name, value in raw.items():
                category = _category(name, strict=False)
                if category in _SUPPORTED:
                    if category in nodes:
                        raise PromptExtractionError("invalid_format", "カテゴリが重複")
                    nodes[category] = value
                    has_explicit = True
            if has_explicit:
                if "general" not in nodes and "tags" in raw:
                    nodes["general"] = raw["tags"]
                if "general" not in nodes and "tag" in raw:
                    nodes["general"] = raw["tag"]
            elif "tags" in raw or "tag" in raw:
                nodes = {"general": raw.get("tags", raw.get("tag"))}
            elif raw and all(isinstance(v, (int, float)) for v in raw.values()):
                nodes = {"general": raw}
            else:
                raise PromptExtractionError("invalid_format", "Tagger応答にtagsがない")
        else:
            nodes = {"general": raw}

        tags: list[ScoredTag] = []
        rating = ""
        for original, entries in nodes.items():
            records = _entries(entries)
            if original == "rating":
                rating = max(records, key=lambda item: item[1])[0] if records else ""
                continue
            limit = self.character_confidence if original == "character" else self.min_confidence
            for label, confidence in records:
                if confidence < limit:
                    continue
                tags.append(
                    ScoredTag(label, confidence, self.classify(label, original), original)
                )
        if not tags:
            raise PromptExtractionError("empty", "分類可能なタグが得られなかった")
        return TagSet(tuple(tags), caption=caption, rating=rating)


def _category(value: object, *, strict: bool = True) -> str:
    if not isinstance(value, str):
        if strict:
            raise ValueError("タグカテゴリは文字列が必要")
        return ""
    name = value.casefold().strip().replace("-", "_")
    category = _ALIAS.get(name, name)
    if strict and category not in _SUPPORTED:
        raise ValueError("未対応のタグカテゴリ: " + value)
    return category


class TagFormatter:
    """タグを用途別に整形。RatingやCopyrightを勝手にPromptへ混ぜない。"""

    def __init__(
        self,
        *,
        include_style: bool = False,
        include_copyright: bool = False,
        include_rating: bool = False,
        exclude_quality_fillers: bool = True,
    ) -> None:
        self.include_style = include_style
        self.include_copyright = include_copyright
        self.include_rating = include_rating
        self.exclude_quality_fillers = exclude_quality_fillers

    @property
    def config_fingerprint(self) -> str:
        return f"{self.include_style}:{self.include_copyright}:{self.include_rating}:{self.exclude_quality_fillers}"

    def format(self, tags: TagSet) -> PreparedTags:
        grouped: dict[str, list[str]] = {}
        positive: list[str] = []
        seen: set[str] = set()
        for tag in tags.tags:
            if self.exclude_quality_fillers and tag.label.casefold() in _QUALITY_FILLERS:
                continue
            grouped.setdefault(tag.category, []).append(tag.label)
            if tag.category == "copyright" and not self.include_copyright:
                continue
            if tag.category == "style" and not self.include_style:
                continue
            if tag.label.casefold() not in seen:
                seen.add(tag.label.casefold())
                positive.append(tag.label)
        if self.include_rating and tags.rating:
            grouped["rating"] = [tags.rating]
            positive.append(tags.rating)
        return PreparedTags(
            normalized=tags,
            positive_tags=tuple(positive),
            category_tags={key: tuple(items) for key, items in grouped.items()},
        )


class TaggingPipeline:
    """独立Taggerの順序付きN段フォールバック。Caption結果を流用しない。"""

    def __init__(
        self,
        providers: Sequence[TagProvider],
        *,
        normalizer: TagNormalizer | None = None,
        formatter: TagFormatter | None = None,
    ) -> None:
        self.providers = tuple(providers)
        if not self.providers:
            raise ValueError("少なくとも1つのTaggerが必要")
        ids = [(provider.provider_id, provider.model_id) for provider in self.providers]
        if len(set(ids)) != len(ids):
            raise ValueError("Taggerの重複登録はできない")
        self.normalizer = normalizer or TagNormalizer()
        self.formatter = formatter or TagFormatter()

    def cache_key(self, image: bytes) -> str:
        payload = {
            "image": hashlib.sha256(image).hexdigest(),
            "providers": [
                [
                    model.provider_id,
                    model.model_id,
                    getattr(model, "config_fingerprint", ""),
                ]
                for model in self.providers
            ],
            "normalizer": self.normalizer.config_fingerprint,
            "formatter": self.formatter.config_fingerprint,
            "tag_schema_version": "idraw-tag-normalizer/v1",
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest()

    def extract(self, image: bytes, mime_type: str) -> TagExtractionResult:
        if not image or mime_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("PNG/JPEG/WebP画像が必要")
        attempts: list[ExtractionAttempt] = []
        for index, provider in enumerate(self.providers):
            try:
                payload = provider.tag(image, mime_type)
                prepared = self.formatter.format(self.normalizer.normalize(payload))
                attempts.append(ExtractionAttempt(provider.provider_id, provider.model_id, "ok"))
                return TagExtractionResult(
                    prepared=prepared,
                    model_id=provider.model_id,
                    attempts=tuple(attempts),
                    fallback_used=index > 0,
                    cache_key=self.cache_key(image),
                )
            except PromptExtractionError as exc:
                attempts.append(ExtractionAttempt(provider.provider_id, provider.model_id, exc.kind))
                if exc.kind == "invalid_input":
                    break
            except (TimeoutError, OSError):
                attempts.append(ExtractionAttempt(provider.provider_id, provider.model_id, "unavailable"))
        report = ",".join(f"{a.provider_id}:{a.status}" for a in attempts)
        raise PromptExtractionError("all_taggers_failed", report)

    def extract_image_file(self, image_path: str | Path) -> TagExtractionResult:
        path = Path(image_path)
        if path.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("画像ファイルが32MiBを超えています")
        image = path.read_bytes()
        return self.extract(image, _detect_mime(image))

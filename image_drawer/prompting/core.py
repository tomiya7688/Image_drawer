"""画像のみを入力に、人が書く短い文章にも整合する学習プロンプトを作る。"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass
from typing import Protocol


class PromptExtractionError(RuntimeError):
    """検出・ルーティング用に機械判定可能な失敗理由を保持する。"""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@dataclass(frozen=True)
class VisionResponse:
    content: str = ""
    refusal: bool = False
    finish_reason: str = ""


class VisionProvider(Protocol):
    provider_id: str
    model_id: str

    def describe(self, image: bytes, mime_type: str, instruction: str) -> VisionResponse:
        """画像を使った文章生成。API・ローカルQwen共通の入口。"""


@dataclass(frozen=True)
class Subject:
    name: str
    count: int | None
    attributes: tuple[str, ...]


@dataclass(frozen=True)
class SceneDescription:
    short: str
    standard: str
    detailed: str
    summary: str
    subjects: tuple[Subject, ...]
    relations: tuple[str, ...]
    tags: tuple[str, ...]
    style: str
    composition: str
    background: str


@dataclass(frozen=True)
class ExtractionAttempt:
    provider_id: str
    model_id: str
    status: str


@dataclass(frozen=True)
class ExtractionResult:
    scene: SceneDescription
    selected_model: str
    attempts: tuple[ExtractionAttempt, ...]
    fallback_used: bool
    verifier_status: str
    cache_key: str

    def training_prompt(self, seed: int) -> str:
        """長いキャプションだけに過学習せず、短文も教材に含める。"""
        return random.Random(seed).choices(
            [self.scene.short, self.scene.standard, self.scene.detailed],
            weights=[40, 40, 20],
            k=1,
        )[0]


CAPTION_INSTRUCTION = """画像の内容を日本語の画像生成プロンプトとして正確に記述してください。
必ず次のJSONオブジェクトのみを出力してください。
{
  "summary": "画像に写る主要な対象と場面",
  "subjects": [{"name": "対象", "count": 1, "attributes": ["見える属性"]}],
  "relations": ["AがBの左にある"],
  "background": "背景",
  "composition": "構図・視点",
  "style": "画風・質感",
  "tags": ["画像から分かる内容タグ"],
  "short_prompt": "人間が短く入力する自然なプロンプト",
  "standard_prompt": "色、数、位置関係を含む自然なプロンプト",
  "detailed_prompt": "構図、背景、画風を含む詳しいプロンプト"
}
対象・数・色・位置関係・画風に注意して、画像にない特徴を捏造しない。
読めない部分は推測せずcountにnull、説明不明な項目に空文字を使える。
short/standard/detailedは同じ画像の説明粒度違い。人間らしい日本語で書く。
汎用の品質語、負のプロンプト、作者の推測、ratingを説明文に混入させない。
"""

VERIFY_INSTRUCTION = """入力画像と候補の文章が一致するか、次のJSONのみ返してください。
{"supported":true,"unsupported_claims":[],"missing_facts":[]}
supported=falseは対象、色、数、位置関係など重要な不一致がある場合。
候補JSON:
"""

_REFUSALS = (
    "i cannot assist",
    "i can't assist",
    "i cannot help",
    "i can't help",
    "i'm sorry, but i can't",
    "sorry, i can't",
    "申し訳ありませんが",
    "この画像には対応できません",
    "この画像を説明できません",
)


def normalize_prompt(prompt: str) -> str:
    """生成時も同じ軽量正規化を使う。否定や修飾を削除しない。"""
    if not isinstance(prompt, str):
        raise TypeError("promptは文字列で指定する")
    result = re.sub(r"\s+", " ", prompt).strip()
    if not result:
        raise ValueError("空のpromptは使用できない")
    return result


def _text(obj: object, name: str, optional: bool = False) -> str:
    if obj is None and optional:
        return ""
    if not isinstance(obj, str):
        raise PromptExtractionError("invalid_format", f"{name}は文字列が必要")
    value = re.sub(r"\s+", " ", obj).strip()
    if len(value) > 1500:
        raise PromptExtractionError("invalid_format", f"{name}が長すぎる")
    if not value and not optional:
        raise PromptExtractionError("empty", f"{name}が空")
    return value


def _list(obj: object, name: str) -> tuple[str, ...]:
    if obj is None:
        return ()
    if not isinstance(obj, list) or len(obj) > 64:
        raise PromptExtractionError("invalid_format", f"{name}の配列形式が不正")
    result: list[str] = []
    seen: set[str] = set()
    for item in obj:
        value = _text(item, name, optional=True)
        if value and value.casefold() not in seen:
            result.append(value)
            seen.add(value.casefold())
    return tuple(result)


def _json(text: str) -> dict:
    value = text.strip()
    if not value:
        raise PromptExtractionError("empty", "VLM出力が空")
    if any(value.casefold().startswith(phrase) for phrase in _REFUSALS):
        raise PromptExtractionError("refused", "VLMが画像説明を拒否")
    fence = chr(96) * 3
    if value.startswith(fence):
        match = re.fullmatch(
            re.escape(fence) + r"(?:json)?\s*\n(.*?)\n" + re.escape(fence),
            value,
            re.DOTALL,
        )
        if match is None:
            raise PromptExtractionError("invalid_format", "JSON以外のMarkdown")
        value = match.group(1)
    try:
        result = json.loads(value)
    except (ValueError, TypeError) as exc:
        raise PromptExtractionError("invalid_format", "出力をJSONとして読めない") from exc
    if not isinstance(result, dict):
        raise PromptExtractionError("invalid_format", "JSON objectを期待した")
    return result


def parse_scene(response: VisionResponse) -> SceneDescription:
    """拒否・不正構造・画像内容のないテンプレート文を拒絶する。"""
    if response.refusal or response.finish_reason == "content_filter":
        raise PromptExtractionError("refused", "モデルが拒否した")
    data = _json(response.content)
    summary = _text(data.get("summary"), "summary")
    short = _text(data.get("short_prompt", summary), "short_prompt")
    standard = _text(data.get("standard_prompt"), "standard_prompt")
    detailed = _text(data.get("detailed_prompt", standard), "detailed_prompt")
    if standard.casefold() in {"an image", "a picture", "画像", "イラスト", "picture"}:
        raise PromptExtractionError("invalid_format", "内容を説明していない")
    raw_subjects = data.get("subjects", [])
    if not isinstance(raw_subjects, list) or len(raw_subjects) > 32:
        raise PromptExtractionError("invalid_format", "subjectsの形式が不正")
    subjects: list[Subject] = []
    for item in raw_subjects:
        if not isinstance(item, dict):
            raise PromptExtractionError("invalid_format", "subjectの形式が不正")
        count = item.get("count")
        if count is not None and (type(count) is not int or not 1 <= count <= 100):
            raise PromptExtractionError("invalid_format", "countは1..100またはnull")
        attributes = _list(item.get("attributes", []), "attributes")
        subjects.append(Subject(_text(item.get("name"), "subject.name"), count, attributes))
    return SceneDescription(
        short=short,
        standard=standard,
        detailed=detailed,
        summary=summary,
        subjects=tuple(subjects),
        relations=_list(data.get("relations", []), "relations"),
        tags=_list(data.get("tags", []), "tags"),
        style=_text(data.get("style", ""), "style", optional=True),
        composition=_text(data.get("composition", ""), "composition", optional=True),
        background=_text(data.get("background", ""), "background", optional=True),
    )


def verify_scene(
    verifier: VisionProvider,
    image: bytes,
    mime_type: str,
    scene: SceneDescription,
) -> None:
    proposal = json.dumps(
        {
            "summary": scene.summary,
            "subjects": [
                {"name": s.name, "count": s.count, "attributes": s.attributes}
                for s in scene.subjects
            ],
            "relations": scene.relations,
            "prompts": [scene.short, scene.standard, scene.detailed],
        },
        ensure_ascii=False,
    )
    reply = verifier.describe(image, mime_type, VERIFY_INSTRUCTION + proposal)
    if reply.refusal or reply.finish_reason == "content_filter":
        raise PromptExtractionError("verifier_refused", "検証モデルが拒否した")
    result = _json(reply.content)
    if result.get("supported") is not True or result.get("unsupported_claims"):
        raise PromptExtractionError("unverified_facts", "画像と候補の内容が一致しない")


class ImagePromptPipeline:
    """精度優先のモデルを試し、拒否・出力不備時は指定Qwen等へ自動切替。"""

    def __init__(
        self,
        primary: VisionProvider,
        fallback: VisionProvider,
        *,
        verifier: VisionProvider | None = None,
        profile_id: str = "human-prompt-aligned-v1",
        fallback_on_unavailable: bool = True,
    ) -> None:
        if (primary.provider_id, primary.model_id) == (
            fallback.provider_id,
            fallback.model_id,
        ):
            raise ValueError("primaryとfallbackは異なるモデルが必要")
        self.primary = primary
        self.fallback = fallback
        self.verifier = verifier
        self.profile_id = profile_id
        self.fallback_on_unavailable = fallback_on_unavailable

    def _cache_key(self, image: bytes, mime_type: str) -> str:
        settings = {
            "image": hashlib.sha256(image).hexdigest(),
            "mime": mime_type,
            "profile": self.profile_id,
            "caption_instruction_sha256": hashlib.sha256(
                CAPTION_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
            "verifier_instruction_sha256": hashlib.sha256(
                VERIFY_INSTRUCTION.encode("utf-8")
            ).hexdigest(),
            "primary": [
                self.primary.provider_id,
                self.primary.model_id,
                getattr(self.primary, "config_fingerprint", ""),
            ],
            "fallback": [
                self.fallback.provider_id,
                self.fallback.model_id,
                getattr(self.fallback, "config_fingerprint", ""),
            ],
            "verifier": (
                [
                    self.verifier.provider_id,
                    self.verifier.model_id,
                    getattr(self.verifier, "config_fingerprint", ""),
                ]
                if self.verifier
                else None
            ),
        }
        canonical = json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    def extract(self, image: bytes, mime_type: str) -> ExtractionResult:
        if not isinstance(image, bytes) or not image:
            raise ValueError("画像が空")
        if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise ValueError("PNG/JPEG/WebPのみ対応")
        attempts: list[ExtractionAttempt] = []
        for index, model in enumerate((self.primary, self.fallback)):
            try:
                scene = parse_scene(model.describe(image, mime_type, CAPTION_INSTRUCTION))
                verification = "not_run"
                if self.verifier:
                    verify_scene(self.verifier, image, mime_type, scene)
                    verification = "verified_by_model"
                attempts.append(ExtractionAttempt(model.provider_id, model.model_id, "ok"))
                return ExtractionResult(
                    scene=scene,
                    selected_model=model.model_id,
                    attempts=tuple(attempts),
                    fallback_used=index == 1,
                    verifier_status=verification,
                    cache_key=self._cache_key(image, mime_type),
                )
            except PromptExtractionError as exc:
                attempts.append(ExtractionAttempt(model.provider_id, model.model_id, exc.kind))
                if index == 1 or exc.kind == "invalid_input":
                    break
                if exc.kind == "unavailable" and not self.fallback_on_unavailable:
                    break
        cause = ",".join(f"{a.provider_id}:{a.status}" for a in attempts)
        raise PromptExtractionError("all_models_failed", cause)

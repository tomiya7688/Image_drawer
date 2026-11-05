"""image_drawer.promptingの抽出Pipelineを、DatasetのCaption/Tag抽出器として包むアダプタ。

Datasetが永続化の所有者であり、promptingは抽出ロジックの提供者にとどまる。
promptingはDatasetのmanifestやsplitを知らず、bytesとMIMEを受け取って結果を返す。
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any

from image_drawer.dataset.models import ExtractorInfo
from image_drawer.dataset.registry import ExtractorRegistry
from image_drawer.prompting import core as prompting_core
from image_drawer.prompting.core import ExtractionResult, ImagePromptPipeline, PromptExtractionError
from image_drawer.prompting.registry import PreprocessorRegistry
from image_drawer.prompting.tagging import TaggingPipeline, TagExtractionResult, display_label

ADAPTER_VERSION = "prompting-adapter-v1"
PNG_MAX_BYTES = 8 * 1024 * 1024
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
DEFAULT_PROMPT_CATEGORIES = frozenset(
    {"general", "clothes", "pose", "expression", "background", "situation"}
)


# {
#   責務: [
#     encode_png: 正規化済みのRGB画像を、メタデータを含まない固定設定のPNGのbytesにする
#   ]
#   処理: [
#     画素だけを新しい画像へ写し(ICC・EXIF・dpiなどを持ち越さない)、固定設定で符号化する。
#     上限を超えても縮小せず失敗にする(品質を勝手に下げた画像を送らない)
#   ]
#   引数: [
#     image: PIL画像
#     max_bytes: 許容するPNGの最大バイト数
#   ]
#   戻り値: [
#     data: PNGのbytes
#   ]
#   エラー: [
#     PromptExtractionError: PNGがmax_bytesを超えた(kind=image_too_large)
#   ]
# }
def encode_png(image: Any, max_bytes: int = PNG_MAX_BYTES) -> bytes:
    from PIL import Image

    rgb = image.convert("RGB")
    clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG", compress_level=6)
    data = buffer.getvalue()
    if len(data) > max_bytes:
        raise PromptExtractionError("image_too_large", f"PNGが上限{max_bytes}バイトを超えた")
    return data


def _sha256_json(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _fingerprints(models: list[Any]) -> list[list[str]]:
    return [[m.provider_id, m.model_id, str(getattr(m, "config_fingerprint", ""))] for m in models]


def _endpoint_host(model: Any) -> str:
    from urllib.parse import urlsplit

    return (urlsplit(str(getattr(model, "endpoint", ""))).hostname or "").casefold()


# {
#   責務: [
#     is_external_provider: 画像が外部のホストへ送られうるプロバイダか判定する
#   ]
#   処理: [
#     endpointのホストがlocalhost・127.0.0.1・::1以外、またはdata_egress="external"の宣言があれば外部。
#     endpointを持たない(ネットワークを使わない)プロバイダは外部ではない
#   ]
#   戻り値: [
#     external: 外部ならTrue
#   ]
# }
def is_external_provider(model: Any) -> bool:
    if getattr(model, "data_egress", "local") == "external":
        return True
    host = _endpoint_host(model)
    return bool(host) and host not in LOCAL_HOSTS


def _egress(models: list[Any]) -> dict[str, Any]:
    hosts = sorted({_endpoint_host(m) for m in models if is_external_provider(m)})
    return {"sent": bool(hosts), "hosts": hosts}


def _attempts(attempts: Any) -> list[dict[str, str]]:
    return [
        {"provider_id": a.provider_id, "model_id": a.model_id, "status": a.status} for a in attempts
    ]


def _profile(extractor_id: str) -> str:
    return extractor_id.split(":", 2)[-1]


# {
#   責務: [
#     PromptingCaptionExtractor: ImagePromptPipelineを、DatasetのCaptionExtractorとして包む
#   ]
#   フィールド: [
#     info: ExtractorInfo(capabilities=["caption"])。model_hashは列の全モデル(検証器を含む)の
#       config_fingerprintを順序つきで並べたSHA-256、config_hashは指示文などのSHA-256
#     pipeline: 包んだImagePromptPipeline
#   ]
#   処理: [
#     画像をメタデータなしPNGにして1回だけPipelineを呼び、同じ画像への続けての問い合わせ
#     (extract→extract_variants→extract_with_trace)では結果を再利用する。失敗は再利用しない
#   ]
# }
class PromptingCaptionExtractor:
    # {
    #   責務: [__init__: Pipelineと識別情報から、キャッシュ鍵に反応するExtractorInfoを組み立てる]
    #   引数: [
    #     pipeline: 包むPipeline
    #     extractor_id: "prompting:caption:<profile_id>"
    #     settings_sha256: 設定ファイルのSHA-256
    #   ]
    # }
    def __init__(
        self, pipeline: ImagePromptPipeline, *, extractor_id: str, settings_sha256: str
    ) -> None:
        self.pipeline = pipeline
        self._memo: tuple[str, ExtractionResult] | None = None
        models = list(pipeline.providers)
        verifier = [pipeline.verifier] if pipeline.verifier is not None else []
        external = any(is_external_provider(m) for m in models + verifier)
        self.info = ExtractorInfo(
            id=extractor_id,
            version=f"{ADAPTER_VERSION}:{pipeline.profile_id}",
            capabilities=["caption"],
            model_hash=_sha256_json(
                {"providers": _fingerprints(models), "verifier": _fingerprints(verifier)}
            ),
            config_hash=_sha256_json(
                {
                    "caption_instruction": hashlib.sha256(
                        prompting_core.CAPTION_INSTRUCTION.encode("utf-8")
                    ).hexdigest(),
                    "verify_instruction": hashlib.sha256(
                        prompting_core.VERIFY_INSTRUCTION.encode("utf-8")
                    ).hexdigest(),
                    "fallback_on_unavailable": pipeline.fallback_on_unavailable,
                    "settings_sha256": settings_sha256,
                }
            ),
            license="provider-defined",
            runtime="remote" if external else "local",
        )

    def _run(self, image: Any) -> ExtractionResult:
        png = encode_png(image)
        digest = hashlib.sha256(png).hexdigest()
        if self._memo is not None and self._memo[0] == digest:
            return self._memo[1]
        result = self.pipeline.extract(png, "image/png")
        self._memo = (digest, result)
        return result

    # {
    #   責務: [extract: 標準文(scene.standard)を返す]
    #   引数: [image: 正規化済みのRGB画像]
    #   戻り値: [caption: 標準文]
    #   エラー: [PromptExtractionError: 全モデルが失敗した]
    # }
    def extract(self, image: Any) -> str:
        return self._run(image).scene.standard

    # {
    #   責務: [extract_variants: 短文・標準文・詳細文を返す(任意の拡張)]
    #   戻り値: [variants: {"short", "standard", "detailed"}]
    # }
    def extract_variants(self, image: Any) -> dict[str, str]:
        scene = self._run(image).scene
        return {"short": scene.short, "standard": scene.standard, "detailed": scene.detailed}

    # {
    #   責務: [extract_with_trace: 標準文と、選ばれたモデル・試行・検証・送信の記録を返す(任意の拡張)]
    #   戻り値: [(caption, trace)。traceに鍵や画像、応答本文は含めない]
    # }
    def extract_with_trace(self, image: Any) -> tuple[str, dict[str, Any]]:
        result = self._run(image)
        used = list(self.pipeline.providers)
        if result.verifier_status != "not_run" and self.pipeline.verifier is not None:
            used.append(self.pipeline.verifier)
        trace = {
            "selected_model": result.selected_model,
            "fallback_used": result.fallback_used,
            "attempts": _attempts(result.attempts),
            "verifier_status": result.verifier_status,
            "egress": _egress(used),
        }
        return result.scene.standard, trace


# {
#   責務: [
#     PromptingTagExtractor: TaggingPipelineを、DatasetのTagExtractorとして包む
#   ]
#   フィールド: [
#     info: ExtractorInfo(capabilities=["tag"])
#     include_categories: 学習用promptの文に使うタグのカテゴリ。他のカテゴリもtagsには保存する
#   ]
# }
class PromptingTagExtractor:
    # {
    #   責務: [__init__: Pipelineと識別情報から、キャッシュ鍵に反応するExtractorInfoを組み立てる]
    #   引数: [
    #     pipeline: 包むTaggingPipeline
    #     extractor_id: "prompting:tags:<profile_id>"
    #     settings_sha256: 設定ファイルのSHA-256
    #     include_categories: promptに使うカテゴリ
    #   ]
    # }
    def __init__(
        self,
        pipeline: TaggingPipeline,
        *,
        extractor_id: str,
        settings_sha256: str,
        include_categories: frozenset[str] = DEFAULT_PROMPT_CATEGORIES,
    ) -> None:
        self.pipeline = pipeline
        self.include_categories = frozenset(include_categories)
        self._memo: tuple[str, TagExtractionResult] | None = None
        models = list(pipeline.providers)
        self.info = ExtractorInfo(
            id=extractor_id,
            version=f"{ADAPTER_VERSION}:{_profile(extractor_id)}",
            capabilities=["tag"],
            model_hash=_sha256_json({"providers": _fingerprints(models)}),
            config_hash=_sha256_json(
                {
                    "normalizer": pipeline.normalizer.config_fingerprint,
                    "formatter": pipeline.formatter.config_fingerprint,
                    "include_categories": sorted(self.include_categories),
                    "settings_sha256": settings_sha256,
                }
            ),
            license="provider-defined",
            languages=["en"],
            runtime="remote" if any(is_external_provider(m) for m in models) else "local",
        )

    def _run(self, image: Any) -> TagExtractionResult:
        png = encode_png(image)
        digest = hashlib.sha256(png).hexdigest()
        if self._memo is not None and self._memo[0] == digest:
            return self._memo[1]
        result = self.pipeline.extract(png, "image/png")
        self._memo = (digest, result)
        return result

    def _tags(self, result: TagExtractionResult) -> list[dict[str, Any]]:
        kept = {
            (category, label.casefold())
            for category, labels in result.prepared.category_tags.items()
            for label in labels
        }
        tags = []
        for tag in result.prepared.normalized.tags:
            label = display_label(tag.label)
            if (tag.category, label.casefold()) not in kept:
                continue  # 品質語など、整形器が除いたタグ
            tags.append(
                {
                    "label": label,
                    "confidence": tag.confidence,
                    "category": tag.category,
                    "in_prompt": tag.category in self.include_categories,
                }
            )
        return tags

    # {
    #   責務: [extract: カテゴリつきのタグを返す。promptに使わないカテゴリはin_prompt=falseで保存する]
    #   戻り値: [tags: [{"label", "confidence", "category", "in_prompt"}]]
    #   エラー: [PromptExtractionError: 全Taggerが失敗した]
    # }
    def extract(self, image: Any) -> list[dict[str, Any]]:
        return self._tags(self._run(image))

    # {
    #   責務: [extract_with_trace: タグと、選ばれたTagger・試行・送信の記録を返す(任意の拡張)]
    # }
    def extract_with_trace(self, image: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        result = self._run(image)
        trace = {
            "selected_model": result.model_id,
            "fallback_used": result.fallback_used,
            "attempts": _attempts(result.attempts),
            "rating": result.prepared.normalized.rating,
            "egress": _egress(list(self.pipeline.providers)),
        }
        return self._tags(result), trace


# {
#   責務: [
#     register_prompting_extractors: 設定済みのCaption列とTagger列を、それぞれ1つの抽出器として登録する
#   ]
#   処理: [
#     1: 登録するidを先に作り、既存のidと重複していないか確認する(途中まで登録した状態を残さない)
#     2: chainが設定されている種類だけ、Pipelineを作って抽出器にし、registryへ登録する
#   ]
#   引数: [
#     registry: Datasetの抽出器registry
#     preprocessors: load_settings済みのPreprocessorRegistry
#     settings_sha256: 設定ファイルのSHA-256(config_hashに入る)
#   ]
#   戻り値: [
#     ids: 登録したextractor idの一覧
#   ]
#   エラー: [
#     ValueError: chainが無い、またはidが登録済み
#   ]
# }
def register_prompting_extractors(
    registry: ExtractorRegistry,
    preprocessors: PreprocessorRegistry,
    *,
    settings_sha256: str,
) -> list[str]:
    profile = preprocessors.profile_id
    caption_id, tag_id = f"prompting:caption:{profile}", f"prompting:tags:{profile}"
    wanted = [
        (caption_id, bool(preprocessors.caption_chain)),
        (tag_id, bool(preprocessors.tag_chain)),
    ]
    if not any(enabled for _, enabled in wanted):
        raise ValueError("Caption chainもTagger chainも設定されていません")
    known = {info.id for infos in registry.describe().values() for info in infos}
    if any(enabled and name in known for name, enabled in wanted):
        raise ValueError(f"抽出器が登録済みです: {[n for n, e in wanted if e and n in known]}")
    ids: list[str] = []
    if preprocessors.caption_chain:
        captioner = PromptingCaptionExtractor(
            preprocessors.build_caption_pipeline(),
            extractor_id=caption_id,
            settings_sha256=settings_sha256,
        )
        registry.register_captioner(captioner)
        ids.append(caption_id)
    if preprocessors.tag_chain:
        tagger = PromptingTagExtractor(
            preprocessors.build_tag_pipeline(),
            extractor_id=tag_id,
            settings_sha256=settings_sha256,
        )
        registry.register_tagger(tagger)
        ids.append(tag_id)
    return ids

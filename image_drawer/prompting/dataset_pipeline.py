"""Dataset全体を①タグ→②Caption→③プロンプトの3パスで前処理する。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

from image_drawer.dataset import DatasetManifest, ExtractorInfo, QualityIssue

from .core import PromptExtractionError, SceneDescription, Subject, _detect_mime
from .prompt_composer import COMPOSER_VERSION, PromptComposer
from .registry import ImageTrainingPreprocessor
from .tagging import PreparedTags, ScoredTag, TagSet


# {
#   責務: [DatasetPreprocessingReport: 3段階ごとの件数・エラーを記録する]
#   フィールド: [count, tags_ok, captions_ok, prompts_ok, failures]
# }
@dataclass(frozen=True)
class DatasetPreprocessingReport:
    images: int
    tags_ok: int
    captions_ok: int
    prompts_ok: int
    failures: tuple[tuple[str, str, str], ...]
    dataset_hash: str


def _stage_key(image: bytes, config_hash: str) -> str:
    value = hashlib.sha256(image).hexdigest() + ":" + config_hash
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_stage(path: Path, expected_key: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        digest = data.pop("payload_sha256")
        actual = hashlib.sha256(
            json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if digest != actual or data.get("cache_key") != expected_key:
            return None
        # 一時的な拒否・ネットワーク障害は次回の実行時に再試行させる。
        if data.get("status") != "ok":
            return None
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_stage(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(data)
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def _read_image(dataset_dir: Path, relative_path: str) -> bytes:
    root = dataset_dir.resolve()
    path = (dataset_dir / relative_path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Dataset画像が不正なパスです")
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("画像が32MiBを超えています")
    return path.read_bytes()


def _restore_tags(data: dict) -> PreparedTags:
    normalized = data["prepared"]["normalized"]
    original = TagSet(
        tuple(ScoredTag(**entry) for entry in normalized["tags"]),
        caption=normalized["caption"],
        rating=normalized["rating"],
    )
    return PreparedTags(
        normalized=original,
        positive_tags=tuple(data["prepared"]["positive_tags"]),
        category_tags={
            key: tuple(value)
            for key, value in data["prepared"]["category_tags"].items()
        },
    )


def _restore_caption(data: dict) -> SceneDescription:
    scene = data["scene"]
    subjects = tuple(
        Subject(
            name=subject["name"],
            count=subject["count"],
            attributes=tuple(subject["attributes"]),
        )
        for subject in scene["subjects"]
    )
    return SceneDescription(
        short=scene["short"],
        standard=scene["standard"],
        detailed=scene["detailed"],
        summary=scene["summary"],
        subjects=subjects,
        relations=tuple(scene["relations"]),
        tags=tuple(scene["tags"]),
        style=scene["style"],
        composition=scene["composition"],
        background=scene["background"],
    )


# {
#   責務: [preprocess_dataset: 全画像を3つの独立したパスで処理する]
#   処理: [①全画像タグ→②全画像Caption→③整合チェック・プロンプト→manifest保存]
#   引数: [dataset_dir: 登録済みDataset, pipeline: 設定済み前処理, seed: 粒度選択]
#   戻り値: [DatasetPreprocessingReport: 正常件数と失敗分母]
# }
def preprocess_dataset(
    dataset_dir: Path,
    pipeline: ImageTrainingPreprocessor,
    *,
    seed: int = 0,
) -> DatasetPreprocessingReport:
    # Datasetからの無断外部送信を防ぐため、この専用入口はローカル接続のみを認める。
    providers = (
        *pipeline.caption_pipeline.providers,
        *pipeline.tag_pipeline.providers,
    )
    for provider in providers:
        backend = getattr(provider, "backend", provider)
        endpoint = getattr(backend, "endpoint", None)
        if endpoint is None:
            continue
        if urlsplit(endpoint).hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Dataset3パスの抽出はローカルAPIのみ対応です")
    manifest = DatasetManifest.load(dataset_dir)
    # 結果が異なるモデル列や整形設定で混ざらないよう、キャッシュ空間を分ける。
    identities = [
        (
            provider.provider_id,
            provider.model_id,
            getattr(provider, "config_fingerprint", ""),
        )
        for provider in (
            *pipeline.tag_pipeline.providers,
            *pipeline.caption_pipeline.providers,
        )
    ]
    profile = {
        "models": identities,
        "normalizer": pipeline.tag_pipeline.normalizer.config_fingerprint,
        "tag_formatter": pipeline.tag_pipeline.formatter.config_fingerprint,
        "composer": COMPOSER_VERSION,
        "seed": seed,
        "custom_prompt_formatter": getattr(
            pipeline.prompt_formatter, "__qualname__", None
        ),
    }
    profile_hash = hashlib.sha256(
        json.dumps(profile, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    cache_root = dataset_dir / "cache" / "preprocessing-v2" / profile_hash

    # ①登録画像を全てタグ処理してキャッシュする。GPUには同時に大量画像を置かない。
    tag_results: dict[str, dict] = {}
    for item in manifest.images:
        image = _read_image(dataset_dir, item.image_path)
        mime_type = _detect_mime(image)
        key = pipeline.tag_pipeline.cache_key(image)
        file = cache_root / "tags" / f"{item.image_sha256}.json"
        record = _read_stage(file, key)
        if record is None:
            try:
                extracted = pipeline.tag_pipeline.extract(image, mime_type)
                record = {
                    "cache_key": key,
                    "status": "ok",
                    "model_id": extracted.model_id,
                    "attempts": [asdict(a) for a in extracted.attempts],
                    "prepared": asdict(extracted.prepared),
                }
            except PromptExtractionError as exc:
                record = {"cache_key": key, "status": exc.kind, "error": str(exc)}
            _write_stage(file, record)
        tag_results[item.image_id] = record

    # ②タグが失敗した画像もCaptionを独立実行し、全画像の結果を先に保存する。
    caption_results: dict[str, dict] = {}
    for item in manifest.images:
        image = _read_image(dataset_dir, item.image_path)
        mime_type = _detect_mime(image)
        key = pipeline.caption_pipeline._cache_key(image, mime_type)
        file = cache_root / "captions" / f"{item.image_sha256}.json"
        record = _read_stage(file, key)
        if record is None:
            try:
                extracted = pipeline.caption_pipeline.extract(image, mime_type)
                record = {
                    "cache_key": key,
                    "status": "ok",
                    "model_id": extracted.selected_model,
                    "attempts": [asdict(a) for a in extracted.attempts],
                    "scene": asdict(extracted.scene),
                }
            except PromptExtractionError as exc:
                record = {"cache_key": key, "status": exc.kind, "error": str(exc)}
            _write_stage(file, record)
        caption_results[item.image_id] = record

    # ③双方の成果を検証して、自然な学習Promptを統合する。
    failures: list[tuple[str, str, str]] = []
    count = 0
    for item in manifest.images:
        tag = tag_results[item.image_id]
        caption = caption_results[item.image_id]
        item.quality_issues = []
        if tag["status"] != "ok" or caption["status"] != "ok":
            # 過去の抽出器が作ったタグ・Captionを今回の教師として残さない。
            item.training_prompt = ""
            item.tags = []
            item.caption = ""
            for stage, record in (("tags", tag), ("caption", caption)):
                if record["status"] != "ok":
                    failures.append((item.image_id, stage, record["status"]))
                    item.quality_issues.append(
                        QualityIssue("preprocessing_failed", f"{stage}: {record['status']}")
                    )
            continue
        prepared = _restore_tags(tag)
        scene = _restore_caption(caption)
        try:
            if pipeline.prompt_formatter is not None:
                prompt = pipeline.prompt_formatter(scene, prepared, seed)
                if not isinstance(prompt, str) or not prompt.strip():
                    raise PromptExtractionError("empty", "独自のPromptFormatterが空を返した")
                included: tuple[str, ...] = ()
                withheld: tuple[tuple[str, str], ...] = ()
            else:
                composed = PromptComposer().compose(scene, prepared, seed)
                prompt, included, withheld = (
                    composed.text,
                    composed.included,
                    composed.withheld,
                )
            item.training_prompt = prompt
            item.caption = scene.standard
            item.tags = [asdict(t) for t in prepared.normalized.tags]
            item.cache_key = _stage_key(
                item.image_sha256.encode("utf-8"),
                profile_hash + tag["cache_key"] + caption["cache_key"],
            )
            _write_stage(
                cache_root / "prompts" / f"{item.image_sha256}.json",
                {
                    "cache_key": item.cache_key,
                    "status": "ok",
                    "prompt": prompt,
                    "included_tags": included,
                    "withheld_tags": withheld,
                },
            )
            count += 1
        except (PromptExtractionError, ValueError) as exc:
            item.training_prompt = ""
            item.tags = []
            item.caption = ""
            failures.append((item.image_id, "compose", type(exc).__name__))
            item.quality_issues.append(QualityIssue("prompt_failed", str(exc)))

    # 完成した結果だけをManifestに反映し、splitと目標画像の隔離を保持する。
    tag_info = ExtractorInfo(
        id="pipeline:tagger",
        version="tagging-v2",
        capabilities=["tag"],
        model_hash=profile_hash,
        config_hash=profile_hash,
    )
    caption_info = ExtractorInfo(
        id="pipeline:caption",
        version="caption-v2",
        capabilities=["caption"],
        model_hash=profile_hash,
        config_hash=profile_hash,
    )
    for item in manifest.images:
        item.tagger = tag_info
        item.captioner = caption_info
        item.prompt_builder_version = COMPOSER_VERSION
    manifest.tagger = tag_info
    manifest.captioner = caption_info
    manifest.prompt_builder_version = COMPOSER_VERSION
    manifest.save(dataset_dir)
    return DatasetPreprocessingReport(
        images=len(manifest.images),
        tags_ok=sum(r["status"] == "ok" for r in tag_results.values()),
        captions_ok=sum(r["status"] == "ok" for r in caption_results.values()),
        prompts_ok=count,
        failures=tuple(failures),
        dataset_hash=manifest.dataset_hash,
    )

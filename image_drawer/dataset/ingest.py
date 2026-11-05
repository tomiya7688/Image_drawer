"""画像だけのDatasetを登録する。正規化・重複排除・分割・抽出・キャッシュ・品質警告を行う。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from image_drawer.core import SerializableModel
from image_drawer.dataset.egress import prepare_egress
from image_drawer.dataset.extraction import extract_with_cache
from image_drawer.dataset.manifest import DatasetManifest
from image_drawer.dataset.models import (
    ExtractorInfo,
    QualityIssue,
    TrainingImage,
)
from image_drawer.dataset.normalize import (
    EXTENSIONS,
    NORMALIZATION_VERSION,
    ImageSignature,
    image_signature,
    group_near_duplicates,
    hash_file,
    load_normalized,
)
from image_drawer.dataset.registry import ExtractorRegistry

PROMPT_BUILDER_VERSION = "prompt-builder-v1"
PROMPT_BUILDER_V2 = "prompt-builder-v2"
SPLIT_RATIOS = (("train", 0.8), ("validation", 0.1), ("test", 0.1))


# {
#   責務: [
#     IngestFailure: 登録できなかった1ファイルの理由を保持する
#   ]
#   フィールド: [
#     path: 対象ファイル, error_type: 例外の種類, message: 理由
#   ]
# }
@dataclass(slots=True)
class IngestFailure(SerializableModel):
    path: str
    error_type: str
    message: str


# {
#   責務: [
#     IngestReport: 登録の結果を数と理由で報告する。除外した件数を隠さない
#   ]
#   フィールド: [
#     added: 新しく登録した画像, reextracted: 抽出器を変えて抽出し直した画像
#     duplicates: 完全一致のため登録しなかった画像
#     quality_warnings: 品質警告のある画像, errors: 登録できなかったファイル
#   ]
# }
@dataclass(slots=True)
class IngestReport(SerializableModel):
    added: int = 0
    reextracted: int = 0
    duplicates: int = 0
    quality_warnings: int = 0
    errors: list[IngestFailure] = field(default_factory=list)


# {
#   責務: [
#     build_prompt: 説明文とタグから、学習用promptを組み立てる(PromptBuilder)
#   ]
#   処理: [
#     説明文を先頭に置き、confidenceの高いタグを最大5個、括弧で補う。
#     抽出器がin_prompt=falseと示したタグ(rating、copyright、style、metaなど)は文に使わない
#   ]
#   引数: [
#     caption: CaptionExtractorの出力
#     tags: TagExtractorの出力
#   ]
#   戻り値: [
#     prompt: 学習用prompt。説明文もタグも無ければ空文字
#   ]
# }
def build_prompt(caption: str, tags: list[dict[str, Any]]) -> str:
    usable = [tag for tag in tags if tag.get("in_prompt", True) is not False]
    labels = [
        tag["label"] for tag in sorted(usable, key=lambda tag: (-tag["confidence"], tag["label"]))
    ][:5]
    if caption and labels:
        return f"{caption}（{'、'.join(labels)}）"
    return caption or "、".join(labels)


# {
#   責務: [
#     prompt_builder_version: 抽出器の組み合わせから、使うPromptBuilderのversionを決める
#   ]
#   処理: [
#     Captionerが短文・標準文・詳細文を返せる(extract_variantsを持つ)場合はv2、他はv1。
#     v2は自然文とタグの矛盾を避けるため、タグを文へ追記しない
#   ]
#   戻り値: [
#     version: "prompt-builder-v1" または "prompt-builder-v2"
#   ]
# }
def prompt_builder_version(captioner: Any) -> str:
    return PROMPT_BUILDER_V2 if hasattr(captioner, "extract_variants") else PROMPT_BUILDER_VERSION


# {
#   責務: [
#     build_prompt_v2: 短文・標準文・詳細文がある画像の学習用promptを作る(PromptBuilder v2)
#   ]
#   処理: [
#     標準文をそのまま使い、タグは追記しない(タグは別に保存する)。
#     変種が無い(抽出に失敗した)場合はv1の組み立てにする
#   ]
#   戻り値: [
#     prompt: 学習用prompt
#   ]
# }
def build_prompt_v2(caption: str, tags: list[dict[str, Any]], variants: dict[str, str]) -> str:
    standard = variants.get("standard", "").strip()
    return standard or build_prompt(caption, tags)


# {
#   責務: [
#     compute_cache_key: 画像と抽出器の選択から、抽出結果の再利用の鍵を作る
#   ]
#   処理: [
#     画像のSHA-256、正規化version、両抽出器のid・version・model_hash・config_hash、
#     PromptBuilderのversionをまとめてSHA-256にする。どれかが変わると鍵も変わる
#   ]
#   引数: [
#     sha256: 画像のSHA-256, tagger, captioner: 抽出器の情報
#     builder_version: PromptBuilderのversion。既定はv1
#   ]
#   戻り値: [
#     key: 16進のSHA-256
#   ]
# }
def compute_cache_key(
    sha256: str,
    tagger: ExtractorInfo,
    captioner: ExtractorInfo,
    builder_version: str = PROMPT_BUILDER_VERSION,
) -> str:
    parts = [sha256, NORMALIZATION_VERSION, builder_version]
    for info in (tagger, captioner):
        parts += [info.id, info.version, info.model_hash, info.config_hash]
    return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()


# {
#   責務: [
#     assign_split: グループIDから、決定的にtrain/validation/testへ割り当てる
#   ]
#   処理: [
#     seedとグループIDのSHA-256を[0,1)の値にし、割合の累積と比べる。同じグループは常に同じsplit
#   ]
#   引数: [
#     group_id: 近重複のグループ
#     seed: 分割の再現用の値
#   ]
#   戻り値: [
#     split: train、validation、test のいずれか
#   ]
# }
def assign_split(group_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) / 2**32
    cumulative = 0.0
    for name, ratio in SPLIT_RATIOS:
        cumulative += ratio
        if value < cumulative:
            return name
    return SPLIT_RATIOS[-1][0]


def _files(root: Path) -> Iterator[Path]:
    def on_error(error: OSError) -> None:
        raise error

    for directory, dirs, files in os.walk(root, followlinks=False, onerror=on_error):
        dirs[:] = sorted(name for name in dirs if not (Path(directory) / name).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix.lower() in EXTENSIONS and not path.is_symlink():
                yield path


def _save_png(image: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(descriptor)
    try:
        image.save(name, format="PNG")
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


# {
#   責務: [
#     _load_stored: 登録済みの正規化PNGを開いてRGB画像として返す
#   ]
#   引数: [
#     path: 正規化PNGのパス
#   ]
#   戻り値: [
#     image: RGB画像(呼び出し側が閉じる)
#   ]
# }
def _load_stored(path: Path) -> Any:
    from PIL import Image

    with Image.open(path) as stored:
        return stored.convert("RGB")


# {
#   責務: [
#     _scan_sources: 元画像を検査・正規化して保存し、近重複の判定に使うハッシュを集める
#   ]
#   処理: [
#     1: ファイルを1件ずつ読む。失敗は報告に残して続ける
#     2: 完全一致(同じSHA-256)は2件目以降を登録せず、重複として数える
#     3: 正規化したPNGを内容のハッシュ名で保存する
#   ]
#   引数: [
#     source_dir: 元画像のディレクトリ
#     dataset_dir: Datasetの保存先
#     report: 結果を書き込む報告
#   ]
#   戻り値: [
#     hashes: 今回読めた画像のSHA-256から近重複判定用の署名への対応
#   ]
# }
def _scan_sources(
    source_dir: Path, dataset_dir: Path, report: IngestReport
) -> dict[str, ImageSignature]:
    hashes: dict[str, ImageSignature] = {}
    for path in _files(source_dir):
        try:
            sha256 = hash_file(path)
            if sha256 in hashes:
                report.duplicates += 1
                continue
            image = load_normalized(path)
            try:
                hashes[sha256] = image_signature(image)
                _save_png(image, dataset_dir / "images" / sha256[:2] / f"{sha256}.png")
            finally:
                image.close()
        except (OSError, ValueError) as exc:
            report.errors.append(IngestFailure(str(path), type(exc).__name__, str(exc)))
    return hashes


# {
#   責務: [
#     _build_entry: 画像1件の抽出・品質警告・分割を行い、manifestの1行を作る
#   ]
#   引数: [
#     sha256: 画像のSHA-256
#     group_id: 近重複のグループ
#     previous: 既存のmanifestの同じ画像。無ければNone
#     context: 登録全体の設定(registry、抽出器、seed)
#     report: 結果を書き込む報告
#   ]
#   戻り値: [
#     entry: TrainingImage
#   ]
# }
def _build_entry(
    sha256: str,
    group_id: str,
    previous: TrainingImage | None,
    context: "_Context",
    report: IngestReport,
) -> TrainingImage:
    key = compute_cache_key(
        sha256, context.tagger.info, context.captioner.info, context.builder_version
    )
    relative = f"images/{sha256[:2]}/{sha256}.png"
    image = _load_stored(context.dataset_dir / relative)
    try:
        extraction = extract_with_cache(
            image,
            key,
            context.dataset_dir / "cache",
            context.registry,
            context.tagger.info.id,
            context.captioner.info.id,
        )
        width, height = image.size
    finally:
        image.close()
    tags, caption, issues = extraction.tags, extraction.caption, extraction.issues
    if context.builder_version == PROMPT_BUILDER_V2:
        prompt = build_prompt_v2(caption, tags, extraction.variants)
    else:
        prompt = build_prompt(caption, tags)
    if not prompt:
        issues.append(QualityIssue("empty_prompt", "training prompt is empty"))
    if previous is None:
        report.added += 1
    elif previous.cache_key != key:
        report.reextracted += 1
    report.quality_warnings += bool(issues)
    return TrainingImage(
        image_id=f"img_{sha256[:16]}",
        image_sha256=sha256,
        group_id=group_id,
        split=assign_split(group_id, context.seed),
        image_path=relative,
        width=width,
        height=height,
        normalization_version=NORMALIZATION_VERSION,
        tags=tags,
        caption=caption,
        training_prompt=prompt,
        tagger=context.tagger.info,
        captioner=context.captioner.info,
        prompt_builder_version=context.builder_version,
        cache_key=key,
        quality_issues=issues,
        caption_variants=extraction.variants,
        extraction_trace=extraction.trace,
    )


@dataclass(slots=True)
class _Context:
    dataset_dir: Path
    registry: ExtractorRegistry
    tagger: Any
    captioner: Any
    seed: int
    builder_version: str = PROMPT_BUILDER_VERSION


# {
#   責務: [
#     register_images: ディレクトリの画像をDatasetに登録または更新し、manifestを書き出す
#   ]
#   処理: [
#     1: 元画像を検査・正規化して保存する(失敗した画像は報告に残す)
#     2: 既存の登録画像と合わせて近重複をグループにし、splitをグループ単位で決める
#     3: 選んだ抽出器でタグ・説明文を抽出する。抽出器を変えた画像は再抽出する
#     4: manifestをハッシュ付きで原子的に書く
#   ]
#   引数: [
#     source_dir: 画像のディレクトリ(中身は変更しない)
#     dataset_dir: Datasetの保存先
#     registry: 抽出器のregistry
#     tagger_id, captioner_id: 利用者が選んだ抽出器
#     seed: 分割の再現用の値
#     allow_external_upload: 実行時の外部送信の許可。外部へ送る抽出器は、Datasetの設定と宣言も必要
#   ]
#   戻り値: [
#     report: 追加・重複・再抽出・警告・失敗の件数と理由
#   ]
#   エラー: [
#     KeyError: 未登録の抽出器
#     ExternalUploadDenied: 外部送信の許可が足りない(画像を読む前に止める)
#   ]
# }
def register_images(
    source_dir: Path,
    dataset_dir: Path,
    registry: ExtractorRegistry,
    *,
    tagger_id: str,
    captioner_id: str,
    seed: int = 0,
    allow_external_upload: bool = False,
) -> IngestReport:
    context = _Context(
        dataset_dir,
        registry,
        registry.tagger(tagger_id),
        registry.captioner(captioner_id),
        seed,
    )
    context.builder_version = prompt_builder_version(context.captioner)
    prepare_egress([context.tagger, context.captioner], dataset_dir, allow_external_upload)
    report = IngestReport()
    previous: dict[str, TrainingImage] = {}
    if (dataset_dir / "manifest.json").is_file():
        previous = {i.image_sha256: i for i in DatasetManifest.load(dataset_dir).images}

    hashes = _scan_sources(source_dir, dataset_dir, report)
    for sha256 in previous.keys() - hashes.keys():
        image = _load_stored(dataset_dir / previous[sha256].image_path)
        try:
            hashes[sha256] = image_signature(image)
        finally:
            image.close()
    groups = group_near_duplicates(hashes)
    images = [
        _build_entry(sha256, groups[sha256], previous.get(sha256), context, report)
        for sha256 in sorted(hashes)
    ]
    DatasetManifest(
        normalization_version=NORMALIZATION_VERSION,
        prompt_builder_version=context.builder_version,
        tagger=context.tagger.info,
        captioner=context.captioner.info,
        split_seed=seed,
        images=images,
    ).save(dataset_dir)
    return report

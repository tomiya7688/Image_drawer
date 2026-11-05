"""抽出器の呼び出しと、その結果のキャッシュ。失敗した結果はキャッシュしない。"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from image_drawer.dataset.models import DISABLED_TAGGER_ID, QualityIssue
from image_drawer.dataset.registry import ExtractorRegistry

LOW_CONFIDENCE = 0.2


# {
#   責務: [
#     Extraction: 1枚の画像の抽出結果を保持する
#   ]
#   フィールド: [
#     tags: タグ(ラベルとconfidence)の一覧, caption: 説明文, issues: 品質警告
#     variants: 短文・標準文・詳細文(抽出器がextract_variantsを持つ場合だけ)
#     trace: 抽出器ごとの記録。キーは "caption" と "tags"(extract_with_traceを持つ場合だけ)
#   ]
# }
@dataclass(slots=True)
class Extraction:
    tags: list[dict[str, Any]] = field(default_factory=list)
    caption: str = ""
    issues: list[QualityIssue] = field(default_factory=list)
    variants: dict[str, str] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)


# {
#   責務: [
#     _valid_tags: タグの一覧が label(文字列)と confidence(数値)を持つ辞書の一覧か確認する
#   ]
#   戻り値: [
#     result: 正しければTrue
#   ]
# }
def _valid_tags(tags: Any) -> bool:
    return isinstance(tags, list) and all(
        isinstance(tag, dict)
        and isinstance(tag.get("label"), str)
        and type(tag.get("confidence")) in (int, float)
        for tag in tags
    )


def _valid_variants(variants: Any) -> bool:
    return isinstance(variants, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in variants.items()
    )


# {
#   責務: [
#     _read_cache: キャッシュを読み、形式を検証する。壊れていれば使わない
#   ]
#   処理: [
#     JSONの構造・型を確認し、1つでも不正なら None を返して呼び出し側が再抽出する。
#     variantsとtraceは古いキャッシュに無いため、無ければ空として読む
#   ]
#   戻り値: [
#     result: Extraction。無い・壊れている場合は None
#   ]
# }
def _read_cache(path: Path) -> Extraction | None:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        tags, caption, raw_issues = data["tags"], data["caption"], data["issues"]
        variants, trace = data.get("variants", {}), data.get("trace", {})
        if not (
            _valid_tags(tags)
            and isinstance(caption, str)
            and isinstance(raw_issues, list)
            and _valid_variants(variants)
            and isinstance(trace, dict)
        ):
            return None
        issues = [QualityIssue.from_dict(item) for item in raw_issues]
        return Extraction(tags, caption, issues, variants, trace)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


# {
#   責務: [
#     _write_cache: 抽出結果を一時ファイルに書いてから置き換え、途中で壊れたキャッシュを残さない
#   ]
#   引数: [
#     path: キャッシュのパス, result: 保存する抽出結果
#   ]
# }
def _write_cache(path: Path, result: Extraction) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            payload = {
                "tags": result.tags,
                "caption": result.caption,
                "issues": [issue.to_dict() for issue in result.issues],
                "variants": result.variants,
                "trace": result.trace,
            }
            json.dump(payload, stream, ensure_ascii=False)
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


# {
#   責務: [
#     failure_message: 抽出器の例外を品質警告の文にする。分類(kind)を持つ例外は"kind: 内容"にする
#   ]
#   引数: [
#     role: "tagger" または "captioner", extractor_id: 抽出器のid, exc: 例外
#   ]
#   戻り値: [
#     message: 利用者向けの説明。例外の内容にAPIキーや画像を含めない責任は抽出器にある
#   ]
# }
def failure_message(role: str, extractor_id: str, exc: Exception) -> str:
    kind = getattr(exc, "kind", None)
    detail = f"{kind}: {exc}" if isinstance(kind, str) and kind else str(exc)
    return f"{role} {extractor_id}: {detail}"


def _extract_tags(extractor: Any, image: Any, extractor_id: str, out: Extraction) -> None:
    try:
        if hasattr(extractor, "extract_with_trace"):
            tags, out.trace["tags"] = extractor.extract_with_trace(image)
        else:
            tags = extractor.extract(image)
        if _valid_tags(tags):
            out.tags = tags
        else:
            out.trace.pop("tags", None)
            msg = f"tagger {extractor_id}: bad tags"
            out.issues.append(QualityIssue("extractor_invalid_output", msg))
    except Exception as exc:  # 抽出器は外部モデルを含みうるため、種類を問わず警告にする
        msg = failure_message("tagger", extractor_id, exc)
        out.issues.append(QualityIssue("extractor_failed", msg))


def _extract_caption(extractor: Any, image: Any, extractor_id: str, out: Extraction) -> None:
    try:
        if hasattr(extractor, "extract_with_trace"):
            caption, out.trace["caption"] = extractor.extract_with_trace(image)
        else:
            caption = extractor.extract(image)
        if isinstance(caption, str):
            out.caption = caption
        else:
            out.trace.pop("caption", None)
            msg = f"captioner {extractor_id}: not text"
            out.issues.append(QualityIssue("extractor_invalid_output", msg))
            return
        if hasattr(extractor, "extract_variants"):
            variants = extractor.extract_variants(image)
            if _valid_variants(variants):
                out.variants = variants
            else:
                msg = f"captioner {extractor_id}: bad variants"
                out.issues.append(QualityIssue("extractor_invalid_output", msg))
    except Exception as exc:
        msg = failure_message("captioner", extractor_id, exc)
        out.issues.append(QualityIssue("extractor_failed", msg))


# {
#   責務: [
#     extract_with_cache: 抽出結果をキャッシュから再利用し、無ければ抽出して保存する
#   ]
#   処理: [
#     1: cache_keyのファイルがあれば読む。抽出器や設定が変わると鍵が変わるので古い結果は使われない
#     2: 無ければ抽出し、失敗は品質警告にして空の結果で続ける(画像を黙って除外しない)。
#        抽出器がextract_with_trace/extract_variantsを持てば使う(任意の拡張)
#     3: extractor_failedを含む結果は保存しない。一時的な障害が、復旧後も残らないようにする
#   ]
#   戻り値: [
#     result: Extraction
#   ]
# }
def extract_with_cache(
    image: Any,
    key: str,
    cache_dir: Path,
    registry: ExtractorRegistry,
    tagger_id: str,
    captioner_id: str,
) -> Extraction:
    cache_path = cache_dir / f"{key}.json"
    cached = _read_cache(cache_path)
    if cached is not None:
        return cached
    out = Extraction()
    _extract_tags(registry.tagger(tagger_id), image, tagger_id, out)
    _extract_caption(registry.captioner(captioner_id), image, captioner_id, out)
    if not out.caption.strip():
        out.issues.append(QualityIssue("empty_caption", "caption is empty"))
    if not out.tags and tagger_id != DISABLED_TAGGER_ID:
        out.issues.append(QualityIssue("empty_tags", "no tags were extracted"))
    elif out.tags and max(tag["confidence"] for tag in out.tags) < LOW_CONFIDENCE:
        out.issues.append(QualityIssue("low_confidence", "all tags are below the confidence floor"))
    if not any(issue.code == "extractor_failed" for issue in out.issues):
        _write_cache(cache_path, out)
    return out

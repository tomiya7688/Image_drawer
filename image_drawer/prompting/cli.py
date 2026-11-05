"""画像だけを指定して文章とタグを両方生成するCLI。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .core import PromptExtractionError
from .registry import PreprocessorRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="設定済みCaption/Taggerを使い、画像だけから学習前プロンプトと分類タグを生成"
    )
    parser.add_argument("--settings", required=True, help="一度用意するアプリ設定JSON")
    parser.add_argument("--image", required=True, help="登録したい画像ファイル")
    parser.add_argument("--seed", type=int, default=0, help="学習プロンプトの長さを再現するseed")
    args = parser.parse_args(argv)
    try:
        registry = PreprocessorRegistry.from_settings_file(args.settings)
        result = registry.build().preprocess_file(args.image)
    except KeyError as exc:
        parser.exit(2, f"設定に必須項目がありません: {exc}\n")
    except (ValueError, OSError, PromptExtractionError) as exc:
        parser.exit(1, f"前処理に失敗しました: {exc}\n")

    payload = {
        "image_sha256": result.image_sha256,
        "training_prompt": result.training_prompt(args.seed),
        "caption": {
            "model_id": result.caption.selected_model,
            "fallback_used": result.caption.fallback_used,
            "verification": result.caption.verifier_status,
            "short": result.caption.scene.short,
            "standard": result.caption.scene.standard,
            "detailed": result.caption.scene.detailed,
            "subjects": [asdict(item) for item in result.caption.scene.subjects],
            "relations": list(result.caption.scene.relations),
            "attempts": [asdict(item) for item in result.caption.attempts],
        },
        "tagger": {
            "model_id": result.tags.model_id,
            "fallback_used": result.tags.fallback_used,
            "attempts": [asdict(item) for item in result.tags.attempts],
            "positive_tags": list(result.formatted_tags),
            "categories": result.tag_categories,
            "rating": result.tags.prepared.normalized.rating,
            "scores": [asdict(item) for item in result.tags.prepared.normalized.tags],
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

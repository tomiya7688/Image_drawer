"""画像のみのDataset登録CLI。登録、抽出器の一覧、内容の確認を1つの入口にまとめる。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

from image_drawer.dataset import DatasetManifest, create_default_registry, register_images
from image_drawer.dataset.prompting_adapters import register_prompting_extractors
from image_drawer.dataset.upload_plan import plan_upload
from image_drawer.prompting.registry import PreprocessorRegistry


# {
#   責務: [
#     _load_extractor_settings: 抽出器の設定JSONを読み、promptingの抽出器をregistryへ登録する
#   ]
#   引数: [
#     registry: 登録先, path: 設定JSONのパス
#   ]
#   戻り値: [
#     ids: 登録した抽出器のid(prompting:caption:<profile>、prompting:tags:<profile>)
#   ]
#   エラー: [
#     ValueError, KeyError, OSError: 設定が不正、または読めない
#   ]
# }
def _load_extractor_settings(registry, path: str) -> list[str]:
    raw = Path(path).read_bytes()
    settings = json.loads(raw.decode("utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("設定JSONはobject")
    preprocessors = PreprocessorRegistry()
    preprocessors.load_settings(settings)
    return register_prompting_extractors(
        registry, preprocessors, settings_sha256=hashlib.sha256(raw).hexdigest()
    )


# {
#   責務: [
#     main: サブコマンド(register、extractors、show)を実行する
#   ]
#   引数: [
#     argv: コマンドライン引数
#   ]
#   戻り値: [
#     exit_code: 成功は0、登録できなかった画像があれば1、実行できなければ2
#   ]
# }
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="image-drawer-dataset")
    commands = parser.add_subparsers(dest="command", required=True)
    register = commands.add_parser("register", help="画像のディレクトリをDatasetに登録する")
    register.add_argument("source")
    register.add_argument("--dataset", required=True)
    register.add_argument(
        "--tagger", default=None, help="既定: 設定があればprompting、なければcolor-layout-tags"
    )
    register.add_argument(
        "--captioner",
        default=None,
        help="既定: 設定があればprompting、なければcolor-layout-caption",
    )
    register.add_argument("--seed", type=int, default=0)
    register.add_argument("--extractor-settings", help="外部モデルを使う抽出器の設定JSON")
    register.add_argument(
        "--allow-external-upload",
        action="store_true",
        help="外部ホストへ画像を送ることを今回の実行で許可する(設定ファイルだけでは送らない)",
    )
    register.add_argument(
        "--dry-run",
        action="store_true",
        help="送信先・枚数・総バイト数を表示するだけで何も送らない",
    )
    commands.add_parser("extractors", help="登録済みの抽出器を一覧する")
    show = commands.add_parser("show", help="Datasetの概要を表示する")
    show.add_argument("--dataset", required=True)
    args = parser.parse_args(argv)
    registry = create_default_registry()
    try:
        if args.command == "extractors":
            print(
                json.dumps(
                    {k: [i.to_dict() for i in v] for k, v in registry.describe().items()},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
        if args.command == "show":
            manifest = DatasetManifest.load(Path(args.dataset))
            splits: dict[str, int] = {}
            for image in manifest.images:
                splits[image.split] = splits.get(image.split, 0) + 1
            print(
                json.dumps(
                    {
                        "images": len(manifest.images),
                        "splits": splits,
                        "dataset_hash": manifest.dataset_hash,
                        "tagger": manifest.tagger.id,
                        "captioner": manifest.captioner.id,
                    },
                    ensure_ascii=False,
                )
            )
            return 0
        ids = (
            _load_extractor_settings(registry, args.extractor_settings)
            if args.extractor_settings
            else []
        )
        tagger_id = args.tagger or next(
            (i for i in ids if i.startswith("prompting:tags:")), "color-layout-tags"
        )
        captioner_id = args.captioner or next(
            (i for i in ids if i.startswith("prompting:caption:")), "color-layout-caption"
        )
        if args.dry_run:
            plan = plan_upload(
                Path(args.source),
                Path(args.dataset),
                registry.tagger(tagger_id),
                registry.captioner(captioner_id),
                allow_external_upload=args.allow_external_upload,
            )
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        report = register_images(
            Path(args.source),
            Path(args.dataset),
            registry,
            tagger_id=tagger_id,
            captioner_id=captioner_id,
            seed=args.seed,
            allow_external_upload=args.allow_external_upload,
        )
    except (OSError, ValueError, KeyError) as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(report.to_json())
    return 1 if report.errors else 0

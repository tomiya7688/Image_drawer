"""画像のみのDataset登録CLI。登録、抽出器の一覧、内容の確認を1つの入口にまとめる。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from image_drawer.dataset import DatasetManifest, create_default_registry, register_images


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
    register.add_argument("--tagger", default="color-layout-tags")
    register.add_argument("--captioner", default="color-layout-caption")
    register.add_argument("--seed", type=int, default=0)
    register.add_argument(
        "--confirm-rights",
        action="store_true",
        help="登録する画像の利用権利を確認済みとして記録する",
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
        report = register_images(
            Path(args.source),
            Path(args.dataset),
            registry,
            tagger_id=args.tagger,
            captioner_id=args.captioner,
            seed=args.seed,
            rights_status="confirmed" if args.confirm_rights else "unconfirmed",
        )
    except (OSError, ValueError, KeyError) as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2
    print(report.to_json())
    return 1 if report.errors else 0

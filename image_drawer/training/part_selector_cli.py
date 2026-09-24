"""PartSelector baseline学習CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from image_drawer.training.part_selector import (
    evaluate_part_selector_checkpoint,
    train_part_selector_from_config,
)
from image_drawer.training.results import load_training_dataset


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-drawer-train-part-selector",
        description="baseline PartSelector rankerを学習・評価します。",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train")
    train.add_argument("config", type=Path)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("checkpoint", type=Path)
    evaluate.add_argument("dataset", type=Path)
    evaluate.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
    )
    evaluate.add_argument("--top-k", type=int)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.command == "train":
        result = train_part_selector_from_config(args.config)
        print(
            json.dumps(
                {
                    "status": "ok",
                    "dataset_id": result.dataset_id,
                    "checkpoint_path": result.checkpoint_path,
                    "best_epoch": result.best_epoch,
                    "best_validation_metrics": (
                        result.best_validation_metrics.to_dict()
                    ),
                    "test_metrics": (
                        result.test_metrics.to_dict()
                        if result.test_metrics is not None
                        else None
                    ),
                    "feature_schema": result.feature_schema.to_dict(),
                    "feature_schema_hash": (
                        result.feature_schema.identity_hash
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    dataset = load_training_dataset(args.dataset)
    metrics = evaluate_part_selector_checkpoint(
        args.checkpoint,
        dataset,
        split=args.split,
        top_k=args.top_k,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "split": args.split,
                "metrics": metrics.to_dict(),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

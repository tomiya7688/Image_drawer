"""WorkflowSelector baseline学習CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from image_drawer.training.results import load_training_dataset
from image_drawer.training.workflow_selector import (
    evaluate_workflow_selector_checkpoint,
    score_workflow_selector_checkpoint,
    train_workflow_selector_from_config,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-drawer-train-workflow-selector",
        description="baseline WorkflowSelector rankerを学習・評価します。",
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

    score = subparsers.add_parser("score")
    score.add_argument("checkpoint", type=Path)
    score.add_argument("prompt")
    score.add_argument(
        "--context-json",
        type=Path,
        help="prompt/task/style/resource context JSON object",
    )
    score.add_argument(
        "--candidates-json",
        type=Path,
        help="optional workflow candidate array; checkpoint workflows are default",
    )
    return parser


def _load_json(path: Path | None):
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.command == "train":
        result = train_workflow_selector_from_config(args.config)
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
                    "label_objective": result.metadata.get(
                        "label_objective"
                    ),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "evaluate":
        dataset = load_training_dataset(args.dataset)
        metrics = evaluate_workflow_selector_checkpoint(
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

    context = _load_json(args.context_json)
    if context is not None and not isinstance(context, dict):
        raise ValueError("context JSON must be an object")
    candidates = _load_json(args.candidates_json)
    if candidates is not None and not isinstance(candidates, list):
        raise ValueError("candidates JSON must be an array")
    ranking = score_workflow_selector_checkpoint(
        args.checkpoint,
        args.prompt,
        candidates=candidates,
        context=context,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "prompt": args.prompt,
                "ranking": ranking,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

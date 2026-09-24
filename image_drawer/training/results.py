"""学習データセットのJSONL export / load。"""

from __future__ import annotations

import json
from pathlib import Path

from image_drawer.training.models import TrainingDataset


def export_training_dataset(
    dataset: TrainingDataset,
    directory: str | Path,
) -> Path:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)

    (root / "dataset.json").write_text(
        dataset.to_json(indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "build_config.json").write_text(
        json.dumps(
            dataset.build_config.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    by_split = {"train": [], "validation": [], "test": []}
    for example in dataset.examples:
        split = example.get("split")
        if split not in by_split:
            raise ValueError(f"unknown dataset split: {split}")
        by_split[split].append(example)

    for split, examples in by_split.items():
        with (root / f"{split}.jsonl").open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as handle:
            for example in examples:
                handle.write(
                    json.dumps(
                        example,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

    manifest = {
        "dataset_id": dataset.id,
        "task": dataset.task,
        "split_counts": dataset.split_counts(),
        "source_ids": dataset.source_ids,
        "metadata": dataset.metadata,
    }
    (root / "manifest.json").write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def load_training_dataset(path: str | Path) -> TrainingDataset:
    source = Path(path)
    if source.is_dir():
        source = source / "dataset.json"
    return TrainingDataset.from_json(source.read_text(encoding="utf-8"))

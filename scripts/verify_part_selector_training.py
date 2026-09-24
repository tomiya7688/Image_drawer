"""built wheelのPartSelector学習をfresh venvで検証する。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def _candidate(part_id: str, tags: list[str], retrieval_score: float, downstream: float):
    return {
        "part_id": part_id,
        "retrieval_score": retrieval_score,
        "part_metadata": {
            "category": "face",
            "bbox": [0, 0, 16, 16],
            "tags": tags,
            "attributes": {},
            "quality": {"downstream_evaluator_score": downstream},
            "extraction_method": "fixture",
            "extraction_version": "v1",
            "metadata": {"search_text": " ".join(tags)},
        },
        "embedding_refs": {},
        "embedding_identity": {
            "family": "metadata-text",
            "model": "feature-hash",
            "version": "v1",
            "dimensions": 32,
        },
    }


def _dataset() -> dict:
    examples = []
    source_ids = []
    for split, count, offset in (
        ("train", 4, 0),
        ("validation", 2, 10),
        ("test", 2, 20),
    ):
        for local_index in range(count):
            index = offset + local_index
            group = f"group-{split}-{index}"
            retrieval = f"retrieval-{split}-{index}"
            trajectory = f"trajectory-{group}"
            source_ids.append(trajectory)
            prompt = f"target token {index}"
            positive = _candidate(
                f"{group}-positive",
                ["target", "token", str(index)],
                1.0,
                0.9,
            )
            negatives = [
                _candidate(
                    f"{group}-negative-a",
                    ["other", "blue"],
                    -0.5,
                    0.2,
                ),
                _candidate(
                    f"{group}-negative-b",
                    ["unrelated", "green"],
                    -1.0,
                    0.1,
                ),
            ]
            for candidate in [positive, *negatives]:
                selected = candidate["part_id"] == positive["part_id"]
                examples.append(
                    {
                        "id": f"point-{candidate['part_id']}",
                        "split": split,
                        "group_id": group,
                        "example_kind": "pointwise",
                        "prompt": prompt,
                        "context": {
                            "category": "face",
                            "retrieval_artifact_id": retrieval,
                            "retrieval_filters": {},
                        },
                        "candidates": [candidate],
                        "label": {
                            "selected": selected,
                            "value": 1 if selected else 0,
                        },
                        "source_trajectory_ids": [trajectory],
                        "metadata": {},
                    }
                )
            for negative in negatives:
                examples.append(
                    {
                        "id": (
                            f"pair-{positive['part_id']}-"
                            f"{negative['part_id']}"
                        ),
                        "split": split,
                        "group_id": group,
                        "example_kind": "pairwise",
                        "prompt": prompt,
                        "context": {
                            "category": "face",
                            "retrieval_artifact_id": retrieval,
                            "retrieval_filters": {},
                        },
                        "candidates": [positive, negative],
                        "label": {
                            "preferred_part_id": positive["part_id"],
                            "rejected_part_id": negative["part_id"],
                        },
                        "source_trajectory_ids": [trajectory],
                        "metadata": {},
                    }
                )
    return {
        "id": "installed-ranker-fixture",
        "task": "part_selector",
        "build_config": {
            "id": "installed-build",
            "split": {
                "train": 0.5,
                "validation": 0.25,
                "test": 0.25,
                "seed": 1,
            },
            "part_example_mode": "both",
            "max_pairs_per_group": 64,
            "metadata": {},
        },
        "examples": examples,
        "source_ids": source_ids,
        "metadata": {"generated_at": "fixed"},
    }


def _config(dataset: Path, output: Path) -> str:
    return f"""[training]
id = "installed-part-selector"
dataset_path = "{dataset.as_posix()}"
output_dir = "{output.as_posix()}"
seed = 123
epochs = 8
batch_size = 4
learning_rate = 0.05
weight_decay = 0.0
hidden_dimensions = [16]
objective = "pairwise"
text_dimensions = 8
top_k = 2
selection_metric = "mrr"
"""


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: verify_part_selector_training.py "
            "<venv-python> <trainer-cli>"
        )
    python = os.path.abspath(sys.argv[1])
    trainer = os.path.abspath(sys.argv[2])
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE"):
        env.pop(key, None)

    def run(*args: str):
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise AssertionError(
                f"exit={result.returncode}: "
                f"{result.stdout}\n{result.stderr}"
            )
        return result

    with tempfile.TemporaryDirectory() as temporary:
        cwd = Path(temporary)
        dataset_path = cwd / "dataset.json"
        dataset_path.write_text(
            json.dumps(_dataset(), sort_keys=True),
            encoding="utf-8",
        )

        summaries = []
        for name in ("first", "second"):
            output = cwd / name
            config = cwd / f"{name}.toml"
            config.write_text(
                _config(dataset_path, output),
                encoding="utf-8",
            )
            trained = run(trainer, "train", str(config))
            summary = json.loads(trained.stdout)
            assert summary["status"] == "ok"
            assert summary["dataset_id"] == "installed-ranker-fixture"
            assert summary["feature_schema"]["version"] == (
                "part-selector-feature-v1"
            )
            assert summary["feature_schema_hash"].startswith("sha256:")
            checkpoint = Path(summary["checkpoint_path"])
            assert checkpoint.is_file()
            assert (output / "metrics.jsonl").is_file()
            assert (output / "training_result.json").is_file()

            evaluated = run(
                trainer,
                "evaluate",
                str(checkpoint),
                str(dataset_path),
                "--split",
                "test",
                "--top-k",
                "2",
            )
            evaluation = json.loads(evaluated.stdout)
            assert evaluation["status"] == "ok"
            assert evaluation["split"] == "test"
            assert evaluation["metrics"]["group_count"] == 2
            assert evaluation["metrics"]["pairwise_accuracy"] is not None
            assert evaluation["metrics"]["mrr"] is not None
            summaries.append((summary, evaluation))

        assert summaries[0][0]["best_epoch"] == summaries[1][0]["best_epoch"]
        assert (
            summaries[0][0]["best_validation_metrics"]
            == summaries[1][0]["best_validation_metrics"]
        )
        assert summaries[0][1]["metrics"] == summaries[1][1]["metrics"]

        inspect = run(
            python,
            "-I",
            "-c",
            """
import importlib.metadata
import sys
from pathlib import Path
import image_drawer
from image_drawer.training import load_part_selector_checkpoint
assert Path(image_drawer.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
requirements = importlib.metadata.metadata('image-drawer').get_all('Requires-Dist') or []
assert any('torch' in item and 'training' in item for item in requirements)
model, schema, metadata = load_part_selector_checkpoint(sys.argv[1])
assert metadata['model_type'] == 'part-selector-mlp'
assert metadata['dataset_id'] == 'installed-ranker-fixture'
assert metadata['feature_schema_hash'] == schema.identity_hash
assert metadata['validation_metrics']['mrr'] is not None
print('installed PartSelector checkpoint verified')
""",
            summaries[0][0]["checkpoint_path"],
        )
        assert "checkpoint verified" in inspect.stdout

    print("built PartSelector training smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

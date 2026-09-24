"""built wheelのWorkflowSelector学習をfresh venvで検証する。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


WORKFLOWS = ("portrait", "landscape", "generic")


def _candidate(key: str, objective: float) -> dict:
    return {
        "workflow_key": key,
        "workflow_id": f"workflow-{key}",
        "workflow_version": "v1",
        "status": "success",
        "objective_value": objective,
        "objective_components": {
            "component:quality": objective,
            "failure_rate": 0.0,
        },
        "workflow_metadata": {
            "content_hash": f"sha256:{key}",
            "backends": {
                "retrieve": "mock",
                "compose": "mock",
                "evaluate": "evaluation",
            },
            "metadata": {
                "template": key,
                "task": key,
            },
        },
    }


def _example(split: str, index: int, target: str, tie: bool = False) -> dict:
    objectives = {
        "portrait": 0.15,
        "landscape": 0.15,
        "generic": 0.05,
    }
    objectives[target] = 1.0
    if tie:
        objectives["generic"] = 0.99
    candidates = [
        _candidate(key, objectives[key])
        for key in WORKFLOWS
    ]
    return {
        "id": f"example-{split}-{index}",
        "split": split,
        "group_id": f"group-{split}-{index}",
        "prompt": f"{target} request sample {index}",
        "context": {
            "experiment_id": "installed-workflow-experiment",
            "prompt_case_id": f"prompt-{split}-{index}",
            "repeat_index": 0,
            "prompt_metadata": {
                "task": target,
                "style": target,
            },
        },
        "candidates": candidates,
        "label": {
            "selected_workflow_key": target,
            "objective": {
                "weights": {"component:quality": 1.0},
                "metadata": {"name": "quality-only"},
            },
        },
        "source_run_ids": [
            f"run-{split}-{index}-{key}"
            for key in WORKFLOWS
        ],
        "source_trajectory_ids": [
            f"trajectory-{split}-{index}-{key}"
            for key in WORKFLOWS
        ],
        "metadata": {"candidate_count": len(candidates)},
    }


def _dataset() -> dict:
    examples = []
    for index in range(6):
        examples.append(
            _example(
                "train",
                index,
                "portrait" if index % 2 == 0 else "landscape",
            )
        )
    examples.append(_example("validation", 10, "portrait"))
    examples.append(_example("validation", 11, "landscape", tie=True))
    examples.append(_example("test", 20, "portrait"))
    examples.append(_example("test", 21, "landscape"))
    source_ids = [
        run_id
        for example in examples
        for run_id in example["source_run_ids"]
    ]
    return {
        "id": "installed-workflow-ranker-fixture",
        "task": "workflow_selector",
        "build_config": {
            "id": "installed-workflow-build",
            "split": {
                "train": 0.6,
                "validation": 0.2,
                "test": 0.2,
                "seed": 1,
            },
            "part_example_mode": "both",
            "max_pairs_per_group": 64,
            "metadata": {},
        },
        "examples": examples,
        "source_ids": source_ids,
        "metadata": {
            "generated_at": "fixed",
            "source_experiment_id": "installed-workflow-experiment",
            "objective": {
                "weights": {"component:quality": 1.0},
                "metadata": {"name": "quality-only"},
            },
        },
    }


def _config(dataset: Path, output: Path) -> str:
    return f"""[training]
id = "installed-workflow-selector"
dataset_path = "{dataset.as_posix()}"
output_dir = "{output.as_posix()}"
seed = 123
epochs = 12
batch_size = 4
learning_rate = 0.05
weight_decay = 0.0
hidden_dimensions = [16]
text_dimensions = 8
top_k = 2
tie_tolerance = 0.02
selection_metric = "top1_accuracy"
final_quality_metric = "component:quality"
"""


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: verify_workflow_selector_training.py "
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
        context_path = cwd / "context.json"
        context_path.write_text(
            json.dumps(
                {
                    "prompt_metadata": {
                        "task": "portrait",
                        "style": "portrait",
                    }
                }
            ),
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
            assert (
                summary["dataset_id"]
                == "installed-workflow-ranker-fixture"
            )
            assert summary["feature_schema"]["version"] == (
                "workflow-selector-feature-v1"
            )
            assert summary["feature_schema_hash"].startswith("sha256:")
            assert summary["label_objective"] == {
                "weights": {"component:quality": 1.0},
                "metadata": {"name": "quality-only"},
            }
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
            assert evaluation["metrics"]["example_count"] == 2
            assert evaluation["metrics"]["top1_accuracy"] is not None
            assert evaluation["metrics"]["top_k_accuracy"] is not None
            assert (
                evaluation["metrics"]["mean_objective_regret"]
                is not None
            )
            assert (
                evaluation["metrics"][
                    "downstream_final_quality_mean"
                ]
                is not None
            )

            scored = run(
                trainer,
                "score",
                str(checkpoint),
                "portrait request for a new user",
                "--context-json",
                str(context_path),
            )
            score_payload = json.loads(scored.stdout)
            assert score_payload["status"] == "ok"
            assert len(score_payload["ranking"]) == 3
            assert {
                item["workflow_key"]
                for item in score_payload["ranking"]
            } == set(WORKFLOWS)
            assert all(
                "objective_value" not in item
                and "status" not in item
                for item in score_payload["ranking"]
            )
            summaries.append((summary, evaluation, score_payload))

        assert summaries[0][0]["best_epoch"] == summaries[1][0]["best_epoch"]
        assert (
            summaries[0][0]["best_validation_metrics"]
            == summaries[1][0]["best_validation_metrics"]
        )
        assert summaries[0][1]["metrics"] == summaries[1][1]["metrics"]
        assert summaries[0][2]["ranking"] == summaries[1][2]["ranking"]

        inspect = run(
            python,
            "-I",
            "-c",
            """
import sys
from pathlib import Path
import image_drawer
from image_drawer.training import load_workflow_selector_checkpoint
assert Path(image_drawer.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
model, schema, metadata = load_workflow_selector_checkpoint(sys.argv[1])
assert metadata['model_type'] == 'workflow-selector-mlp'
assert metadata['dataset_id'] == 'installed-workflow-ranker-fixture'
assert metadata['feature_schema_hash'] == schema.identity_hash
assert len(metadata['known_workflows']) == 3
assert metadata['label_objective']['weights'] == {'component:quality': 1.0}
print('installed WorkflowSelector checkpoint verified')
""",
            summaries[0][0]["checkpoint_path"],
        )
        assert "checkpoint verified" in inspect.stdout

    print("built WorkflowSelector training smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

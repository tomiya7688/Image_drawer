"""Experiment result persistence and analysis-friendly export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from image_drawer.search.models import ExperimentResult


def _safe_name(value: str) -> str:
    return "".join(
        character
        if character.isalnum() or character in "._-"
        else "_"
        for character in value
    ).strip("._") or "record"


def export_experiment(
    result: ExperimentResult,
    directory: str | Path,
) -> Path:
    """Persist one self-contained experiment result directory."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    trajectories = root / "trajectories"
    trajectories.mkdir(parents=True, exist_ok=True)

    (root / "experiment.json").write_text(
        result.to_json(indent=2) + "\n",
        encoding="utf-8",
    )

    for run in result.runs:
        if run.trajectory is None:
            continue
        path = trajectories / f"{_safe_name(run.id)}.json"
        path.write_text(
            run.trajectory.to_json(indent=2) + "\n",
            encoding="utf-8",
        )

    with (root / "runs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_id",
                "workflow_key",
                "prompt_case_id",
                "prompt",
                "repeat_index",
                "seed",
                "status",
                "runtime_seconds",
                "trajectory_id",
                "error",
            ],
        )
        writer.writeheader()
        for run in result.runs:
            writer.writerow(
                {
                    "run_id": run.id,
                    "workflow_key": run.workflow_key,
                    "prompt_case_id": run.prompt_case_id,
                    "prompt": run.prompt,
                    "repeat_index": run.repeat_index,
                    "seed": "" if run.seed is None else run.seed,
                    "status": run.status,
                    "runtime_seconds": f"{run.runtime_seconds:.9f}",
                    "trajectory_id": (
                        run.trajectory.id
                        if run.trajectory is not None
                        else ""
                    ),
                    "error": run.error or "",
                }
            )

    component_names = sorted(
        {
            component
            for metric in result.aggregate_metrics.values()
            for component in metric.component_means
        }
    )
    with (root / "workflow_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fields = [
            "workflow_key",
            "run_count",
            "success_count",
            "failure_count",
            "failure_rate",
            "runtime_seconds_mean",
            "runtime_seconds_min",
            "runtime_seconds_max",
            "evaluation_count",
            "score_overall_mean",
            *[f"component:{name}" for name in component_names],
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, metric in result.aggregate_metrics.items():
            row = {
                "workflow_key": key,
                "run_count": metric.run_count,
                "success_count": metric.success_count,
                "failure_count": metric.failure_count,
                "failure_rate": metric.failure_rate,
                "runtime_seconds_mean": metric.runtime_seconds_mean,
                "runtime_seconds_min": metric.runtime_seconds_min,
                "runtime_seconds_max": metric.runtime_seconds_max,
                "evaluation_count": metric.evaluation_count,
                "score_overall_mean": (
                    ""
                    if metric.score_overall_mean is None
                    else metric.score_overall_mean
                ),
            }
            for name in component_names:
                row[f"component:{name}"] = metric.component_means.get(
                    name, ""
                )
            writer.writerow(row)

    workflow_dir = root / "workflows"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    for snapshot in result.workflows:
        (workflow_dir / f"{_safe_name(snapshot.key)}.dsl").write_text(
            snapshot.canonical_dsl,
            encoding="utf-8",
        )
        (workflow_dir / f"{_safe_name(snapshot.key)}.json").write_text(
            json.dumps(
                snapshot.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return root


def load_experiment(path: str | Path) -> ExperimentResult:
    source = Path(path)
    if source.is_dir():
        source = source / "experiment.json"
    return ExperimentResult.from_json(source.read_text(encoding="utf-8"))

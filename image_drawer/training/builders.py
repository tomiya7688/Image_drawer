"""Trajectory / ExperimentResult から学習例を構築する。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from statistics import fmean
from typing import Any, Iterable, Mapping, Protocol

from image_drawer.core import Artifact, Part, Trajectory
from image_drawer.search import ExperimentResult, ObjectiveSpec
from image_drawer.training.models import (
    DatasetBuildConfig,
    PartCandidateFeatures,
    PartSelectorExample,
    TrainingDataset,
    WorkflowCandidateFeatures,
    WorkflowSelectorExample,
)
from image_drawer.training.splitting import assign_split, prompt_group_id


class PartRepositoryLike(Protocol):
    def get(self, part_id: str) -> Part:
        ...


def _stable_id(prefix: str, payload: object) -> str:
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    return f"{prefix}-{digest[:24]}"


def _artifact_by_id(trajectory: Trajectory) -> dict[str, Artifact]:
    return {artifact.id: artifact for artifact in trajectory.artifacts}


def _execution_output_artifact(
    trajectory: Trajectory,
    step_type: str,
) -> list[tuple[Any, Artifact]]:
    by_id = _artifact_by_id(trajectory)
    output: list[tuple[Any, Artifact]] = []
    for execution in trajectory.executions:
        if execution.step_type != step_type:
            continue
        if not execution.output_artifact_ids:
            continue
        artifact = by_id.get(execution.output_artifact_ids[0])
        if artifact is not None:
            output.append((execution, artifact))
    return output


def _part_set_payload(artifact: Artifact) -> tuple[list[str], list[float]]:
    payload = artifact.metadata.get("part_set")
    if isinstance(payload, dict):
        part_ids = payload.get("part_ids", [])
        scores = payload.get("retrieval_scores", [])
    else:
        part_ids = artifact.metadata.get("value", [])
        scores = artifact.metadata.get("retrieval_scores", [])
    if not isinstance(part_ids, list) or not all(
        isinstance(value, str) for value in part_ids
    ):
        return [], []
    if not isinstance(scores, list):
        scores = []
    normalized_scores: list[float] = []
    for index in range(len(part_ids)):
        if index < len(scores) and type(scores[index]) in (int, float):
            normalized_scores.append(float(scores[index]))
        else:
            normalized_scores.append(float("nan"))
    return list(part_ids), normalized_scores


def _selected_ids_for_retrieval(
    trajectory: Trajectory,
    retrieval_artifact: Artifact,
) -> list[str]:
    by_id = _artifact_by_id(trajectory)
    selected: list[str] = []
    for execution in trajectory.executions:
        if execution.step_type != "SELECT_PARTS":
            continue
        if retrieval_artifact.id not in execution.input_artifact_ids:
            continue
        if not execution.output_artifact_ids:
            continue
        output_artifact = by_id.get(execution.output_artifact_ids[0])
        if output_artifact is None:
            continue
        values = output_artifact.metadata.get("value", [])
        if isinstance(values, list):
            selected.extend(
                value for value in values if isinstance(value, str)
            )
    if selected:
        return list(dict.fromkeys(selected))

    # SELECT_PARTSを経由しないworkflowでもCOMPOSEがPart IDを保持していれば利用する。
    for artifact in trajectory.artifacts:
        values = artifact.metadata.get("selected_part_ids")
        if isinstance(values, list):
            selected.extend(
                value for value in values if isinstance(value, str)
            )
    return list(dict.fromkeys(selected))


def _part_features(
    part_id: str,
    retrieval_score: float | None,
    embedding_identity: Mapping[str, Any] | None,
    repository: PartRepositoryLike | None,
) -> PartCandidateFeatures:
    metadata: dict[str, Any] = {}
    embedding_refs: dict[str, str] = {}
    if repository is not None:
        try:
            part = repository.get(part_id)
        except KeyError:
            part = None
        if part is not None:
            metadata = {
                "category": part.category,
                "source_image_id": part.source_image_id,
                "bbox": list(part.bbox),
                "tags": list(part.tags),
                "attributes": dict(part.attributes),
                "quality": dict(part.quality),
                "extraction_method": part.extraction_method,
                "extraction_version": part.extraction_version,
                "metadata": dict(part.metadata),
            }
            embedding_refs = dict(part.embedding_refs)
    return PartCandidateFeatures(
        part_id=part_id,
        retrieval_score=retrieval_score,
        part_metadata=metadata,
        embedding_refs=embedding_refs,
        embedding_identity=dict(embedding_identity or {}),
    )


def build_part_selector_dataset(
    trajectories: Iterable[Trajectory],
    config: DatasetBuildConfig,
    *,
    repository: PartRepositoryLike | None = None,
    generated_at: str | None = None,
) -> TrainingDataset:
    trajectories = list(trajectories)
    examples: list[PartSelectorExample] = []
    used_trajectory_ids: list[str] = []
    observed_embeddings: dict[str, dict[str, Any]] = {}
    evaluator_versions: set[tuple[str, str | None]] = set()

    for trajectory in trajectories:
        group_id = prompt_group_id(trajectory.prompt)
        split = assign_split(group_id, config.split)
        trajectory_used = False

        for score in trajectory.evaluations:
            evaluator_versions.add(
                (score.evaluator, score.evaluator_version)
            )

        identities = trajectory.metadata.get("execution_identity", {})
        if not isinstance(identities, dict):
            identities = {}

        for execution, retrieval_artifact in _execution_output_artifact(
            trajectory,
            "RETRIEVE_PARTS",
        ):
            part_ids, raw_scores = _part_set_payload(retrieval_artifact)
            if not part_ids:
                continue
            selected_ids = set(
                _selected_ids_for_retrieval(
                    trajectory,
                    retrieval_artifact,
                )
            )
            selected_ids &= set(part_ids)
            rejected_ids = [
                part_id for part_id in part_ids if part_id not in selected_ids
            ]
            if not selected_ids or not rejected_ids:
                continue

            identity_group = identities.get(execution.step_id, {})
            embedding_identity = (
                identity_group.get("embedding", {})
                if isinstance(identity_group, dict)
                else {}
            )
            if isinstance(embedding_identity, dict) and embedding_identity:
                key = json.dumps(
                    embedding_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                observed_embeddings[key] = dict(embedding_identity)

            score_map = {
                part_id: (
                    None
                    if index >= len(raw_scores)
                    or raw_scores[index] != raw_scores[index]
                    else raw_scores[index]
                )
                for index, part_id in enumerate(part_ids)
            }
            feature_map = {
                part_id: _part_features(
                    part_id,
                    score_map[part_id],
                    embedding_identity
                    if isinstance(embedding_identity, dict)
                    else {},
                    repository,
                )
                for part_id in part_ids
            }
            context = {
                "prompt": trajectory.prompt,
                "workflow_id": trajectory.workflow_id,
                "workflow_version": trajectory.workflow_version,
                "retrieval_step_id": execution.step_id,
                "retrieval_artifact_id": retrieval_artifact.id,
                "query_id": retrieval_artifact.metadata.get("query_id"),
                "category": retrieval_artifact.metadata.get("category"),
                "retrieval_filters": (
                    retrieval_artifact.metadata.get("part_set", {})
                    .get("metadata", {})
                    .get("filters", {})
                    if isinstance(
                        retrieval_artifact.metadata.get("part_set"), dict
                    )
                    else {}
                ),
            }

            if config.part_example_mode in {"pointwise", "both"}:
                for part_id in part_ids:
                    label = {
                        "selected": part_id in selected_ids,
                        "value": 1 if part_id in selected_ids else 0,
                    }
                    payload = {
                        "trajectory_id": trajectory.id,
                        "retrieval_artifact_id": retrieval_artifact.id,
                        "part_id": part_id,
                        "kind": "pointwise",
                    }
                    examples.append(
                        PartSelectorExample(
                            id=_stable_id("part-example", payload),
                            split=split,
                            group_id=group_id,
                            example_kind="pointwise",
                            prompt=trajectory.prompt,
                            context=dict(context),
                            candidates=[feature_map[part_id]],
                            label=label,
                            source_trajectory_ids=[trajectory.id],
                            metadata={
                                "seed": trajectory.input_metadata.get("seed"),
                            },
                        )
                    )

            if config.part_example_mode in {"pairwise", "both"}:
                pairs = [
                    (positive, negative)
                    for positive in sorted(selected_ids)
                    for negative in sorted(rejected_ids)
                ][: config.max_pairs_per_group]
                for positive, negative in pairs:
                    payload = {
                        "trajectory_id": trajectory.id,
                        "retrieval_artifact_id": retrieval_artifact.id,
                        "preferred": positive,
                        "rejected": negative,
                        "kind": "pairwise",
                    }
                    examples.append(
                        PartSelectorExample(
                            id=_stable_id("part-example", payload),
                            split=split,
                            group_id=group_id,
                            example_kind="pairwise",
                            prompt=trajectory.prompt,
                            context=dict(context),
                            candidates=[
                                feature_map[positive],
                                feature_map[negative],
                            ],
                            label={
                                "preferred_part_id": positive,
                                "rejected_part_id": negative,
                            },
                            source_trajectory_ids=[trajectory.id],
                            metadata={
                                "seed": trajectory.input_metadata.get("seed"),
                            },
                        )
                    )
            trajectory_used = True

        if trajectory_used:
            used_trajectory_ids.append(trajectory.id)

    generated = generated_at or datetime.now(timezone.utc).isoformat()
    dataset_id = _stable_id(
        "dataset",
        {
            "task": "part_selector",
            "build_config": config.to_dict(),
            "source_ids": sorted(set(used_trajectory_ids)),
            "example_ids": [example.id for example in examples],
        },
    )
    return TrainingDataset(
        id=dataset_id,
        task="part_selector",
        build_config=config,
        examples=[example.to_dict() for example in examples],
        source_ids=sorted(set(used_trajectory_ids)),
        metadata={
            "generated_at": generated,
            "source_trajectory_ids": sorted(set(used_trajectory_ids)),
            "embedding_versions": [
                payload
                for _, payload in sorted(observed_embeddings.items())
            ],
            "evaluator_versions": [
                {
                    "name": name,
                    "version": version,
                }
                for name, version in sorted(
                    evaluator_versions,
                    key=lambda item: (item[0], item[1] or ""),
                )
            ],
            "split_policy": {
                "grouping": "normalized_prompt",
                **config.split.to_dict(),
            },
            "example_mode": config.part_example_mode,
        },
    )


def _mean_run_metric(run, metric_name: str) -> float | None:
    if metric_name == "failure_rate":
        return 1.0 if run.status == "failure" else 0.0
    if metric_name == "runtime_seconds_mean":
        return float(run.runtime_seconds)
    trajectory = run.trajectory
    if trajectory is None:
        return None
    if metric_name == "score_overall_mean":
        values = [
            float(score.overall)
            for score in trajectory.evaluations
            if score.overall is not None
        ]
        return fmean(values) if values else None
    if metric_name.startswith("component:"):
        component = metric_name.split(":", 1)[1]
        values = [
            float(score.components[component])
            for score in trajectory.evaluations
            if component in score.components
        ]
        return fmean(values) if values else None
    raise ValueError(f"unsupported workflow objective metric: {metric_name}")


def _workflow_run_objective(
    run,
    objective: ObjectiveSpec,
) -> tuple[float | None, dict[str, float]]:
    components: dict[str, float] = {}
    for name in objective.weights:
        value = _mean_run_metric(run, name)
        if value is None:
            return None, components
        components[name] = value
    aggregate = sum(
        components[name] * float(weight)
        for name, weight in objective.weights.items()
    )
    return aggregate, components


def build_workflow_selector_dataset(
    experiment: ExperimentResult,
    config: DatasetBuildConfig,
    objective: ObjectiveSpec,
    *,
    generated_at: str | None = None,
) -> TrainingDataset:
    snapshots = {snapshot.key: snapshot for snapshot in experiment.workflows}
    grouped: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for run in experiment.runs:
        grouped[(run.prompt_case_id, run.repeat_index)].append(run)

    examples: list[WorkflowSelectorExample] = []
    source_run_ids: set[str] = set()
    source_trajectory_ids: set[str] = set()

    for (prompt_case_id, repeat_index), runs in sorted(grouped.items()):
        prompts = {run.prompt for run in runs}
        if len(prompts) != 1:
            raise ValueError(
                "workflow comparison group contains inconsistent prompts"
            )
        prompt = next(iter(prompts))
        group_id = prompt_group_id(prompt)
        split = assign_split(group_id, config.split)
        candidates: list[WorkflowCandidateFeatures] = []
        scoreable: list[tuple[float, str]] = []

        for run in sorted(runs, key=lambda item: item.workflow_key):
            snapshot = snapshots.get(run.workflow_key)
            if snapshot is None:
                raise ValueError(
                    f"missing WorkflowSnapshot for {run.workflow_key}"
                )
            value, components = _workflow_run_objective(run, objective)
            candidates.append(
                WorkflowCandidateFeatures(
                    workflow_key=run.workflow_key,
                    workflow_id=snapshot.workflow_id,
                    workflow_version=snapshot.workflow_version,
                    status=run.status,
                    objective_value=value,
                    objective_components=components,
                    workflow_metadata={
                        "content_hash": snapshot.content_hash,
                        "backends": dict(snapshot.backends),
                        "metadata": dict(snapshot.metadata),
                    },
                )
            )
            source_run_ids.add(run.id)
            if run.trajectory is not None:
                source_trajectory_ids.add(run.trajectory.id)
            if value is not None:
                scoreable.append((value, run.workflow_key))

        if len(candidates) < 2 or not scoreable:
            continue
        scoreable.sort(key=lambda item: (-item[0], item[1]))
        selected_key = scoreable[0][1]
        example_id = _stable_id(
            "workflow-example",
            {
                "experiment_id": experiment.id,
                "prompt_case_id": prompt_case_id,
                "repeat_index": repeat_index,
                "workflow_keys": [
                    candidate.workflow_key for candidate in candidates
                ],
                "objective": objective.to_dict(),
            },
        )
        examples.append(
            WorkflowSelectorExample(
                id=example_id,
                split=split,
                group_id=group_id,
                prompt=prompt,
                context={
                    "experiment_id": experiment.id,
                    "prompt_case_id": prompt_case_id,
                    "repeat_index": repeat_index,
                    "prompt_metadata": next(
                        (
                            dict(run.metadata.get("prompt_metadata", {}))
                            for run in runs
                            if isinstance(
                                run.metadata.get("prompt_metadata"), dict
                            )
                        ),
                        {},
                    ),
                },
                candidates=candidates,
                label={
                    "selected_workflow_key": selected_key,
                    "objective": objective.to_dict(),
                },
                source_run_ids=[run.id for run in runs],
                source_trajectory_ids=[
                    run.trajectory.id
                    for run in runs
                    if run.trajectory is not None
                ],
                metadata={
                    "candidate_count": len(candidates),
                },
            )
        )

    generated = generated_at or datetime.now(timezone.utc).isoformat()
    dataset_id = _stable_id(
        "dataset",
        {
            "task": "workflow_selector",
            "build_config": config.to_dict(),
            "experiment_id": experiment.id,
            "objective": objective.to_dict(),
            "example_ids": [example.id for example in examples],
        },
    )
    return TrainingDataset(
        id=dataset_id,
        task="workflow_selector",
        build_config=config,
        examples=[example.to_dict() for example in examples],
        source_ids=sorted(source_run_ids),
        metadata={
            "generated_at": generated,
            "source_experiment_id": experiment.id,
            "source_run_ids": sorted(source_run_ids),
            "source_trajectory_ids": sorted(source_trajectory_ids),
            "objective": objective.to_dict(),
            "experiment_environment": dict(
                experiment.metadata.get("environment", {})
            ),
            "observed_versions": dict(
                experiment.metadata.get("observed_versions", {})
            ),
            "split_policy": {
                "grouping": "normalized_prompt",
                **config.split.to_dict(),
            },
        },
    )

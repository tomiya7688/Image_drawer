"""PyTorchによるbaseline WorkflowSelector ranker。"""

from __future__ import annotations

import json
import math
import random
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from image_drawer.core import SerializableModel
from image_drawer.training.models import TrainingDataset
from image_drawer.training.results import load_training_dataset
from image_drawer.training.workflow_selector_features import (
    WorkflowSelectorFeatureSchema,
    encode_workflow_selector_candidate,
    workflow_static_record,
)


Metadata = dict[str, Any]


@dataclass(slots=True)
class WorkflowSelectorTrainConfig(SerializableModel):
    id: str
    dataset_path: str
    output_dir: str
    seed: int = 0
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    hidden_dimensions: list[int] = field(default_factory=lambda: [64, 32])
    text_dimensions: int = 32
    top_k: int = 2
    tie_tolerance: float = 0.0
    selection_metric: str = "top1_accuracy"
    final_quality_metric: str = "component:quality"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("training config id must not be empty")
        if not self.dataset_path:
            raise ValueError("dataset_path must not be empty")
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        if type(self.epochs) is not int or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        if type(self.batch_size) is not int or self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not self.hidden_dimensions or any(
            type(value) is not int or value <= 0
            for value in self.hidden_dimensions
        ):
            raise ValueError("hidden_dimensions must contain positive integers")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.tie_tolerance < 0:
            raise ValueError("tie_tolerance must be non-negative")
        if self.selection_metric not in {
            "top1_accuracy",
            "top_k_accuracy",
            "mean_objective_regret",
        }:
            raise ValueError("unsupported selection_metric")
        if not self.final_quality_metric:
            raise ValueError("final_quality_metric must not be empty")


@dataclass(slots=True)
class WorkflowSelectorMetrics(SerializableModel):
    split: str
    example_count: int
    scoreable_example_count: int
    tied_example_count: int
    top1_accuracy: float | None
    top_k_accuracy: float | None
    mean_objective_regret: float | None
    selected_objective_mean: float | None
    best_objective_mean: float | None
    downstream_final_quality_mean: float | None = None


@dataclass(slots=True)
class WorkflowSelectorTrainingResult(SerializableModel):
    config: WorkflowSelectorTrainConfig
    dataset_id: str
    checkpoint_path: str
    feature_schema: WorkflowSelectorFeatureSchema
    best_epoch: int
    best_validation_metrics: WorkflowSelectorMetrics
    test_metrics: WorkflowSelectorMetrics | None
    metrics_log_path: str
    metadata: Metadata = field(default_factory=dict)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "WorkflowSelector training requires image-drawer[training]"
        ) from exc
    return torch


def _model_factory(torch, input_dimensions: int, hidden_dimensions: Sequence[int]):
    layers = []
    previous = input_dimensions
    for hidden in hidden_dimensions:
        layers.append(torch.nn.Linear(previous, hidden))
        layers.append(torch.nn.ReLU())
        previous = hidden
    layers.append(torch.nn.Linear(previous, 1))
    return torch.nn.Sequential(*layers)


def _set_deterministic(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _split_examples(
    dataset: TrainingDataset,
    split: str,
) -> list[dict[str, Any]]:
    return [
        dict(example)
        for example in dataset.examples
        if example.get("split") == split
        and isinstance(example.get("candidates"), list)
    ]


def _scoreable_candidates(
    example: Mapping[str, Any],
) -> list[dict[str, Any]]:
    candidates = example.get("candidates", [])
    output: list[dict[str, Any]] = []
    if not isinstance(candidates, list):
        return output
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        value = candidate.get("objective_value")
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            continue
        workflow_key = candidate.get("workflow_key")
        if not isinstance(workflow_key, str) or not workflow_key:
            continue
        output.append(dict(candidate))
    return output


def _positive_keys(
    candidates: Sequence[Mapping[str, Any]],
    tie_tolerance: float,
) -> set[str]:
    if not candidates:
        return set()
    best = max(float(item["objective_value"]) for item in candidates)
    return {
        str(item["workflow_key"])
        for item in candidates
        if best - float(item["objective_value"]) <= tie_tolerance
    }


def _pairwise_instances(
    examples: Sequence[Mapping[str, Any]],
    schema: WorkflowSelectorFeatureSchema,
    tie_tolerance: float,
) -> list[tuple[list[float], list[float]]]:
    pairs: list[tuple[list[float], list[float]]] = []
    for example in examples:
        candidates = _scoreable_candidates(example)
        positives = _positive_keys(candidates, tie_tolerance)
        if not positives:
            continue
        prompt = str(example.get("prompt", ""))
        context = example.get("context", {})
        context = dict(context) if isinstance(context, Mapping) else {}
        by_key = {
            str(candidate["workflow_key"]): candidate
            for candidate in candidates
        }
        negatives = sorted(set(by_key) - positives)
        for positive in sorted(positives):
            for negative in negatives:
                pairs.append(
                    (
                        encode_workflow_selector_candidate(
                            prompt,
                            by_key[positive],
                            schema,
                            context=context,
                        ),
                        encode_workflow_selector_candidate(
                            prompt,
                            by_key[negative],
                            schema,
                            context=context,
                        ),
                    )
                )
    return pairs


def _batches(
    items: Sequence[Any],
    batch_size: int,
    seed: int,
) -> list[list[Any]]:
    indexes = list(range(len(items)))
    random.Random(seed).shuffle(indexes)
    return [
        [items[index] for index in indexes[start : start + batch_size]]
        for start in range(0, len(indexes), batch_size)
    ]


def _quality_value(
    candidate: Mapping[str, Any],
    metric: str,
) -> float | None:
    if metric == "objective_value":
        value = candidate.get("objective_value")
        if type(value) in (int, float) and math.isfinite(float(value)):
            return float(value)
        return None
    components = candidate.get("objective_components", {})
    if not isinstance(components, Mapping):
        return None
    value = components.get(metric)
    if type(value) in (int, float) and math.isfinite(float(value)):
        return float(value)
    return None


def score_workflow_candidates(
    model,
    schema: WorkflowSelectorFeatureSchema,
    prompt: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    torch = _require_torch()
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not candidates:
        raise ValueError("at least one workflow candidate is required")
    features = [
        encode_workflow_selector_candidate(
            prompt,
            candidate,
            schema,
            context=context,
        )
        for candidate in candidates
    ]
    model.eval()
    with torch.no_grad():
        scores = model(
            torch.tensor(features, dtype=torch.float32)
        ).squeeze(-1).tolist()
    ranked = [
        {
            **workflow_static_record(candidate),
            "score": float(score),
        }
        for candidate, score in zip(candidates, scores)
    ]
    ranked.sort(
        key=lambda item: (-item["score"], item["workflow_key"])
    )
    return ranked


def evaluate_workflow_selector_model(
    model,
    dataset: TrainingDataset,
    schema: WorkflowSelectorFeatureSchema,
    *,
    split: str,
    top_k: int,
    tie_tolerance: float,
    final_quality_metric: str,
) -> WorkflowSelectorMetrics:
    examples = _split_examples(dataset, split)
    if not examples:
        return WorkflowSelectorMetrics(
            split=split,
            example_count=0,
            scoreable_example_count=0,
            tied_example_count=0,
            top1_accuracy=None,
            top_k_accuracy=None,
            mean_objective_regret=None,
            selected_objective_mean=None,
            best_objective_mean=None,
            downstream_final_quality_mean=None,
        )

    top1_hits = 0
    topk_hits = 0
    regrets: list[float] = []
    selected_values: list[float] = []
    best_values: list[float] = []
    downstream: list[float] = []
    scoreable_count = 0
    tied_count = 0

    for example in examples:
        candidates = _scoreable_candidates(example)
        if len(candidates) < 2:
            continue
        positives = _positive_keys(candidates, tie_tolerance)
        if not positives:
            continue
        if len(positives) > 1:
            tied_count += 1
        scoreable_count += 1

        prompt = str(example.get("prompt", ""))
        context = example.get("context", {})
        context = dict(context) if isinstance(context, Mapping) else {}
        ranking = score_workflow_candidates(
            model,
            schema,
            prompt,
            candidates,
            context=context,
        )
        ranked_keys = [item["workflow_key"] for item in ranking]
        selected_key = ranked_keys[0]
        if selected_key in positives:
            top1_hits += 1
        if positives.intersection(ranked_keys[: min(top_k, len(ranked_keys))]):
            topk_hits += 1

        by_key = {
            str(candidate["workflow_key"]): candidate
            for candidate in candidates
        }
        best_objective = max(
            float(candidate["objective_value"])
            for candidate in candidates
        )
        selected_objective = float(
            by_key[selected_key]["objective_value"]
        )
        regrets.append(best_objective - selected_objective)
        selected_values.append(selected_objective)
        best_values.append(best_objective)

        quality = _quality_value(
            by_key[selected_key],
            final_quality_metric,
        )
        if quality is not None:
            downstream.append(quality)

    return WorkflowSelectorMetrics(
        split=split,
        example_count=len(examples),
        scoreable_example_count=scoreable_count,
        tied_example_count=tied_count,
        top1_accuracy=(
            top1_hits / scoreable_count if scoreable_count else None
        ),
        top_k_accuracy=(
            topk_hits / scoreable_count if scoreable_count else None
        ),
        mean_objective_regret=(
            sum(regrets) / len(regrets) if regrets else None
        ),
        selected_objective_mean=(
            sum(selected_values) / len(selected_values)
            if selected_values
            else None
        ),
        best_objective_mean=(
            sum(best_values) / len(best_values)
            if best_values
            else None
        ),
        downstream_final_quality_mean=(
            sum(downstream) / len(downstream)
            if downstream
            else None
        ),
    )


def _selection_value(
    metrics: WorkflowSelectorMetrics,
    name: str,
) -> float:
    value = getattr(metrics, name)
    if value is None:
        return float("-inf")
    if name == "mean_objective_regret":
        return -float(value)
    return float(value)


def _known_workflows(dataset: TrainingDataset) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for example in dataset.examples:
        candidates = example.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            static = workflow_static_record(candidate)
            key = static["workflow_key"]
            if not key:
                continue
            current = unique.get(key)
            if current is not None and current != static:
                raise ValueError(
                    f"workflow key {key} has conflicting static metadata"
                )
            unique[key] = static
    return [unique[key] for key in sorted(unique)]


def _save_checkpoint(
    torch,
    path: Path,
    *,
    model,
    config: WorkflowSelectorTrainConfig,
    dataset: TrainingDataset,
    schema: WorkflowSelectorFeatureSchema,
    epoch: int,
    metrics: WorkflowSelectorMetrics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "workflow-selector-mlp",
            "model_state_dict": model.state_dict(),
            "input_dimensions": schema.total_dimensions,
            "hidden_dimensions": list(config.hidden_dimensions),
            "feature_schema": schema.to_dict(),
            "feature_schema_hash": schema.identity_hash,
            "dataset_id": dataset.id,
            "training_config": config.to_dict(),
            "label_objective": dataset.metadata.get("objective"),
            "known_workflows": _known_workflows(dataset),
            "best_epoch": epoch,
            "validation_metrics": metrics.to_dict(),
        },
        path,
    )


def load_workflow_selector_checkpoint(path: str | Path):
    torch = _require_torch()
    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    schema = WorkflowSelectorFeatureSchema.from_dict(
        checkpoint["feature_schema"]
    )
    if checkpoint.get("feature_schema_hash") != schema.identity_hash:
        raise ValueError("checkpoint feature schema hash mismatch")
    model = _model_factory(
        torch,
        int(checkpoint["input_dimensions"]),
        list(checkpoint["hidden_dimensions"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key != "model_state_dict"
    }
    return model, schema, metadata


def evaluate_workflow_selector_checkpoint(
    checkpoint_path: str | Path,
    dataset: TrainingDataset | str | Path,
    *,
    split: str = "test",
    top_k: int | None = None,
) -> WorkflowSelectorMetrics:
    if not isinstance(dataset, TrainingDataset):
        dataset = load_training_dataset(dataset)
    model, schema, metadata = load_workflow_selector_checkpoint(
        checkpoint_path
    )
    config = WorkflowSelectorTrainConfig.from_dict(
        metadata["training_config"]
    )
    return evaluate_workflow_selector_model(
        model,
        dataset,
        schema,
        split=split,
        top_k=top_k or config.top_k,
        tie_tolerance=config.tie_tolerance,
        final_quality_metric=config.final_quality_metric,
    )


def score_workflow_selector_checkpoint(
    checkpoint_path: str | Path,
    prompt: str,
    *,
    candidates: Sequence[Mapping[str, Any]] | None = None,
    context: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    model, schema, metadata = load_workflow_selector_checkpoint(
        checkpoint_path
    )
    active_candidates = (
        list(candidates)
        if candidates is not None
        else list(metadata.get("known_workflows", []))
    )
    return score_workflow_candidates(
        model,
        schema,
        prompt,
        active_candidates,
        context=context,
    )


def train_workflow_selector(
    dataset: TrainingDataset,
    config: WorkflowSelectorTrainConfig,
) -> WorkflowSelectorTrainingResult:
    if dataset.task != "workflow_selector":
        raise ValueError(
            "WorkflowSelector trainer requires workflow_selector dataset"
        )

    torch = _require_torch()
    _set_deterministic(torch, config.seed)
    schema = WorkflowSelectorFeatureSchema(
        text_dimensions=config.text_dimensions
    )
    train_examples = _split_examples(dataset, "train")
    validation_examples = _split_examples(dataset, "validation")
    if not train_examples:
        raise ValueError("training split has no WorkflowSelector examples")
    if not validation_examples:
        raise ValueError("validation split has no WorkflowSelector examples")

    train_instances = _pairwise_instances(
        train_examples,
        schema,
        config.tie_tolerance,
    )
    if not train_instances:
        raise ValueError("training split produced no ranking pairs")

    model = _model_factory(
        torch,
        schema.total_dimensions,
        config.hidden_dimensions,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"
    log_path = output_dir / "metrics.jsonl"
    if log_path.exists():
        log_path.unlink()

    best_epoch = 0
    best_metrics: WorkflowSelectorMetrics | None = None
    best_value = float("-inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch in _batches(
            train_instances,
            config.batch_size,
            config.seed + epoch,
        ):
            optimizer.zero_grad()
            positive = torch.tensor(
                [item[0] for item in batch],
                dtype=torch.float32,
            )
            negative = torch.tensor(
                [item[1] for item in batch],
                dtype=torch.float32,
            )
            positive_scores = model(positive).squeeze(-1)
            negative_scores = model(negative).squeeze(-1)
            loss = torch.nn.functional.softplus(
                -(positive_scores - negative_scores)
            ).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = evaluate_workflow_selector_model(
            model,
            dataset,
            schema,
            split="validation",
            top_k=config.top_k,
            tie_tolerance=config.tie_tolerance,
            final_quality_metric=config.final_quality_metric,
        )
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": sum(losses) / len(losses),
                        "validation": validation.to_dict(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )

        selection = _selection_value(
            validation,
            config.selection_metric,
        )
        if selection > best_value:
            best_value = selection
            best_epoch = epoch
            best_metrics = validation
            _save_checkpoint(
                torch,
                checkpoint_path,
                model=model,
                config=config,
                dataset=dataset,
                schema=schema,
                epoch=epoch,
                metrics=validation,
            )

    if best_metrics is None or not checkpoint_path.is_file():
        raise RuntimeError("training did not produce a checkpoint")

    test_examples = _split_examples(dataset, "test")
    test_metrics = (
        evaluate_workflow_selector_checkpoint(
            checkpoint_path,
            dataset,
            split="test",
            top_k=config.top_k,
        )
        if test_examples
        else None
    )
    result = WorkflowSelectorTrainingResult(
        config=config,
        dataset_id=dataset.id,
        checkpoint_path=str(checkpoint_path),
        feature_schema=schema,
        best_epoch=best_epoch,
        best_validation_metrics=best_metrics,
        test_metrics=test_metrics,
        metrics_log_path=str(log_path),
        metadata={
            "torch_version": torch.__version__,
            "deterministic_algorithms": True,
            "training_example_count": len(train_examples),
            "validation_example_count": len(validation_examples),
            "test_example_count": len(test_examples),
            "known_workflow_count": len(_known_workflows(dataset)),
            "label_objective": dataset.metadata.get("objective"),
        },
    )
    (output_dir / "training_result.json").write_text(
        result.to_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def load_workflow_selector_train_config(
    path: str | Path,
) -> WorkflowSelectorTrainConfig:
    source = Path(path)
    if source.suffix.lower() == ".toml":
        payload = tomllib.loads(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("training config must be .toml or .json")
    if "training" in payload:
        payload = payload["training"]
    if not isinstance(payload, Mapping):
        raise ValueError("training config must contain an object")
    normalized = dict(payload)
    base = source.parent
    for key in ("dataset_path", "output_dir"):
        value = normalized.get(key)
        if isinstance(value, str) and not Path(value).is_absolute():
            normalized[key] = str((base / value).resolve())
    return WorkflowSelectorTrainConfig.from_dict(normalized)


def train_workflow_selector_from_config(
    path: str | Path,
) -> WorkflowSelectorTrainingResult:
    config = load_workflow_selector_train_config(path)
    dataset = load_training_dataset(config.dataset_path)
    return train_workflow_selector(dataset, config)

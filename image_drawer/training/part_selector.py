"""PyTorchによるbaseline PartSelector ranker。"""

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
from image_drawer.training.part_selector_features import (
    PartSelectorFeatureSchema,
    encode_part_selector_candidate,
)
from image_drawer.training.results import load_training_dataset


Metadata = dict[str, Any]


@dataclass(slots=True)
class PartSelectorTrainConfig(SerializableModel):
    id: str
    dataset_path: str
    output_dir: str
    seed: int = 0
    epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 0.01
    weight_decay: float = 0.0
    hidden_dimensions: list[int] = field(default_factory=lambda: [64, 32])
    objective: str = "pairwise"
    text_dimensions: int = 32
    top_k: int = 3
    selection_metric: str = "mrr"

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
        if self.objective not in {"pairwise", "pointwise"}:
            raise ValueError("objective must be pairwise or pointwise")
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        if self.selection_metric not in {
            "mrr",
            "ndcg",
            "pairwise_accuracy",
            "top1_accuracy",
            "top_k_recall",
        }:
            raise ValueError("unsupported selection_metric")


@dataclass(slots=True)
class PartSelectorMetrics(SerializableModel):
    split: str
    group_count: int
    single_preferred_group_count: int
    pair_count: int
    top1_accuracy: float | None
    top_k_recall: float | None
    pairwise_accuracy: float | None
    mrr: float | None
    ndcg: float | None
    downstream_evaluator_score_mean: float | None = None


@dataclass(slots=True)
class PartSelectorTrainingResult(SerializableModel):
    config: PartSelectorTrainConfig
    dataset_id: str
    checkpoint_path: str
    feature_schema: PartSelectorFeatureSchema
    best_epoch: int
    best_validation_metrics: PartSelectorMetrics
    test_metrics: PartSelectorMetrics | None
    metrics_log_path: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(slots=True)
class _RankingGroup:
    key: str
    prompt: str
    example_context: dict[str, Any]
    candidates: dict[str, dict[str, Any]]
    positives: set[str]
    negatives: set[str]


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PartSelector training requires image-drawer[training]"
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


def _example_group_key(example: Mapping[str, Any]) -> str:
    context = example.get("context", {})
    retrieval = (
        context.get("retrieval_artifact_id")
        if isinstance(context, Mapping)
        else None
    )
    return f"{example.get('group_id', '')}:{retrieval or 'retrieval'}"


def _groups(dataset: TrainingDataset, split: str) -> list[_RankingGroup]:
    grouped: dict[str, _RankingGroup] = {}
    for raw in dataset.examples:
        if raw.get("split") != split:
            continue
        kind = raw.get("example_kind")
        if kind not in {"pointwise", "pairwise"}:
            continue
        key = _example_group_key(raw)
        group = grouped.get(key)
        if group is None:
            context = raw.get("context", {})
            group = _RankingGroup(
                key=key,
                prompt=str(raw.get("prompt", "")),
                example_context=(
                    dict(context) if isinstance(context, Mapping) else {}
                ),
                candidates={},
                positives=set(),
                negatives=set(),
            )
            grouped[key] = group

        candidates = raw.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            part_id = candidate.get("part_id")
            if isinstance(part_id, str):
                group.candidates.setdefault(part_id, dict(candidate))

        label = raw.get("label", {})
        if not isinstance(label, Mapping):
            continue
        if kind == "pointwise" and len(candidates) == 1:
            part_id = candidates[0].get("part_id")
            if isinstance(part_id, str):
                if bool(label.get("selected")):
                    group.positives.add(part_id)
                    group.negatives.discard(part_id)
                elif part_id not in group.positives:
                    group.negatives.add(part_id)
        elif kind == "pairwise":
            preferred = label.get("preferred_part_id")
            rejected = label.get("rejected_part_id")
            if isinstance(preferred, str):
                group.positives.add(preferred)
                group.negatives.discard(preferred)
            if (
                isinstance(rejected, str)
                and rejected not in group.positives
            ):
                group.negatives.add(rejected)

    return [
        group
        for _, group in sorted(grouped.items())
        if group.positives and group.negatives
    ]


def _feature_example(group: _RankingGroup) -> dict[str, Any]:
    return {
        "prompt": group.prompt,
        "context": group.example_context,
    }


def _pairwise_instances(
    groups: Sequence[_RankingGroup],
    schema: PartSelectorFeatureSchema,
) -> list[tuple[list[float], list[float]]]:
    pairs: list[tuple[list[float], list[float]]] = []
    for group in groups:
        example = _feature_example(group)
        for positive in sorted(group.positives):
            if positive not in group.candidates:
                continue
            for negative in sorted(group.negatives):
                if negative not in group.candidates:
                    continue
                pairs.append(
                    (
                        encode_part_selector_candidate(
                            example,
                            group.candidates[positive],
                            schema,
                        ),
                        encode_part_selector_candidate(
                            example,
                            group.candidates[negative],
                            schema,
                        ),
                    )
                )
    return pairs


def _pointwise_instances(
    groups: Sequence[_RankingGroup],
    schema: PartSelectorFeatureSchema,
) -> list[tuple[list[float], float]]:
    output: list[tuple[list[float], float]] = []
    for group in groups:
        example = _feature_example(group)
        for part_id, candidate in sorted(group.candidates.items()):
            if part_id not in group.positives and part_id not in group.negatives:
                continue
            output.append(
                (
                    encode_part_selector_candidate(
                        example,
                        candidate,
                        schema,
                    ),
                    1.0 if part_id in group.positives else 0.0,
                )
            )
    return output


def _batches(items: Sequence[Any], batch_size: int, seed: int) -> list[list[Any]]:
    indexes = list(range(len(items)))
    random.Random(seed).shuffle(indexes)
    return [
        [items[index] for index in indexes[start : start + batch_size]]
        for start in range(0, len(indexes), batch_size)
    ]


def _downstream_score(candidate: Mapping[str, Any]) -> float | None:
    metadata = candidate.get("part_metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    quality = metadata.get("quality", {})
    if not isinstance(quality, Mapping):
        return None
    for key in (
        "downstream_evaluator_score",
        "evaluator_score",
        "overall",
    ):
        value = quality.get(key)
        if type(value) in (int, float) and math.isfinite(float(value)):
            return float(value)
    return None


def evaluate_part_selector_model(
    model,
    dataset: TrainingDataset,
    schema: PartSelectorFeatureSchema,
    *,
    split: str,
    top_k: int,
) -> PartSelectorMetrics:
    torch = _require_torch()
    groups = _groups(dataset, split)
    if not groups:
        return PartSelectorMetrics(
            split=split,
            group_count=0,
            single_preferred_group_count=0,
            pair_count=0,
            top1_accuracy=None,
            top_k_recall=None,
            pairwise_accuracy=None,
            mrr=None,
            ndcg=None,
            downstream_evaluator_score_mean=None,
        )

    model.eval()
    top1_hits = 0
    single_count = 0
    recalls: list[float] = []
    pair_hits = 0
    pair_count = 0
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    downstream: list[float] = []

    with torch.no_grad():
        for group in groups:
            example = _feature_example(group)
            candidate_ids = sorted(group.candidates)
            features = [
                encode_part_selector_candidate(
                    example,
                    group.candidates[part_id],
                    schema,
                )
                for part_id in candidate_ids
            ]
            scores_tensor = model(
                torch.tensor(features, dtype=torch.float32)
            ).squeeze(-1)
            scores = {
                part_id: float(score)
                for part_id, score in zip(
                    candidate_ids, scores_tensor.tolist()
                )
            }
            ranking = sorted(
                candidate_ids,
                key=lambda part_id: (-scores[part_id], part_id),
            )

            if len(group.positives) == 1:
                single_count += 1
                if ranking[0] in group.positives:
                    top1_hits += 1

            effective_k = min(top_k, len(ranking))
            retrieved = set(ranking[:effective_k])
            recalls.append(
                len(retrieved & group.positives)
                / len(group.positives)
            )

            first_positive_rank = next(
                (
                    index + 1
                    for index, part_id in enumerate(ranking)
                    if part_id in group.positives
                ),
                None,
            )
            if first_positive_rank is not None:
                reciprocal_ranks.append(1.0 / first_positive_rank)

            dcg = 0.0
            for index, part_id in enumerate(ranking):
                if part_id in group.positives:
                    dcg += 1.0 / math.log2(index + 2.0)
            ideal_hits = min(len(group.positives), len(ranking))
            idcg = sum(
                1.0 / math.log2(index + 2.0)
                for index in range(ideal_hits)
            )
            ndcgs.append(dcg / idcg if idcg else 0.0)

            for positive in group.positives:
                for negative in group.negatives:
                    if positive not in scores or negative not in scores:
                        continue
                    pair_count += 1
                    if scores[positive] > scores[negative]:
                        pair_hits += 1

            selected_score = _downstream_score(
                group.candidates[ranking[0]]
            )
            if selected_score is not None:
                downstream.append(selected_score)

    return PartSelectorMetrics(
        split=split,
        group_count=len(groups),
        single_preferred_group_count=single_count,
        pair_count=pair_count,
        top1_accuracy=(
            top1_hits / single_count if single_count else None
        ),
        top_k_recall=sum(recalls) / len(recalls) if recalls else None,
        pairwise_accuracy=(
            pair_hits / pair_count if pair_count else None
        ),
        mrr=(
            sum(reciprocal_ranks) / len(reciprocal_ranks)
            if reciprocal_ranks
            else None
        ),
        ndcg=sum(ndcgs) / len(ndcgs) if ndcgs else None,
        downstream_evaluator_score_mean=(
            sum(downstream) / len(downstream)
            if downstream
            else None
        ),
    )


def _selection_value(metrics: PartSelectorMetrics, name: str) -> float:
    value = getattr(metrics, name)
    if value is None:
        return float("-inf")
    return float(value)


def _set_deterministic(torch, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _save_checkpoint(
    torch,
    path: Path,
    *,
    model,
    config: PartSelectorTrainConfig,
    dataset: TrainingDataset,
    schema: PartSelectorFeatureSchema,
    epoch: int,
    metrics: PartSelectorMetrics,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "model_type": "part-selector-mlp",
            "model_state_dict": model.state_dict(),
            "input_dimensions": schema.total_dimensions,
            "hidden_dimensions": list(config.hidden_dimensions),
            "feature_schema": schema.to_dict(),
            "feature_schema_hash": schema.identity_hash,
            "dataset_id": dataset.id,
            "training_config": config.to_dict(),
            "best_epoch": epoch,
            "validation_metrics": metrics.to_dict(),
        },
        path,
    )


def load_part_selector_checkpoint(path: str | Path):
    """checkpointをCPUへ安全ロードしてmodel/schema/metadataを返す。"""
    torch = _require_torch()
    checkpoint = torch.load(
        Path(path),
        map_location="cpu",
        weights_only=True,
    )
    schema = PartSelectorFeatureSchema.from_dict(
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


def evaluate_part_selector_checkpoint(
    checkpoint_path: str | Path,
    dataset: TrainingDataset | str | Path,
    *,
    split: str = "test",
    top_k: int | None = None,
) -> PartSelectorMetrics:
    if not isinstance(dataset, TrainingDataset):
        dataset = load_training_dataset(dataset)
    model, schema, metadata = load_part_selector_checkpoint(
        checkpoint_path
    )
    config = PartSelectorTrainConfig.from_dict(
        metadata["training_config"]
    )
    return evaluate_part_selector_model(
        model,
        dataset,
        schema,
        split=split,
        top_k=top_k or config.top_k,
    )


def train_part_selector(
    dataset: TrainingDataset,
    config: PartSelectorTrainConfig,
) -> PartSelectorTrainingResult:
    if dataset.task != "part_selector":
        raise ValueError("PartSelector trainer requires part_selector dataset")

    torch = _require_torch()
    _set_deterministic(torch, config.seed)
    schema = PartSelectorFeatureSchema(
        text_dimensions=config.text_dimensions
    )
    train_groups = _groups(dataset, "train")
    validation_groups = _groups(dataset, "validation")
    if not train_groups:
        raise ValueError("training split has no rankable Part groups")
    if not validation_groups:
        raise ValueError("validation split has no rankable Part groups")

    if config.objective == "pairwise":
        train_instances = _pairwise_instances(train_groups, schema)
    else:
        train_instances = _pointwise_instances(train_groups, schema)
    if not train_instances:
        raise ValueError("training split produced no training instances")

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
    best_metrics: PartSelectorMetrics | None = None
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
            if config.objective == "pairwise":
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
            else:
                features = torch.tensor(
                    [item[0] for item in batch],
                    dtype=torch.float32,
                )
                labels = torch.tensor(
                    [item[1] for item in batch],
                    dtype=torch.float32,
                )
                logits = model(features).squeeze(-1)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    labels,
                )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation = evaluate_part_selector_model(
            model,
            dataset,
            schema,
            split="validation",
            top_k=config.top_k,
        )
        log_record = {
            "epoch": epoch,
            "train_loss": sum(losses) / len(losses),
            "validation": validation.to_dict(),
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    log_record,
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

    test_groups = _groups(dataset, "test")
    test_metrics = (
        evaluate_part_selector_checkpoint(
            checkpoint_path,
            dataset,
            split="test",
            top_k=config.top_k,
        )
        if test_groups
        else None
    )
    result = PartSelectorTrainingResult(
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
            "training_group_count": len(train_groups),
            "validation_group_count": len(validation_groups),
            "test_group_count": len(test_groups),
        },
    )
    (output_dir / "training_result.json").write_text(
        result.to_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def load_part_selector_train_config(
    path: str | Path,
) -> PartSelectorTrainConfig:
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
    return PartSelectorTrainConfig.from_dict(normalized)


def train_part_selector_from_config(
    path: str | Path,
) -> PartSelectorTrainingResult:
    config = load_part_selector_train_config(path)
    dataset = load_training_dataset(config.dataset_path)
    return train_part_selector(dataset, config)

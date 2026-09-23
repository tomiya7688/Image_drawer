"""Selection policy over structured evaluator-specific ScoreSets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from image_drawer.core import Score, ScoreSet


@dataclass(frozen=True, slots=True)
class SelectionResult:
    selected_artifact_ids: tuple[str, ...]
    ranking: tuple[str, ...]
    aggregate_scores: dict[str, float]
    strategy: str
    weights: dict[str, float]


def parse_component_weights(values: Sequence[str]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            raise ValueError(
                "weights must contain strings in component=value form"
            )
        component, raw_weight = value.split("=", 1)
        component = component.strip()
        if not component:
            raise ValueError("weight component name must not be empty")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(
                f"invalid weight for component {component}: {raw_weight}"
            ) from exc
        if component in weights and weights[component] != weight:
            raise ValueError(f"conflicting weight for component {component}")
        weights[component] = weight
    return weights


def _overall(score: Score) -> float:
    if score.overall is None:
        raise ValueError(
            f"Score {score.id} has no overall value; use weighted selection"
        )
    return float(score.overall)


def _weighted(score: Score, weights: Mapping[str, float]) -> float:
    if not weights:
        raise ValueError("weighted selection requires component weights")
    missing = sorted(set(weights) - set(score.components))
    if missing:
        raise ValueError(
            f"Score {score.id} lacks weighted components: {', '.join(missing)}"
        )
    return sum(score.components[name] * weight for name, weight in weights.items())


def select_from_scores(
    score_set: ScoreSet,
    *,
    strategy: str,
    top: int,
    weights: Mapping[str, float] | None = None,
) -> SelectionResult:
    if type(top) is not int or top <= 0:
        raise ValueError("top must be a positive integer")
    if top > len(score_set.scores):
        raise ValueError(
            f"top={top} exceeds candidate count {len(score_set.scores)}"
        )
    if strategy not in {"best", "top_k", "weighted"}:
        raise ValueError(f"unsupported selection strategy: {strategy}")

    active_weights = dict(weights or {})
    aggregates: dict[str, float] = {}
    for candidate_id, score in zip(
        score_set.candidate_artifact_ids,
        score_set.scores,
    ):
        aggregate = (
            _weighted(score, active_weights)
            if strategy == "weighted"
            else _overall(score)
        )
        aggregates[candidate_id] = aggregate

    ranking = sorted(
        score_set.candidate_artifact_ids,
        key=lambda artifact_id: (-aggregates[artifact_id], artifact_id),
    )
    effective_top = 1 if strategy == "best" else top
    return SelectionResult(
        selected_artifact_ids=tuple(ranking[:effective_top]),
        ranking=tuple(ranking),
        aggregate_scores=aggregates,
        strategy=strategy,
        weights=active_weights,
    )

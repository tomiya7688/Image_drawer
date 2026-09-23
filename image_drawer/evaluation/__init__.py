"""Evaluator interfaces, deterministic baselines, and selection policies."""

from image_drawer.evaluation.base import (
    EvaluationIdentity,
    Evaluator,
    EvaluatorRegistry,
    artifact_candidates,
)
from image_drawer.evaluation.mock import DeterministicMockEvaluator
from image_drawer.evaluation.selection import (
    SelectionResult,
    parse_component_weights,
    select_from_scores,
)


def create_default_evaluator_registry() -> EvaluatorRegistry:
    registry = EvaluatorRegistry()
    registry.register(DeterministicMockEvaluator())
    return registry


__all__ = [
    "DeterministicMockEvaluator",
    "EvaluationIdentity",
    "Evaluator",
    "EvaluatorRegistry",
    "SelectionResult",
    "artifact_candidates",
    "create_default_evaluator_registry",
    "parse_component_weights",
    "select_from_scores",
]

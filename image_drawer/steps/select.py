"""Generic SELECT Step over evaluator-specific ScoreSets."""

from __future__ import annotations

from image_drawer.core import Artifact, ScoreSet
from image_drawer.evaluation import (
    artifact_candidates,
    parse_component_weights,
    select_from_scores,
)
from image_drawer.steps.base import ParameterSpec, Step, StepContext, StepSchema


class SelectStep(Step):
    backend = "selection"
    schema = StepSchema(
        step_type="SELECT",
        input_types=(("Image", "ImageSet"), "ScoreSet"),
        output_type="*",
        parameters={
            "strategy": ParameterSpec(
                str,
                default="best",
                choices=("best", "top_k", "weighted"),
            ),
            "top": ParameterSpec(int, default=1),
            "weights": ParameterSpec(list, default=[]),
        },
        capabilities=frozenset(
            {"selection", "top-k", "weighted-components", "deterministic"}
        ),
    )

    def run(self, inputs, params, context: StepContext):
        candidates_artifact, scores_artifact = inputs
        candidates = artifact_candidates(candidates_artifact)
        score_payload = scores_artifact.metadata.get("score_set")
        if not isinstance(score_payload, dict):
            raise ValueError(
                "SELECT requires ScoreSet artifact metadata.score_set"
            )
        score_set = ScoreSet.from_dict(score_payload)

        candidate_ids = [candidate.id for candidate in candidates]
        if candidate_ids != score_set.candidate_artifact_ids:
            raise ValueError(
                "candidate order/identity does not match ScoreSet"
            )

        weights = parse_component_weights(params["weights"])
        result = select_from_scores(
            score_set,
            strategy=params["strategy"],
            top=params["top"],
            weights=weights,
        )
        by_id = {candidate.id: candidate for candidate in candidates}
        selected = [
            by_id[artifact_id]
            for artifact_id in result.selected_artifact_ids
        ]
        selection_metadata = {
            "strategy": result.strategy,
            "top": len(result.selected_artifact_ids),
            "selected_artifact_ids": list(result.selected_artifact_ids),
            "ranking": list(result.ranking),
            "aggregate_scores": dict(result.aggregate_scores),
            "weights": dict(result.weights),
            "evaluator": score_set.evaluator,
            "evaluator_version": score_set.evaluator_version,
            "score_set_id": score_set.id,
        }
        parents = [candidates_artifact.id, scores_artifact.id]

        if len(selected) == 1:
            candidate = selected[0]
            metadata = dict(candidate.metadata)
            metadata["selection"] = selection_metadata
            metadata["selected_from_artifact_id"] = candidate.id
            return context.make_artifact(
                "Image",
                parent_artifact_ids=parents,
                uri=candidate.uri,
                model=candidate.model,
                version=candidate.version,
                seed=candidate.seed,
                scores=[score_set.score_for(candidate.id)],
                metadata=metadata,
            )

        return context.make_artifact(
            "ImageSet",
            parent_artifact_ids=parents,
            scores=[
                score_set.score_for(candidate.id)
                for candidate in selected
            ],
            metadata={
                "candidates": [candidate.to_dict() for candidate in selected],
                "value": [candidate.id for candidate in selected],
                "selection": selection_metadata,
            },
        )

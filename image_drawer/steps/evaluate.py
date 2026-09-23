"""Generic EVALUATE Step backed by an EvaluatorRegistry."""

from __future__ import annotations

from image_drawer.evaluation import EvaluatorRegistry, artifact_candidates
from image_drawer.steps.base import ParameterSpec, Step, StepContext, StepSchema


class EvaluateStep(Step):
    backend = "evaluation"
    schema = StepSchema(
        step_type="EVALUATE",
        input_types=(("Image", "ImageSet"),),
        output_type="ScoreSet",
        parameters={
            "evaluator": ParameterSpec(str, default="mock"),
        },
        capabilities=frozenset({"evaluation", "batch", "structured-scores"}),
    )

    def __init__(self, evaluators: EvaluatorRegistry) -> None:
        self.evaluators = evaluators

    def run(self, inputs, params, context: StepContext):
        candidates = artifact_candidates(inputs[0])
        evaluator = self.evaluators.resolve(params["evaluator"])
        prompt = next(
            (
                value
                for value in context.external_inputs.values()
                if isinstance(value, str)
            ),
            "",
        )
        score_set = evaluator.evaluate(
            candidates,
            context={
                "run_id": context.run_id,
                "step_id": context.step_id,
                "prompt": prompt,
            },
        )
        return context.make_artifact(
            "ScoreSet",
            parent_artifact_ids=[inputs[0].id],
            model=evaluator.identity.name,
            version=evaluator.identity.version,
            scores=list(score_set.scores),
            metadata={
                "score_set": score_set.to_dict(),
                "candidate_artifact_ids": list(
                    score_set.candidate_artifact_ids
                ),
                "execution_identity": {
                    "evaluator": evaluator.identity.to_dict(),
                },
            },
        )

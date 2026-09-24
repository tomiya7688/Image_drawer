"""modelなしでRuntimeを検証するためのdeterministic mock Step。"""

from __future__ import annotations

from typing import Any

from image_drawer.core import Artifact, Score
from image_drawer.steps.base import ParameterSpec, Step, StepContext, StepSchema
from image_drawer.steps.registry import StepRegistry


def _values(artifact: Artifact) -> list[Any]:
    value = artifact.metadata.get("value", [])
    return value if isinstance(value, list) else [value]


class InputStep(Step):
    backend = "mock"
    schema = StepSchema(
        step_type="INPUT",
        input_types=(),
        output_type="Text",
        capabilities=frozenset({"external_input"}),
    )

    def run(self, inputs, params, context):
        if context.step_id not in context.external_inputs:
            raise ValueError(f"missing external input for {context.step_id}")
        value = context.external_inputs[context.step_id]
        if not isinstance(value, str):
            raise TypeError("mock INPUT expects a string")
        return context.make_artifact("Text", metadata={"value": value})


class RetrievePartsStep(Step):
    backend = "mock"
    schema = StepSchema(
        step_type="RETRIEVE_PARTS",
        input_types=("Text",),
        output_type="PartSet",
        parameters={
            "category": ParameterSpec(str, default="generic"),
            "top": ParameterSpec(int, default=20),
        },
        capabilities=frozenset({"retrieval", "deterministic"}),
    )

    def run(self, inputs, params, context):
        prompt = inputs[0].metadata["value"]
        category = params["category"]
        top = params["top"]
        values = [f"{category}:{prompt}:{index}" for index in range(top)]
        scores = [round(1.0 - (index / max(top, 1)), 6) for index in range(top)]
        return context.make_artifact(
            "PartSet",
            parent_artifact_ids=[inputs[0].id],
            metadata={
                "value": values,
                "retrieval_scores": scores,
                "category": category,
            },
        )


class SelectPartsStep(Step):
    backend = "mock"
    schema = StepSchema(
        step_type="SELECT_PARTS",
        input_types=("PartSet",),
        output_type="PartSet",
        parameters={"top": ParameterSpec(int, default=5)},
        capabilities=frozenset({"selection", "deterministic"}),
    )

    def run(self, inputs, params, context):
        selected = _values(inputs[0])[: params["top"]]
        retrieval_scores = inputs[0].metadata.get("retrieval_scores", [])
        return context.make_artifact(
            "PartSet",
            parent_artifact_ids=[inputs[0].id],
            metadata={
                "value": selected,
                "retrieval_scores": retrieval_scores[: params["top"]],
                "selected": True,
            },
        )


class ComposeStep(Step):
    backend = "mock"
    schema = StepSchema(
        step_type="COMPOSE",
        input_types=("PartSet",),
        output_type="Image",
        parameters={"count": ParameterSpec(int, default=1)},
        capabilities=frozenset({"composition", "deterministic"}),
    )

    def run(self, inputs, params, context):
        parts = _values(inputs[0])
        rendered = "|".join(str(part) for part in parts)
        return context.make_artifact(
            "Image",
            parent_artifact_ids=[inputs[0].id],
            model="mock-compose",
            version="v1",
            metadata={
                "value": f"mock-image[{rendered}]",
                "count": params["count"],
            },
        )


class EvaluateStep(Step):
    backend = "mock"
    schema = StepSchema(
        step_type="EVALUATE",
        input_types=(("Image", "ImageSet"),),
        output_type="ScoreSet",
        parameters={"evaluator": ParameterSpec(str, default="mock")},
        capabilities=frozenset({"evaluation", "deterministic"}),
    )

    def run(self, inputs, params, context):
        rendered = str(inputs[0].metadata.get("value", ""))
        quality = round((sum(rendered.encode("utf-8")) % 1000) / 1000, 6)
        score = Score(
            id=context.score_id(),
            evaluator=params["evaluator"],
            evaluator_version="v1",
            overall=quality,
            components={
                "quality": quality,
                "determinism": 1.0,
            },
        )
        return context.make_artifact(
            "ScoreSet",
            parent_artifact_ids=[inputs[0].id],
            scores=[score],
            metadata={
                "value": [quality],
                "scored_artifact_ids": [inputs[0].id],
            },
        )


class SelectStep(Step):
    backend = "mock"
    schema = StepSchema(
        step_type="SELECT",
        input_types=(("Image", "ImageSet"), "ScoreSet"),
        output_type="Image",
        parameters={"top": ParameterSpec(int, default=1)},
        capabilities=frozenset({"selection", "deterministic"}),
    )

    def run(self, inputs, params, context):
        candidates, scores = inputs
        return context.make_artifact(
            "Image",
            parent_artifact_ids=[candidates.id, scores.id],
            metadata={
                "value": candidates.metadata.get("value"),
                "top": params["top"],
                "selected": True,
            },
        )


class OutputStep(Step):
    backend = "mock"
    schema = StepSchema(
        step_type="OUTPUT",
        input_types=("*",),
        output_type="*",
        capabilities=frozenset({"workflow_output"}),
    )

    def run(self, inputs, params, context):
        source = inputs[0]
        return context.make_artifact(
            source.artifact_type,
            parent_artifact_ids=[source.id],
            uri=source.uri,
            model=source.model,
            version=source.version,
            seed=source.seed,
            scores=source.scores,
            metadata=source.metadata,
        )


def create_mock_registry() -> StepRegistry:
    registry = StepRegistry()
    for step in (
        InputStep(),
        RetrievePartsStep(),
        SelectPartsStep(),
        ComposeStep(),
        EvaluateStep(),
        SelectStep(),
        OutputStep(),
    ):
        registry.register(step, default=True)
    return registry

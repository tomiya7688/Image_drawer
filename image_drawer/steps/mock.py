"""modelなしでRuntimeを検証するためのdeterministic mock Step。"""

from __future__ import annotations

from typing import Any

from image_drawer.core import Artifact
from image_drawer.evaluation.mock import MockEvaluator, image_candidates
from image_drawer.evaluation.openclip_quality import OpenClipQualityEvaluator
from image_drawer.evaluation.select_images import select_images
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
        if params["evaluator"] != MockEvaluator.name:
            raise ValueError(f"unknown mock evaluator: {params['evaluator']}")
        score_set = MockEvaluator().evaluate(image_candidates(inputs[0]), context)
        return context.make_artifact(
            "ScoreSet",
            parent_artifact_ids=[inputs[0].id],
            scores=score_set.scores,
            metadata={
                "value": [score.overall for score in score_set.scores],
                "scored_artifact_ids": score_set.candidate_artifact_ids,
                "score_set": score_set.to_dict(),
            },
        )


# {
#   責務: [
#     OpenClipQualityStep: OpenCLIP品質評価をRuntimeから呼び出す
#   ]
#   フィールド: [
#     schema: 入出力とローカル重みの指定
#   ]
#   処理: [
#     候補ごとの相対得点をScoreSet Artifactへ保存する
#   ]
# }
class OpenClipQualityStep(Step):
    backend = "openclip_quality"
    schema = StepSchema(
        step_type="EVALUATE",
        input_types=(("Image", "ImageSet"),),
        output_type="ScoreSet",
        parameters={"weights_path": ParameterSpec(str, required=True)},
        capabilities=frozenset({"evaluation", "local-model"}),
    )

    # {
    #   責務: [
    #     run: 指定された画像候補を評価する
    #   ]
    #   処理: [
    #     ローカル重みを読み、得点集合と由来を記録する
    #   ]
    #   引数: [
    #     inputs: 画像または画像集合
    #     params: 重みの保存場所
    #     context: 実行時の識別子
    #   ]
    #   戻り値: [
    #     artifact: 得点集合のArtifact
    #   ]
    # }
    def run(self, inputs, params, context):
        evaluator = OpenClipQualityEvaluator(params["weights_path"])
        score_set = evaluator.evaluate(image_candidates(inputs[0]), context)
        return context.make_artifact(
            "ScoreSet",
            parent_artifact_ids=[inputs[0].id],
            scores=score_set.scores,
            metadata={"score_set": score_set.to_dict()},
        )


# {
#   責務: [
#     select_output_type: topの値から、SELECTの出力型を決める
#   ]
#   引数: [
#     params: 解決済みのパラメータ
#   ]
#   戻り値: [
#     artifact_type: top=1はImage、それ以外はImageSet
#   ]
# }
def select_output_type(params: dict) -> str:
    return "Image" if params.get("top", 1) == 1 else "ImageSet"


# {
#   責務: [
#     SelectStep: 得点集合に従って上位の画像を選び、根拠をArtifactに残す
#   ]
#   フィールド: [
#     schema: top、weights、componentを受け取り、出力型をtopで決める
#   ]
#   処理: [
#     総合得点・評価軸・重み付き平均のいずれかで順位づけし、全候補の順位と生の成分を記録する
#   ]
# }
class SelectStep(Step):
    backend = "mock"
    schema = StepSchema(
        step_type="SELECT",
        input_types=(("Image", "ImageSet"), "ScoreSet"),
        output_type="Image",
        parameters={
            "top": ParameterSpec(int, default=1),
            "weights": ParameterSpec(dict, default={}),
            "component": ParameterSpec(str, default=""),
        },
        capabilities=frozenset({"selection", "deterministic"}),
        output_resolver=select_output_type,
    )

    # {
    #   責務: [run: 評価結果に基づき画像を選ぶ]
    #   処理: [入力を検証し、top=1はImage、top>=2はImageSetとして選択結果と由来を返す]
    #   引数: [inputs: 候補と得点, params: top・weights・component, context: 実行情報]
    #   戻り値: [artifact: 選択画像(top=1)または選択画像集合(top>=2)]
    # }
    def run(self, inputs, params, context):
        candidates, scores = inputs
        selection = select_images(
            candidates,
            scores,
            top=params["top"],
            weights=params.get("weights", {}),
            component=params.get("component", ""),
        )
        best, best_score = selection.candidates[0], selection.scores[0]
        record = {
            "mode": "best" if params["top"] == 1 else "top_k",
            "top": params["top"],
            "method": selection.method,
            "weights": selection.weights,
            "component": selection.component,
            "candidate_artifact_id": best.id,
            "score_id": best_score.id,
            "overall": best_score.overall,
            "candidate_artifact_ids": [item.id for item in selection.candidates],
            "score_ids": [item.id for item in selection.scores],
            "evaluator": selection.score_set.evaluator,
            "evaluator_version": selection.score_set.evaluator_version,
            "ranking": selection.ranking,
        }
        parents = [candidates.id, scores.id]
        if params["top"] != 1:
            return context.make_artifact(
                "ImageSet",
                parent_artifact_ids=parents,
                metadata={
                    "candidates": [item.to_dict() for item in selection.candidates],
                    "selected": True,
                    "selection": record,
                },
            )
        metadata = dict(best.metadata)
        # 生成元の実行情報をSELECT自身の実行情報として記録しない。
        metadata.pop("execution_identity", None)
        metadata.update({"selected": True, "selection": record})
        return context.make_artifact(
            "Image",
            parent_artifact_ids=parents,
            uri=best.uri,
            model=best.model,
            version=best.version,
            seed=best.seed,
            scores=best.scores,
            metadata=metadata,
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
    registry.register(OpenClipQualityStep())
    return registry

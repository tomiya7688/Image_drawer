"""上位k件・評価軸・重み付き平均によるSELECTと、出力型の解決を確認する。"""

import pytest

from image_drawer.core import Artifact, Score, ScoreSet, StepSpec, WorkflowSpec
from image_drawer.dsl import parse_workflow, serialize_workflow
from image_drawer.evaluation.select_images import select_images
from image_drawer.runtime import WorkflowRuntime, WorkflowValidationError, validate_workflow
from image_drawer.steps import Step, StepContext, StepSchema, create_mock_registry
from image_drawer.steps.mock import SelectStep

# 候補ごとの (overall, quality, sharpness)。overall順は a > b > c、
# quality順は c > b > a、sharpness順は b > a > c になる。
RAW = {"a": (0.9, 0.1, 0.5), "b": (0.6, 0.5, 0.9), "c": (0.3, 0.9, 0.1)}


# {
#   責務: [
#     make_inputs: 3候補のImageSetと、成分付きScoreSetのArtifactを作る
#   ]
#   戻り値: [
#     images, scores: SELECTへ渡す入力
#   ]
# }
def make_inputs(raw=RAW):
    ids = list(raw)
    candidates = [
        Artifact(id=name, artifact_type="Image", uri=f"{name}.png", metadata={"value": name})
        for name in ids
    ]
    images = Artifact(
        id="set",
        artifact_type="ImageSet",
        metadata={"candidates": [candidate.to_dict() for candidate in candidates]},
    )
    score_set = ScoreSet(
        id="scores",
        evaluator="mock",
        evaluator_version="v1",
        candidate_artifact_ids=ids,
        scores=[
            Score(
                id=f"score-{name}",
                evaluator="mock",
                evaluator_version="v1",
                overall=overall,
                components={"quality": quality, "sharpness": sharpness},
            )
            for name, (overall, quality, sharpness) in raw.items()
        ],
    )
    scores = Artifact(
        id="scores",
        artifact_type="ScoreSet",
        parent_artifact_ids=["set"],
        scores=score_set.scores,
        metadata={"score_set": score_set.to_dict()},
    )
    return images, scores


def chosen_ids(selection):
    return [candidate.id for candidate in selection.candidates]


def test_top_k_by_overall_keeps_rank_order():
    selection = select_images(*make_inputs(), top=2)
    assert chosen_ids(selection) == ["a", "b"]
    assert selection.method == "overall"
    assert [item["candidate_artifact_id"] for item in selection.ranking] == ["a", "b", "c"]


def test_component_changes_the_ranking_without_a_canonical_score():
    assert chosen_ids(select_images(*make_inputs(), top=1, component="quality")) == ["c"]
    assert chosen_ids(select_images(*make_inputs(), top=1, component="sharpness")) == ["b"]


def test_weighted_aggregate_over_components():
    # a: (0.1*1 + 0.5*3)/4 = 0.4 / b: (0.5 + 2.7)/4 = 0.8 / c: (0.9 + 0.3)/4 = 0.3
    selection = select_images(*make_inputs(), top=2, weights={"quality": 1, "sharpness": 3})
    assert chosen_ids(selection) == ["b", "a"]
    assert selection.method == "weighted"
    assert selection.ranking[0]["aggregate"] == pytest.approx(0.8)
    assert selection.weights == {"quality": 1.0, "sharpness": 3.0}


def test_raw_components_remain_available_after_selection():
    selection = select_images(*make_inputs(), top=1, weights={"quality": 1})
    by_id = {item["candidate_artifact_id"]: item for item in selection.ranking}
    assert by_id["a"]["components"] == {"quality": 0.1, "sharpness": 0.5}
    assert by_id["a"]["overall"] == 0.9
    assert selection.scores[0].components == {"quality": 0.9, "sharpness": 0.1}


def test_ties_are_resolved_by_candidate_id():
    raw = {"b": (0.5, 0.5, 0.5), "a": (0.5, 0.5, 0.5), "c": (0.1, 0.1, 0.1)}
    assert chosen_ids(select_images(*make_inputs(raw), top=2)) == ["a", "b"]


def test_weighted_selection_does_not_require_overall():
    raw = {"a": (None, 0.1, 0.5), "b": (None, 0.5, 0.9)}
    images, scores = make_inputs(raw)
    assert chosen_ids(select_images(images, scores, weights={"sharpness": 1})) == ["b"]
    with pytest.raises(ValueError, match="overall"):
        select_images(images, scores)


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"weights": {"quality": -1}}, "non-negative"),
        ({"weights": {"quality": 0}}, "positive sum"),
        ({"weights": {"quality": True}}, "finite"),
        ({"weights": {"quality": float("nan")}}, "finite"),
        ({"weights": {"missing": 1}}, "no component"),
        ({"component": "missing"}, "no component"),
        ({"weights": {"quality": 1}, "component": "quality"}, "cannot be combined"),
        ({"top": 0}, "between 1 and 3"),
        ({"top": 4}, "between 1 and 3"),
    ],
)
def test_invalid_selection_options_are_rejected(options, message):
    with pytest.raises(ValueError, match=message):
        select_images(*make_inputs(), **options)


def test_non_finite_scores_and_overflowing_aggregates_are_rejected():
    with pytest.raises(ValueError, match="finite"):
        ScoreSet(
            id="x",
            evaluator="mock",
            evaluator_version="v1",
            candidate_artifact_ids=["a"],
            scores=[
                Score(
                    id="s",
                    evaluator="mock",
                    evaluator_version="v1",
                    components={"quality": float("nan")},
                )
            ],
        )
    huge = {"a": (0.5, 1e308, 1e308), "b": (0.4, 1e308, 1e308)}
    with pytest.raises(ValueError, match="aggregate score must be finite"):
        select_images(*make_inputs(huge), weights={"quality": 1e308, "sharpness": 1e308})


# {
#   責務: [
#     MakeSetStep: テスト用にImageSetを生成するStep。ImageSetを返すStepが他にないため用意する
#   ]
# }
class MakeSetStep(Step):
    backend = "test"
    schema = StepSchema(step_type="MAKE_SET", input_types=("Text",), output_type="ImageSet")

    def run(self, inputs, params, context):
        images, _ = make_inputs()
        return context.make_artifact(
            "ImageSet",
            parent_artifact_ids=[inputs[0].id],
            metadata={"candidates": images.metadata["candidates"]},
        )


# {
#   責務: [
#     build_workflow: INPUT→MAKE_SET→EVALUATE→SELECT→OUTPUTのWorkflowを作る
#   ]
#   引数: [
#     select_parameters: SELECTに渡すパラメータ
#   ]
# }
def build_workflow(select_parameters):
    return WorkflowSpec(
        id="select-set",
        version="v1",
        nodes=[
            StepSpec(id="prompt", type="INPUT"),
            StepSpec(id="images", type="MAKE_SET"),
            StepSpec(id="evaluate", type="EVALUATE"),
            StepSpec(id="select", type="SELECT", parameters=select_parameters),
            StepSpec(id="output", type="OUTPUT"),
        ],
        edges=[
            ("prompt", "images"),
            ("images", "evaluate"),
            ("images", "select"),
            ("evaluate", "select"),
            ("select", "output"),
        ],
        inputs=["prompt"],
        outputs=["output"],
    )


def make_registry():
    registry = create_mock_registry()
    registry.register(MakeSetStep())
    return registry


def test_runtime_selects_top_k_image_set_and_records_the_decision():
    result = WorkflowRuntime(make_registry()).execute(
        build_workflow({"top": 2, "component": "quality"}),
        external_inputs={"prompt": "cat"},
        run_id="topk",
    )
    output = result.outputs["output"]
    assert output.artifact_type == "ImageSet"
    selection = output.metadata["selection"]
    assert selection["mode"] == "top_k"
    assert selection["method"] == "component"
    assert len(selection["candidate_artifact_ids"]) == 2
    assert len(selection["ranking"]) == 3
    record = next(item for item in result.trajectory.selections if item["step_id"] == "select")
    assert record["candidate_artifact_ids"] == selection["candidate_artifact_ids"]
    assert record["ranking"] == selection["ranking"]
    # 選択後も、評価で記録した生得点はTrajectoryに1件ずつ残る。
    assert len(result.trajectory.evaluations) == 3


def test_output_type_follows_top_during_validation():
    registry = make_registry()
    assert registry.resolve("SELECT").schema.output_for({"top": 1}) == "Image"
    assert registry.resolve("SELECT").schema.output_for({"top": 3}) == "ImageSet"

    class NeedsImage(Step):
        backend = "test"
        schema = StepSchema(step_type="NEEDS_IMAGE", input_types=("Image",), output_type="Image")

        def run(self, inputs, params, context):
            return inputs[0]

    registry.register(NeedsImage())

    def workflow(top):
        spec = build_workflow({"top": top})
        spec.nodes.insert(-1, StepSpec(id="needs_image", type="NEEDS_IMAGE"))
        spec.edges = [edge for edge in spec.edges if edge != ("select", "output")]
        spec.edges += [("select", "needs_image"), ("needs_image", "output")]
        return spec

    validate_workflow(workflow(1), registry)
    with pytest.raises(WorkflowValidationError, match="expects Image, got ImageSet"):
        validate_workflow(workflow(2), registry)


def test_dsl_round_trips_weights_for_select():
    source = """\
INPUT prompt: Text

images = MAKE_SET(prompt)
scores = EVALUATE(images)
best = SELECT(images, scores=scores, top=2, weights={"quality": 1, "sharpness": 3})

OUTPUT best
"""
    workflow = parse_workflow(source, make_registry())
    select = next(node for node in workflow.nodes if node.type == "SELECT")
    assert select.parameters["weights"] == {"quality": 1, "sharpness": 3}
    canonical = serialize_workflow(workflow, make_registry())
    assert parse_workflow(canonical, make_registry()) == workflow


def test_select_step_defaults_keep_the_single_best_behavior():
    images, scores = make_inputs()
    context = StepContext(run_id="run", step_id="select", backend="mock", external_inputs={})
    artifact = SelectStep().run([images, scores], {"top": 1}, context)
    assert artifact.artifact_type == "Image"
    assert artifact.metadata["selection"]["candidate_artifact_id"] == "a"
    assert artifact.metadata["selection"]["mode"] == "best"

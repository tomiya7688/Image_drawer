"""評価報告書・ゲート・claim_levelを確認する(設計 9・10節)。

トイPolicyの結果は配線の確認であり、モデルの精度や学習の効果の根拠にしない。
"""

import copy
import json
from pathlib import Path

import jsonschema
import pytest
from test_eval_runner import PaintPolicy, FailingPolicy, make_runner, run_toy

from image_drawer.generalization.compare import compare_all
from image_drawer.generalization.eval_set import EvalItem, EvalSet
from image_drawer.generalization.gates import (
    GATE_IDS,
    GateInputs,
    decide_claim_level,
    evaluate_gates,
)
from image_drawer.generalization.report import (
    HEADINGS,
    build_report,
    render_markdown,
    validate_claim_text,
)

SCHEMA = Path(__file__).parent.parent / "docs/developers/schemas/eval_report_v1.schema.json"
COMPONENTS = ["rgb_128", "edge_128", "rgb_hist_16"]
HELDOUT = ["heldout_ssim64"]
GOOD = {"delta": 0.2, "ci95": [0.1, 0.3], "win_rate": 0.9, "n_groups": 120, "effect_size": 1.0}


# {
#   責務: [ passing_inputs: G0〜G10の全てを満たす(試験用の)入力。各ゲートの失敗は、ここから1項目を崩して確かめる ]
# }
def passing_inputs() -> GateInputs:
    rows = [
        {"a": "after", "b": other, "metric": metric, "subset": subset, **GOOD}
        for other in ("before", "random_valid")
        for subset in ("test_unseen_image", "test_unseen_prompt")
        for metric in [*COMPONENTS, *HELDOUT]
    ]
    human = {
        "raters": 6,
        "excluded_raters": 0,
        "min_raters_per_item": 3,
        "catch_pass_rate": 1.0,
        "tasks": {t: {"ci95": [0.6, 0.8], "kappa": 0.5} for t in ("T1", "T2")},
    }
    return GateInputs(
        policy_kind="trained", worktree_clean=True, reproduced=True, target_isolation_ok=True,
        replay_exact_rate=1.0, auditor_disagreements=0, completion_rate=0.98, audit_pass_rate=1.0,
        test_groups=120, shared_groups=0, ledger_retest=False, exclusions_disclosed=True,
        weights_differ=True, training_seeds=3, inference_seeds=5, comparisons=rows,
        reward_components=COMPONENTS, heldout_metrics=HELDOUT,
        median_gain_positive_all_seeds=True,
        controls={
            "prompt_shuffle_delta": {"delta": 0.1, "ci95": [0.05, 0.15]},
            "probe_hit_rate": {"before": 0.2, "after": 0.6},
            "nearest_train_hit_rate": {"before": 0.02, "after": 0.01},
            "generalization_gap": 0.01,
        },
        human=human,
    )  # fmt: skip


# {
#   責務: [ passed_ids: 判定したゲートのうち合格したIDの集合 ]
# }
def passed_ids(inputs: GateInputs) -> set[str]:
    return {g["gate_id"] for g in evaluate_gates(inputs) if g["passed"]}


# {
#   責務: [ toy_report: 試験用Policyの実行から、報告書の辞書を作る ]
# }
def toy_report(tmp_path, kind: str = "scripted", **options) -> dict:
    output = run_toy(
        tmp_path, {"before": make_runner(FailingPolicy()), "after": make_runner(PaintPolicy())}
    )
    items = [EvalItem(i, "h", g, "黒") for i, g in (("img1", "g1"), ("img2", "g2"))]
    eval_set = EvalSet("set1", "t", "dh", 7, items, eval_set_sha256="x")
    comparison = compare_all(output, resamples=100)
    inputs = GateInputs(policy_kind=kind, worktree_clean=True, reproduced=True, target_isolation_ok=True,
        replay_exact_rate=1.0, auditor_disagreements=0)  # fmt: skip
    return build_report(
        comparison, output, eval_set, inputs, code={"git_commit": "abc"}, protocol={"inference_seeds_per_item": 2},
        reproduction={"commands": ["image-drawer-evaluate run ..."]}, report_id="r1",
        created_at="2026-11-05T00:00:00+00:00", resamples=100, **options,
    )  # fmt: skip


def test_report_json_validates_against_schema(tmp_path):
    report = toy_report(tmp_path)
    jsonschema.validate(report, json.loads(SCHEMA.read_text(encoding="utf-8")))
    broken = copy.deepcopy(report)
    broken["claim_level"] = "excellent"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, json.loads(SCHEMA.read_text(encoding="utf-8")))


def test_report_marks_unverified_layers_explicitly(tmp_path):
    report = toy_report(tmp_path)
    text = " ".join(report["unverified"])
    assert "人間評価" in text and "OpenCLIP" in text and "試験用" in text
    assert report["human_eval"] is None
    assert "未実施" in render_markdown(report)


def test_claim_level_is_none_without_trained_policy(tmp_path):
    assert toy_report(tmp_path, kind="none")["claim_level"] == "none"
    assert decide_claim_level("none", evaluate_gates(passing_inputs())) == "none"


def test_claim_level_l1_for_scripted_policy_runs(tmp_path):
    assert toy_report(tmp_path, kind="scripted")["claim_level"] == "pipeline_verified"
    assert "G2" not in passed_ids(GateInputs(policy_kind="scripted"))


def test_claim_level_l3_requires_every_gate_g0_to_g10():
    inputs = passing_inputs()
    assert passed_ids(inputs) == set(GATE_IDS)
    assert decide_claim_level("trained", evaluate_gates(inputs)) == "high_accuracy_claim_allowed"
    breakers = {
        "G0": lambda i: setattr(i, "worktree_clean", False),
        "G1": lambda i: setattr(i, "test_groups", 99),
        "G2": lambda i: setattr(i, "weights_differ", False),
        "G3": lambda i: setattr(i, "inference_seeds", 4),
        "G4": lambda i: setattr(i, "median_gain_positive_all_seeds", False),
        "G5": lambda i: [
            r.update(ci95=[-0.3, 0.1]) for r in i.comparisons if r["metric"] in HELDOUT
        ],
        "G6": lambda i: [
            r.update(ci95=[-0.1, 0.1]) for r in i.comparisons if r["b"] == "random_valid"
        ],
        "G7": lambda i: i.controls["prompt_shuffle_delta"].update(ci95=[-0.1, 0.1]),
        "G8": lambda i: setattr(i, "completion_rate", 0.9),
        "G9": lambda i: setattr(i, "human", None),
        "G10": lambda i: i.controls["nearest_train_hit_rate"].update(after=0.2),
    }
    for gate, breaker in breakers.items():
        broken = passing_inputs()
        breaker(broken)
        assert gate in set(GATE_IDS) - passed_ids(broken), gate
        assert (
            decide_claim_level("trained", evaluate_gates(broken)) != "high_accuracy_claim_allowed"
        )


def test_gate_g4_fails_when_any_component_interval_includes_zero():
    inputs = passing_inputs()
    for row in inputs.comparisons:
        if (row["b"], row["subset"], row["metric"]) == ("before", "test_unseen_prompt", "edge_128"):
            row["ci95"] = [-0.01, 0.2]
    assert "G4" not in passed_ids(inputs)
    assert decide_claim_level("trained", evaluate_gates(inputs)) != "improvement_observed"


def test_gate_g1_fails_when_ledger_has_retest():
    inputs = passing_inputs()
    inputs.ledger_retest = True
    assert "G1" not in passed_ids(inputs)
    inputs.ledger_retest = None  # 帳簿を確認できなければ合格にしない
    assert "G1" not in passed_ids(inputs)


def test_gate_g9_fails_without_human_evaluation():
    inputs = passing_inputs()
    inputs.human = None
    gate = next(g for g in evaluate_gates(inputs) if g["gate_id"] == "G9")
    assert gate["passed"] is False and "not performed" in gate["evidence"]
    assert decide_claim_level("trained", evaluate_gates(inputs)) == "improvement_observed"


def test_report_markdown_has_required_headings_and_random_and_worst_examples(tmp_path):
    report = toy_report(tmp_path)
    text = render_markdown(report)
    for number, heading in enumerate(HEADINGS, start=1):
        assert f"## {number}. {heading}" in text
    examples = report["results"]["examples"]
    assert examples["random"] and examples["worst"]
    assert "ランダムに選んだ例" in text and "得点が最も低い例" in text
    scores = [e["score"] for e in examples["worst"]]
    assert scores == sorted(scores)


def test_report_rejects_disallowed_phrases_without_l3(tmp_path):
    with pytest.raises(ValueError, match="disallowed"):
        validate_claim_text("このモデルは高精度です", "improvement_observed")
    validate_claim_text("高精度 (report r1)", "high_accuracy_claim_allowed", "r1")
    report = toy_report(tmp_path)
    report["limitations"].append("必ず収束する")
    with pytest.raises(ValueError, match="disallowed"):
        render_markdown(report)


def test_reproduction_commands_reproduce_metrics_from_frozen_inputs(tmp_path):
    first, second = toy_report(tmp_path), toy_report(tmp_path)

    # 同じ凍結入力(seed・評価seed)から、同じ指標が得られる。
    def metrics_of(report):
        return {n: c["metrics"] for n, c in report["results"]["by_condition"].items()}

    assert metrics_of(first) == metrics_of(second)
    assert first["results"]["comparisons"] == second["results"]["comparisons"]
    assert first["reproduction"]["commands"] == ["image-drawer-evaluate run ..."]

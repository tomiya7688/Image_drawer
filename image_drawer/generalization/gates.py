"""公表条件のゲート(G0〜G10)と、主張の水準(claim_level)。数値は設計書10.2節の暫定値で、結果を見て変更しない。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from image_drawer.generalization.metrics_quality import reward_gain_without_heldout_gain

MIN_TEST_GROUPS = 100
MIN_TRAINING_SEEDS = 3
MIN_INFERENCE_SEEDS = 5
MIN_COMPLETION_RATE = 0.95
MAX_NEAREST_TRAIN_HIT_RATE = 0.05
MIN_HUMAN_KAPPA = 0.4
MIN_HUMAN_RATERS = 5
MIN_RATERS_PER_ITEM = 3
HUMAN_WIN_RATE_FLOOR = 0.5
GATE_IDS = tuple(f"G{i}" for i in range(11))
L2_GATES = ("G0", "G1", "G2", "G3", "G4", "G6")
SUBSETS = ("test_unseen_image", "test_unseen_prompt")


# {
#   責務: [
#     GateInputs: ゲート判定に必要な、実測または確認済みの事実。未確認の値はNone(合格にしない)
#   ]
#   フィールド: [
#     policy_kind: "trained"(学習したPolicy)、"scripted"(試験用)、"none"
#     worktree_clean, reproduced, target_isolation_ok: G0の確認結果
#     replay_exact_rate, auditor_disagreements, completion_rate, audit_pass_rate: 工程の指標
#     test_groups, shared_groups, ledger_retest, exclusions_disclosed: G1
#     weights_differ: beforeとafterの重みが異なるか(G2)
#     training_seeds, inference_seeds: G3
#     comparisons: compareの比較(subsetごとのa,b,metric,ci95)
#     reward_components, heldout_metrics: 成分名と独立指標名
#     median_gain_positive_all_seeds: 全学習seedでgainの中央値が正か(G4)
#     controls: prompt_shuffle_delta(ci95)、probe_hit_rate・nearest_train_hit_rate(before/after)、generalization_gap
#     human: collect_human_resultsの結果と確認項目の情報。実施しなければNone
#   ]
# }
@dataclass(slots=True)
class GateInputs:
    policy_kind: str = "none"
    worktree_clean: bool | None = None
    reproduced: bool | None = None
    target_isolation_ok: bool | None = None
    replay_exact_rate: float | None = None
    auditor_disagreements: int | None = None
    completion_rate: float | None = None
    audit_pass_rate: float | None = None
    test_groups: int = 0
    shared_groups: int | None = None
    ledger_retest: bool | None = None
    exclusions_disclosed: bool = False
    weights_differ: bool | None = None
    training_seeds: int = 0
    inference_seeds: int = 0
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    reward_components: list[str] = field(default_factory=list)
    heldout_metrics: list[str] = field(default_factory=list)
    median_gain_positive_all_seeds: bool | None = None
    controls: dict[str, Any] = field(default_factory=dict)
    human: dict[str, Any] | None = None


# {
#   責務: [ _lower: 比較の信頼区間の下限。該当する比較が無ければNone ]
# }
def _bound(inputs: GateInputs, other: str, subset: str, metric: str, index: int) -> float | None:
    for row in inputs.comparisons:
        if (row["a"], row["b"], row["subset"], row["metric"]) == ("after", other, subset, metric):
            return row["ci95"][index]
    return None


# {
#   責務: [ _all_lower_positive: 指定した指標の全てで区間の下限が0を超えるか(比較が欠ければ偽) ]
# }
def _all_lower_positive(inputs: GateInputs, other: str, subsets, metrics) -> bool:
    if not metrics:
        return False
    bounds = [_bound(inputs, other, s, m, 0) for s in subsets for m in metrics]
    return all(b is not None and b > 0 for b in bounds)


# {
#   責務: [ _gate_g0: 再現可能性と配線の正しさ ]
# }
def _gate_g0(i: GateInputs) -> tuple[bool, str]:
    checks = {
        "worktree_clean": i.worktree_clean is True,
        "reproduced": i.reproduced is True,
        "target_isolation_ok": i.target_isolation_ok is True,
        "replay_exact_rate=1.0": i.replay_exact_rate == 1.0,
        "auditor_disagreements=0": i.auditor_disagreements == 0,
    }
    return all(checks.values()), ", ".join(f"{k}={'ok' if v else 'NG'}" for k, v in checks.items())


# {
#   責務: [ _gate_g4_g5_g6: 改善・独立指標・対照の3ゲート ]
# }
def _improvement_gates(i: GateInputs) -> dict[str, tuple[bool, str]]:
    g4 = (
        _all_lower_positive(i, "before", SUBSETS, i.reward_components)
        and i.median_gain_positive_all_seeds is True
    )
    everything = [*i.reward_components, *i.heldout_metrics]
    degraded = [
        m
        for m in everything
        for s in SUBSETS
        if (_bound(i, "before", s, m, 1) is not None and _bound(i, "before", s, m, 1) < 0)
    ]
    heldout_gain = any(
        _all_lower_positive(i, "before", ("test_unseen_image",), [m]) for m in i.heldout_metrics
    )
    warning = reward_gain_without_heldout_gain(
        {m: _bound(i, "before", "test_unseen_image", m, 0) or 0.0 for m in i.reward_components},
        {m: _bound(i, "before", "test_unseen_image", m, 0) or 0.0 for m in i.heldout_metrics},
    )
    g5 = heldout_gain and not degraded and not warning
    g6 = _all_lower_positive(i, "before", ("test_unseen_image",), i.reward_components) and (
        _all_lower_positive(i, "random_valid", ("test_unseen_image",), i.reward_components)
    )
    return {
        "G4": (
            g4,
            f"component intervals above 0 in both subsets; seeds_consistent={i.median_gain_positive_all_seeds}",
        ),
        "G5": (g5, f"heldout_gain={heldout_gain}, degraded={degraded}, warning={warning}"),
        "G6": (g6, "after exceeds before and random_valid on every component"),
    }


# {
#   責務: [ _gate_g7_g10: prompt忠実度と記憶の検査 ]
# }
def _control_gates(i: GateInputs) -> dict[str, tuple[bool, str]]:
    c = i.controls
    shuffle = c.get("prompt_shuffle_delta", {}).get("ci95")
    probe = c.get("probe_hit_rate", {})
    near = c.get("nearest_train_hit_rate", {})
    g7 = bool(shuffle) and shuffle[0] > 0 and bool(probe) and probe["after"] > probe["before"]
    g10 = (
        bool(near)
        and near["after"] <= MAX_NEAREST_TRAIN_HIT_RATE
        and near["after"] <= near["before"]
        and c.get("generalization_gap") is not None
    )
    return {
        "G7": (g7, f"prompt_shuffle_ci={shuffle}, probe={probe}"),
        "G10": (g10, f"nearest_train_hit_rate={near}, gap={c.get('generalization_gap')}"),
    }


# {
#   責務: [ _gate_g9: 独立した人間評価 ]
# }
def _gate_g9(i: GateInputs) -> tuple[bool, str]:
    human = i.human
    if human is None:
        return False, "human evaluation was not performed"
    tasks = human.get("tasks", {})
    ok = bool(tasks) and all(
        t["ci95"][0] is not None
        and t["ci95"][0] > HUMAN_WIN_RATE_FLOOR
        and t["kappa"] is not None
        and t["kappa"] >= MIN_HUMAN_KAPPA
        for t in tasks.values()
    )
    ok = ok and human.get("raters", 0) - human.get("excluded_raters", 0) >= MIN_HUMAN_RATERS
    ok = ok and human.get("min_raters_per_item", 0) >= MIN_RATERS_PER_ITEM
    ok = ok and human.get("catch_pass_rate", 0.0) >= 0.9
    return ok, f"tasks={sorted(tasks)}, raters={human.get('raters')}"


# {
#   責務: [
#     evaluate_gates: G0〜G10を判定し、根拠を付けて返す
#   ]
#   戻り値: [ gates: [{gate_id, passed, evidence}]。判定できない項目は合格にしない ]
# }
def evaluate_gates(i: GateInputs) -> list[dict[str, Any]]:
    results: dict[str, tuple[bool, str]] = {"G0": _gate_g0(i)}
    results["G1"] = (
        i.test_groups >= MIN_TEST_GROUPS
        and i.shared_groups == 0
        and i.exclusions_disclosed
        and i.ledger_retest is False,
        f"test_groups={i.test_groups}, shared_groups={i.shared_groups}, retest={i.ledger_retest}, exclusions_disclosed={i.exclusions_disclosed}",
    )
    results["G2"] = (
        i.policy_kind == "trained" and i.weights_differ is True,
        f"policy_kind={i.policy_kind}, weights_differ={i.weights_differ}",
    )
    results["G3"] = (
        i.training_seeds >= MIN_TRAINING_SEEDS and i.inference_seeds >= MIN_INFERENCE_SEEDS,
        f"training_seeds={i.training_seeds}, inference_seeds={i.inference_seeds}",
    )
    results.update(_improvement_gates(i))
    results.update(_control_gates(i))
    results["G8"] = (
        (i.completion_rate or 0.0) >= MIN_COMPLETION_RATE and i.audit_pass_rate == 1.0,
        f"completion_rate={i.completion_rate}, audit_pass_rate={i.audit_pass_rate}",
    )
    results["G9"] = _gate_g9(i)
    return [
        {"gate_id": gate, "passed": bool(results[gate][0]), "evidence": results[gate][1]}
        for gate in GATE_IDS
    ]


# {
#   責務: [
#     decide_claim_level: ゲートの結果から主張の水準を決める
#   ]
#   処理: [
#     学習したPolicyでなければ、G0合格の試験用Policyだけがpipeline_verified。それ以外はnone。
#     学習したPolicyでは、G0〜G10が全て合格でhigh_accuracy_claim_allowed、G0〜G4とG6でimprovement_observed、
#     G0だけでpipeline_verified
#   ]
# }
def decide_claim_level(policy_kind: str, gates: list[dict[str, Any]]) -> str:
    passed = {g["gate_id"] for g in gates if g["passed"]}
    if policy_kind != "trained":
        return "pipeline_verified" if policy_kind == "scripted" and "G0" in passed else "none"
    if set(GATE_IDS) <= passed:
        return "high_accuracy_claim_allowed"
    if set(L2_GATES) <= passed:
        return "improvement_observed"
    return "pipeline_verified" if "G0" in passed else "none"

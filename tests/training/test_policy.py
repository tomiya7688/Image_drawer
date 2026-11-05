"""Policy・action_codec・target隔離の契約テスト(設計書8.1・8.2節)。"""

from __future__ import annotations

import ast
import builtins
import dataclasses
import pathlib
import typing

import pytest

np = pytest.importorskip("numpy")

from image_drawer.drawing import DrawingAction, Engine  # noqa: E402
from image_drawer.drawing.stage import Budget, StageController, judge_style_intent  # noqa: E402
from image_drawer.training import TrainConfig  # noqa: E402
from image_drawer.training import action_codec as codec  # noqa: E402
from image_drawer.training import policy as policy_module  # noqa: E402
from image_drawer.training.observation import ObservationBuilder  # noqa: E402
from image_drawer.training.policy import ParametricPolicy, PolicySample  # noqa: E402
from image_drawer.training.policy_features import build_features  # noqa: E402
from image_drawer.training.policy_params import masked_softmax  # noqa: E402

PACKAGE = pathlib.Path(policy_module.__file__).parent
POLICY_SIDE_MODULES = (
    "policy.py",
    "observation.py",
    "action_codec.py",
    "text_encoder.py",
    "policy_params.py",
    "policy_features.py",
    "image_ops.py",
)
FORBIDDEN_TYPE_NAMES = {
    "RewardTarget",
    "TargetFeatures",
    "RewardBreakdown",
    "RewardEvaluator",
    "PolicyInput",
    "TrialReward",
}


def make_observation(side: int = 16):
    engine = Engine(side, side, seed=1)
    budget = Budget(24)
    controller = StageController(engine, judge_style_intent("test"), budget)
    text = np.linspace(-1, 1, 8, dtype=np.float32)
    observation = ObservationBuilder(tail_length=8, summary_side=4).build(
        text, controller, engine, budget
    )
    return observation, controller, engine


def make_policy(side: int = 16, seed: int = 0) -> ParametricPolicy:
    config = TrainConfig(canvas_side=side, summary_side=4, max_actions=24)
    return ParametricPolicy(config, 8, np.random.default_rng(seed))


def _expand_type_names(annotation: object, seen: set[object]) -> set[str]:
    names: set[str] = set()
    if annotation in seen:
        return names
    seen.add(annotation)
    for argument in typing.get_args(annotation):
        names |= _expand_type_names(argument, seen)
    name = getattr(annotation, "__name__", None)
    if name:
        names.add(name)
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        for hint in typing.get_type_hints(annotation).values():
            names |= _expand_type_names(hint, seen)
    return names


def test_policy_act_signature_excludes_reward_target() -> None:
    names: set[str] = set()
    for function in (ParametricPolicy.act, policy_module.DrawingPolicy.act):
        for hint in typing.get_type_hints(function).values():
            names |= _expand_type_names(hint, set())
    assert {"Observation", "PolicyDecision", "Generator"} <= names
    assert not names & FORBIDDEN_TYPE_NAMES


def _imported_modules(path: pathlib.Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules |= {f"{node.module}.{alias.name}" for alias in node.names}
    return modules


def test_policy_modules_do_not_import_dataset_or_reward() -> None:
    banned_parts = ("dataset", "reward", "rollout", "evaluation", "trainer", "integrity")
    for filename in POLICY_SIDE_MODULES:
        for module in _imported_modules(PACKAGE / filename):
            parts = module.split(".")
            assert not any(part.startswith(banned_parts) for part in parts), (filename, module)
            assert parts[:2] != ["image_drawer", "steps"] and parts[:2] != ["image_drawer", "gui"]


def test_policy_act_performs_no_file_io(monkeypatch: pytest.MonkeyPatch) -> None:
    observation, _controller, _engine = make_observation()
    policy = make_policy()

    def forbidden(*args, **kwargs):
        raise AssertionError("act中にファイルを開いてはいけません")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(pathlib.Path, "open", forbidden)
    assert policy.act(observation, np.random.default_rng(0)).action.actor == "model"


def test_policy_same_seed_same_action() -> None:
    observation, _c, _e = make_observation()
    policy = make_policy()
    first = policy.act(observation, np.random.default_rng(5))
    second = policy.act(observation, np.random.default_rng(5))
    assert first.action == second.action and first.sample.choice == second.sample.choice
    actions = {repr(policy.act(observation, np.random.default_rng(s)).action) for s in range(30)}
    assert len(actions) > 1


def test_policy_masks_kinds_by_stage() -> None:
    observation, _c, _e = make_observation()
    assert codec.legal_kinds("rough") == codec.ACTION_KINDS
    assert codec.legal_kinds("finish") == ()
    mask = codec.kind_mask(observation)
    assert mask.all()
    # 選べるレイヤーが無ければselect_layerの確率は厳密に0。
    no_layers = dataclasses.replace(observation, layer_summary=())
    mask = codec.kind_mask(no_layers)
    assert not mask[codec.ACTION_KINDS.index("select_layer")]
    probabilities = masked_softmax(np.zeros(6), mask)
    assert probabilities[codec.ACTION_KINDS.index("select_layer")] == 0.0
    policy = make_policy()
    for seed in range(40):
        decision = policy.act(no_layers, np.random.default_rng(seed))
        assert decision.action.type != "select_layer"
    # 工程finishでは選べる種別が無く、例外になる。
    with pytest.raises(ValueError):
        policy.act(dataclasses.replace(observation, stage_id="finish"), np.random.default_rng(0))


def _samples(policy: ParametricPolicy) -> list[PolicySample]:
    observation, _c, _e = make_observation()
    features = build_features(observation)
    kinds, layers = codec.kind_mask(observation), codec.layer_mask(observation)
    choices = [
        codec.ActionChoice(0, layer_index=0),
        codec.ActionChoice(1, color_id=17),
        codec.ActionChoice(2, brush_index=3),
        codec.ActionChoice(3, size_level=4, opacity_level=7),
        codec.ActionChoice(4, stroke_u=(0.1, -0.3, 0.7, 0.2, -0.5, 0.4)),
        codec.ActionChoice(5),
    ]
    return [PolicySample(features, "rough", kinds, layers, c, 0.15) for c in choices]


def test_policy_gradient_matches_finite_difference() -> None:
    policy = make_policy()
    rng = np.random.default_rng(1)
    for sample in _samples(policy):
        grads = policy.new_grads()
        base = policy.accumulate_log_prob_grad(sample, 1.0, grads)
        entropy_grads = policy.new_grads()
        policy.accumulate_entropy_grad(sample, 1.0, entropy_grads)
        for name, array in policy.params.items():
            for _ in range(6):
                index = tuple(rng.integers(0, size) for size in array.shape)
                original = array[index]
                numeric = []
                for delta in (1e-5, -1e-5):
                    array[index] = original + delta
                    probe = policy.new_grads()
                    logp = policy.accumulate_log_prob_grad(sample, 0.0, probe)
                    entropy = policy.accumulate_entropy_grad(sample, 0.0, policy.new_grads())
                    numeric.append((logp, entropy))
                array[index] = original
                assert (numeric[0][0] - numeric[1][0]) / 2e-5 == pytest.approx(
                    grads.arrays[name][index], abs=1e-6
                ), name
                assert (numeric[0][1] - numeric[1][1]) / 2e-5 == pytest.approx(
                    entropy_grads.arrays[name][index], abs=1e-6
                ), name
        assert base <= 0.0 + 1e9  # log確率が有限であること
        assert np.isfinite(base)


def test_action_codec_roundtrips_through_engine() -> None:
    observation, controller, engine = make_observation(side=16)
    config = TrainConfig(canvas_side=16, summary_side=4, max_actions=24)
    stroke = lambda u: codec.ActionChoice(4, stroke_u=u)  # noqa: E731
    choices = [
        codec.ActionChoice(0, layer_index=0),
        codec.ActionChoice(1, color_id=255),
        codec.ActionChoice(2, brush_index=11),
        codec.ActionChoice(2, brush_index=0),
        codec.ActionChoice(3, size_level=18, opacity_level=10),
        codec.ActionChoice(3, size_level=0, opacity_level=0),
        # 座標0(sigmoid極小)と座標1(極大)の端を含める。
        stroke((-60.0,) * 6),
        stroke((60.0,) * 6),
        stroke((-60.0, 60.0, 0.0, 0.0, 60.0, -60.0)),
        codec.ActionChoice(5),
    ]
    for choice in choices:
        action = codec.decode_action(choice, observation, 16, config.stroke_samples, 5)
        result = controller.propose(action)
        if action.type == "stroke_v2":
            assert result.accepted, result.reason
        if action.type in ("select_layer", "select_color", "set_brush_params"):
            assert result.accepted, result.reason
    samples = codec.build_stroke_samples((-60.0,) * 6, 16, 6, 5)
    assert samples[0]["x_q8"] == 0 and samples[0]["time_us"] == 0
    samples = codec.build_stroke_samples((60.0,) * 6, 16, 6, 5)
    assert samples[-1]["x_q8"] == 15 * 256 and samples[-1]["y_q8"] == 15 * 256
    assert engine.state.history  # 実Engineに受理されたイベントが記録される


def test_action_codec_rejects_out_of_range_indices() -> None:
    observation, _c, _e = make_observation()

    def decode(choice):
        return codec.decode_action(choice, observation, 16, 6, 5)

    for bad in (
        codec.ActionChoice(1, color_id=256),
        codec.ActionChoice(1, color_id=-1),
        codec.ActionChoice(2, brush_index=12),
        codec.ActionChoice(3, size_level=19, opacity_level=0),
        codec.ActionChoice(3, size_level=0, opacity_level=11),
        codec.ActionChoice(0, layer_index=16),
        codec.ActionChoice(0, layer_index=5),
        codec.ActionChoice(0),
        codec.ActionChoice(4),
        codec.ActionChoice(9),
    ):
        with pytest.raises(ValueError):
            decode(bad)


def test_policy_state_dict_roundtrip_and_rejects_bad_shapes() -> None:
    policy, other = make_policy(seed=0), make_policy(seed=1)
    assert policy.params_sha256() != other.params_sha256()
    other.load_state_dict(policy.state_dict())
    assert policy.params_sha256() == other.params_sha256()
    broken = policy.state_dict()
    broken["kind"] = broken["kind"][:, :-1]
    with pytest.raises(ValueError):
        other.load_state_dict(broken)
    assert isinstance(DrawingAction("advance_stage", {}), DrawingAction)

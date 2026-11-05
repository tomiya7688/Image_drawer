"""離散ストローク表現の受け入れテスト(design_stroke_action.md 5節)。"""

from __future__ import annotations

import ast
import inspect
import json
import math
import pathlib

import pytest

np = pytest.importorskip("numpy")

from image_drawer.drawing import RENDERER_V2, DrawingAction, Engine  # noqa: E402
from image_drawer.drawing.stage import Budget, StageController, judge_style_intent  # noqa: E402
from image_drawer.training import (  # noqa: E402
    CheckpointMismatchError,
    HashedNgramTextEncoder,
    TrainConfig,
)
from image_drawer.training import action_codec as codec  # noqa: E402
from image_drawer.training import stroke_geometry as geo  # noqa: E402
from image_drawer.training.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from image_drawer.training.observation import ObservationBuilder  # noqa: E402
from image_drawer.training.optimizer import BaselineTable  # noqa: E402
from image_drawer.training.policy import ParametricPolicy, PolicySample  # noqa: E402
from image_drawer.training.policy_features import build_features  # noqa: E402
from image_drawer.training.policy_params import AdamState  # noqa: E402

SHAPE = geo.StrokeShape()
SIDE = 64
CELL_Q8 = 256 * (SIDE - 1) / SHAPE.grid


def replay_pick(**values):
    """名前->添字の辞書で選ぶ。stroke_dir_i / stroke_len_i は dirs / lens の列でも渡せる。"""

    def pick(name: str, mask):
        return values[name]

    return pick


def walk(shape=SHAPE, prev_end=None, mode=0, start=(10, 10), dirs=(0,), lens=(2,)):
    values = {
        "stroke_mode": mode,
        "stroke_start": start[1] * shape.grid + start[0],
        "stroke_segments": len(dirs) - 1,
    }
    for index, (d, n) in enumerate(zip(dirs, lens)):
        values[f"stroke_dir_{index}"] = d
        values[f"stroke_len_{index}"] = n
    return geo.walk_stroke(shape, prev_end, replay_pick(**values))


def samples_for(result, shape=SHAPE, count=12, side=SIDE):
    return codec.build_stroke_samples(result.path, shape, side, count, 5)


def distance_to_line(point, a, b) -> float:
    (px, py), (ax, ay), (bx, by) = point, a, b
    dx, dy = bx - ax, by - ay
    length = math.hypot(dx, dy)
    if length == 0:
        return math.hypot(px - ax, py - ay)
    return abs(dx * (ay - py) - (ax - px) * dy) / length


def rendered_ink(samples) -> np.ndarray:
    """samplesを実Engineへ描き、描かれた画素の真偽配列を返す。"""
    engine = Engine(SIDE, SIDE, RENDERER_V2, seed=1)
    budget = Budget(24)
    controller = StageController(engine, judge_style_intent("test"), budget)
    for action in (
        DrawingAction("select_layer", {"layer_id": "sys_rough_001"}),
        DrawingAction("select_color", {"color_id": 0}),
        DrawingAction("select_brush", {"brush_id": "ink_gpen"}),
        DrawingAction("set_brush_params", {"size_level_id": 4, "opacity_level_id": 10}),
        DrawingAction("stroke_v2", {"samples": samples}),
    ):
        result = controller.propose(action)
        assert result.accepted, result.reason
    rgba = np.frombuffer(engine.render().rgba, dtype=np.uint8).reshape(SIDE, SIDE, 4)
    return rgba[..., :3].sum(axis=2) < 3 * 250


def make_observation(side: int = 16, tools=()):
    engine = Engine(side, side, seed=1)
    budget = Budget(24)
    controller = StageController(engine, judge_style_intent("test"), budget)
    for action in tools:
        assert controller.propose(action).accepted
    text = np.linspace(-1, 1, 8, dtype=np.float32)
    return ObservationBuilder(tail_length=8, summary_side=4).build(text, controller, engine, budget)


def make_policy(**overrides) -> ParametricPolicy:
    config = TrainConfig(canvas_side=16, summary_side=4, max_actions=24, **overrides)
    return ParametricPolicy(config, 8, np.random.default_rng(0))


def random_pick(rng):
    def pick(name: str, mask):
        if name == "stroke_segments":
            return int(rng.integers(SHAPE.max_segments))
        return int(rng.choice(np.flatnonzero(mask)))

    return pick


def test_stroke_single_segment_is_straight() -> None:
    result = walk(start=(5, 10), dirs=(3,), lens=(4,))  # dir3=67.5度, len=8マス
    assert result.segments == 1 and len(result.path) == 2
    samples = samples_for(result)
    assert len(samples) == 12
    points = [(s["x_q8"], s["y_q8"]) for s in samples]
    for point in points:
        assert distance_to_line(point, points[0], points[-1]) <= CELL_Q8
    # 始点・終点は始点マス中心と整数変位で決まる。
    dx, dy = geo.displacement(SHAPE, 3, 4)
    assert (result.path[0], result.path[1]) == ((5, 10), (5 + dx, 10 + dy))


def test_stroke_directions_cover_16_angles() -> None:
    assert geo.DIRECTIONS == 16
    for direction in range(16):
        theta = math.radians(direction * 22.5)
        for length_index, length in enumerate(SHAPE.lengths):
            dx, dy = geo.displacement(SHAPE, direction, length_index)
            assert abs(dx - length * math.cos(theta)) <= 0.5 + 1e-9
            assert abs(dy - length * math.sin(theta)) <= 0.5 + 1e-9
    # +xが0番、yは下向き正で時計回り。
    assert geo.displacement(SHAPE, 0, 3) == (5, 0)
    assert geo.displacement(SHAPE, 4, 3) == (0, 5)
    assert geo.displacement(SHAPE, 8, 3) == (-5, 0)
    assert geo.displacement(SHAPE, 12, 3) == (0, -5)
    # 長さが十分大きいと実際の角度が22.5度刻みの理想角に近い(最大長)。
    for direction in range(16):
        dx, dy = geo.displacement(SHAPE, direction, len(SHAPE.lengths) - 1)
        angle = math.degrees(math.atan2(dy, dx)) % 360
        error = abs(angle - direction * 22.5)
        assert min(error, 360 - error) < 3.0
    # 端点は選んだ長さの通りに動く(始点から終点まで)。
    result = walk(start=(16, 16), dirs=(0,), lens=(3,))
    assert result.path[-1] == (21, 16)


def test_stroke_relative_direction_zero_is_straight() -> None:
    # 5+3マスの2本は8マスの1本と同じ変位。水平は(16,8)、斜めは(4,4)+(2,2)=(6,6)で(14,14)。
    for direction, end in ((0, (16, 8)), (2, (14, 14))):
        two = walk(start=(8, 8), dirs=(direction, 0), lens=(3, 2))
        one = walk(start=(8, 8), dirs=(direction,), lens=(4,))
        assert two.path[-1] == one.path[-1] == end
        a, b = samples_for(two), samples_for(one)
        line = [(s["x_q8"], s["y_q8"]) for s in b]
        for sample in a:
            point = (sample["x_q8"], sample["y_q8"])
            assert distance_to_line(point, line[0], line[-1]) <= CELL_Q8
        ink_a, ink_b = rendered_ink(a), rendered_ink(b)
        union = (ink_a | ink_b).sum()
        assert union > 0 and (ink_a & ink_b).sum() / union >= 0.95


def test_stroke_outside_canvas_is_masked() -> None:
    corner = geo.valid_moves(SHAPE, (0, 0))
    assert not corner[8].any() and not corner[12].any()  # 左・上
    assert corner[0].all() and corner[4].all()  # 右・下は13マスまで盤面内
    # 外へ出る選択をpickで返すと拒否される。
    with pytest.raises(ValueError):
        walk(start=(0, 0), dirs=(8,), lens=(0,))
    # 乱数でサンプルしても、全端点が格子内に収まる。
    rng = np.random.default_rng(3)
    for _ in range(300):
        result = geo.walk_stroke(SHAPE, None, random_pick(rng))
        for gx, gy in result.path:
            assert 0 <= gx < SHAPE.grid and 0 <= gy < SHAPE.grid


def test_stroke_samples_are_deterministic() -> None:
    result = walk(start=(12, 12), dirs=(5, 2, 15), lens=(3, 1, 2))
    first, second = samples_for(result), samples_for(result)
    assert first == second
    again = walk(start=(12, 12), dirs=(5, 2, 15), lens=(3, 1, 2))
    assert samples_for(again) == first
    # time_usは0から単調、筆圧は固定値、Replay(JSON)往復で不変。
    assert [s["time_us"] for s in first] == [i * codec.STROKE_POINT_STEP_US for i in range(12)]
    assert {s["pressure_level_id"] for s in first} == {5}
    assert json.loads(json.dumps(first)) == first
    # decode_action経由でも同じ。
    observation = make_observation()
    choice = codec.ActionChoice(
        4, stroke_mode=0, stroke_start=12 * 32 + 12, stroke_segments=3,
        stroke_dirs=(5, 2, 15), stroke_lens=(3, 1, 2),
    )  # fmt: skip
    action = codec.decode_action(choice, observation, SIDE, 12, 5)
    assert action.payload["samples"] == first


def test_stroke_continue_starts_at_previous_end() -> None:
    result = walk(mode=1, prev_end=(7, 9), dirs=(0,), lens=(1,))
    assert result.path[0] == (7, 9)
    assert [name for name, _i, _m in result.terms][:2] == ["stroke_mode", "stroke_segments"]
    q8 = samples_for(result)[0]
    expected = round((7 + 0.5) / SHAPE.grid * (SIDE - 1) * 256)
    assert q8["x_q8"] == expected
    # 直前がなければcontinueはマスされ、newだけが選べる。
    seen = {}

    def pick(name, mask):
        seen[name] = None if mask is None else mask.copy()
        return {"stroke_mode": 0, "stroke_start": 0, "stroke_segments": 0}.get(name, 0)

    geo.walk_stroke(SHAPE, None, pick)
    assert list(seen["stroke_mode"]) == [True, False]
    with pytest.raises(ValueError):
        walk(mode=1, prev_end=None, dirs=(0,), lens=(1,))
    geo.walk_stroke(SHAPE, (4, 4), pick)
    assert list(seen["stroke_mode"]) == [True, True]


def _stroke_sample(policy: ParametricPolicy, segments: int, mode: int = 0, prev_end=None):
    observation = make_observation()
    rng = np.random.default_rng(segments)

    def pick(name: str, mask):
        if name == "stroke_mode":
            return mode
        if name == "stroke_segments":
            return segments - 1
        if name == "stroke_start":
            return 16 * 32 + 16  # 中央から短く動かし、どのマスクも複数の選択肢を持たせる
        if name.startswith("stroke_len_"):
            return int(np.flatnonzero(mask)[0])
        return int(rng.choice(np.flatnonzero(mask)))

    walked = geo.walk_stroke(policy.config.stroke_shape, prev_end, pick)
    choice = codec.ActionChoice(
        4, stroke_mode=walked.mode, stroke_start=walked.start_cell,
        stroke_segments=walked.requested, stroke_dirs=walked.dirs, stroke_lens=walked.lens,
    )  # fmt: skip
    # 勾配の検査なので種別は全て選べるマスクにする(実際のマスクはtest_legal_mask.py)。
    kinds, layers = np.ones(6, dtype=bool), codec.layer_mask(observation)
    return PolicySample(
        build_features(observation), "rough", kinds, layers, choice, stroke_prev_end=prev_end
    ), walked


def test_stroke_policy_gradient_matches_numeric() -> None:
    policy = make_policy()
    rng = np.random.default_rng(1)
    for segments, mode, prev_end in ((1, 0, None), (3, 1, (6, 9)), (4, 0, None)):
        sample, _walked = _stroke_sample(policy, segments, mode, prev_end)
        grads, entropy_grads = policy.new_grads(), policy.new_grads()
        policy.accumulate_log_prob_grad(sample, 1.0, grads)
        policy.accumulate_entropy_grad(sample, 1.0, entropy_grads)
        for name, array in policy.params.items():
            if not name.startswith("stroke_"):
                continue
            for _ in range(8):
                index = tuple(int(rng.integers(0, size)) for size in array.shape)
                original = array[index]
                numeric = []
                for delta in (1e-5, -1e-5):
                    array[index] = original + delta
                    logp = policy.accumulate_log_prob_grad(sample, 0.0, policy.new_grads())
                    entropy = policy.accumulate_entropy_grad(sample, 0.0, policy.new_grads())
                    numeric.append((logp, entropy))
                array[index] = original
                assert (numeric[0][0] - numeric[1][0]) / 2e-5 == pytest.approx(
                    grads.arrays[name][index], abs=1e-6
                ), name
                assert (numeric[0][1] - numeric[1][1]) / 2e-5 == pytest.approx(
                    entropy_grads.arrays[name][index], abs=1e-6
                ), name


def test_stroke_mask_is_sequential_and_inclusive_of_border() -> None:
    shape = geo.StrokeShape(grid=14, max_segments=4, lengths=(1, 13))
    # 境界上(インデックス13)への移動は有効。0から13マス右は有効、14は盤面外。
    assert geo.valid_moves(shape, (0, 3))[0][1]
    assert not geo.valid_moves(shape, (1, 3))[0][1]
    # 逐次: 1本目の終点(13,3)から右へは動けないが、左へは動ける。
    seen = {}

    def pick(name, mask):
        seen[name] = None if mask is None else mask.copy()
        return {
            "stroke_mode": 0, "stroke_start": 3 * 14 + 0, "stroke_segments": 1,
            "stroke_dir_0": 0, "stroke_len_0": 1, "stroke_dir_1": 8, "stroke_len_1": 0,
        }[name]  # fmt: skip

    result = geo.walk_stroke(shape, None, pick)
    assert result.path == ((0, 3), (13, 3), (12, 3))
    # 2本目の相対方向0(直進=右)は(13,3)からは出てしまう。反転(8)は有効。
    assert not seen["stroke_dir_1"][0] and seen["stroke_dir_1"][8]


def test_stroke_all_invalid_truncates_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    real = geo.valid_moves
    calls = {"count": 0}

    def fake(shape, pos):
        calls["count"] += 1
        if calls["count"] == 1:
            return real(shape, pos)
        return np.zeros((geo.DIRECTIONS, len(shape.lengths)), dtype=bool)

    monkeypatch.setattr(geo, "valid_moves", fake)

    def pick(name, mask):
        return {"stroke_mode": 0, "stroke_start": 5, "stroke_segments": 3}.get(name, 0)

    result = geo.walk_stroke(SHAPE, None, pick)
    assert result.requested == 4 and result.segments == 1
    assert result.truncated == result.requested - result.segments
    monkeypatch.undo()
    # 0本のときはstroke_v2を選べない。
    tiny = geo.StrokeShape(grid=1, max_segments=2, lengths=(1,))
    observation = make_observation()
    tools = (
        DrawingAction("select_layer", {"layer_id": "sys_rough_001"}),
        DrawingAction("select_color", {"color_id": 0}),
        DrawingAction("select_brush", {"brush_id": "ink_gpen"}),
    )
    observation = make_observation(tools=tools)
    assert not codec.kind_mask(observation, tiny)[codec.ACTION_KINDS.index("stroke_v2")]
    assert codec.kind_mask(observation, SHAPE)[codec.ACTION_KINDS.index("stroke_v2")]
    with pytest.raises(ValueError):
        geo.walk_stroke(tiny, None, pick)


def test_stroke_unused_heads_excluded_from_logprob_and_entropy() -> None:
    policy = make_policy()
    for segments in range(1, policy.config.stroke_max_segments + 1):
        sample, walked = _stroke_sample(policy, segments, mode=0)
        grads, entropy_grads = policy.new_grads(), policy.new_grads()
        logp = policy.accumulate_log_prob_grad(sample, 1.0, grads)
        entropy = policy.accumulate_entropy_grad(sample, 1.0, entropy_grads)
        names = [name for name, _i, _m in walked.terms]
        used = {f"stroke_dir_{i}" for i in range(segments)} | {
            f"stroke_len_{i}" for i in range(segments)
        }
        for i in range(policy.config.stroke_max_segments):
            for head in (f"stroke_dir_{i}", f"stroke_len_{i}"):
                active = head in used
                assert (head in names) == active
                assert bool(np.any(grads.arrays[head])) == active, head
                assert bool(np.any(entropy_grads.arrays[head])) == active, head
        # newなのでstartは含まれ、continueなら含まれない。
        assert "stroke_start" in names
        features = sample.features
        expected_logp = expected_entropy = 0.0
        for name, index, mask in walked.terms:
            probabilities = policy._probabilities(name, features, mask)
            expected_logp += math.log(probabilities[index])
        # kindの項(stroke_v2のlog確率とentropy)を除いた分が一致する。
        kind_probs = policy._probabilities("kind", features, sample.kind_mask)
        kind_entropy = -(kind_probs[kind_probs > 0] * np.log(kind_probs[kind_probs > 0])).sum()
        expected_logp += math.log(kind_probs[4])
        for name, _index, mask in walked.terms:
            p = policy._probabilities(name, features, mask)
            expected_entropy += -(p[p > 0] * np.log(p[p > 0])).sum()
        expected_entropy += kind_entropy
        assert logp == pytest.approx(expected_logp)
        assert entropy == pytest.approx(expected_entropy)
    continued, walked = _stroke_sample(policy, 2, mode=1, prev_end=(4, 4))
    assert "stroke_start" not in [name for name, _i, _m in walked.terms]
    grads = policy.new_grads()
    policy.accumulate_log_prob_grad(continued, 1.0, grads)
    assert not grads.arrays["stroke_start"].any() and grads.arrays["stroke_mode"].any()


def test_stroke_diagonal_rounding_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    def half_away(value: float) -> int:
        return int(math.copysign(math.floor(abs(value) + 0.5), value))

    for direction in range(16):
        theta = direction * 22.5 * math.pi / 180
        for index, length in enumerate(SHAPE.lengths):
            expected = (half_away(length * math.cos(theta)), half_away(length * math.sin(theta)))
            assert geo.displacement(SHAPE, direction, index) == expected
            assert geo.displacement(SHAPE, direction, index) == expected  # 再現性
    assert geo.displacement(SHAPE, 2, 0) == (1, 1) and geo.displacement(SHAPE, 1, 0) == (1, 0)
    assert geo.displacement(SHAPE, 6, 3) == (-4, 4)  # 5*cos(135度)=-3.54 -> -4
    # 変位が(0,0)になる選択は無効。
    zero = np.zeros((geo.DIRECTIONS, len(SHAPE.lengths), 2), dtype=int)
    monkeypatch.setattr(geo, "displacement_table", lambda lengths: zero)
    assert not geo.valid_moves(SHAPE, (10, 10)).any()


def test_stroke_sample_budget_rejects_small_stroke_samples() -> None:
    with pytest.raises(ValueError):
        TrainConfig(stroke_max_segments=4, stroke_samples=4)
    assert TrainConfig(stroke_max_segments=4, stroke_samples=5).stroke_samples == 5
    with pytest.raises(ValueError):
        TrainConfig(stroke_lengths=(0, 1))
    result = walk(start=(2, 2), dirs=(0, 0, 0, 0), lens=(5, 1, 3, 0))  # 13,2,5,1マス
    assert len(result.path) == 5
    with pytest.raises(ValueError):
        codec.build_stroke_samples(result.path, SHAPE, SIDE, 4, 5)
    samples = codec.build_stroke_samples(result.path, SHAPE, SIDE, 12, 5)
    assert len(samples) == 12
    # 余り7点は長い線分から順に配る: 長さ順(13,5,2,1)に 2,2,2,1 ではなく巡回配分。
    xs = [s["x_q8"] for s in samples]
    assert xs == sorted(xs)
    vertex_x = [round((gx + 0.5) / SHAPE.grid * (SIDE - 1) * 256) for gx, _gy in result.path]
    inner = [sum(1 for x in xs if a < x < b) for a, b in zip(vertex_x, vertex_x[1:])]
    assert sum(inner) == 12 - 5 and inner[0] >= inner[1] >= inner[3]
    assert set(vertex_x) <= set(xs)


def _checkpoint_dir(tmp_path, config):
    policy = ParametricPolicy(config, 8, np.random.default_rng(0))
    encoder = HashedNgramTextEncoder(dim=8)
    save_checkpoint(
        tmp_path, 1, policy, AdamState(policy.state_dict()), BaselineTable(), config, encoder
    )
    return policy, encoder


def test_checkpoint_rejects_old_stroke_head(tmp_path: pathlib.Path) -> None:
    config = TrainConfig(canvas_side=16, summary_side=4, max_actions=24)
    policy, encoder = _checkpoint_dir(tmp_path, config)
    sidecar = tmp_path / "iter_1.json"
    fresh = ParametricPolicy(config, 8, np.random.default_rng(5))
    assert (
        load_checkpoint(
            sidecar, fresh, AdamState(fresh.state_dict()), BaselineTable(), encoder, config
        )
        == 1
    )
    assert fresh.params_sha256() == policy.params_sha256()
    # 旧構成(head構成hashなし)は理由つきで拒否する。
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    del meta["stroke_head_hash"]
    sidecar.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpointのstroke head構成が一致しません"):
        load_checkpoint(
            sidecar, fresh, AdamState(fresh.state_dict()), BaselineTable(), encoder, config
        )
    # 構成(格子)が違うPolicyにも拒否する。
    other_config = TrainConfig(canvas_side=16, summary_side=4, max_actions=24, stroke_grid=16)
    _checkpoint_dir(tmp_path / "other", other_config)
    other = ParametricPolicy(config, 8, np.random.default_rng(5))
    with pytest.raises(CheckpointMismatchError, match="stroke head構成"):
        load_checkpoint(
            tmp_path / "other" / "iter_1.json",
            other,
            AdamState(other.state_dict()),
            BaselineTable(),
            encoder,
            config,
        )


def test_policy_stroke_has_no_target_input() -> None:
    package = pathlib.Path(geo.__file__).parent
    banned = ("dataset", "reward", "rollout", "evaluation", "trainer", "integrity")
    tree = ast.parse((package / "stroke_geometry.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            assert not any(part.startswith(banned) for part in name.split(".")), name
    forbidden = {"target", "reward", "image", "path_target", "target_image"}
    for function in (
        ParametricPolicy.act,
        geo.walk_stroke,
        codec.decode_action,
        codec.build_stroke_samples,
    ):
        assert not forbidden & set(inspect.signature(function).parameters)
    fields = {field.name for field in codec.ActionChoice.__dataclass_fields__.values()}
    assert not any("target" in name or "image" in name for name in fields)
    # 実際にactしたストロークの選択は観測と乱数だけで決まる。
    policy = make_policy()
    observation = make_observation()
    first = policy.act(observation, np.random.default_rng(4))
    again = policy.act(observation, np.random.default_rng(4))
    assert first.sample.choice == again.sample.choice and first.action == again.action

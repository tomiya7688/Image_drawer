"""DrawingPolicy(線形softmaxの各head、ストロークは離散)。入力はObservationと乱数だけ。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray

from image_drawer.drawing import DrawingAction
from image_drawer.training.action_codec import (
    ActionChoice,
    brush_mask,
    decode_action,
    kind_mask,
    layer_mask,
    resolve_stroke,
)
from image_drawer.training.config import TrainConfig
from image_drawer.training.observation import ACTION_KINDS, Observation
from image_drawer.training.policy_features import build_features, feature_dim
from image_drawer.training.policy_params import (
    HEAD_SIZES,
    Params,
    PolicyGrads,
    categorical_entropy,
    init_params,
    masked_softmax,
    params_sha256,
)
from image_drawer.training.stroke_geometry import GridPoint, StrokeShape, StrokeWalk, walk_stroke

POLICY_VERSION = "parametric-softmax-stroke-v2"
MLP_POLICY_VERSION = "parametric-softmax-stroke-mlp-v1"
TRUNK = "trunk"  # 隠れ層の重み(隠れ幅 x 特徴次元)のparams名
PRESSURE_LEVEL = 5  # 筆圧は固定(中央のlevel)。設計書4.2節。


# {
#   責務: [
#     PolicySample: 勾配計算に必要な特徴量・マスク・選択結果・直前ストロークの終点
#   ]
#   フィールド: [ stroke_prev_end: 選択時点の直前ストロークの終点(continueとマスの再計算用) ]
# }
@dataclass(frozen=True)
class PolicySample:
    features: NDArray[np.float64]
    stage_id: str
    kind_mask: NDArray[np.bool_]
    layer_mask: NDArray[np.bool_]
    choice: ActionChoice
    stroke_prev_end: GridPoint | None = None


# {
#   責務: [ PolicyDecision: 提案するAction(actor=model)とその勾配用サンプル ]
# }
@dataclass(frozen=True)
class PolicyDecision:
    action: DrawingAction
    sample: PolicySample


# {
#   責務: [
#     DrawingPolicy: 描画Policyの契約。actの引数はObservationと乱数だけで、目標・Rewardを受け取らない
#   ]
# }
class DrawingPolicy(Protocol):
    policy_version: str

    def act(self, observation: Observation, rng: np.random.Generator) -> PolicyDecision: ...

    def accumulate_log_prob_grad(
        self, sample: PolicySample, weight: float, grads: PolicyGrads
    ) -> float: ...

    def accumulate_entropy_grad(
        self, sample: PolicySample, weight: float, grads: PolicyGrads
    ) -> float: ...

    def new_grads(self) -> PolicyGrads: ...

    def apply_update(self, deltas: Mapping[str, NDArray[np.float64]]) -> None: ...

    def state_dict(self) -> dict[str, NDArray[np.float64]]: ...

    def load_state_dict(self, state: Mapping[str, NDArray[np.float64]]) -> None: ...

    def params_sha256(self) -> str: ...


# {
#   責務: [
#     StrokeTracker: 直前に受理されたストロークの終点を覚える(continue用)。観測のaction_tailだけで同期する
#   ]
#   処理: [ action_tailが空なら新しいtrialとして忘れる。提案したストロークは次の観測で受理を確認する ]
# }
class StrokeTracker:
    def __init__(self) -> None:
        self.last_end: GridPoint | None = None
        self.pending_end: GridPoint | None = None

    # {
    #   責務: [ previous_end: 観測に合わせて状態を更新し、選択時点の直前ストロークの終点を返す ]
    # }
    def previous_end(self, observation: Observation) -> GridPoint | None:
        tail = observation.action_tail
        if not tail:
            self.last_end = None
        elif self.pending_end is not None and tail[-1].kind == "stroke_v2" and tail[-1].accepted:
            self.last_end = self.pending_end
        self.pending_end = None
        return self.last_end

    def propose(self, end: GridPoint) -> None:
        self.pending_end = end


# {
#   責務: [ stroke_fields: StrokeWalkをActionChoiceのストローク欄にする ]
# }
def stroke_fields(walked: StrokeWalk) -> dict[str, object]:
    return {
        "stroke_mode": walked.mode,
        "stroke_start": walked.start_cell,
        "stroke_segments": walked.requested,
        "stroke_dirs": walked.dirs,
        "stroke_lens": walked.lens,
    }


# {
#   責務: [
#     categorical_terms: 選んだ種別が使うカテゴリ分布(head名, 選択, マスク)の列を返す。
#     ストロークは使う線分のhead・使うstartだけ(逐次マスはwalk_strokeで再現する)
#   ]
# }
def categorical_terms(
    sample: PolicySample, shape: StrokeShape
) -> list[tuple[str, int, NDArray[np.bool_] | None]]:
    choice = sample.choice
    terms: list[tuple[str, int, NDArray[np.bool_] | None]] = [
        ("kind", choice.kind_index, sample.kind_mask)
    ]
    kind = ACTION_KINDS[choice.kind_index]
    if kind == "select_layer":
        terms.append(("layer", int(choice.layer_index), sample.layer_mask))
    elif kind == "select_color":
        terms.append(("color", int(choice.color_id), None))
    elif kind == "select_brush":
        terms.append(("brush", int(choice.brush_index), brush_mask()))
    elif kind == "set_brush_params":
        terms.append(("size", int(choice.size_level), None))
        terms.append(("opacity", int(choice.opacity_level), None))
    elif kind == "stroke_v2":
        terms.extend(resolve_stroke(choice, shape, sample.stroke_prev_end).terms)
    return terms


# {
#   責務: [ all_head_sizes: 固定headとストロークheadを合わせた出力数 ]
# }
def all_head_sizes(shape: StrokeShape) -> dict[str, int]:
    return {**HEAD_SIZES, **shape.head_sizes()}


# {
#   責務: [
#     ParametricPolicy: 線形softmaxのheadを持つPolicy(ストロークは始点マス+方向x長さの離散)。
#     DrawingPolicyを実装する(継承しない)
#   ]
#   フィールド: [ config, params: head名->重み, 温度はconfig.softmax_temperature ]
# }
class ParametricPolicy:
    def __init__(self, config: TrainConfig, text_dim: int, rng: np.random.Generator) -> None:
        self.config = config
        self.text_dim = text_dim
        self.feature_dim = feature_dim(text_dim, config.summary_side)
        self.hidden = config.policy_hidden
        self.policy_version = MLP_POLICY_VERSION if self.hidden else POLICY_VERSION
        self.params: Params = init_params(
            self.feature_dim + self.hidden, rng, all_head_sizes(config.stroke_shape)
        )
        if self.hidden:
            self.params[TRUNK] = rng.normal(
                0.0, 1.0 / math.sqrt(self.feature_dim), size=(self.hidden, self.feature_dim)
            )
        self.tracker = StrokeTracker()

    # {
    #   責務: [ _expand: 特徴に隠れ層(tanh(trunk x 特徴))を連結する。線形ならそのまま ]
    #   戻り値: [ (headへの入力, 隠れ層の出力またはNone) ]
    # }
    def _expand(
        self, features: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
        if not self.hidden:
            return features, None
        hidden = np.tanh(self.params[TRUNK] @ features)
        return np.concatenate([features, hidden]), hidden

    # {
    #   責務: [ _add_grad: headの勾配を足し、隠れ層があればtrunkへ逆伝播する ]
    # }
    def _add_grad(
        self,
        grads: PolicyGrads,
        name: str,
        output_grad: NDArray[np.float64],
        inputs: NDArray[np.float64],
        features: NDArray[np.float64],
        hidden: NDArray[np.float64] | None,
    ) -> None:
        grads.add(name, output_grad, inputs)
        if hidden is not None:
            back = self.params[name][:, self.feature_dim :].T @ output_grad
            grads.add(TRUNK, back * (1.0 - hidden * hidden), features)

    # {
    #   責務: [ _logits: headの出力(重み x 特徴) ]
    # }
    def _logits(self, name: str, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.params[name] @ features

    # {
    #   責務: [ _probabilities: 温度付きのマスク付きsoftmax(logits / T) ]
    # }
    def _probabilities(
        self, name: str, features: NDArray[np.float64], mask: NDArray[np.bool_] | None
    ) -> NDArray[np.float64]:
        return masked_softmax(self._logits(name, features) / self.config.softmax_temperature, mask)

    # {
    #   責務: [ act: 観測から種別とパラメータをサンプルし、actor=modelのActionにする ]
    #   処理: [ 1: 特徴化 2: 種別をマスク付きsoftmaxでサンプル 3: 種別が使うheadだけサンプル 4: 変換 ]
    # }
    def act(self, observation: Observation, rng: np.random.Generator) -> PolicyDecision:
        features = build_features(observation)
        inputs, _hidden = self._expand(features)
        shape = self.config.stroke_shape
        kinds, layers = kind_mask(observation, shape), layer_mask(observation)
        prev_end = self.tracker.previous_end(observation)

        def draw(name: str, mask: NDArray[np.bool_] | None = None) -> int:
            probabilities = self._probabilities(name, inputs, mask)
            return int(rng.choice(len(probabilities), p=probabilities))

        kind_index = draw("kind", kinds)
        kind = ACTION_KINDS[kind_index]
        fields: dict[str, object] = {}
        if kind == "select_layer":
            fields["layer_index"] = draw("layer", layers)
        elif kind == "select_color":
            fields["color_id"] = draw("color")
        elif kind == "select_brush":
            fields["brush_index"] = draw("brush", brush_mask())
        elif kind == "set_brush_params":
            fields["size_level"] = draw("size")
            fields["opacity_level"] = draw("opacity")
        elif kind == "stroke_v2":
            walked = walk_stroke(shape, prev_end, draw)
            fields = stroke_fields(walked)
            self.tracker.propose(walked.path[-1])
        choice = ActionChoice(kind_index, **fields)  # type: ignore[arg-type]
        sample = PolicySample(features, observation.stage_id, kinds, layers, choice, prev_end)
        action = decode_action(
            choice,
            observation,
            self.config.canvas_side,
            self.config.stroke_samples,
            PRESSURE_LEVEL,
            shape,
            prev_end,
        )
        return PolicyDecision(action, sample)

    # {
    #   責務: [ accumulate_log_prob_grad: weight x grad log pi を足し、log piを返す ]
    #   処理: [ 各カテゴリは (onehot - p) / T を特徴との外積でgradsへ足す。使わないheadは0 ]
    # }
    def accumulate_log_prob_grad(
        self, sample: PolicySample, weight: float, grads: PolicyGrads
    ) -> float:
        total = 0.0
        temperature = self.config.softmax_temperature
        inputs, hidden = self._expand(sample.features)
        for name, index, mask in categorical_terms(sample, self.config.stroke_shape):
            probabilities = self._probabilities(name, inputs, mask)
            total += math.log(probabilities[index])
            output_grad = -probabilities
            output_grad[index] += 1.0
            self._add_grad(
                grads, name, weight * output_grad / temperature, inputs, sample.features, hidden
            )
        return total

    # {
    #   責務: [ accumulate_entropy_grad: 選んだ種別のカテゴリ分布のエントロピー勾配をweight倍で足す ]
    #   戻り値: [ entropy: 使ったカテゴリ分布のエントロピーの合計 ]
    # }
    def accumulate_entropy_grad(
        self, sample: PolicySample, weight: float, grads: PolicyGrads
    ) -> float:
        total = 0.0
        temperature = self.config.softmax_temperature
        inputs, hidden = self._expand(sample.features)
        for name, _index, mask in categorical_terms(sample, self.config.stroke_shape):
            probabilities = self._probabilities(name, inputs, mask)
            entropy = categorical_entropy(probabilities)
            total += entropy
            safe = np.where(probabilities > 0, probabilities, 1.0)
            self._add_grad(
                grads,
                name,
                -weight * probabilities * (np.log(safe) + entropy) / temperature,
                inputs,
                sample.features,
                hidden,
            )
        return total

    def new_grads(self) -> PolicyGrads:
        return PolicyGrads(self.params)

    def apply_update(self, deltas: Mapping[str, NDArray[np.float64]]) -> None:
        for name, delta in deltas.items():
            self.params[name] = self.params[name] + delta

    def state_dict(self) -> dict[str, NDArray[np.float64]]:
        return {name: value.copy() for name, value in self.params.items()}

    # {
    #   責務: [ load_state_dict: 重みを読み込む。名前・形状が違えば拒否する ]
    #   エラー: [ ValueError: head名または形状の不一致 ]
    # }
    def load_state_dict(self, state: Mapping[str, NDArray[np.float64]]) -> None:
        if set(state) != set(self.params):
            raise ValueError(f"head名が一致しません: {sorted(state)}")
        for name, value in state.items():
            if value.shape != self.params[name].shape:
                raise ValueError(f"{name}の形状が違います: {value.shape}")
        self.params = {name: np.array(value, dtype=np.float64) for name, value in state.items()}

    def params_sha256(self) -> str:
        return params_sha256(self.params)


# {
#   責務: [
#     RandomLegalPolicy: 工程別の候補Actionから一様に選ぶベースライン(学習しない)
#   ]
#   処理: [ 種別・色・ブラシ・段階・レイヤーは一様、ストロークは逐次マスで許可された選択肢から一様 ]
# }
class RandomLegalPolicy:
    policy_version = "random-legal-v2"

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.tracker = StrokeTracker()

    def act(self, observation: Observation, rng: np.random.Generator) -> PolicyDecision:
        shape = self.config.stroke_shape
        kinds, layers = kind_mask(observation, shape), layer_mask(observation)
        prev_end = self.tracker.previous_end(observation)
        kind_index = int(rng.choice(np.flatnonzero(kinds)))
        kind = ACTION_KINDS[kind_index]

        def uniform(name: str, mask: NDArray[np.bool_] | None) -> int:
            if mask is None:  # segments head(マスなし)
                return int(rng.integers(shape.max_segments))
            return int(rng.choice(np.flatnonzero(mask)))

        def stroke() -> dict[str, object]:
            walked = walk_stroke(shape, prev_end, uniform)
            self.tracker.propose(walked.path[-1])
            return stroke_fields(walked)

        pick: dict[str, Callable[[], dict[str, object]]] = {
            "select_layer": lambda: {"layer_index": int(rng.choice(np.flatnonzero(layers)))},
            "select_color": lambda: {"color_id": int(rng.integers(HEAD_SIZES["color"]))},
            "select_brush": lambda: {"brush_index": int(rng.choice(np.flatnonzero(brush_mask())))},
            "set_brush_params": lambda: {
                "size_level": int(rng.integers(HEAD_SIZES["size"])),
                "opacity_level": int(rng.integers(HEAD_SIZES["opacity"])),
            },
            "stroke_v2": stroke,
        }
        fields = pick.get(kind, lambda: {})()
        choice = ActionChoice(kind_index, **fields)  # type: ignore[arg-type]
        features = build_features(observation)
        sample = PolicySample(features, observation.stage_id, kinds, layers, choice, prev_end)
        action = decode_action(
            choice,
            observation,
            self.config.canvas_side,
            self.config.stroke_samples,
            PRESSURE_LEVEL,
            shape,
            prev_end,
        )
        return PolicyDecision(action, sample)

    def params_sha256(self) -> str:
        return params_sha256({})

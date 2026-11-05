"""DrawingPolicy(線形softmax+ガウスの各head)。入力はObservationと乱数だけ。"""

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
    decode_action,
    kind_mask,
    layer_mask,
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

POLICY_VERSION = "parametric-softmax-gauss-v1"
PRESSURE_LEVEL = 5  # 筆圧は固定(中央のlevel)。設計書4.2節。
_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


# {
#   責務: [ PolicySample: 勾配計算に必要な特徴量・マスク・選択結果・ガウス幅 ]
# }
@dataclass(frozen=True)
class PolicySample:
    features: NDArray[np.float64]
    stage_id: str
    kind_mask: NDArray[np.bool_]
    layer_mask: NDArray[np.bool_]
    choice: ActionChoice
    sigma: float


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
    stroke_sigma: float

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
#   責務: [ categorical_terms: 選んだ種別が使うカテゴリ分布(head名, 選択, マスク)の列を返す ]
# }
def categorical_terms(
    sample: PolicySample,
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
        terms.append(("brush", int(choice.brush_index), None))
    elif kind == "set_brush_params":
        terms.append(("size", int(choice.size_level), None))
        terms.append(("opacity", int(choice.opacity_level), None))
    return terms


# {
#   責務: [
#     ParametricPolicy: 線形softmaxとガウスのheadを持つPolicy。DrawingPolicyを実装する(継承しない)
#   ]
#   フィールド: [ config, params: head名->重み, stroke_sigma: 現在のガウス幅 ]
# }
class ParametricPolicy:
    policy_version = POLICY_VERSION

    def __init__(self, config: TrainConfig, text_dim: int, rng: np.random.Generator) -> None:
        self.config = config
        self.text_dim = text_dim
        self.feature_dim = feature_dim(text_dim, config.summary_side)
        self.params: Params = init_params(self.feature_dim, rng)
        self.stroke_sigma = config.sigma_start

    # {
    #   責務: [ _logits: headの出力(重み x 特徴) ]
    # }
    def _logits(self, name: str, features: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.params[name] @ features

    # {
    #   責務: [ act: 観測から種別とパラメータをサンプルし、actor=modelのActionにする ]
    #   処理: [ 1: 特徴化 2: 種別をマスク付きsoftmaxでサンプル 3: 種別が使うheadだけサンプル 4: 変換 ]
    # }
    def act(self, observation: Observation, rng: np.random.Generator) -> PolicyDecision:
        features = build_features(observation)
        kinds, layers = kind_mask(observation), layer_mask(observation)
        sigma = self.stroke_sigma

        def draw(name: str, mask: NDArray[np.bool_] | None = None) -> int:
            probabilities = masked_softmax(self._logits(name, features), mask)
            return int(rng.choice(len(probabilities), p=probabilities))

        kind_index = draw("kind", kinds)
        kind = ACTION_KINDS[kind_index]
        fields: dict[str, object] = {}
        if kind == "select_layer":
            fields["layer_index"] = draw("layer", layers)
        elif kind == "select_color":
            fields["color_id"] = draw("color")
        elif kind == "select_brush":
            fields["brush_index"] = draw("brush")
        elif kind == "set_brush_params":
            fields["size_level"] = draw("size")
            fields["opacity_level"] = draw("opacity")
        elif kind == "stroke_v2":
            mean = self._logits("stroke", features)
            latent = mean + sigma * rng.standard_normal(HEAD_SIZES["stroke"])
            fields["stroke_u"] = tuple(float(value) for value in latent)
        choice = ActionChoice(kind_index, **fields)  # type: ignore[arg-type]
        sample = PolicySample(features, observation.stage_id, kinds, layers, choice, sigma)
        action = decode_action(
            choice, observation, self.config.canvas_side, self.config.stroke_samples, PRESSURE_LEVEL
        )
        return PolicyDecision(action, sample)

    # {
    #   責務: [ accumulate_log_prob_grad: weight x grad log pi を足し、log piを返す ]
    #   処理: [ 各カテゴリは onehot - p、ガウスは (u-mu)/sigma^2 を特徴との外積でgradsへ足す ]
    # }
    def accumulate_log_prob_grad(
        self, sample: PolicySample, weight: float, grads: PolicyGrads
    ) -> float:
        total = 0.0
        for name, index, mask in categorical_terms(sample):
            probabilities = masked_softmax(self._logits(name, sample.features), mask)
            total += math.log(probabilities[index])
            output_grad = -probabilities
            output_grad[index] += 1.0
            grads.add(name, weight * output_grad, sample.features)
        if sample.choice.stroke_u is not None:
            mean = self._logits("stroke", sample.features)
            residual = np.asarray(sample.choice.stroke_u) - mean
            sigma = sample.sigma
            total += float((-0.5 * (residual / sigma) ** 2 - math.log(sigma) - _LOG_SQRT_2PI).sum())
            grads.add("stroke", weight * residual / sigma**2, sample.features)
        return total

    # {
    #   責務: [ accumulate_entropy_grad: 選んだ種別のカテゴリ分布のエントロピー勾配をweight倍で足す ]
    #   戻り値: [ entropy: カテゴリ分布のエントロピーの合計(ガウス幅は固定のため0寄与) ]
    # }
    def accumulate_entropy_grad(
        self, sample: PolicySample, weight: float, grads: PolicyGrads
    ) -> float:
        total = 0.0
        for name, _index, mask in categorical_terms(sample):
            probabilities = masked_softmax(self._logits(name, sample.features), mask)
            entropy = categorical_entropy(probabilities)
            total += entropy
            safe = np.where(probabilities > 0, probabilities, 1.0)
            grads.add(name, -weight * probabilities * (np.log(safe) + entropy), sample.features)
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
#   処理: [ 種別・色・ブラシ・段階・レイヤーは一様、ストロークの座標は[0,1]の一様乱数 ]
# }
class RandomLegalPolicy:
    policy_version = "random-legal-v1"

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.stroke_sigma = config.sigma_start

    def act(self, observation: Observation, rng: np.random.Generator) -> PolicyDecision:
        kinds, layers = kind_mask(observation), layer_mask(observation)
        kind_index = int(rng.choice(np.flatnonzero(kinds)))
        kind = ACTION_KINDS[kind_index]
        pick: dict[str, Callable[[], dict[str, object]]] = {
            "select_layer": lambda: {"layer_index": int(rng.choice(np.flatnonzero(layers)))},
            "select_color": lambda: {"color_id": int(rng.integers(HEAD_SIZES["color"]))},
            "select_brush": lambda: {"brush_index": int(rng.integers(HEAD_SIZES["brush"]))},
            "set_brush_params": lambda: {
                "size_level": int(rng.integers(HEAD_SIZES["size"])),
                "opacity_level": int(rng.integers(HEAD_SIZES["opacity"])),
            },
            # sigmoidの逆関数で一様座標を潜在値にする(端の0/1は避ける)。
            "stroke_v2": lambda: {
                "stroke_u": tuple(
                    float(math.log(p / (1 - p)))
                    for p in rng.uniform(1e-6, 1 - 1e-6, HEAD_SIZES["stroke"])
                )
            },
        }
        fields = pick.get(kind, lambda: {})()
        choice = ActionChoice(kind_index, **fields)  # type: ignore[arg-type]
        features = build_features(observation)
        sample = PolicySample(features, observation.stage_id, kinds, layers, choice, 1.0)
        action = decode_action(
            choice, observation, self.config.canvas_side, self.config.stroke_samples, PRESSURE_LEVEL
        )
        return PolicyDecision(action, sample)

    def params_sha256(self) -> str:
        return params_sha256({})

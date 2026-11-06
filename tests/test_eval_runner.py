"""評価ランナー(同条件の生成・失敗の保持・終了コード・比較)を確認する。

ここで使うPolicyは配線を確かめる試験用で、モデルの精度や学習の効果を示すものではない。
"""

import numpy as np
import pytest
from PIL import Image

from image_drawer.dataset import PolicyInput, RewardTarget
from image_drawer.drawing import DrawingAction
from image_drawer.generalization.compare import compare_conditions
from image_drawer.generalization.controls import PROBES, probe_hit_rate_control
from image_drawer.generalization.runner import RunOutput, run_generations
from image_drawer.generalization.stats import inference_seed
from image_drawer.training.action_codec import ActionChoice, layer_mask
from image_drawer.training.config import TrainConfig
from image_drawer.training.observation import ObservationBuilder
from image_drawer.training.policy import PolicyDecision, PolicySample, RandomLegalPolicy
from image_drawer.training.reward import RewardEvaluator
from image_drawer.training.rollout import RolloutRunner
from image_drawer.training.text_encoder import get_text_encoder

CONFIG = TrainConfig(canvas_side=16, max_actions=80)


# {
#   責務: [
#     PaintPolicy: 試験用。各工程で最初の書込レイヤーを選び、黒で全面を塗って次へ進む(目標画像は見ない)
#   ]
# }
class PaintPolicy:
    policy_version = "test-paint-v1"

    def act(self, observation, rng) -> PolicyDecision:
        tool, tail = observation.tool_state, observation.action_tail
        last = tail[-1] if tail else None
        moved = last is not None and last.kind == "advance_stage" and last.accepted
        rejected = last is not None and not last.accepted
        if not tool.layer_selected or moved or rejected:
            legal = np.flatnonzero(layer_mask(observation))
            layer_id = observation.layer_summary[int(legal[0])].layer_id
            action = DrawingAction("select_layer", {"layer_id": layer_id})
        elif not tool.brush_selected or (last and last.kind == "select_layer"):
            action = DrawingAction("select_brush", {"brush_id": "ink_gpen"})
        elif not tool.color_selected or (last and last.kind == "select_brush"):
            action = DrawingAction(
                "select_color", {"color_id": 1 if observation.stage_id == "refine" else 0}
            )
        elif not tool.size_set or (last and last.kind == "select_color"):
            action = DrawingAction(
                "set_brush_params", {"size_level_id": 18, "opacity_level_id": 10}
            )
        elif last is not None and last.kind == "stroke_v2":
            action = DrawingAction("advance_stage", {})
        else:
            sample = {
                "x_q8": 8 * 256, "y_q8": 8 * 256, "time_us": 0, "pressure_level_id": 10,
                "tilt_x_level_id": None, "tilt_y_level_id": None,
            }  # fmt: skip
            action = DrawingAction("stroke_v2", {"samples": [sample]})
        empty = np.zeros(1, dtype=bool)
        dummy = PolicySample(np.zeros(1), observation.stage_id, empty, empty, ActionChoice(0))
        return PolicyDecision(action, dummy)

    def params_sha256(self) -> str:
        return "paint"


# {
#   責務: [ FailingPolicy: 試験用。常に例外を出し、生成をpolicy_errorにする ]
# }
class FailingPolicy(PaintPolicy):
    policy_version = "test-failing-v1"

    def act(self, observation, rng):
        raise RuntimeError("toy failure")

    def params_sha256(self) -> str:
        return "failing"


# {
#   責務: [ make_runner: Policyから試験用のRolloutRunnerを作る ]
# }
def make_runner(policy, config: TrainConfig = CONFIG) -> RolloutRunner:
    builder = ObservationBuilder(config.tail_length, config.summary_side)
    return RolloutRunner(policy, get_text_encoder(config.text_encoder_id), builder, config)


# {
#   責務: [ make_samples: 黒い目標画像2枚を作り、(入力, 目標)の組を返す ]
# }
def make_samples(tmp_path):
    samples = []
    for image_id in ("img1", "img2"):
        path = tmp_path / f"{image_id}.png"
        Image.new("RGB", (16, 16), (0, 0, 0)).save(path)
        samples.append((PolicyInput(image_id, "黒"), RewardTarget(image_id, str(path), "sha")))
    return samples


# {
#   責務: [ run_toy: 試験用の条件でrun_generationsを実行する ]
# }
def run_toy(tmp_path, conditions, **options) -> RunOutput:
    evaluator = RewardEvaluator(CONFIG.reward, CONFIG.canvas_side)
    groups = {"img1": "g1", "img2": "g2"}
    return run_generations(
        conditions, evaluator, make_samples(tmp_path), groups, "set1", "validation",
        seeds_per_item=2, **options,
    )  # fmt: skip


def test_runner_uses_same_seeds_budget_and_canvas_for_all_conditions(tmp_path):
    output = run_toy(
        tmp_path,
        {"before": make_runner(RandomLegalPolicy(CONFIG)), "after": make_runner(PaintPolicy())},
    )
    seeds = {(g.condition, g.image_id, g.k): g.seed for g in output.generations}
    for (condition, image_id, k), seed in seeds.items():
        assert seed == inference_seed("set1", image_id, k) == seeds[("after", image_id, k)]
    other = TrainConfig(canvas_side=8, max_actions=80)
    with pytest.raises(ValueError, match="share canvas and budget"):
        run_toy(
            tmp_path,
            {"before": make_runner(PaintPolicy()), "after": make_runner(PaintPolicy(), other)},
        )


def test_runner_keeps_failed_generations_in_denominators(tmp_path):
    output = run_toy(
        tmp_path, {"before": make_runner(FailingPolicy()), "after": make_runner(PaintPolicy())}
    )
    assert len([g for g in output.generations if g.condition == "before"]) == 4
    assert output.failures["before"]["policy_error"] == 4
    assert output.process["before"]["generations"] == 4
    assert output.process["before"]["completion_rate"] == 0.0
    assert output.process["after"]["completion_rate"] == 1.0, output.failures["after"]


def test_runner_exit_code_is_1_when_failures_exist_and_results_are_saved(tmp_path):
    output = run_toy(
        tmp_path, {"before": make_runner(FailingPolicy()), "after": make_runner(PaintPolicy())}
    )
    assert output.exit_code == 1
    saved = tmp_path / "run.json"
    saved.write_text(output.to_json(indent=2), encoding="utf-8")
    reloaded = RunOutput.from_json(saved.read_text(encoding="utf-8"))
    assert len(reloaded.generations) == len(output.generations) and reloaded.exit_code == 1
    clean = run_toy(tmp_path, {"after": make_runner(PaintPolicy())})
    assert clean.exit_code == 0 and clean.process["after"]["audit_pass_rate"] == 1.0


def test_runner_blocks_comparison_across_different_hardware_ids(tmp_path):
    output = run_toy(
        tmp_path,
        {"before": make_runner(FailingPolicy()), "after": make_runner(PaintPolicy())},
        hardware_ids={"before": "machine-a", "after": "machine-b"},
    )
    with pytest.raises(ValueError, match="hardware_id"):
        compare_conditions(output, "after", "before", resamples=50)


def test_runner_before_after_with_toy_policy_shows_expected_direction(tmp_path):
    # 配線の確認だけ。before(全生成が失敗しblank得点)とafter(黒で塗る)の差が、黒い目標で正になる。
    output = run_toy(
        tmp_path, {"before": make_runner(FailingPolicy()), "after": make_runner(PaintPolicy())}
    )
    rows = {r["metric"]: r for r in compare_conditions(output, "after", "before", resamples=200)}
    assert {"rgb_128", "edge_128", "rgb_hist_16", "heldout_ssim64"} <= set(rows)
    assert rows["rgb_128"]["delta"] > 0 and rows["heldout_ssim64"]["delta"] > 0
    assert rows["rgb_128"]["n_groups"] == 2


def test_compare_pairs_only_shared_inference_seeds_and_reports_missing_blank(tmp_path):
    output = run_toy(
        tmp_path, {"before": make_runner(FailingPolicy()), "after": make_runner(PaintPolicy())}
    )
    # beforeからk=1を欠かすと、共通のk=0だけで対応づける(位置でずらして比べない)。
    output.generations = [
        g for g in output.generations if not (g.condition == "before" and g.k == 1)
    ]
    rows = compare_conditions(output, "after", "before", resamples=50)
    assert rows and all(r["n_items"] == 2 for r in rows)
    # blankの指標が欠けた実行結果は、KeyErrorでなく原因の分かるエラーにする。
    del output.blank_metrics["img1"]["rgb_128"]
    with pytest.raises(ValueError, match="blank metric"):
        compare_conditions(output, "after", "before", resamples=50)


# {
#   責務: [ RecordingPolicy: 試験用。生成ごとに観測のtext_conditionを記録する(PaintPolicyと同じ描画) ]
# }
class RecordingPolicy(PaintPolicy):
    def __init__(self) -> None:
        self.texts: list[bytes] = []

    def act(self, observation, rng):
        self.texts.append(observation.text_condition.tobytes())
        return super().act(observation, rng)


def distinct_prompt_samples(tmp_path):
    samples = []
    for image_id, prompt in (("img1", "赤い丸"), ("img2", "青い四角")):
        path = tmp_path / f"{image_id}.png"
        Image.new("RGB", (16, 16), (0, 0, 0)).save(path)
        samples.append((PolicyInput(image_id, prompt), RewardTarget(image_id, str(path), "sha")))
    return samples


def test_runner_shuffle_prompts_records_swapped_generations_apart_from_normal_ones(tmp_path):
    policy = RecordingPolicy()
    evaluator = RewardEvaluator(CONFIG.reward, CONFIG.canvas_side)
    groups = {"img1": "g1", "img2": "g2"}
    kwargs = dict(seeds_per_item=2)
    plain = run_generations(
        {"after": make_runner(PaintPolicy())}, evaluator, distinct_prompt_samples(tmp_path),
        groups, "set1", "validation", **kwargs,
    )  # fmt: skip
    output = run_generations(
        {"after": make_runner(policy)}, evaluator, distinct_prompt_samples(tmp_path),
        groups, "set1", "validation", shuffle_prompts=True, **kwargs,
    )  # fmt: skip
    assert plain.shuffled == []
    assert len(output.shuffled) == len(output.generations) == 4
    assert {(g.image_id, g.k, g.seed) for g in output.shuffled} == {
        (g.image_id, g.k, g.seed) for g in output.generations
    }
    assert all(g.condition == "after" for g in output.shuffled)

    # 通常の生成・工程指標・コスト・終了コードは、入れ替え生成の有無で変わらない
    def key(g):
        return (g.image_id, g.k, g.failure_kind, g.metrics, g.audit_passed, g.replay_exact)

    assert [key(g) for g in output.generations] == [key(g) for g in plain.generations]
    assert output.process == plain.process and output.exit_code == plain.exit_code
    # 方策が見たpromptは2種類だけ(各画像の自分のpromptと、入れ替え先のprompt)
    assert len(set(policy.texts)) == 2
    assert RunOutput.from_json(output.to_json()).shuffled == output.shuffled


# {
#   責務: [
#     ColorByPromptPolicy: 試験用。観測のtext_conditionが一致したprobeのpromptの色で、最後のrefine工程を塗る
#     (同じ色の塗り重ねは画素を変えず工程を抜けられないため、他の工程はPaintPolicyの黒のまま)
#   ]
# }
class ColorByPromptPolicy(PaintPolicy):
    def __init__(self) -> None:
        encoder = get_text_encoder(CONFIG.text_encoder_id)
        self.vectors = [(encoder.encode(prompt), color) for prompt, color in PROBES]

    def act(self, observation, rng):
        decision = super().act(observation, rng)
        if decision.action.type == "select_color" and observation.stage_id == "refine":
            for vector, color in self.vectors:
                if np.allclose(observation.text_condition, vector):
                    return PolicyDecision(
                        DrawingAction("select_color", {"color_id": color}), decision.sample
                    )
        return decision


def test_runner_probe_generates_with_probe_prompts_and_scores_the_expected_color(tmp_path):
    output = run_toy(
        tmp_path,
        {"before": make_runner(PaintPolicy()), "after": make_runner(ColorByPromptPolicy())},
        probe=True,
    )
    assert len(output.probes) == len(PROBES) * 2 * 2  # probe x seed(2) x 条件(2)
    seeds = {(p.prompt, p.k): p.seed for p in output.probes}
    assert all(p.seed == seeds[(p.prompt, p.k)] for p in output.probes)  # seedは条件に依存しない
    control = probe_hit_rate_control(output)
    assert control == {"before": 0.0, "after": 1.0, "n": len(PROBES) * 2}
    assert RunOutput.from_json(output.to_json()).probes == output.probes
    plain = run_toy(tmp_path, {"before": make_runner(PaintPolicy())})
    assert plain.probes == []

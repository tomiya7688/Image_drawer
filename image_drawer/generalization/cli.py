"""image-drawer-evaluate: 評価集合の凍結、実行、比較、報告書、人間評価のコマンド。

終了コード: 0 成功、1 失敗の生成を含む(結果は保存済み)、2 実行不能、3 Replay不一致・監査器の不一致。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from image_drawer.generalization.eval_set import EvalSet, build_eval_set
from image_drawer.generalization.freeze import (
    LEDGER_NAME,
    LedgerError,
    authorize_run,
    create_freeze_manifest,
)

EXIT_UNRUNNABLE = 2
METRICS_VERSION = "metrics_v1"


# {
#   責務: [ _build_parser: サブコマンドと引数を定義する ]
# }
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="image-drawer-evaluate", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="評価集合を凍結する")
    plan.add_argument("--dataset", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--eval-set-id", default=None)
    plan.add_argument("--probes", type=Path, default=None, help="prompt_probes_v1.json")
    freeze = sub.add_parser("freeze", help="test評価の凍結宣言を作る")
    freeze.add_argument("--eval-set", type=Path, required=True)
    freeze.add_argument("--before", type=Path, required=True)
    freeze.add_argument("--after", type=Path, required=True)
    freeze.add_argument("--analysis-plan", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    run = sub.add_parser("run", help="条件ごとに生成して指標を計算する")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--eval-set", type=Path, required=True)
    run.add_argument("--split", choices=("validation", "test"), required=True)
    run.add_argument("--condition", action="append", default=[], help="名前=checkpointのjson")
    run.add_argument("--freeze-manifest", type=Path, default=None)
    run.add_argument("--train-config", type=Path, default=None)
    run.add_argument("--seeds", type=int, default=5)
    run.add_argument(
        "--shuffle-prompts",
        action="store_true",
        help="promptを別の画像のものに入れ替えた生成も行い、prompt_shuffle_deltaを計算できるようにする",
    )
    run.add_argument(
        "--probe",
        action="store_true",
        help="色名などの構成的なpromptでも生成し、probe_hit_rateを計算できるようにする",
    )
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--ledger", type=Path, default=None)
    compare = sub.add_parser("compare", help="実行結果を比較する")
    compare.add_argument("--run", type=Path, required=True)
    compare.add_argument("--eval-set", type=Path, default=None)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--resamples", type=int, default=10000)
    compare.add_argument("--rng-seed", type=int, default=0)
    report = sub.add_parser("report", help="報告書(JSONとMarkdown)を作る")
    report.add_argument("--compare", type=Path, required=True)
    report.add_argument("--run", type=Path, required=True)
    report.add_argument("--eval-set", type=Path, required=True)
    report.add_argument("--human", type=Path, default=None)
    report.add_argument("--out", type=Path, required=True)
    report.add_argument("--markdown", type=Path, required=True)
    pack = sub.add_parser("human-pack", help="人間評価の盲検パックを作る")
    pack.add_argument("--run", type=Path, required=True)
    pack.add_argument("--plan", type=Path, required=True)
    pack.add_argument("--eval-set", type=Path, required=True)
    pack.add_argument("--out", type=Path, required=True)
    collect = sub.add_parser("human-collect", help="人間評価の回答を集計する")
    collect.add_argument("--pack", type=Path, required=True)
    collect.add_argument("--plan", type=Path, required=True)
    collect.add_argument("--eval-set", type=Path, required=True)
    collect.add_argument("--responses", type=Path, required=True)
    collect.add_argument("--out", type=Path, required=True)
    return parser


# {
#   責務: [ _write_json: JSONを整形して書く(親ディレクトリも作る) ]
# }
def _write_json(path: Path, document: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


# {
#   責務: [ _cmd_plan: DatasetのtestをスナップショットしてEvalSetを保存する ]
# }
def _cmd_plan(args: argparse.Namespace) -> int:
    templates = json.loads(args.probes.read_text(encoding="utf-8")) if args.probes else []
    eval_set = build_eval_set(
        args.dataset, args.eval_set_id or args.out.stem, probe_templates=templates
    )
    eval_set.save(args.out)
    print(
        f"eval_set {eval_set.eval_set_id}: {len(eval_set.items)} items, sha256={eval_set.eval_set_sha256}"
    )
    return 0


# {
#   責務: [ _cmd_freeze: checkpointと評価集合から凍結宣言を保存する ]
# }
def _cmd_freeze(args: argparse.Namespace) -> int:
    from image_drawer.training.experiment import git_commit
    from image_drawer.training.reward_spec import default_stage_reward_spec

    eval_set = EvalSet.load(args.eval_set)
    spec = default_stage_reward_spec()
    config_text = json.dumps({"final": dict(spec.final_weights)}, sort_keys=True)
    manifest = create_freeze_manifest(
        eval_set,
        before_checkpoint=args.before,
        after_checkpoint=args.after,
        analysis_plan=args.analysis_plan,
        git_commit=git_commit(),
        reward_version=spec.profile_id,
        reward_weights_sha256=spec.sha256,
        metrics_version=METRICS_VERSION,
        metrics_config_sha256=hashlib.sha256(config_text.encode()).hexdigest(),
    )
    manifest.save(args.out)
    print(f"freeze manifest {manifest.digest()}")
    return 0


# {
#   責務: [ _load_runner: checkpointのjson、またはrandom_validからRolloutRunnerを作る ]
# }
def _load_runner(spec: str, config):
    import numpy as np

    from image_drawer.training.checkpoint import load_checkpoint
    from image_drawer.training.observation import ObservationBuilder
    from image_drawer.training.optimizer import BaselineTable
    from image_drawer.training.policy import ParametricPolicy, RandomLegalPolicy
    from image_drawer.training.policy_params import AdamState
    from image_drawer.training.rollout import RolloutRunner
    from image_drawer.training.text_encoder import get_text_encoder

    encoder = get_text_encoder(config.text_encoder_id)
    if spec == "random":
        policy = RandomLegalPolicy(config)
    else:
        policy = ParametricPolicy(config, encoder.dim, np.random.Generator(np.random.PCG64(0)))
        load_checkpoint(
            Path(spec), policy, AdamState(policy.state_dict()), BaselineTable(), encoder, config
        )
    builder = ObservationBuilder(config.tail_length, config.summary_side)
    return RolloutRunner(policy, encoder, builder, config)


# {
#   責務: [ _make_image_sink: 人間評価用に、k=0の生成画像をPNGで保存する関数を返す ]
# }
def _make_image_sink(out: Path, side: int):
    from PIL import Image

    def sink(result, trial) -> None:
        if result.k != 0 or trial.final_composite is None:
            return
        directory = out / "images" / result.condition
        directory.mkdir(parents=True, exist_ok=True)
        image = Image.frombytes("RGBA", (side, side), trial.final_composite)
        white = Image.new("RGB", (side, side), (255, 255, 255))
        white.paste(image, mask=image.getchannel("A"))
        white.save(directory / f"{result.image_id}.png")

    return sink


# {
#   責務: [ _parse_conditions: --conditionを検証し、名前 -> 指定の辞書にする ]
#   エラー: [ ValueError: 0件、名前または指定が空、名前の重複、予約名blank ]
# }
def _parse_conditions(items: list[str]) -> dict[str, str]:
    if not items:
        raise ValueError("at least one --condition name=path is required")
    specs: dict[str, str] = {}
    for item in items:
        name, _, spec = item.partition("=")
        if not name.strip() or not spec.strip():
            raise ValueError(f"--condition must be name=path: {item!r}")
        if name in specs or name == "blank":
            raise ValueError(f"--condition name is duplicated or reserved: {name}")
        specs[name] = spec
    return specs


# {
#   責務: [ _cmd_run: 規則を確認し、条件ごとに生成して結果を保存する ]
#   処理: [ testでは凍結宣言と帳簿を要求する。失敗した生成があっても結果は保存し、終了コード1を返す ]
# }
def _cmd_run(args: argparse.Namespace) -> int:
    from image_drawer.dataset import iter_samples
    from image_drawer.dataset.manifest import DatasetManifest
    from image_drawer.generalization.runner import run_generations
    from image_drawer.training.config import TrainConfig
    from image_drawer.training.reward import RewardEvaluator

    eval_set = EvalSet.load(args.eval_set)
    # 台帳に記録する前に、条件と評価集合の整合を確認する(実行不能な入力で台帳を汚さない)。
    specs = _parse_conditions(args.condition)
    samples = list(iter_samples(args.dataset, args.split))
    if args.split == "test":
        # 評価集合とDatasetが一致しなければ、部分的な結果を保存せずに実行不能とする。
        wanted = {item.image_id for item in eval_set.items}
        missing = sorted(wanted - {pair[0].image_id for pair in samples})
        if missing:
            raise ValueError(f"eval_set items are missing from the dataset test split: {missing}")
        samples = [pair for pair in samples if pair[0].image_id in wanted]
    ledger = args.ledger or args.out.parent / LEDGER_NAME
    try:
        authorize_run(args.split, eval_set, args.freeze_manifest, ledger)
    except LedgerError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_UNRUNNABLE
    config = (
        TrainConfig.from_dict(json.loads(args.train_config.read_text(encoding="utf-8")))
        if args.train_config
        else TrainConfig()
    )
    conditions = {name: _load_runner(spec, config) for name, spec in specs.items()}
    manifest = DatasetManifest.load(args.dataset)
    groups = {image.image_id: image.group_id for image in manifest.images}
    output = run_generations(
        conditions,
        RewardEvaluator(config.reward, config.canvas_side),
        samples,
        groups,
        eval_set.eval_set_id,
        args.split,
        seeds_per_item=args.seeds,
        dataset_exclusions=len(eval_set.excluded),
        sink=_make_image_sink(args.out, config.canvas_side),
        shuffle_prompts=args.shuffle_prompts,
        probe=args.probe,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run.json").write_text(output.to_json(indent=2), encoding="utf-8")
    print(f"run saved: {len(output.generations)} generations, exit_code={output.exit_code}")
    return output.exit_code


# {
#   責務: [ _cmd_compare: 実行結果を比較して保存する ]
# }
def _cmd_compare(args: argparse.Namespace) -> int:
    from image_drawer.generalization.compare import compare_all
    from image_drawer.generalization.runner import RunOutput

    run_path = args.run / "run.json" if args.run.is_dir() else args.run
    output = RunOutput.from_json(run_path.read_text(encoding="utf-8"))
    unseen = None
    if args.eval_set:
        unseen = {i.image_id for i in EvalSet.load(args.eval_set).unseen_prompt_items()}
    result = compare_all(
        output, unseen_prompt_ids=unseen, resamples=args.resamples, rng_seed=args.rng_seed
    )
    result["bootstrap"] = {"resamples": args.resamples, "rng_seed": args.rng_seed}
    _write_json(args.out, result)
    return 0


# {
#   責務: [ main: コマンドを実行する。実行不能な入力は終了コード2にする ]
# }
def main(argv: list[str] | None = None) -> int:
    from image_drawer.generalization import cli_report

    args = _build_parser().parse_args(argv)
    handlers = {
        "plan": _cmd_plan,
        "freeze": _cmd_freeze,
        "run": _cmd_run,
        "compare": _cmd_compare,
        "report": cli_report.cmd_report,
        "human-pack": cli_report.cmd_human_pack,
        "human-collect": cli_report.cmd_human_collect,
    }
    try:
        return handlers[args.command](args)
    except (OSError, ValueError, KeyError, LedgerError) as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_UNRUNNABLE


if __name__ == "__main__":
    sys.exit(main())

"""学習CLI: `python -m image_drawer.training train / evaluate`(設計書2節・8.4節)。

終了コード: 0=完了(evaluateの判定がnot_learned/inconclusiveでも0。判定は出力のverdictを見る)、
1=学習が中断した(半数超のtrialが例外)、2=実行できない(入力不正・checkpoint不一致など)。
このCLIは学習の成功を主張しない。判定は`ClaimGate`の出力をそのまま保存する。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from image_drawer.dataset import DatasetManifest, iter_samples
from image_drawer.training.checkpoint import load_checkpoint
from image_drawer.training.config import TrainConfig
from image_drawer.training.evaluation import (
    MIN_IMAGES,
    MIN_SEEDS,
    ClaimGate,
    UpdateEvidence,
    run_evaluation,
)
from image_drawer.training.integrity import verify_allowed_actions, verify_trial_replay
from image_drawer.training.observation import ObservationBuilder
from image_drawer.training.optimizer import NON_UPDATING_END_REASONS, BaselineTable, mean_log_prob
from image_drawer.training.policy import ParametricPolicy, RandomLegalPolicy
from image_drawer.training.policy_params import AdamState
from image_drawer.training.reward import RewardEvaluator
from image_drawer.training.rollout import RolloutRunner
from image_drawer.training.trainer import STREAM_INIT, Trainer, default_text_encoder, derive_seed

PROBE_STEP_LIMIT = 500
PROBE_SEEDS = 2


# {
#   責務: [ _load_config: 設定JSON(TrainConfig.to_dictの部分集合)を読み、--seed/--iterationsで上書きする ]
#   エラー: [ ValueError, OSError: 設定が読めない・不正 ]
# }
def _load_config(path: str | None, seed: int | None, iterations: int | None) -> TrainConfig:
    data: dict[str, Any] = {}
    if path:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("設定JSONはobjectにしてください")
        data = loaded
    if seed is not None:
        data["master_seed"] = seed
    if iterations is not None:
        data["iterations"] = iterations
    return TrainConfig.from_dict(data)


# {
#   責務: [ _new_policy: configの初期化seedでParametricPolicyを作る(trainと同じ導出) ]
# }
def _new_policy(config: TrainConfig, dim: int) -> ParametricPolicy:
    rng = np.random.default_rng(derive_seed(config.master_seed, STREAM_INIT))
    return ParametricPolicy(config, dim, rng)


# {
#   責務: [ _sidecar_for: 記録したnpzのパスから、iter_N.jsonの場所を決める(移動された出力も探す) ]
# }
def _sidecar_for(npz: str, experiment_dir: Path) -> Path:
    path = Path(npz)
    if not path.exists():
        path = experiment_dir / "checkpoints" / path.name
    return path.with_suffix(".json")


# {
#   責務: [ _load_policy: checkpointを検証付きで読み、評価用の幅(sigma_end)を設定したPolicyを返す ]
#   エラー: [ CheckpointMismatchError(ValueError): SHA-256・版・形状・設定の不一致 ]
# }
def _load_policy(config: TrainConfig, encoder, sidecar: Path) -> ParametricPolicy:
    policy = _new_policy(config, encoder.dim)
    load_checkpoint(
        sidecar, policy, AdamState(policy.state_dict()), BaselineTable(), encoder, config
    )
    policy.stroke_sigma = config.sigma_end
    return policy


# {
#   責務: [ _train: 学習を実行し、experiment.jsonとmetrics.jsonlを出力先へ書く ]
#   戻り値: [ exit_code: 0=完了、1=中断 ]
# }
def _train(args: argparse.Namespace) -> int:
    config = _load_config(args.config, args.seed, args.iterations)
    encoder = default_text_encoder(config)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("reports") / "training" / f"exp-{config.config_hash()[:8]}"
    )
    trainer = Trainer(
        config,
        _new_policy(config, encoder.dim),
        encoder,
        RewardEvaluator(config.reward, config.canvas_side),
        output_dir,
    )
    record = trainer.train(
        Path(args.dataset),
        resume_from=Path(args.resume) if args.resume else None,
        stop_after_iteration=args.stop_after,
    )
    summary = {
        "experiment_id": record.experiment_id,
        "output_dir": str(output_dir),
        "counts": dict(record.counts),
        "policy_before": record.policy_before.path,
        "policy_after": record.policy_after.path,
        "abort_reason": record.abort_reason,
        "claim": None,
        "note": "学習の成功は主張しない。判定は evaluate で行う",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if record.abort_reason else 0


# {
#   責務: [ _probe_evidence: M1の根拠。更新前後のPolicyで、同じ記録済みActionの対数確率を比べる ]
#   処理: [ 更新前Policyのtrialからsampleを取り、整合検査(Replay一致・許可Action)も実行する ]
#   戻り値: [ (evidence, integrity_ok) ]
# }
def _probe_evidence(
    config: TrainConfig, encoder, before: ParametricPolicy, after: ParametricPolicy, dataset: Path
) -> tuple[UpdateEvidence, bool]:
    builder = ObservationBuilder(config.tail_length, config.summary_side)
    runner = RolloutRunner(before, encoder, builder, config)
    samples = list(iter_samples(dataset, "train"))[:1]
    recorded, integrity_ok = [], True
    for policy_input, _target in samples:
        for seed in range(PROBE_SEEDS):
            trial = runner.run_trial(policy_input, seed, None)
            if trial.end_reason in NON_UPDATING_END_REASONS:
                continue
            try:
                verify_trial_replay(trial)
                verify_allowed_actions(trial)
            except ValueError:
                integrity_ok = False
            recorded.extend(step.sample for step in trial.steps)
    recorded = recorded[:PROBE_STEP_LIMIT]
    start, end = before.state_dict(), after.state_dict()
    delta = float(np.sqrt(sum(float(((end[k] - start[k]) ** 2).sum()) for k in start)))
    changed = bool(recorded) and mean_log_prob(after, recorded) != mean_log_prob(before, recorded)
    return UpdateEvidence(delta, changed), integrity_ok and bool(recorded)


# {
#   責務: [ _evaluate: held-out(test split)で学習前・学習後・ランダムを測り、ClaimGateの判定を保存する ]
#   処理: [ 同じ画像・seed・Budgetで3者を測定。claim.jsonとevaluation.jsonlを書き、experiment.jsonのclaimを更新 ]
#   戻り値: [ exit_code: 0(判定はverdictに出力) ]
# }
def _evaluate(args: argparse.Namespace) -> int:
    if args.images < MIN_IMAGES or args.seeds < MIN_SEEDS:
        raise ValueError(f"--imagesは{MIN_IMAGES}以上、--seedsは{MIN_SEEDS}以上にしてください")
    experiment_dir = Path(args.experiment_dir)
    record_path = experiment_dir / "experiment.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    config = TrainConfig.from_dict(record["config"])
    dataset = Path(args.dataset)
    manifest = DatasetManifest.load(dataset)
    if (manifest.dataset_hash or manifest.compute_hash()) != record["dataset_hash"]:
        raise ValueError("Datasetが学習時と異なります(dataset_hashが一致しません)")
    encoder = default_text_encoder(config)
    before = _load_policy(
        config, encoder, _sidecar_for(record["policy_before"]["path"], experiment_dir)
    )
    after = _load_policy(
        config, encoder, _sidecar_for(record["policy_after"]["path"], experiment_dir)
    )
    held_out = list(iter_samples(dataset, "test"))
    if not held_out:
        raise ValueError("test splitに評価できる画像がありません")
    held_out = sorted(held_out, key=lambda pair: pair[0].image_id)[: args.images]
    seeds = list(range(args.seeds))
    evaluator = RewardEvaluator(config.reward, config.canvas_side)
    builder = ObservationBuilder(config.tail_length, config.summary_side)

    def measure(policy) -> list:
        return run_evaluation(
            RolloutRunner(policy, encoder, builder, config), evaluator, held_out, seeds
        )

    tables = {
        "before": measure(before),
        "after": measure(after),
        "baseline": measure(RandomLegalPolicy(config)),
    }
    evidence, integrity_ok = _probe_evidence(config, encoder, before, after, dataset)
    claim = ClaimGate().evaluate(
        tables["before"], tables["after"], tables["baseline"], evidence, integrity_ok
    )
    claim_json = {
        **claim.to_json(),
        "images": len(held_out),
        "seeds": len(seeds),
        "scope": "合成した単純な画像セットでの測定。実画像・高画質・収束は主張しない",
    }
    with (experiment_dir / "evaluation.jsonl").open("w", encoding="utf-8") as handle:
        for name, rows in tables.items():
            for row in rows:
                handle.write(
                    json.dumps({"policy": name, **row.__dict__}, ensure_ascii=False) + "\n"
                )
    (experiment_dir / "claim.json").write_text(
        json.dumps(claim_json, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    record["claim"] = claim_json
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(claim_json, ensure_ascii=False, indent=2))
    return 0


# {
#   責務: [ main: train / evaluate を実行する ]
#   戻り値: [ exit_code: 0=完了、1=学習が中断、2=実行できない ]
# }
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m image_drawer.training")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="画像のみのDatasetでPolicyを学習する")
    train.add_argument("dataset", help="image-drawer-datasetで作ったDatasetのディレクトリ")
    train.add_argument("--config", help="TrainConfigの設定JSON(省略した項目は既定値)")
    train.add_argument("--output-dir", help="既定: reports/training/exp-<設定ハッシュ先頭8桁>")
    train.add_argument("--seed", type=int, help="master_seedを上書きする")
    train.add_argument("--iterations", type=int, help="iterationsを上書きする")
    train.add_argument("--resume", help="再開するcheckpointのiter_N.json")
    train.add_argument("--stop-after", type=int, help="このiterationで一旦止める(再開の試験用)")
    evaluate = commands.add_parser("evaluate", help="held-outで学習前後とランダムを比べて判定する")
    evaluate.add_argument("experiment_dir", help="trainの出力先(experiment.jsonがある場所)")
    evaluate.add_argument("--dataset", required=True, help="学習に使ったDatasetのディレクトリ")
    evaluate.add_argument("--images", type=int, default=MIN_IMAGES, help="held-outの最大画像数")
    evaluate.add_argument("--seeds", type=int, default=MIN_SEEDS, help="画像ごとの評価seed数")
    args = parser.parse_args(argv)
    try:
        return _train(args) if args.command == "train" else _evaluate(args)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(
            json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())

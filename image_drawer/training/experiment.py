"""ExperimentRecord(再現に必要な全情報)と、experiment.json・metrics.jsonlの書き出し。"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from image_drawer.dataset.manifest import DatasetManifest
from image_drawer.drawing import PALETTE_ID, RENDERER_V2, default_catalog
from image_drawer.drawing.stage import default_stage_table
from image_drawer.training.checkpoint import CheckpointRef
from image_drawer.training.config import TrainConfig
from image_drawer.training.reward_spec import default_stage_reward_spec
from image_drawer.training.text_encoder import TextEncoder


# {
#   責務: [ ExperimentRecord: 実験の再現に必要な項目(設計書3.5・6節)。claimは評価後に付く ]
#   フィールド: [ abort_reason: 学習を中断した理由(中断しなければNone。設計書への追加項目) ]
# }
@dataclass(frozen=True)
class ExperimentRecord:
    experiment_id: str
    git_commit: str
    dataset_hash: str
    extractors: Mapping[str, str]
    text_encoder: Mapping[str, str]
    palette_id: str
    brush_set_id: str
    control_set_id: str
    renderer_profile: str
    stage_workflow_sha256: str
    reward: Mapping[str, str]
    config: Mapping[str, Any]
    seeds: Mapping[str, int]
    policy_before: CheckpointRef
    policy_after: CheckpointRef
    counts: Mapping[str, int]
    metrics_path: str
    claim: Mapping[str, Any] | None
    environment: Mapping[str, str]
    abort_reason: str | None = None

    # {
    #   責務: [ missing_fields: 空の必須項目名を返す(abort_reasonとclaimは空でよい) ]
    # }
    def missing_fields(self) -> list[str]:
        optional = {"claim", "abort_reason", "counts"}
        return [
            name
            for name, value in asdict(self).items()
            if name not in optional and value in ("", None, {}, [])
        ]

    # {
    #   責務: [ to_json: JSON互換の辞書にする ]
    # }
    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# {
#   責務: [ git_commit: 現在のコミットID。取得できなければunknown ]
# }
def git_commit(root: Path | None = None) -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root or Path.cwd(), timeout=10, text=True,
            stderr=subprocess.DEVNULL,
        )  # fmt: skip
        return output.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


# {
#   責務: [ build_record: configと実行環境、Dataset manifestからExperimentRecordを組み立てる ]
#   引数: [ dataset_dir, config, encoder, experiment_id, before/after: checkpoint, counts, seeds,
#           metrics_path, claim, abort_reason ]
# }
def build_record(
    dataset_dir: Path,
    config: TrainConfig,
    encoder: TextEncoder,
    experiment_id: str,
    before: CheckpointRef,
    after: CheckpointRef,
    counts: Mapping[str, int],
    metrics_path: str,
    claim: Mapping[str, Any] | None = None,
    abort_reason: str | None = None,
) -> ExperimentRecord:
    manifest = DatasetManifest.load(dataset_dir)
    catalog = default_catalog()
    reward = default_stage_reward_spec()

    def describe(info) -> str:
        return f"{info.id}@{info.version}#model={info.model_hash}#config={info.config_hash}"

    return ExperimentRecord(
        experiment_id=experiment_id,
        git_commit=git_commit(),
        dataset_hash=manifest.dataset_hash or manifest.compute_hash(),
        extractors={
            "tagger": describe(manifest.tagger),
            "captioner": describe(manifest.captioner),
            "prompt_builder_version": manifest.prompt_builder_version,
        },
        text_encoder={
            "id": config.text_encoder_id,
            "version": encoder.encoder_version,
            "dim": str(encoder.dim),
        },
        palette_id=PALETTE_ID,
        brush_set_id=catalog.brush_sha256,
        control_set_id=catalog.control_sha256,
        renderer_profile=RENDERER_V2,
        stage_workflow_sha256=default_stage_table().workflow_sha256,
        reward={
            "reward_profile_id": config.reward.reward_profile_id,
            "reward_impl_version": config.reward.reward_impl_version,
            "stage_reward_v2_sha256": reward.sha256,
        },
        config=config.to_dict(),
        seeds={"master_seed": config.master_seed, "split_seed": manifest.split_seed},
        policy_before=before,
        policy_after=after,
        counts=dict(counts),
        metrics_path=metrics_path,
        claim=claim,
        environment={
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        abort_reason=abort_reason,
    )


# {
#   責務: [ write_record: experiment.jsonを書き出す ]
# }
def write_record(record: ExperimentRecord, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "experiment.json"
    path.write_text(
        json.dumps(record.to_json(), ensure_ascii=False, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return path


# {
#   責務: [ append_metrics: metrics.jsonlへ1行追記する ]
# }
def append_metrics(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")

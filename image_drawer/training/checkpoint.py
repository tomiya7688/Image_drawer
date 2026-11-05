"""checkpoint(.npz + sidecar JSON)の保存・読込・SHA-256検証。pickleは使わない。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from image_drawer.training.config import TrainConfig
from image_drawer.training.optimizer import BaselineTable
from image_drawer.training.policy import DrawingPolicy
from image_drawer.training.policy_params import AdamState
from image_drawer.training.text_encoder import CheckpointMismatchError, TextEncoder


# {
#   責務: [ CheckpointRef: 保存したcheckpointの場所・SHA-256・policy_version ]
# }
@dataclass(frozen=True)
class CheckpointRef:
    path: str
    sha256: str
    policy_version: str


# {
#   責務: [ _file_sha256: ファイルのSHA-256 ]
# }
def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# {
#   責務: [ save_checkpoint: 重み・Adam状態・工程別baselineを.npzへ、検証情報をJSONへ書く ]
#   処理: [ 一時ファイルへ書いてから置換する。npzはpickle不要の数値・文字列配列だけ ]
#   引数: [ directory, iteration, policy, adam, baselines, config, encoder ]
#   戻り値: [ ref: .npzのパスとSHA-256 ]
# }
def save_checkpoint(
    directory: Path,
    iteration: int,
    policy: DrawingPolicy,
    adam: AdamState,
    baselines: BaselineTable,
    config: TrainConfig,
    encoder: TextEncoder,
) -> CheckpointRef:
    directory.mkdir(parents=True, exist_ok=True)
    npz_path = directory / f"iter_{iteration}.npz"
    arrays: dict[str, np.ndarray] = {}
    for name, value in policy.state_dict().items():
        arrays[f"p/{name}"] = value
        arrays[f"m/{name}"] = adam.m[name]
        arrays[f"v/{name}"] = adam.v[name]
    arrays["adam_step"] = np.array([adam.step_count], dtype=np.int64)
    stages = sorted(baselines.values)
    arrays["baseline_stages"] = np.array(stages, dtype="U32")
    arrays["baseline_values"] = np.array([baselines.values[s] for s in stages], dtype=np.float64)
    temporary = directory / f"iter_{iteration}.tmp.npz"
    np.savez(temporary, **arrays)
    os.replace(temporary, npz_path)
    digest = _file_sha256(npz_path)
    sidecar = {
        "npz": npz_path.name,
        "npz_sha256": digest,
        "policy_version": policy.policy_version,
        "params_sha256": policy.params_sha256(),
        "encoder_version": encoder.encoder_version,
        "dim": encoder.dim,
        "config_hash": config.config_hash(),
        "iteration": iteration,
    }
    # sidecarも一時ファイル経由で置換し、書込み途中の中断で壊れたJSONを残さない。
    sidecar_path = directory / f"iter_{iteration}.json"
    temporary_sidecar = directory / f"iter_{iteration}.tmp.json"
    temporary_sidecar.write_text(
        json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )
    os.replace(temporary_sidecar, sidecar_path)
    return CheckpointRef(str(npz_path), digest, policy.policy_version)


# {
#   責務: [ load_checkpoint: SHA-256・版・形状を検証して状態を復元する ]
#   引数: [ sidecar_path: iter_N.json, policy, adam, baselines, encoder, config: 与えたら設定ハッシュも検証 ]
#   戻り値: [ iteration: 保存時のiteration ]
#   エラー: [ CheckpointMismatchError: 改ざん・版/次元/形状/設定の不一致・pickle配列 ]
# }
def load_checkpoint(
    sidecar_path: Path,
    policy: DrawingPolicy,
    adam: AdamState,
    baselines: BaselineTable,
    encoder: TextEncoder,
    config: TrainConfig | None = None,
) -> int:
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        npz_path = sidecar_path.parent / sidecar["npz"]
        if _file_sha256(npz_path) != sidecar["npz_sha256"]:
            raise CheckpointMismatchError(f"SHA-256が一致しません: {npz_path.name}")
        if sidecar["policy_version"] != policy.policy_version:
            raise CheckpointMismatchError("policy_versionが一致しません")
        if sidecar["encoder_version"] != encoder.encoder_version or sidecar["dim"] != encoder.dim:
            raise CheckpointMismatchError("TextEncoderの版または次元が一致しません")
        if config is not None and sidecar["config_hash"] != config.config_hash():
            raise CheckpointMismatchError("設定のハッシュが一致しません")
        with np.load(npz_path, allow_pickle=False) as data:
            weights = {k[2:]: data[k] for k in data.files if k.startswith("p/")}
            first = {k[2:]: data[k] for k in data.files if k.startswith("m/")}
            second = {k[2:]: data[k] for k in data.files if k.startswith("v/")}
            if data["adam_step"].size != 1:
                raise CheckpointMismatchError("adam_stepの形状が不正です")
            step = int(data["adam_step"].reshape(-1)[0])
            stages = [str(s) for s in data["baseline_stages"]]
            values = [float(v) for v in data["baseline_values"]]
    except CheckpointMismatchError:
        raise
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError) as error:
        raise CheckpointMismatchError(f"checkpointを読めません: {error}") from error
    # 状態を書き換える前に、全ての配列を検証する(失敗時にpolicy/Adam/baselineを混在させない)。
    current = policy.state_dict()
    if len(stages) != len(values):
        raise CheckpointMismatchError("baseline_stagesとbaseline_valuesの長さが一致しません")
    if set(weights) != set(current) or set(first) != set(weights) or set(second) != set(weights):
        raise CheckpointMismatchError("重みまたはAdam状態のhead名が一致しません")
    for name, array in weights.items():
        if array.shape != current[name].shape or first[name].shape != array.shape:
            raise CheckpointMismatchError(f"{name}の形状が一致しません")
        if second[name].shape != array.shape:
            raise CheckpointMismatchError(f"{name}のAdam 2次モーメントの形状が一致しません")
    try:
        policy.load_state_dict(weights)
    except ValueError as error:
        raise CheckpointMismatchError(str(error)) from error
    adam.m, adam.v, adam.step_count = first, second, step
    baselines.values = dict(zip(stages, values))
    return int(sidecar["iteration"])

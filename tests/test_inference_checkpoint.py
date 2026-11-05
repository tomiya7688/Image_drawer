"""S4: checkpointをPolicyとして登録する経路の配線確認（品質は確認しない）。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inference_helpers import SIDE

np = pytest.importorskip("numpy")

from image_drawer.drawing.episode_io import dumps_episode  # noqa: E402
from image_drawer.inference import DrawRequest  # noqa: E402
from image_drawer.inference.cli import main  # noqa: E402
from image_drawer.inference.policy import PolicyError, PolicyRegistry  # noqa: E402
from image_drawer.inference.session import generate_episode  # noqa: E402
from image_drawer.inference.training_policy import register_checkpoint  # noqa: E402
from image_drawer.training import TrainConfig, get_text_encoder  # noqa: E402
from image_drawer.training.checkpoint import save_checkpoint  # noqa: E402
from image_drawer.training.optimizer import BaselineTable  # noqa: E402
from image_drawer.training.policy import ParametricPolicy  # noqa: E402
from image_drawer.training.policy_params import AdamState  # noqa: E402

CONFIG = TrainConfig(canvas_side=SIDE, summary_side=4, max_actions=40)


def make_checkpoint(directory: Path) -> Path:
    encoder = get_text_encoder(CONFIG.text_encoder_id)
    policy = ParametricPolicy(CONFIG, encoder.dim, np.random.default_rng(1))
    adam = AdamState(policy.state_dict())
    save_checkpoint(directory, 3, policy, adam, BaselineTable(), CONFIG, encoder)
    return directory / "iter_3.json"


def test_checkpoint_policy_is_registered_as_trained_and_deterministic(tmp_path: Path) -> None:
    sidecar = make_checkpoint(tmp_path)
    registry = PolicyRegistry()
    register_checkpoint(registry, "ckpt", sidecar, CONFIG)
    (entry,) = registry.describe()
    digest = json.loads(sidecar.read_text(encoding="utf-8"))["npz_sha256"]
    assert entry["policy_trained"] is True and entry["checkpoint_sha256"] == digest
    assert entry["canvas"] == [SIDE, SIDE]

    request = DrawRequest("猫", "ckpt", max_actions=30, seed=2, width=SIDE, height=SIDE)
    first, second = generate_episode(request, registry), generate_episode(request, registry)
    assert dumps_episode(first.episode) == dumps_episode(second.episode)
    assert first.policy["checkpoint_sha256"] == digest and first.accepted_actions > 0


def test_checkpoint_registration_rejects_tampered_or_mismatched_files(tmp_path: Path) -> None:
    sidecar = make_checkpoint(tmp_path)
    other = TrainConfig(canvas_side=SIDE, summary_side=4, max_actions=41)
    with pytest.raises(PolicyError):
        register_checkpoint(PolicyRegistry(), "ckpt", sidecar, other)
    sidecar.with_suffix(".npz").write_bytes(b"tampered")
    with pytest.raises(PolicyError):
        register_checkpoint(PolicyRegistry(), "ckpt", sidecar, CONFIG)


def test_draw_cli_registers_checkpoint_with_train_config(tmp_path: Path, capsys) -> None:
    sidecar = make_checkpoint(tmp_path / "ckpt")
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(CONFIG.to_dict()), encoding="utf-8")
    code = main(
        ["--list-policies", "--checkpoint", f"ckpt={sidecar}", "--train-config", str(config_path)]
    )
    listed = json.loads(capsys.readouterr().out)["policies"]
    assert code == 0
    assert {p["policy_id"]: p["policy_trained"] for p in listed} == {
        "random-legal-v1": False,
        "ckpt": True,
    }
    assert main(["--list-policies", "--checkpoint", f"ckpt={sidecar}"]) == 2
    capsys.readouterr()

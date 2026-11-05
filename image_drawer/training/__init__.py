"""Image Drawerの学習(Trainer)パッケージ。numpy(extras `train`)が必要です。"""

from __future__ import annotations

try:
    import numpy as _numpy  # noqa: F401
except ImportError as error:  # pragma: no cover - numpyが無い環境の案内
    raise ImportError(
        "image_drawer.trainingにはnumpyが必要です。"
        ' `pip install "image-drawer[train]"` を実行してください。'
    ) from error

from image_drawer.training.config import (
    REWARD_COMPONENTS,
    STAGE_IDS,
    CurriculumConfig,
    RewardConfig,
    TrainConfig,
)
from image_drawer.training.text_encoder import (
    CheckpointMismatchError,
    HashedNgramTextEncoder,
    TextEncoder,
    check_encoder_matches,
    get_text_encoder,
    register_text_encoder,
)

__all__ = [
    "REWARD_COMPONENTS",
    "STAGE_IDS",
    "CheckpointMismatchError",
    "CurriculumConfig",
    "HashedNgramTextEncoder",
    "RewardConfig",
    "TextEncoder",
    "TrainConfig",
    "check_encoder_matches",
    "get_text_encoder",
    "register_text_encoder",
]

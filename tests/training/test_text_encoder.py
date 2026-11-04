"""TextEncoder(設計書8.2節)の受け入れテスト。"""

from __future__ import annotations

import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")

from image_drawer.training import (  # noqa: E402
    CheckpointMismatchError,
    HashedNgramTextEncoder,
    check_encoder_matches,
    get_text_encoder,
    register_text_encoder,
)


def test_text_encoder_is_deterministic_and_unit_norm() -> None:
    encoder = HashedNgramTextEncoder()
    first = encoder.encode("赤い四角を左に描く")
    second = encoder.encode("赤い四角を左に描く")
    assert first.shape == (encoder.dim,) and first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert abs(float(np.linalg.norm(first)) - 1.0) < 1e-5
    assert not np.array_equal(first, encoder.encode("青い四角を右に描く"))

    # PYTHONHASHSEEDが違うプロセスでも同じ値になる(hash()に依存しない)。
    code = (
        "from image_drawer.training import HashedNgramTextEncoder as E;"
        "print(E().encode('red square').tolist())"
    )
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            env={"PYTHONHASHSEED": seed, "PATH": ""},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2")
    }
    assert len(outputs) == 1
    assert outputs == {str(HashedNgramTextEncoder().encode("red square").tolist()) + "\n"}


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_text_encoder_rejects_empty_prompt(prompt: str) -> None:
    with pytest.raises(ValueError):
        HashedNgramTextEncoder().encode(prompt)


def test_text_encoder_registry_rejects_dim_mismatch_checkpoint() -> None:
    register_text_encoder("test-hashed-dim16", lambda: HashedNgramTextEncoder(dim=16))
    encoder = get_text_encoder("test-hashed-dim16")
    assert encoder.dim == 16
    check_encoder_matches(encoder, "hashed-ngram-v1", 16)
    with pytest.raises(CheckpointMismatchError):
        check_encoder_matches(encoder, "hashed-ngram-v1", 64)
    with pytest.raises(CheckpointMismatchError):
        check_encoder_matches(encoder, "other-v2", 16)


def test_text_encoder_registry_rejects_unknown_and_duplicate_ids() -> None:
    with pytest.raises(ValueError):
        get_text_encoder("no-such-encoder")
    with pytest.raises(ValueError):
        register_text_encoder("hashed-ngram-v1", lambda: HashedNgramTextEncoder(dim=8))


def test_text_encoder_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        HashedNgramTextEncoder(dim=0)
    with pytest.raises(ValueError):
        HashedNgramTextEncoder(ngram_range=(3, 1))

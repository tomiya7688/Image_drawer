"""color-keyword-v1: 共通テンプレートが長いpromptでも色の違いがベクトルに現れることを確認する。"""

import numpy as np
import pytest

from image_drawer.training.text_encoder import get_text_encoder

RED = "白と赤が中心の正方形の画像（明るい、正方形、白、赤）"
BLUE = "白と青が中心の正方形の画像（明るい、正方形、白、青）"


def cosine(first, second) -> float:
    return float(first @ second)


def test_color_keyword_encoder_is_unit_norm_deterministic_and_has_the_declared_dim():
    encoder = get_text_encoder("color-keyword-v1")
    vector = encoder.encode(RED)
    assert vector.shape == (encoder.dim,) and vector.dtype == np.float32
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)
    assert np.array_equal(vector, encoder.encode(RED))
    assert np.isfinite(vector).all()


def test_color_keyword_encoder_separates_red_and_blue_more_than_hashed_ngram():
    hashed = get_text_encoder("hashed-ngram-v1")
    colored = get_text_encoder("color-keyword-v1")
    before = cosine(hashed.encode(RED), hashed.encode(BLUE))
    after = cosine(colored.encode(RED), colored.encode(BLUE))
    assert after < before - 0.2  # 赤と青のpromptが、はっきり離れる


def test_color_keyword_encoder_rejects_blank_and_handles_prompts_without_color_words():
    encoder = get_text_encoder("color-keyword-v1")
    with pytest.raises(ValueError):
        encoder.encode("   ")
    vector = encoder.encode("色の語を含まない文章")
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-6)

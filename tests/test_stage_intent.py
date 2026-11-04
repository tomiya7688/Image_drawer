"""S1b-3: StyleIntent（promptだけから決まる規則判定器）。"""

from __future__ import annotations

import inspect

import pytest

from image_drawer.drawing.stage import StageConfigError
from image_drawer.drawing.stage.intent import (
    FLAT_POSITIVE,
    MONOCHROME_POSITIVE,
    STYLE_INTENT_VERSION,
    StyleIntent,
    judge_style_intent,
)


@pytest.mark.parametrize("phrase", MONOCHROME_POSITIVE)
def test_style_intent_monochrome_positive_phrases(phrase):
    intent = judge_style_intent(f"猫を{phrase}で描く")
    assert intent.monochrome_line_art_only is True
    assert intent.flat_style_no_shading is False


@pytest.mark.parametrize("phrase", FLAT_POSITIVE)
def test_style_intent_flat_positive_phrases(phrase):
    intent = judge_style_intent(f"猫の絵、{phrase}")
    assert intent.flat_style_no_shading is True
    assert intent.monochrome_line_art_only is False


def test_style_intent_negative_cue_overrides_positive_phrase():
    assert judge_style_intent("白黒の線画だがカラーも使う").monochrome_line_art_only is False
    assert judge_style_intent("lineart only, colorful").monochrome_line_art_only is False
    assert judge_style_intent("影なしだが立体感を出す").flat_style_no_shading is False
    assert judge_style_intent("ベタ塗り、ハイライトあり").flat_style_no_shading is False


@pytest.mark.parametrize("prompt", ["線画を描いて", "絵を描いて", "猫の絵"])
def test_style_intent_ambiguous_prompt_is_false(prompt):
    intent = judge_style_intent(prompt)
    assert (intent.monochrome_line_art_only, intent.flat_style_no_shading) == (False, False)


def test_style_intent_empty_prompt_is_false_for_both_flags():
    intent = judge_style_intent("")
    assert (intent.monochrome_line_art_only, intent.flat_style_no_shading) == (False, False)


def test_style_intent_normalizes_width_case_and_whitespace():
    assert judge_style_intent("ＬＩＮＥＡＲＴ　ＯＮＬＹ").monochrome_line_art_only is True
    assert judge_style_intent("Line Art\tOnly").monochrome_line_art_only is True
    assert judge_style_intent("線 画 の み").monochrome_line_art_only is True
    assert judge_style_intent("影、なし").flat_style_no_shading is True
    assert judge_style_intent("ﾉｰｼｪｰﾃﾞｨﾝｸﾞ").flat_style_no_shading is True
    assert judge_style_intent("NO SHADING").flat_style_no_shading is True


def test_style_intent_records_version_and_prompt():
    intent = judge_style_intent("影なし")
    assert intent.version == STYLE_INTENT_VERSION == "style-intent-rules-v1"
    assert intent.prompt == "影なし"
    assert isinstance(intent, StyleIntent)


def test_style_intent_signature_takes_only_prompt():
    assert list(inspect.signature(judge_style_intent).parameters) == ["prompt"]


@pytest.mark.parametrize("bad", [None, 3, b"x", ["a"]])
def test_style_intent_rejects_non_string_prompt(bad):
    with pytest.raises(StageConfigError):
        judge_style_intent(bad)

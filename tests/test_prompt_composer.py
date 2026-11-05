"""Captionとタグを自然言語へ統合し、矛盾やメタタグは保留する。"""

from __future__ import annotations

import json

from image_drawer.prompting.core import VisionResponse, parse_scene
from image_drawer.prompting.prompt_composer import PromptComposer
from image_drawer.prompting.tagging import TagFormatter, TagNormalizer


def test_composer_uses_tag_hints_without_repeating_subject():
    scene = parse_scene(
        VisionResponse(
            json.dumps(
                {
                    "summary": "青い髪の女の子",
                    "short_prompt": "青い髪の女の子",
                    "standard_prompt": "青い髪の女の子",
                    "detailed_prompt": "青い髪の女の子",
                },
                ensure_ascii=False,
            )
        )
    )
    raw = TagNormalizer().normalize(
        {
            "general": {
                "blue_hair": 0.97,
                "red_hair": 0.98,
                "smile": 0.94,
                "unknown_item": 0.99,
            },
            "style": {"anime_style": 0.99},
            "rating": {"explicit": 0.99},
        }
    )
    result = PromptComposer().compose(scene, TagFormatter().format(raw), 0)
    assert result.text == "青い髪の女の子、笑顔"
    assert "blue_hair" in result.included
    assert "smile" in result.included
    assert ("red_hair", "conflicts_with_caption") in result.withheld
    assert ("unknown_item", "not_mapped_to_natural_japanese") in result.withheld
    assert ("anime_style", "excluded_category") in result.withheld
    assert "explicit" not in result.text


def test_low_confidence_and_excluded_categories_do_not_enter_prose():
    scene = parse_scene(
        VisionResponse(
            json.dumps(
                {
                    "summary": "An apple",
                    "standard_prompt": "An apple",
                    "detailed_prompt": "An apple",
                }
            )
        )
    )
    raw = TagNormalizer(min_confidence=0).normalize(
        {
            "general": {"blue_dress": 0.2, "sky": 0.9},
            "copyright": {"fictional_series": 0.99},
        }
    )
    result = PromptComposer().compose(scene, TagFormatter().format(raw), 5)
    assert result.text == "An apple; sky"
    assert ("blue_dress", "low_confidence") in result.withheld
    assert ("fictional_series", "excluded_category") in result.withheld

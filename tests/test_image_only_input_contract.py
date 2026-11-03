"""画像だけで学習できる入力契約と内部テキスト条件を検査する。"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "docs" / "developers" / "training_input_v1.json"


def _contract() -> dict:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def test_training_requires_only_image():
    spec = _contract()
    assert spec["schema_version"] == "idraw-training-input/v1"
    assert spec["external_required_inputs"] == ["image"]
    assert spec["external_optional_inputs"] == []
    assert spec["mandatory_manual_fields"] == []
    assert spec["external_auxiliary_files_required"] is False
    assert set(spec["never_request_user_files"]) == {
        "txt",
        "tags",
        "captions",
        "prompts",
        "rights_info",
        "action_traces",
        "extra_metadata",
    }


def test_text_is_generated_internally_and_extractor_is_switchable():
    spec = _contract()
    controls = spec["selectors"]
    assert controls["TagExtractor"]["user_switchable"] is True
    assert controls["CaptionExtractor"]["user_switchable"] is True
    assert controls["PromptBuilder"]["internal"] is True
    assert controls["TextEncoder"]["internal"] is True
    assert spec["text_condition"]["required_during_txt2img_training"]
    assert spec["text_condition"]["required_nonempty_training_prompt"]
    assert "without_requesting_external_text" in (spec["text_condition"]["missing_condition"])
    assert spec["interfaces"]["learning_input"] == "image"
    assert spec["interfaces"]["inference_input"] == "user_prompt"


def test_policy_cannot_see_target_image():
    boundaries = _contract()["leakage_prevention"]
    assert boundaries["target_image_visible_to_policy"] is False
    assert boundaries["target_image_features_visible_to_policy"] is False
    assert boundaries["target_image_visible_to_evaluator"] is True
    assert boundaries["training_prompt_generated_without_manual_text"] is True


def test_training_docs_display_same_three_sentences():
    statements = (
        "学習に必要な外部入力は画像のみとする。",
        "ユーザーおよび開発者は、画像以外の補助データを用意する必要はない。",
        "タグ・説明文・プロンプトは、必要に応じてシステム側で自動抽出する。",
    )
    paths = (
        "README.md",
        "docs/kadoka/concept.md",
        "docs/developers/start_here.md",
        "docs/developers/drawing_data_quality_v1.md",
        "docs/developers/image_only_learning_v1.md",
        "docs/developers/training_and_evaluation.md",
        "docs/developers/drawing_policy_accuracy_v1.md",
        "docs/developers/drawing_process_acceptance.md",
    )
    for name in paths:
        content = (ROOT / name).read_text(encoding="utf-8")
        for sentence in statements:
            assert sentence in content, name


def test_no_manual_rights_field_in_dataset_example():
    doc = ROOT / "docs" / "developers" / "drawing_data_quality_v1.md"
    content = doc.read_text(encoding="utf-8")
    assert "rights_status" not in content
    assert "使用権限を確認したデータを利用" not in content
    assert "TextEncoder" in content

"""S1b-6: 目標画像を扱う型・モジュールへ到達できないことの固定。"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import subprocess
import sys
from pathlib import Path

from image_drawer.drawing.stage import (
    Budget,
    StageController,
    StyleIntent,
    judge_style_intent,
)

ROOT = Path(__file__).resolve().parents[1]
STAGE_DIR = ROOT / "image_drawer" / "drawing" / "stage"
FORBIDDEN_PREFIXES = (
    "image_drawer.evaluation", "image_drawer.part_bank", "image_drawer.dataset",
    "image_drawer.training", "image_drawer.steps", "PIL", "numpy",
)  # fmt: skip
FORBIDDEN_NAME_PARTS = ("target", "image", "reference", "pixels_of")
RUNTIME_SCRIPT = """
import sys
import image_drawer.drawing.stage
print("\\n".join(sorted(sys.modules)))
"""


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def test_stage_package_does_not_import_evaluation_or_image_modules():
    for path in sorted(STAGE_DIR.glob("*.py")):
        for module in imported_modules(path):
            assert not module.startswith(FORBIDDEN_PREFIXES), (path.name, module)
    done = subprocess.run(
        [sys.executable, "-c", RUNTIME_SCRIPT],
        cwd=ROOT, capture_output=True, text=True, timeout=120, check=True,
    )  # fmt: skip
    loaded = set(done.stdout.split())
    leaked = [name for name in loaded if name.startswith(FORBIDDEN_PREFIXES)]
    assert leaked == []


def parameter_names(callable_object) -> list[str]:
    return [name for name in inspect.signature(callable_object).parameters if name != "self"]


def test_stage_public_signatures_have_no_target_image_parameter():
    names = (
        parameter_names(StageController.__init__)
        + parameter_names(StageController.propose)
        + parameter_names(StageController.episode)
        + parameter_names(judge_style_intent)
        + [field.name for field in dataclasses.fields(StyleIntent)]
        + [field.name for field in dataclasses.fields(Budget)]
    )
    assert names  # 空の検査にならないこと
    assert not [n for n in names if any(part in n.lower() for part in FORBIDDEN_NAME_PARTS)]


def test_style_intent_result_depends_only_on_prompt():
    assert parameter_names(judge_style_intent) == ["prompt"]
    first = judge_style_intent("白黒の線画、影なし")
    assert first == judge_style_intent("白黒の線画、影なし")
    assert first != judge_style_intent("カラーの絵")

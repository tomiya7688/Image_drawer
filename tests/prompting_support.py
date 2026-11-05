"""DatasetとPromptingの統合テストで共有する擬似プロバイダと画像の用意。ネットワークは使わない。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from image_drawer.prompting import VisionResponse
from image_drawer.prompting.registry import PreprocessorRegistry

SCENE = {
    "summary": "赤いリンゴ",
    "subjects": [{"name": "リンゴ", "count": 1, "attributes": ["赤い"]}],
    "short_prompt": "赤いリンゴ",
    "standard_prompt": "白い背景に赤いリンゴが一つ",
    "detailed_prompt": "中央に一つの赤いリンゴ、白背景のシンプルなイラスト",
}


def scene_response(**overrides) -> VisionResponse:
    return VisionResponse(json.dumps({**SCENE, **overrides}, ensure_ascii=False))


@dataclass
class FakeVision:
    """最後の応答を繰り返す擬似VLM。呼び出しと受け取った画像を記録する。"""

    provider_id: str = "fake"
    model_id: str = "vlm"
    config_fingerprint: str = "fp-vlm"
    responses: list = field(default_factory=lambda: [scene_response()])
    calls: int = 0
    images: list = field(default_factory=list)
    instructions: list = field(default_factory=list)

    def describe(self, image: bytes, mime_type: str, instruction: str) -> VisionResponse:
        self.calls += 1
        self.images.append((image, mime_type))
        self.instructions.append(instruction)
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, BaseException):
            raise item
        return item


@dataclass
class FakeTagger:
    """最後の応答を繰り返す擬似Tagger。"""

    provider_id: str = "faketag"
    model_id: str = "tagger"
    config_fingerprint: str = "fp-tag"
    responses: list = field(
        default_factory=lambda: [
            {
                "general": {"apple": 0.9, "red": 0.8},
                "copyright": {"some_series": 0.9},
                "rating": {"safe": 0.99},
            }
        ]
    )
    calls: int = 0

    def tag(self, image: bytes, mime_type: str) -> object:
        self.calls += 1
        item = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        if isinstance(item, BaseException):
            raise item
        return item


def make_preprocessors(captions=None, taggers=None, verifier=None, **configure):
    """擬似プロバイダを登録したPreprocessorRegistry。chainは渡した順。"""
    registry = PreprocessorRegistry()
    names = []
    for index, model in enumerate(captions or []):
        registry.register_caption(f"c{index}", model)
        names.append(f"c{index}")
    tag_names = []
    for index, model in enumerate(taggers or []):
        registry.register_tagger(f"t{index}", model)
        tag_names.append(f"t{index}")
    if verifier is not None:
        registry.register_verifier("v", verifier)
    if names or tag_names:
        # configureは両chainを要求するため、未使用側は直接設定する
        registry.caption_chain = tuple(names)
        registry.tag_chain = tuple(tag_names)
        registry.verifier_id = "v" if verifier is not None else None
    return registry


def solid(path: Path, color, size=(32, 24)) -> Path:
    Image.new("RGB", size, color).save(path)
    return path


def write_images(directory: Path, count: int = 3) -> Path:
    """色の違う画像をcount枚(最大6)置いたディレクトリ。近重複にならない色を使う。"""
    directory.mkdir(parents=True, exist_ok=True)
    colors = [
        (220, 40, 40),
        (50, 80, 220),
        (50, 170, 70),
        (240, 220, 40),
        (140, 60, 190),
        (20, 20, 20),
    ]
    for index in range(count):
        solid(directory / f"img{index}.png", colors[index])
    return directory

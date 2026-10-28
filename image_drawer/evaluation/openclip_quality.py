"""OpenCLIPによるイラスト品質の相対評価。"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from image_drawer.core import Artifact, Score, ScoreSet

if TYPE_CHECKING:
    from image_drawer.steps.base import StepContext


# {
#   責務: [
#     OpenClipQualityEvaluator: 画像と品質を表す文章の類似度を比較する
#   ]
#   フィールド: [
#     weights_path: ローカルのOpenCLIP重み
#   ]
#   処理: [
#     画像と二つの文章のcosine類似度の差を候補ごとに返す
#   ]
# }
class OpenClipQualityEvaluator:
    name = "openclip_quality"
    prompts = ("a high quality illustration", "a low quality illustration")

    # {
    #   責務: [
    #     __init__: オフラインで読み込む重みの場所を保持する
    #   ]
    #   処理: [
    #     安全な形式の既存ファイルを指定する
    #   ]
    #   引数: [
    #     weights_path: safetensors形式の重み
    #   ]
    #   戻り値: [
    #     なし
    #   ]
    # }
    def __init__(self, weights_path: str) -> None:
        self.weights_path = Path(weights_path)
        if self.weights_path.suffix != ".safetensors" or not self.weights_path.is_file():
            raise ValueError("weights_path must point to an existing .safetensors file")
        with self.weights_path.open("rb") as weights:
            self.version = "ViT-B-32/sha256:" + hashlib.file_digest(weights, "sha256").hexdigest()

    # {
    #   責務: [
    #     evaluate: 画像候補をOpenCLIPで評価する
    #   ]
    #   処理: [
    #     ローカル画像を読み込み、候補と得点の順序を保つ
    #   ]
    #   引数: [
    #     candidates: ローカル画像を指す候補
    #     context: 実行時の識別子
    #   ]
    #   戻り値: [
    #     score_set: 文章との類似度とその差を含む得点集合
    #   ]
    # }
    def evaluate(self, candidates: list[Artifact], context: StepContext) -> ScoreSet:
        if not candidates:
            raise ValueError("OpenCLIP quality evaluation requires at least one candidate")
        for candidate in candidates:
            if not candidate.uri or not Path(candidate.uri).is_file():
                raise ValueError("OpenCLIP quality evaluation requires a local image file")
        import open_clip
        import torch
        from PIL import Image

        # モデル重みは指定されたローカルファイルからだけ読み込む。
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(self.weights_path), device="cpu"
        )
        with self.weights_path.open("rb") as weights:
            loaded_version = "ViT-B-32/sha256:" + hashlib.file_digest(weights, "sha256").hexdigest()
        if loaded_version != self.version:
            raise ValueError("OpenCLIP weights changed while loading")
        model.eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        scores = []
        with torch.inference_mode():
            text = model.encode_text(tokenizer(list(self.prompts)))
            text = text / text.norm(dim=-1, keepdim=True)
            for index, candidate in enumerate(candidates):
                with Image.open(candidate.uri) as image:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    background.alpha_composite(rgba)
                    pixels = preprocess(background.convert("RGB")).unsqueeze(0)
                vector = model.encode_image(pixels)
                vector = vector / vector.norm(dim=-1, keepdim=True)
                high, low = (vector @ text.T).squeeze(0).tolist()
                scores.append(
                    Score(
                        id=context.score_id(str(index)),
                        evaluator=self.name,
                        evaluator_version=self.version,
                        components={"high_quality_similarity": high, "low_quality_similarity": low},
                        overall=high - low,
                        metadata={"candidate_artifact_id": candidate.id},
                    )
                )
        return ScoreSet(
            id=context.artifact_id(),
            evaluator=self.name,
            evaluator_version=self.version,
            candidate_artifact_ids=[candidate.id for candidate in candidates],
            scores=scores,
            metadata={"prompts": list(self.prompts)},
        )

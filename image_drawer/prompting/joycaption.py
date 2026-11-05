"""JoyCaption Beta Oneの自然文を捏造せず画像学習用のCaption形式へ適合させる。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .core import (
    PromptExtractionError,
    VisionResponse,
    _REFUSALS,
)
from .openai_compatible import OpenAICompatibleVisionProvider

# 公式が案内するStraightforward Caption方式を使用する。JSON出力を要求しない。
JOYCAPTION_INSTRUCTION = (
    "Write a straightforward caption for this image in 100 words or less. "
    "Describe visible subjects, their colors and number, poses, surroundings, "
    "and spatial relationships. Do not infer unobservable details or identities. "
    "Avoid generic quality phrases and introductions such as 'This image shows'."
)
_ADAPTER_VERSION = "joycaption-prose-v1"


# {
#   責務: [JoyCaptionVisionProvider: 自然文専用モデルを画像説明Pipelineへ接続する]
#   フィールド: [backend: OpenAI互換APIへの接続, provider_idとmodel_idはbackendから取得]
#   処理: [自然文を要求し、既存のJSON形式へ観察事実を追加せず変換する]
# }
@dataclass(frozen=True)
class JoyCaptionVisionProvider:
    """モデルが生成した説明文から、根拠のない構造化特徴を作らない。"""

    backend: OpenAICompatibleVisionProvider

    @property
    def provider_id(self) -> str:
        return self.backend.provider_id

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    @property
    def endpoint(self) -> str:
        return self.backend.endpoint

    @property
    def data_egress(self) -> str:
        return self.backend.data_egress

    @property
    def allowed_hosts(self) -> tuple[str, ...]:
        return self.backend.allowed_hosts

    @property
    def config_fingerprint(self) -> str:
        data = ":".join(
            (
                self.backend.config_fingerprint,
                _ADAPTER_VERSION,
                JOYCAPTION_INSTRUCTION,
            )
        )
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    # {
    #   責務: [describe: 自然文を抽出し、Caption PipelineのJSON契約に変換する]
    #   処理: [推奨指示で生成し、拒否や空文は維持し、短文・標準文・詳細文を抽出]
    #   引数: [image: 画像bytes, mime_type: 画像種別, instruction: 共通Pipelineの指示]
    #   戻り値: [VisionResponse: JSON形式の文章と元の拒否情報]
    #   エラー: [PromptExtractionError: 長すぎる文章など検証不能な応答]
    # }
    def describe(
        self,
        image: bytes,
        mime_type: str,
        instruction: str,
    ) -> VisionResponse:
        response = self.backend.describe(image, mime_type, JOYCAPTION_INSTRUCTION)
        if response.refusal or response.finish_reason == "content_filter":
            return response
        caption = re.sub(r"\s+", " ", response.content).strip()
        if not caption or any(caption.casefold().startswith(s) for s in _REFUSALS):
            return response
        if caption.casefold().startswith(("sorry, i cannot", "sorry, i can't")):
            return VisionResponse(refusal=True, finish_reason=response.finish_reason)
        # JSONらしい応答は既存の厳密なparse_sceneに検証させる。
        if caption.startswith(("{", "[", "`")):
            return response
        if len(caption) > 1500:
            raise PromptExtractionError("invalid_format", "JoyCaptionの説明が1500文字を超えた")
        # 文章の一部だけを切り出し、対象・個数・位置関係を推定して追加しない。
        sentences = re.split(r"(?<=[.!?。！？])\s+", caption)
        short = sentences[0]
        standard = " ".join(sentences[:2])
        scene = {
            "summary": caption,
            "subjects": [],
            "relations": [],
            "short_prompt": short,
            "standard_prompt": standard,
            "detailed_prompt": caption,
            "background": "",
            "composition": "",
            "style": "",
            "tags": [],
        }
        return VisionResponse(
            content=json.dumps(scene, ensure_ascii=False),
            finish_reason=response.finish_reason,
        )

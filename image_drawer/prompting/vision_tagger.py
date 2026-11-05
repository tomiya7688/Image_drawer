"""画像のタグをCaptionとは独立したVLM推論で抽出する。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .core import PromptExtractionError, _json
from .openai_compatible import OpenAICompatibleVisionProvider

TAGGING_INSTRUCTION = """入力画像を解析し、見える内容から画像生成向けのタグを抽出してください。
Captionとは独立した推論です。画像の内容を見ずにタグを推測しないでください。
以下の形式のJSON objectだけを出力してください。
{
  "general": {"subject_or_appearance": 0.9},
  "character": {},
  "copyright": {},
  "style": {},
  "clothes": {},
  "pose": {},
  "expression": {},
  "background": {},
  "situation": {}
}
各カテゴリは{タグ: 信頼度0.0〜1.0}です。
一般タグは対象・髪色・人数等、キャラクター・作品名は確実な時だけ出力してください。
分からないカテゴリは空のobjectにし、画像にない特徴や品質語を追加しないでください。
ratingは内容タグと混ぜず、必要なら別のratingカテゴリとして返してください。
NSFWを含む画像も、対応可能なら見える事実だけを中立的なタグにしてください。
"""
_TAGGING_VERSION = "vision-tagger-independent/v1"


# {
#   責務: [VisionTaggerProvider: Caption生成とは別の画像推論で分類Tagを返す]
#   フィールド: [backend: OpenAI互換の画像処理モデル]
#   処理: [タグ抽出専用の指示を送り、JSON形式と拒否状態を検証する]
# }
@dataclass(frozen=True)
class VisionTaggerProvider:
    """Captionと同じ重みも利用可能だが、画像を別の推論として再評価する。"""

    backend: OpenAICompatibleVisionProvider

    @property
    def provider_id(self) -> str:
        return self.backend.provider_id

    @property
    def model_id(self) -> str:
        return self.backend.model_id

    @property
    def config_fingerprint(self) -> str:
        value = ":".join(
            (self.backend.config_fingerprint, _TAGGING_VERSION, TAGGING_INSTRUCTION)
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    # {
    #   責務: [tag: 画像のみから独立したTaggerの構造化推論を実行する]
    #   処理: [専用指示を送信し、拒否・空・JSON不正を後続Taggerへの切替理由にする]
    #   引数: [image: 静止画像のbytes, mime_type: 検証済みMIME]
    #   戻り値: [object: TagNormalizerへ渡すカテゴリー別JSON]
    #   エラー: [PromptExtractionError: VLM拒否・形式不正・一時障害]
    # }
    def tag(self, image: bytes, mime_type: str) -> object:
        response = self.backend.describe(image, mime_type, TAGGING_INSTRUCTION)
        if response.refusal or response.finish_reason == "content_filter":
            raise PromptExtractionError("refused", "VLM Taggerが画像を拒否した")
        return _json(response.content)

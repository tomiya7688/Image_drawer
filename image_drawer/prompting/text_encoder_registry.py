"""テキストエンコーダーをJSON設定と開発者APIから自由に登録する。"""

from __future__ import annotations

from typing import Callable, Mapping

from .text_condition import (
    HashedNgramTextEncoder,
    SentenceTransformerEncoder,
    TextConditionEncoder,
    TextEncoderProvider,
)

EncoderFactory = Callable[[dict], TextEncoderProvider]


# {
#   責務: [TextEncoderRegistry: 任意の実行環境・モデルのエンコーダーをユーザーが選択する]
#   フィールド: [_adapters: 明示的に信頼したAdapter factory]
#   処理: [固定モデルを強制せず、互換性のあるモデル列だけを構築する]
# }
class TextEncoderRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, EncoderFactory] = {}
        self.register_adapter("sentence_transformer", self._sentence_transformer)
        self.register_adapter("hashed_ngram", self._hashed_ngram)

    # {
    #   責務: [register_adapter: Python側で検査したEncoderファクトリを追加する]
    #   引数: [name: Adapter名, factory: ユーザー定義factory]
    #   戻り値: [None]
    # }
    def register_adapter(self, name: str, factory: EncoderFactory) -> None:
        if not isinstance(name, str) or not name.strip() or name in self._adapters:
            raise ValueError("text encoder adapter名が空、または登録済み")
        if not callable(factory):
            raise TypeError("factoryはcallable")
        self._adapters[name] = factory

    @staticmethod
    def _sentence_transformer(entry: dict) -> SentenceTransformerEncoder:
        return SentenceTransformerEncoder(
            encoder_id=entry["id"],
            model_id=entry["model_id"],
            model_revision=entry.get("model_revision", "main"),
            device=entry["device"],
            dimension=entry.get("dimension", 384),
        )

    @staticmethod
    def _hashed_ngram(entry: dict) -> HashedNgramTextEncoder:
        return HashedNgramTextEncoder(
            encoder_id=entry["id"], dimension=entry.get("dimension", 64)
        )

    # {
    #   責務: [build: エンコーダーを指定順に構成し、意味空間の互換性を検証する]
    #   引数: [settings: アプリ全体のJSON設定object]
    #   戻り値: [TextConditionEncoder: モデル別のGPU/CPUフォールバック列]
    # }
    def build(self, settings: Mapping) -> TextConditionEncoder:
        config = settings.get("text_encoder")
        if not isinstance(config, dict):
            raise ValueError("text_encoder設定が必要")
        items = config.get("providers", [])
        if not isinstance(items, list) or not 1 <= len(items) <= 256:
            raise ValueError("text_encoder.providersは1〜256個必要")
        providers: list[TextEncoderProvider] = []
        for entry in items:
            if not isinstance(entry, dict):
                raise ValueError("text_encoder providerはobject")
            name = entry.get("adapter")
            if name not in self._adapters:
                raise ValueError("未登録のtext encoder adapter: " + str(name))
            providers.append(self._adapters[name](entry))
        return TextConditionEncoder(providers)

"""学習と推論で同じ意味空間を使うテキスト条件化と自動CPUオフロード。"""

from __future__ import annotations

import gc
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .core import normalize_prompt


# {
#   責務: [TextEncodingError: モデル切替が必要な失敗を分類する]
#   フィールド: [kind: 失敗理由（oom/unavailable/invalid_output/all_failed）]
# }
class TextEncodingError(RuntimeError):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


# {
#   責務: [TextEncoderProvider: ユーザー登録するテキスト特徴抽出器の境界]
#   フィールド: [encoder_id, model_id, model_revision, dimension, feature_space_id]
# }
class TextEncoderProvider(Protocol):
    encoder_id: str
    model_id: str
    model_revision: str
    dimension: int
    feature_space_id: str

    def encode(self, text: str) -> object: ...
    def release(self) -> None: ...


# {
#   責務: [EncoderAttempt: 各デバイスへの試行と採択状況を記録する]
#   フィールド: [encoder_id: 登録ID, status: ok/oom/unavailable/invalid_output]
# }
@dataclass(frozen=True)
class EncoderAttempt:
    encoder_id: str
    status: str


# {
#   責務: [EncodedText: Promptと互換性ある意味特徴をPolicyへ渡す]
#   フィールド: [values: 正規化済み特徴, feature_space_id: 学習と推論の一致確認]
# }
@dataclass(frozen=True)
class EncodedText:
    values: tuple[float, ...]
    feature_space_id: str
    model_id: str
    model_revision: str
    encoder_id: str
    attempts: tuple[EncoderAttempt, ...]


# {
#   責務: [SentenceTransformerEncoder: 軽量SentenceTransformerを必要な時だけ読む]
#   フィールド: [encoder_id, model_id, model_revision, device, dimension]
#   処理: [GPUかCPUで埋め込みを計算し、終了後はモデルとGPUキャッシュを解放]
# }
class SentenceTransformerEncoder:
    def __init__(
        self,
        encoder_id: str,
        model_id: str,
        device: str,
        dimension: int = 384,
        model_revision: str = "main",
    ) -> None:
        if device not in {"cpu", "cuda"} or dimension <= 0:
            raise ValueError("text encoderのdeviceまたはdimensionが不正")
        if not encoder_id or not model_id or not model_revision:
            raise ValueError("text encoder ID/モデル/改訂を指定してください")
        self.encoder_id = encoder_id
        self.model_id = model_id
        self.model_revision = model_revision
        self.device = device
        self.dimension = dimension
        self.feature_space_id = f"sentence-transformer:{model_id}@{model_revision}"
        self._model: object | None = None

    # {
    #   責務: [encode: ローカル埋め込みモデルでテキストを数値へ変換する]
    #   処理: [モデルを遅延ロードし、推論のみ実行して数値ベクトルを返す]
    #   引数: [text: 正規化済み非空文字列]
    #   戻り値: [object: ベクトル配列]
    # }
    def encode(self, text: str) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise TextEncodingError("unavailable", "sentence-transformersが未導入") from exc
            self._model = SentenceTransformer(
                self.model_id, revision=self.model_revision, device=self.device
            )
        return self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

    # {
    #   責務: [release: 埋め込みモデルのメモリを毎回解放する]
    #   処理: [参照を破棄し、可能ならPyTorchの未使用GPUキャッシュを開放する]
    #   戻り値: [None: 次の画像モデルが利用できるようにする]
    # }
    def release(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


# {
#   責務: [HashedNgramTextEncoder: 実モデルが無い場合の明示的な軽量ベースライン]
#   フィールド: [encoder_id, dimension, feature_space_id]
#   処理: [文字N-gramを固定SHAに埋め込み。学習済みの意味特徴とは称しない]
# }
class HashedNgramTextEncoder:
    def __init__(self, encoder_id: str = "hashed-v1", dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimensionは正の整数")
        self.encoder_id = encoder_id
        self.model_id = "hashed-ngram-v1"
        self.model_revision = "v1"
        self.dimension = dimension
        self.feature_space_id = f"hashed-ngram-v1:dim{dimension}"

    def encode(self, text: str) -> tuple[float, ...]:
        chars = normalize_prompt(text)
        buckets = [0.0] * self.dimension
        for n in (1, 2, 3):
            for i in range(max(0, len(chars) - n + 1)):
                digest = hashlib.sha256(chars[i : i + n].encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                buckets[index] += 1.0 if digest[8] % 2 == 0 else -1.0
        return tuple(buckets)

    def release(self) -> None:
        return None


# {
#   責務: [TextConditionEncoder: 同じ意味空間のGPU→CPU自動フォールバック]
#   フィールド: [providers: 登録順の同一モデル/同一改訂の実行先]
#   処理: [失敗したモデルを解放して次のデバイスを試す。別モデルは混ぜない]
# }
class TextConditionEncoder:
    def __init__(self, providers: list[TextEncoderProvider]) -> None:
        if not providers:
            raise ValueError("テキストエンコーダーが未登録")
        first = providers[0]
        identities = {(p.model_id, p.model_revision, p.dimension, p.feature_space_id)
                      for p in providers}
        if len(identities) != 1:
            raise ValueError("互換性のないエンコーダーをフォールバックに混在できない")
        if len({p.encoder_id for p in providers}) != len(providers):
            raise ValueError("encoder_idが重複しています")
        self.providers = tuple(providers)
        self.feature_space_id = first.feature_space_id
        self.dimension = first.dimension

    # {
    #   責務: [encode: 学習/推論で同じ非空Promptを意味ベクトルに変換する]
    #   処理: [実行エラーならrelease後に同一モデルの別デバイスへ移り、無効値は拒否]
    #   引数: [text: 元の学習用または推論用Prompt]
    #   戻り値: [EncodedText: 正規化ベクトルと使ったデバイスの来歴]
    # }
    def encode(self, text: str) -> EncodedText:
        prompt = normalize_prompt(text)
        attempts: list[EncoderAttempt] = []
        for provider in self.providers:
            try:
                result = provider.encode(prompt)
                values = tuple(float(v) for v in result)
                if len(values) != provider.dimension or not all(
                    math.isfinite(value) for value in values
                ):
                    raise TextEncodingError("invalid_output", "埋め込みの次元または数値が不正")
                norm = math.sqrt(sum(v * v for v in values))
                if norm <= 1e-12:
                    raise TextEncodingError("invalid_output", "ゼロ埋め込みは使用しない")
                normalized = tuple(v / norm for v in values)
                attempts.append(EncoderAttempt(provider.encoder_id, "ok"))
                return EncodedText(
                    normalized,
                    provider.feature_space_id,
                    provider.model_id,
                    provider.model_revision,
                    provider.encoder_id,
                    tuple(attempts),
                )
            except (OSError, MemoryError, RuntimeError, ValueError, TypeError) as exc:
                if isinstance(exc, TextEncodingError):
                    kind = exc.kind
                else:
                    msg = str(exc).casefold()
                    kind = "oom" if isinstance(exc, MemoryError) or "out of memory" in msg else "unavailable"
                attempts.append(EncoderAttempt(provider.encoder_id, kind))
            finally:
                # GPUメモリを保持したまま次のモデルへ移らない。
                provider.release()
        details = ",".join(f"{p.encoder_id}:{p.status}" for p in attempts)
        raise TextEncodingError("all_encoders_failed", details)

    def validate_checkpoint_space(self, checkpoint_feature_space_id: str, dim: int) -> None:
        if checkpoint_feature_space_id != self.feature_space_id or dim != self.dimension:
            raise TextEncodingError(
                "incompatible_checkpoint",
                "学習checkpointとテキストエンコーダーの意味空間が一致しない",
            )


# {
#   責務: [load_text_encoder_from_settings: アプリ設定から実行先とフォールバックを構築する]
#   処理: [固定したモデル改訂・次元が一致したGPU→CPU列だけを許可する]
#   引数: [settings_path: ユーザー設定JSON]
#   戻り値: [TextConditionEncoder: 逐次実行するエンコーダー]
# }
def load_text_encoder_from_settings(settings_path: str | Path) -> TextConditionEncoder:
    import json

    from .text_encoder_registry import TextEncoderRegistry

    settings = json.loads(Path(settings_path).read_text(encoding="utf-8"))
    return TextEncoderRegistry().build(settings)

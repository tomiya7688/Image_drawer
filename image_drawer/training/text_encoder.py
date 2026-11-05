"""文章を固定長ベクトルへ変える TextEncoder(Protocol、ハッシュn-gram、登録口)。"""

from __future__ import annotations

import hashlib
from typing import Callable, Protocol

import numpy as np
from numpy.typing import NDArray


# {
#   責務: [
#     TextEncoder: training_promptをfloat32ベクトルへ変換する契約
#   ]
#   フィールド: [
#     encoder_version: checkpointとExperimentRecordへ保存する版文字列
#     dim: 出力ベクトルの次元
#   ]
# }
class TextEncoder(Protocol):
    encoder_version: str
    dim: int

    # {
    #   責務: [encode: 文章を形状(dim,)、有限値、L2ノルム1のベクトルにする]
    #   エラー: [ValueError: 空文字または空白のみ]
    # }
    def encode(self, training_prompt: str) -> NDArray[np.float32]: ...


# {
#   責務: [
#     HashedNgramTextEncoder: 文字n-gramをSHA-256で符号付きバケットへ加算し正規化する。
#     学習パラメータを持たず、Pythonのhash()に依存しない
#   ]
#   フィールド: [
#     dim: バケット数
#     ngram_range: (最小n, 最大n)
#     encoder_version: "hashed-ngram-v1"
#   ]
#   処理: [
#     1: 空白を畳んだ文字列から各nのn-gramを取り出す
#     2: SHA-256の先頭バイトでバケットと符号を決めて加算する
#     3: L2正規化する(全て相殺された場合は先頭n-gramのバケットへ退避して非零にする)
#   ]
# }
class HashedNgramTextEncoder:
    encoder_version = "hashed-ngram-v1"

    # {
    #   責務: [__init__: 次元とn-gram範囲を検証して保持する]
    #   引数: [dim: 1以上, ngram_range: 1以上でmin<=max]
    #   エラー: [ValueError: 範囲が不正]
    # }
    def __init__(self, dim: int = 64, ngram_range: tuple[int, int] = (1, 3)) -> None:
        low, high = ngram_range
        if dim < 1 or low < 1 or high < low:
            raise ValueError(f"dimとngram_rangeが不正です: dim={dim}, ngram_range={ngram_range}")
        self.dim = dim
        self.ngram_range = (low, high)

    # {
    #   責務: [encode: 文章を決定的なL2ノルム1のfloat32ベクトルにする]
    #   引数: [training_prompt: 学習用の文章]
    #   戻り値: [vector: 形状(dim,)のfloat32]
    #   エラー: [ValueError: 空文字・空白のみ]
    # }
    def encode(self, training_prompt: str) -> NDArray[np.float32]:
        # 空白の揺れで別ベクトルにならないよう、連続空白を1つに畳む。
        text = " ".join(training_prompt.split())
        if not text:
            raise ValueError("training_promptが空です")

        # n-gramごとにSHA-256でバケットと符号を決め、float64で足し込む。
        vector = np.zeros(self.dim, dtype=np.float64)
        low, high = self.ngram_range
        for size in range(low, high + 1):
            for start in range(len(text) - size + 1):
                digest = hashlib.sha256(text[start : start + size].encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self.dim
                vector[bucket] += 1.0 if digest[8] & 1 else -1.0

        # 符号の相殺で零ベクトルになった場合も、ノルム1の契約を守る。
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[
                int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % self.dim
            ] = 1.0
            norm = 1.0
        return (vector / norm).astype(np.float32)


_ENCODER_FACTORIES: dict[str, Callable[[], TextEncoder]] = {
    "hashed-ngram-v1": HashedNgramTextEncoder,
}


# {
#   責務: [register_text_encoder: TextEncoderの生成関数をIDで登録する]
#   引数: [encoder_id: 一意なID, factory: 引数なしで生成する関数]
#   エラー: [ValueError: IDが空、または既に別の関数で登録済み]
# }
def register_text_encoder(encoder_id: str, factory: Callable[[], TextEncoder]) -> None:
    if not encoder_id:
        raise ValueError("encoder_idは空にできません")
    # 登録の上書きは、同じIDで別の挙動になりcheckpointの再現性を壊すため拒否する。
    existing = _ENCODER_FACTORIES.get(encoder_id)
    if existing is not None and existing is not factory:
        raise ValueError(f"encoder_idは登録済みです: {encoder_id}")
    _ENCODER_FACTORIES[encoder_id] = factory


# {
#   責務: [get_text_encoder: 登録IDからTextEncoderを生成する]
#   引数: [encoder_id: register_text_encoderで登録したID]
#   戻り値: [encoder: 新しいTextEncoder]
#   エラー: [ValueError: 未登録のID]
# }
def get_text_encoder(encoder_id: str) -> TextEncoder:
    factory = _ENCODER_FACTORIES.get(encoder_id)
    if factory is None:
        raise ValueError(f"未登録のtext_encoder_idです: {encoder_id}")
    return factory()


# {
#   責務: [check_encoder_matches: checkpointの版・次元と現在のencoderの一致を確認する]
#   引数: [encoder: 現在のencoder, encoder_version: 保存済みの版, dim: 保存済みの次元]
#   エラー: [CheckpointMismatchError: 版または次元が違う]
# }
def check_encoder_matches(encoder: TextEncoder, encoder_version: str, dim: int) -> None:
    if encoder.encoder_version != encoder_version or encoder.dim != dim:
        raise CheckpointMismatchError(
            f"TextEncoderが一致しません: checkpoint=({encoder_version}, {dim}) "
            f"現在=({encoder.encoder_version}, {encoder.dim})"
        )


# {
#   責務: [CheckpointMismatchError: checkpointと実行環境の版・形状・SHA-256の不一致を表す]
# }
class CheckpointMismatchError(ValueError):
    pass

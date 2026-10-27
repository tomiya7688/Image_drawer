"""候補画像と評価器固有の得点を対応付ける記録。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from image_drawer.core.models import Score
from image_drawer.core.serialization import SerializableModel


# {
#   責務: [
#     ScoreSet: 一つの評価器による候補ごとの生得点と由来を保持する
#   ]
#   フィールド: [
#     id: 得点集合の識別子
#     evaluator: 評価器の名前
#     evaluator_version: 評価器のバージョン
#     candidate_artifact_ids: 得点と同順の候補Artifact ID
#     scores: 評価器が返した各候補の得点
#     metadata: 追加の由来情報
#   ]
#   処理: [
#     候補と得点の件数、評価器の一致、数値の有限性を検証する
#   ]
# }
@dataclass(slots=True)
class ScoreSet(SerializableModel):
    id: str
    evaluator: str
    evaluator_version: str
    candidate_artifact_ids: list[str] = field(default_factory=list)
    scores: list[Score] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # {
    #   責務: [
    #     __post_init__: 作成時の得点集合を検証する
    #   ]
    #   処理: [
    #     不整合を検出して生成を拒否する
    #   ]
    #   引数: [
    #     なし
    #   ]
    #   戻り値: [
    #     なし
    #   ]
    # }
    def __post_init__(self) -> None:
        self.validate()

    # {
    #   責務: [
    #     validate: 現在の候補と得点の整合性を確認する
    #   ]
    #   処理: [
    #     ID、件数、評価器とバージョン、得点の型と有限性を検証する
    #   ]
    #   引数: [
    #     なし
    #   ]
    #   戻り値: [
    #     なし
    #   ]
    # }
    def validate(self) -> None:
        for name, value in (
            ("id", self.id),
            ("evaluator", self.evaluator),
            ("evaluator_version", self.evaluator_version),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"ScoreSet {name} must be a non-empty string")
        if not isinstance(self.candidate_artifact_ids, list) or not isinstance(self.scores, list):
            raise TypeError("ScoreSet candidate_artifact_ids and scores must be lists")
        if len(self.candidate_artifact_ids) != len(self.scores):
            raise ValueError("ScoreSet candidate and score counts must match")
        if any(
            not isinstance(candidate, str) or not candidate.strip()
            for candidate in self.candidate_artifact_ids
        ):
            raise ValueError("ScoreSet candidate IDs must be non-empty strings")
        if len(set(self.candidate_artifact_ids)) != len(self.candidate_artifact_ids):
            raise ValueError("ScoreSet candidate IDs must be unique")
        if not isinstance(self.metadata, dict):
            raise TypeError("ScoreSet metadata must be a dict")
        # 評価器ごとの成分を加工せず、異なる評価器や非有限値の混入だけを防ぐ。
        seen_score_ids: set[str] = set()
        for score in self.scores:
            if not isinstance(score, Score):
                raise TypeError("ScoreSet scores must contain Score records")
            if not isinstance(score.id, str) or not score.id.strip() or score.id in seen_score_ids:
                raise ValueError("ScoreSet score IDs must be unique non-empty strings")
            seen_score_ids.add(score.id)
            if (
                score.evaluator != self.evaluator
                or score.evaluator_version != self.evaluator_version
            ):
                raise ValueError("ScoreSet score evaluator and version must match")
            if not isinstance(score.metadata, dict):
                raise TypeError("Score metadata must be a dict")
            if not isinstance(score.components, dict):
                raise TypeError("Score components must be a dict")
            for name, value in score.components.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("Score component names must be non-empty strings")
                self._validate_number(value)
            if score.overall is not None:
                self._validate_number(score.overall)

    # {
    #   責務: [
    #     _validate_number: 並べ替え可能な有限数値を保証する
    #   ]
    #   処理: [
    #     真偽値、非数値、非有限値を拒否する
    #   ]
    #   引数: [
    #     value: 一つの得点
    #   ]
    #   戻り値: [
    #     なし
    #   ]
    # }
    @staticmethod
    def _validate_number(value: object) -> None:
        if type(value) not in (int, float):
            raise ValueError("Score values must be finite numbers, not booleans")
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError("Score value is outside the finite range") from exc
        if not finite:
            raise ValueError("Score values must be finite numbers, not booleans")

    # {
    #   責務: [
    #     scores_by_candidate: 候補IDをキーにした得点の対応表を作る
    #   ]
    #   処理: [
    #     変更後の整合性を再確認し、一回の走査で対応表を返す
    #   ]
    #   引数: [
    #     なし
    #   ]
    #   戻り値: [
    #     scores: 候補IDと生得点の対応表
    #   ]
    # }
    def scores_by_candidate(self) -> dict[str, Score]:
        self.validate()
        return dict(zip(self.candidate_artifact_ids, self.scores))

    # {
    #   責務: [
    #     score_for: 指定候補の生得点を取得する
    #   ]
    #   処理: [
    #     不整合を拒否し、未登録の候補にはKeyErrorを返す
    #   ]
    #   引数: [
    #     artifact_id: 候補のArtifact ID
    #   ]
    #   戻り値: [
    #     score: 対応する生得点
    #   ]
    # }
    def score_for(self, artifact_id: str) -> Score:
        return self.scores_by_candidate()[artifact_id]

    # {
    #   責務: [
    #     from_dict: 外部の保存形式を型検証して復元する
    #   ]
    #   処理: [
    #     配列と各得点の辞書の形式を確かめ、既存の復元処理へ渡す
    #   ]
    #   引数: [
    #     data: 保存された得点集合
    #   ]
    #   戻り値: [
    #     score_set: 復元後の得点集合
    #   ]
    # }
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScoreSet:
        if not isinstance(data, dict):
            raise TypeError("ScoreSet.from_dict expects a dict")
        for name in ("candidate_artifact_ids", "scores"):
            if name in data and not isinstance(data[name], list):
                raise TypeError(f"ScoreSet {name} must be a list")
        if "metadata" in data and not isinstance(data["metadata"], dict):
            raise TypeError("ScoreSet metadata must be a dict")
        # 共通デシリアライザーが誤った型を反復処理する前に、外部形式を検証する。
        for item in data.get("scores", []):
            if not isinstance(item, dict):
                raise TypeError("ScoreSet scores must contain objects")
            for name in ("components", "metadata"):
                if name in item and not isinstance(item[name], dict):
                    raise TypeError(f"Score {name} must be an object")
        return super(ScoreSet, cls).from_dict(data)

    # {
    #   責務: [
    #     to_dict: 現在も整合する得点集合だけを保存形式へ変換する
    #   ]
    #   処理: [
    #     保存前に再検証し、既存のJSON変換を利用する
    #   ]
    #   引数: [
    #     なし
    #   ]
    #   戻り値: [
    #     record: JSON互換の辞書
    #   ]
    # }
    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return SerializableModel.to_dict(self)

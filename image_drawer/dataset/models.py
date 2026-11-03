"""画像のみのDatasetで使う記録。Policyが見る入力と、Evaluatorだけが見る目標を別の型にする。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from image_drawer.core import SerializableModel

SPLITS = ("train", "validation", "test")


# {
#   責務: [
#     ExtractorInfo: 抽出器を一意に識別し、cache_keyと再現性の記録に使う情報を保持する
#   ]
#   フィールド: [
#     id: 抽出器の識別名
#     version: 抽出器のversion
#     capabilities: "tag" または "caption" の一覧
#     model_hash: モデルの重みなどのハッシュ。モデルを持たない抽出器は "none"
#     config_hash: 設定のハッシュ
#     license: ライセンスの表記
#     languages: 出力できる言語
#     runtime: 実行環境(例: cpu)
#   ]
# }
@dataclass(slots=True)
class ExtractorInfo(SerializableModel):
    id: str
    version: str
    capabilities: list[str]
    model_hash: str = "none"
    config_hash: str = "none"
    license: str = "project"
    languages: list[str] = field(default_factory=lambda: ["ja"])
    runtime: str = "cpu"


# {
#   責務: [
#     QualityIssue: 抽出結果の品質警告を保持する。警告は画像を除外する理由にしない
#   ]
#   フィールド: [
#     code: 警告の種類(empty_prompt、low_confidence、extractor_failed など)
#     message: 利用者向けの説明
#   ]
# }
@dataclass(slots=True)
class QualityIssue(SerializableModel):
    code: str
    message: str


# {
#   責務: [
#     TrainingImage: 登録した画像1件と、その抽出結果・分割・権利状態を保持する
#   ]
#   フィールド: [
#     image_id: Dataset内の一意なID
#     image_sha256: 正規化前のファイル内容のSHA-256
#     group_id: 同一作品とみなす近重複のグループ。分割はグループ単位で行う
#     split: train、validation、test のいずれか
#     image_path: Datasetディレクトリからの相対パス(正規化後のPNG)
#     width, height: 正規化後の画素数
#     normalization_version: 画像の正規化処理のversion
#     tags: ラベルとconfidenceの一覧
#     caption: CaptionExtractorの出力
#     training_prompt: PromptBuilderが組み立てた学習用prompt
#     tagger, captioner: 使った抽出器
#     prompt_builder_version: PromptBuilderのversion
#     cache_key: 画像と抽出器の組み合わせから決まる再利用の鍵
#     quality_issues: 品質警告
#   ]
# }
@dataclass(slots=True)
class TrainingImage(SerializableModel):
    image_id: str
    image_sha256: str
    group_id: str
    split: str
    image_path: str
    width: int
    height: int
    normalization_version: str
    tags: list[dict[str, Any]]
    caption: str
    training_prompt: str
    tagger: ExtractorInfo
    captioner: ExtractorInfo
    prompt_builder_version: str
    cache_key: str
    quality_issues: list[QualityIssue] = field(default_factory=list)


# {
#   責務: [
#     PolicyInput: 描画Policyに渡してよい入力だけを持つ。目標画像の情報を含めない
#   ]
#   フィールド: [
#     image_id: 結果を対応づけるID(画像の内容ではない)
#     prompt: 学習用prompt
#   ]
# }
@dataclass(frozen=True, slots=True)
class PolicyInput:
    image_id: str
    prompt: str


# {
#   責務: [
#     RewardTarget: 目標画像への参照。Trainerの評価器だけが読み、Policyには渡さない
#   ]
#   フィールド: [
#     image_id: 対応するPolicyInputのID
#     image_path: 目標画像の絶対パス
#     image_sha256: 目標画像のハッシュ
#   ]
# }
@dataclass(frozen=True, slots=True)
class RewardTarget:
    image_id: str
    image_path: str
    image_sha256: str

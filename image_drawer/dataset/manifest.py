"""Datasetのmanifest(登録内容の一覧)の保存・読込・ハッシュと、Policy入力と目標の分離した読み出し。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from image_drawer.core import SerializableModel
from image_drawer.dataset.models import (
    SPLITS,
    ExtractorInfo,
    PolicyInput,
    RewardTarget,
    TrainingImage,
)

MANIFEST_NAME = "manifest.json"
# 後から足した任意の項目。空のときはdataset_hashの計算から除き、古いmanifestのハッシュを保つ
_OPTIONAL_IMAGE_FIELDS = ("caption_variants", "extraction_trace")
VARIANT_NAMES = ("short", "standard", "detailed")
VARIANT_WEIGHTS = (0.4, 0.4, 0.2)


# {
#   責務: [
#     DatasetManifest: 登録した全画像と、抽出器・正規化・分割の選択を、再現できる形で保持する
#   ]
#   フィールド: [
#     normalization_version: 画像の正規化処理のversion
#     prompt_builder_version: PromptBuilderのversion
#     tagger, captioner: 利用者が選んだ抽出器
#     split_seed: グループ単位の分割に使った値
#     images: 登録した画像。image_id順
#     dataset_hash: 画像のSHA-256・抽出結果・選択から決まるハッシュ。内容が変われば変わる
#   ]
# }
@dataclass(slots=True)
class DatasetManifest(SerializableModel):
    normalization_version: str
    prompt_builder_version: str
    tagger: ExtractorInfo
    captioner: ExtractorInfo
    split_seed: int
    images: list[TrainingImage] = field(default_factory=list)
    dataset_hash: str = ""

    # {
    #   責務: [
    #     compute_hash: 再現性の記録に使うハッシュを計算して保存する
    #   ]
    #   処理: [
    #     dataset_hash以外の全内容を、キー順に整えたJSONにしてSHA-256を取る。
    #     任意の項目(caption_variants、extraction_trace)は空なら含めない
    #   ]
    #   戻り値: [
    #     digest: 16進のSHA-256
    #   ]
    # }
    def compute_hash(self) -> str:
        body = self.to_dict()
        body["dataset_hash"] = ""
        for image in body["images"]:
            for name in _OPTIONAL_IMAGE_FIELDS:
                if not image.get(name):
                    image.pop(name, None)
        payload = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.dataset_hash = hashlib.sha256(payload).hexdigest()
        return self.dataset_hash

    # {
    #   責務: [
    #     validate: manifestの内容が仕様を満たすことを確認する
    #   ]
    #   処理: [
    #     分割値と権利状態の範囲、image_idの重複を検査する
    #   ]
    #   エラー: [
    #     ValueError: 範囲外の値、またはimage_idの重複
    #   ]
    # }
    def validate(self) -> None:
        ids = [image.image_id for image in self.images]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate image_id in manifest")
        for image in self.images:
            if image.split not in SPLITS:
                raise ValueError(f"invalid split: {image.split}")

    # {
    #   責務: [
    #     save: manifestを原子的に書き出す
    #   ]
    #   処理: [
    #     検証とハッシュ計算の後、一時ファイルに書いて置き換え、途中で壊れた状態を残さない
    #   ]
    #   引数: [
    #     dataset_dir: Datasetのディレクトリ
    #   ]
    # }
    def save(self, dataset_dir: Path) -> None:
        self.validate()
        self.compute_hash()
        dataset_dir.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=dataset_dir, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(self.to_json(indent=2))
            os.replace(name, dataset_dir / MANIFEST_NAME)
        finally:
            Path(name).unlink(missing_ok=True)

    # {
    #   責務: [
    #     load: manifestを読み込み、検証と保存済みハッシュの照合を行う
    #   ]
    #   引数: [
    #     dataset_dir: Datasetのディレクトリ
    #   ]
    #   戻り値: [
    #     manifest: 読み込んだmanifest
    #   ]
    #   エラー: [
    #     ValueError: 内容がハッシュと一致しない(手で書き換えられた)、または検証に失敗
    #   ]
    # }
    @classmethod
    def load(cls, dataset_dir: Path) -> DatasetManifest:
        manifest = cls.from_json((dataset_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
        manifest.validate()
        stored = manifest.dataset_hash
        if manifest.compute_hash() != stored:
            raise ValueError("manifest hash does not match its content")
        return manifest


# {
#   責務: [
#     choose_variant: 画像ごとに決定的に、短文40%・標準文40%・詳細文20%から1つ選ぶ
#   ]
#   処理: [
#     SHA256("{seed}:{image_id}")の先頭32bitを[0,1)にし、割合の累積と比べる(比率は#78の仮の分布)
#   ]
#   戻り値: [
#     name: "short"、"standard"、"detailed" のいずれか
#   ]
# }
def choose_variant(seed: int, image_id: str) -> str:
    value = int(hashlib.sha256(f"{seed}:{image_id}".encode()).hexdigest()[:8], 16) / 2**32
    cumulative = 0.0
    for name, weight in zip(VARIANT_NAMES, VARIANT_WEIGHTS):
        cumulative += weight
        if value < cumulative:
            return name
    return VARIANT_NAMES[-1]


def _prompt_of(image: TrainingImage, variant: str) -> str:
    text = image.caption_variants.get(variant, "")
    return text if text.strip() else image.training_prompt


# {
#   責務: [
#     iter_samples: 指定したsplitの画像を、Policyへ渡す入力と、評価器だけが読む目標に分けて返す
#   ]
#   処理: [
#     1: 学習用promptが空の画像は、品質警告に記録したまま対象から外す
#     2: 画像1件ごとに PolicyInput(prompt) と RewardTarget(画像パス)を別の型で返す
#     3: variant_seedを指定すると、画像ごとに決定的に文の長さを選ぶ。指定しなければprompt_variantを使う。
#        変種が無い画像は学習用promptを使う
#   ]
#   引数: [
#     dataset_dir: Datasetのディレクトリ
#     split: train、validation、test のいずれか
#     prompt_variant: "standard"(既定)、"short"、"detailed"
#     variant_seed: 指定すると40:40:20で選ぶ。位置引数の契約(dataset_dir, split)は変わらない
#   ]
#   戻り値: [
#     samples: (PolicyInput, RewardTarget)の組を、image_id順に返すイテレーター
#   ]
#   エラー: [
#     ValueError: splitまたはprompt_variantが不正
#   ]
# }
def iter_samples(
    dataset_dir: Path,
    split: str,
    *,
    prompt_variant: str = "standard",
    variant_seed: int | None = None,
) -> Iterator[tuple[PolicyInput, RewardTarget]]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    if prompt_variant not in VARIANT_NAMES:
        raise ValueError(f"prompt_variant must be one of {VARIANT_NAMES}")
    manifest = DatasetManifest.load(dataset_dir)
    # 学習用promptが空の例は、失敗として記録した上で学習の対象から外す。外部テキストは要求しない。
    selected = [i for i in manifest.images if i.split == split and i.training_prompt.strip()]

    def variant_of(image: TrainingImage) -> str:
        if variant_seed is None:
            return prompt_variant
        return choose_variant(variant_seed, image.image_id)

    return (
        (
            PolicyInput(image_id=image.image_id, prompt=_prompt_of(image, variant_of(image))),
            RewardTarget(
                image_id=image.image_id,
                image_path=str(dataset_dir / image.image_path),
                image_sha256=image.image_sha256,
            ),
        )
        for image in selected
    )

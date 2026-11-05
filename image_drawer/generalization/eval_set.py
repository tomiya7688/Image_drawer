"""凍結した評価集合(eval_set)と、構成的な未見promptのprobe。Datasetが後で変わっても評価集合は変わらない。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from image_drawer.core import SerializableModel
from image_drawer.dataset.manifest import DatasetManifest

EVAL_SET_SCHEMA = "idraw-eval-set/v1"


# {
#   責務: [
#     normalize_prompt: prompt一致の比較用に、空白と大文字小文字を揃える
#   ]
#   引数: [
#     prompt: 学習用または評価用のprompt
#   ]
#   戻り値: [
#     normalized: 正規化した文字列
#   ]
# }
def normalize_prompt(prompt: str) -> str:
    return re.sub(r"\s+", " ", prompt.strip()).casefold()


# {
#   責務: [
#     EvalItem: 評価集合の画像1件を、後から変わらない値で保持する
#   ]
#   フィールド: [
#     image_id, image_sha256, group_id: S2のmanifestの値
#     prompt: 標準の1文
#     prompt_variants: 短・標準・詳細(無ければ空)
#     prompt_seen_in_train: 正規化したpromptがtrainまたはvalidationの学習用promptに一致するか
#     extractor_ids: タグ付けと説明文の抽出器ID
#     prompt_builder_version: PromptBuilderのversion
#   ]
# }
@dataclass(slots=True)
class EvalItem(SerializableModel):
    image_id: str
    image_sha256: str
    group_id: str
    prompt: str
    prompt_variants: dict[str, str] = field(default_factory=dict)
    prompt_seen_in_train: bool = False
    extractor_ids: list[str] = field(default_factory=list)
    prompt_builder_version: str = ""


# {
#   責務: [
#     EvalProbe: 基準itemのpromptから、宣言した1属性だけを入れ替えた構成的な未見prompt
#   ]
#   フィールド: [
#     probe_id: probeの識別子
#     base_image_id: 基準のitem
#     attribute: 変更する属性(color、count、positionなど)
#     original, replacement: 入れ替え前後の語
#     prompt: 入れ替え後のprompt
#   ]
# }
@dataclass(slots=True)
class EvalProbe(SerializableModel):
    probe_id: str
    base_image_id: str
    attribute: str
    original: str
    replacement: str
    prompt: str


# {
#   責務: [
#     EvalSet: 評価に使うtest画像とprobeのスナップショット。eval_set_sha256で全体を固定する
#   ]
#   フィールド: [
#     eval_set_id, created_at: 識別子と作成日時
#     dataset_hash, split_seed: S2のmanifestの値
#     items, probes: 評価対象
#     train_group_ids_sha256: trainとvalidationのgroup_idの分離確認用ハッシュ
#     excluded: 学習用promptが空で対象外となったtest画像のimage_id
#     eval_set_sha256: eval_set_sha256以外の全内容のハッシュ
#   ]
# }
@dataclass(slots=True)
class EvalSet(SerializableModel):
    eval_set_id: str
    created_at: str
    dataset_hash: str
    split_seed: int
    items: list[EvalItem]
    probes: list[EvalProbe] = field(default_factory=list)
    train_group_ids_sha256: str = ""
    excluded: list[str] = field(default_factory=list)
    schema: str = EVAL_SET_SCHEMA
    eval_set_sha256: str = ""

    # {
    #   責務: [
    #     compute_hash: eval_set_sha256以外の全内容のSHA-256を計算して保存する
    #   ]
    #   戻り値: [
    #     digest: 16進のSHA-256
    #   ]
    # }
    def compute_hash(self) -> str:
        body = self.to_dict()
        body["eval_set_sha256"] = ""
        payload = json.dumps(body, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.eval_set_sha256 = hashlib.sha256(payload).hexdigest()
        return self.eval_set_sha256

    # {
    #   責務: [
    #     save: 評価集合を原子的に書き出す
    #   ]
    #   引数: [
    #     path: 保存先のJSONファイル
    #   ]
    # }
    def save(self, path: Path) -> None:
        self.compute_hash()
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(self.to_json(indent=2))
            os.replace(name, path)
        finally:
            Path(name).unlink(missing_ok=True)

    # {
    #   責務: [
    #     load: 評価集合を読み込み、保存済みハッシュと照合する
    #   ]
    #   エラー: [
    #     ValueError: 内容がハッシュと一致しない(書き換えられた)
    #   ]
    # }
    @classmethod
    def load(cls, path: Path) -> EvalSet:
        eval_set = cls.from_json(path.read_text(encoding="utf-8"))
        stored = eval_set.eval_set_sha256
        if eval_set.compute_hash() != stored:
            raise ValueError("eval_set hash does not match its content")
        return eval_set

    # {
    #   責務: [
    #     unseen_prompt_items: 厳密な未見prompt(trainとvalidationに一致しない)のitemだけを返す
    #   ]
    # }
    def unseen_prompt_items(self) -> list[EvalItem]:
        return [item for item in self.items if not item.prompt_seen_in_train]


def _group_ids_hash(group_ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(group_ids)).encode("utf-8")).hexdigest()


# {
#   責務: [
#     _probe_pattern: probeの置換対象の検索パターンを作る
#   ]
#   処理: [
#     英数字だけの語は、前後が英数字・下線でない位置だけに一致させる(catがcertificateに一致しない)。
#     日本語など語の境界が空白で決まらない文字列は、大文字小文字を区別する完全一致の部分文字列とする
#   ]
# }
def _probe_pattern(original: str) -> re.Pattern[str]:
    if not original:
        raise ValueError("probe original must not be empty")
    escaped = re.escape(original)
    if re.fullmatch(r"[A-Za-z0-9_]+", original):
        return re.compile(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])")
    return re.compile(escaped)


# {
#   責務: [
#     build_probes: probeテンプレートから、1属性だけ入れ替えたpromptを作る
#   ]
#   処理: [
#     1: 基準itemのpromptに、originalがちょうど1回だけ現れることを確認する
#     2: originalをreplacementに入れ替える。他の語は変えない
#   ]
#   引数: [
#     items: 基準になる評価item
#     templates: probe_id、base_image_id、attribute、original、replacementを持つ辞書の列
#   ]
#   戻り値: [
#     probes: 作ったprobe
#   ]
#   エラー: [
#     ValueError: 基準itemが無い、originalが0回または複数回現れる、入れ替えが無変更
#   ]
# }
def build_probes(items: list[EvalItem], templates: list[dict[str, str]]) -> list[EvalProbe]:
    by_id = {item.image_id: item for item in items}
    probes: list[EvalProbe] = []
    for template in templates:
        base = by_id.get(template["base_image_id"])
        if base is None:
            raise ValueError(f"probe base item not found: {template['base_image_id']}")
        original, replacement = template["original"], template["replacement"]
        # 宣言した属性以外を動かさないため、置換対象が一意でなければ拒否する。
        pattern = _probe_pattern(original)
        if original == replacement or len(pattern.findall(base.prompt)) != 1:
            raise ValueError(f"probe {template['probe_id']}: original must appear exactly once")
        probes.append(
            EvalProbe(
                probe_id=template["probe_id"],
                base_image_id=base.image_id,
                attribute=template["attribute"],
                original=original,
                replacement=replacement,
                prompt=pattern.sub(lambda _match: replacement, base.prompt, count=1),
            )
        )
    return probes


# {
#   責務: [
#     build_eval_set: DatasetのtestをスナップショットしたEvalSetを作る
#   ]
#   処理: [
#     1: manifestを検証付きで読む
#     2: testのgroup_idがtrain・validationと共有されていないことを確認する(再検査)
#     3: 学習用promptが空のtest画像はexcludedに残し、itemにしない
#     4: promptがtrain・validationと正規化後に一致するかを各itemに記録する
#   ]
#   引数: [
#     dataset_dir: Datasetのディレクトリ
#     eval_set_id: 評価集合の識別子
#     probe_templates: build_probesへ渡すテンプレート
#     created_at: 作成日時。省略すると現在時刻
#   ]
#   エラー: [
#     ValueError: testのgroup_idがtrainまたはvalidationに存在する
#   ]
# }
def build_eval_set(
    dataset_dir: Path,
    eval_set_id: str,
    *,
    probe_templates: list[dict[str, str]] | None = None,
    created_at: str | None = None,
) -> EvalSet:
    manifest = DatasetManifest.load(dataset_dir)
    test = [i for i in manifest.images if i.split == "test"]
    others = [i for i in manifest.images if i.split != "test"]
    other_groups = {i.group_id for i in others}
    shared = sorted({i.group_id for i in test} & other_groups)
    if shared:
        raise ValueError(f"test group_id shared with train/validation: {shared}")
    seen_prompts = {
        normalize_prompt(i.training_prompt) for i in others if i.training_prompt.strip()
    }
    items: list[EvalItem] = []
    excluded: list[str] = []
    for image in sorted(test, key=lambda i: i.image_id):
        if not image.training_prompt.strip():
            excluded.append(image.image_id)
            continue
        items.append(
            EvalItem(
                image_id=image.image_id,
                image_sha256=image.image_sha256,
                group_id=image.group_id,
                prompt=image.training_prompt,
                prompt_variants=dict(image.caption_variants),
                prompt_seen_in_train=normalize_prompt(image.training_prompt) in seen_prompts,
                extractor_ids=[image.tagger.id, image.captioner.id],
                prompt_builder_version=image.prompt_builder_version,
            )
        )
    eval_set = EvalSet(
        eval_set_id=eval_set_id,
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
        dataset_hash=manifest.dataset_hash,
        split_seed=manifest.split_seed,
        items=items,
        probes=build_probes(items, probe_templates or []),
        train_group_ids_sha256=_group_ids_hash(other_groups),
        excluded=excluded,
    )
    eval_set.compute_hash()
    return eval_set

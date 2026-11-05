"""独立抽出したCaptionとTagを照合し、人間が入力する文章に寄せて統合する。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from .core import PromptExtractionError, SceneDescription, normalize_prompt
from .tagging import PreparedTags

COMPOSER_VERSION = "prompt-composer-v2"
_TAG_THRESHOLD = 0.85
_ALLOWED_CATEGORIES = frozenset(
    {"general", "clothes", "pose", "expression", "background", "situation"}
)
_JP_ALIASES = {
    "red apple": "赤いリンゴ",
    "apple": "リンゴ",
    "red hair": "赤い髪",
    "blue hair": "青い髪",
    "standing": "立っている",
    "sitting": "座っている",
    "smile": "笑顔",
    "blue dress": "青いドレス",
    "red dress": "赤いドレス",
    "white background": "白い背景",
    "sky": "空",
}
_OPPOSITES = {
    "red hair": ("青い髪", "blue hair"),
    "blue hair": ("赤い髪", "red hair"),
    "standing": ("座っている", "sitting", "lying"),
    "sitting": ("立っている", "standing"),
    "red dress": ("青いドレス", "blue dress"),
    "blue dress": ("赤いドレス", "red dress"),
}


# {
#   責務: [ComposedPrompt: 統合した自然文とタグの採用・保留理由を保持する]
#   フィールド: [text: 学習条件, included: 使ったタグ, withheld: 保留したタグと理由]
#   処理: [編集・学習用プロンプトの来歴を追えるようにする]
# }
@dataclass(frozen=True)
class ComposedPrompt:
    text: str
    included: tuple[str, ...]
    withheld: tuple[tuple[str, str], ...]
    composer_version: str = COMPOSER_VERSION


# {
#   責務: [PromptComposer: Taggerの結果とCaptionの矛盾・重複を整形する]
#   フィールド: [threshold: 本文へ追加するタグのconfidence閾値]
#   処理: [安全な既知タグだけを自然文に統合し、不明なタグは別成果物に保留]
# }
class PromptComposer:
    def __init__(self, threshold: float = _TAG_THRESHOLD) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("タグ補足のconfidence閾値は0〜1")
        self.threshold = threshold

    # {
    #   責務: [compose: Captionと独立タグから非空の学習プロンプトを作る]
    #   処理: [カテゴリ制限→低信頼除外→重複検知→矛盾除外→既知表現の統合]
    #   引数: [scene: Caption結果, tags: 分類済みTagger結果, seed: 粒度選択]
    #   戻り値: [ComposedPrompt: 人間用の文章とタグ来歴]
    # }
    def compose(self, scene: SceneDescription, tags: PreparedTags, seed: int) -> ComposedPrompt:
        import random

        source = random.Random(seed).choices(
            [scene.short, scene.standard, scene.detailed], weights=[40, 40, 20], k=1
        )[0]
        text = normalize_prompt(source)
        included: list[str] = []
        withheld: list[tuple[str, str]] = []
        has_japanese = bool(re.search(r"[ぁ-んァ-ン一-龥]", text))
        normalized = unicodedata.normalize("NFKC", text).casefold()
        seen: set[str] = set()
        for tag in tags.normalized.tags:
            label = tag.label.replace("_", " ").casefold().strip()
            if label in seen:
                continue
            seen.add(label)
            if tag.category not in _ALLOWED_CATEGORIES:
                withheld.append((tag.label, "excluded_category"))
                continue
            if tag.confidence < self.threshold:
                withheld.append((tag.label, "low_confidence"))
                continue
            phrase = _JP_ALIASES.get(label) if has_japanese else label
            if not phrase:
                withheld.append((tag.label, "untranslated_tag"))
                continue
            if any(
                opposite.casefold() in normalized
                for opposite in _OPPOSITES.get(label, ())
            ):
                withheld.append((tag.label, "conflicts_with_caption"))
                continue
            if phrase.casefold() in normalized or label in normalized:
                included.append(tag.label)
                continue
            # 意味が曖昧な未知の英語Tagを日本語Captionに混ぜない。
            if has_japanese and label not in _JP_ALIASES:
                withheld.append((tag.label, "not_mapped_to_natural_japanese"))
                continue
            delimiter = "、" if has_japanese else "; "
            text += delimiter + phrase
            normalized = unicodedata.normalize("NFKC", text).casefold()
            included.append(tag.label)
        if not text.strip():
            raise PromptExtractionError("empty", "プロンプト統合後に文章が空")
        return ComposedPrompt(
            text=normalize_prompt(text),
            included=tuple(included),
            withheld=tuple(withheld),
        )

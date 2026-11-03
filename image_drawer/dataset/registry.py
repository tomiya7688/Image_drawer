"""TagExtractor・CaptionExtractor・PromptBuilderを名前で登録・解決するregistry。"""

from __future__ import annotations

from typing import Any, Protocol

from image_drawer.dataset.models import ExtractorInfo


# {
#   責務: [
#     TagExtractor: 画像からタグ(ラベルとconfidence)を取り出す共通interface
#   ]
#   フィールド: [
#     info: 抽出器の識別情報。capabilitiesに "tag" を含む
#   ]
# }
class TagExtractor(Protocol):
    info: ExtractorInfo

    def extract(self, image: Any) -> list[dict[str, Any]]: ...


# {
#   責務: [
#     CaptionExtractor: 画像から説明文を取り出す共通interface
#   ]
#   フィールド: [
#     info: 抽出器の識別情報。capabilitiesに "caption" を含む
#   ]
# }
class CaptionExtractor(Protocol):
    info: ExtractorInfo

    def extract(self, image: Any) -> str: ...


# {
#   責務: [
#     ExtractorRegistry: 抽出器をidで登録し、利用者の選択から解決する。モデル名をコードに埋め込まない
#   ]
#   フィールド: [
#     _taggers, _captioners: idから抽出器への対応
#   ]
#   処理: [
#     1: 登録時にcapabilitiesとidの重複を検証する
#     2: 解決時に未登録のidを、登録済みの一覧つきのエラーにする
#   ]
# }
class ExtractorRegistry:
    def __init__(self) -> None:
        self._taggers: dict[str, TagExtractor] = {}
        self._captioners: dict[str, CaptionExtractor] = {}

    # {
    #   責務: [
    #     register_tagger: TagExtractorを登録する
    #   ]
    #   引数: [
    #     extractor: 登録する抽出器
    #   ]
    #   エラー: [
    #     ValueError: capabilitiesに "tag" がない、またはidが登録済み
    #   ]
    # }
    def register_tagger(self, extractor: TagExtractor) -> None:
        self._register(self._taggers, extractor, "tag")

    # {
    #   責務: [
    #     register_captioner: CaptionExtractorを登録する
    #   ]
    #   引数: [
    #     extractor: 登録する抽出器
    #   ]
    #   エラー: [
    #     ValueError: capabilitiesに "caption" がない、またはidが登録済み
    #   ]
    # }
    def register_captioner(self, extractor: CaptionExtractor) -> None:
        self._register(self._captioners, extractor, "caption")

    # {
    #   責務: [
    #     tagger: idからTagExtractorを取得する
    #   ]
    #   引数: [
    #     extractor_id: 利用者が選んだ抽出器のid
    #   ]
    #   戻り値: [
    #     extractor: 登録済みの抽出器
    #   ]
    #   エラー: [
    #     KeyError: 未登録のid
    #   ]
    # }
    def tagger(self, extractor_id: str) -> TagExtractor:
        return self._resolve(self._taggers, extractor_id, "tag")

    # {
    #   責務: [
    #     captioner: idからCaptionExtractorを取得する
    #   ]
    #   引数: [
    #     extractor_id: 利用者が選んだ抽出器のid
    #   ]
    #   戻り値: [
    #     extractor: 登録済みの抽出器
    #   ]
    #   エラー: [
    #     KeyError: 未登録のid
    #   ]
    # }
    def captioner(self, extractor_id: str) -> CaptionExtractor:
        return self._resolve(self._captioners, extractor_id, "caption")

    # {
    #   責務: [
    #     describe: 登録済みの抽出器の情報を一覧する(GUI・CLI・設定が同じ一覧を使う)
    #   ]
    #   戻り値: [
    #     infos: kind("tag"/"caption")ごとのExtractorInfo
    #   ]
    # }
    def describe(self) -> dict[str, list[ExtractorInfo]]:
        return {
            "tag": [extractor.info for extractor in self._taggers.values()],
            "caption": [extractor.info for extractor in self._captioners.values()],
        }

    @staticmethod
    def _register(table: dict, extractor: Any, capability: str) -> None:
        info = extractor.info
        if capability not in info.capabilities:
            raise ValueError(f"extractor {info.id} lacks capability: {capability}")
        if not info.id.strip() or info.id in table:
            raise ValueError(f"extractor id must be unique and non-empty: {info.id!r}")
        table[info.id] = extractor

    @staticmethod
    def _resolve(table: dict, extractor_id: str, kind: str) -> Any:
        try:
            return table[extractor_id]
        except KeyError:
            raise KeyError(
                f"unknown {kind} extractor: {extractor_id} (registered: {sorted(table)})"
            ) from None

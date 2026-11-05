"""Dataset画面(G3)の状態。画像ディレクトリと抽出器の選択だけを入力とし、CLIと同じ関数を呼ぶ。

prompt・タグ・Captionの手入力欄は持たない（抽出結果は閲覧だけ）。
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_drawer.dataset import (
    DatasetManifest,
    ExtractorRegistry,
    IngestReport,
    create_default_registry,
    register_images,
)

FAILURES = (OSError, ValueError, KeyError)


# {
#   責務: [ ExtractorEntry: 一覧に出す抽出器1件（ExtractorRegistry.describe()の内容）。external=外部へ画像を送りうる ]
# }
@dataclass(frozen=True)
class ExtractorEntry:
    id: str
    version: str
    capabilities: tuple[str, ...]
    languages: tuple[str, ...]
    runtime: str
    model_hash: str
    external: bool


# {
#   責務: [ ImageRow: 抽出結果の閲覧1行。編集できない ]
# }
@dataclass(frozen=True)
class ImageRow:
    image_id: str
    split: str
    tags: tuple[str, ...]
    caption: str
    training_prompt: str
    tagger: str
    captioner: str
    cache_key: str
    quality_codes: tuple[str, ...]


# {
#   責務: [ DatasetSummary: Datasetの概要（image-drawer-dataset showと同じ内容＋閲覧用の行） ]
# }
@dataclass(frozen=True)
class DatasetSummary:
    images: int
    splits: dict[str, int]
    dataset_hash: str
    tagger: str
    captioner: str
    quality_issues: int
    rows: tuple[ImageRow, ...]


def summarize_dataset(dataset_dir: Path) -> DatasetSummary:
    manifest = DatasetManifest.load(dataset_dir)
    splits: dict[str, int] = {}
    for image in manifest.images:
        splits[image.split] = splits.get(image.split, 0) + 1
    rows = tuple(
        ImageRow(
            i.image_id,
            i.split,
            tuple(str(t.get("name", t)) for t in i.tags),
            i.caption,
            i.training_prompt,
            f"{i.tagger.id}@{i.tagger.version}",
            f"{i.captioner.id}@{i.captioner.version}",
            i.cache_key,
            tuple(q.code for q in i.quality_issues),
        )  # fmt: skip
        for i in manifest.images
    )
    return DatasetSummary(
        len(manifest.images), splits, manifest.dataset_hash, manifest.tagger.id,
        manifest.captioner.id, sum(len(r.quality_codes) for r in rows), rows,
    )  # fmt: skip


@dataclass(frozen=True)
class RegistrationIdle:
    pass


@dataclass(frozen=True)
class Registering:
    pass


@dataclass(frozen=True)
class RegistrationDone:
    report: IngestReport
    summary: DatasetSummary
    updated_cache_keys: int


@dataclass(frozen=True)
class RegistrationFailed:
    reason: str


RegistrationState = RegistrationIdle | Registering | RegistrationDone | RegistrationFailed


# {
#   責務: [
#     DatasetPanelModel: 抽出器の一覧・選択・登録の状態遷移。登録はimage_drawer.datasetのregister_imagesをそのまま呼ぶ
#   ]
# }
class DatasetPanelModel:
    def __init__(
        self,
        registry: ExtractorRegistry | None = None,
        *,
        register: Callable[..., IngestReport] = register_images,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.register = register
        self.state: RegistrationState = RegistrationIdle()
        entries = self.entries()
        self.tagger_id = entries["tag"][0].id
        self.captioner_id = entries["caption"][0].id
        self._messages: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._thread: threading.Thread | None = None

    def entries(self) -> dict[str, list[ExtractorEntry]]:
        described = self.registry.describe()
        external = {
            "tag": self._external_ids(self.registry.tagger, described["tag"]),
            "caption": self._external_ids(self.registry.captioner, described["caption"]),
        }
        return {
            kind: [
                ExtractorEntry(
                    i.id,
                    i.version,
                    tuple(i.capabilities),
                    tuple(i.languages),
                    i.runtime,
                    i.model_hash,
                    i.id in external[kind],
                )  # fmt: skip
                for i in infos
            ]
            for kind, infos in described.items()
        }

    @staticmethod
    def _external_ids(resolve: Callable[[str], Any], infos: list[Any]) -> set[str]:
        found = set()
        for info in infos:
            extractor = resolve(info.id)
            if hasattr(extractor, "external_destinations") and extractor.external_destinations():
                found.add(info.id)
        return found

    def select(self, *, tagger_id: str | None = None, captioner_id: str | None = None) -> None:
        if tagger_id is not None:
            self.registry.tagger(tagger_id)  # 未登録はKeyError
            self.tagger_id = tagger_id
        if captioner_id is not None:
            self.registry.captioner(captioner_id)
            self.captioner_id = captioner_id

    # {
    #   責務: [ external_warnings: 選択中の抽出器が外部へ画像を送りうる場合の警告文 ]
    # }
    def external_warnings(self) -> list[str]:
        flags = {e.id: e.external for kind in self.entries().values() for e in kind}
        return [
            f"{name}は外部へ画像を送る可能性があります（許可しない限り送信されません）"
            for name in (self.tagger_id, self.captioner_id)
            if flags.get(name)
        ]

    # {
    #   責務: [ selection_differences: 選択とDatasetのmanifestに記録された抽出器の差分を文にする ]
    # }
    def selection_differences(self, dataset_dir: Path) -> list[str]:
        if not (Path(dataset_dir) / "manifest.json").is_file():
            return []
        manifest = DatasetManifest.load(Path(dataset_dir))
        pairs = (
            ("Tagger", self.tagger_id, manifest.tagger.id),
            ("Captioner", self.captioner_id, manifest.captioner.id),
        )
        return [f"{n}: 選択={sel} / 記録={rec}" for n, sel, rec in pairs if sel != rec]

    # {
    #   責務: [ start: 登録(または抽出器を変えた再抽出)を別スレッドで開始する。登録中は開始しない ]
    # }
    def start(
        self,
        source_dir: Path,
        dataset_dir: Path,
        *,
        seed: int = 0,
        allow_external_upload: bool = False,
    ) -> bool:
        if isinstance(self.state, Registering):
            return False
        self.state = Registering()
        self._messages = queue.Queue()
        self._thread = threading.Thread(
            target=self._work,
            args=(Path(source_dir), Path(dataset_dir), seed, allow_external_upload, self._messages),
            daemon=True,
        )
        self._thread.start()
        return True

    def _previous_keys(self, dataset_dir: Path) -> dict[str, str]:
        if not (dataset_dir / "manifest.json").is_file():
            return {}
        return {i.image_id: i.cache_key for i in DatasetManifest.load(dataset_dir).images}

    def _work(self, source_dir, dataset_dir, seed, allow_external_upload, messages) -> None:
        try:
            before = self._previous_keys(dataset_dir)
            report = self.register(
                source_dir, dataset_dir, self.registry,
                tagger_id=self.tagger_id, captioner_id=self.captioner_id, seed=seed,
                allow_external_upload=allow_external_upload,
            )  # fmt: skip
            summary = summarize_dataset(dataset_dir)
            updated = sum(
                1 for r in summary.rows if before.get(r.image_id, r.cache_key) != r.cache_key
            )
        except FAILURES as error:
            messages.put(("failed", str(error)))
        else:
            messages.put(("done", report, summary, updated))

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    def poll(self) -> RegistrationState:
        while True:
            try:
                kind, *rest = self._messages.get_nowait()
            except queue.Empty:
                return self.state
            if not isinstance(self.state, Registering):
                continue
            if kind == "failed":
                self.state = RegistrationFailed(rest[0])
            else:
                self.state = RegistrationDone(*rest)

    def reset(self) -> None:
        if not isinstance(self.state, Registering):
            self.state = RegistrationIdle()

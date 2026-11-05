"""Episode v2のcanonicalな保存・厳密な読込・1ファイルの原子的な置換。

v1のEpisodeは読込のみ対応し、保存はv2だけ（v1 fixtureを書き換えない）。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Final

from .engine import RENDERER_V2
from .episode import DrawingEpisode, loads_strict
from .episode_groups import truncate_to_last_complete_group, validate_episode_structure
from .models import validate_canvas
from .validation import DrawingError, EpisodeFormatError, EpisodeStorageError

EPISODE_ARTIFACT_TYPE: Final[str] = "Episode"
MAX_EPISODE_BYTES: Final[int] = 16 * 1024 * 1024
SCHEMA_V1: Final[str] = "drawing-episode/v1"
SCHEMA_V2: Final[str] = "drawing-episode/v2"
BOM: Final[bytes] = b"\xef\xbb\xbf"


# {
#   責務: [ _compact: キー辞書順・空白なしのJSON文字列にする ]
# }
def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


# {
#   責務: [ _canonical_text: 最上位キー辞書順、eventsは1件1行のcanonical文書を作る（末尾LF1つ） ]
# }
def _canonical_text(document: dict[str, Any]) -> str:
    parts = []
    for key in sorted(document):
        if key == "events":
            lines = ",\n".join(_compact(item) for item in document[key])
            parts.append(f'"events":[\n{lines}\n]')
        else:
            parts.append(f"{_compact(key)}:{_compact(document[key])}")
    return "{" + ",".join(parts) + "}\n"


# {
#   責務: [ _to_bytes: 文字列またはバイト列をサイズ・BOM・UTF-8の規則で文字列へ復号する ]
#   エラー: [ EpisodeFormatError: サイズ超過・BOM・不正なUTF-8 ]
# }
def _to_text(data: str | bytes) -> str:
    try:
        raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    except UnicodeEncodeError as exc:
        raise EpisodeFormatError(f"input is not valid UTF-8: {exc}") from exc
    if len(raw) > MAX_EPISODE_BYTES:
        raise EpisodeFormatError(f"episode exceeds {MAX_EPISODE_BYTES} bytes")
    if raw.startswith(BOM):
        raise EpisodeFormatError("a BOM is not allowed")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EpisodeFormatError(f"input is not valid UTF-8: {exc}") from exc


# {
#   責務: [ _parse: 厳密JSONを読み、schema_versionで分岐してDrawingEpisodeにする（構造検証の前まで） ]
#   エラー: [ EpisodeFormatError: JSON・識別・header・eventの違反 ]
# }
def _parse(text: str) -> DrawingEpisode:
    try:
        document = loads_strict(text)
        if not isinstance(document, dict):
            raise DrawingError("the top level must be an object")
        version = document.get("schema_version")
        if version not in (SCHEMA_V1, SCHEMA_V2):
            raise DrawingError(f"unsupported schema_version: {version!r}")
        canvas = document.get("canvas")
        if isinstance(canvas, dict):
            validate_canvas(canvas.get("width"), canvas.get("height"), document.get("seed", 0))
        episode = DrawingEpisode.from_dict(document)
    except (DrawingError, ValueError, RecursionError) as exc:
        if isinstance(exc, EpisodeFormatError):
            raise
        raise EpisodeFormatError(str(exc)) from exc
    return episode


# {
#   責務: [ loads_episode: 文字列/バイト列をEpisodeにする。v2はグループ・工程連鎖まで検査する ]
#   引数: [ text: 文書, recover: Trueのときだけ末尾の未確定グループを切り詰めてdraftにする ]
#   エラー: [ EpisodeFormatError: いずれかの検査に失敗（回復は構造の末尾切り詰めだけ） ]
# }
def loads_episode(text: str | bytes, *, recover: bool = False) -> DrawingEpisode:
    episode = _parse(_to_text(text))
    if episode.renderer_version != RENDERER_V2:
        return episode
    if recover:
        episode = truncate_to_last_complete_group(episode)
    validate_episode_structure(episode)
    return episode


# {
#   責務: [ dumps_episode: v2 Episodeをcanonical形式の文字列にする。読み戻せない内容は拒否する ]
#   エラー: [ EpisodeFormatError: v1、または検証に失敗するEpisode ]
# }
def dumps_episode(episode: DrawingEpisode) -> str:
    if episode.renderer_version != RENDERER_V2:
        raise EpisodeFormatError("only v2 episodes can be saved")
    try:
        text = _canonical_text(episode.to_dict())
    except (DrawingError, TypeError, ValueError) as exc:
        raise EpisodeFormatError(f"episode cannot be serialized: {exc}") from exc
    loads_episode(text)  # 書き出す内容が必ず読み戻せることを保証する
    return text


# {
#   責務: [ episode_sha256: canonical形式のバイト列のSHA-256（16進） ]
# }
def episode_sha256(episode: DrawingEpisode) -> str:
    return hashlib.sha256(dumps_episode(episode).encode("utf-8")).hexdigest()


# {
#   責務: [ _sync_directory: POSIXでだけ保存先ディレクトリをfsyncする（置換の永続化） ]
# }
def _sync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# {
#   責務: [ save_episode: 検証→canonical化→一時ファイル書込→fsync→os.replace→ディレクトリfsyncで1ファイルを置換する ]
#   処理: [ 失敗時は一時ファイルを削除し、既存の保存先を変更しない。保存先ディレクトリは作らない ]
#   戻り値: [ sha256: 保存したバイト列のSHA-256（16進） ]
#   エラー: [ EpisodeFormatError: 検証失敗, EpisodeStorageError: ファイル操作の失敗 ]
# }
def save_episode(episode: DrawingEpisode, path: str | os.PathLike[str]) -> str:
    data = dumps_episode(episode).encode("utf-8")
    target = Path(path)
    temp_path: str | None = None
    try:
        descriptor, temp_path = tempfile.mkstemp(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
        )
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        temp_path = None
        _sync_directory(target.parent)
    except OSError as exc:
        raise EpisodeStorageError(f"cannot save {target.name}: {exc}") from exc
    finally:
        if temp_path is not None:
            with contextlib.suppress(OSError):
                os.unlink(temp_path)
    return hashlib.sha256(data).hexdigest()


# {
#   責務: [ load_episode: ファイルを読み、loads_episodeでEpisodeにする ]
#   エラー: [ EpisodeStorageError: 読めない, EpisodeFormatError: 内容が不正 ]
# }
def load_episode(path: str | os.PathLike[str], *, recover: bool = False) -> DrawingEpisode:
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_EPISODE_BYTES + 1)
    except OSError as exc:
        raise EpisodeStorageError(f"cannot read {os.fspath(path)}: {exc}") from exc
    return loads_episode(data, recover=recover)


# {
#   責務: [ episode_artifact_metadata: Episode Artifactのmetadataに入れる値を返す ]
# }
def episode_artifact_metadata(episode: DrawingEpisode, sha256: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_V2,
        "episode_sha256": sha256,
        "status": episode.status,
        "termination_reason": episode.termination_reason,
        "event_count": len(episode.events),
        "group_count": episode.events[-1].group.id + 1,
        "renderer_version": episode.renderer_version,
        "workflow_id": episode.workflow_id,
        "canvas": {"width": episode.width, "height": episode.height},
    }

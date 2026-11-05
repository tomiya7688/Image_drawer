"""保存したEpisode(またはdraft)の検証つき再生。CLI・Step・GUIが共通に呼ぶ。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from image_drawer.drawing import DrawingEpisode, loads_strict
from image_drawer.drawing.engine import RenderedImage
from image_drawer.drawing.episode_io import loads_episode
from image_drawer.drawing.replay import iter_replay, replay
from image_drawer.drawing.validation import DrawingError
from image_drawer.inference.artifacts import DRAFT_SCHEMA


# {
#   責務: [ ReplayMismatchError: Episodeの検証失敗・ハッシュ不一致・再生画素の不一致（CLIの終了コード3） ]
# }
class ReplayMismatchError(ValueError):
    pass


# {
#   責務: [ ReplayOutput: 再生結果。stage_framesは退出した工程ごとの、その時点の合成画像 ]
# }
@dataclass(frozen=True)
class ReplayOutput:
    episode: DrawingEpisode
    image: RenderedImage
    rgba_sha256: str
    stage_frames: dict[str, RenderedImage] = field(default_factory=dict)


# {
#   責務: [ load_episode_document: episode.jsonまたはdraft.jsonを読み、(Episode, Draftの中断理由)を返す ]
#   処理: [ expected_sha256があればファイルのバイト列と照合する。draft.jsonは接頭辞Episodeを取り出す ]
#   エラー: [ OSError: 読めない, ReplayMismatchError: ハッシュ不一致・形式不正 ]
# }
def load_episode_document(
    path: str | Path, *, expected_sha256: str | None = None
) -> tuple[DrawingEpisode, str | None]:
    data = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ReplayMismatchError("Episodeファイルのハッシュが記録と一致しません")
    try:
        document = loads_strict(data.decode("utf-8"))
        if isinstance(document, dict) and document.get("schema_version") == DRAFT_SCHEMA:
            prefix = json.dumps(document.get("episode_prefix"), ensure_ascii=False)
            return loads_episode(prefix), str(document.get("reason"))
        return loads_episode(data), None
    except (DrawingError, UnicodeDecodeError, ValueError, RecursionError) as error:
        raise ReplayMismatchError(f"Episodeを検証できません: {error}") from error


# {
#   責務: [ load_episode_file: load_episode_documentのEpisodeだけを返す ]
# }
def load_episode_file(path: str | Path, *, expected_sha256: str | None = None) -> DrawingEpisode:
    return load_episode_document(path, expected_sha256=expected_sha256)[0]


# {
#   責務: [ replay_episode_file: 検証して全Actionを再生する。履歴を再生せずに画像を返さない ]
#   引数: [ expected_episode_sha256/expected_rgba_sha256: 記録されたハッシュ, frames: 工程退出時の画像も得る ]
#   エラー: [ ReplayMismatchError ]
# }
def replay_episode_file(
    path: str | Path,
    *,
    expected_episode_sha256: str | None = None,
    expected_rgba_sha256: str | None = None,
    frames: bool = False,
) -> ReplayOutput:
    episode = load_episode_file(path, expected_sha256=expected_episode_sha256)
    try:
        result = replay(episode)
        stage_frames = _stage_frames(episode) if frames else {}
    except DrawingError as error:
        raise ReplayMismatchError(f"再生に失敗しました: {error}") from error
    digest = hashlib.sha256(result.image.rgba).hexdigest()
    if expected_rgba_sha256 is not None and digest != expected_rgba_sha256:
        raise ReplayMismatchError("再生した画素が記録されたハッシュと一致しません")
    return ReplayOutput(episode, result.image, digest, stage_frames)


# {
#   責務: [ _stage_frames: exit_stageを含む適用単位の末尾で画像を取り、退出した工程名で返す ]
# }
def _stage_frames(episode: DrawingEpisode) -> dict[str, RenderedImage]:
    frames: dict[str, RenderedImage] = {}
    pending: list[str] = []
    for frame in iter_replay(episode):
        if frame.event.type == "exit_stage":
            pending.append(str(frame.event.payload["stage_id"]))
        if frame.is_group_end and pending:
            image = frame.capture().image
            frames.update({stage_id: image for stage_id in pending})
            pending = []
    return frames

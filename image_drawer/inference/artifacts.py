"""生成結果の保存。episode.json・draft.json・image.png・attempts.jsonを原子的に書く。"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_drawer.drawing import PALETTE_ID, default_catalog
from image_drawer.drawing.episode_io import dumps_episode
from image_drawer.drawing.stage.controller import AttemptRecord
from image_drawer.inference.png import encode_png_rgba
from image_drawer.inference.session import COMPLETE, DrawOutcome

EPISODE_FILE = "episode.json"
DRAFT_FILE = "draft.json"
IMAGE_FILE = "image.png"
ATTEMPTS_FILE = "attempts.json"
KNOWN_FILES = (EPISODE_FILE, DRAFT_FILE, IMAGE_FILE, ATTEMPTS_FILE)
DRAFT_SCHEMA = "idraw-draft/v1"
ATTEMPTS_SCHEMA = "idraw-attempts/v1"


# {
#   責務: [ OutputConflictError: 出力先が空でない（--overwriteなし）・ディレクトリでない ]
# }
class OutputConflictError(ValueError):
    pass


# {
#   責務: [ WrittenOutcome: 保存したファイルの場所とハッシュ。episode_pathとdraft_pathはどちらか一方 ]
# }
@dataclass(frozen=True)
class WrittenOutcome:
    out_dir: Path
    episode_path: Path | None
    draft_path: Path | None
    image_path: Path
    attempts_path: Path
    episode_sha256: str | None
    image_rgba_sha256: str
    png_sha256: str
    attempts_sha256: str


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(document: Any) -> bytes:
    text = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    return (text + "\n").encode("utf-8")


# {
#   責務: [ write_atomic: 一時ファイルへ書いてから改名する。途中の破損ファイルを目的の名前で残さない ]
# }
def write_atomic(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# {
#   責務: [ attempts_bytes: 拒否を含む全提案（試行番号・Action・受理か・理由・工程・seq）のJSON ]
# }
def attempts_bytes(attempts: tuple[AttemptRecord, ...]) -> bytes:
    rows = [
        {
            "attempt": record.attempt,
            "accepted": record.accepted,
            "reason": record.reason,
            "stage_id": record.stage_id,
            "seqs": list(record.seqs),
            "action": {
                "type": record.action.type,
                "payload": record.action.payload,
                "actor": record.action.actor,
                "time_ms": record.action.time_ms,
            },
        }
        for record in attempts
    ]
    return _json_bytes({"schema_version": ATTEMPTS_SCHEMA, "attempts": rows})


# {
#   責務: [ draft_bytes: finishを含まない接頭辞Episodeと中断理由のJSON。episode.jsonの名前は使わない ]
# }
def draft_bytes(outcome: DrawOutcome) -> bytes:
    last = outcome.episode.events[-1]
    document = {
        "schema_version": DRAFT_SCHEMA,
        "reason": outcome.reason,
        "stage_id": last.stage_id,
        "episode_prefix": outcome.episode.to_dict(),
    }
    return _json_bytes(document)


# {
#   責務: [ check_out_dir: 出力先を作らずに衝突だけ確認する。空でない既存先は--overwrite時だけ許す ]
#   エラー: [ OutputConflictError ]
# }
def check_out_dir(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and not out_dir.is_dir():
        raise OutputConflictError(f"出力先がディレクトリではありません: {out_dir}")
    if out_dir.is_dir() and any(out_dir.iterdir()) and not overwrite:
        raise OutputConflictError(
            f"出力先が空ではありません（--overwriteで既知のファイルだけ上書き）: {out_dir}"
        )


# {
#   責務: [ prepare_out_dir: 衝突を確認してから出力先を作る ]
# }
def prepare_out_dir(out_dir: Path, overwrite: bool) -> None:
    check_out_dir(out_dir, overwrite)
    out_dir.mkdir(parents=True, exist_ok=True)


# {
#   責務: [ write_outcome: 結果を保存する。completeはepisode.json、draftはdraft.jsonを書く ]
#   処理: [ 上書き時は、反対側の古い既知ファイル（draft時のepisode.json等）だけ削除する。他は触らない ]
# }
def write_outcome(
    outcome: DrawOutcome, out_dir: Path, *, overwrite: bool = False
) -> WrittenOutcome:
    prepare_out_dir(out_dir, overwrite)
    complete = outcome.status == COMPLETE
    main_name = EPISODE_FILE if complete else DRAFT_FILE
    main_bytes = (
        dumps_episode(outcome.episode).encode("utf-8") if complete else draft_bytes(outcome)
    )
    png = encode_png_rgba(outcome.image.width, outcome.image.height, outcome.image.rgba)
    attempts = attempts_bytes(outcome.attempts)
    if overwrite:
        (out_dir / (DRAFT_FILE if complete else EPISODE_FILE)).unlink(missing_ok=True)
    main_path = out_dir / main_name
    write_atomic(main_path, main_bytes)
    write_atomic(out_dir / IMAGE_FILE, png)
    write_atomic(out_dir / ATTEMPTS_FILE, attempts)
    return WrittenOutcome(
        out_dir=out_dir,
        episode_path=main_path if complete else None,
        draft_path=None if complete else main_path,
        image_path=out_dir / IMAGE_FILE,
        attempts_path=out_dir / ATTEMPTS_FILE,
        episode_sha256=sha256_hex(main_bytes) if complete else None,
        image_rgba_sha256=sha256_hex(outcome.image.rgba),
        png_sha256=sha256_hex(png),
        attempts_sha256=sha256_hex(attempts),
    )


# {
#   責務: [ execution_identity: 再現に効くPolicy・renderer・palette・brush manifestの識別子 ]
# }
def execution_identity(outcome: DrawOutcome) -> dict[str, Any]:
    policy = outcome.policy
    return {
        "policy_id": policy["policy_id"],
        "policy_version": policy["policy_version"],
        "checkpoint_sha256": policy["checkpoint_sha256"],
        "renderer_version": outcome.episode.renderer_version,
        "palette_id": PALETTE_ID,
        "brush_manifest_sha256": default_catalog().brush_sha256,
    }


# {
#   責務: [ episode_metadata: Episode Artifactのmetadata（件数・ハッシュ・Policy。Action本体は入れない） ]
# }
def episode_metadata(outcome: DrawOutcome, written: WrittenOutcome) -> dict[str, Any]:
    episode = outcome.episode
    return {
        "episode_id": episode.episode_id,
        "renderer_version": episode.renderer_version,
        "event_count": len(episode.events),
        "accepted_actions": outcome.accepted_actions,
        "rejected_attempts": outcome.rejected_attempts,
        "system_events": outcome.system_events,
        "episode_sha256": written.episode_sha256,
        "final_image_rgba_sha256": written.image_rgba_sha256,
        "attempts_sha256": written.attempts_sha256,
        "policy_id": outcome.policy["policy_id"],
        "checkpoint_sha256": outcome.policy["checkpoint_sha256"],
        "policy_trained": outcome.policy["policy_trained"],
        "stage_summary": outcome.stage_summary,
        "status": outcome.status,
        "execution_identity": execution_identity(outcome),
    }

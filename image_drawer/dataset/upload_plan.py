"""外部へ画像を送る前に、送信先・枚数・総バイト数を数える(--dry-run)。何も送らず、何も書かない。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from image_drawer.dataset.egress import read_dataset_external_upload
from image_drawer.dataset.extraction import _read_cache
from image_drawer.dataset.ingest import _files, compute_cache_key, prompt_builder_version
from image_drawer.dataset.normalize import hash_file, load_normalized
from image_drawer.dataset.png_encode import encode_png


# {
#   責務: [
#     plan_upload: 登録を実行した場合に外部へ送られる画像の枚数と総バイト数を、送信先ごとに数える
#   ]
#   処理: [
#     1: 元画像を重複なく読み、キャッシュがある画像は送らないものとして除く
#     2: 残りを正規化して、実際に送るPNGの大きさを数える(Datasetにも元画像にも書き込まない)
#     3: 外部へ送りうる抽出器の送信先ごとに、同じ枚数・バイト数を報告する
#   ]
#   引数: [
#     source_dir: 元画像のディレクトリ, dataset_dir: Datasetの保存先
#     tagger, captioner: 選んだ抽出器, allow_external_upload: 実行時の許可
#   ]
#   戻り値: [
#     plan: destinations(送信先ごとの枚数・バイト数)、images、bytes、permissions、errors
#   ]
# }
def plan_upload(
    source_dir: Path,
    dataset_dir: Path,
    tagger: Any,
    captioner: Any,
    *,
    allow_external_upload: bool,
) -> dict[str, Any]:
    destinations = [
        {**d, "extractor": ext.info.id}
        for ext in (tagger, captioner)
        for d in (ext.external_destinations() if hasattr(ext, "external_destinations") else [])
    ]
    images = total = 0
    errors: list[str] = []
    seen: set[str] = set()
    version = prompt_builder_version(captioner)
    for path in _files(source_dir) if destinations else []:
        try:
            sha256 = hash_file(path)
            if sha256 in seen:
                continue
            seen.add(sha256)
            key = compute_cache_key(sha256, tagger.info, captioner.info, version)
            if _read_cache(dataset_dir / "cache" / f"{key}.json") is not None:
                continue
            image = load_normalized(path)
            try:
                total += len(encode_png(image))
            finally:
                image.close()
            images += 1
        except (OSError, ValueError) as exc:
            errors.append(f"{path.name}: {type(exc).__name__}")
    for destination in destinations:
        destination.update(images=images, bytes=total)
    return {
        "dry_run": True,
        "images": images,
        "bytes": total,
        "destinations": destinations,
        "permissions": {
            "dataset_external_upload": read_dataset_external_upload(dataset_dir),
            "allow_external_upload_flag": allow_external_upload,
        },
        "errors": errors,
    }

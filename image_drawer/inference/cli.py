"""image-drawer-draw: 文章だけから描く、保存したEpisodeを再生する。画像や下書きは入力に取らない。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from image_drawer.inference.artifacts import (
    IMAGE_FILE,
    OutputConflictError,
    check_out_dir,
    prepare_out_dir,
    write_atomic,
    write_outcome,
)
from image_drawer.inference.png import PngError, decode_png_rgba, encode_png_rgba
from image_drawer.inference.policy import PolicyError, PolicyRegistry
from image_drawer.inference.replay import ReplayMismatchError, replay_episode_file
from image_drawer.inference.session import DEFAULT_SIDE, DrawRequest, generate_episode

EXIT_OK, EXIT_DRAFT, EXIT_UNRUNNABLE, EXIT_MISMATCH = 0, 1, 2, 3
UNTRAINED_NOTICE = "未学習またはテスト用のPolicyです。絵の品質は評価されていません。"


# {
#   責務: [ create_parser: 引数定義。画像・下書き・Episodeを生成の入力にする引数は持たない ]
# }
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-drawer-draw",
        description="文章だけから描く。--replayで保存したEpisodeを再生する。",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt", help="描く内容の文章")
    source.add_argument("--prompt-file", help="UTF-8のpromptファイル")
    source.add_argument("--replay", metavar="EPISODE", help="episode.jsonまたはdraft.jsonを再生")
    source.add_argument("--list-policies", action="store_true", help="登録済みPolicyを一覧")
    parser.add_argument("--policy", help="登録済みPolicyのID（描画時は必須）")
    parser.add_argument("--max-actions", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, help="既定: Policyのcanvas、無ければ256")
    parser.add_argument("--height", type=int, help="既定: Policyのcanvas、無ければ256")
    parser.add_argument("--out-dir", help="出力先。空でない既存ディレクトリはエラー")
    parser.add_argument("--overwrite", action="store_true", help="既知の出力ファイルだけ上書き")
    parser.add_argument("--frames", choices=("none", "stages"), default="none")
    parser.add_argument(
        "--expect-image", metavar="PNG", help="再生画素と一致するか確認（再生のみ）"
    )
    parser.add_argument(
        "--checkpoint", action="append", default=[], metavar="ID=SIDECAR_JSON",
        help="学習済みcheckpointをPolicyとして登録（--train-configが必要）",
    )  # fmt: skip
    parser.add_argument("--train-config", help="checkpointの学習設定JSON（TrainConfig.to_dict）")
    return parser


# {
#   責務: [ default_policies: 製品の既定のPolicy登録簿。numpyが無ければ空 ]
# }
def default_policies(checkpoints: Sequence[str], train_config: str | None) -> PolicyRegistry:
    registry = PolicyRegistry()
    try:
        from image_drawer.inference.training_policy import (
            register_checkpoint,
            register_random_legal,
        )
        from image_drawer.training import TrainConfig
    except ImportError:
        if checkpoints:
            raise PolicyError("checkpointの利用にはnumpy(extras train)が必要です") from None
        return registry
    register_random_legal(registry)
    if checkpoints:
        if not train_config:
            raise PolicyError("--checkpointには--train-configが必要です")
        config = TrainConfig.from_dict(json.loads(Path(train_config).read_text(encoding="utf-8")))
        for item in checkpoints:
            policy_id, separator, path = item.partition("=")
            if not separator or not policy_id or not path:
                raise PolicyError(f"--checkpointはID=PATHの形で指定してください: {item}")
            register_checkpoint(registry, policy_id, Path(path), config)
    return registry


def _emit(stream: Any, document: dict[str, Any]) -> None:
    print(json.dumps(document, sort_keys=True), file=stream)


def _fail(code: int, kind: str, message: str) -> int:
    _emit(sys.stderr, {"error": kind, "message": message})
    return code


# {
#   責務: [ _draw: promptから描いて保存する。完成は0、Draftは1、実行不能は2 ]
# }
def _draw(args: argparse.Namespace, registry: PolicyRegistry) -> int:
    if not args.policy:
        return _fail(EXIT_UNRUNNABLE, "usage", "--policyを指定してください")
    if not args.out_dir:
        return _fail(EXIT_UNRUNNABLE, "usage", "--out-dirを指定してください")
    prompt = args.prompt
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").rstrip("\r\n")
    loaded = registry.resolve(args.policy)
    side = loaded.canvas or (DEFAULT_SIDE, DEFAULT_SIDE)
    request = DrawRequest(
        prompt,
        args.policy,
        args.max_actions,
        args.seed,
        side[0] if args.width is None else args.width,
        side[1] if args.height is None else args.height,
    )
    out_dir = Path(args.out_dir)
    check_out_dir(out_dir, args.overwrite)
    outcome = generate_episode(request, registry)
    written = write_outcome(outcome, out_dir, overwrite=args.overwrite)
    document: dict[str, Any] = {
        "status": outcome.status,
        "image_path": str(written.image_path),
        "attempts_path": str(written.attempts_path),
        "accepted_actions": outcome.accepted_actions,
        "rejected_attempts": outcome.rejected_attempts,
        "policy_id": outcome.policy["policy_id"],
        "checkpoint_sha256": outcome.policy["checkpoint_sha256"],
        "policy_trained": outcome.policy["policy_trained"],
        "image_rgba_sha256": written.image_rgba_sha256,
    }
    if written.episode_path is not None:
        document["episode_path"] = str(written.episode_path)
    else:
        document["draft_path"] = str(written.draft_path)
        document["reason"] = outcome.reason
    if not outcome.policy["policy_trained"]:
        document["notice"] = UNTRAINED_NOTICE
    _emit(sys.stdout, document)
    return EXIT_OK if written.episode_path is not None else EXIT_DRAFT


# {
#   責務: [ _replay: 保存したEpisodeを検証して再生する。不一致・検証失敗は3 ]
# }
def _replay(args: argparse.Namespace) -> int:
    if not args.out_dir:
        return _fail(EXIT_UNRUNNABLE, "usage", "--out-dirを指定してください")
    out_dir = Path(args.out_dir)
    check_out_dir(out_dir, args.overwrite)
    output = replay_episode_file(args.replay, frames=args.frames == "stages")
    if args.expect_image:
        width, height, expected = decode_png_rgba(Path(args.expect_image).read_bytes())
        image = output.image
        if (width, height, expected) != (image.width, image.height, image.rgba):
            raise ReplayMismatchError("再生画素が--expect-imageと一致しません")
    prepare_out_dir(out_dir, args.overwrite)
    png = encode_png_rgba(output.image.width, output.image.height, output.image.rgba)
    write_atomic(out_dir / IMAGE_FILE, png)
    frames = []
    for stage_id, frame in output.stage_frames.items():
        path = out_dir / f"frame_{stage_id}.png"
        write_atomic(path, encode_png_rgba(frame.width, frame.height, frame.rgba))
        frames.append(str(path))
    _emit(
        sys.stdout,
        {
            "status": "replayed",
            "episode_id": output.episode.episode_id,
            "episode_status": output.episode.status,
            "image_path": str(out_dir / IMAGE_FILE),
            "image_rgba_sha256": output.rgba_sha256,
            "frames": frames,
        },
    )
    return EXIT_OK


# {
#   責務: [
#     main: 描画・再生・Policy一覧を実行する。policiesは登録簿の差し替え口（テスト・組み込み用）
#   ]
#   戻り値: [ 0=完成/一致, 1=Draft, 2=実行不能, 3=再生の検証失敗・不一致 ]
# }
def main(argv: Sequence[str] | None = None, *, policies: PolicyRegistry | None = None) -> int:
    args = create_parser().parse_args(argv)
    try:
        registry = policies or default_policies(args.checkpoint, args.train_config)
        if args.list_policies:
            _emit(sys.stdout, {"policies": registry.describe()})
            return EXIT_OK
        if args.replay:
            return _replay(args)
        return _draw(args, registry)
    except ReplayMismatchError as error:
        return _fail(EXIT_MISMATCH, "replay_mismatch", str(error))
    except (PolicyError, OutputConflictError, PngError) as error:
        return _fail(EXIT_UNRUNNABLE, type(error).__name__, str(error))
    except (OSError, ValueError) as error:
        return _fail(EXIT_UNRUNNABLE, type(error).__name__, str(error))


if __name__ == "__main__":
    sys.exit(main())

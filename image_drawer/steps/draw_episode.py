"""文章だけから描くDRAW_EPISODEと、保存したEpisodeを再生するREPLAY_EPISODE。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from image_drawer.core import Artifact
from image_drawer.inference.artifacts import (
    sha256_hex,
    episode_metadata,
    write_atomic,
    write_outcome,
)
from image_drawer.inference.png import encode_png_rgba
from image_drawer.inference.policy import PolicyRegistry
from image_drawer.inference.replay import ReplayMismatchError, replay_episode_file
from image_drawer.inference.session import DrawRequest, generate_episode
from image_drawer.steps.base import ParameterSpec, Step, StepContext, StepResult, StepSchema
from image_drawer.steps.registry import StepRegistry

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


# {
#   責務: [ DrawIncomplete: 描画がDraftで終わった。Episode Artifactは作らずdraft.jsonの場所を残す ]
# }
class DrawIncomplete(RuntimeError):
    def __init__(self, reason: str, draft_path: str) -> None:
        super().__init__(f"描画が完成しませんでした: {reason} (draft: {draft_path})")
        self.reason = reason
        self.draft_path = draft_path


def _step_dir(root: Path, context: StepContext) -> Path:
    return root / _UNSAFE.sub("_", context.run_id) / _UNSAFE.sub("_", context.step_id)


def _image_metadata(width: int, height: int, rgba_sha256: str, png_sha256: str) -> dict[str, Any]:
    return {
        "width": width,
        "height": height,
        "rgba_sha256": rgba_sha256,
        "png_sha256": png_sha256,
    }


# {
#   責務: [
#     DrawEpisodeStep: Text(prompt)だけを入力に、StageController経由で描く。主出力Episode、追加Artifact Image
#   ]
#   エラー: [ ValueError: 範囲外のパラメータ・Policy未登録・canvas不一致, DrawIncomplete: Draft ]
# }
class DrawEpisodeStep(Step):
    backend = "inference"
    schema = StepSchema(
        step_type="DRAW_EPISODE",
        input_types=("Text",),
        output_type="Episode",
        parameters={
            "policy": ParameterSpec(str, required=True),
            "max_actions": ParameterSpec(int, default=1000),
            "seed": ParameterSpec(int, default=0),
            "width": ParameterSpec(int, default=256),
            "height": ParameterSpec(int, default=256),
        },
        capabilities=frozenset({"draws", "writes_files"}),
    )

    def __init__(self, policies: PolicyRegistry, output_dir: str | Path) -> None:
        self.policies = policies
        self.output_dir = Path(output_dir)

    def run(self, inputs: list[Artifact], params: dict[str, Any], context: StepContext):
        prompt = inputs[0].metadata.get("value")
        if not isinstance(prompt, str):
            raise TypeError("DRAW_EPISODE expects a Text artifact string value")
        request = DrawRequest(
            prompt,
            params["policy"],
            params["max_actions"],
            params["seed"],
            params["width"],
            params["height"],
        )
        outcome = generate_episode(request, self.policies)
        written = write_outcome(outcome, _step_dir(self.output_dir, context))
        if written.draft_path is not None:
            raise DrawIncomplete(outcome.reason, str(written.draft_path.resolve()))
        policy = outcome.policy
        episode = context.make_artifact(
            "Episode",
            parent_artifact_ids=[inputs[0].id],
            uri=str(written.episode_path.resolve()),
            model=policy["policy_id"],
            version=policy["policy_version"],
            seed=request.seed,
            metadata=episode_metadata(outcome, written),
        )
        image = context.make_artifact(
            "Image",
            suffix="image",
            parent_artifact_ids=[episode.id],
            uri=str(written.image_path.resolve()),
            seed=request.seed,
            metadata={
                **_image_metadata(
                    outcome.image.width,
                    outcome.image.height,
                    written.image_rgba_sha256,
                    written.png_sha256,
                ),
                "episode_artifact_id": episode.id,
            },
        )
        return StepResult(output=episode, additional_artifacts=[image])


# {
#   責務: [
#     ReplayEpisodeStep: Episodeを検証して全Actionを再生し、Imageを返す。履歴を再生せずに画像を返さない
#   ]
#   エラー: [ ValueError: ファイルのハッシュ不一致・再生画素のハッシュ不一致・Episode不正 ]
# }
class ReplayEpisodeStep(Step):
    backend = "inference"
    schema = StepSchema(
        step_type="REPLAY_EPISODE",
        input_types=("Episode",),
        output_type="Image",
        capabilities=frozenset({"writes_files"}),
    )

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def run(self, inputs: list[Artifact], params: dict[str, Any], context: StepContext):
        source = inputs[0]
        if not source.uri:
            raise ValueError("Episode artifactにuriがありません")
        try:
            output = replay_episode_file(
                source.uri,
                expected_episode_sha256=source.metadata.get("episode_sha256"),
                expected_rgba_sha256=source.metadata.get("final_image_rgba_sha256"),
            )
        except ReplayMismatchError as error:
            raise ValueError(f"Replay不一致: {error}") from error
        png = encode_png_rgba(output.image.width, output.image.height, output.image.rgba)
        target = _step_dir(self.output_dir, context)
        target.mkdir(parents=True, exist_ok=True)
        write_atomic(target / "image.png", png)
        return context.make_artifact(
            "Image",
            parent_artifact_ids=[source.id],
            uri=str((target / "image.png").resolve()),
            metadata={
                **_image_metadata(
                    output.image.width, output.image.height, output.rgba_sha256, sha256_hex(png)
                ),
                "episode_artifact_id": source.id,
            },
        )


# {
#   責務: [ register_drawing_steps: DRAW_EPISODEとREPLAY_EPISODEを既存のregistryへ足す ]
#   エラー: [ ValueError: 同じ(step_type, backend)が登録済み ]
# }
def register_drawing_steps(
    registry: StepRegistry, *, policies: PolicyRegistry, output_dir: str | Path
) -> None:
    registry.register(DrawEpisodeStep(policies, output_dir), default=True)
    registry.register(ReplayEpisodeStep(output_dir), default=True)

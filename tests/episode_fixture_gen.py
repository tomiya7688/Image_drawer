"""S1c-4: 7工程fixtureの生成。スクリプト化した提案をStageControllerへ与え、受理ごとにEngine状態を記録する。

frames fixtureはReplayerではなくControllerのEngine状態から作る。これにより、
Replayerの出力を独立した記録と比較できる。
"""

from __future__ import annotations

import json
from pathlib import Path

from stage_helpers import (
    FULL_COVER_LEVEL,
    advance,
    engine_of,
    line_samples,
    make_controller,
    model,
    sample,
    setup_actions,
)

from image_drawer.drawing import DrawingAction, DrawingEpisode
from image_drawer.drawing.episode_io import dumps_episode
from image_drawer.drawing.stage.controller import StageController

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "drawing_episode_v2"
WIDTH, HEIGHT = 12, 8
REVIEW_PROMPT = "白黒の線画だけ、影なしで描いて"
FRAMES_SCHEMA = "drawing-replay-frames/v2"
NAMES = ("full_seven_stage", "review_only_stages", "draft_budget_exhausted")


# {
#   責務: [ Recorder: Controllerへ提案を与え、グループが確定するたびにEngine状態を記録する ]
#   フィールド: [ controller: 対象, frames: 記録した状態（after_seq・stage_id・合成・各レイヤー） ]
# }
class Recorder:
    def __init__(self, controller: StageController) -> None:
        self.controller = controller
        self.frames: list[dict] = []
        self._record()

    def _record(self) -> None:
        engine = engine_of(self.controller)
        state = engine.state
        self.frames.append(
            {
                "after_seq": len(state.history) - 1,
                "stage_id": state.stage_id,
                "rgba_hex": engine.render().rgba.hex(),
                "layers": [
                    {
                        "id": layer.id,
                        "role": layer.role,
                        "visible": layer.visible,
                        "rgba_hex": bytes(layer.pixels.data).hex(),
                    }  # fmt: skip
                    for layer in state.layers
                ],
            }
        )

    def accept(self, action: DrawingAction) -> None:
        result = self.controller.propose(action)
        assert result.accepted, result.reason
        self._record()

    def accept_all(self, actions: list[DrawingAction]) -> None:
        for action in actions:
            self.accept(action)

    def cover(self, layer_id: str, color: int) -> None:
        self.accept_all(setup_actions(layer_id, "ink_gpen", color))
        params = {"size_level_id": FULL_COVER_LEVEL, "opacity_level_id": 10}
        self.accept(model("set_brush_params", params))
        self.accept(model("stroke_v2", {"samples": [sample(WIDTH // 2, HEIGHT // 2)]}))

    def line(self, layer_id: str, color: int, start: tuple, end: tuple, brush: str = "ink_gpen"):
        self.accept_all(setup_actions(layer_id, brush, color))
        self.accept(model("stroke_v2", {"samples": line_samples(start, end)}))


# {
#   責務: [ _script_full: 7工程すべてで描く提案列を与える ]
# }
def _script_full(rec: Recorder) -> None:
    rec.line("sys_rough_001", 100, (2, 2), (9, 6))
    rec.accept(advance())
    rec.line("sys_sketch_001", 0, (2, 2), (9, 6))
    rec.accept(advance())
    rec.line("sys_final_drawing_001", 0, (1, 1), (10, 1))
    rec.line("sys_final_drawing_001", 0, (1, 5), (10, 5))
    rec.accept(advance())
    rec.cover("sys_base_color_001", 180)
    rec.accept(advance())
    rec.line("sys_shade_001", 5, (3, 3), (8, 3))
    layer = {"layer_id": "hl_001", "name": "ハイライト", "role": "highlight"}
    rec.accept(model("create_layer", layer))
    rec.line("hl_001", 215, (1, 5), (10, 5))
    rec.accept(advance())
    rec.accept(model("create_layer", {"layer_id": "fix_001", "name": "修正", "role": "correction"}))
    rec.line("fix_001", 36, (2, 7), (5, 7))
    rec.accept_all(setup_actions("sys_base_color_001", "eraser_hard", 0))
    rec.accept(model("set_brush_params", {"size_level_id": 3, "opacity_level_id": 10}))
    rec.accept(model("erase_v2", {"samples": line_samples((10, 7), (11, 7))}))
    rec.accept(advance())


# {
#   責務: [ _script_review: 基本色・陰影・修正を理由付きレビューで退出する提案列を与える ]
# }
def _script_review(rec: Recorder) -> None:
    rec.line("sys_rough_001", 100, (2, 2), (9, 6))
    rec.accept(advance())
    rec.line("sys_sketch_001", 0, (2, 2), (9, 6))
    rec.accept(advance())
    rec.line("sys_final_drawing_001", 0, (1, 1), (10, 1))
    rec.accept(advance())
    rec.accept(advance("monochrome_line_art_only"))
    rec.accept(advance("flat_style_no_shading"))
    rec.accept(advance("no_correction_needed"))


# {
#   責務: [ _script_draft: max_actions切れで閉じるまで、final_drawingの途中まで提案を与える ]
# }
def _script_draft(rec: Recorder) -> None:
    rec.line("sys_rough_001", 100, (2, 2), (9, 6))
    rec.accept(advance())
    rec.line("sys_sketch_001", 0, (2, 2), (9, 6))
    rec.accept(advance())
    rec.accept(model("select_layer", {"layer_id": "sys_final_drawing_001"}))
    rec.accept(model("select_brush", {"brush_id": "ink_gpen"}))


SCRIPTS = {
    "full_seven_stage": (_script_full, "猫", 1000),
    "review_only_stages": (_script_review, REVIEW_PROMPT, 1000),
    "draft_budget_exhausted": (_script_draft, "猫", 12),
}


# {
#   責務: [ build_fixture: 名前のスクリプトをControllerで実行し、Episodeと記録したframesを返す ]
# }
def build_fixture(name: str) -> tuple[DrawingEpisode, list[dict]]:
    script, prompt, max_actions = SCRIPTS[name]
    controller = make_controller(prompt, WIDTH, HEIGHT, max_actions)
    recorder = Recorder(controller)
    script(recorder)
    episode = controller.episode()
    episode.metadata = {"source": "scripted-fixture", "fixture": name}
    return episode, recorder.frames


# {
#   責務: [ frames_text: frames fixtureを1frame1行のJSON文書にする ]
# }
def frames_text(name: str, frames: list[dict]) -> str:
    head = {
        "schema_version": FRAMES_SCHEMA,
        "ref_episode": f"{name}.episode.json",
        "canvas": {"width": WIDTH, "height": HEIGHT},
    }
    dump = lambda value: json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )  # noqa: E731
    body = ",\n".join(dump(frame) for frame in frames)
    return f'{dump(head)[:-1]},"frames":[\n{body}\n]}}\n'


# {
#   責務: [ regenerate: 3つのfixtureをCRLF変換なしで書き直す ]
# }
def regenerate() -> None:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        episode, frames = build_fixture(name)
        (FIXTURE_DIR / f"{name}.episode.json").write_bytes(dumps_episode(episode).encode("utf-8"))
        (FIXTURE_DIR / f"{name}.frames.json").write_bytes(frames_text(name, frames).encode("utf-8"))
        print(f"wrote {name}: {len(episode.events)} events, {len(frames)} frames")


__all__ = ["FULL_COVER_LEVEL", "NAMES", "FIXTURE_DIR", "build_fixture", "frames_text", "regenerate"]

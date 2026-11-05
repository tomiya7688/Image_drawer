"""Replay画面の表示データ（viewmodel）。画面ライブラリをimportせず、状態遷移と表示内容を検査できる。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from image_drawer.drawing import DrawingEpisode, palette_rgb
from image_drawer.drawing.engine import RENDERER_V2, RenderedImage
from image_drawer.drawing.models import LAYER_ROLES_Z_ORDER, STAGE_IDS
from image_drawer.drawing.pixels import composite_on_white
from image_drawer.gui.replay_cursor import ReplayCursor
from image_drawer.inference.replay import ReplayMismatchError, load_episode_document

HIGH_ACCURACY_CLAIM = "high_accuracy_claim_allowed"
THUMBNAIL_SIDE = 32


# {
#   責務: [ StageItem: 工程バーの1工程。statusは unreached|in_progress|done|reviewed|interrupted ]
# }
@dataclass(frozen=True)
class StageItem:
    stage_id: str
    status: str
    review_reason: str | None


# {
#   責務: [ LayerRow: レイヤー一覧の1行。サムネイルは最大32pxのRGBA ]
# }
@dataclass(frozen=True)
class LayerRow:
    layer_id: str
    name: str
    role: str | None
    visible: bool
    has_pixels: bool
    thumbnail_size: tuple[int, int]
    thumbnail_rgba: bytes


# {
#   責務: [ BrushPanel: 現在のseq時点の選択状態の表示専用データ。凍結され、変更操作を持たない ]
# }
@dataclass(frozen=True)
class BrushPanel:
    color_id: int | None
    color_rgb: tuple[int, int, int] | None
    brush_id: str | None
    size_level_id: int | None
    opacity_level_id: int | None
    pressure_level_id: int | None


# {
#   責務: [ ActionRow / RejectedRow: Action履歴と拒否された提案の1行 ]
# }
@dataclass(frozen=True)
class ActionRow:
    seq: int
    actor: str
    type: str
    summary: str
    stage_id: str | None
    applied: bool


@dataclass(frozen=True)
class RejectedRow:
    attempt: int
    stage_id: str
    reason: str
    action_type: str


# {
#   責務: [ policy_badge: Policyの表示文言。「高精度」は評価報告書の参照つきで許可されたときだけ ]
# }
def policy_badge(
    trained: bool | None, *, claim_level: str | None = None, report_ref: str | None = None
) -> str:
    if trained is None:
        return "Policy不明（記録なし）"
    if not trained:
        return "未学習／テスト用"
    if claim_level == HIGH_ACCURACY_CLAIM and report_ref:
        return f"高精度（報告書: {report_ref}）"
    return "学習済み（評価は別途）"


def _summary(event) -> str:
    payload = event.payload
    if event.type in ("stroke_v2", "erase_v2"):
        return f"{len(payload.get('samples', []))}点"
    return ", ".join(f"{key}={value}" for key, value in payload.items())[:60]


# {
#   責務: [ build_stage_bar: seqまでのeventから7工程の進行状況とレビュー理由コードを出す ]
# }
def build_stage_bar(episode: DrawingEpisode, seq: int) -> list[StageItem]:
    entered: list[str] = []
    exited: set[str] = set()
    reasons: dict[str, str] = {}
    for event in episode.events:
        if event.seq > seq:
            break
        if event.type == "enter_stage":
            entered.append(str(event.payload["stage_id"]))
        elif event.type == "exit_stage":
            exited.add(str(event.payload["stage_id"]))
        elif event.type == "advance_stage" and event.payload.get("review_reason"):
            reasons[str(event.stage_id)] = str(event.payload["review_reason"])
    at_end = seq >= episode.events[-1].seq
    items = []
    for stage_id in STAGE_IDS:
        reason = reasons.get(stage_id)
        if stage_id not in entered:
            status = "unreached"
        elif stage_id in exited:
            status = "reviewed" if reason else "done"
        elif episode.status == "draft" and at_end and stage_id == entered[-1]:
            status = "interrupted"
        else:
            status = "in_progress"
        items.append(StageItem(stage_id, status, reason))
    return items


def _thumbnail(layer, width: int, height: int) -> tuple[tuple[int, int], bytes]:
    tw, th = min(width, THUMBNAIL_SIDE), min(height, THUMBNAIL_SIDE)
    data = layer.pixels.data
    out = bytearray()
    for ty in range(th):
        for tx in range(tw):
            offset = ((ty * height // th) * width + (tx * width // tw)) * 4
            out += data[offset : offset + 4]
    return (tw, th), bytes(out)


def _z_order(layers):
    if layers and layers[0].role is not None:
        return sorted(layers, key=lambda layer: LAYER_ROLES_Z_ORDER.index(layer.role))
    return list(layers)


# {
#   責務: [
#     ViewerModel: 読み込んだEpisodeを見るための表示データ。操作はカーソル移動と閲覧用の表示切替だけ
#   ]
#   フィールド: [ draft_reason: Draftの中断理由（完成ならNone）, policy_trained: 不明ならNone ]
# }
class ViewerModel:
    def __init__(
        self,
        episode: DrawingEpisode,
        *,
        draft_reason: str | None = None,
        policy_trained: bool | None = None,
        rejected: list[RejectedRow] | None = None,
    ) -> None:
        self.episode = episode
        self.cursor = ReplayCursor(episode)
        self.draft_reason = draft_reason
        self.policy_trained = policy_trained
        self.rejected = rejected or []
        self.hidden_view_layers: set[str] = set()

    @property
    def is_draft(self) -> bool:
        return self.episode.status != "complete" or self.draft_reason is not None

    @property
    def draft_badge(self) -> str | None:
        if not self.is_draft:
            return None
        reason = self.draft_reason or self.episode.termination_reason or "unknown"
        return f"未完成（{reason}）"

    @property
    def policy_badge(self) -> str:
        return policy_badge(self.policy_trained)

    # {
    #   責務: [ canvas: 現在位置の合成画像。閲覧用の非表示指定があれば反映する（Episodeは変えない） ]
    # }
    def canvas(self) -> RenderedImage:
        if not self.hidden_view_layers:
            return self.cursor.image()
        state = self.cursor.state
        layers = [layer for layer in state.layers if layer.visible]
        if state.profile == RENDERER_V2:
            layers = _z_order(layers)
        shown = [layer.pixels for layer in layers if layer.id not in self.hidden_view_layers]
        return RenderedImage(
            state.width, state.height, composite_on_white(shown, state.width, state.height)
        )

    def toggle_view_layer(self, layer_id: str) -> None:
        self.hidden_view_layers ^= {layer_id}

    def stage_bar(self) -> list[StageItem]:
        return build_stage_bar(self.episode, self.cursor.seq)

    # {
    #   責務: [ layer_panel: 前面から順のレイヤー行。順序はroleの固定z順（同roleは後に作った方が前） ]
    # }
    def layer_panel(self) -> list[LayerRow]:
        state = self.cursor.state
        rows = []
        for layer in reversed(_z_order(state.layers)):
            size, thumb = _thumbnail(layer, state.width, state.height)
            shown = layer.visible and layer.id not in self.hidden_view_layers
            rows.append(
                LayerRow(
                    layer.id,
                    layer.name,
                    layer.role,
                    shown,
                    any(layer.pixels.data[3::4]),
                    size,
                    thumb,
                )  # fmt: skip
            )
        return rows

    def brush_panel(self) -> BrushPanel:
        state = self.cursor.state
        color = state.selected_color_id
        stroke = self.cursor.last_stroke()
        samples = stroke.payload.get("samples") if stroke else None
        return BrushPanel(
            color,
            palette_rgb(color) if color is not None else None,
            state.selected_brush_id,
            state.size_level_id,
            state.opacity_level_id,
            samples[0].get("pressure_level_id") if samples else None,
        )

    def action_history(self) -> list[ActionRow]:
        seq = self.cursor.seq
        return [
            ActionRow(e.seq, e.actor, e.type, _summary(e), e.stage_id, e.seq <= seq)
            for e in self.episode.events
        ]


# {
#   責務: [ load_rejected: attempts.jsonから拒否された提案だけを読む ]
# }
def load_rejected(path: Path) -> list[RejectedRow]:
    rows = json.loads(path.read_text(encoding="utf-8")).get("attempts", [])
    return [
        RejectedRow(r["attempt"], r["stage_id"], r["reason"], r["action"]["type"])
        for r in rows
        if not r["accepted"]
    ]


@dataclass(frozen=True)
class ViewerEmpty:
    pass


@dataclass(frozen=True)
class ViewerLoaded:
    model: ViewerModel


@dataclass(frozen=True)
class ViewerLoadError:
    reason: str


ViewerState = ViewerEmpty | ViewerLoaded | ViewerLoadError


# {
#   責務: [ ViewerController: Empty -> Loaded/LoadError の遷移だけを行う ]
# }
class ViewerController:
    def __init__(self) -> None:
        self.state: ViewerState = ViewerEmpty()

    def load(self, path: str | Path, *, policy_trained: bool | None = None) -> ViewerState:
        source = Path(path)
        try:
            episode, reason = load_episode_document(source)
            attempts = source.with_name("attempts.json")
            rejected = load_rejected(attempts) if attempts.is_file() else []
            model = ViewerModel(
                episode, draft_reason=reason, policy_trained=policy_trained, rejected=rejected
            )
            model.cursor.seek_index(model.cursor.last_index)
        except (OSError, ValueError, KeyError, ReplayMismatchError) as error:
            self.state = ViewerLoadError(str(error))
        else:
            self.state = ViewerLoaded(model)
        return self.state

    def clear(self) -> None:
        self.state = ViewerEmpty()


__all__ = [
    "ActionRow", "BrushPanel", "LayerRow", "RejectedRow", "StageItem", "ViewerController",
    "ViewerEmpty", "ViewerLoadError", "ViewerLoaded", "ViewerModel", "build_stage_bar",
    "load_rejected", "policy_badge",
]  # fmt: skip

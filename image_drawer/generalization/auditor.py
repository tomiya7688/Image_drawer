"""保存されたDrawingEpisodeだけから、固定工程の遵守を独立に再計算する監査器(StageAuditor)。

Controllerの試行ログ・内部状態は読まない。Controllerが規則どおり受理していれば合格し、
規則を破ったEpisode(手で書き換えた、別の経路で作った)は不合格になる。
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass, field

from image_drawer.drawing import DrawingEpisode, DrawingError, Engine
from image_drawer.drawing.models import STAGE_IDS, DrawingAction
from image_drawer.drawing.stage.table import StageTable, default_stage_table

REQUIRED_STAGES = ("rough", "sketch", "final_drawing")
PAINTING_TYPES = ("stroke_v2", "erase_v2")
FORBIDDEN_TYPES = ("skip_stage", "set_phase")


# {
#   責務: [
#     AuditViolation: 監査で見つけた規則違反1件
#   ]
#   フィールド: [
#     code: 違反の種類(stage_event_missing、stage_order、finish、forbidden_action、
#       forbidden_layer_write、required_stage_without_effective_stroke、review_reason_not_allowed、
#       replay_error)
#     seq: 該当するeventのseq。特定できなければNone
#     detail: 利用者向けの説明
#   ]
# }
@dataclass(frozen=True, slots=True)
class AuditViolation:
    code: str
    seq: int | None
    detail: str


# {
#   責務: [
#     StrokeRecord: 描画event1件の、画素への効果と経路長(有効stroke率・ドット率の元データ)
#   ]
#   フィールド: [
#     seq, stage_id, action_type: 該当event
#     effective: 許可された書込レイヤーの画素を1つ以上変えたか
#     changed_pixels: 変わったRGBA画素数
#     path_length_px: サンプル間の距離の合計(画素単位)
#   ]
# }
@dataclass(frozen=True, slots=True)
class StrokeRecord:
    seq: int
    stage_id: str
    action_type: str
    effective: bool
    changed_pixels: int
    path_length_px: float


# {
#   責務: [
#     path_length_px: サンプル列の経路長を画素単位で求める
#   ]
# }
def path_length_px(samples: list[dict]) -> float:
    total = 0.0
    for first, second in zip(samples, samples[1:]):
        total += math.hypot(second["x_q8"] - first["x_q8"], second["y_q8"] - first["y_q8"])
    return total / 256.0


# {
#   責務: [
#     AuditResult: 監査の結果と、後続の指標が使う工程別の集計
#   ]
#   フィールド: [
#     violations: 規則違反。空なら合格
#     review_only_stages: レビューのみで通過した工程
#     effective_strokes_by_stage: 工程ごとの、画素を変えたstroke_v2の数
#     changed_pixels_by_stage: 工程ごとの、有効なstroke_v2が変えた画素数(RGBAの1画素単位)の合計
#     stroke_records: 描画event(stroke_v2・erase_v2)ごとの効果。指標の元データ
#   ]
# }
@dataclass(slots=True)
class AuditResult:
    violations: list[AuditViolation] = field(default_factory=list)
    review_only_stages: list[str] = field(default_factory=list)
    effective_strokes_by_stage: dict[str, int] = field(default_factory=dict)
    changed_pixels_by_stage: dict[str, int] = field(default_factory=dict)
    stroke_records: list[StrokeRecord] = field(default_factory=list)

    # {
    #   責務: [ passed: 規則違反が1件も無いか ]
    # }
    @property
    def passed(self) -> bool:
        return not self.violations


# {
#   責務: [
#     _check_stage_events: 7工程の入場・退出が順に1回ずつ、finishが1回で最後であることを確認する
#   ]
# }
def _check_stage_events(events: list[DrawingAction], violations: list[AuditViolation]) -> None:
    expected: list[tuple[str, str]] = []
    for stage_id in STAGE_IDS:
        expected += [("enter_stage", stage_id), ("exit_stage", stage_id)]
    seen = [
        (event.type, event.payload.get("stage_id"))
        for event in events
        if event.type in ("enter_stage", "exit_stage")
    ]
    if seen != expected:
        # 欠落・重複(余分)・順序違反を区別し、原因を特定しやすくする。
        missing = list((Counter(expected) - Counter(seen)).elements())
        extra = list((Counter(seen) - Counter(expected)).elements())
        if missing or extra:
            violations.append(
                AuditViolation(
                    "stage_event_missing",
                    None,
                    f"missing stage events: {missing}; extra or duplicated stage events: {extra}",
                )
            )
        else:
            violations.append(
                AuditViolation("stage_order", None, f"stage events are out of order: {seen}")
            )
    finishes = [event for event in events if event.type == "finish"]
    if len(finishes) != 1 or not events or events[-1].type != "finish":
        violations.append(
            AuditViolation(
                "finish", None, f"finish must appear exactly once and last (found {len(finishes)})"
            )
        )


# {
#   責務: [
#     _check_advance_events: レビュー退出の理由コードが、その工程で許可されたものか確認する
#   ]
#   戻り値: [
#     review_stages: レビューのみで通過した工程
#   ]
# }
def _check_advance_events(
    events: list[DrawingAction], table: StageTable, violations: list[AuditViolation]
) -> list[str]:
    review_stages: list[str] = []
    for event in events:
        if event.type != "advance_stage" or "review_reason" not in event.payload:
            continue
        reason = event.payload["review_reason"]
        stage_id = event.stage_id
        allowed = table.spec(stage_id).review_only_reason_codes if stage_id in STAGE_IDS else ()
        if reason not in allowed:
            violations.append(
                AuditViolation(
                    "review_reason_not_allowed",
                    event.seq,
                    f"{reason!r} is not allowed in {stage_id}",
                )
            )
        review_stages.append(stage_id)
    return review_stages


# {
#   責務: [
#     _check_forbidden_actions: skip_stageとv2でのset_phaseが無いことを確認する
#   ]
# }
def _check_forbidden_actions(events: list[DrawingAction], violations: list[AuditViolation]) -> None:
    for event in events:
        if event.type in FORBIDDEN_TYPES:
            violations.append(
                AuditViolation("forbidden_action", event.seq, f"{event.type} is not allowed in v2")
            )


# {
#   責務: [
#     _check_drawing: Engineで再適用し、書込先レイヤーと有効なstrokeを工程ごとに集計する
#   ]
#   処理: [
#     1: Engineをheaderから作り、eventを1件ずつ適用する。拒否されたらreplay_errorを記録して打ち切る
#     2: 描画eventでは、適用前後の選択レイヤーの画素を比べて、変わった画素数を数える
#     3: 選択レイヤーのroleがその工程の書込roleでなければ、forbidden_layer_writeとする
#     4: 必須工程に、専用roleへの有効なstroke_v2が1件も無ければ違反とする
#   ]
# }
def _check_drawing(episode: DrawingEpisode, table: StageTable, result: AuditResult) -> None:
    try:
        engine = Engine(
            episode.width, episode.height, profile=episode.renderer_version, seed=episode.seed
        )
    except DrawingError as exc:
        result.violations.append(
            AuditViolation("replay_error", None, f"header is not replayable: {exc}")
        )
        return
    strokes = {stage_id: 0 for stage_id in STAGE_IDS}
    changed = {stage_id: 0 for stage_id in STAGE_IDS}
    for event in episode.events:
        state = engine.state
        painting = event.type in PAINTING_TYPES
        layer = next((x for x in state.layers if x.id == state.active_layer_id), None)
        before = bytes(layer.pixels.data) if painting and layer is not None else None
        try:
            engine.apply(copy.deepcopy(event))
        except DrawingError as exc:
            result.violations.append(AuditViolation("replay_error", event.seq, str(exc)))
            return
        if before is None or event.stage_id not in STAGE_IDS:
            continue
        spec = table.spec(event.stage_id)
        if layer.role not in spec.write_layer_roles:
            result.violations.append(
                AuditViolation(
                    "forbidden_layer_write",
                    event.seq,
                    f"{event.type} wrote to {layer.id!r} (role={layer.role}) in {event.stage_id}",
                )
            )
        after = layer.pixels.data
        pixels = sum(before[i : i + 4] != after[i : i + 4] for i in range(0, len(after), 4))
        effective = pixels > 0 and layer.role in spec.write_layer_roles
        samples = event.payload.get("samples", [])
        result.stroke_records.append(
            StrokeRecord(
                event.seq, event.stage_id, event.type, effective, pixels, path_length_px(samples)
            )
        )
        if event.type == "stroke_v2" and effective:
            strokes[event.stage_id] += 1
            changed[event.stage_id] += pixels
    result.effective_strokes_by_stage = strokes
    result.changed_pixels_by_stage = changed
    for stage_id in REQUIRED_STAGES:
        if strokes[stage_id] == 0:
            result.violations.append(
                AuditViolation(
                    "required_stage_without_effective_stroke",
                    None,
                    f"{stage_id} has no effective stroke_v2 on its own layer role",
                )
            )


# {
#   責務: [
#     audit_episode: DrawingEpisodeだけから固定工程の遵守を再計算する
#   ]
#   処理: [
#     1: 7工程の入場・退出とfinishの順序を確認する
#     2: skip_stageとset_phaseが無いことを確認する
#     3: レビュー退出の理由コードを確認する
#     4: Engineで再適用して、書込先レイヤーと必須工程の有効strokeを確認する
#   ]
#   引数: [
#     episode: 保存されたEpisode。Controllerの試行ログは受け取らない
#     table: 工程表。省略すると既定の工程表
#   ]
#   戻り値: [
#     result: 違反と工程別の集計
#   ]
# }
def audit_episode(episode: DrawingEpisode, table: StageTable | None = None) -> AuditResult:
    table = table or default_stage_table()
    result = AuditResult()
    _check_stage_events(episode.events, result.violations)
    _check_forbidden_actions(episode.events, result.violations)
    result.review_only_stages = _check_advance_events(episode.events, table, result.violations)
    _check_drawing(episode, table, result)
    return result

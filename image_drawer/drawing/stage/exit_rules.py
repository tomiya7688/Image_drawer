"""advance_stage（工程退出）の審査。状態を読むだけで、変更しない。"""

from __future__ import annotations

from typing import Any

from ..models import DrawingState
from .intent import StyleIntent
from .progress import StageProgress, has_visible_paint
from .table import StageSpec

PAINT_REMAINS_STAGES = ("rough", "sketch", "final_drawing")
INTENT_FLAG_BY_REASON = {
    "monochrome_line_art_only": "monochrome_line_art_only",
    "flat_style_no_shading": "flat_style_no_shading",
}


# {
#   責務: [ check_advance: advance_stageのpayloadと退出条件を審査し、拒否理由を返す ]
#   処理: [
#     1: payloadが{}または{review_reason}か 2: 通常退出は有効筆跡数と残存画素 3: レビュー退出は
#     許可コード・描画済みでないこと・StyleIntentの裏付け
#   ]
#   引数: [ spec: 現在工程, state: 状態, progress: 工程内計数, intent: StyleIntent, payload: 提案のpayload ]
#   戻り値: [ reason: "<code>: <detail>"。受理可能ならNone ]
# }
def check_advance(
    spec: StageSpec,
    state: DrawingState,
    progress: StageProgress,
    intent: StyleIntent,
    payload: dict[str, Any],
) -> str | None:
    if payload and set(payload) != {"review_reason"}:
        return "invalid_payload: advance_stage accepts only {} or {review_reason}"
    if not payload:
        return _check_normal_exit(spec, state, progress)
    return _check_review_exit(spec, progress, intent, payload["review_reason"])


# {
#   責務: [ _check_normal_exit: 通常退出（筆跡あり）の条件を確認する ]
# }
def _check_normal_exit(spec: StageSpec, state: DrawingState, progress: StageProgress) -> str | None:
    # 必須描画のない工程（refine）は筆跡と消去のどちらでもよい。他は有効なstroke_v2が必要。
    refine_like = spec.min_effectful_strokes == 0
    done = progress.total if refine_like else progress.effective_strokes
    if done < max(1, spec.min_effectful_strokes):
        return f"stage_not_ready: no effectful stroke in {spec.id}"
    if spec.id in PAINT_REMAINS_STAGES and not has_visible_paint(state, spec.layer_roles):
        return f"stage_not_ready: no visible paint remains on {spec.id} layers"
    return None


# {
#   責務: [ _check_review_exit: レビュー退出の理由コード・描画済みでないこと・StyleIntentを確認する ]
# }
def _check_review_exit(
    spec: StageSpec, progress: StageProgress, intent: StyleIntent, reason: Any
) -> str | None:
    if type(reason) is not str or reason not in spec.review_only_reason_codes:
        return f"review_reason_not_allowed: {reason!r} is not allowed in {spec.id}"
    if progress.total > 0:
        return f"review_after_drawing: effective strokes/erases exist in {spec.id}"
    flag = INTENT_FLAG_BY_REASON.get(reason)
    if flag is not None and not getattr(intent, flag):
        return f"review_intent_not_confirmed: StyleIntent.{flag} is false"
    return None

"""推論ループ。PolicyのActionをStageController.proposeへ渡すことだけでキャンバスを変える。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Event
from typing import Any

from image_drawer.drawing import RENDERER_V2, DrawingEpisode, Engine
from image_drawer.drawing.engine import RenderedImage
from image_drawer.drawing.models import MAX_CANVAS_SIDE
from image_drawer.drawing.stage import (
    MAX_ACTIONS_LIMIT,
    Budget,
    ProposalResult,
    StageConfigError,
    StageController,
    judge_style_intent,
)
from image_drawer.drawing.stage.controller import AttemptRecord
from image_drawer.drawing.validation import DrawingError
from image_drawer.inference.policy import PolicyError, PolicyRegistry

DEFAULT_SIDE = 256
MAX_SEED = 0xFFFFFFFF
COMPLETE = "complete"
DRAFT = "draft"
EVENT_LIMIT_MARK = "event limit exceeded"
ProgressCallback = Callable[[ProposalResult, StageController], None]


# {
#   責務: [ DrawRequestError: 生成要求の値が範囲外・canvasがPolicyと不一致（実行不能） ]
# }
class DrawRequestError(PolicyError):
    pass


# {
#   責務: [ DrawRequest: 1回の生成要求。参照画像・下書きを持つフィールドは作らない ]
# }
@dataclass(frozen=True)
class DrawRequest:
    prompt: str
    policy: str
    max_actions: int = 1000
    seed: int = 0
    width: int = DEFAULT_SIDE
    height: int = DEFAULT_SIDE

    # {
    #   責務: [ validate: 型と範囲を検査する（boolは整数として受け付けない） ]
    #   エラー: [ DrawRequestError ]
    # }
    def validate(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise DrawRequestError("promptは空でない文字列にしてください")
        if not isinstance(self.policy, str) or not self.policy:
            raise DrawRequestError("policyを指定してください")
        limits = {
            "max_actions": (self.max_actions, 1, MAX_ACTIONS_LIMIT),
            "seed": (self.seed, 0, MAX_SEED),
            "width": (self.width, 1, MAX_CANVAS_SIDE),
            "height": (self.height, 1, MAX_CANVAS_SIDE),
        }
        for name, (value, low, high) in limits.items():
            if type(value) is not int or not low <= value <= high:
                raise DrawRequestError(f"{name}は{low}..{high}の整数にしてください: {value!r}")


# {
#   責務: [ DrawOutcome: 生成結果。completeは完成Episode、draftはfinishを含まない接頭辞 ]
#   フィールド: [
#     status: complete|draft, reason: finished または中断理由, episode: 受理済みイベントだけのEpisode,
#     image: その時点の合成画像, attempts: 拒否を含む全提案, policy: Policyの識別情報,
#     stage_summary: 工程ごとの受理Action数とレビュー理由コード
#   ]
# }
@dataclass(frozen=True)
class DrawOutcome:
    status: str
    reason: str
    episode: DrawingEpisode
    image: RenderedImage
    attempts: tuple[AttemptRecord, ...]
    policy: dict[str, Any]
    request: DrawRequest
    stage_summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def accepted_actions(self) -> int:
        return sum(1 for event in self.episode.events if event.actor == "model")

    @property
    def system_events(self) -> int:
        return sum(1 for event in self.episode.events if event.actor == "system")

    @property
    def rejected_attempts(self) -> int:
        return sum(1 for record in self.attempts if not record.accepted)


# {
#   責務: [ summarize_stages: 工程ごとの受理modelAction数とレビュー理由コードを集計する ]
# }
def summarize_stages(episode: DrawingEpisode) -> dict[str, dict[str, Any]]:
    counts: Counter[str] = Counter()
    reviews: dict[str, list[str]] = {}
    for event in episode.events:
        if event.actor != "model" or event.stage_id is None:
            continue
        counts[event.stage_id] += 1
        reason = event.payload.get("review_reason") if event.type == "advance_stage" else None
        if reason is not None:
            reviews.setdefault(event.stage_id, []).append(str(reason))
    return {
        stage: {"accepted_actions": counts[stage], "review_reasons": reviews.get(stage, [])}
        for stage in counts
    }


# {
#   責務: [ generate_episode: promptだけからEpisodeを生成する。Engine/DrawingStateは直接書き換えない ]
#   処理: [
#     1: 要求とPolicyのcanvasを検査 2: Engine・Controllerを作る
#     3: 閉じるまで next_action -> propose を繰り返す 4: 閉じた理由でcomplete/draftを決める
#   ]
#   引数: [ cancel: セットされたら次の提案前に止めてdraft(cancelled)にする, on_progress: 提案ごとの通知 ]
#   エラー: [ DrawRequestError, UnknownPolicyError: 実行不能（保存するものは無い） ]
# }
def generate_episode(
    request: DrawRequest,
    policies: PolicyRegistry,
    *,
    cancel: Event | None = None,
    on_progress: ProgressCallback | None = None,
) -> DrawOutcome:
    request.validate()
    loaded = policies.resolve(request.policy)
    if loaded.canvas is not None and loaded.canvas != (request.width, request.height):
        raise DrawRequestError(
            f"canvasがPolicyと違います: 指定={request.width}x{request.height} "
            f"Policy={loaded.canvas[0]}x{loaded.canvas[1]}"
        )
    engine = Engine(request.width, request.height, RENDERER_V2, seed=request.seed)
    budget = Budget(request.max_actions)
    try:
        controller = StageController(engine, judge_style_intent(request.prompt), budget)
    except StageConfigError as error:
        raise DrawRequestError(str(error)) from error

    reason: str | None = None
    try:
        session = loaded.open_session(request.prompt, request.seed)
    except Exception:
        reason, session = "policy_error", None
    while reason is None and session is not None and not controller.closed:
        if cancel is not None and cancel.is_set():
            reason = "cancelled"
            break
        try:
            action = session.next_action(controller, engine, budget)
        except Exception:
            reason = "policy_error"
            break
        try:
            result = controller.propose(action)
        except DrawingError as error:
            limited = EVENT_LIMIT_MARK in str(error)
            reason = "event_limit_exceeded" if limited else controller.termination_reason
            reason = reason or "engine_error"
            break
        if on_progress is not None:
            on_progress(result, controller)
        if not result.accepted and EVENT_LIMIT_MARK in (result.reason or ""):
            reason = "event_limit_exceeded"
    reason = reason or controller.termination_reason or "engine_error"
    episode = controller.episode()
    return DrawOutcome(
        status=COMPLETE if reason == "finished" else DRAFT,
        reason=reason,
        episode=episode,
        image=engine.render(),
        attempts=controller.attempts(),
        policy=loaded.describe(),
        request=request,
        stage_summary=summarize_stages(episode),
    )

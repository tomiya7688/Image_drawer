"""StageController: modelの提案Actionを審査し、固定7工程を原子的なグループで記録する。"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace

from ..engine import RENDERER_V2, Engine
from ..episode import DrawingEpisode
from ..models import MAX_LAYER_AREA, MAX_TIME_MS, DrawingAction, EventGroup
from ..validation import DrawingError, require_int
from .budget import Budget
from .envelope import check_envelope, snapshot_action
from .episode_build import build_episode
from .errors import StageClosedError, StageConfigError, StageInvariantError
from .groups import build_initial_group, build_transition_group, with_group
from .intent import StyleIntent
from .progress import StageProgress, pixels_changed, snapshot_pixels
from .rules import PAINTING_TYPES, check_proposal
from .table import default_stage_table

REQUIRED_LAYER_COUNT = 5  # rough・sketch・final_drawing・base_color・shade


# {
#   責務: [ ProposalResult: 1件の提案の結果。受理時は原子的グループの全イベントを返す ]
#   フィールド: [
#     accepted: 受理か, events: 受理時のイベント列（拒否時は空）,
#     reason: 拒否時の"<code>"または"<code>: <detail>"（受理時はNone）, attempt: 1始まりの通し番号
#   ]
# }
@dataclass(frozen=True)
class ProposalResult:
    accepted: bool
    events: list[DrawingAction]
    reason: str | None
    attempt: int


# {
#   責務: [ AttemptRecord: 受理・拒否を通した試行1件のログ ]
#   フィールド: [
#     attempt: 通し番号, action: 提案時点のコピー, accepted/reason: 結果,
#     stage_id: 審査した時点の工程, seqs: 受理時に割り当てたseq（拒否時は空）
#   ]
# }
@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    action: DrawingAction
    accepted: bool
    reason: str | None
    stage_id: str
    seqs: tuple[int, ...]


# {
#   責務: [
#     StageController: Engineの外側で固定7工程を強制する。目標画像は受け取らない
#   ]
#   フィールド: [
#     _engine/_table/_intent/_budget: 構成, _progress: 工程内の有効筆跡数, _events: 受理済みイベント,
#     _attempts: 試行ログ, _next_group_id: 次のグループ番号, _accepted_actions: 受理したmodelイベント数,
#     _termination: 閉じた理由（開いていればNone）
#   ]
#   処理: [
#     1: 構築時にgroup 0（rough入場）を記録 2: propose()が審査→実行→記録→閉じる判定を行う
#     3: 拒否はEngine・イベント・seq・group番号を変えず、試行ログにだけ残す
#   ]
# }
class StageController:
    def __init__(self, engine: Engine, intent: StyleIntent, budget: Budget) -> None:
        self._validate_arguments(engine, intent, budget)
        self._engine = engine
        self._table = default_stage_table()
        self._intent = intent
        self._budget = budget
        self._progress = StageProgress()
        self._events: list[DrawingAction] = []
        self._attempts: list[AttemptRecord] = []
        self._next_group_id = 0
        self._accepted_actions = 0
        self._termination: str | None = None
        initial = build_initial_group(self._table, 0)
        try:
            self._commit(initial)
        except DrawingError as exc:
            raise StageConfigError(f"cannot enter the initial stage: {exc}") from exc

    # {
    #   責務: [ _validate_arguments: 構築引数と画布が工程を完走できる条件を満たすか確認する ]
    #   エラー: [ StageConfigError: 型違い・v2でない・使用済み・必須5レイヤーが入らない画布 ]
    # }
    @staticmethod
    def _validate_arguments(engine: Engine, intent: StyleIntent, budget: Budget) -> None:
        if not isinstance(engine, Engine) or engine.state.profile != RENDERER_V2:
            raise StageConfigError(f"engine must be an unused {RENDERER_V2} Engine")
        if engine.state.history or engine.state.layers:
            raise StageConfigError("engine must not have any history or layers")
        if not isinstance(intent, StyleIntent) or not isinstance(budget, Budget):
            raise StageConfigError("intent must be a StyleIntent and budget a Budget")
        state = engine.state
        if state.width * state.height * REQUIRED_LAYER_COUNT > MAX_LAYER_AREA:
            raise StageConfigError("canvas is too large to hold the five mandatory stage layers")

    @property
    def stage_id(self) -> str:
        return self._engine.state.stage_id

    @property
    def finished(self) -> bool:
        return self._engine.state.finished

    @property
    def closed(self) -> bool:
        return self._termination is not None

    @property
    def termination_reason(self) -> str | None:
        return self._termination

    @property
    def accepted_action_count(self) -> int:
        return self._accepted_actions

    @property
    def attempt_count(self) -> int:
        return len(self._attempts)

    # {
    #   責務: [ attempts: 受理・拒否を通した試行ログを返す ]
    # }
    def attempts(self) -> tuple[AttemptRecord, ...]:
        return tuple(self._attempts)

    # {
    #   責務: [ episode: 受理済みイベントだけからDrawingEpisodeを作る（閉じる前ならdraft） ]
    #   引数: [ episode_id: 省略時はprompt・画布・seed・workflow_idから決まるID ]
    # }
    def episode(self, episode_id: str | None = None) -> DrawingEpisode:
        return build_episode(
            self._engine, self._events, self._intent, self._budget, self._table,
            self._termination, episode_id,
        )  # fmt: skip

    # {
    #   責務: [ propose: modelの提案Actionを1件審査し、受理なら原子的に記録する ]
    #   処理: [
    #     1: 閉じていれば例外 2: 試行番号を割り当て 3: 基本の正当性 4: payloadを複製
    #     5: 工程による権限と時刻 6: 実行と記録 7: 閉じる判定
    #   ]
    #   引数: [ action: modelのDrawingAction（seq・stage_id・groupは空） ]
    #   戻り値: [ result: 受理（イベント列）または拒否（理由） ]
    #   エラー: [ StageClosedError: 閉じている, StageInvariantError: 審査済みグループの適用失敗 ]
    # }
    def propose(self, action: DrawingAction) -> ProposalResult:
        if self.closed:
            raise StageClosedError(f"controller is closed: {self._termination}")
        attempt = len(self._attempts) + 1
        stage_id = self.stage_id
        snapshot = snapshot_action(action)
        reason = check_envelope(action)
        if reason is not None:
            return self._reject(attempt, snapshot, stage_id, reason)
        if snapshot.type != action.type:
            return self._reject(
                attempt, snapshot, stage_id, "invalid_envelope: proposal cannot be copied"
            )
        proposal = replace(snapshot, seq=None)
        reason = check_proposal(
            self._table, self._engine.state, self._progress, self._intent, proposal
        ) or self._check_time(proposal)
        if reason is not None:
            return self._reject(attempt, snapshot, stage_id, reason)
        if proposal.type == "advance_stage":
            return self._accept_advance(attempt, snapshot, proposal, stage_id)
        return self._accept_action(attempt, snapshot, proposal, stage_id)

    # {
    #   責務: [ _last_time_ms: 直前に記録したイベントの時刻 ]
    # }
    def _last_time_ms(self) -> int:
        return self._events[-1].time_ms

    # {
    #   責務: [ _check_time: 提案のtime_msがEngineの規則（範囲・非減少）を満たすか確認する ]
    # }
    def _check_time(self, proposal: DrawingAction) -> str | None:
        if proposal.time_ms is None:
            return None
        try:
            require_int(proposal.time_ms, "time_ms", 0, MAX_TIME_MS)
            if proposal.time_ms < self._last_time_ms():
                raise DrawingError("time_ms must be non-decreasing")
        except DrawingError as exc:
            return f"engine_rejected: {exc}"
        return None

    # {
    #   責務: [ _commit: 注釈済みイベントをEngineへ適用し、受理済みイベントへ追加する ]
    #   処理: [ 単独の通常Action（描画を含み得る）はapply、複数件のグループはapply_group（全件成功か全件不変） ]
    #   戻り値: [ applied: seq・time_msが確定したイベント列 ]
    # }
    def _commit(self, events: list[DrawingAction]) -> list[DrawingAction]:
        if len(events) == 1 and events[0].actor == "model":
            applied = [self._engine.apply(events[0])]
        else:
            applied = self._engine.apply_group(events)
        self._events.extend(applied)
        self._next_group_id += 1
        return applied

    # {
    #   責務: [ _accept_action: 通常Action1件をEngineへ適用して記録する。画素を変えた描画は計数する ]
    # }
    def _accept_action(
        self, attempt: int, snapshot: DrawingAction, proposal: DrawingAction, stage_id: str
    ) -> ProposalResult:
        state = self._engine.state
        painting = proposal.type in PAINTING_TYPES
        layer = next((item for item in state.layers if item.id == state.active_layer_id), None)
        before = snapshot_pixels(layer) if painting and layer is not None else None
        time_ms = proposal.time_ms if proposal.time_ms is not None else self._last_time_ms()
        event = replace(proposal, stage_id=stage_id, time_ms=time_ms)
        event = replace(event, group=EventGroup(self._next_group_id, 0, 1))
        try:
            applied = self._commit([event])
        except DrawingError as exc:
            return self._reject(attempt, snapshot, stage_id, f"engine_rejected: {exc}")
        if before is not None and pixels_changed(before, layer):
            self._progress.record(proposal.type)
        return self._accept(attempt, snapshot, stage_id, applied)

    # {
    #   責務: [ _accept_advance: 遷移（またはfinalize）グループを原子的に適用する ]
    #   エラー: [ StageInvariantError: 審査済みなのに適用が失敗した。Controllerを閉じる ]
    # }
    def _accept_advance(
        self, attempt: int, snapshot: DrawingAction, proposal: DrawingAction, stage_id: str
    ) -> ProposalResult:
        time_ms = proposal.time_ms if proposal.time_ms is not None else self._last_time_ms()
        events = build_transition_group(self._table, self._engine.state, proposal)
        events = with_group(events, self._next_group_id, time_ms)
        try:
            applied = self._commit(events)
        except Exception as exc:
            self._termination = "internal_error"
            reason = f"internal_error: {exc}"
            self._attempts.append(AttemptRecord(attempt, snapshot, False, reason, stage_id, ()))
            raise StageInvariantError(reason) from exc
        self._progress.reset()
        return self._accept(attempt, snapshot, stage_id, applied)

    # {
    #   責務: [ _accept: 受理を試行ログへ記録し、閉じる判定をして結果を返す ]
    # }
    def _accept(
        self, attempt: int, snapshot: DrawingAction, stage_id: str, applied: list[DrawingAction]
    ) -> ProposalResult:
        self._accepted_actions += sum(1 for event in applied if event.actor == "model")
        seqs = tuple(event.seq for event in applied)
        self._attempts.append(AttemptRecord(attempt, snapshot, True, None, stage_id, seqs))
        self._update_closed_state()
        return ProposalResult(True, list(applied), None, attempt)

    # {
    #   責務: [ _reject: 拒否を試行ログにだけ記録し、閉じる判定をして結果を返す（状態は変えない） ]
    # }
    def _reject(
        self, attempt: int, snapshot: DrawingAction, stage_id: str, reason: str
    ) -> ProposalResult:
        self._attempts.append(AttemptRecord(attempt, snapshot, False, reason, stage_id, ()))
        self._update_closed_state()
        return ProposalResult(False, [], reason, attempt)

    # {
    #   責務: [ _update_closed_state: finished → action → attemptの優先順で閉じる理由を決める ]
    # }
    def _update_closed_state(self) -> None:
        if self.finished:
            self._termination = "finished"
        elif self._accepted_actions >= self._budget.max_actions:
            self._termination = "action_budget_exhausted"
        elif len(self._attempts) >= self._budget.max_attempts:
            self._termination = "attempt_budget_exhausted"

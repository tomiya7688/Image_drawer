"""受理済みイベントだけからDrawingEpisodeを組み立てる。"""

from __future__ import annotations

import copy
from dataclasses import asdict

from ..catalog import canonical_sha256
from ..engine import RENDERER_V2, Engine
from ..episode import DrawingEpisode
from ..models import DrawingAction
from .budget import Budget
from .intent import StyleIntent
from .table import StageTable

IN_PROGRESS = "in_progress"


# {
#   責務: [ default_episode_id: prompt・画布・seed・workflow_idだけから決まる既定のEpisode ID ]
#   戻り値: [ episode_id: "ep-" + canonical JSONのSHA-256の先頭16桁 ]
# }
def default_episode_id(engine: Engine, intent: StyleIntent, table: StageTable) -> str:
    state = engine.state
    identity = {
        "prompt": intent.prompt,
        "canvas": {"width": state.width, "height": state.height},
        "seed": state.seed,
        "workflow_id": table.workflow_id,
    }
    return "ep-" + canonical_sha256(identity)[:16]


# {
#   責務: [ build_episode: 受理済みイベントの複製からDrawingEpisodeを作る ]
#   処理: [
#     1: 完了（finish済み）ならstatus=complete、それ以外はdraft 2: 閉じていなければin_progress
#     3: StyleIntent・Budget・workflow識別子をheader情報として付ける
#   ]
#   引数: [
#     engine: 画布寸法とseedの出所, events: 受理済みイベント, intent/budget/table: header情報,
#     termination: 閉じた理由（開いていればNone）, episode_id: 省略時は既定ID
#   ]
#   戻り値: [ episode: 呼び出し側の変更が波及しない独立したEpisode ]
# }
def build_episode(
    engine: Engine,
    events: list[DrawingAction],
    intent: StyleIntent,
    budget: Budget,
    table: StageTable,
    termination: str | None,
    episode_id: str | None,
) -> DrawingEpisode:
    state = engine.state
    intent_header = {key: value for key, value in asdict(intent).items() if key != "prompt"}
    return DrawingEpisode(
        episode_id=episode_id or default_episode_id(engine, intent, table),
        origin="mixed",
        prompt=intent.prompt,
        width=state.width,
        height=state.height,
        renderer_version=RENDERER_V2,
        events=copy.deepcopy(events),
        seed=state.seed,
        workflow_id=table.workflow_id,
        workflow_sha256=table.workflow_sha256,
        style_intent=intent_header,
        budget={"max_actions": budget.max_actions, "max_attempts": budget.max_attempts},
        status="complete" if termination == "finished" else "draft",
        termination_reason=termination if termination is not None else IN_PROGRESS,
    )

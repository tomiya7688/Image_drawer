"""推論（S4）のテストが共有する台本Policy。製品の登録簿には載せない（水準Aの配線確認用）。"""

from __future__ import annotations

from collections.abc import Sequence

from stage_helpers import STAGE_ORDER, advance, draw_actions

from image_drawer.drawing import DrawingAction
from image_drawer.inference import LoadedPolicy, PolicyRegistry

SIDE = 16


# {
#   責務: [ complete_actions: rough〜refineを描いてfinishまで進める台本 ]
# }
def complete_actions() -> list[DrawingAction]:
    actions: list[DrawingAction] = []
    for stage_id in STAGE_ORDER:
        actions += [*draw_actions(stage_id), advance()]
    return actions


# {
#   責務: [ ScriptedSession: 台本を順に返す。尽きたら最後を繰り返す ]
# }
class ScriptedSession:
    def __init__(self, actions: Sequence[DrawingAction], raise_at: int | None = None) -> None:
        self.actions = list(actions)
        self.raise_at = raise_at
        self.calls = 0

    def next_action(self, controller, engine, budget) -> DrawingAction:
        index = self.calls
        self.calls += 1
        if self.raise_at is not None and index >= self.raise_at:
            raise RuntimeError("scripted failure")
        return self.actions[min(index, len(self.actions) - 1)]


# {
#   責務: [ scripted_policy: 台本のLoadedPolicy。seedや画面に依らず同じ列を返す ]
# }
def scripted_policy(
    actions: Sequence[DrawingAction] | None = None,
    policy_id: str = "scripted",
    raise_at: int | None = None,
    canvas: tuple[int, int] | None = (SIDE, SIDE),
) -> LoadedPolicy:
    script = list(actions) if actions is not None else complete_actions()
    return LoadedPolicy(
        policy_id, "scripted-test", False, "", canvas,
        lambda prompt, seed: ScriptedSession(script, raise_at),
    )  # fmt: skip


def scripted_registry(**kwargs) -> PolicyRegistry:
    registry = PolicyRegistry()
    registry.register(scripted_policy(**kwargs))
    return registry

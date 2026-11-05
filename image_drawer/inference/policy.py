"""推論に使うPolicyの窓口。目標画像を受け取らず、Controllerと自分の状態だけを渡す。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from image_drawer.drawing import DrawingAction, Engine
from image_drawer.drawing.stage import Budget, StageController


# {
#   責務: [ PolicyError: Policyの登録・解決に関する失敗の基底 ]
# }
class PolicyError(ValueError):
    pass


# {
#   責務: [ UnknownPolicyError: 未登録のPolicy IDを指定した ]
# }
class UnknownPolicyError(PolicyError):
    pass


# {
#   責務: [ PolicySession: 1回の生成で使うPolicy。次のActionを返す ]
#   引数: [ controller/engine/budget: 自分の状態だけ。Datasetや目標は渡さない ]
# }
class PolicySession(Protocol):
    def next_action(
        self, controller: StageController, engine: Engine, budget: Budget
    ) -> DrawingAction: ...


# {
#   責務: [ LoadedPolicy: 登録済みPolicyの識別情報と、生成ごとにセッションを開く関数 ]
#   フィールド: [
#     policy_id: 登録ID, policy_version: 実装の版, trained: 学習済みか（未学習・テスト用はFalse）,
#     checkpoint_sha256: checkpointのSHA-256（無ければ空）, canvas: 学習時の(幅,高さ)。固定しなければNone,
#     open_session: (prompt, seed)から1回分のPolicySessionを作る
#   ]
# }
@dataclass(frozen=True)
class LoadedPolicy:
    policy_id: str
    policy_version: str
    trained: bool
    checkpoint_sha256: str
    canvas: tuple[int, int] | None
    open_session: Callable[[str, int], PolicySession]

    # {
    #   責務: [ describe: 一覧表示とEpisode metadataに使う識別情報（呼び出し関数は含めない） ]
    # }
    def describe(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "policy_trained": self.trained,
            "checkpoint_sha256": self.checkpoint_sha256,
            "canvas": list(self.canvas) if self.canvas else None,
        }


# {
#   責務: [ PolicyRegistry: Policy IDからLoadedPolicyを引く。同じIDの二重登録は拒否する ]
# }
class PolicyRegistry:
    def __init__(self) -> None:
        self._policies: dict[str, LoadedPolicy] = {}

    def register(self, loaded: LoadedPolicy) -> None:
        if not loaded.policy_id:
            raise PolicyError("policy_idは空にできません")
        if loaded.policy_id in self._policies:
            raise PolicyError(f"Policyは登録済みです: {loaded.policy_id}")
        self._policies[loaded.policy_id] = loaded

    def resolve(self, policy_id: str) -> LoadedPolicy:
        try:
            return self._policies[policy_id]
        except KeyError:
            known = ", ".join(sorted(self._policies)) or "(なし)"
            raise UnknownPolicyError(
                f"未登録のPolicyです: {policy_id}（登録済み: {known}）"
            ) from None

    def describe(self) -> list[dict[str, Any]]:
        return [self._policies[key].describe() for key in sorted(self._policies)]

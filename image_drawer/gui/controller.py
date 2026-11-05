"""生成(G2)とDataset登録(G3)の状態遷移。画面ライブラリをimportせず、CLIと同じサービス関数を呼ぶ。"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_drawer.inference.artifacts import OutputConflictError, check_out_dir, write_outcome
from image_drawer.inference.policy import PolicyError, PolicyRegistry
from image_drawer.inference.session import DrawOutcome, DrawRequest, generate_episode


@dataclass(frozen=True)
class GenerationIdle:
    pass


@dataclass(frozen=True)
class Generating:
    accepted_actions: int = 0
    stage_id: str | None = None


@dataclass(frozen=True)
class GenerationComplete:
    episode_path: Path
    policy_trained: bool


@dataclass(frozen=True)
class GenerationDraft:
    reason: str
    draft_path: Path
    policy_trained: bool


@dataclass(frozen=True)
class GenerationFailed:
    reason: str


GenerationState = (
    GenerationIdle | Generating | GenerationComplete | GenerationDraft | GenerationFailed
)
FAILURES = (PolicyError, OutputConflictError, OSError, ValueError)


# {
#   責務: [
#     GenerationController: Idle -> Generating -> Complete/Draft/Failed の遷移だけを行う。
#     生成は別スレッドでserviceを呼び、画面部品には触れない。結果はキュー経由でpoll()が取り込む
#   ]
#   フィールド: [ service: 生成関数（既定はCLI・Stepと同じgenerate_episode） ]
# }
class GenerationController:
    def __init__(
        self,
        policies: PolicyRegistry,
        *,
        service: Callable[..., DrawOutcome] = generate_episode,
    ) -> None:
        self.policies = policies
        self.service = service
        self.state: GenerationState = GenerationIdle()
        self._cancel = threading.Event()
        self._messages: queue.Queue[tuple[Any, ...]] = queue.Queue()
        self._thread: threading.Thread | None = None

    # {
    #   責務: [ start: 生成を開始する。Generating中は何もしない ]
    #   戻り値: [ started: 開始したか ]
    # }
    def start(self, request: DrawRequest, out_dir: Path, *, overwrite: bool = False) -> bool:
        if isinstance(self.state, Generating):
            return False
        self._cancel = threading.Event()
        self._messages = queue.Queue()
        self.state = Generating()
        self._thread = threading.Thread(
            target=self._work,
            args=(request, Path(out_dir), overwrite, self._cancel, self._messages),
            daemon=True,
        )
        self._thread.start()
        return True

    def _work(self, request, out_dir, overwrite, cancel, messages) -> None:
        def on_progress(result, controller) -> None:
            if result.accepted:
                messages.put(("progress", controller.accepted_action_count, controller.stage_id))

        try:
            check_out_dir(out_dir, overwrite)
            outcome = self.service(request, self.policies, cancel=cancel, on_progress=on_progress)
            written = write_outcome(outcome, out_dir, overwrite=overwrite)
        except FAILURES as error:
            messages.put(("failed", str(error)))
        else:
            messages.put(("done", outcome, written))

    def stop(self) -> None:
        if isinstance(self.state, Generating):
            self._cancel.set()

    def wait(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)

    # {
    #   責務: [ poll: ワーカーからの通知を取り込み、状態を更新する（画面スレッドから呼ぶ） ]
    # }
    def poll(self) -> GenerationState:
        while True:
            try:
                kind, *rest = self._messages.get_nowait()
            except queue.Empty:
                return self.state
            if not isinstance(self.state, Generating):
                continue
            if kind == "progress":
                self.state = Generating(rest[0], rest[1])
            elif kind == "failed":
                self.state = GenerationFailed(rest[0])
            else:
                outcome, written = rest
                trained = bool(outcome.policy["policy_trained"])
                if written.episode_path is not None:
                    self.state = GenerationComplete(written.episode_path, trained)
                else:
                    self.state = GenerationDraft(outcome.reason, written.draft_path, trained)

    # {
    #   責務: [ reset: Complete/Draft/FailedからIdleへ戻る（生成中は戻れない） ]
    # }
    def reset(self) -> None:
        if not isinstance(self.state, Generating):
            self.state = GenerationIdle()

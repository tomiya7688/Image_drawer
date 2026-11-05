"""S1b-6: 決定性（同じ提案列は同じ結果）。"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from pathlib import Path

from stage_helpers import engine_of, full_run, make_controller, pixels_hex

ROOT = Path(__file__).resolve().parents[1]
HASH_SCRIPT = """
import sys
sys.path.insert(0, "tests")
from stage_helpers import full_run, make_controller
from image_drawer.drawing.catalog import canonical_sha256

controller = make_controller("白黒の線画")
full_run(controller)
events = [
    [e.to_dict(), e.stage_id, [e.group.id, e.group.index, e.group.size]]
    for e in controller.episode().events
]
print(canonical_sha256(events))
"""


def test_same_proposals_give_identical_events_and_pixels():
    first, second = make_controller("猫", seed=7), make_controller("猫", seed=7)
    full_run(first)
    full_run(second)
    assert first.episode().events == second.episode().events
    assert pixels_hex(engine_of(first)) == pixels_hex(engine_of(second))


def test_controller_does_not_use_time_or_random(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("time/random must not be used")

    for module, names in (
        (time, ("time", "monotonic", "perf_counter", "time_ns")),
        (random, ("random", "randint", "choice", "shuffle", "getrandbits")),
        (os, ("urandom",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    controller = make_controller("猫")
    full_run(controller)
    assert controller.finished


def run_with_hash_seed(seed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    done = subprocess.run(
        [sys.executable, "-c", HASH_SCRIPT],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120, check=True,
    )  # fmt: skip
    return done.stdout.strip()


def test_controller_result_is_independent_of_pythonhashseed():
    digests = {run_with_hash_seed(seed) for seed in ("0", "1", "424242")}
    assert len(digests) == 1 and len(next(iter(digests))) == 64

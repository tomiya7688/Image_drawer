"""Deterministic evaluator used for tests and architecture baselines."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from image_drawer.core import Artifact, Score, ScoreSet
from image_drawer.evaluation.base import EvaluationIdentity


def _unit_hash(payload: object) -> float:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    return round(value / ((1 << 64) - 1), 6)


class DeterministicMockEvaluator:
    """Structured mock evaluator with reproducible raw components."""

    identity = EvaluationIdentity(name="mock", version="v1")

    def evaluate(
        self,
        artifacts: Sequence[Artifact],
        *,
        context: Mapping[str, Any] | None = None,
    ) -> ScoreSet:
        if not artifacts:
            raise ValueError("evaluator requires at least one candidate")

        prompt = (context or {}).get("prompt", "")
        scores: list[Score] = []
        for artifact in artifacts:
            explicit = artifact.metadata.get("mock_scores")
            if explicit is not None:
                if not isinstance(explicit, dict) or not explicit:
                    raise ValueError(
                        "mock_scores must be a non-empty component mapping"
                    )
                components = {}
                for name, value in explicit.items():
                    if not isinstance(name, str) or not name:
                        raise ValueError("mock score component names must be strings")
                    if type(value) not in (int, float):
                        raise TypeError(
                            f"mock score component {name} must be numeric"
                        )
                    components[name] = float(value)
            else:
                stable_artifact = artifact.to_dict()
                stable_artifact["scores"] = []
                components = {
                    "quality": _unit_hash(
                        {"artifact": stable_artifact, "component": "quality"}
                    ),
                    "prompt_alignment": _unit_hash(
                        {
                            "artifact_id": artifact.id,
                            "prompt": prompt,
                            "component": "prompt_alignment",
                        }
                    ),
                    "determinism": 1.0,
                }

            overall = sum(components.values()) / len(components)
            score_identity = {
                "evaluator": self.identity.to_dict(),
                "artifact_id": artifact.id,
                "components": components,
                "overall": overall,
                "prompt": prompt,
            }
            score_id = "score_" + hashlib.sha256(
                json.dumps(
                    score_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            scores.append(
                Score(
                    id=score_id,
                    evaluator=self.identity.name,
                    evaluator_version=self.identity.version,
                    overall=overall,
                    components=components,
                    metadata={"artifact_id": artifact.id},
                )
            )

        set_payload = {
            "evaluator": self.identity.to_dict(),
            "candidate_artifact_ids": [artifact.id for artifact in artifacts],
            "score_ids": [score.id for score in scores],
        }
        set_id = "scoreset_" + hashlib.sha256(
            json.dumps(
                set_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return ScoreSet(
            id=set_id,
            evaluator=self.identity.name,
            evaluator_version=self.identity.version,
            candidate_artifact_ids=[artifact.id for artifact in artifacts],
            scores=scores,
            metadata={"context": dict(context or {})},
        )

"""Real RETRIEVE_PARTS Step backed by the local Part Bank."""

from __future__ import annotations

from typing import Mapping

from image_drawer.part_bank.embedding import PartEmbedder
from image_drawer.part_bank.index import PartIndex
from image_drawer.part_bank.repository import SQLitePartRepository
from image_drawer.steps.base import ParameterSpec, Step, StepContext, StepSchema
from image_drawer.steps.mock import create_mock_registry
from image_drawer.steps.registry import StepRegistry


def _parse_filters(values: list[str]) -> dict[str, str]:
    filters: dict[str, str] = {}
    for value in values:
        if not isinstance(value, str) or "=" not in value:
            raise ValueError("filters must contain strings in key=value form")
        key, item = value.split("=", 1)
        key = key.strip()
        item = item.strip()
        if not key or not item:
            raise ValueError("filters must contain non-empty key=value pairs")
        if key in filters and filters[key] != item:
            raise ValueError(f"conflicting filter values for {key}")
        filters[key] = item
    return filters


class PartBankRetrievePartsStep(Step):
    """Search persisted Part embeddings without exposing index implementation."""

    backend = "part-bank"
    schema = StepSchema(
        step_type="RETRIEVE_PARTS",
        input_types=("Text",),
        output_type="PartSet",
        parameters={
            "category": ParameterSpec(str, default="generic"),
            "top": ParameterSpec(int, default=20),
            "embedding_model": ParameterSpec(str, default="default"),
            "filters": ParameterSpec(list, default=[]),
        },
        capabilities=frozenset({"retrieval", "local-index"}),
    )

    def __init__(
        self,
        repository: SQLitePartRepository,
        embedder: PartEmbedder,
        *,
        index: PartIndex | None = None,
    ) -> None:
        self.repository = repository
        self.embedder = embedder
        self.index = index

    def run(self, inputs, params, context: StepContext):
        prompt = inputs[0].metadata.get("value")
        if not isinstance(prompt, str):
            raise TypeError("RETRIEVE_PARTS expects Text artifact string value")

        requested_model = params["embedding_model"]
        if requested_model not in {"default", self.embedder.identity.key}:
            raise ValueError(
                "requested embedding_model is unavailable: "
                f"{requested_model}; available={self.embedder.identity.key}"
            )

        category = params["category"]
        active_category = None if category == "*" else category
        filters = _parse_filters(params["filters"])
        result = self.repository.query(
            prompt,
            embedder=self.embedder,
            category=active_category,
            top_k=params["top"],
            filters=filters,
            index=self.index,
        )
        identity = {
            "embedding": result.metadata["embedding"],
            "index": result.metadata["index"],
        }
        return context.make_artifact(
            "PartSet",
            parent_artifact_ids=[inputs[0].id],
            model=self.embedder.identity.model,
            version=self.embedder.identity.version,
            metadata={
                "value": list(result.part_ids),
                "part_set": result.to_dict(),
                "query_id": result.query_id,
                "category": result.category,
                "retrieval_scores": list(result.retrieval_scores),
                "execution_identity": identity,
            },
        )


def create_part_bank_registry(
    repository: SQLitePartRepository,
    embedder: PartEmbedder,
    *,
    index: PartIndex | None = None,
) -> StepRegistry:
    """Create the standard registry with real retrieval as the default backend."""
    registry = create_mock_registry()
    registry.register(
        PartBankRetrievePartsStep(repository, embedder, index=index),
        default=True,
    )
    return registry

"""Artifact lineage reconstruction helpers."""

from __future__ import annotations

from collections.abc import Iterable

from image_drawer.core.models import Artifact


def build_artifact_lineage(
    artifacts: Iterable[Artifact],
    artifact_id: str,
) -> dict[str, tuple[str, ...]]:
    """Return the reachable parent graph for one artifact.

    The returned mapping includes the requested artifact and every reachable
    ancestor. Missing referenced parents and cycles are treated as invalid
    provenance and reported explicitly.
    """
    by_id = {artifact.id: artifact for artifact in artifacts}
    if artifact_id not in by_id:
        raise KeyError(f"unknown artifact id: {artifact_id}")

    lineage: dict[str, tuple[str, ...]] = {}
    visiting: set[str] = set()

    def visit(current_id: str) -> None:
        if current_id in lineage:
            return
        if current_id in visiting:
            raise ValueError(f"artifact lineage contains a cycle at {current_id}")

        try:
            artifact = by_id[current_id]
        except KeyError as exc:
            raise KeyError(f"missing parent artifact: {current_id}") from exc

        visiting.add(current_id)
        parents = tuple(artifact.parent_artifact_ids)
        lineage[current_id] = parents
        for parent_id in parents:
            visit(parent_id)
        visiting.remove(current_id)

    visit(artifact_id)
    return lineage

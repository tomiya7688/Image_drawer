"""Artifact lineageを再構築するhelper。"""

from __future__ import annotations

from collections.abc import Iterable

from image_drawer.core.models import Artifact


def build_artifact_lineage(
    artifacts: Iterable[Artifact],
    artifact_id: str,
) -> dict[str, tuple[str, ...]]:
    """1個のArtifactから到達可能な親graphを返す。

    戻り値には指定Artifactと到達可能な全ancestorを含める。
    duplicate ID、参照先parent欠損、cycleは不正なprovenanceとして明示的に報告する。
    """
    by_id: dict[str, Artifact] = {}
    for artifact in artifacts:
        if artifact.id in by_id:
            raise ValueError(f"duplicate artifact id: {artifact.id}")
        by_id[artifact.id] = artifact

    if artifact_id not in by_id:
        raise KeyError(f"unknown artifact id: {artifact_id}")

    lineage: dict[str, tuple[str, ...]] = {}
    visiting: set[str] = set()

    def visit(current_id: str) -> None:
        if current_id in visiting:
            raise ValueError(f"artifact lineage contains a cycle at {current_id}")
        if current_id in lineage:
            return

        try:
            artifact = by_id[current_id]
        except KeyError as exc:
            raise KeyError(f"missing parent artifact: {current_id}") from exc

        visiting.add(current_id)
        parents = tuple(artifact.parent_artifact_ids)
        for parent_id in parents:
            visit(parent_id)
        visiting.remove(current_id)
        lineage[current_id] = parents

    visit(artifact_id)
    return lineage

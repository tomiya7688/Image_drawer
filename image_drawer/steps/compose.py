"""Real deterministic COMPOSE Step backed by Part Bank crops."""

from __future__ import annotations

import re
from pathlib import Path

from image_drawer.part_bank.rendering import (
    build_composition,
    fixed_grid_layout,
    render_composition,
    save_png_atomic,
)
from image_drawer.part_bank.repository import SQLitePartRepository
from image_drawer.steps.base import (
    ParameterSpec,
    Step,
    StepContext,
    StepResult,
    StepSchema,
)

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_component(value: str) -> str:
    cleaned = _SAFE_COMPONENT.sub("_", value).strip("._")
    return cleaned or "unnamed"


class PartBankComposeStep(Step):
    """Create a structured Composition and its deterministic PNG raster."""

    backend = "part-bank"
    schema = StepSchema(
        step_type="COMPOSE",
        input_types=("PartSet",),
        output_type="Image",
        parameters={
            "canvas_width": ParameterSpec(int, default=512),
            "canvas_height": ParameterSpec(int, default=512),
            "background": ParameterSpec(str, default="transparent"),
            "spacing": ParameterSpec(int, default=0),
            "layout_mode": ParameterSpec(
                str,
                default="fixed_grid",
                choices=("fixed_grid",),
            ),
        },
        capabilities=frozenset(
            {"composition", "deterministic", "structured-output"}
        ),
    )

    def __init__(
        self,
        repository: SQLitePartRepository,
        bank_dir: str | Path,
    ) -> None:
        self.repository = repository
        self.bank_dir = Path(bank_dir).resolve()

    def run(self, inputs, params, context: StepContext):
        part_ids = inputs[0].metadata.get("value")
        if not isinstance(part_ids, list) or not all(
            isinstance(part_id, str) for part_id in part_ids
        ):
            raise TypeError("COMPOSE expects PartSet metadata.value Part IDs")
        if not part_ids:
            raise ValueError("COMPOSE requires at least one selected Part")
        if len(set(part_ids)) != len(part_ids):
            raise ValueError("COMPOSE PartSet must not contain duplicate Part IDs")

        parts = [self.repository.get(part_id) for part_id in part_ids]
        if params["layout_mode"] != "fixed_grid":
            raise ValueError(f"unsupported layout_mode: {params['layout_mode']}")

        layout = fixed_grid_layout(
            parts,
            canvas_width=params["canvas_width"],
            canvas_height=params["canvas_height"],
            spacing=params["spacing"],
        )
        background = (
            None
            if params["background"] == "transparent"
            else params["background"]
        )
        composition = build_composition(
            parts,
            layout,
            background=background,
        )

        image = render_composition(
            composition,
            {part.id: part for part in parts},
            self.bank_dir,
        )
        relative_uri = (
            Path("runs")
            / _safe_component(context.run_id)
            / f"{_safe_component(context.step_id)}.png"
        ).as_posix()
        try:
            checksum = save_png_atomic(
                image,
                self.bank_dir / relative_uri,
            )
        finally:
            image.close()

        composition_artifact = context.make_artifact(
            "Composition",
            parent_artifact_ids=[inputs[0].id],
            model="structured-composition",
            version="v1",
            metadata={
                "composition": composition.to_dict(),
                "layout": layout.to_dict(),
                "selected_part_ids": part_ids,
                "lineage": {"selected_part_ids": part_ids},
            },
        )
        composition_artifact.id = f"{context.artifact_id()}:composition"

        image_artifact = context.make_artifact(
            "Image",
            parent_artifact_ids=[composition_artifact.id],
            uri=relative_uri,
            model="pillow-compose",
            version="v1",
            metadata={
                "checksum": checksum,
                "canvas_size": list(composition.canvas_size),
                "composition_artifact_id": composition_artifact.id,
                "selected_part_ids": part_ids,
                "lineage": {"selected_part_ids": part_ids},
                "execution_identity": {
                    "renderer": {
                        "backend": "pillow",
                        "version": "v1",
                    }
                },
            },
        )
        return StepResult(
            primary=image_artifact,
            additional_artifacts=[composition_artifact],
        )

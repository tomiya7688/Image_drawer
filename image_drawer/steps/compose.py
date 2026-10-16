"""Part Bankのcrop画像からCompositionとPNGを生成する決定的なCOMPOSE backend。"""

from __future__ import annotations

import hashlib
import io
import math
import os
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from image_drawer.core import Composition, Layout, Part, PartPlacement
from image_drawer.part_bank.repository import SQLitePartRepository
from image_drawer.steps.base import ParameterSpec, Step, StepContext, StepResult, StepSchema

_MAX_CANVAS_PIXELS = 16_000_000
_MAX_EDGE = 4096
_MAX_PARTS = 256


def _number(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


def _positive_integer(value: Any, name: str) -> int:
    if type(value) is not int or not (0 < value <= _MAX_EDGE):
        raise ValueError(f"{name} must be an integer between 1 and {_MAX_EDGE}")
    return value


def _background(value: Any) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(channel) is not int or not 0 <= channel <= 255 for channel in value)
    ):
        raise ValueError("background must be [red, green, blue, alpha] in 0..255")
    return tuple(value)


def _bank_file(bank_dir: Path, uri: str) -> Path:
    """外部Part metadata経由でbank外を読み込ませない。symlinkも確認する。"""
    if not isinstance(uri, str) or not uri or "\\" in uri:
        raise ValueError("Part media URI must be a bank-relative POSIX path")
    relative = PurePosixPath(uri)
    if relative.is_absolute() or ".." in relative.parts or ":" in relative.parts[0]:
        raise ValueError("Part media URI must stay within the Part Bank")
    path = (bank_dir / relative).resolve()
    if not path.is_relative_to(bank_dir):
        raise ValueError("Part media URI resolves outside the Part Bank")
    return path


def _manual_placements(slots: list[Any], parts: list[Part]) -> list[PartPlacement]:
    by_id = {part.id: part for part in parts}
    remaining = [part.id for part in parts]
    placements: list[PartPlacement] = []
    allowed = {
        "part_id", "category", "x", "y", "width", "height",
        "scale_x", "scale_y", "rotation", "z_index", "opacity",
    }
    for index, slot in enumerate(slots):
        if not isinstance(slot, dict):
            raise ValueError(f"slots[{index}] must be a JSON object")
        extra = set(slot) - allowed
        if extra:
            raise ValueError(f"slots[{index}] has unknown fields: {sorted(extra)}")
        category = slot.get("category")
        if category is not None and (not isinstance(category, str) or not category):
            raise ValueError(f"slots[{index}].category must be a non-empty string")
        part_id = slot.get("part_id")
        if part_id is None:
            part_id = next(
                (candidate for candidate in remaining
                 if category is None or by_id[candidate].category == category),
                None,
            )
            if part_id is None:
                raise ValueError(f"slots[{index}] has no matching unused Part")
        elif not isinstance(part_id, str) or part_id not in by_id:
            raise ValueError(f"slots[{index}].part_id is not in the selected PartSet")
        part = by_id[part_id]
        if category is not None and part.category != category:
            raise ValueError(f"slots[{index}] category does not match {part_id}")
        if part_id in remaining:
            remaining.remove(part_id)

        if "x" not in slot or "y" not in slot:
            raise ValueError(f"slots[{index}] requires x and y")
        x = _number(slot["x"], f"slots[{index}].x")
        y = _number(slot["y"], f"slots[{index}].y")
        width_specified = "width" in slot
        height_specified = "height" in slot
        if width_specified != height_specified:
            raise ValueError(f"slots[{index}] must specify width and height together")
        if width_specified and ("scale_x" in slot or "scale_y" in slot):
            raise ValueError(f"slots[{index}] cannot combine dimensions and scales")
        if width_specified:
            width = _number(slot["width"], f"slots[{index}].width", positive=True)
            height = _number(slot["height"], f"slots[{index}].height", positive=True)
            scale_x = width / part.bbox[2]
            scale_y = height / part.bbox[3]
        else:
            scale_x = _number(slot.get("scale_x", 1), f"slots[{index}].scale_x", positive=True)
            scale_y = _number(slot.get("scale_y", 1), f"slots[{index}].scale_y", positive=True)
        rotation = _number(slot.get("rotation", 0), f"slots[{index}].rotation")
        opacity = _number(slot.get("opacity", 1), f"slots[{index}].opacity")
        if not 0 <= opacity <= 1:
            raise ValueError(f"slots[{index}].opacity must be in 0..1")
        z_index = slot.get("z_index", index)
        if type(z_index) is not int:
            raise ValueError(f"slots[{index}].z_index must be an integer")

        placements.append(PartPlacement(
            id=f"placement-{index}", part_id=part_id, x=x, y=y,
            scale_x=scale_x, scale_y=scale_y, rotation=rotation,
            z_index=z_index, opacity=opacity,
            transform_metadata={"slot_index": index, "category": part.category},
        ))
    return placements


def _grid_placements(parts: list[Part], width: int, height: int) -> list[PartPlacement]:
    columns = math.ceil(math.sqrt(len(parts)))
    rows = math.ceil(len(parts) / columns)
    if width < columns or height < rows:
        raise ValueError("canvas is too small for a non-overlapping grid")
    placements: list[PartPlacement] = []
    for index, part in enumerate(parts):
        column, row = index % columns, index // columns
        left, right = width * column // columns, width * (column + 1) // columns
        top, bottom = height * row // rows, height * (row + 1) // rows
        part_width, part_height = part.bbox[2:]
        scale = min((right - left) / part_width, (bottom - top) / part_height)
        scaled_width = max(1, round(part_width * scale))
        scaled_height = max(1, round(part_height * scale))
        placements.append(PartPlacement(
            id=f"placement-{index}", part_id=part.id,
            x=left + (right - left - scaled_width) // 2,
            y=top + (bottom - top - scaled_height) // 2,
            scale_x=scaled_width / part_width,
            scale_y=scaled_height / part_height,
            z_index=index,
            transform_metadata={"slot_index": index, "category": part.category},
        ))
    return placements


def _render(
    bank_dir: Path,
    parts: dict[str, Part],
    composition: Composition,
) -> bytes:
    """同じcrop・mask・Compositionから同じPNG byte列を生成する。"""
    try:
        from PIL import Image, ImageChops
    except ImportError as exc:
        raise RuntimeError("COMPOSE requires the image-drawer[part-bank] extra") from exc

    width, height = composition.canvas_size
    with Image.new("RGBA", (width, height), tuple(composition.background)) as canvas:
        for placement in sorted(composition.placements, key=lambda item: item.z_index):
            part = parts[placement.part_id]
            with Image.open(_bank_file(bank_dir, part.crop_uri)) as source:
                source.load()
                if source.size != tuple(part.bbox[2:]):
                    raise ValueError(f"crop size does not match Part bbox: {part.id}")
                crop = source.convert("RGBA")
            target_size = (
                max(1, round(crop.width * placement.scale_x)),
                max(1, round(crop.height * placement.scale_y)),
            )
            if any(size > _MAX_EDGE for size in target_size):
                raise ValueError(f"transformed Part exceeds {_MAX_EDGE}px: {part.id}")
            with crop.resize(target_size, resample=Image.Resampling.BICUBIC) as layer:
                if part.mask_uri is not None:
                    with Image.open(_bank_file(bank_dir, part.mask_uri)) as raw_mask:
                        raw_mask.load()
                        if raw_mask.size != crop.size:
                            raise ValueError(f"mask size does not match crop: {part.id}")
                        if "A" in raw_mask.getbands() or "transparency" in raw_mask.info:
                            mask = raw_mask.convert("RGBA").getchannel("A")
                        else:
                            mask = raw_mask.convert("L")
                    with mask.resize(target_size, resample=Image.Resampling.BICUBIC) as resized_mask:
                        layer.putalpha(ImageChops.multiply(layer.getchannel("A"), resized_mask))
                if placement.opacity < 1:
                    opacity_byte = round(placement.opacity * 255)
                    layer.putalpha(layer.getchannel("A").point(
                        lambda alpha: (alpha * opacity_byte + 127) // 255
                    ))
                with layer.rotate(
                    placement.rotation, resample=Image.Resampling.BICUBIC, expand=True
                ) as transformed:
                    x, y = round(placement.x), round(placement.y)
                    left, top = max(x, 0), max(y, 0)
                    right = min(x + transformed.width, width)
                    bottom = min(y + transformed.height, height)
                    if left < right and top < bottom:
                        with transformed.crop((
                            left - x, top - y, right - x, bottom - y
                        )) as clipped:
                            canvas.alpha_composite(clipped, (left, top))
        buffer = io.BytesIO()
        canvas.save(buffer, format="PNG")
    return buffer.getvalue()


class PartBankComposeStep(Step):
    """選択済みPartSetを読み、独立Composition ArtifactとImage Artifactを返す。"""

    backend = "pillow"
    schema = StepSchema(
        step_type="COMPOSE",
        input_types=("PartSet",),
        output_type="Image",
        parameters={
            "canvas_width": ParameterSpec(int, default=512),
            "canvas_height": ParameterSpec(int, default=512),
            "background": ParameterSpec(list, default=[255, 255, 255, 255]),
            "slots": ParameterSpec(list, default=[]),
        },
        capabilities=frozenset({"composition", "deterministic", "raster", "layout"}),
    )

    def __init__(
        self,
        repository: SQLitePartRepository,
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        self.repository = repository
        self.bank_dir = repository.bank_dir
        self.output_dir = (
            Path(output_dir).resolve() if output_dir is not None
            else self.bank_dir / "runs" / "renders"
        )

    def run(self, inputs, params, context: StepContext) -> StepResult:
        width = _positive_integer(params["canvas_width"], "canvas_width")
        height = _positive_integer(params["canvas_height"], "canvas_height")
        if width * height > _MAX_CANVAS_PIXELS:
            raise ValueError(f"canvas cannot exceed {_MAX_CANVAS_PIXELS} pixels")
        background = _background(params["background"])
        slots = params["slots"]
        if not isinstance(slots, list):
            raise ValueError("slots must be a list")

        ids = inputs[0].metadata.get("value")
        if not isinstance(ids, list) or not ids or len(ids) > _MAX_PARTS:
            raise ValueError(f"COMPOSE expects 1..{_MAX_PARTS} selected Part IDs")
        if any(not isinstance(part_id, str) or not part_id for part_id in ids):
            raise ValueError("selected Part IDs must be non-empty strings")
        if len(set(ids)) != len(ids):
            raise ValueError("selected Part IDs must be unique")
        parts = [self.repository.get(part_id) for part_id in ids]
        if any(part.bbox[2] <= 0 or part.bbox[3] <= 0 for part in parts):
            raise ValueError("Part bbox dimensions must be positive")

        placements = (
            _manual_placements(slots, parts) if slots
            else _grid_placements(parts, width, height)
        )
        used_ids = list(dict.fromkeys(placement.part_id for placement in placements))
        layout = Layout(
            id=f"layout:{context.run_id}:{context.step_id}",
            canvas_width=width, canvas_height=height,
            slots=[
                {
                    "part_id": placement.part_id, "x": placement.x, "y": placement.y,
                    "scale_x": placement.scale_x, "scale_y": placement.scale_y,
                    "rotation": placement.rotation, "z_index": placement.z_index,
                    "opacity": placement.opacity,
                }
                for placement in placements
            ],
            metadata={"mode": "manual" if slots else "grid"},
        )
        composition = Composition(
            id=f"composition:{context.run_id}:{context.step_id}",
            canvas_size=(width, height), placements=placements,
            background=list(background),
            metadata={
                "layout_id": layout.id,
                "selected_part_ids": list(ids),
                "used_part_ids": used_ids,
                "unused_part_ids": [part_id for part_id in ids if part_id not in used_ids],
                "part_sources": {
                    part.id: {"source_image_id": part.source_image_id,
                              "crop_uri": part.crop_uri, "mask_uri": part.mask_uri,
                              "extraction_method": part.extraction_method,
                              "extraction_version": part.extraction_version}
                    for part in parts if part.id in used_ids
                },
            },
        )
        image_bytes = _render(self.bank_dir, {part.id: part for part in parts}, composition)
        checksum = hashlib.sha256(image_bytes).hexdigest()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        image_path = self.output_dir / f"{checksum}.png"
        if not image_path.exists():
            fd, temporary_path = tempfile.mkstemp(prefix=".compose-", dir=self.output_dir)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(image_bytes)
                os.replace(temporary_path, image_path)
            finally:
                Path(temporary_path).unlink(missing_ok=True)

        composition_artifact = context.make_artifact(
            "Composition", suffix="composition",
            parent_artifact_ids=[inputs[0].id],
            version="v1",
            metadata={"composition": composition.to_dict(), "layout": layout.to_dict(),
                      "part_ids": used_ids},
        )
        image_artifact = context.make_artifact(
            "Image", parent_artifact_ids=[composition_artifact.id],
            uri=image_path.as_uri(), version="v1",
            metadata={
                "composition_artifact_id": composition_artifact.id,
                "part_ids": used_ids,
                "width": width, "height": height, "format": "PNG",
                "checksum": f"sha256:{checksum}",
            },
        )
        return StepResult(output=image_artifact, additional_artifacts=[composition_artifact])

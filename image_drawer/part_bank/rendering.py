"""Deterministic Part placement and raster composition utilities."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from image_drawer.core import Composition, Layout, Part, PartPlacement


def _stable_id(prefix: str, payload: object) -> str:
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return f"{prefix}_{digest}"


def _positive_int(name: str, value: object) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def fixed_grid_layout(
    parts: Sequence[Part],
    *,
    canvas_width: int,
    canvas_height: int,
    spacing: int = 0,
) -> Layout:
    """Build a deterministic grid layout in Part order."""
    width = _positive_int("canvas_width", canvas_width)
    height = _positive_int("canvas_height", canvas_height)
    if type(spacing) is not int or spacing < 0:
        raise ValueError("spacing must be a non-negative integer")

    count = len(parts)
    slots: list[dict[str, Any]] = []
    if count:
        columns = math.ceil(math.sqrt(count))
        rows = math.ceil(count / columns)
        for index, part in enumerate(parts):
            row, column = divmod(index, columns)
            left = round(column * width / columns) + spacing
            right = round((column + 1) * width / columns) - spacing
            top = round(row * height / rows) + spacing
            bottom = round((row + 1) * height / rows) - spacing
            if right <= left or bottom <= top:
                raise ValueError("spacing leaves an empty fixed-grid slot")
            slots.append(
                {
                    "part_id": part.id,
                    "category": part.category,
                    "x": left,
                    "y": top,
                    "width": right - left,
                    "height": bottom - top,
                    "rotation": 0.0,
                    "z_index": index,
                    "opacity": 1.0,
                }
            )

    payload = {
        "canvas_width": width,
        "canvas_height": height,
        "slots": slots,
        "mode": "fixed_grid",
        "spacing": spacing,
    }
    return Layout(
        id=_stable_id("layout", payload),
        canvas_width=width,
        canvas_height=height,
        slots=slots,
        metadata={"mode": "fixed_grid", "spacing": spacing},
    )


def _slot_for_part(
    part: Part,
    index: int,
    slots: Sequence[dict[str, Any]],
    used: set[int],
) -> tuple[int, dict[str, Any]]:
    for slot_index, slot in enumerate(slots):
        if slot_index in used:
            continue
        if slot.get("part_id") == part.id:
            return slot_index, slot
    for slot_index, slot in enumerate(slots):
        if slot_index in used:
            continue
        category = slot.get("category")
        if category in (None, "*", part.category):
            return slot_index, slot
    if index < len(slots) and index not in used:
        return index, slots[index]
    raise ValueError(f"no layout slot available for Part {part.id}")


def placements_from_layout(
    parts: Sequence[Part],
    layout: Layout,
) -> list[PartPlacement]:
    """Resolve Parts into validated placements using layout slot metadata."""
    if layout.canvas_width <= 0 or layout.canvas_height <= 0:
        raise ValueError("layout canvas dimensions must be positive")
    if len(layout.slots) < len(parts):
        raise ValueError("layout has fewer slots than selected Parts")

    placements: list[PartPlacement] = []
    used: set[int] = set()
    for index, part in enumerate(parts):
        slot_index, slot = _slot_for_part(part, index, layout.slots, used)
        used.add(slot_index)

        slot_width = float(slot.get("width", 0))
        slot_height = float(slot.get("height", 0))
        if slot_width <= 0 or slot_height <= 0:
            raise ValueError("layout slot width/height must be positive")
        _, _, source_width, source_height = part.bbox
        if source_width <= 0 or source_height <= 0:
            raise ValueError(f"Part {part.id} has invalid bbox dimensions")

        contain_scale = min(
            slot_width / source_width,
            slot_height / source_height,
        )
        scale_x = float(slot.get("scale_x", contain_scale))
        scale_y = float(slot.get("scale_y", contain_scale))
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("PartPlacement scale must be positive")

        rendered_width = source_width * scale_x
        rendered_height = source_height * scale_y
        x = float(
            slot.get(
                "placement_x",
                float(slot.get("x", 0)) + (slot_width - rendered_width) / 2,
            )
        )
        y = float(
            slot.get(
                "placement_y",
                float(slot.get("y", 0)) + (slot_height - rendered_height) / 2,
            )
        )
        opacity = float(slot.get("opacity", 1.0))
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("PartPlacement opacity must be in [0, 1]")

        payload = {
            "layout_id": layout.id,
            "slot_index": slot_index,
            "part_id": part.id,
            "x": x,
            "y": y,
            "scale_x": scale_x,
            "scale_y": scale_y,
            "rotation": float(slot.get("rotation", 0.0)),
            "z_index": int(slot.get("z_index", index)),
            "opacity": opacity,
        }
        placements.append(
            PartPlacement(
                id=_stable_id("placement", payload),
                part_id=part.id,
                x=x,
                y=y,
                scale_x=scale_x,
                scale_y=scale_y,
                rotation=payload["rotation"],
                z_index=payload["z_index"],
                opacity=opacity,
                transform_metadata={
                    "layout_id": layout.id,
                    "slot_index": slot_index,
                },
            )
        )
    return placements


def build_composition(
    parts: Sequence[Part],
    layout: Layout,
    *,
    background: object = None,
) -> Composition:
    placements = placements_from_layout(parts, layout)
    payload = {
        "canvas_size": [layout.canvas_width, layout.canvas_height],
        "placements": [placement.to_dict() for placement in placements],
        "background": background,
    }
    return Composition(
        id=_stable_id("composition", payload),
        canvas_size=(layout.canvas_width, layout.canvas_height),
        placements=placements,
        background=background,
        metadata={
            "layout_id": layout.id,
            "selected_part_ids": [part.id for part in parts],
        },
    )


def _bank_path(bank_dir: Path, uri: str) -> Path:
    if not isinstance(uri, str) or not uri:
        raise ValueError("Part file URI must be a non-empty bank-relative path")
    root = bank_dir.resolve()
    path = (root / Path(uri)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Part URI escapes Part Bank: {uri}")
    return path


def _background_rgba(value: object) -> tuple[int, int, int, int]:
    from PIL import ImageColor

    if value is None or value == "transparent":
        return (0, 0, 0, 0)
    if isinstance(value, str):
        return ImageColor.getcolor(value, "RGBA")
    if isinstance(value, (list, tuple)) and len(value) in (3, 4):
        channels = [int(channel) for channel in value]
        if any(channel < 0 or channel > 255 for channel in channels):
            raise ValueError("background channels must be in [0, 255]")
        if len(channels) == 3:
            channels.append(255)
        return tuple(channels)  # type: ignore[return-value]
    raise ValueError("background must be transparent, a color string, or RGB/RGBA")


def _transform_part(
    part: Part,
    placement: PartPlacement,
    bank_dir: Path,
):
    try:
        from PIL import Image, ImageChops
    except ImportError as exc:
        raise RuntimeError(
            "install the image-drawer[part-bank] extra to render compositions"
        ) from exc

    crop_path = _bank_path(bank_dir, part.crop_uri)
    with Image.open(crop_path) as opened:
        opened.load()
        image = opened.convert("RGBA")

    if part.mask_uri:
        mask_path = _bank_path(bank_dir, part.mask_uri)
        with Image.open(mask_path) as opened_mask:
            opened_mask.load()
            mask = opened_mask.convert("L")
        if mask.size != image.size:
            mask.close()
            image.close()
            raise ValueError(
                f"mask size {mask.size} does not match crop size {image.size}"
            )
        alpha = image.getchannel("A")
        combined = ImageChops.multiply(alpha, mask)
        image.putalpha(combined)
        alpha.close()
        combined.close()
        mask.close()

    if placement.scale_x <= 0 or placement.scale_y <= 0:
        image.close()
        raise ValueError("PartPlacement scale must be positive")
    if not 0.0 <= placement.opacity <= 1.0:
        image.close()
        raise ValueError("PartPlacement opacity must be in [0, 1]")

    width = max(1, round(image.width * placement.scale_x))
    height = max(1, round(image.height * placement.scale_y))
    if (width, height) != image.size:
        resized = image.resize(
            (width, height),
            resample=Image.Resampling.LANCZOS,
        )
        image.close()
        image = resized

    if placement.rotation % 360:
        rotated = image.rotate(
            placement.rotation,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )
        image.close()
        image = rotated

    if placement.opacity < 1.0:
        alpha = image.getchannel("A")
        table = [round(value * placement.opacity) for value in range(256)]
        adjusted = alpha.point(table)
        image.putalpha(adjusted)
        alpha.close()
        adjusted.close()

    return image


def render_composition(
    composition: Composition,
    parts: Mapping[str, Part],
    bank_dir: str | Path,
):
    """Rasterize a Composition deterministically in z-order."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "install the image-drawer[part-bank] extra to render compositions"
        ) from exc

    width, height = composition.canvas_size
    _positive_int("composition canvas width", width)
    _positive_int("composition canvas height", height)
    canvas = Image.new("RGBA", (width, height), _background_rgba(composition.background))

    ordered = sorted(
        enumerate(composition.placements),
        key=lambda item: (item[1].z_index, item[0], item[1].id),
    )
    for _, placement in ordered:
        try:
            part = parts[placement.part_id]
        except KeyError as exc:
            canvas.close()
            raise KeyError(
                f"Composition references unknown Part: {placement.part_id}"
            ) from exc
        transformed = _transform_part(part, placement, Path(bank_dir))
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.paste(
            transformed,
            (round(placement.x), round(placement.y)),
        )
        transformed.close()
        composited = Image.alpha_composite(canvas, layer)
        layer.close()
        canvas.close()
        canvas = composited
    return canvas


def save_png_atomic(image, path: str | Path) -> str:
    """Write a deterministic PNG atomically and return its SHA-256."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        image.save(
            temporary,
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
        os.replace(temporary, target)
        return f"sha256:{digest}"
    finally:
        temporary.unlink(missing_ok=True)

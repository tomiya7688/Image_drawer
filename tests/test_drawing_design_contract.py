"""設計書の数値契約とゴールデンデータを独立検証するテスト。

製品用DrawingEngineの実装ではない。今後の実装は別途同じfixtureを読み、
同じ出力を返す必要がある。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "drawing_process_v1"
SCHEMA = (
    Path(__file__).parent.parent
    / "docs"
    / "developers"
    / "schemas"
    / "drawing_episode_v1.schema.json"
)


def palette(color_id: int) -> tuple[int, int, int]:
    """仕様どおりの固定256色を求める。"""
    if not 0 <= color_id <= 255 or type(color_id) is not int:
        raise ValueError("invalid color_id")
    if color_id < 216:
        return (
            51 * (color_id // 36),
            51 * ((color_id // 6) % 6),
            51 * (color_id % 6),
        )
    value = (255 * (color_id - 215) * 2 + 41) // 82
    return value, value, value


def round_half_up(numerator: int, denominator: int) -> int:
    return (2 * numerator + denominator) // (2 * denominator)


def over(source: list[int], destination: list[int]) -> list[int]:
    """RGBA8の整数source-over。"""
    source_alpha, dest_alpha = source[3], destination[3]
    denominator = 255 * source_alpha + dest_alpha * (255 - source_alpha)
    if denominator == 0:
        return [0, 0, 0, 0]
    output = [
        round_half_up(
            source[channel] * source_alpha * 255
            + destination[channel] * dest_alpha * (255 - source_alpha),
            denominator,
        )
        for channel in range(3)
    ]
    return output + [round_half_up(denominator, 255)]


def line_contains(
    point: tuple[int, int],
    start: list[int],
    end: list[int],
    width: int,
) -> bool:
    """浮動小数を使わない線分との距離比較。"""
    ux, uy = point[0] - start[0], point[1] - start[1]
    vx, vy = end[0] - start[0], end[1] - start[1]
    length = vx * vx + vy * vy
    dot = ux * vx + uy * vy
    if length == 0 or dot <= 0:
        numerator, denominator = ux * ux + uy * uy, 1
    elif dot >= length:
        rx, ry = point[0] - end[0], point[1] - end[1]
        numerator, denominator = rx * rx + ry * ry, 1
    else:
        numerator = (ux * vy - uy * vx) ** 2
        denominator = length
    return 4 * numerator <= width * width * denominator


def pixels_covered(points: list[list[int]], width: int, x: int, y: int) -> bool:
    segments = (
        list(zip(points, points[1:]))
        if len(points) > 1
        else [(points[0], points[0])]
    )
    return any(line_contains((x, y), a, b, width) for a, b in segments)


def replay(episode: dict) -> dict[int, dict]:
    """小さなfixtureだけを再生する、設計上の独立oracle。"""
    width, height = episode["canvas"]["width"], episode["canvas"]["height"]
    layers: list[dict] = []
    layer_by_id: dict[str, dict] = {}
    active = None
    color = None
    brush = None
    phase = None
    observed = {}
    previous_time = -1

    for seq, item in enumerate(episode["events"]):
        assert item["seq"] == seq
        assert item["time_ms"] >= previous_time
        assert item["actor"] in {"human", "model", "system", "synthetic", "derived"}
        previous_time = item["time_ms"]
        payload = item["payload"]
        action_type = item["type"]
        if action_type == "create_layer":
            assert payload["layer_id"] not in layer_by_id
            layer = {
                "id": payload["layer_id"],
                "rows": [
                    [[0, 0, 0, 0] for _ in range(width)]
                    for _ in range(height)
                ],
            }
            layers.append(layer)
            layer_by_id[layer["id"]] = layer
        elif action_type == "select_layer":
            active = layer_by_id[payload["layer_id"]]
        elif action_type == "set_phase":
            phase = payload["phase"]
        elif action_type == "select_brush":
            brush = payload["brush_id"]
        elif action_type == "select_color":
            color = payload["color_id"]
        elif action_type in {"stroke", "erase"}:
            assert active is not None and brush == "hard_round_v1" and phase
            if action_type == "stroke":
                assert color is not None
            for y in range(height):
                for x in range(width):
                    if pixels_covered(
                        payload["points"], payload["width_px"], x, y
                    ):
                        active["rows"][y][x] = (
                            [0, 0, 0, 0]
                            if action_type == "erase"
                            else over(
                                [*palette(color), payload["opacity_u8"]],
                                active["rows"][y][x],
                            )
                        )
        elif action_type == "finish":
            assert seq == len(episode["events"]) - 1
        else:
            raise AssertionError(action_type)

        image = []
        for y in range(height):
            row = []
            for x in range(width):
                pixel = [255, 255, 255, 255]
                for layer in layers:
                    pixel = over(layer["rows"][y][x], pixel)
                row.append(pixel)
            image.append(row)
        observed[seq] = {
            "rgba_rows": image,
            "layers": {
                layer["id"]: [
                    [pixel[:] for pixel in row]
                    for row in layer["rows"]
                ]
                for layer in layers
            },
        }
    return observed


def test_palette_is_fixed_and_nonoverlapping():
    colors = [palette(i) for i in range(256)]
    assert len(set(colors)) == 256
    assert colors[0] == (0, 0, 0)
    assert colors[180] == (255, 0, 0)
    assert colors[216] == (6, 6, 6)
    assert colors[255] == (249, 249, 249)


def test_schema_declares_all_actor_fields_and_budgets():
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert schema["properties"]["events"]["maxItems"] == 20000
    assert "mixed" in schema["properties"]["origin"]["enum"]
    for action in schema["properties"]["events"]["items"]["oneOf"]:
        assert "actor" in action["required"]
        assert action["properties"]["seq"]["maximum"] == 19999
        assert action["properties"]["time_ms"]["maximum"] == 604800000


def test_main_golden_episode_matches_every_checkpoint():
    episode = json.loads(
        (FIXTURES / "golden_episode.json").read_text(encoding="utf-8")
    )
    checkpoints = json.loads(
        (FIXTURES / "golden_frames.json").read_text(encoding="utf-8")
    )["checkpoints"]
    assert all(item["actor"] == "synthetic" for item in episode["events"])
    frames = replay(episode)
    for checkpoint in checkpoints:
        state = frames[checkpoint["after_seq"]]
        assert state["rgba_rows"] == checkpoint["rgba_rows"]
        for key, expected in checkpoint.get("layer_pixels", {}).items():
            layer_id, pair = key.split("@")
            x, y = (int(value) for value in pair.split(","))
            assert state["layers"][layer_id][y][x] == expected


def test_geometry_and_overlap_golden_cases():
    cases = json.loads(
        (FIXTURES / "golden_geometry.json").read_text(encoding="utf-8")
    )["cases"]
    for item in cases:
        frames = replay(item["episode"])
        for checkpoint in item["frames"]:
            state = frames[checkpoint["after_seq"]]
            assert state["rgba_rows"] == checkpoint["rgba_rows"]
            for layer in checkpoint["layer_rgba"]:
                assert state["layers"][layer["id"]] == layer["rows"]


def reject_decimal(value: str):
    raise ValueError(f"integer required, got {value}")


def reject_duplicate(pairs):
    data = {}
    for key, value in pairs:
        if key in data:
            raise ValueError(f"duplicate key: {key}")
        data[key] = value
    return data


@pytest.mark.parametrize(
    "payload",
    ['{"color_id": 5.0}', '{"color_id": 5e0}', '{"seq": 0, "seq": 1}'],
)
def test_strict_json_loader_rejects_ambiguous_numeric_and_duplicates(payload):
    with pytest.raises(ValueError):
        json.loads(payload, parse_float=reject_decimal, object_pairs_hook=reject_duplicate)

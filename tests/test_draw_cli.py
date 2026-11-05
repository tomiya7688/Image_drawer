"""S4-4: image-drawer-draw。水準A（配線）の確認で、モデルの品質は確認しない。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from inference_helpers import SIDE, scripted_registry

from image_drawer.inference.cli import create_parser, main
from image_drawer.inference.png import decode_png_rgba, encode_png_rgba


def invoke(capsys, *argv: str, registry=None) -> tuple[int, str, str]:
    code = main(list(argv), policies=registry or scripted_registry())
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def draw(capsys, out: Path, *extra: str, registry=None):
    return invoke(
        capsys, "--prompt", "猫", "--policy", "scripted", "--out-dir", str(out), *extra,
        registry=registry,
    )  # fmt: skip


def test_draw_cli_writes_episode_png_attempts_and_exit_0(tmp_path: Path, capsys) -> None:
    code, out, _err = draw(capsys, tmp_path / "o")
    assert code == 0
    document = json.loads(out)
    assert document["status"] == "complete" and document["rejected_attempts"] == 0
    for key in ("episode_path", "image_path", "attempts_path"):
        assert Path(document[key]).is_file()
    width, height, rgba = decode_png_rgba(Path(document["image_path"]).read_bytes())
    assert (width, height) == (SIDE, SIDE) and len(rgba) == SIDE * SIDE * 4


def test_draw_cli_returns_1_and_writes_draft_when_budget_exhausted(tmp_path: Path, capsys) -> None:
    code, out, _err = draw(capsys, tmp_path / "o", "--max-actions", "5")
    document = json.loads(out)
    assert code == 1 and document["status"] == "draft"
    assert document["reason"] == "action_budget_exhausted"
    assert Path(document["draft_path"]).name == "draft.json"
    assert Path(document["image_path"]).is_file() and "episode_path" not in document
    assert not (tmp_path / "o" / "episode.json").exists()


def test_draw_cli_returns_2_for_unknown_policy(tmp_path: Path, capsys) -> None:
    code, out, err = invoke(
        capsys, "--prompt", "猫", "--policy", "nope", "--out-dir", str(tmp_path / "o")
    )
    assert code == 2 and out == ""
    assert json.loads(err)["error"] == "UnknownPolicyError"
    assert not (tmp_path / "o").exists()


def test_draw_cli_returns_2_for_non_empty_out_dir_without_overwrite(tmp_path: Path, capsys) -> None:
    target = tmp_path / "o"
    target.mkdir()
    (target / "keep.txt").write_text("x", encoding="utf-8")
    code, _out, err = draw(capsys, target)
    assert code == 2 and json.loads(err)["error"] == "OutputConflictError"
    assert [p.name for p in target.iterdir()] == ["keep.txt"]
    code, _out, _err = draw(capsys, target, "--overwrite")
    assert code == 0 and (target / "keep.txt").exists() and (target / "episode.json").exists()


def test_draw_cli_returns_2_when_canvas_differs_from_policy_canvas(tmp_path: Path, capsys) -> None:
    code, _out, err = draw(capsys, tmp_path / "o", "--width", "32", "--height", "32")
    assert code == 2 and json.loads(err)["error"] == "DrawRequestError"
    assert not (tmp_path / "o").exists()


def test_draw_cli_replay_reproduces_image_and_returns_0(tmp_path: Path, capsys) -> None:
    _code, out, _err = draw(capsys, tmp_path / "o")
    drawn = json.loads(out)
    code, out, _err = invoke(
        capsys, "--replay", drawn["episode_path"], "--out-dir", str(tmp_path / "r"),
        "--frames", "stages", "--expect-image", drawn["image_path"],
    )  # fmt: skip
    replayed = json.loads(out)
    assert code == 0 and replayed["image_rgba_sha256"] == drawn["image_rgba_sha256"]
    assert (tmp_path / "r" / "image.png").read_bytes() == Path(drawn["image_path"]).read_bytes()
    names = {Path(p).name for p in replayed["frames"]}
    assert names == {
        f"frame_{s}.png"
        for s in ("rough", "sketch", "final_drawing", "base_color", "shading", "refine")
    }
    assert all(Path(p).is_file() for p in replayed["frames"])


def test_draw_cli_replay_returns_3_for_tampered_episode(tmp_path: Path, capsys) -> None:
    _code, out, _err = draw(capsys, tmp_path / "o")
    path = Path(json.loads(out)["episode_path"])
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["events"][10]
    path.write_text(json.dumps(document), encoding="utf-8")
    code, out, err = invoke(capsys, "--replay", str(path), "--out-dir", str(tmp_path / "r"))
    assert code == 3 and out == "" and json.loads(err)["error"] == "replay_mismatch"
    assert not (tmp_path / "r").exists()
    missing = invoke(
        capsys, "--replay", str(tmp_path / "none.json"), "--out-dir", str(tmp_path / "r")
    )
    assert missing[0] == 2


def test_draw_cli_replay_returns_3_when_expected_image_differs(tmp_path: Path, capsys) -> None:
    _code, out, _err = draw(capsys, tmp_path / "o")
    drawn = json.loads(out)
    _width, _height, rgba = decode_png_rgba(Path(drawn["image_path"]).read_bytes())
    changed = bytearray(rgba)
    changed[0] ^= 0xFF
    other = tmp_path / "other.png"
    other.write_bytes(encode_png_rgba(SIDE, SIDE, bytes(changed)))
    code, _out, err = invoke(
        capsys, "--replay", drawn["episode_path"], "--out-dir", str(tmp_path / "r"),
        "--expect-image", str(other),
    )  # fmt: skip
    assert code == 3 and "expect-image" in json.loads(err)["message"]


def test_draw_cli_replays_draft_json(tmp_path: Path, capsys) -> None:
    _code, out, _err = draw(capsys, tmp_path / "o", "--max-actions", "5")
    drawn = json.loads(out)
    code, out, _err = invoke(
        capsys, "--replay", drawn["draft_path"], "--out-dir", str(tmp_path / "r"),
        "--expect-image", drawn["image_path"],
    )  # fmt: skip
    assert code == 0 and json.loads(out)["episode_status"] == "draft"


def test_draw_cli_has_no_image_input_option() -> None:
    options = {
        option
        for action in create_parser()._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    assert options == {
        "--help", "--prompt", "--prompt-file", "--replay", "--list-policies", "--policy",
        "--max-actions", "--seed", "--width", "--height", "--out-dir", "--overwrite",
        "--frames", "--expect-image", "--checkpoint", "--train-config",
    }  # fmt: skip
    assert not [o for o in options if any(w in o for w in ("image", "sketch", "draft", "ref"))
                and o != "--expect-image"]  # fmt: skip


def test_draw_cli_stdout_is_single_json_line_and_marks_untrained_policy(
    tmp_path: Path, capsys
) -> None:
    _code, out, _err = draw(capsys, tmp_path / "o")
    assert out.count("\n") == 1 and out.endswith("\n")
    document = json.loads(out)
    assert document["policy_trained"] is False and "品質は評価されていません" in document["notice"]
    code, out, _err = invoke(capsys, "--list-policies")
    listed = json.loads(out)["policies"]
    assert code == 0 and listed[0]["policy_id"] == "scripted" and listed[0]["canvas"] == [16, 16]


def test_draw_cli_prompt_file_and_default_registry(tmp_path: Path, capsys) -> None:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("赤い四角\n", encoding="utf-8")
    code, out, _err = invoke(
        capsys,
        "--prompt-file",
        str(prompt),
        "--policy",
        "scripted",
        "--out-dir",
        str(tmp_path / "o"),
    )
    assert code == 0
    episode = json.loads(Path(json.loads(out)["episode_path"]).read_text(encoding="utf-8"))
    assert episode["prompt"] == "赤い四角"
    code = main(["--list-policies"])
    listed = json.loads(capsys.readouterr().out)["policies"]
    assert code == 0 and [p["policy_id"] for p in listed] in (["random-legal-v1"], [])


def test_draw_cli_and_step_produce_identical_episode_for_same_request(
    tmp_path: Path, capsys
) -> None:
    from image_drawer.dsl import parse_workflow
    from image_drawer.runtime import WorkflowRuntime
    from image_drawer.steps import create_mock_registry, register_drawing_steps

    _code, out, _err = draw(capsys, tmp_path / "cli", "--seed", "7")
    registry = create_mock_registry()
    register_drawing_steps(registry, policies=scripted_registry(), output_dir=tmp_path / "step")
    source = (
        'INPUT prompt: Text\nepisode = DRAW_EPISODE(prompt, policy="scripted", seed=7, '
        "width=16, height=16)\nOUTPUT episode\n"
    )
    result = WorkflowRuntime(registry).execute(
        parse_workflow(source, registry), external_inputs={"prompt": "猫"}, run_id="run-1"
    )
    step_episode = next(a for a in result.trajectory.artifacts if a.artifact_type == "Episode")
    cli_path = Path(json.loads(out)["episode_path"])
    assert Path(step_episode.uri).read_bytes() == cli_path.read_bytes()

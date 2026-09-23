"""Prompt-set loading utilities."""

from __future__ import annotations

import json
from pathlib import Path

from image_drawer.search.models import PromptCase


def _validate_prompt(prompt: object, *, location: str) -> str:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"{location}: prompt must be a non-empty string")
    return prompt.strip()


def load_prompt_set(path: str | Path) -> list[PromptCase]:
    source = Path(path)
    suffix = source.suffix.lower()

    if suffix in {"", ".txt"}:
        prompts: list[PromptCase] = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            prompts.append(
                PromptCase(
                    id=f"prompt-{len(prompts):04d}",
                    prompt=stripped,
                    metadata={"source_line": line_number},
                )
            )
        if not prompts:
            raise ValueError("prompt set is empty")
        return prompts

    if suffix == ".jsonl":
        prompts = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if isinstance(payload, str):
                prompt = _validate_prompt(
                    payload, location=f"line {line_number}"
                )
                metadata = {}
                prompt_id = f"prompt-{len(prompts):04d}"
            elif isinstance(payload, dict):
                prompt = _validate_prompt(
                    payload.get("prompt"), location=f"line {line_number}"
                )
                metadata = payload.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"line {line_number}: metadata must be an object"
                    )
                prompt_id = str(
                    payload.get("id", f"prompt-{len(prompts):04d}")
                )
                if not prompt_id:
                    raise ValueError(
                        f"line {line_number}: id must not be empty"
                    )
            else:
                raise ValueError(
                    f"line {line_number}: expected string or object"
                )
            prompts.append(
                PromptCase(
                    id=prompt_id,
                    prompt=prompt,
                    metadata=dict(metadata),
                )
            )
        if not prompts:
            raise ValueError("prompt set is empty")
        if len({item.id for item in prompts}) != len(prompts):
            raise ValueError("prompt IDs must be unique")
        return prompts

    if suffix == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("JSON prompt set must be an array")
        prompts = []
        for index, item in enumerate(payload):
            if isinstance(item, str):
                prompts.append(
                    PromptCase(
                        id=f"prompt-{index:04d}",
                        prompt=_validate_prompt(
                            item, location=f"item {index}"
                        ),
                    )
                )
            elif isinstance(item, dict):
                prompt = _validate_prompt(
                    item.get("prompt"), location=f"item {index}"
                )
                metadata = item.get("metadata", {})
                if not isinstance(metadata, dict):
                    raise ValueError(
                        f"item {index}: metadata must be an object"
                    )
                prompts.append(
                    PromptCase(
                        id=str(item.get("id", f"prompt-{index:04d}")),
                        prompt=prompt,
                        metadata=dict(metadata),
                    )
                )
            else:
                raise ValueError(
                    f"item {index}: expected string or object"
                )
        if not prompts:
            raise ValueError("prompt set is empty")
        if len({item.id for item in prompts}) != len(prompts):
            raise ValueError("prompt IDs must be unique")
        return prompts

    raise ValueError(
        f"unsupported prompt-set format: {source.suffix}; "
        "use .txt, .jsonl, or .json"
    )

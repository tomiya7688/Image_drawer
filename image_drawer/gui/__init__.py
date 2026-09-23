"""Image Drawer workflow workbench entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from image_drawer.gui.controller import (
    DEFAULT_WORKFLOW_DSL,
    WorkbenchController,
)
from image_drawer.part_bank import MetadataHashEmbedder, SQLitePartRepository
from image_drawer.runtime import WorkflowRuntime
from image_drawer.steps import create_mock_registry, create_part_bank_registry


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="image-drawer-gui",
        description="Image Drawer workflow workbench.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Run a headless entrypoint check suitable for CI.",
    )
    parser.add_argument(
        "--bank",
        type=Path,
        help="Part Bank directory containing metadata.sqlite3.",
    )
    parser.add_argument(
        "--workflow",
        type=Path,
        help="Workflow DSL file to open on startup.",
    )
    parser.add_argument(
        "--embedding-dimensions",
        type=int,
        default=128,
        help="MetadataHashEmbedder dimensions for the local Part Bank backend.",
    )
    return parser


def _build_controller(args: argparse.Namespace):
    repository = None
    if args.bank is None:
        registry = create_mock_registry()
        controller = WorkbenchController(
            registry,
            WorkflowRuntime(registry),
        )
    else:
        bank = args.bank.resolve()
        metadata = bank / "metadata.sqlite3"
        if not metadata.is_file():
            raise FileNotFoundError(
                f"Part Bank metadata database not found: {metadata}"
            )
        repository = SQLitePartRepository(metadata)
        embedder = MetadataHashEmbedder(
            dimensions=args.embedding_dimensions
        )
        registry = create_part_bank_registry(
            repository,
            embedder,
            bank_dir=bank,
        )
        controller = WorkbenchController(
            registry,
            WorkflowRuntime(registry),
            bank_dir=bank,
        )
    if args.workflow is not None:
        controller.load_workflow(args.workflow)
    return controller, repository


def main(argv: Sequence[str] | None = None) -> int:
    args = create_parser().parse_args(argv)
    if args.check:
        registry = create_mock_registry()
        controller = WorkbenchController(registry)
        assert controller.validate() is None
        assert controller.canonical_dsl() == DEFAULT_WORKFLOW_DSL
        print(
            json.dumps(
                {"component": "gui", "status": "ok"},
                sort_keys=True,
            )
        )
        return 0

    controller, repository = _build_controller(args)
    try:
        from image_drawer.gui.app import run_app

        return run_app(controller)
    finally:
        if repository is not None:
            repository.close()


__all__ = [
    "DEFAULT_WORKFLOW_DSL",
    "WorkbenchController",
    "create_parser",
    "main",
]

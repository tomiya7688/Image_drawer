"""固定7工程を強制するStageController（S1b）。目標画像は受け取らない。"""

from .errors import StageClosedError, StageConfigError, StageInvariantError, StageTableError
from .table import StageSpec, StageTable, default_stage_table, load_stage_table, parse_stage_table

__all__ = [
    "StageClosedError",
    "StageConfigError",
    "StageInvariantError",
    "StageSpec",
    "StageTable",
    "StageTableError",
    "default_stage_table",
    "load_stage_table",
    "parse_stage_table",
]

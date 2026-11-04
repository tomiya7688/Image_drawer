"""固定7工程を強制するStageController（S1b）。目標画像は受け取らない。"""

from .budget import MAX_ACTIONS_LIMIT, Budget
from .errors import StageClosedError, StageConfigError, StageInvariantError, StageTableError
from .intent import STYLE_INTENT_VERSION, StyleIntent, judge_style_intent
from .table import StageSpec, StageTable, default_stage_table, load_stage_table, parse_stage_table

__all__ = [
    "MAX_ACTIONS_LIMIT",
    "STYLE_INTENT_VERSION",
    "Budget",
    "StageClosedError",
    "StageConfigError",
    "StageInvariantError",
    "StageSpec",
    "StageTable",
    "StageTableError",
    "StyleIntent",
    "default_stage_table",
    "judge_style_intent",
    "load_stage_table",
    "parse_stage_table",
]

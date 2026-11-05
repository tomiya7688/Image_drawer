"""工程の遵守・有効stroke・拒否・計算コストの指標(目標画像を読まない)。

受理済みActionの遵守率は構造上100%になるため成果として扱わず、提案レベルの率と監査結果から測る。
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from statistics import median

from image_drawer.drawing import DrawingEpisode
from image_drawer.drawing.stage.controller import AttemptRecord
from image_drawer.generalization.auditor import REQUIRED_STAGES, AuditResult, audit_episode

# 工程の規則による拒否の理由コード(Controllerの"<code>: <detail>"の<code>)
STAGE_RULE_CODES = frozenset(
    {
        "action_not_allowed",
        "stage_not_ready",
        "layer_not_selectable",
        "layer_not_writable",
        "review_reason_not_allowed",
        "review_after_drawing",
        "review_intent_not_confirmed",
        "role_not_creatable",
    }
)
DOT_STROKE_LIMIT_PX = 1.0


# {
#   責務: [
#     ProcessSample: 1回の生成の、提案ログと監査結果
#   ]
#   フィールド: [
#     attempts: 受理・拒否を通した提案の記録
#     audit: 保存Episodeの監査結果。生成が失敗してEpisodeが無ければNone
#     is_complete: finishに達した完成か
#     attempt_limit_hit: 試行上限に達して終了したか
#   ]
# }
@dataclass(frozen=True)
class ProcessSample:
    attempts: tuple[AttemptRecord, ...]
    audit: AuditResult | None
    is_complete: bool
    attempt_limit_hit: bool = False


# {
#   責務: [
#     make_process_sample: Episodeと提案ログからProcessSampleを作る(監査はEpisodeだけで行う)
#   ]
# }
def make_process_sample(
    episode: DrawingEpisode,
    attempts: Sequence[AttemptRecord],
    *,
    attempt_limit_hit: bool = False,
) -> ProcessSample:
    return ProcessSample(
        attempts=tuple(attempts),
        audit=audit_episode(episode),
        is_complete=episode.status == "complete",
        attempt_limit_hit=attempt_limit_hit,
    )


# {
#   責務: [
#     ProcessMetrics: 工程・stroke・拒否の指標。分母が0の率はNone(0%と区別する)
#   ]
#   フィールド: [
#     generations, completed: 生成数と完成数
#     completion_rate, audit_pass_rate, proposal_legal_rate, stage_violation_rate: 設計6.2節
#     auditor_controller_disagreements: 完成なのに監査が不合格だった件数(0でなければバグ)
#     review_only_rate: 工程ごとの、レビューのみで通過した生成 ÷ 完成数
#     accepted_stroke_rate, effective_stroke_rate, dot_stroke_rate: 設計6.3節
#     pixels_changed_per_required_stage: 必須工程ごとの、生成ごとの変更画素数の中央値
#     rejection_rate, rejection_by_reason, rejection_by_stage: 設計6.4節。理由別・工程別は拒否数
#     attempt_limit_hit_rate, max_consecutive_rejections: 試行上限に達した割合と連続拒否の最大長
#   ]
# }
@dataclass(slots=True)
class ProcessMetrics:
    generations: int
    completed: int
    completion_rate: float | None
    audit_pass_rate: float | None
    proposal_legal_rate: float | None
    stage_violation_rate: float | None
    auditor_controller_disagreements: int
    review_only_rate: dict[str, float | None]
    accepted_stroke_rate: float | None
    effective_stroke_rate: float | None
    dot_stroke_rate: float | None
    pixels_changed_per_required_stage: dict[str, float | None]
    rejection_rate: float | None
    rejection_by_reason: dict[str, int] = field(default_factory=dict)
    rejection_by_stage: dict[str, int] = field(default_factory=dict)
    attempt_limit_hit_rate: float | None = None
    max_consecutive_rejections: int = 0


# {
#   責務: [ _ratio: 分母が0ならNoneを返す ]
# }
def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


# {
#   責務: [ _reason_code: 拒否理由"<code>: <detail>"から<code>だけを取り出す ]
# }
def _reason_code(reason: str | None) -> str:
    return (reason or "unknown").split(":", 1)[0].strip()


# {
#   責務: [ _max_rejection_run: 提案の並びの中の、連続した拒否の最大長 ]
# }
def _max_rejection_run(attempts: Sequence[AttemptRecord]) -> int:
    longest = run = 0
    for attempt in attempts:
        run = 0 if attempt.accepted else run + 1
        longest = max(longest, run)
    return longest


# {
#   責務: [
#     compute_process_metrics: 生成の集合から、工程・stroke・拒否の指標を計算する
#   ]
#   処理: [
#     1: 失敗した生成も分母に残す(完成率・拒否率の分母に含める)
#     2: 提案ログから、受理率・工程規則による拒否・理由別/工程別の拒否・連続拒否を数える
#     3: 監査結果から、有効stroke・ドット・必須工程の変更画素数・レビューのみの工程を数える
#   ]
# }
def compute_process_metrics(samples: Sequence[ProcessSample]) -> ProcessMetrics:
    attempts = [a for sample in samples for a in sample.attempts]
    rejected = [a for a in attempts if not a.accepted]
    completed = [s for s in samples if s.is_complete]
    audited = [s for s in samples if s.audit is not None]
    records = [r for s in audited for r in s.audit.stroke_records if r.action_type == "stroke_v2"]
    proposed_strokes = [a for a in attempts if a.action.type == "stroke_v2"]
    stage_rule = [a for a in rejected if _reason_code(a.reason) in STAGE_RULE_CODES]
    passed = [s for s in completed if s.audit is not None and s.audit.passed]
    review_counts = Counter(
        stage for s in completed if s.audit is not None for stage in s.audit.review_only_stages
    )
    stage_ids = ("rough", "sketch", "final_drawing", "base_color", "shading", "refine")
    pixels = {
        stage: (
            median(s.audit.changed_pixels_by_stage.get(stage, 0) for s in audited)
            if audited
            else None
        )
        for stage in REQUIRED_STAGES
    }
    return ProcessMetrics(
        generations=len(samples),
        completed=len(completed),
        completion_rate=_ratio(len(completed), len(samples)),
        audit_pass_rate=_ratio(len(passed), len(completed)),
        proposal_legal_rate=_ratio(len(attempts) - len(rejected), len(attempts)),
        stage_violation_rate=_ratio(len(stage_rule), len(attempts)),
        auditor_controller_disagreements=sum(
            1 for s in completed if s.audit is not None and not s.audit.passed
        ),
        review_only_rate={
            stage: _ratio(review_counts[stage], len(completed)) for stage in stage_ids
        },
        accepted_stroke_rate=_ratio(
            sum(a.accepted for a in proposed_strokes), len(proposed_strokes)
        ),
        effective_stroke_rate=_ratio(sum(r.effective for r in records), len(records)),
        dot_stroke_rate=_ratio(
            sum(r.path_length_px < DOT_STROKE_LIMIT_PX for r in records), len(records)
        ),
        pixels_changed_per_required_stage=pixels,
        rejection_rate=_ratio(len(rejected), len(attempts)),
        rejection_by_reason=dict(Counter(_reason_code(a.reason) for a in rejected)),
        rejection_by_stage=dict(Counter(a.stage_id for a in rejected)),
        attempt_limit_hit_rate=_ratio(sum(s.attempt_limit_hit for s in samples), len(samples)),
        max_consecutive_rejections=max(
            (_max_rejection_run(s.attempts) for s in samples), default=0
        ),
    )


# {
#   責務: [
#     GenerationCost: 1生成の計算コスト(設計6.5節)。取得できない値はNone
#   ]
#   フィールド: [
#     accepted_actions, attempts, system_events: 件数
#     renderer_points_used, renderer_area_used: EngineのDrawingState.points_used/area_used
#     wall_policy_seconds, wall_engine_seconds, wall_total_seconds: 実時間
#     renderer_points_used など取得できない値はNone。peak_memory_mb: 取得できなければNone(推定値を入れない)
#   ]
# }
@dataclass(frozen=True, slots=True)
class GenerationCost:
    accepted_actions: int
    attempts: int
    system_events: int
    renderer_points_used: int | None
    renderer_area_used: int | None
    wall_policy_seconds: float | None
    wall_engine_seconds: float | None
    wall_total_seconds: float
    peak_memory_mb: float | None = None


# {
#   責務: [ _percentile90: 値の90パーセンタイル(線形補間なし、最近傍の順位) ]
# }
def _percentile90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)]


# {
#   責務: [
#     summarize_costs: 項目ごとに中央値とp90を返す
#   ]
#   処理: [
#     Noneの値は集計から除く。全生成でNoneの項目は、中央値もp90もNoneにする(0や推定値を入れない)
#   ]
#   戻り値: [
#     summary: 項目名 -> {"median", "p90"}
#   ]
# }
def summarize_costs(costs: Sequence[GenerationCost]) -> dict[str, dict[str, float | None]]:
    summary: dict[str, dict[str, float | None]] = {}
    for name in GenerationCost.__slots__:
        values = [getattr(cost, name) for cost in costs if getattr(cost, name) is not None]
        summary[name] = {
            "median": float(median(values)) if values else None,
            "p90": float(_percentile90(values)) if values else None,
        }
    return summary

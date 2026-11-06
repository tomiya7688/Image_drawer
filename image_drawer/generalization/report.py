"""評価報告書(idraw-eval-report/v1)のJSONとMarkdownの生成。未確認の範囲を明示し、許さない表現を拒否する。"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from statistics import mean
from typing import Any

from image_drawer.generalization.compare import metric_names, summarize_condition
from image_drawer.generalization.eval_set import EvalSet
from image_drawer.generalization.gates import GateInputs, decide_claim_level, evaluate_gates
from image_drawer.generalization.runner import RunOutput
from image_drawer.generalization.stats import cluster_bootstrap

SCHEMA_VERSION = "idraw-eval-report/v1"
DISALLOWED_PHRASES = ("高精度", "必ず収束", "プロ品質")
EXAMPLE_COUNT = 12
HEADINGS = (
    "結論と主張の水準",
    "条件",
    "データと分割",
    "結果",
    "工程の遵守と有効stroke、拒否",
    "失敗の集計",
    "prompt忠実度と記憶の検査",
    "人間評価",
    "計算コスト",
    "制限と未確認の範囲",
    "再現手順",
    "付録: 出力例",
)
_ALLOWED = {
    "none": ["評価していない"],
    "pipeline_verified": ["配線とReplayの一致を確認した(テスト用Policy)"],
    "improvement_observed": ["(条件)で学習前より改善が観測された"],
    "high_accuracy_claim_allowed": ["(データ・解像度・件数の範囲で)高精度"],
}
_DISALLOWED = [
    "モデルの品質・精度・学習の効果(L1以下)",
    "イラスト全般・あらゆるpromptへの一般化",
    "人間の絵師の筆跡を学習した、元画像から描画行動を推定した",
]


# {
#   責務: [
#     validate_claim_text: L3の報告書への参照が無い限り、「高精度」などの語を含む文を拒否する
#   ]
#   引数: [ text: 報告書または公表文, claim_level: 水準, report_id: L3のときに参照する報告書ID ]
#   エラー: [ ValueError: 許されない表現を含む ]
# }
def validate_claim_text(text: str, claim_level: str, report_id: str | None = None) -> None:
    permitted = claim_level == "high_accuracy_claim_allowed" and report_id and report_id in text
    for phrase in DISALLOWED_PHRASES:
        if phrase in text and not permitted:
            raise ValueError(f"disallowed phrase without an L3 report reference: {phrase}")


# {
#   責務: [ _condition_metrics: 条件ごとの指標をITTと条件付きで、項目単位のgroupブートストラップ区間とともに返す ]
# }
def _condition_metrics(output: RunOutput, condition: str, resamples: int, rng_seed: int) -> dict:
    summary = summarize_condition(output, condition)
    rows = [g for g in output.generations if g.condition == condition]
    groups = {g.image_id: g.group_id for g in rows}
    metrics: dict[str, Any] = {}
    for metric, by_reporting in summary.items():
        for reporting, info in by_reporting.items():
            per_item: dict[str, list[float]] = {}
            for g in rows:
                blank = output.blank_metrics[g.image_id][metric]
                value = None if g.failure_kind or metric not in g.metrics else g.metrics[metric]
                if value is None and reporting == "itt":
                    value = blank
                if value is not None:
                    per_item.setdefault(g.image_id, []).append(value)
            ids = sorted(per_item)
            interval = [None, None]
            if ids:
                _, lower, upper = cluster_bootstrap(
                    [mean(per_item[i]) for i in ids],
                    [groups[i] for i in ids],
                    resamples=resamples,
                    rng_seed=rng_seed,
                )
                interval = [lower, upper]
            metrics.setdefault(metric, {})[reporting] = {
                "mean": info["mean"],
                "ci95": interval,
                "n": len(ids),
                "denominator": info["denominator"],
                "failures_included": info["failures_included"],
                "reporting": reporting,
            }
    return metrics


# {
#   責務: [ _examples: 出力例。乱数で選んだ件と、得点が最も低い件を載せる(良い例だけを載せない) ]
# }
def _examples(output: RunOutput, rng_seed: int) -> dict[str, list[dict[str, Any]]]:
    metric = "rgb_128" if "rgb_128" in metric_names(output) else (metric_names(output) or [""])[0]
    rows = [g for g in output.generations if g.condition == "after"]

    def score(g) -> float:
        return (
            output.blank_metrics[g.image_id].get(metric, 0.0)
            if g.failure_kind
            else g.metrics.get(metric, 0.0)
        )

    def brief(g) -> dict[str, Any]:
        return {"image_id": g.image_id, "k": g.k, "failure_kind": g.failure_kind, "score": score(g)}

    shuffled = sorted(rows, key=lambda g: (g.image_id, g.k))
    random.Random(rng_seed).shuffle(shuffled)
    worst = sorted(rows, key=lambda g: (score(g), g.image_id, g.k))
    return {
        "metric": metric,
        "random": [brief(g) for g in shuffled[:EXAMPLE_COUNT]],
        "worst": [brief(g) for g in worst[:EXAMPLE_COUNT]],
    }


# {
#   責務: [ _unverified: 実測していない項目を列挙する(未実測を実測として書かない) ]
# }
def _unverified(inputs: GateInputs, output: RunOutput) -> list[str]:
    notes = []
    if inputs.policy_kind != "trained":
        notes.append("学習したPolicyの評価(試験用または未指定のPolicyのため、モデルの品質は未測定)")
    if inputs.human is None:
        notes.append("独立した人間評価(未実施)")
    missing = [
        name
        for name in ("prompt_shuffle_delta", "probe_hit_rate", "nearest_train_hit_rate")
        if name not in inputs.controls
    ]
    if missing:
        notes.append("・".join(missing) + "(未計算)")
    if inputs.reproduced is not True:
        notes.append("新しい環境での再現(未確認)")
    notes.append("OpenCLIP系の独立指標(heldout_openclip_*)は測定していない")
    notes.append(
        "peak_memory_mb・renderer_points_used・renderer_area_used・Policyとengineの時間内訳は未取得"
    )
    return notes


# {
#   責務: [
#     build_report: 比較・実行結果・評価集合・確認事実から報告書(JSON互換の辞書)を作る
#   ]
#   処理: [ ゲートを判定し、claim_levelと許す/許さない表現、制限、未確認の範囲を記録する ]
#   引数: [
#     comparison: compare_allの結果, output: RunOutput, eval_set: 評価集合, inputs: ゲート判定の入力
#     code: git_commitなどのコード情報, protocol: 推論seed・予算などの条件, reproduction: 再現コマンド
#     report_id, created_at, status, rng_seed, resamples
#   ]
# }
def build_report(
    comparison: dict[str, Any],
    output: RunOutput,
    eval_set: EvalSet,
    inputs: GateInputs,
    *,
    code: dict[str, Any],
    protocol: dict[str, Any],
    reproduction: dict[str, Any],
    report_id: str,
    created_at: str | None = None,
    status: str = "draft",
    conditions: list[dict[str, Any]] | None = None,
    resamples: int = 2000,
    rng_seed: int = 0,
) -> dict[str, Any]:
    gates = evaluate_gates(inputs)
    level = decide_claim_level(inputs.policy_kind, gates)
    names = sorted({g.condition for g in output.generations})
    by_condition = {
        name: {
            "n_items": len({g.image_id for g in output.generations if g.condition == name}),
            "n_generations": sum(1 for g in output.generations if g.condition == name),
            "metrics": _condition_metrics(output, name, resamples, rng_seed),
            "failure_breakdown": {
                kind: {
                    "count": count,
                    "rate": count / max(1, sum(g.condition == name for g in output.generations)),
                }
                for kind, count in output.failures.get(name, {}).items()
            },
            "cost": output.costs.get(name, {}),
        }
        for name in names
    }
    after_process = output.process.get("after") or next(iter(output.process.values()), {})
    return {
        "schema_version": SCHEMA_VERSION,
        "report_id": report_id,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "status": status,
        "claim_level": level,
        "code": code,
        "dataset": {
            "dataset_hash": eval_set.dataset_hash,
            "split_seed": eval_set.split_seed,
            "eval_set_id": eval_set.eval_set_id,
            "eval_set_sha256": eval_set.eval_set_sha256,
            "counts": {"test": {"images": len(eval_set.items), "groups": inputs.test_groups}},
            "exclusions": [
                {"image_id": i, "reason": "empty training prompt"} for i in eval_set.excluded
            ],
        },
        "conditions": conditions or [],
        "protocol": protocol,
        "results": {
            "by_condition": by_condition,
            "comparisons": comparison.get("comparisons", []),
            "controls": inputs.controls,
            "process": {
                "completion_rate": after_process.get("completion_rate"),
                "audit_pass_rate": after_process.get("audit_pass_rate"),
                "auditor_controller_disagreements": inputs.auditor_disagreements,
                "replay_exact_rate": inputs.replay_exact_rate,
            },
            "examples": _examples(output, rng_seed),
        },
        "human_eval": inputs.human,
        "gates": gates,
        "claims": {"allowed": _ALLOWED[level], "disallowed": _DISALLOWED},
        "limitations": [
            "結果は、使ったデータの種類・件数・解像度・seed数の範囲だけに当てはまる",
            "ハードウェアが異なる実行どうしは比較していない",
        ],
        "unverified": _unverified(inputs, output),
        "reproduction": reproduction,
    }


# {
#   責務: [ _format_value: 表の数値を表示用に整える(未計算は「未測定」) ]
# }
def _format_value(value: Any) -> str:
    return "未測定" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


# {
#   責務: [ _result_lines: 条件ごとの指標(ITT)と比較の表の行を作る ]
# }
def _result_lines(report: dict[str, Any]) -> list[str]:
    lines = [
        "| 条件 | 指標 | ITT平均 | 95%区間 | 条件付き平均 |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for name, info in report["results"]["by_condition"].items():
        for metric, by_reporting in info["metrics"].items():
            itt, cond = by_reporting["itt"], by_reporting["conditional"]
            interval = " 〜 ".join(_format_value(v) for v in itt["ci95"])
            lines.append(
                f"| {name} | {metric} | {_format_value(itt['mean'])} | {interval} | {_format_value(cond['mean'])} |"
            )
    lines += [
        "",
        "| 比較 | 部分集合 | 指標 | 差 | 95%区間 | 勝率 |",
        "| --- | --- | --- | ---: | --- | ---: |",
    ]
    for row in report["results"]["comparisons"]:
        interval = " 〜 ".join(_format_value(v) for v in row["ci95"])
        lines.append(
            f"| {row['a']} − {row['b']} | {row['subset']} | {row['metric']} | {_format_value(row['delta'])} | {interval} | {_format_value(row['win_rate'])} |"
        )
        if row["ci95"][0] <= 0 <= row["ci95"][1]:
            lines.append("|  |  | 信頼区間が0を含むため、改善を確認できない |  |  |  |")
    return lines


# {
#   責務: [
#     render_markdown: 報告書の辞書を、必須の12見出しを持つMarkdownにする
#   ]
#   処理: [ 許されない表現を検査し、未確認の範囲と、良い例だけでない出力例(乱数と最低得点)を載せる ]
#   エラー: [ ValueError: L3の報告書への参照なしに禁止表現を含む ]
# }
def render_markdown(report: dict[str, Any]) -> str:
    level, report_id = report["claim_level"], report["report_id"]
    process, examples = report["results"]["process"], report["results"]["examples"]
    sections: dict[str, list[str]] = {
        HEADINGS[0]: [
            f"- report_id: {report_id}",
            f"- claim_level: {level}",
            f"- 使える表現: {' / '.join(report['claims']['allowed'])}",
            f"- 使えない表現: {' / '.join(report['claims']['disallowed'])}",
            *[
                f"- {g['gate_id']}: {'合格' if g['passed'] else '不合格'} ({g['evidence']})"
                for g in report["gates"]
            ],
        ],
        HEADINGS[1]: [f"- コード: {report['code']}", f"- 条件: {report['conditions']}"],
        HEADINGS[2]: [
            f"- {report['dataset']['counts']}",
            f"- eval_set_id: {report['dataset']['eval_set_id']}",
            f"- 除外: {len(report['dataset']['exclusions'])}件",
        ],
        HEADINGS[3]: _result_lines(report),
        HEADINGS[4]: [
            f"- 完成率: {_format_value(process['completion_rate'])}",
            f"- 監査合格率: {_format_value(process['audit_pass_rate'])}",
            f"- 監査器とControllerの不一致: {_format_value(process['auditor_controller_disagreements'])}",
        ],
        HEADINGS[5]: [
            f"- {name}: {info['failure_breakdown']}"
            for name, info in report["results"]["by_condition"].items()
        ],
        HEADINGS[6]: [f"- {report['results']['controls'] or '未計算'}"],
        HEADINGS[7]: [f"- {report['human_eval'] if report['human_eval'] else '未実施'}"],
        HEADINGS[8]: [
            f"- {name}: {info['cost']}" for name, info in report["results"]["by_condition"].items()
        ],
        HEADINGS[9]: [
            *[f"- 制限: {t}" for t in report["limitations"]],
            *[f"- 未確認: {t}" for t in report["unverified"]],
        ],
        HEADINGS[10]: [f"- {c}" for c in report["reproduction"]["commands"]],
        HEADINGS[11]: [
            f"指標: {examples['metric']}",
            "ランダムに選んだ例:",
            *[f"- {e}" for e in examples["random"]],
            "得点が最も低い例:",
            *[f"- {e}" for e in examples["worst"]],
        ],
    }
    parts = ["# 汎化評価報告書", ""]
    for number, heading in enumerate(HEADINGS, start=1):
        parts += [f"## {number}. {heading}", "", *sections[heading], ""]
    text = "\n".join(parts)
    validate_claim_text(text, level, report_id)
    return text

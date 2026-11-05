"""image-drawer-evaluateの報告書・人間評価のサブコマンド。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from image_drawer.generalization.eval_set import EvalSet
from image_drawer.generalization.gates import GateInputs
from image_drawer.generalization.human import (
    HumanEvalPlan,
    build_human_pack,
    collect_human_results,
)


# {
#   責務: [ _git: gitコマンドの出力。取得できなければ空文字 ]
# }
def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, timeout=20, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return ""


# {
#   責務: [
#     cmd_report: 比較と実行結果から報告書のJSONとMarkdownを作る
#   ]
#   処理: [
#     ゲート判定に使う事実のうち、このコマンドが実測できるもの(作業ツリー、Replay一致率、完成率など)だけを入れる。
#     学習したPolicyの確認・再現の確認・対照指標は、実施した結果をまだ受け取らないため未確認として扱う
#   ]
# }
def cmd_report(args: argparse.Namespace) -> int:
    from image_drawer.generalization.report import build_report, render_markdown
    from image_drawer.generalization.runner import RunOutput

    comparison = json.loads(args.compare.read_text(encoding="utf-8"))
    output = RunOutput.from_json((args.run / "run.json").read_text(encoding="utf-8"))
    eval_set = EvalSet.load(args.eval_set)
    human = json.loads(args.human.read_text(encoding="utf-8")) if args.human else None
    after = output.process.get("after", {})
    exact = [g.replay_exact for g in output.generations if g.replay_exact is not None]
    inputs = GateInputs(
        policy_kind="none",
        worktree_clean=_git("status", "--porcelain") == "" if _git("rev-parse", "HEAD") else None,
        replay_exact_rate=(sum(exact) / len(exact)) if exact else None,
        auditor_disagreements=after.get("auditor_controller_disagreements"),
        completion_rate=after.get("completion_rate"),
        audit_pass_rate=after.get("audit_pass_rate"),
        test_groups=len({i.group_id for i in eval_set.items}),
        exclusions_disclosed=True,
        comparisons=comparison.get("comparisons", []),
        human=human,
    )
    report = build_report(
        comparison,
        output,
        eval_set,
        inputs,
        code={"git_commit": _git("rev-parse", "HEAD") or "unknown"},
        protocol={
            "hardware_ids": output.hardware_ids,
            "bootstrap": comparison.get("bootstrap", {}),
        },
        reproduction={
            "commands": [" ".join(sys.argv)],
            "input_hashes": {"eval_set": eval_set.eval_set_sha256},
        },
        report_id=f"{eval_set.eval_set_id}-{output.split}",
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print(f"report {report['report_id']}: claim_level={report['claim_level']}")
    return 0


# {
#   責務: [ cmd_human_pack: 実行結果の画像から人間評価の盲検パックを作る(k=0の画像と白紙を使う) ]
# }
def cmd_human_pack(args: argparse.Namespace) -> int:
    from PIL import Image

    plan = HumanEvalPlan.from_json(args.plan.read_text(encoding="utf-8"))
    eval_set = EvalSet.load(args.eval_set)
    images_dir = args.run / "images"
    if not images_dir.is_dir():
        raise ValueError(f"generated images not found: {images_dir}")
    images: dict[tuple[str, str], bytes] = {}
    for condition_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        for item_id in plan.item_ids:
            path = condition_dir / f"{item_id}.png"
            if path.is_file():
                images[(item_id, condition_dir.name)] = path.read_bytes()
    if not images:
        raise ValueError(f"no generated images for the planned items in {images_dir}")
    # 白紙は確認項目に使う。大きさは生成画像と同じにする。
    import io

    size = Image.open(io.BytesIO(next(iter(images.values())))).size
    buffer = io.BytesIO()
    Image.new("RGB", size, (255, 255, 255)).save(buffer, format="PNG")
    for item_id in plan.item_ids:
        images[(item_id, "blank")] = buffer.getvalue()
    prompts = {item.image_id: item.prompt for item in eval_set.items}
    key_path = args.out.with_name(args.out.name + "_blind_key.json")
    count = build_human_pack(plan, images, prompts, args.out, key_path)
    print(f"human pack: {count} pairs; blind key: {key_path} (評価者へ渡さない)")
    return 0


# {
#   責務: [ cmd_human_collect: 回答を集計してJSONで保存する ]
# }
def cmd_human_collect(args: argparse.Namespace) -> int:
    plan = HumanEvalPlan.from_json(args.plan.read_text(encoding="utf-8"))
    eval_set = EvalSet.load(args.eval_set)
    groups = {item.image_id: item.group_id for item in eval_set.items}
    key_path = args.pack.with_name(args.pack.name + "_blind_key.json")
    result = collect_human_results(plan, key_path, args.responses, groups)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0

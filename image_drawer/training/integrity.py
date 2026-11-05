"""trialの整合検査(Replay一致、許可Action型)と、禁止importの静的検査。"""

from __future__ import annotations

import ast
from pathlib import Path

from image_drawer.drawing.replay import replay
from image_drawer.drawing.stage import default_stage_table
from image_drawer.training.rollout import Trial

# 禁止する依存(Part配置・img2img・推論入口・GUI)。設計書4.6節。
FORBIDDEN_IMPORT_PREFIXES = (
    "image_drawer.part_bank",
    "image_drawer.steps",
    "image_drawer.gui",
    "image_drawer.evaluation",
)
FORBIDDEN_IMPORT_NAMES = ("retrieve_parts", "compose", "part_bank")
CONTROLLER_ONLY_ACTORS = ("system",)


# {
#   責務: [ IntegrityError: trialが学習に使えない(外部画素・許可外Actionなど)ことを表す ]
# }
class IntegrityError(ValueError):
    pass


# {
#   責務: [ verify_trial_replay: EpisodeをEngineでReplayし、保存した合成と画素一致することを確認する ]
#   処理: [ Engineを経由しない画素が混ざれば、Replayの合成と食い違う ]
#   エラー: [ IntegrityError: 不一致、またはReplay失敗 ]
# }
def verify_trial_replay(trial: Trial) -> None:
    if trial.final_composite is None:
        raise IntegrityError("final_compositeがありません")
    try:
        result = replay(trial.episode)
    except Exception as error:  # Replay失敗は整合性の失敗として扱う。
        raise IntegrityError(f"Replayに失敗しました: {error}") from error
    if result.image.rgba != trial.final_composite:
        raise IntegrityError("ReplayとTrialの最終合成が画素単位で一致しません")


# {
#   責務: [ verify_allowed_actions: actor=modelのeventが許可された描画v2の種別だけであることを確認する ]
#   エラー: [ IntegrityError: pasteやcomposeなどの許可外の種別 ]
# }
def verify_allowed_actions(trial: Trial) -> None:
    allowed = default_stage_table().allowed_model_actions
    for event in trial.episode.events:
        if event.actor == "model" and event.type not in allowed:
            raise IntegrityError(f"許可されていないAction種別です: {event.type}")


# {
#   責務: [ imported_modules: ファイルのimport先(from x import y を x と x.y の両方)を返す ]
# }
def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules |= {alias.name for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
            modules |= {f"{node.module}.{alias.name}" for alias in node.names}
    return modules


# {
#   責務: [ forbidden_imports: パッケージ内の全ファイルから禁止importを探し「ファイル: モジュール」で返す ]
# }
def forbidden_imports(package_dir: Path) -> list[str]:
    found: list[str] = []
    for path in sorted(package_dir.glob("*.py")):
        for module in imported_modules(path):
            parts = module.split(".")
            prefixed = any(
                module == prefix or module.startswith(prefix + ".")
                for prefix in FORBIDDEN_IMPORT_PREFIXES
            )
            named = any(
                part in FORBIDDEN_IMPORT_NAMES or part.startswith("select_") for part in parts
            )
            if prefixed or named:
                found.append(f"{path.name}: {module}")
    return found

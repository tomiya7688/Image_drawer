"""凍結宣言とtest帳簿。固定したtestを調整に使わないことを、実行時に強制する。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from image_drawer.core import SerializableModel
from image_drawer.generalization.eval_set import EvalSet

LEDGER_NAME = "test_ledger.jsonl"


# {
#   責務: [
#     sha256_file: ファイルの内容のSHA-256を返す
#   ]
# }
def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# {
#   責務: [
#     FreezeManifest: test評価の前に固定する入力の宣言
#   ]
#   フィールド: [
#     eval_set_id, eval_set_sha256: 評価集合
#     before_checkpoint_sha256, after_checkpoint_sha256: 比較する2つのcheckpoint
#     git_commit: コードのcommit
#     reward_version, reward_weights_sha256, metrics_version, metrics_config_sha256: 指標の版と設定
#     inference_seed_count, max_actions: 推論の条件
#     analysis_plan_sha256: 分析計画のハッシュ
#     created_at: 宣言の日時
#   ]
# }
@dataclass(slots=True)
class FreezeManifest(SerializableModel):
    eval_set_id: str
    eval_set_sha256: str
    before_checkpoint_sha256: str
    after_checkpoint_sha256: str
    git_commit: str
    reward_version: str
    reward_weights_sha256: str
    metrics_version: str
    metrics_config_sha256: str
    inference_seed_count: int
    max_actions: int
    analysis_plan_sha256: str
    created_at: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    # {
    #   責務: [
    #     digest: 凍結宣言全体のSHA-256(帳簿に記録する値)
    #   ]
    # }
    def digest(self) -> str:
        payload = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    # {
    #   責務: [
    #     save: 凍結宣言をJSONで書き出す
    #   ]
    # }
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=2), encoding="utf-8")


# {
#   責務: [
#     create_freeze_manifest: 評価集合とcheckpointから凍結宣言を作る
#   ]
#   引数: [
#     eval_set: 凍結済みの評価集合
#     before_checkpoint, after_checkpoint: checkpointファイル
#     analysis_plan: 分析計画のファイル
#     そのほか: コードと指標の版
#   ]
# }
def create_freeze_manifest(
    eval_set: EvalSet,
    *,
    before_checkpoint: Path,
    after_checkpoint: Path,
    analysis_plan: Path,
    git_commit: str,
    reward_version: str,
    reward_weights_sha256: str,
    metrics_version: str,
    metrics_config_sha256: str,
    inference_seed_count: int = 5,
    max_actions: int = 1000,
    created_at: str | None = None,
) -> FreezeManifest:
    return FreezeManifest(
        eval_set_id=eval_set.eval_set_id,
        eval_set_sha256=eval_set.eval_set_sha256,
        before_checkpoint_sha256=sha256_file(before_checkpoint),
        after_checkpoint_sha256=sha256_file(after_checkpoint),
        git_commit=git_commit,
        reward_version=reward_version,
        reward_weights_sha256=reward_weights_sha256,
        metrics_version=metrics_version,
        metrics_config_sha256=metrics_config_sha256,
        inference_seed_count=inference_seed_count,
        max_actions=max_actions,
        analysis_plan_sha256=sha256_file(analysis_plan),
        created_at=created_at or datetime.now(timezone.utc).isoformat(),
    )


# {
#   責務: [
#     LedgerError: testの実行が規則により拒否されたことを表す(終了コード2に対応)
#   ]
# }
class LedgerError(RuntimeError):
    pass


# {
#   責務: [
#     read_ledger: 帳簿の全行を読む。無ければ空
#   ]
# }
def read_ledger(ledger_path: Path) -> list[dict]:
    if not ledger_path.exists():
        return []
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# {
#   責務: [
#     authorize_run: 実行するsplitに応じて、凍結宣言の要求と帳簿への記録を行う
#   ]
#   処理: [
#     1: validationは凍結宣言を要求せず、帳簿に触れない
#     2: testは凍結宣言が無ければ拒否する
#     3: 宣言のeval_set_sha256が評価集合と一致しなければ拒否する
#     4: 同じeval_set_idに別の凍結宣言の記録があればretest=Trueで追記する。
#        同じ宣言の再実行はretest=Falseで追記し、rerun=Trueとする
#   ]
#   引数: [
#     split: validation または test
#     eval_set: 評価集合
#     freeze_manifest_path: 凍結宣言のJSON(testでは必須)
#     ledger_path: 追記専用の帳簿
#     now: 日時。省略すると現在時刻
#   ]
#   戻り値: [
#     entry: 追記した帳簿の行。validationはNone
#   ]
#   エラー: [
#     LedgerError: 凍結宣言が無い、評価集合と一致しない、splitが不正
#   ]
# }
def authorize_run(
    split: str,
    eval_set: EvalSet,
    freeze_manifest_path: Path | None,
    ledger_path: Path,
    *,
    now: str | None = None,
) -> dict | None:
    if split == "validation":
        return None
    if split != "test":
        raise LedgerError(f"unsupported split for evaluation: {split}")
    if freeze_manifest_path is None or not freeze_manifest_path.exists():
        raise LedgerError("test run requires --freeze-manifest")
    manifest = FreezeManifest.from_json(freeze_manifest_path.read_text(encoding="utf-8"))
    if manifest.eval_set_sha256 != eval_set.eval_set_sha256:
        raise LedgerError("freeze manifest does not match the eval_set")
    freeze_digest = manifest.digest()
    previous = [e for e in read_ledger(ledger_path) if e["eval_set_id"] == eval_set.eval_set_id]
    retest = any(e["freeze_manifest_sha256"] != freeze_digest for e in previous)
    rerun = any(e["freeze_manifest_sha256"] == freeze_digest for e in previous)
    entry = {
        "eval_set_id": eval_set.eval_set_id,
        "freeze_manifest_sha256": freeze_digest,
        "run_index": len(previous) + 1,
        "retest": retest or any(e.get("retest") for e in previous),
        "rerun": rerun,
        "timestamp": now or datetime.now(timezone.utc).isoformat(),
    }
    # 追記専用: 既存の行を書き換えず、1行を末尾に足すだけにする。
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    return entry

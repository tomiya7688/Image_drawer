> **対象読者:** 実装・レビューを担当する開発者とコーディングエージェント
>
> **目的:** よく触る場所と、変更時に必ず一緒に直す箇所を1表にまとめ、探索と手戻りを減らす。
>
> **状態:** 2026-11-05。探索を繰り返した場所だけを追記する。

# 変更箇所ガイド

## 1. 変更の種類 → 触る場所

| やりたいこと | 触る場所 | 一緒に直す箇所 |
| --- | --- | --- |
| 描画の新しいActionを足す | `image_drawer/drawing/handlers*.py`、`models.py` | [drawing_engine_v2.md](drawing_engine_v2.md)、`episode_events.py`、schema JSON、テスト |
| 工程の規則を変える | `image_drawer/drawing/stage/rules.py`、`table.py` | `data/drawing_stage_workflow_v2.json`（コピー）、[固定工程](fixed_drawing_workflow_v2.md) |
| Episodeの形式を変える | `drawing/episode*.py`、`replay.py` | [schema](schemas/drawing_episode_v2.schema.json)、fixture再生成（`tests/episode_fixture_gen.py`） |
| 画像登録・抽出を変える | `image_drawer/dataset/`、`prompting/` | [image_dataset.md](image_dataset.md)、manifestのhash契約 |
| Workflowの新しいStep | `image_drawer/steps/`、`dsl/` | [workflow_dsl.md](workflow_dsl.md)、`StepSchema`、`runtime/validation.py` |
| 学習の部品 | `image_drawer/training/` | [design_trainer.md](design_trainer.md)、target隔離の契約テスト |

## 2. 毎回やること（スクリプト）

| 手順 | コマンド |
| --- | --- |
| 作業前の確認 | `python scripts/check_local.py` |
| 設計書のテスト名が実在するか | `python scripts/check_design_coverage.py [設計書]` |
| レビュー資料を作る（12000バイト以下に分割） | `python scripts/review_packet.py <base> <head> <出力先>` |
| 外部レビューを並列起動（Ori含む） | `scripts/review_run.ps1 -Dir <出力先>` |
| PR作成からマージ | `python scripts/pr_flow.py --title … --body-file … --review-dir <出力先> --merge` |

## 3. 落としやすい点

- 新しい設計書を実装したら、`scripts/check_design_coverage.py` の `IMPLEMENTED` へ追加する。
- レビューは毎回6者＋Ori＋CodeRabbitを実行し、`summary.txt` を証跡にする。省略は理由つきで記録する。
- `pyproject.toml` の依存は他ブランチと衝突しやすい。マージ時に両方残す。
- pytestは `--basetemp` をEドライブに指定する。`test_symlinks_are_not_followed` はWindows権限の既知の失敗。

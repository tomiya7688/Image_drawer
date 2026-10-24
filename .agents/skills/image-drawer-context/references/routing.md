# 作業別の入口

調査対象の行だけを使ってください。確認時点はmain `b5b491b`（2026-10-22）です。表は現在の実装を保証しないため、編集前に対象の定義と利用箇所を確かめてください。

| 作業 | 実装・代表的な確認先 | 詳細が必要な場合 |
| --- | --- | --- |
| 共通レコード、JSON、由来情報 | [core/models.py](../../../../image_drawer/core/models.py)、[serialization.py](../../../../image_drawer/core/serialization.py) → [test_core_models.py](../../../../tests/test_core_models.py) | [アーキテクチャ](../../../../docs/developers/architecture.md) |
| 実行順、Step間の型、途中失敗 | [runtime/engine.py](../../../../image_drawer/runtime/engine.py)のWorkflowRuntime、[validation.py](../../../../image_drawer/runtime/validation.py) → [test_runtime.py](../../../../tests/test_runtime.py) | [アーキテクチャ](../../../../docs/developers/architecture.md) |
| DSL解析・書き戻し | [dsl/parser.py](../../../../image_drawer/dsl/parser.py)のparse_workflow、[serializer.py](../../../../image_drawer/dsl/serializer.py) → [test_dsl.py](../../../../tests/test_dsl.py) | [DSL仕様](../../../../docs/developers/workflow_dsl.md) |
| 画像取り込み、重複、SQLite | [part_bank/ingest.py](../../../../image_drawer/part_bank/ingest.py)、[repository.py](../../../../image_drawer/part_bank/repository.py) → [test_part_bank.py](../../../../tests/test_part_bank.py) | [Part Bank実装](../../../../docs/developers/part_bank_v0.md) |
| 埋め込み、検索、RETRIEVE_PARTS | [part_bank/embedding.py](../../../../image_drawer/part_bank/embedding.py)、[index.py](../../../../image_drawer/part_bank/index.py)、[steps/retrieve_parts.py](../../../../image_drawer/steps/retrieve_parts.py) → [test_part_retrieval.py](../../../../tests/test_part_retrieval.py) | [Part Bank仕様](../../../../docs/developers/part_bank.md) |
| GUI起動 | [gui/__init__.py](../../../../image_drawer/gui/__init__.py)、[test_gui.py](../../../../tests/test_gui.py)。このmainではplaceholderです | [GUI設計](../../../../docs/developers/gui.md)。過去のWorkbenchは[統合メモ](integration.md) |
| インストール、配布、CI | [pyproject.toml](../../../../pyproject.toml)、[ci.yml](../../../../.github/workflows/ci.yml)、[verify_built_artifact.py](../../../../scripts/verify_built_artifact.py)、[verify_part_bank_artifact.py](../../../../scripts/verify_part_bank_artifact.py) | [操作例](operations.md) |
| この共有コンテキストの変更 | [AGENTS.md](../../../../AGENTS.md)、[容量・リンク確認](../../../../scripts/check_agent_context.py) | [スキル入口](../SKILL.md) |

## 表にない機能

エラー文、CLI名、Step名などを手掛かりに `rg -n` で定義と呼び出し元を探してください。mainにない実装を、設計書や保全ブランチにあるだけで利用可能と判断しないでください。
文書再編でパスが変わったら、同じ内容を別表に増やさず、この表のリンクを修正してください。

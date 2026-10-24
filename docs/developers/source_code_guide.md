> **対象読者:** 開発者  
> **目的:** コードレビューや実装開始時に、最初に読むsourceとcomponent間の関係を素早く把握する。

# ソースコードガイド

この文書は「Image Drawerのコードを読みたいが、どこから入ればよいか」を解決するための案内です。

## 1. 最短の入口

最初に読むfile:

- [tests/test_runtime.py](../../tests/test_runtime.py)

ここには現在の中核処理が最小構成でまとまっています。

流れ:

```text
WorkflowSpecを作る
-> StepRegistryを用意する
-> validate_workflow()
-> WorkflowRuntime.execute()
-> Artifact / Trajectoryを確認する
```

このtestを読んだあと、実装本体へ降りるのが最も追いやすいです。

## 2. 読む順番

### 2.1 Runtime

- [image_drawer/runtime/engine.py](../../image_drawer/runtime/engine.py)
- [image_drawer/runtime/validation.py](../../image_drawer/runtime/validation.py)

`WorkflowRuntime.execute()` がWorkflow実行の中心です。

validationでは、

- graph構造
- Step間の依存関係
- Artifact type
- Step parameter

を実行前に確認します。

### 2.2 Core data model

- [image_drawer/core/models.py](../../image_drawer/core/models.py)
- [image_drawer/core/serialization.py](../../image_drawer/core/serialization.py)
- [image_drawer/core/lineage.py](../../image_drawer/core/lineage.py)

ここでは次を定義します。

- `Artifact`
- `Score`
- `SourceImage`
- `Part`
- `PartSet`
- `Layout`
- `Composition`
- `StepSpec`
- `WorkflowSpec`
- `StepExecution`
- `Trajectory`

Image Drawerの内部data flowを理解したい場合はここが中心です。

### 2.3 Step

- [image_drawer/steps/base.py](../../image_drawer/steps/base.py)
- [image_drawer/steps/registry.py](../../image_drawer/steps/registry.py)
- [image_drawer/steps/mock.py](../../image_drawer/steps/mock.py)
- [image_drawer/steps/retrieve_parts.py](../../image_drawer/steps/retrieve_parts.py)

`Step` はWorkflowから呼び出される操作の共通interfaceです。

新しい処理を追加する場合は、基本的にRuntime本体を変更せず、

1. `Step` を実装
2. `StepSchema` を定義
3. `StepRegistry` へ登録

という流れにします。

### 2.4 DSL

- [image_drawer/dsl/parser.py](../../image_drawer/dsl/parser.py)
- [image_drawer/dsl/serializer.py](../../image_drawer/dsl/serializer.py)
- [tests/test_dsl.py](../../tests/test_dsl.py)

DSLはGUIで編集するWorkflowの保存・交換形式です。

内部ではDSLを直接実行せず、

```text
DSL
-> WorkflowSpec
-> validation
-> WorkflowRuntime
```

という流れです。

### 2.5 Part Bank

- [image_drawer/part_bank/ingest.py](../../image_drawer/part_bank/ingest.py)
- [image_drawer/part_bank/repository.py](../../image_drawer/part_bank/repository.py)
- [image_drawer/part_bank/extractor.py](../../image_drawer/part_bank/extractor.py)
- [image_drawer/part_bank/embedding.py](../../image_drawer/part_bank/embedding.py)
- [image_drawer/part_bank/index.py](../../image_drawer/part_bank/index.py)
- [image_drawer/part_bank/cli.py](../../image_drawer/part_bank/cli.py)
- [tests/test_part_bank.py](../../tests/test_part_bank.py)
- [tests/test_part_retrieval.py](../../tests/test_part_retrieval.py)

大量画像をSourceImage / Partへ変換し、保存・embedding・検索する層です。

現在はPart Bankだけ独立CLIがあります。

```text
image-drawer-part-bank ingest ...
```

## 3. 実行entrypoint

entrypoint定義は [pyproject.toml](../../pyproject.toml) にあります。

現在公開しているcommand:

### image-drawer

実装:

- [image_drawer/cli.py](../../image_drawer/cli.py)
- [image_drawer/__main__.py](../../image_drawer/__main__.py)

注意:

現時点ではM0のinput/output smoke contractです。

まだWorkflow Runtimeを操作する完成CLIではありません。

### image-drawer-gui

実装:

- [image_drawer/gui/](../../image_drawer/gui/)

現段階では本格的なWorkflow workbench実装前のentrypointです。

### image-drawer-part-bank

実装:

- [image_drawer/part_bank/cli.py](../../image_drawer/part_bank/cli.py)

現在の実機能を持つCLIの1つで、画像directoryのingestを行えます。

## 4. 目的別の入口

| やりたいこと | 最初に見る場所 |
|---|---|
| Workflow全体の実行を理解したい | [tests/test_runtime.py](../../tests/test_runtime.py) |
| Runtimeを変更したい | [image_drawer/runtime/engine.py](../../image_drawer/runtime/engine.py) |
| Workflow validationを変更したい | [image_drawer/runtime/validation.py](../../image_drawer/runtime/validation.py) |
| Artifact / Trajectoryを変更したい | [image_drawer/core/models.py](../../image_drawer/core/models.py) |
| 新しいStepを追加したい | [image_drawer/steps/base.py](../../image_drawer/steps/base.py) |
| Step backend登録を変更したい | [image_drawer/steps/registry.py](../../image_drawer/steps/registry.py) |
| DSLを変更したい | [image_drawer/dsl/parser.py](../../image_drawer/dsl/parser.py) |
| Part ingestを変更したい | [image_drawer/part_bank/ingest.py](../../image_drawer/part_bank/ingest.py) |
| Part保存形式を変更したい | [image_drawer/part_bank/repository.py](../../image_drawer/part_bank/repository.py) |
| Part検索を変更したい | [image_drawer/part_bank/embedding.py](../../image_drawer/part_bank/embedding.py) / [index.py](../../image_drawer/part_bank/index.py) |
| End-to-End testを追加したい | [tests/](../../tests/) |

## 5. 現在の全体像

```text
DSL
 |
 v
WorkflowSpec
 |
 v
validate_workflow
 |
 v
WorkflowRuntime
 |
 +---- StepRegistry ---- Step implementation
 |
 +---- Artifact / Score
 |
 v
Trajectory
```

Part Bankを使う場合:

```text
source images
 |
 v
Part Bank ingest
 |
 v
SourceImage / Part
 |
 v
embedding / index
 |
 v
RETRIEVE_PARTS Step
 |
 v
WorkflowRuntime
```

## 6. 設計文書との対応

コードだけでなく設計意図も確認したい場合:

- 全体構成: [architecture.md](architecture.md)
- DSL: [workflow_dsl.md](workflow_dsl.md)
- Part Bank: [part_bank.md](part_bank.md)
- Workflow Search: [workflow_search.md](workflow_search.md)
- 実装順: [実装計画](../kadoka/implementation_plan.md)

コードが設計文書と食い違う場合は、Issue/PRで差分を明示して更新します。

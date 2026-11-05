# はじめに

**対象読者:** Image Drawerを試したい利用者  
**目的:** 現在実装済みの機能と、利用可能なentrypointを把握する。

## 現在の状態

Image Drawerは開発中です。

現時点では完成した画像制作GUIではなく、Workflow RuntimeやPart Bankなどの基盤を段階的に実装しています。

## 現在利用できるentrypoint

### Part Bank

local画像directoryをPart Bankへingestできます。

```bash
python -m pip install -e ".[part-bank]"
image-drawer-part-bank ingest /path/to/images \
  --bank ./data/part-bank \
  --dataset my-dataset \
  --split train
```

詳細な内部仕様は開発者向けの [Part Bank v0実装仕様](../developers/part_bank_v0.md) にあります。

### image-drawer

```bash
image-drawer --input hello
```

現在はM0のsmoke / plumbing確認用entrypointです。完成したWorkflow実行CLIではありません。

### 文章だけから描く（image-drawer-draw）

```bash
image-drawer-draw --prompt "赤いリンゴ" --policy random-legal-v1 --out-dir out/apple
```

登録されているPolicyは未学習のテスト用で、絵の品質は評価されていません。詳しくは [文章だけから描く](draw.md)。

### GUI

```bash
image-drawer-gui --check   # headlessの起動確認
image-drawer-gui           # Replay・生成・Datasetの画面（tkinterが必要）
```

画面の操作・見た目は自動テストでは確認していません。

## 今後

利用者向けGUIやWorkflow実行手順が実装された時点で、このfolderへ操作手順を追加します。

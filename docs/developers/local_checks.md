# ローカル確認と整形

## 背景と目的

PR作成前に構文不正やテスト失敗を検出するため、短い検査を一つの入口へ集約します。GitHub Actionsのビルドと配布検証も引き続き実行します。

## 準備と実行方法

Python 3.11以上とGitが必要です。ルートで `python -m pip install -e ".[dev]"` を実行して開発用依存を導入してください。Ruffは `pyproject.toml` で **0.15.22** に固定しています。

PR前には次の1コマンドを実行してください。

```powershell
python scripts/check_local.py
```

この入口はorigin/mainとの共通祖先からの変更（commit済み・stage済み・未stage）と未追跡ファイルを調べます。Git管理外の生成物は読みません。変更Pythonに構文、Ruff format/checkと軽いlintを適用し、既存の文書検査と全pytestを呼びます。現在のpytestは小規模のため全件を使い、変更とテストの対応表による漏れを防ぎます。Makefileのtestと同じpytestを利用し、テスト判定を再実装しません。

結果はPASS / FAIL / SKIP / UNVERIFIEDと所要時間をまとめて表示します。Python変更がない検査はSKIPになりますが、文書・pytestは実行します。テスト収集0件はpytestの失敗終了として扱います。必須検査の失敗、依存不足、起動不能、120秒タイムアウトは失敗終了します。対象探索や構文が失敗した場合は、後続の検査をSKIPとして表示して中断します。結果と未確認範囲をPR本文に記載してください。

## 実装担当者の整形

実装後は担当するPythonだけを明示して整形し、通常の確認コマンドを再実行してください。指摘修正後も同じ手順を実行してください。

```powershell
python scripts/check_local.py --format scripts/check_local.py tests/test_check_local.py
python scripts/check_local.py
```

`--format` は明示したファイルを書き換えます。通常のチェックは自動修正・依存導入・Git操作を行いません。Ruffの除外はデータ・モデル・生成レポート・vendor・原文保持のtests/fixturesです。担当外のdirtyファイルを指定しないでください。MarkdownやYAMLは今回のPython formatterの対象外です。

## CIとの分担と制約

CIも同じ入口を比較元SHA付きで呼びます。通常利用にフラグは不要です。配布ビルド、wheel導入、画像出力smokeはCIに残します。画像素材全件、GPU、実モデル、実GUI操作はこの入口で検証しません。

origin/mainがなければfetchしてから再実行してください。チェック自体はネットワークに接続しません。pytestはルート内の.tmpへ実行ごとの専用一時フォルダを作り、終了時にその専用フォルダだけを片付けます。Git管理外のキャッシュが生じる場合があります。Windowsでsymlink権限や一時フォルダのアクセス権が不足した場合はFAILになり、成功扱いにはしません。既存の未コミット文書が容量制限を超えている場合も文書検査がFAILになります。

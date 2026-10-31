# Image Drawer

Image Drawer の最上位目的は、**人間の絵師がキャンバスを見て判断し、色と道具を選び、筆で描き、レイヤーを重ね、修正しながら完成させる制作過程そのものをAIに学習させること**です。

```text
現在の状態と制作履歴
    -> 次の描画行動を予測（色・筆・レイヤー・軌跡・修正など）
    -> 行動を実行してキャンバスを更新
    -> 結果を観察して次の行動を予測
```

色は**固定256色パレット**から選び、ブラシは色と独立した要素です。下書き・線画・着色は人間の制作上の工程を表し、**「着色＝一括塗りつぶし」ではありません**。

まず[コンセプト](docs/kadoka/concept.md)と[制作過程v1の正本](docs/developers/drawing_process_v1.md)・[受け入れ試験](docs/developers/drawing_process_acceptance.md)を読んでください。既存のPart Bank・COMPOSEは補助的な素材機能であり、筆による制作行動の学習の代わりではありません。

## 文書は読者別に分かれています

詳しい文書一覧は [docs/README.md](docs/README.md) にあります。

### Image Drawerを使いたい

→ [利用者向け文書](docs/users/)

現在使える機能、起動方法、CLI/GUIの使い方を確認します。

### コードを読みたい・実装したい

→ [開発者向け文書](docs/developers/)  
→ [ソースコードガイド](docs/developers/source_code_guide.md)

architecture、DSL、Part Bank、学習/評価、コードレビューの入口を確認します。

### Kadokaが方針・実装順を確認したい

→ [Kadoka向け文書](docs/kadoka/)

プロジェクトのコンセプト、優先順位、実装計画を確認します。

### KadokaのCodexが実装作業をする

→ [KadokaのCodex向け作業規約](docs/kadoka-codex/)

Issue/PR運用、参照すべき正本、test/build要件、禁止事項を確認します。

## 現在の状態

現在は基盤実装段階です。Workflow Runtime、DSL、Part Bankなどの既存機能がありますが、**制作行動の記録・完全再生・次行動の学習が実装済みという意味ではありません**。[新しい実装計画](docs/kadoka/implementation_plan.md)に基づきIssue / Pull Request単位で進めます。

完成した画像制作GUIはまだ提供していません。利用可能なentrypointは [利用者向けの「はじめに」](docs/users/getting_started.md) を参照してください。

## 言語

プロジェクト独自の説明文は日本語を正本とします。詳細は [言語運用方針](docs/developers/language_policy.md) を参照してください。

Apache-2.0や第三者ライセンスなど、原文自体に法的意味がある文書は公式原文を正本とします。

## PR前のローカル確認

開発用依存を `python -m pip install -e ".[dev]"` で導入し、ルートで `python scripts/check_local.py` を実行してください。変更Pythonの構文・Ruff整形/lint、差分整合性、既存の文書検査・pytestを実行し、結果と未確認範囲を集約します。配布・GPU・実GUIの検証は別に行います。詳しくは[開発手順](docs/developers/local_checks.md)を参照してください。

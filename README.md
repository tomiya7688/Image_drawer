# Image Drawer

Image Drawer の最上位目的は、**AI自身が実際にブラシで描き、描いた結果を評価して、絵を描く行動を学習すること**です。制作工程は**ラフ→下書き→本描き→基本色→陰影・仕上げ→修正・確認→完成**に固定しますが、人間の筆運びを模倣する必要はありません。

**学習入力の原則**

> 学習に必要な外部入力は画像のみとする。
> ユーザーおよび開発者は、画像以外の補助データを用意する必要はない。
> タグ・説明文・プロンプトは、必要に応じてシステム側で自動抽出する。

学習では画像から内部的に文章条件を生成して描画AIに渡します。生成時はユーザーが文章を入力します。学習用のTXT・タグ・キャプション・権利情報・制作履歴・追加メタデータの提出欄は必要ありません。[画像から文章条件を作る設計](docs/developers/image_only_learning_v1.md)を参照してください。

**主機能はtxt2img（Text-to-Image）です。** 推論では文章（prompt）だけを必須入力とし、AIが白紙から下書き・線画・着色・修正を描画行動として逐次実行して、完成画像を返します。**学習素材として用意するのは画像だけ**で、タグとpromptは利用者が選んだ抽出器が自動生成します。AIは白紙に描いた結果を目標画像と比較され、行動を改善します。推論時には参照画像・下書き・実演履歴を要求しません。操作列は再生可能な付随出力として記録します。

```text
現在の状態と制作履歴
    -> 次の描画行動を予測（色・筆・レイヤー・軌跡・修正など）
    -> 行動を実行してキャンバスを更新
    -> 結果を観察して次の行動を予測
```

色は**固定256色パレット**から選び、ブラシは色と独立した要素です。下書き・線画・着色は人間の制作上の工程を表し、**「着色＝一括塗りつぶし」ではありません**。

**初見の方は [設計ガイド](docs/developers/start_here.md) から読んでください。** 「画像だけで学習」「自動タグ・prompt抽出器を切り替えられる」「ラフ→下書き→本描きの順序固定」の全体像を説明しています。詳細は[コンセプト](docs/kadoka/concept.md)と[制作過程v1](docs/developers/drawing_process_v1.md)・[受け入れ試験](docs/developers/drawing_process_acceptance.md)。既存のPart Bank・COMPOSEは補助的な素材機能であり、筆による制作行動の学習の代わりではありません。

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

現在は基盤実装段階です。Workflow Runtime、DSL、Part Bankなどの既存機能がありますが、**制作行動の記録・完全再生・画像のみで描いて上達する学習が実装済みという意味ではありません**。[新しい実装計画](docs/kadoka/implementation_plan.md)に基づきIssue / Pull Request単位で進めます。

完成した画像制作GUIはまだ提供していません。利用可能なentrypointは [利用者向けの「はじめに」](docs/users/getting_started.md) を参照してください。

## 言語

プロジェクト独自の説明文は日本語を正本とします。詳細は [言語運用方針](docs/developers/language_policy.md) を参照してください。

Apache-2.0や第三者ライセンスなど、原文自体に法的意味がある文書は公式原文を正本とします。

## PR前のローカル確認

開発用依存を `python -m pip install -e ".[dev]"` で導入し、ルートで `python scripts/check_local.py` を実行してください。変更Pythonの構文・Ruff整形/lint、差分整合性、既存の文書検査・pytestを実行し、結果と未確認範囲を集約します。配布・GPU・実GUIの検証は別に行います。詳しくは[開発手順](docs/developers/local_checks.md)を参照してください。

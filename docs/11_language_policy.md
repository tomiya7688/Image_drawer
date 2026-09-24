# 言語運用方針

## 1. 正本

Image Drawerでは、プロジェクト独自の説明文について**日本語を正本**とします。

対象:

- README
- docs/ 配下の仕様・設計・調査文書
- 補助README
- Python docstring
- コードコメント
- Issue本文
- Pull Request本文
- 開発者向け運用説明

英訳を用意する場合は参考訳です。内容が食い違う場合は日本語版を優先します。

## 2. 英語のまま保持するもの

次は可読性・互換性・外部仕様との一致を優先し、必要に応じて英語を使用します。

- 変数名、関数名、class名、module名
- API名
- DSL keyword
- CLI command / option
- file/path名
- schema field名
- protocol名
- model名
- library / framework名
- 論文title
- 固有名詞
- log/exception messageのうち機械処理や外部互換性が必要なもの

説明文まで無理に英訳へ合わせる必要はありません。

## 3. ライセンス文書の例外

Apache-2.0本文、第三者library/model/datasetのlicense本文など、**原文自体に法的意味がある文書は公式原文を正本**とします。

これらを日本語で説明する場合は別の解説文書として作成し、翻訳がlicense原文を置き換えるものではないことを明記します。

## 4. Code comment / docstring

新規codeでは説明commentとdocstringを原則日本語で記述します。

推奨:

    def select_parts(...):
        """現在の構図と候補集合から使用するPartを選択する。"""

        # 同一Source由来の候補へ偏りすぎないよう、上限を適用する。
        ...

避けるもの:

- codeを読めば明らかな処理を逐語的に説明するcomment
- 日本語と英語で同じ内容を二重記述するcomment
- 実装とずれた古いcomment

commentは「何をしているか」より、必要に応じて「なぜそうするか」「どの制約があるか」を説明します。

## 5. 用語

技術用語は不自然に日本語化しすぎません。

例:

- Workflow
- Step
- Runtime
- Artifact
- Trajectory
- Evaluator
- Part Bank
- embedding
- checkpoint
- provenance

本文は日本語で書き、識別子や一般に定着した技術語は英語表記を許容します。

## 6. Issue / Pull Request

IssueとPull Requestの説明は原則日本語で記述します。

code identifier、error message、command output、引用する外部仕様などは原文のままで構いません。

## 7. Review

文書・commentの変更reviewでは、翻訳の自然さよりも次を優先します。

1. 日本語だけで仕様を理解できること
2. 技術的意味が変わっていないこと
3. identifierと外部仕様が正確であること
4. 正本と参考訳の関係が曖昧でないこと

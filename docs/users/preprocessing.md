> **対象読者：** 学習前処理を自分の環境で使いたい利用者
> **目的：** 画像だけを登録し、任意モデルによるCaption生成・タグ抽出・分類を設定する
> **状態：** 登録・CLI基盤を追加。実モデルの導入・GUI設定画面は別途実装する

# 画像だけで学習前の文章とタグを生成する

**学習用に用意するのは画像ファイルだけです。** TXT、タグ、説明文、制作履歴、補助メタデータは画像ごとに必要ありません。

画像は **①文章を作るCaptioner** と **②タグを作るTagger** にそれぞれ渡します。2つの処理は独立しており、一方の出力をもう一方の結果として扱いません。

## モデルを登録する

[設定例JSON](preprocessor_settings.example.json)をコピーし、使用したいモデル名と接続先を記入します。設定JSONは**アプリの設定**であり、学習画像ごとの追加素材ではありません。

- \`providers\`：Captioner・Tagger・Verifierを好きな名前で登録（最大256件）
- \`caption_chain\`：精度が高いモデルから、拒否が少なく一定の画像説明品質があるモデルへ、**任意の段数で**並べる。Qwenである必要はない
- \`tag_chain\`：Taggerの試行順。タグ側も複数モデルへフォールバックできる
- \`tag_normalizer\`：信頼度閾値・タグ分類・ユーザー定義カテゴリ
- \`tag_formatter\`：タグの重複除去・カテゴリ別整形・画風/作品名/rating/metaの扱い

文章側はOpenAI互換Vision APIとJoyCaption専用Adapter、タグ側はTagger Service HTTP/PixAI系HTTPの接続方式を提供します。外部モデルの導入と起動はユーザーの環境に応じて行います。

### 標準img2txtフォールバックの決定（2026-11-05）

**JoyCaption Beta One**（`fancyfeast/llama-joycaption-beta-one-hf-llava`）を標準フォールバックに採用しました。開発元がSFW/NSFW、写真・イラスト・アニメを画像説明の対象に明示し、Diffusion学習用キャプションへ特化しているためです。設定例の順序は`Qwen/Qwen3-VL-8B-Instruct`（構造化Captionの初段）→`JoyCaption Beta One`（低制限の代替）です。**この二つ以外にも登録・追加・並べ替えが可能**で、Qwenの利用も必須ではありません。

JoyCaptionは一般的なJSON指示への追従が安定しないため、`adapter: "joycaption"`がモデルに自然文を要求し、学習用の短文・標準文・詳細文へ変換します。元の文章にない対象や個数・位置関係を推測で補完しません。Taggerは引き続き独立実行します。

JoyCaptionのモデルサービスを起動する例（GPU・vLLMの準備が必要）：

```bash
vllm serve fancyfeast/llama-joycaption-beta-one-hf-llava --host 127.0.0.1 --port 8001 --max-model-len 4096
```

モデル作者の案内では、bf16モデル本体はVRAM約17GB、24GB以上のGPUが目安です。量子化で削減できますが、提供するAPIの互換性を別途確認してください。初段のQwenとTaggerも**それぞれ別に起動**する必要があります。モデルを自動ダウンロード・同梱する仕組みは未実装です。

**注意点：** 作者はJoyCaptionが複数対象・左右・OCRを取り違えることや、NSFWの一部で拒否する場合があることを明記しています。独立した比較ベンチマークは未実施で、拒否率や正確性の実測値を保証しません。配布コードのApache 2.0とモデルの基盤であるLlama 3.1の重みの利用条件は別途確認し、外部重みをImage_drawerへ同梱しません。

- [JoyCaption公式リポジトリ](https://github.com/fpgaminer/joycaption)
- [公式モデル配布](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)

## 画像を処理する

\`\`\`bash
image-drawer-preprocess --settings path/to/my-settings.json --image path/to/picture.png
\`\`\`

画像はPNG/JPEG/WebPを中身から自動判定し、結果をJSONで表示します。生成された短文・標準文・詳細文、整形済みタグと種類、各Tagのconfidence、利用したモデルとfallback履歴を確認できます。

例の\`REPLACE_WITH_...\`は実際のモデルではありません。使用するモデルサービスを起動して接続先を設定してください。現時点では単一画像を試すCLIであり、学習用Dataset/Trainerへの接続は別作業です。

## タグを種類別に整理する

例えばTaggerが以下を出した場合：

\`\`\`text
general: red_dress(0.94), standing(0.90), smile(0.91)
character: example_character(0.92)
copyright: example_series(0.91)
rating: general(0.80)
style: watercolor(0.93)
meta: absurdres(0.93)
\`\`\`

システムは服・ポーズ・表情・キャラクター・作品名・rating・画風・meta（画像解像度などの付帯情報）を分類します。信頼度と元の表記は残し、学習用に使うタグ列だけを整形します。**既定ではrating・作品名・画風・metaを肯定タグ列へ勝手に混ぜません。** `tag_formatter.include_meta`を`true`にした場合だけmetaを肯定タグ列へ含めます。

\`tag_normalizer.category_overrides\`で分類を上書きでき、\`extra_categories\`で独自カテゴリを作れます。Captionerが作る自然文とは別の結果として管理します。

## 独自の前処理も登録できる

開発者は[レジストリ](../../image_drawer/prompting/registry.py)の\`register_caption()\`、\`register_tagger()\`、\`register_verifier()\`、\`register_normalizer()\`、\`register_tag_formatter()\`、\`register_prompt_formatter()\`を使って独自のモデルや処理を追加できます。

新しいAPI形式は\`register_adapter()\`で信頼できるPythonコードのFactoryを追加できます。設定JSONから勝手にコードをimport/evalしません。

詳細設計：[Caption・Tagの独立実行と任意N段フォールバック](../developers/prompt_extraction_v1.md)。

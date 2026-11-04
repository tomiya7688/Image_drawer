> **対象読者：** はじめてImage Drawerの学習前処理を読む人、VLM・Tagger・学習Dataset・評価の実装者
> **目的：** 画像しか用意しなくても、人間が生成時に書くプロンプトと整合する文章条件を自動生成する。画像説明モデル・タグ抽出器・分類器を自由に登録し、失敗時は設定した順序で何段でも切り替える。
> **状態：** プロンプト抽出の**ルーティング基盤と契約テストは実装対象**。実モデルの導入・正答率・画像から最終イラストまでの学習効果は**未測定**。

# 学習前の画像→高精度Caption生成＋独立Tag抽出・分類・整形

## 1. 利用者が準備するもの

> 学習に必要な外部入力は画像のみとする。
> ユーザーおよび開発者は、画像以外の補助データを用意する必要はない。
> タグ・説明文・プロンプトは、必要に応じてシステム側で自動抽出する。

画像だけを登録すれば、ソフトの内部でプロンプトを作る。TXT、タグ、キャプション、制作履歴、追加メタデータをユーザーに提出させない。モデルの接続先や使用モデルはアプリの**設定**であり、画像1枚ごとの追加学習素材ではない。

このプロンプトは**txt2imgの重要な教師条件**。画像に「青い髪の少女が2人、左に犬」と写っているのに「少女のイラスト」だけを教師にすれば、生成時に「青い髪」「2人」「左」の意味を描画AIが学びにくい。**画像に忠実であること**と**人が普段使う文章に近いこと**の両方を目標とする。

## 2. 一枚の画像から必ず別々に生成するもの

```text
学習画像のみ
 ├─ Caption生成（高精度モデル → 低拒否率モデル → 任意の追加モデル）
 │     ↓
 │   主体・色・数・位置関係を抽出し、必要なら独立VLM照合
 │     ↓
 │   人間の短文／標準文／詳細文をそれぞれ生成
 │
 └─ Tag抽出（Tagger A → Tagger B → 任意の追加Tagger）
       ↓
     一般／キャラクター／作品／rating／style分類
       ↓
     衣服・姿勢・表情・背景・状況などへ分類・重複除去
       ↓
     confidenceで選別、ratingなどは本文へ暗黙追加しない
            ↓
自然文Caption と 分類・整形Tag を両方保存（別々の成果物）
            ↓
CaptionをTextEncoder → DrawingPolicy
            ↓
固定7工程で白紙から描き、Evaluatorだけが目標画像で評価
```

**抽出器から得るのは元画像からの説明であり、人間の過去の筆跡ではない。** DrawingPolicyは正解画像を観測しない。

### 例：一枚のイラスト

見本に「赤いリンゴが一つ、白い背景に置かれた絵」がある場合、内部では以下を生成する。

| 内容 | 出力例 |
| --- | --- |
| 対象 | リンゴ、個数1、赤色 |
| 背景 | 白 |
| 構図 | 中央 |
| 短いプロンプト | 赤いリンゴ |
| 普通のプロンプト | 白い背景に赤いリンゴが一つ |
| 詳しいプロンプト | 中央に一つの赤いリンゴ、白背景のシンプルなイラスト |

同じ画像に対して別のTaggerも実行し、`apple`等のタグをカテゴリ・confidence付きで整形する。**CaptionerのJSON内に入っているtagsだけでTagger処理を済ませてはいけない**。

この3文は**画像にある事実をどこまで詳しく言うかだけが違う**。詳しい方で勝手に「透明な皿」「葉の水滴」などを追加しない。

## 3. なぜ1つの長いキャプションに固定しないか

[DALL·E 3のrecaptioning研究](https://cdn.openai.com/papers/dall-e-3.pdf)と[RECAP](https://arxiv.org/abs/2310.16656)は、画像の説明をより正確で詳しいものに置き換えることが、テキスト指示への忠実度の改善につながると報告した。

一方、[2025年のsynthetic captions比較研究](https://ml-research.github.io/human-centered-genai/projects/t2i-synthetic-captions/index.html)は、詳細な説明文ばかりだと画質や多様性が犠牲になる場合があり、**説明の長さを変えて学習させる方法**が有効と報告した。

したがって最初は次の**仮の分布**を使う（実測済みの最適比率ではない）。

| 学習時に選ぶ説明 | 割合 |
| --- | ---: |
| 人間が入力しそうな短文 | 40% |
| 色・個数・関係を含む普通の文章 | 40% |
| 背景・構図・画風まで説明した詳細文 | 20% |

学習時は同じ画像から抽出した3候補を保存し、seedでどれを使ったか再現できるようにする。**生成時にユーザーが短文を入力しても使えること**を未見のプロンプトで評価する。色・個数・位置・否定を消すような過度な正規化はしない。

## 4. 精度向上のため何を構造化するか

出力を単なる一行の文章にせず、**次の情報を分けて取り出す**。

| 項目 | 何を検証するか |
| --- | --- |
| 主な対象 | 何が描かれているか |
| 個数 | 一人か二人か、1個か複数か。分からないときはnull |
| 色・見た目 | 対象と色が結び付いているか |
| 位置関係 | 左/右、前/後、上/下を取り違えないか |
| 背景 | 場所、物体、背景色 |
| 構図 | 全身・上半身、視点、遠近 |
| 画風・質感 | 線画、厚塗り、水彩など観測できる特徴 |
| タグ | 内容タグなど。Ratingと混ぜない |

[ComfyUI Support ToolのTagger/image-to-text移行設計](https://github.com/tomiya7688/comfyUI_support_tools/blob/main/doc/gui/tagger_image_to_text_migration.md)を参考に、**タグ、caption、style、ratingを同じ一つの文章に無条件で混ぜない**。別のTaggerを使う場合もカテゴリー・信頼度を区別し、モデルの自動タグが間違っている可能性を残す。

[CVPR 2026のCCCaption](https://openaccess.thecvf.com/content/CVPR2026/html/Tang_CCCaption_Dual-Reward_Reinforcement_Learning_for_Complete_and_Correct_Image_Captioning_CVPR_2026_paper.html)の考え方を参考に、**Completeness（重要情報の取りこぼし）とCorrectness（画像にない情報の捏造）を分けて評価**する。

現在のコードでは構造化JSONの検査と、**設定された独立VLMによる支持／不一致の再照合**を提供する。ただし照合モデル自身も間違えるので、`verified_by_model`は真実の保証ではない。実画像の正答率・検証器間の偏りは実測で報告する。

## 5. 任意N段のモデルルーティング（Qwen固定ではない）

**どのモデルを何個登録するかはユーザーが決める。** 少なくとも文章生成に1モデル・Taggerに1モデルが必要。2段階という上限も、Qwenという名称も決め打ちにしない。

Captionerは最初に**精度が高いモデル**、次から**拒否が少なく画像を十分に説明できるモデル**を並べる。例：

\`\`\`text
Caption: 高精度VLM → 幅広い画像に対応するVLM → 別のローカルVLM → …
Tags:    Tagger A → Tagger B → Tagger C → …
\`\`\`

- **実際にどのモデルが拒否しにくいか**は、ユーザーの運用環境と検証結果による。Qwen系は候補の一例であり「無制限」とは保証しない。局所の「拒否が少ない」モデルを置くのは性能と画像対応範囲のバランスを取るため。
- 文章生成モデルは\`refusal\`・\`content_filter\`・空出力・不正JSON・重要な画像事実の誤り・検証器拒否などを区別。後続モデルがあれば**次の登録モデル**を試す。
- Taggerにも独立したモデル列がある。タグ側の\`refused\`・空応答・不正応答・一時障害時は次のTaggerへ移る。
- 認証・API設定不備、入力画像不正を「拒否」と偽らない。モデルが途中で成功したら**その系統の残りのモデルは呼ばない**。
- 途中まで成功していても両処理が完了しなければ、例全体を成功扱いしない。失敗した側だけでなく他方の実行結果も記録する。
- すべてのモデルが失敗しても「画像」「イラスト」などのダミープロンプトやタグを作らず、画像IDと失敗理由を保存する。ユーザーに補助ファイルを要求しない。

独立したVerifierを登録できる。候補に見えない事実を足していないか・重要事実が欠けていないかを画像と照合する。ただしVerifierも誤るので\`verified_by_model\`は絶対的正解ではない。**最後のCaptionモデルが有効な文章を生成し、Verifierだけが拒否した場合**は\`verifier_refused_unverified\`として未照合扱いで保持する。

### 5.1 標準フォールバックモデルの調査・決定（2026-11-05）

学習用画像説明の低拒否率モデルとして、**JoyCaption Beta One**（`fancyfeast/llama-joycaption-beta-one-hf-llava`）を標準の**img2txtフォールバックに採用**する。初段の標準設定例には`Qwen/Qwen3-VL-8B-Instruct`を置くが、モデル名や切替段数は固定しない。

| モデル | 調査した根拠・制限 | 判断 |
| --- | --- | --- |
| **JoyCaption Beta One** | 作者がSFW・NSFW双方と写真・アニメ等の幅広い画像への対応を明示。Diffusion学習用の文章生成に特化。自然文で応答し、複数対象・左右・OCRに既知の誤り、まれな拒否あり | **標準フォールバック** |
| `Felldude/Qwen3-VL-8B-Instruct-Uncensored-V2` | Apache 2.0表記のNSFW向け調整モデル。作者は画像を動画として説明する不具合を報告 | ユーザー追加候補 |
| `GitMylo/nsfwcaption-qwen3-vl-8b-v2-safetensors` | 作者はキャプション用途の低拒否率を明記。ただし人物やキャラクター名を取り違える旨を報告 | ユーザー追加候補 |
| `Qwen/Qwen3-VL-8B-Instruct` | 汎用画像理解とOpenAI互換推論サービスに対応。Apache 2.0のモデルカードあり | 構造化Caption初段の標準例 |

**出典（モデル開発元）**：
[JoyCaption公式](https://github.com/fpgaminer/joycaption)・[公式モデル](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava)・
[Felldude V2](https://huggingface.co/Felldude/Qwen3-VL-8B-Instruct-Uncensored-V2)・
[GitMylo V2](https://huggingface.co/GitMylo/nsfwcaption-qwen3-vl-8b-v2-safetensors)・
[Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。

**選定の意味**：モデル作者が明記した学習目的・対応範囲・既知の制約を根拠に、採用モデル名まで決めた。ただし画像を使った独立の精度比較や拒否率実測は行っていない。JoyCaptionが無拒否であるとの主張や、他候補を数値で下回る／上回るといった未測定の断言はしない。

**実装上の必須差分**：JoyCaptionはJSON指示に安定して従う汎用VLMではなく、作者推奨のStraightforward Captionで自然文を返す。このため`image_drawer.prompting.joycaption.JoyCaptionVisionProvider`が自然文を取り出し、文章そのものから短文（冒頭1文）・標準文（冒頭2文）・詳細文（全文）を決定的に組み立てる。元の説明文にないsubjects・relations等を想像して埋めず空配列として保持し、別Taggerの結果を流用しない。日本語と英語の学習条件の混在はTextEncoderとの整合を別途評価する。

**実行環境・権利**：JoyCaptionの公式案内はbf16でVRAM約17GB、24GB以上のGPUを推奨し、vLLMによるOpenAI互換サーバーも示している。モデルカードに`meta-llama/Llama-3.1-8B-Instruct`が基盤として記されている。JoyCaptionコードのApache 2.0とモデル重みの許諾条件は混同せず、導入時に基盤モデルの条件を確認する。Image_drawer本体へモデル重みは同梱しない。

## 6. タグの種類・分類・整形も独立した必須工程

**Caption生成とTag抽出は必ず両方実行**する。文章生成器が\`tags\`を返したからといって、独立Taggerの推論を省略しない。

ComfyUI Support Tools の[Tagger正規化実装](https://github.com/tomiya7688/comfyUI_support_tools/blob/main/src/comfyui_support_tools/shared/tag_result_normalizer.py)と[カテゴリ分類器](https://github.com/tomiya7688/comfyUI_support_tools/blob/main/scripts/backend/tag_category_splitter.py)を参考に、次の分類を使う。

| 分類 | 主な内容 | 初期の扱い |
| --- | --- | --- |
| \`general\` | 対象・色など一般に観察できる特徴 | 内容タグとして保持 |
| \`character\` | 固有のキャラクター名 | 専用confidence閾値 |
| \`copyright\` | 作品名・シリーズ名 | 保存するが肯定タグに自動追加しない |
| \`rating\` | モデルの評価ラベル | **自然文には混ぜず独立保存** |
| \`style\` | 水彩、線画、アニメ調など | 別カテゴリに保存。本文への統合は選択式 |
| \`meta\` | 解像度・由来等の付帯タグ | 別カテゴリに保存し、肯定タグへの統合は既定で無効 |
| \`clothes\` / \`pose\` / \`expression\` | 服・姿勢・表情 | \`general\`の一般タグから必要な分だけ分類 |
| \`background\` / \`situation\` | 背景・時間帯・状況 | 表現上の特徴として分類 |
| ユーザー独自カテゴリ | 例：髪色・照明・小物 | 設定にカテゴリ名と個別ルールを追加 |

Tagger応答は**PixAI風のフラットなタグ辞書、カテゴリ付きJSON、文字列やscore付きタグ列**を正規化する。余分なスペース・重複・confidenceのしきい値を処理し、元のラベルと元スコアを保持する。下流へ渡すタグ名だけ可読化する（\`blue_dress\`→\`blue dress\`）。**顔文字は文字列を壊さず保持**する。低信頼の固有キャラクター名は別の高い閾値で除く。

\`rating\`、\`copyright\`、\`style\`、\`meta\`の扱いは個別設定で変えられる。**Tagger結果のRatingを「画像に描かれた対象」の説明へ黙って混ぜない。** 画像から説明文を作るVLMと、タグ抽出器が出した特徴が矛盾する場合も、単純連結で誤情報を増やさず別々に記録して比較可能にする。

## 7. 利用者によるモデル・前処理の自由登録

利用者向けの設定方法は[学習前処理の設定](../users/preprocessing.md)、[設定例](../users/preprocessor_settings.example.json)を参照。

\`PreprocessorRegistry\`が次の独立した登録を提供する。

- \`register_caption\`：任意の画像説明モデル
- \`register_tagger\`：任意のタグ抽出モデル
- \`register_verifier\`：画像とCaptionを比較するモデル
- \`register_normalizer\`：カテゴリ分類・confidence正規化の処理
- \`register_tag_formatter\`：タグの整形・必要カテゴリの出力
- \`register_prompt_formatter\`：Caption/タグを使う学習条件生成の独自処理
- \`register_adapter\`：追加したい独自API・モデル形式の接続処理

**登録数も優先順位も固定しない。** 同じモデルを重複登録して無限に回ることは防ぎ、CaptionとTaggerは別の配列に登録する。ユーザーは接続先・model ID・優先順位・カテゴリルール・信頼度・整形設定をアプリ設定として記録する。追加の学習素材ファイルを要求しない。

JSON設定から登録できる標準接続は**OpenAI互換Vision API**と**Tagger Service/PixAI HTTP**。ローカルモデルでもWeb APIでも、画像を処理できる接続であれば登録可能。未知APIは信頼できるプラグインPythonからFactoryを登録する。**設定JSONそのものを任意コードとして実行しない**。APIキーは環境変数名で指定し、学習ログへ保存しない。

\`\`\`bash
image-drawer-preprocess --settings my-settings.json --image picture.png
\`\`\`

このコマンドでは**学習画像を1枚指定するだけ**で、CaptionとTaggerを別々に走らせた結果とモデル採択履歴を取得できる。設定JSONはアプリ全体の設定で、画像ごとの追加入力データではない。

## 8. txt2imgの学習条件と評価

Captionは短文40%、標準文40%、詳細文20%の**仮の初期比率**で使用。Tag結果は独立したカテゴリ別特徴として保存し、タグ式の条件で学習するかは別途Trainer側で選択する。**自然文とタグ列の意味が矛盾する場合は単純に結合して学習しない。**

生成時に人間が入力した文章と、学習時の自動文章は**同じTextEncoder・正規化**で解釈する。例えば「赤い髪」「二人」「右」「影なし」を学習・推論のどちらでも保持する。

受け入れ試験では、モデルを選ぶ設定を変更したときのキャッシュ識別、4段以上のモデル切替、タグ整形とカテゴリ別表示、Primary成功時の後続スキップ、全部失敗したときの理由記録を必須とする。

実画像の精度は以下を**別々に測る**：Captionの対象・色・個数・位置関係、誤記述率、タグのカテゴリ別Precision/Recall、Characterタグの誤検出、推論モデルの拒否率・Fallback率、短文/詳細文の生成指示への追従、試行コスト。比較結果がないモデルを「無制限」「最も精度が高い」とは宣言しない。

## 9. 実装済み・未実装

**実装済み**：任意N段のCaptionモデル切替、独立TaggerとそのN段切替、Comfy系タグの読み取り・カテゴリ分類・整形、登録レジストリとユーザー設定JSON、OpenAI互換Vision APIおよびTagger Service/PixAI HTTPアダプタ、1枚を処理するCLI、モックと静的契約テスト。

**未実装**：実VLM/Taggerの導入・起動・モデルごとの精度比較、設定を操作するGUI、大量画像Datasetの収集・キャッシュ永続化、TextEncoderとTrainerへのEnd-to-End接続。CIに成功しても実画像での学習精度改善を主張しない。

**固定する元の製品仕様**：ユーザーが用意するのは画像だけ。学習中は正解画像をDrawingPolicyへ渡さない。生成時はテキストのみから、固定7工程・256色・12ブラシでAI自身が実際に描く。

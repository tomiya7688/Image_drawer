> **対象読者：** Dataset前処理、TextEncoder、DrawingPolicy、Trainer、CIの開発者
> **目的：** Dataset全体への3パス処理と画像から文章条件、白紙描画学習までの境界を確定する
> **状態：** 前処理・TextEncoderのAPIと単体テストを実装。TrainerとのEnd-to-End学習は未完成。

# Datasetの3段前処理 → TextEncoder → 自己描画学習

## 1. 学習画像からプロンプトを作る（時間順）

```text
Dataset（画像のみ、重複をグループ分割）
       │
       ▼
① 全画像に Tagger を適用  ──> タグ/分類/confidence/失敗理由を永続化
       │
       ▼
② 全画像に Captioner を適用 ──> 短文/標準文/詳細文/採用モデルを永続化
       │
       ▼
③ PromptComposer で合成・整形 ──> 重要タグと自然文の重複・明白な矛盾を除外
       │                          補足未採用タグも理由つきで保存
       ▼
完成した自然文プロンプト（CaptionとTagの双方が必要）
       │
       ▼
④ TextEncoder（ユーザー選択、GPU→同一モデルCPUのフォールバック）
       │
       ▼
text_condition（固定次元の特徴ベクトル）
       │
       ▼
DrawingPolicy（自分の白紙キャンバスとAction履歴）
       │
       ▼
固定7工程の実Stroke → 最終画像
       │
       ▼
Evaluatorのみがtarget画像と比較 → Reward → TrainerがPolicy更新
```

**①→②→③は登録済みDataset全体について独立したパスとして行う。**
以前の画像ごとにCaption→Tagへ実行する前処理とは異なる。`image-drawer-dataset register`で
元画像登録を行った後、次の新しいCLIから実行する。

```powershell
image-drawer-preprocess --settings docs/users/preprocessor_settings.example.json --dataset E:\datasets\my-drawings --seed 11
```

`image_drawer/prompting/dataset_pipeline.py`が3パスを実行し、各画像で
`cache/preprocessing-v2/<profile_hash>/{tags,captions,prompts}/`へ原子的にJSONを保存する。
一時的な失敗は次回再試行し、**成功した段階だけを再利用**する。
③まで成功した画像だけが`training_prompt`を持つ。失敗時に旧ラベルを残さず、
`DatasetManifest`のハッシュ、抽出器バージョン、splitは保全する。
失敗件数は分母込みで報告し、画像ごとに手作業のTXTを要求しない。

### 3段目の合成規則

`PromptComposer`はCaptionとタグを無条件に連結しない。タグの
confidence（暫定0.85以上）とカテゴリで判定し、すでにCaptionにある語は重複させない。
色/姿勢の既知の矛盾は保留し、rating/copyright/style/metaを肯定文へ黙って入れない。
日本語Captionと英語Tagが混ざる場合、**辞書にないTagを不自然な英語文字列として
追記せず保留**する。採用・保留と理由を`included_tags/withheld_tags`へ保存する。
これらは**ルールベースの暫定baseline**であり、全言語の意味矛盾を判別できるものではない。
実画像の調整精度は別途測り、必要なら登録可能な`PromptFormatter`へ差し替える。

## 2. TextEncoderとフォールバックの意味

TextEncoderは自然文をベクトルへ変換するだけで、画像を解説するVLMではない。
通常のローカル埋め込みモデルは安全ポリシーによる文章生成拒否を行わない。
失敗として扱うのは、GPU OOM、モデル未導入、読み込みエラー、不正な次元/NaN/ゼロベクトル。
**フォールバックの初期設定はモデルを変更しない**。

| 設定項目 | 通常8GB |
| --- | --- |
| モデル | [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) |
| 出力次元 | 384 |
| 第1候補 | 同じモデルのCUDA実行 |
| 第2候補 | **GPUモデルをrelease→キャッシュ解放してから**CPU実行 |
| 永続化 | model ID、モデルrevision、feature_space_id、dimension、採用デバイス、各試行失敗 |

`TextEncoderRegistry`で独自Adapterを登録でき、設定JSONから`text_encoder.providers`を
自由に並べ替えられる。ただし、**異なるモデルやrevision、次元を同一checkpointの
フォールバックに混ぜることは不合格**。見かけ上どちらも384次元でも意味空間が
別なら学習結果が壊れる。別のEncoderを使う場合は新しいcheckpointとして学習し直すか、
あらかじめ学習した投影Adapterを使い互換性を検証する。

モデルrevisionは実際のcommit SHAへ固定して実験を再現すること。
学習と生成は同一の`TextConditionEncoder`/正規化ルールを使い、
checkpointの`feature_space_id`と次元が一致しない場合は停止する。

動作確認のCLI：

```powershell
python -m pip install -e ".[text-encoder]"
image-drawer-text-encode --settings docs/users/preprocessor_settings.example.json --prompt "赤いリンゴが一つ"
```

開発用baselineとして`hashed_ngram`（64次元、追加モデル不要）も登録可能だが、
言語意味の精度はTransformerと同等ではなく、**別の意味空間**として扱う。

## 3. 8GB VRAMと自動メモリ管理

`llama.cpp`ルーターの`--models-max 1 --models-memory-margin 2048`で
Caption/Tag用画像モデルを1つずつオンデマンドでロード/アンロードする。
`--fit on --fit-target 2048 --gpu-layers auto`でGPU層数を自動調整し、
`--no-mmproj-offload`で画像投影器をCPU配置する。
**「モデル解放」と「GPU→CPUへ一部層を配置」は別々の操作**として両方設定する。

VRAMが不足しても`llama.cpp`の自動fitで常に回復できる保証はない。
画像用mmprojの推定が不足する場合があり、実機でピークVRAMを確認する。
TextEncoderは画像前処理3パスの後、GPUで動かなければモデルを解放して
**同じモデルのCPU実行**へ切り替える。CPU fallbackは32GB RAMの範囲で
遅くなる可能性があるが、意味空間を変更しない。

## 4. Trainerへ渡せるものと未完成の境界

TrainerのPolicyが読むのは`prompt→text_condition`と**自身の描画状態だけ**。
正解画像のピクセル/特徴量/差分/Rewardは同一rolloutのPolicyの入力へ渡さず、
Evaluator/Trainerだけが参照する。

今回の実装はDataset3パス・Composer・TextEncoder/互換性検証・CLIまで。
`image_drawer/training/`の学習本体がまだ存在しないため、
「学習前後で実際に絵を描いて改善した」とは主張しない。
学習器を接続するときは[Trainer設計](design_trainer.md)に従い、
学習/推論でTextEncoderが一致するテストと、8GB VRAM実機での学習メモリ計測を追加する。

> **対象読者:** 文章から絵を描かせたい人
>
> **目的:** `image-drawer-draw`の使い方を示す。
>
> **状態:** 配線（水準A）まで。学習済みモデルの品質は評価されていません。

# 文章だけから描く（image-drawer-draw）

リポジトリのルートで `python -m pip install -e ".[train]"` を実行してから使います。画像や下書きは入力に取りません。

```powershell
# 文章から描く（既定のPolicyは未学習の random-legal-v1。絵の品質は評価されていません）
image-drawer-draw --prompt "赤いリンゴが白い背景に一つ" --policy random-legal-v1 --out-dir out/apple --seed 7

# 保存したEpisodeを再生する（工程ごとの画像も保存）
image-drawer-draw --replay out/apple/episode.json --out-dir out/apple-replay --frames stages

# 登録済みPolicyの一覧
image-drawer-draw --list-policies
```

| 引数 | 内容 |
| --- | --- |
| `--prompt` / `--prompt-file` | 描く内容。ファイルはUTF-8 |
| `--policy` | 登録済みPolicyのID |
| `--max-actions` | 受理するActionの上限（1〜19900、既定1000） |
| `--seed` | 乱数seed（0〜4294967295）。同じ条件なら同じ絵になる |
| `--width` / `--height` | 既定はPolicyのcanvas。違う値を指定するとエラー |
| `--out-dir` / `--overwrite` | 出力先。空でない先は`--overwrite`が必要（既知のファイルだけ上書き） |
| `--checkpoint ID=PATH` / `--train-config PATH` | 学習済みcheckpointをPolicyとして登録 |

出力は`episode.json`（完成時）、`image.png`、`attempts.json`です。途中で終わった場合は`episode.json`ではなく`draft.json`を保存します。

| 終了コード | 意味 |
| ---: | --- |
| 0 | 完成した、または再生が一致した |
| 1 | 未完成（予算切れなど）。`draft.json`と画像を保存済み |
| 2 | 実行できない（引数、未登録のPolicy、canvas、出力先、入出力） |
| 3 | 再生のみ。Episodeの検証失敗、`--expect-image`との不一致 |

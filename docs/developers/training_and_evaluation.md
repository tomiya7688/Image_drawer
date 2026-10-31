> **対象読者:** 学習器・評価器・データ収集器を実装する開発者
> **目的:** 「制作過程を学習した」という主張を検証可能にし、画像品質評価と混同しない。

# 学習と評価：DrawingPolicyを学ぶ

## 1. 何を学習するのか

主対象は**次の描画行動の予測**。

```text
入力:  prompt + DrawingState(t) + ActionHistory(<t)
予測:  DrawingAction(t)
適用:  DrawingEngine.apply(state, action) -> DrawingState(t+1)
次へ:  新しい状態で同じpolicyを繰り返す
```

`DrawingAction`の型と意味、固定256色、レイヤーとブラシ、描画計算は [制作過程 v1](drawing_process_v1.md) を参照する。この文書では再定義しない。

**学習の成功と呼ばない例**：

- 画像だけを見て色を塗りつぶす変換器。
- 固定されたストローク列を毎回再生するだけのScript。
- 画像のPartやWorkflow templateをランキングするだけのモデル。
- 人間の制作履歴ではなく、完成PNGから逆算して生成した疑似行動だけを用いた結果。

これらは補助技術・smoke testには使えるが、主目的の達成には該当しない。

## 2. Dataset作成

データ単位は`DrawingEpisode`。**保存された全イベントを最初から順に再生**し、各時点で`state_before`、`history_before`、`target_action`を生成する。

| 分類 | 用途 |
| --- | --- |
| `origin=human` | **出所と利用権限を確認できたものだけ**、人間の制作工程の学習と評価に使用 |
| `origin=model` | モデルの生成行動を観察・比較。人間の教師として扱わない |
| `origin=synthetic` | 描画器・Dataset builder・checkpointの自動テスト用 |
| `origin=derived` | 完成画像等から推定された行動。人間が実際に実施した操作として扱わない |

作成時に`episode_id`、出所・許諾、作品または収集セッションのID、palette/brush/renderer version、prompt、イベント列を保持。作品の途中までをtrainとvalidationへ跨らせない。重複・派生系列も同じsplitに配置し、データ漏洩を防ぐ。人間の実演が確保できなければ学習成果を宣言しない。

`state_before`へ完成キャンバスや将来イベントを混入させてはならない。時刻`time_ms`は記録するが、未来の所要時間を特徴にしない。

## 3. 最初の学習方法

初期ベースラインは**Behavior Cloning（教師あり模倣）**。

予測対象は1回の`DrawingAction`を構成する型とパラメータ全体。

- `type`：`create_layer`、`select_layer`、`set_phase`、`select_color`、`select_brush`、`stroke`、`erase`、`finish`。
- `color_id`：**固定256クラスの分類**。色番号とRGB値の対応はすべてのrunで同じ。
- `brush_id`：色とは独立。v1では`hard_round_v1`のみ（ブラシの選択はActionとして残す）。
- `points`、`width_px`、`opacity_u8`、`layer_id`、`phase`、`finish`は、それぞれAction typeに必要な場合に予測する。
- 人間の未来行動を直接入力してはいけない。学習時は教師Actionをtargetとして使い、推論時はモデルがActionを選ぶ。

**モデルの内部構造**（Transformer、RNN、別の模倣学習器等）は別Issueで比較・決定する。どの実装であっても**同じDrawingActionを出力**しなければならない。予測が契約違反なら`DrawingEngine`が拒否し、失敗をログに残す。仕様外の色や「代わりにPNGを生成する」回避は禁止。

## 4. 2種類の評価を必ず分離する

### 4.1 教師強制による次行動の予測（offline）

`state_before`と`history_before`を入力して、実演の次Actionを予測する。

- Action typeの正解率・混同行列（全イベント対象）。
- `select_color`のみ色番号top-1正解率（出現しないときはN/A）。
- `select_brush`のみbrush選択正解率（v1は1種類であり、難易度が低いことを明記）。
- `stroke/erase`の点列長・位置・線幅・不透明度に関する誤差をパラメータ別に記録。
- レイヤー操作・phase変更・`finish`の正誤を個別に記録。

未来情報を入力に混ぜて正解率を上げた実験は無効。作品単位で分割し、比較したDatasetバージョンとcheckpointを残す。

### 4.2 モデル自身のclosed-loop描画（online rollout）

`DrawingPolicy`が自分の予測した1Actionを`DrawingEngine`へ渡す。次の予測入力は**その結果のState**。人間の実演から次の一手を抜き出して強制適用するテストで代用しない。

測定：

- 有効Action率、拒否されたActionと原因、レイヤー／色／ブラシの使用履歴。
- 最後に`finish`まで到達した割合、最大Action数超過・例外率。
- 制作途中のStateをどこまで再生・検査できるか。
- 中間画像と完成画像の品質・人間評価（自動数値と区別して記録）。
- Action選択と最終品質の関係（同じprompt/seeds・比較条件を保存）。

**推論画像が見栄え良くても、行動履歴がない／再生できないなら不合格。**

## 5. 評価器と選択器の位置づけ

`EVALUATE`、`SELECT`、`ScoreSet`は引き続き使う。Sketch、line、color、finalの各段階の画質や選好を評価してよいが、以下を守る。

- **画像品質のScore**と**描画行動予測のScore**を別の名前・評価器versionで保存。
- 各Scoreのcomponentを潰さず保持し、aggregate方式・weightを記録。
- `SELECT`が画像候補を選んでも、`DrawingPolicy`の次行動予測を学習したことにはならない。
- 人間A/B評価は評価者の判断として保存する。実際に描いたActionとは別のデータ。
- 最終的な「制作過程を学習できた」判定は、描画行動の教師予測とclosed-loopの両方に根拠があること。

## 6. 再現性・機密性とDone

学習runには`dataset_id/split`、`schema_version`、`palette_id`、`brush_set_id`、`renderer_version`、`model/checkpoint_id`、`seed`、学習設定、評価器versionを記録する。モデルのAction出力と拒否／適用結果は`DrawingEpisode`またはrunの関連ログに残す。個人データ・権利のない実演を勝手に学習へ投入しない。

**Done条件**は [受け入れ試験 F](drawing_process_acceptance.md) と [実装計画 S3](../kadoka/implementation_plan.md) に従う。synthetic fixtureだけの模倣学習テストに成功した場合は、**「学習系のテストは成功。実演学習は未検証」**と明記する。

## 7. 後続研究

十分な人間の実演と安定した行動契約が揃った後、モデル比較、action preference、self-correction、Action policyのRL、Workflow Search、Part Bankの選択学習などへ拡張する。上位Workflow最適化の成功を、基礎となる「筆で描く過程」の学習の代わりにはしない。

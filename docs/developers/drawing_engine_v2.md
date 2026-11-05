> **対象読者:** DrawingEngine・StageController・Policy・テストの実装者
> **目的:** S1a（Issue #66）で実装した決定的DrawingEngineの公開APIと、設計書が未確定だったv2の数値カーネルの決定を記録する。
> **状態:** 実装済み（`image_drawer/drawing/`）。画素goldenは**回帰用**であり、画質の良し悪しを主張しません。

# DrawingEngine（S1a）と brush-raster-v2 の数値カーネル

## 1. 範囲

アプリはモデルなしでActionを1件ずつ適用し、同じAction列から必ず同じ画素を生成します。StageController（工程順の強制）、Episodeの永続化とReplayer、Policy、Trainerは対象外です。StageControllerは`Engine.apply()`を包んで利用できます。

| 項目 | 内容 |
| --- | --- |
| `Engine(width, height, profile, seed)` | `profile`は`raster-v1`（旧互換）または`brush-raster-v2`。`seed`はgrain用の固定値（0〜2^32-1） |
| `Engine.apply(DrawingAction)` | 検証してから状態を変更します。不正なActionは`DrawingError`（`ValueError`の派生）で、状態は変わりません |
| `Engine.render()` | 可視レイヤーを白背景へ合成した`RenderedImage`（RGBA8、alpha=255） |
| `Engine.layer_rows(layer_id)` | レイヤー単体のRGBA画素 |
| `DrawingState` / `DrawingAction` / `DrawingEpisode` | `to_dict`/`from_dict`でJSON化できるデータ型。`loads_strict`は小数・指数・NaN・重複キーを拒否します |

v1のAction（`create_layer`/`select_layer`/`set_phase`/`select_color`/`select_brush`/`stroke`/`erase`/`finish`）は`drawing_process_v1.md`のとおりに処理し、`tests/fixtures/drawing_process_v1`の全goldenと画素一致します。

v2のAction: `create_layer{layer_id,name,role}`、`select_layer`、`select_color`、`select_brush`、`set_brush_params{size_level_id,opacity_level_id}`、`stroke_v2{samples}`、`erase_v2{samples}`、`set_layer_visibility{layer_id,visible}`、`finish`。v1とv2で受理する種別は混ぜません（例: v2は`set_phase`と`stroke`を拒否します）。S1bで、v2に工程マーカー`enter_stage{stage_id}`・`exit_stage{stage_id}`・`advance_stage{}`または`{review_reason}`・`clear_active_layer{}`と、`Engine.apply_group`（1〜64件を全件成功か全件不変で適用。画素を変えるActionは不可）を追加しました。マーカーは画素を変えず、工程の順序は判定しません（順序の強制は`image_drawer/drawing/stage/`のStageController）。v2の`finish`は工程設定済みなら`stage_id`を`finish`にします。`DrawingAction`の`stage_id`・`group`は注釈で、`to_dict`には含めません。

## 2. v2の数値カーネル（本書で決めた式）

座標は`q8`（1/256px）で、画素`(x,y)`の中心は`(256x, 256y)`です。計算は整数のみ（`math.sqrt`や浮動小数は使いません）。`round(a/b)`は`floor((2a+b)/(2b))`です。

1. **段階の展開**: サイズ`size_q8 = size_px*256`。不透明度・筆圧`u16 = floor((percent*65535+50)/100)`。筆圧が`null`のときは100%（65535）。
2. **筆圧係数**: `f = min + round((65535-min)*pressure_u16/65535)`。径は`round(size_q8*f_size/65535)`（下限1）、alphaは`round(opacity_u16*f_opacity/65535)`のあと`round(·*flow_u16/65535)`。
3. **スタンプ配置**: 間隔`spacing = max(64, round(size_q8*spacing_q16/65535))`（q8、基準径は筆圧に依存しません）。折れ線の弧長は整数平方根で求め、始点から`spacing`ごとに置きます。終点に届かなければ終点にも1個置きます。座標と筆圧は隣接サンプル間で整数の線形補間（0.5は上）です。1回のActionで20,000個を超えるスタンプは拒否します。
4. **縁の被覆率**: 半径`R = 径/2`（q12 = 1/4096px）。内側半径`R_in = min(R*hardness/65535, max(0, R-0.5px))`、外側半径`R_out = R+0.5px`（縁には常に1px幅のアンチエイリアスが入ります）。画素中心までの距離`r`が`R_in`以下で65535、`R_out`以上で0、間は線形です。`soft_round`は同じ比率`t`にsmoothstep（`t²(3-2t)`）を使います。
5. **tipごとの違い**:
   - `round`/`soft_round`/`grain_round`: 円。
   - `flat`: `angle_cdeg`で回転した楕円（長軸=径、短軸=`径*aspect_u16/65535`、下限0.25px）。短軸方向を長軸と同じ尺度に引き伸ばしてから上の被覆率を使います。回転の`sin/cos`は`Decimal`のテイラー展開で求め、環境のlibmに依存しません。
   - `grain_round`など`grain_u16>0`: 画素ごとに`hash16(seed,x,y) < grain_u16`なら欠落させます。紙目のように位置で決まる固定パターンで、同じseedなら同じです。
6. **画素への適用**: `alpha16 = round(stamp_alpha*coverage/65535)`、`alpha8 = round(alpha16*255/65535)`。`alpha8==0`は何もしません。スタンプ順に整数source-over（v1と同じ式）で、その場で合成します（1ストローク内で重なれば濃くなります）。
7. **消去**: `erase_v2`は`alpha8`を消去強度として`new_alpha = round(Ad*(255-alpha8)/255)`にします。0になった画素は`[0,0,0,0]`です。色は参照せず、他のレイヤーは変更しません。
8. **レイヤー合成**: v1は作成順。v2は`rough, sketch, base_color, shade, final_drawing, highlight, correction`のrole順（同roleは作成順）で下から上へ重ね、非表示レイヤーは除外します。

## 3. 設計書との差分・判断

設計書（`docs/`）は変更していません。次は本実装での判断です。

| 項目 | 判断 |
| --- | --- |
| 数値カーネル | `brush_presets_v2.md`が「未確定」とした丸め・coverage・平筆・grain・source-overを本書§2で決めました。画質調整はしていません |
| データの置き場所 | wheelにも含めるため、2つのJSONを`image_drawer/drawing/data/`へ複製し、`tests/test_drawing_engine_v2.py`で`docs/developers/`の原本とバイト一致を確認します |
| 傾き（tilt）・`time_us` | 検証して保持しますが、描画には使いません（設計書も現Rendererでの使用を定めていません） |
| `aspect_u16` | `flat`だけが使います。`dry_texture`（`grain_round`）の値は使いません |
| `create_layer` | v2は`role`必須（上記7種）。v1は`role`を受け付けません。レイヤー上限は16、`幅*高さ*枚数 <= 4,194,304` |
| `set_layer_visibility` | StageController用のActionとして`Engine`が受理します。非表示レイヤーへの`stroke_v2`/`erase_v2`は拒否します |
| 資源予算 | 点数合計200,000と候補面積合計50,000,000は、Episode全体の事前拒否ではなく**Action単位で累積課金**します。超過したActionは描画前に拒否し、状態は変わりません |
| 純Pythonの性能 | 追加依存はありません。大きなブラシ・キャンバスは遅いため、本書のgoldenは小さなキャンバスです。高速化（numpy等）は別Issueです |
| Episode v2 | `DrawingEpisode`はv2ヘッダ（`seed`、brush/controlのSHA-256）を持ちますが、JSON Schemaファイルは未作成です |

## 4. goldenの位置づけと再生成

`tests/fixtures/drawing_engine_v2/golden_brushes.json`は、20x14のキャンバスで12ブラシそれぞれに固定Actionを適用した結果（レイヤー画素の16進）と、role順・非表示の合成例を保存します。**現在の式が同じ画素を出し続けることを確かめる回帰用**で、画質の良し悪し・メーカー製ブラシとの近さを示しません。

式を意図して変えたときだけ、リポジトリ直下で`python tests/test_drawing_engine_v2.py --regenerate`を実行してfixtureを作り直し、差分をレビューしてください。ブラシ数値やカタログを変える場合は新しい`brush_set_id`が必要です。

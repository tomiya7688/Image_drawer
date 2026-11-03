> **対象読者:** 描画器・収集器・学習器・GUI・テストを実装する開発者
> **目的:** 人間の絵師の**行動と判断の連続**を学習するための、解釈の余地を残さない v1 実装契約。
> **状態:** v1 設計案。PRの承認・マージまでは既存実装に反映済みとみなさない。

# 制作過程の状態・行動・再生仕様 v1（正本）

## 0. 優先順位と読み方

この文書は「何を学習するか／何を再生するか／記録をどう解釈するか」の **唯一の規範的仕様**。関連する仕様に食い違いがある場合、プロジェクトの目的は [コンセプト](../kadoka/concept.md)、具体的なv1契約はこの文書を優先する。受け入れ条件は [描画行動の受け入れ試験](drawing_process_acceptance.md) に置く。

- **必須** はv1で実装・検証する事項。**後続** はv1の完成判定に含めない事項。
- `SKETCH`、`LINE`、`COLOR` は**工程の目的・区間**であり、「画像を一度で生成／変換する魔法の関数」ではない。
- `stroke` はブラシによる一回の描画操作。`fill` は領域塗りつぶし（v1では**未実装**）。`color` は色を使って描き進める工程であり、`fill` の同義語ではない。
- `Part Bank`／`RETRIEVE_PARTS`／`COMPOSE` は既存の**補助的な画像部品処理**で、ここで定義する描画操作・学習を代替しない。
- 完成PNGの生成、固定のストローク列のハードコード、WorkflowやPart選択だけの学習を「絵師の制作過程を学習した」と報告してはならない。

## 1. 学習する対象（最上位の不変条件）

### 1.0 txt2imgの推論契約（必須）

製品の基幹経路は**Text-to-Image（txt2img）**である。外部からの必須入力は`prompt: Text`のみ。canvas寸法とseedは固定defaultまたは任意設定として扱えるが、**外部の画像・下書き・参照画像・人間の制作履歴を必須にしてはいけない**。

1. `prompt`と初期の白い空キャンバス、空レイヤー、空行動履歴から`DrawingState`を作る。状態を外部画像から初期化しない。
2. モデルが`DrawingPolicy(prompt, current_state, history)`で描画Actionを逐次決め、`DrawingEngine`が実行する。下書き・線・色・修正を**モデル自身の操作**で行い、次の決定では自ら更新した状態だけを見る。
3. `finish`まで（または制限到達による明示失敗まで）実行し、**最終画像を必須のユーザー向け出力**として返す。全Actionの`DrawingEpisode`を監査・再生可能な追加成果物として保存する。
4. 既存の`Part Bank → COMPOSE`、人間の手描き履歴の再生、外部画像入力だけで完成画像を生成したものは、**基幹txt2img経路の合格としない**。

**区別**：訓練では人間の制作中の状態・筆跡履歴を教師データとして利用するが、**推論ではそれらを外部入力に要求しない**。画像を内部で観測することはimg2imgではない。外部から入力画像を受け取って変換するモードを後から作る場合は別の明示的モード名を持たせる。


AIが観測するのは「現在のキャンバス、各レイヤー、使用中の色・道具、工程、これまでの行動、prompt」。AIが選ぶのは「**次にどの行動をするか、その具体的なパラメータ**」。実行後の状態を再び観測して次を選ぶ。

```text
prompt + DrawingState + ActionHistory
    -> DrawingPolicy（次の行動を予測）
    -> DrawingAction
    -> DrawingEngine（実行・記録）
    -> 新しいDrawingState / 画像 / ActionHistory
    -> 次の予測 … -> finish
```

人間の絵師が行った操作は同じ `DrawingAction` 形式で収集する。学習・推論・再生で**同一の行動意味論**を使う。画像から逆算した推定ストロークは、人間が実際に行った行動として扱わない。

## 2. v1の境界

| 項目 | v1で必須 | 後続（無断で代替してはいけない） |
| --- | --- | --- |
| 絵師の制作工程 | 描画／色選択／レイヤー操作／消去／工程移行／終了を時系列で記録し再生 | 筆圧・傾き・速度、複雑な補正 |
| パレット | **全エピソードで同じ固定256色**、色番号 `0..255` | 画像ごとの自動パレット生成、色番号の可変化 |
| ブラシ | 色と独立した`brush_id`、v1は再現可能な`hard_round_v1`を1種類 | 複数ブラシ・質感・物理筆モデル |
| レイヤー | 透明RGBA層、作成順が下から上、選択して描画 | blend mode・グループ・レイヤー順変更 |
| 描画 | point列を結ぶ一回のストローク、消しゴム | bucket fill・領域推定による自動着色 |
| 保存 | 描画前後の状態を再生できる行動履歴 | 完成画像のみ保存 |
| 学習 | 人間の操作履歴を教師信号に「次の行動」を予測 | PartSelector/WorkflowSelectorだけを主目標とする学習 |

## 3. パレット：固定256色（色の選択と実画素を分離）

**一つの配列`palette[0..255]`が全描画・収集・推論・テストで共通**。RGB値ではなく色番号 `color_id` を予測する。色番号とRGBの対応は実行中・エピソードごと・モデルごとに変更できない。

v1のパレット識別子は `idraw-palette-256-v1`。外部ライセンスに依存しない初期基準として、次の**厳密な順番・計算**でRGB8を定義する（パレット内容を変える場合は新しい `palette_id` と、別PRでの設計承認が必須）。

- `0 <= i <= 215`：`r = i // 36`、`g = (i // 6) % 6`、`b = i % 6`、`palette[i] = [51*r, 51*g, 51*b]`（6×6×6 = 216色）。
- `216 <= i <= 255`：`k = i - 215`、`v = floor(255*k/41 + 0.5)`、`palette[i] = [v, v, v]`（端点を含まない40段階のグレースケール）。
- 3チャンネルとも整数 `0..255`。全256項目は重複しない。`color_id` は**JSONの整数**であり、`true`、`5.0`、`-1`、`256` は不正。
- 確認点：`0=[0,0,0]`、`5=[0,0,255]`、`30=[0,255,0]`、`180=[255,0,0]`、`215=[255,255,255]`、`216=[6,6,6]`、`255=[249,249,249]`。

**透明度の0〜255（256段階）はパレットに含まない。** 描画色は256色からしか選ばないが、透明度やレイヤーの合成で最終PNGに中間RGB値が現れることは許す。「GIF形式への強制変換」や「出力PNG自体を256色に再量子化」は要求しない。表示・保存用色空間はv1では**sRGBのRGB8/RGBA8数値**に固定し、ICC変換・HDRは後続とする。

## 4. DrawingState：1つの制作途中状態

| フィールド | 型と初期状態 | 更新する行動 |
| --- | --- | --- |
| `canvas.width / height` | 各整数 `1..1024`、RGB8白背景`[255,255,255]`。キャンバス寸法はエピソード中不変 | なし |
| `layers` | 空配列。`Layer{id, name, rgba}`を**下から上**に並べる。最大16層 | `create_layer` |
| `active_layer_id` | `null` | `select_layer` |
| `selected_color_id` | `null` | `select_color` |
| `selected_brush_id` | `null` | `select_brush` |
| `phase` | `null` | `set_phase` |
| `finished` | `false` | `finish` |
| `history` | 空配列 | **全行動** |

各Layerの`rgba`はキャンバスと同じ大きさの8bit RGBA画素配列。新規作成時は全画素`[0,0,0,0]`。同じ`layer_id`の再利用禁止。`create_layer`は配列末尾、つまり**最上層**へ追加。`select_layer`は選択だけで既存画素を変更しない。

**合成は下→上のsource-over、最後に白背景**。上レイヤーの不透明な筆跡は同位置の下書きを覆い、上レイヤー上の消去はその箇所の下書きを再び見せる。「上のレイヤーで色を塗る」は**筆跡を記録すること**であり、下書き領域から自動的に塗り領域を生成する処理を意味しない。

## 5. DrawingAction：順序付きの1操作

以下は**v1で受理するすべての行動型**。辞書に記載していない`payload`キーは不正。型の`int`は真偽値を含まない。`name`を除く文字列IDは`[A-Za-z][A-Za-z0-9_-]{0,63}`、最大64文字。`name`は1〜128文字。各イベントには実行主体の`actor`を記録する（`6）。

| `type` | `payload`（指定キーは必須） | 状態変化 |
| --- | --- | --- |
| `create_layer` | `{"layer_id":"sketch","name":"下書き"}` | 新規透明レイヤーを最上位に追加。重複IDはエラー |
| `select_layer` | `{"layer_id":"sketch"}` | 既存レイヤーを作業対象にする |
| `set_phase` | `{"phase":"sketch"}` | `sketch / line / color / correct / other`から選択。**順序は強制しない** |
| `select_color` | `{"color_id":180}` | 固定パレット番号を選ぶ。画素は変えない |
| `select_brush` | `{"brush_id":"hard_round_v1"}` | 色とは独立した筆を選ぶ。v1はこの1種類のみ |
| `stroke` | `{"points":[[0,0],[2,0]],"width_px":1,"opacity_u8":255}` | 選択レイヤーへ、選択したブラシと色で**一回の筆跡**を描く |
| `erase` | `{"points":[[1,0]],"width_px":1}` | 選択レイヤーの筆跡部分だけを透明へ消去。下層は変更しない |
| `finish` | `{}` | エピソードを終了する。最後に必ず1回のみ |

`stroke`の前に、`active_layer_id`、`phase`、`selected_color_id`、`selected_brush_id` が全て設定されていなければエラー。`erase`の前に、`active_layer_id`、`phase`、`selected_brush_id` が設定されていなければエラー。

- `points`は1〜4096個の`[x,y]`の配列。`x,y`は**キャンバス画素中心を表す整数**、`0 <= x < width`、`0 <= y < height`。`[0,0]`は左上画素の中心。配列順に折れ線を結ぶ。1点の場合は点を描く。
- `width_px` は`1..64`の整数。`opacity_u8`は`1..255`の整数。初期値を暗黙に補わない（全フィールド必須）。
- `set_phase` は学習・表示用の意味的な工程ラベル。`color`フェーズでも`stroke`を使い、下書きへの描画と同じ基本操作で処理する。フェーズで描画機能を制限しない。
- レイヤー追加前の描画、存在しないレイヤー選択、`finish`後の行動、不明な型・キー・範囲外の値は必ず拒否する。黙った補正はしない。

### v1の決定的な描画器（`raster-v1`）

- 対応ブラシは`hard_round_v1`のみ。ストロークの全点・隣接2点からなる線分の**和集合**で対象画素を決める。各画素中心`P=(px,py)`、半径`width_px/2`。以下の**整数演算だけ**を使う。境界を含み、アンチエイリアスなし。
- 線分の両端`A=(ax,ay)`、`B=(bx,by)`に対して、`u=P-A`、`v=B-A`、`L=v.x²+v.y²`、`q=u.x*v.x+u.y*v.y`。`L=0`または`q<=0`なら`N=|P-A|²,D=1`。`q>=L`なら`N=|P-B|²,D=1`。その他は`N=(u.x*v.y-u.y*v.x)²,D=L`。**`4*N <= width_px²*D`ならその画素を描く**。`points`が1点の場合は`A=B`として扱う。オーバーフローしない整数演算で判定し、浮動小数への近似変換は禁止。画布外画素へは書き込まない。
- `stroke`のsrc色は`palette[selected_color_id]`、src alphaは`opacity_u8`。**同じストローク内の複数線分が重なっても1画素に合成するのは最大1回**。セグメントごとに濃くする処理は禁止。別の`stroke`イベントでなぞり直した場合は新しい合成を1回行う。
- Alpha合成の各チャンネルは整数演算で次の通り。`As,Ad`は`0..255`のalpha、`Cs,Cd`は色チャンネルの`0..255`。`D = 255*As + Ad*(255-As)`。`Aout = round_half_up(D/255)`。`D==0`なら`Cout=0`、それ以外は`Cout = round_half_up((Cs*As*255 + Cd*Ad*(255-As))/D)`。`round_half_up(p/q) = floor(p/q + 0.5)`。結果は各8bit整数。これをR/G/Bに適用する。
- `erase`の対象画素も**`stroke`と同じ整数距離・線分和集合・境界・画布外クリップ規則**で求める。選択レイヤーの対象画素だけを`[0,0,0,0]`へ置換し、他のレイヤーを消さない。
- 表示画像はすべての透明レイヤーを順に白背景へ同じ整数source-overで合成し、最終の8bit RGBA画像のalphaは255となる。PNG圧縮バイトの同一性ではなく、**復号後のRGBA画素列**の同一性を検証する。
- v1は筆圧、アンチエイリアス、レイヤーの非`normal`ブレンド、ブラシ形状の自動生成を採用しない。将来追加する場合、ブラシとrendererのバージョンを更新する。

## 6. 保存形式：DrawingEpisode v1

1エピソードはUTF-8 JSON 1ファイルとし、`schema_version`で判別する。画像バイト列はJSONへ埋め込まない。**時系列順の`events`が正本であり、最終画像から行動を推定して補完してはならない。**

```json
{
  "schema_version": "drawing-episode/v1",
  "episode_id": "example01",
  "origin": "synthetic",
  "prompt": "赤い線を下書きの上のレイヤーに描く",
  "canvas": {"width": 3, "height": 2},
  "palette_id": "idraw-palette-256-v1",
  "brush_set_id": "brush-v1",
  "renderer_version": "raster-v1",
  "events": [
    {"seq": 0, "time_ms": 0, "actor": "synthetic", "type": "create_layer", "payload": {"layer_id": "sketch", "name": "下書き"}},
    {"seq": 1, "time_ms": 1, "actor": "synthetic", "type": "select_layer", "payload": {"layer_id": "sketch"}},
    {"seq": 2, "time_ms": 2, "actor": "synthetic", "type": "set_phase", "payload": {"phase": "sketch"}},
    {"seq": 3, "time_ms": 3, "actor": "synthetic", "type": "select_brush", "payload": {"brush_id": "hard_round_v1"}},
    {"seq": 4, "time_ms": 4, "actor": "synthetic", "type": "select_color", "payload": {"color_id": 0}},
    {"seq": 5, "time_ms": 5, "actor": "synthetic", "type": "stroke", "payload": {"points": [[0, 0], [2, 0]], "width_px": 1, "opacity_u8": 255}},
    {"seq": 6, "time_ms": 6, "actor": "synthetic", "type": "create_layer", "payload": {"layer_id": "color", "name": "色"}},
    {"seq": 7, "time_ms": 7, "actor": "synthetic", "type": "select_layer", "payload": {"layer_id": "color"}},
    {"seq": 8, "time_ms": 8, "actor": "synthetic", "type": "set_phase", "payload": {"phase": "color"}},
    {"seq": 9, "time_ms": 9, "actor": "synthetic", "type": "select_color", "payload": {"color_id": 180}},
    {"seq": 10, "time_ms": 10, "actor": "synthetic", "type": "stroke", "payload": {"points": [[1, 0], [1, 1]], "width_px": 1, "opacity_u8": 255}},
    {"seq": 11, "time_ms": 11, "actor": "synthetic", "type": "set_phase", "payload": {"phase": "correct"}},
    {"seq": 12, "time_ms": 12, "actor": "synthetic", "type": "erase", "payload": {"points": [[1, 0]], "width_px": 1}},
    {"seq": 13, "time_ms": 13, "actor": "synthetic", "type": "finish", "payload": {}}
  ],
  "metadata": {"dataset_id": "example", "rights": "demo-only"}
}
```

- `origin`は単一主体の`human`／`model`／`synthetic`／`derived`と、複数主体の`mixed`の5種類。各イベントには必須の`actor`を持たせ、`human`（人間の実入力）、`model`（モデルの予測）、`system`（アプリによる自動操作）、`synthetic`（テスト）、`derived`（推定）のいずれかとする。全イベントactorが同じ`human`／`model`／`synthetic`／`derived`ならそれを`origin`に使い、その他は`mixed`。systemのみの完成episodeは拒否。モデルを人間が修正したときは`actor=human`で区別し、mixed全体を人間実演として学習させない。上記例は`synthetic`。
- `seq`は0から始まる欠番なし整数（`0..19999`）。`time_ms`は開始からの非負整数（`0..604800000`＝7日）で**非減少**。実行順は必ず`seq`で決める。時刻は描画結果に影響しない。syntheticでは連番を使用してよい。
- `episode_id`は空でない識別子。`prompt`は文字列。`metadata`はオブジェクトであり、権利・収集元・許諾の記録を保持する。`origin=human`だけでは学習許諾を示さない。
- `palette_id`、`brush_set_id`、`renderer_version`は上記v1の識別子以外を拒否。将来の形式は新しいschema/versionを使う。
- 行動イベントのトップレベルキーは`seq,time_ms,actor,type,payload`の**5キーのみ**。エピソードの必須トップレベルキーはサンプルに列挙した10キー（`metadata`含む）。未知キーは拒否。**完成Episodeだけが**終端`finish`を必須とする。録画中断の部分履歴は`6.2のDraftとして別管理する。
- JSON numberを符号付き32bitへ自動丸めしない。全数値は**JSON字句として小数点も指数表記もない10進整数表記**のみ。`5.0`、`5e0`、`true`、NaN/Infinity、同名JSONキー、非UTF-8を拒否。JSON Schemaだけでこの字句制約を検査できないため、**厳密なJSON読取層**が必須。`events`は`1..20000`件。
- `DrawingEpisode`は`WorkflowSpec/Trajectory`とは別の型。Workflow Runtimeからは`DrawingEpisode`を指す**型付きArtifact**（例：`DrawingEpisode`）としてURIとバージョンを保持し、`Trajectory`にsource/parents/step等を紐づける。`Trajectory.executions`を描画操作ログの代用にしない。

構造検査用の[JSON Schema](schemas/drawing_episode_v1.schema.json)、[正本ゴールデンエピソード](../../tests/fixtures/drawing_process_v1/golden_episode.json)、[期待RGBA画素](../../tests/fixtures/drawing_process_v1/golden_frames.json)も同梱する。Schemaでは検査できない`seq`連続性・時間順・レイヤー存在・座標のキャンバス内制約は**DrawingEngine側で必ず検証**する。なおJSON Schemaの`integer`は数学的に整数な`5.0`を許容する実装がある。v1のID・座標・時刻・連番は**JSONの整数表記のみ許可**し、小数表記（`5.0`等）と真偽値をJSON読取層／Action validatorで拒否する。

### 6.1 リソースと計算量上限（受理前に検証）

形式上validでも過大なCPU/RAMを消費しないように**実行前に全予算を計算して拒否**する。byte制限はJSON parse前に適用する。

| 予算 | v1上限と計算 |
| --- | --- |
| UTF-8ファイル | 16 MiB（読み込む前に確認） |
| キャンバス | `width,height <= 1024`、レイヤー最大16。さらに`width * height * max(1, layer_count) <= 4,194,304` |
| イベント数 | 最大20,000（`finish`を含む） |
| 全ストローク/消去の座標点 | 合計200,000点 |
| 全線分 | 合計200,000本。1点strokeも1本 |
| 線分の探索候補画素 | 全線分のクリップ済み候補矩形の面積合計が最大50,000,000画素 |

候補矩形の定義：`r=ceil(width_px/2)`、`xmin=max(0,min(ax,bx)-r)`、`xmax=min(width-1,max(ax,bx)+r)`。y軸も同様。`xmin>xmax`または`ymin>ymax`なら面積0、それ以外は`(xmax-xmin+1)*(ymax-ymin+1)`。描画器は**矩形内に限って**`5の整数距離判定を行う。全画面を全線分ごとに走査してはならない。合計上限超過時は**一部を描画する前に**episodeごと拒否する。

この予算はDoS対策のv1実装上限であり、作品品質の保証値ではない。緩和する場合は互換性・測定と共に別実行profile/versionを定義する。

### 6.2 途中保存・クラッシュ復旧と参照の一貫性

- `finish`を持つ完成`DrawingEpisode`と、`finish`を持たない未完成の`DrawingDraft`を**別の状態**として扱う。Draftを完成済み／学習済みと偽らない。
- `Recorder`は承認済みイベントを`seq`順のappend-only journalへ記録する。1イベントごとにflushとfsyncを済ませてから、画面へ「保存済み」を返す。描画中のポインタサンプルはpointer-upまで未確定と表示し、1strokeが確定した時点でeventとして記録する。
- 再起動時はchecksum・seq・actor・状態遷移を検証できる最後のprefixまで復旧し、未確定suffixを報告して隔離する。`finish`を自動挿入しない。受理済みでackされたイベントを失わない。
- 完了時はEpisode JSONを一時ファイルへ書き、fsync→同一filesystem内の原子的rename→ディレクトリfsyncする。その後、Episodeの`uri`・SHA-256・`run_id`・`artifact_id`・`status`を含むmanifest/outboxをトランザクションで確定する。
- manifest参照のないblobは`orphan`として照合・再登録を試みる。blob不在やSHA不一致の参照は`broken`とし、公開／学習へ出さない。破損を隠して正常完了としない。
- Draftから続ける場合は既存の有効prefixと描画状態を完全再生したうえで、新規イベントのseqを継続する。リカバリーした操作の`actor`と時刻を変更しない。

独立した検証項目は[受け入れ試験](drawing_process_acceptance.md)、学習への取り込みは[データ品質仕様](drawing_data_quality_v1.md)で定義する。

## 7. 学習・推論の最小契約

- 学習対象は`policy(prompt, DrawingState, history) -> next DrawingAction`。実演エピソードの`events[n]`より前だけを観測し、**次の操作種別とそのパラメータを**教師信号とする。未来のイベント・最終画像はポリシーへの入力に渡さない。
- `select_color` は固定256クラスの分類。`select_brush` は**別の分類**。`stroke`の座標列・太さ・不透明度、レイヤー選択、工程変更、消去、終了も**学習・予測の対象**。固定スクリプトで毎回同じ操作を再生するだけでは学習ではない。
- v1初期ベースラインは`behavior cloning`（教師あり行動模倣）。評価時は「実演の次の行動の予測」と「予測を閉ループで実際に描かせた結果」を**別々に測定**する。損失とモデル内部構造の選択は後続の学習実装Issueで固定し、この行動・状態APIだけを先に固定する。
- 学習用データには`origin=human`かつ権利・出所を確認できる実行履歴を使う。`synthetic`と`derived`は描画器の試験には使えても「人間の制作過程を学習した」根拠にはしない。`model`は推論ログ。
- 記録する再現性情報：`schema_version`、`palette_id`、`brush_set_id`、`renderer_version`、学習dataset split/version、checkpoint、seed、モデルが予測した各行動（採択・棄却を含むなら区別）、実際に適用した行動、失敗理由。
- まず学習器なしの人間入力→履歴保存→完全再生を成立させ、その後に人間履歴からの学習、モデルの逐次描画を通す。Part検索・構図評価などはこのループを置き換えない。

## 8. Workflowとの責務の違い

| 単位 | 意味 | 担当 |
| --- | --- | --- |
| `DrawingAction` | ある時点で色を選ぶ／筆で描く／レイヤーを選ぶ等の**行動1件** | DrawingEngineとDrawingPolicy |
| `DrawingEpisode` | 1枚を描く過程の、時間順の行動列と再生可能な状態 | Recorder / Replay / 学習Dataset |
| `Workflow Step`（`SKETCH/LINE/COLOR/EVALUATE`など） | 何をする工程を起動し、どこで評価するかという**上位の区間・実行制御** | WorkflowRuntime |
| `Image / Score / PartSet` | 中間結果・評価値・補助データ | Renderer / Evaluator / Part Bank |

例：`COLOR` Stepが1回呼ばれても、その内部ではポリシーが`select_layer`、`select_color`、`stroke`、`erase`など**複数の行動を逐次選んで記録する**。画素を直接塗りつぶす画像変換のみの`COLOR`は本仕様に合格しない。`SKETCH`や`LINE`にも同じ原則が適用される。工程間でCanvasと履歴を引き継ぐこと。

## 9. 実装上の分割と禁止事項

推奨の責務境界（path案ではなく抽象的な所有者）：

- **Palette**：`color_id -> RGB`の純粋な固定表。
- **DrawingEngine**：厳密な行動validation、状態遷移、レイヤー合成。学習器に依存しない。
- **Recorder / Replay**：入力イベントの収集と永続化、行動による再生。予測モデルに依存しない。
- **DrawingPolicy**：状態を読み、次の行動を予測。描画を直接変更せずDrawingEngineに渡す。
- **Workflow adapter**：これらを上位Stepとして登録し、既存Runtime/Artifact/Trajectoryへ橋渡しする。
- **Evaluation**：行動予測と完成物を区別して評価する。

**禁止**：完成PNGを用いて架空の筆跡履歴を「人間が描いた過程」と偽装／`COMPOSE`のPart配置を`stroke`として記録／描画工程の固定順を「人間の制作行動の学習」と呼ぶ／renderer内部でモデルに無断の色を選ばせる／256色paletteを画像ごとに並べ替える。

## 10. 変更手順

この仕様に未記載の意味論（筆圧、bucket、異なる色管理、履歴Undo等）が必要な場合、推測実装せず設計Issue→バージョン更新→受け入れテスト更新→実装PRの順にする。

特に`idraw-palette-256-v1`の**具体的なRGB値の対応**は「256色固定」から設計者がここで導いた**明示的な初期案**であり、既にユーザーが個々の256色を承認した事実を意味しない。このPRのレビューで色テーブル自体を確認し、承認された内容のみ正本とする。勝手に既存の実学習データを移行しない。

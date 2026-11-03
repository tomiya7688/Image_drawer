> **対象読者:** DrawingEngine・入力収集GUI・Dataset・DrawingPolicy・ブラシ描画器の実装者
> **目的:** 共通する定番ブラシを Image Drawer 独自のプリセットとして固定し、道具選択と操作の学習を可能にする。
> **状態:** **採用方針：プリセット＋サイズ/不透明度調整**。以下の初期12種と数値初期値は **v2実装案・レビュー対象**。描画器の完成や精度向上を意味しない。

# 標準ブラシプリセット：brush-v2-core12

**最初に読む：** [ブラシのサイズ・不透明度・筆圧をどう扱うか](brush_control_levels_v2.md)。ここではプリセットの固定特性とRendererへの接続だけを定義する。

## 1. なぜプリセットを使うか

Image Drawer は **txt2img**。モデルは文章だけから白紙上に一筆ずつ描き、その過程を学習する。鉛筆・ペン・塗り筆・水彩風・エアブラシなど、ペイントソフト間で共通する**用途の分類**を Image Drawer 自身のブラシIDへ集約する。

- CLIP STUDIO、Photoshop、Illustrator、MediBang等の**個別ブラシデータ、画像、設定ファイルをコピーしない**。製品固有の名称・質感をそのまま再現する目標ではない。各社の採用率ランキングを根拠にした分類でもない。
- 絵師の行動として学ぶのは **「どのプリセットを選んだか」＋「サイズ/不透明度をどう変えたか」＋「筆圧とペン軌跡をどう動かしたか」**。
- **固定256色の color_id はブラシIDとは完全に独立**。プリセットにRGB、固定色、パレットindexを埋め込まない。sourceの色は現行の選択色から取得する。
- 道具数が多いほど精度が上がるとは限らない。**初期は10種類の描画ブラシ＋2種類の消しゴム＝計12**、用途の重複を抑える。**モデル自身の試行時の利用頻度・Reward改善**を測って拡張する。

## 2. プリセット一覧（初期値の正本）

構造化された唯一の初期値は [brush_presets_v2.json](brush_presets_v2.json) に置く。以下は対応表であり、データと違う値があったら**JSONを正**として修正する。

| ID | 通称 | 動作モード | 筆先 | 役割 |
| --- | --- | --- | --- | --- |
| `pencil_graphite` | 鉛筆（ラフ） | 描画 | 粒状丸筆 | アタリ・下書き |
| `ink_gpen` | Gペン | 描画 | 硬め丸筆 | 筆圧で太さの変わる線画 |
| `ink_fine` | 丸ペン（細線） | 描画 | 硬め丸筆 | 安定した細い線 |
| `marker_chisel` | マーカー | 描画 | 傾いた平筆 | ベースカラーの面塗り |
| `paint_round_opaque` | 丸筆（厚塗り） | 描画 | 丸筆 | 不透明な重ね塗り |
| `paint_flat` | 平筆 | 描画 | 平筆 | 広い面の着色 |
| `watercolor_soft` | 水彩風（透過重ね塗り） | 描画 | 柔らかい丸筆 | 半透明の色重ね |
| `airbrush_soft` | エアブラシ | 描画 | 柔らかい丸筆 | 緩やかな陰影 |
| `pastel_oil` | オイルパステル風 | 描画 | 粒状丸筆 | かすれ・塊のある質感 |
| `dry_texture` | ドライ・テクスチャ | 描画 | 粒状丸筆 | 背景・材質感 |
| `eraser_hard` | 消しゴム（硬め） | 消去 | 硬め丸筆 | 輪郭の修正 |
| `eraser_soft` | 消しゴム（柔らかめ） | 消去 | 柔らかい丸筆 | 段階的な消去 |

**水彩風は「透明に重ねられるブラシ」という意味に限定**する。水のにじみ、紙の吸水、色素拡散の物理シミュレーションはこの12種の選定に含めない。オイルパステルやドライブラシも独自の粒状近似であり、メーカー製の既存筆跡と同一という主張をしない。

## 3. ブラシセットの識別子・データ構造

- `brush_set_id = "brush-v2-core12"`。`schema_version = "idraw-brush-presets/2"`。サイズ・不透明度・筆圧・傾きの段階表は独立した `control_set_id = "brush-controls-v2-levels-20261103"`（[段階表JSON](brush_control_levels_v2.json)）。
- `brush_id` はmanifest内で一意のASCII `[a-z][a-z0-9_]*`で、削除/別の形に再利用禁止。GUI表示名とは独立。
- `mode` は `stroke` または `erase`。`family` はGUI用の分類（`sketch,ink,color,paint,shade,texture,erase`）。
- `tip` は `round / flat / soft_round / grain_round`。道具の見た目のラベルではなく、rendererが実装する幾何カーネルを特定する。
- プリセット値は全て**JSON整数**。実数を使わず、色とブラシを混ぜず、外部ブラシデータを参照しない。`q8` は256を1px、`u16` と `q16` は0〜65535を0〜1とする固定小数のスケールである。
- プリセットのindexはJSON配列の順序が正本。AI学習用のclass番号はこの順番（0〜11）。CSV・Pythonの辞書順やusage rankingで並べ替えない。
- **モデルcheckpointとDrawingEpisodeにはbrush_set_idとmanifest SHA-256（canonical JSON）を保存する**。同時に独立した`control_set_id`と[段階表](brush_control_levels_v2.json)のcanonical SHA-256も保存する。canonical bytesはUTF-8で `json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",",":"), allow_nan=False).encode("utf-8")` と同一の結果になるよう定義し、そのSHA-256を記録する。空白やキー順の違いでhashが変化しない一方、プリセット値や順番の変更ではhashが変わる。異なる構成で同じbrush_set_idを使わず、checksum不一致なら学習・推論・Replayを停止。

各プリセットが持つ固定フィールドは `tip`、`family`、`spacing_q16`、`hardness_u16`、`flow_u16`、`grain_u16`、`aspect_u16`、`angle_cdeg`、`pressure_min_size_u16`、`pressure_min_opacity_u16`。

| フィールド | 型／意味 |
| --- | --- |
| `default_size_level_id` | サイズの初期レベルID、0〜18（[19段階](brush_control_levels_v2.md)のindex） |
| `default_opacity_level_id` | 不透明度の初期レベルID、0〜10（[11段階](brush_control_levels_v2.md)のindex） |
| `spacing_q16` | 筆先径に対する連続スタンプ間隔の割合、1〜65535 |
| `hardness_u16` | 筆先半径に対して不透明度が落ち始める比率、0〜65535 |
| `flow_u16` | 1回のスタンプの描画量、1〜65535 |
| `grain_u16` | 擬似的な粒子欠落率、0〜65535。0は無欠落 |
| `aspect_u16` | `flat`筆先の短軸／長軸比率、1〜65535。丸筆は65535 |
| `angle_cdeg` | 筆先の初期角度、0〜35999（1/100度） |
| `pressure_min_size_u16` | 筆圧0のときの径／基準径の比率、1〜65535 |
| `pressure_min_opacity_u16` | 筆圧0のときの不透明度係数、1〜65535 |

この段階の固定値は**設計者が選んだ初期値**であり、各社の実際の内部パラメータを測ったものでも最適値でもない。学習・再現性評価の開始前に**モデル自身の試し描きの画質と描画結果**を見てセット単位でレビュー・freezeする。人間の実演収集は必要ない。

## 4. AIと人間が変更してよいもの

v2の最初の正式な調整可能項目は **`size_level_id` と `opacity_level_id` の2つだけ**。AIが実際に選べる値は段階表のIDに限る。GUIも同じ段階表で表示する。硬さ・grain・tip・spacingなどはブラシプリセットを選び直すことで切り替える。初期から全パラメータを自由予測させて学習空間を不必要に拡げない。

1. `select_brush({"brush_id":"ink_gpen"})`：選択ブラシを切り替え、`size_level_id`・`opacity_level_id`は**そのプリセットのdefaultへリセット**する。変更前の他のブラシの調整値は引き継がない。
2. `set_brush_params({"size_level_id":7,"opacity_level_id":8})`：10px・80%に変更する例。選択済みブラシに対して両方のIDを**同時に更新**。片方だけ変更するUIも、現在値を含む完全なpayloadを記録する。ブラシ未選択時は拒否。
3. `select_color({"color_id":180})`：固定256色から色を選び、筆先と設定は変更しない。選択色は消しゴムの動作に使わない。
4. `stroke_v2`／`erase_v2`：下記の高精細サンプル列で筆を動かす。`mode=erase`のブラシは`stroke_v2`に使えない。`mode=stroke`のブラシは`erase_v2`に使えない。GUIやモデルが必要なブラシへ`select_brush`を発行する。
5. 切り替え・設定・消去は**それぞれ独立したAction**で記録し、modelの予測対象にする。推論時にrendererが勝手に近いプリセットへ置き換えたり、設定変更を黙って挿入したりしない。

学習器は **`p(action_type)`、`p(brush_id | select_brush)`、`p(size_level_id, opacity_level_id | set_brush_params)`、`p(color_id | select_color)`、`p(stroke/erase samples)`** に分ける。サイズ19クラス・不透明度11クラス・筆圧11クラスと筆圧パターン5クラスの意味は[段階表](brush_control_levels_v2.md)が正本。色とbrushの組み合わせで一つのクラスを作らない。erase用／描画用の合法候補をマスクする。単なるプリセットの頻出一致だけで制作能力を評価しない。

## 5. 高精度のペンサンプル形式（v2・共通）

採用した「高精度のペン操作」を次の正本単位で保持する。`brush-v1`の整数ポイントはこの新形式へ黙って置換しない。

```json
{
  "type": "stroke_v2",
  "payload": {
    "samples": [
      {"x_q8": 2560, "y_q8": 5120, "time_us": 0,
       "pressure_level_id": 5, "tilt_x_level_id": null, "tilt_y_level_id": null},
      {"x_q8": 2816, "y_q8": 5376, "time_us": 16000,
       "pressure_level_id": 7, "tilt_x_level_id": null, "tilt_y_level_id": null}
    ]
  }
}
```

- `x_q8 / y_q8`：画素中心の1/256単位の非負整数。各 `0..(width-1)*256` ／ `0..(height-1)*256`。2Dのサブピクセル精度を保持する。
- `time_us`：一本のストローク開始からの相対microsecond、先頭0、非減少、`0..60,000,000`。
- `pressure_level_id`：0〜10または`null`（未取得）。[筆圧11段階](brush_control_levels_v2.md)のindex。**学習時のモデル出力はこのlevelを直接選ぶ**。任意の人間入力機能を実装した場合だけ、実測センサーの元の細かい値をraw sidecarに保存する。levelに丸めた値を元の実測値として扱わない。
- `tilt_x_level_id, tilt_y_level_id`：両方`null`または両方0〜12。傾きは[15°刻みの13段階](brush_control_levels_v2.md)へ正規化。+xは右、+yは下。原センサーの傾きはraw sidecarに残し、normalizerのversionを保存する。
- `samples`：1〜4096個。**モデルが自分で予測し実際に描いた**座標・筆圧・傾きの時系列が主要な学習データ。任意の人間のセンサー入力を収集するときだけ、そのraw値と正規化versionを別に保存する。ユーザーの人間実演は不要。
- Renderer内部では`pressure_level_id=null`の場合だけ計算上の100%を使用するが、**保存したnullは保持する**。`tilt_x_level_id/tilt_y_level_id=null`は計算上の0°として扱う。時間は現在のRendererではスタンプ密度の制御に使わない。筆圧パターンは学習用の補助予測であり、Rendererは**各サンプルのpressure_level_id**を参照する。
- 実行中のtool選択とサイズ/透明度は**strokeの開始時点で固定**。ストローク中に別の選択が入ったらGUIは順序キューで後続に回し、既存strokeの意味を変えない。
- `stroke_v2`は`DrawingEpisode/v2`の行動であり、旧`drawing-episode/v1`スキーマに勝手に追加しない。Episode v2のJSON Schema・Action validation・Replayテストは**対応Renderer実装Issueの必須成果物**とする。

## 6. 何を描画器が決め、何をプリセットで変えるか

**プリセット＝描画の性質の固定データ、Renderer＝その数値を画素に変える唯一の実装**。プリセット自体は学習済みモデルでも既存アプリのブラシファイルでもない。

最初の`brush-raster-v2` rendererが保証すべき意味：

1. `stroke_v2`は同一の選択色を使い、サンプル順にブラシのスタンプを置く。まず`size_level_id → size_px`、`opacity_level_id → percent`、各点の`pressure_level_id → percent`を[固定段階表](brush_control_levels_v2.json)から解決する。筆圧に応じた**径・不透明度**は `effective_fraction = min_fraction + (65535-min_fraction)*pressure_u16/65535`（各最小値はプリセットから取得）。径とalphaは解決したsize/opacityにこの係数を掛ける。整数丸めの詳細はB1のRenderer契約で定義する。
2. 丸・平・soft・grainの4つのtipが、それぞれ別の**形状・エッジ・粒子欠落**を作る。`hardness`はエッジの強さ、`aspect`と`angle`は平筆の輪郭を決める。`grain`は外部画像でなく、seed付きの**独自決定的なprocedural noise**のみで実現する。
3. `spacing`は筆先径に対する連続スタンプ配置間隔、`flow`は各スタンプのインク量を制御する。1ストローク内で複数スタンプが重なる場合、**v1の「1ストロークの合成は1画素1回」ではなく、v2ではスタンプ順のsource-over**で濃くなる。この差は`renderer_profile`で区別する。
4. `erase_v2`は選択レイヤーのalphaを減らす`destination-out`。RGBを固定256色から選ばず、下のレイヤーを直接変更しない。前景のレイヤーが透明になれば下描きが見える。
5. 同じ`prompt/seed/checkpoint/brush_catalog_hash/renderer_version`とAction列のとき同じRGBAを出す。異なるGPU/言語で合成結果が揺れるならそのバージョンでは合格しない。

**まだ精密数値計算までの完全なRenderer v2仕様ではない。** 特に、スタンプ配置の丸め、サブサンプルのcoverage、平筆の回転、粒子noiseのseed展開、source-overの丸めを確定してから、全ブラシの画素goldenを作る。これらを実装者が独断で解釈しないよう**描画器v2は「設計未完了」としてレビュー停止条件**にする。初期の`hard_round_v1`と`raster-v1`は[制作過程v1](drawing_process_v1.md)に従い引き続き再現試験用として保持する。

## 7. 実装・学習・受け入れの順番

| Stage | 成果物 | 合格条件 |
| --- | --- | --- |
| B0：一覧 | [12種のJSON manifest](brush_presets_v2.json)＋[サイズ/不透明度/筆圧の固定段階表](brush_control_levels_v2.json)・選択/調整意味論 | 12種・19/11/11段階・5パターン、ID重複なし、色と独立、値域テスト成功 |
| B1：描画器 | 4 tip・サイズ/opacity/圧力効果・筆先配置・alpha・grainとeraseの決定的数値仕様 | **全12種×複数筆圧×透明度×反復筆跡**のRGBA goldens。異実装で一致 |
| B2：自己描画Trial記録 | **モデル自身**が使ったプリセット・設定・筆圧・時刻を記録 | 保存→Replayで同じレイヤーRGBA、全Actionを再現 |
| B3：学習 | **画像だけを目標に自分で描いて試す**。色・ブラシ・設定・筆圧・点列を画像Rewardで改善 | 12種・サイズ19・不透明度11・筆圧11・筆圧パターン5種を使い、closed-loop画質・Reward改善を報告 |

B0ができただけでB1～B3が完成したことにはならない。水彩物理やクリスタ独自ブラシ資産の互換性を主張しない。

## 8. 安全・権利・更新ポリシー

- 元ブラシのtip画像・テクスチャ・ABR/SUT/MDPファイル等はこのRepositoryへ**再配布・埋め込みしない**。独自生成のブラシ素材を追加する場合も出所と許諾を記録する。
- プリセットの用途分類に対する参考資料と、内部実装コード・brush dataのライセンスを混同しない。
- ブラシ数の増減は**新しいbrush_set_idとmanifest SHA**で管理。既存IDの意味・値・順序を黙って変更しない。モデルcheckpoint・Episode・評価結果には対応セットを必ず保存する。
- 固定256色パレットとブラシプリセットは独立にバージョン管理。色テーブルの承認前にbrushプリセットの再生品質まで確認済みと報告しない。

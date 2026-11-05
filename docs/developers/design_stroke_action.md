> **対象読者:** Policy・action_codec・Trainerの実装者とレビュー担当
>
> **目的:** 学習初期の絵が「うようよ」になる原因を除き、直線を描けるストローク表現に置き換える。
>
> **状態:** 2026-11-05。実装手順1〜3（離散head・Policy・設定とcheckpoint互換）を実装済み。手順4（直線率の記録）は`metrics.jsonl`に1線分ストロークの割合などを追加した範囲まで。**実測・学習の成功・絵の品質は未確認**です。[design_trainer.md](design_trainer.md) 4.2節の `stroke` head の記述を置き換える。

# ストローク表現の設計（直線を描ける離散Action）

## 1. 問題

現在の `stroke` head は、始点・中間・終点の6座標を連続値のガウスから得ます（`u ~ N(Wx, σ²)`、σは0.15から0.05）。

- 座標が独立にゆれるため、中間点がずれるたびに線が折れ曲がる。σ=0.15はキャンバスの15%で、方向のそろった直線にならない。
- 「まっすぐ進む」を選ぶ手段がなく、直線は座標3つがたまたま一直線になる確率に頼る。
- 連続座標は丸めが細かく、同じ意図でも毎回違う画素になり、Reward の勾配がノイズに埋もれる。

Engine の `stroke_v2{samples}` は変えません。変えるのは、Policy が `samples` を作る手前（`action_codec`）だけです。

## 2. 案

ストロークを「始点の格子マス＋最大 K 本の線分」で表します。線分は **方向 × 長さ** の離散値です（「書いた地点から、どの方向に何マス進むか」）。

| head | 選択肢 | 内容 |
| --- | --- | --- |
| `start` | 格子 `G×G`（既定 G=32） | 始点のマス。マス中心をキャンバス座標へ写す |
| `start_mode` | 2 | `new`（`start`のマス）／`continue`（直前のストロークの終点から続ける） |
| `segments` | K（既定 4） | 線分の数 1〜K |
| `dir[i]` | 16方向（22.5度刻み） | 線分 i の向き。最初の線分は絶対方向、2本目以降は **直前の線分に対する相対方向**（直進を含む 16 通り） |
| `len[i]` | 6段階 `{1,2,3,5,8,13}` マス | 線分 i の長さ。Fibonacci 風で、大きな動きと細かい動きを少ない選択肢で両立する |

- 直線は `segments=1`、折れ線は複数、緩い曲線は小さい相対角を連ねて表す。
- 線分の端点は整数マスに丸め、`samples` は端点を結ぶ **等間隔の点列** にする（線分1本あたり最低2点、全体で `stroke_samples` 上限）。同じ選択からは必ず同じ `samples` ができる。
- キャンバス外へ出る線分は `mask` で選べなくする（位置は観測の `tool_state` だけで決まり、目標画像に依存しない）。
- すべて categorical（softmax）で、`kind`・`color` などの既存 head と同じ扱いにする。ガウスの σ 減衰設定は不要になり、探索は softmax の温度（既定1.0）と entropy bonus で制御する。

方向を 16 にする理由: 8 方向だと 22.5度ずつの斜線（髪・布のひだ）が出せない。32 以上は選択肢が増えて学習初期の探索が薄まる。K・G・長さの段階は設定値にして実測で調整する。

## 2.1 確定規則（実装が一意に決まるための定義）

- **サンプルの順序**: `start_mode`→(`new`なら`start`)→`segments`→線分1の`dir`,`len`→線分2の…の順に、確定済みの現在位置を使って逐次選ぶ。
- **方向番号**: `d=0..15`、角度 `θ=d×22.5度`、`d=0` が画面右(+x)、画面座標のyは下向きを正として時計回りに増える。相対方向は `d_i = (d_{i-1} + r_i) mod 16`（`r_i=0` が直進）。
- **変位の整数化**: 線分 i の変位は `(round(L·cosθ), round(L·sinθ))`（L=長さ段階のマス数、`round` は0.5を0から遠い方へ丸める）で格子単位の整数。丸めたあとの変位が `(0,0)` になる選択（L=1の斜め等は `(1,1)` や `(1,0)` になるため通常起きないが、起きた場合）は無効としてマスする。
- **座標変換**: 格子点 `(gx,gy)` は `x_q8 = round((gx+0.5)/G × (canvas_side-1) × 256)`（マス中心）。前ストロークの終点は格子点（整数）で保持し、`continue` は `start` を使わず（`start`のlogitは使わない）その格子点から始める。直前にストロークがなければ`continue`はマスされる。
- **逐次マスク**: 各段階で、確定済みの現在位置に候補の変位を足した終点が `0≤gx,gy<G` にあるものだけを有効にする（境界上は有効）。線分の途中はキャンバス内の凸領域なので、終点が内側なら途中も内側。全候補が無効のとき: その線分を作らず、そこまでのストロークで確定する（`segments` は実際の線分数に切り詰め、ログに残す）。線分が0本なら`stroke_v2`を出さず、`kind`のマスで `stroke_v2` を選べないようにする。
- **補間**: 線分ごとに端点間を等間隔に補間し、連続する線分の共有端点は1点にまとめる。全体のサンプル数は `stroke_samples` 以下（既定 6→ **12 に引き上げ**、設定値）。線分数×2−(線分数−1)=線分数+1 点が最低限必要なので、`K+1 ≤ stroke_samples` を設定検証する。余りの点は長い線分から順に配る。`time_us`は0から`STROKE_POINT_STEP_US`刻み、`pressure_level_id`は設定の固定値（現行と同じ）。
- **有効なhead**: 実際に使う `dir[i]`・`len[i]`（i < `segments`）、`start`（`new`のときだけ）だけを log-prob と entropy に含める。使わない線分・`start`はサンプルせず、勾配もentropyも0。`segments` は1〜Kの個数で、softmaxの添字 0..K−1 が個数 1..K に対応する。
- **「同じ画素」の基準**: 受け入れテストの直線判定は、`samples` の座標（`x_q8,y_q8`）が始点・終点を結ぶ線から1マス（`256×(canvas_side-1)/G`）以内、Engineで描いた画素が同条件の1本線とのIoU 0.95以上で判定する。
- **checkpoint**: checkpointにhead構成のhash（head名と出力数、G・K・方向数・長さ段階）を保存し、読み込み時に不一致なら `ValueError("checkpointのstroke head構成が一致しません: ...")` で拒否する。

## 3. 代替案と採用しなかった理由

| 案 | 採用しない理由 |
| --- | --- |
| 連続座標のまま σ を小さくする | 直線の問題は残り、探索も止まる。 |
| 描画後に線を平滑化・直線化する | Replay で再現する対象が Action 列からずれる。画像に後処理を足す形になり、Episode と画素の対応が崩れる。 |
| 始点と終点の 2 点だけを選ぶ | 直線は描けるが、曲線が描けず、描ける形が減る。 |
| 全画素の座標を離散化する（G=256 など） | 選択肢が多すぎて 1 ステップあたりの勾配が薄い。粗い格子＋線分の方が少ないパラメータで済む。 |

## 4. 契約への影響

| 項目 | 変更 |
| --- | --- |
| `HEAD_SIZES["stroke"]=6` | 廃止。`start`（G²）、`start_mode`（2）、`segments`（K）、`dir`（16）、`len`（6）に分ける |
| `ActionChoice.stroke_u` | 廃止。`stroke_start`・`stroke_mode`・`stroke_segments`・`stroke_dirs`・`stroke_lens` に置き換える |
| `build_stroke_samples` | 入力を離散選択に変え、格子座標→`x_q8,y_q8` の変換と線分の補間を行う（`time_us`・`pressure_level_id` の契約は同じ） |
| Policy の対数確率・勾配 | 追加 head は softmax の標準勾配。`dir`・`len` は線分数ぶん加算（使わない線分は勾配 0） |
| Engine・Episode・Replay・schema | 変更なし。`stroke_v2` の `samples` の形が同じため |
| target 隔離 | 変更なし。新 head も観測だけを入力とする |
| パラメータ数 | 出力数は `start` 1024、`start_mode` 2、`segments` 4、`dir` 16×4=64、`len` 6×4=24で合計1118(既存head 326のうちstroke 6を除いた320を足すと1438)。単一線形層で `D≈280` の重み+バイアスは約40万個(float64で約3.2MB)。Adamのm・vと勾配バッファを含め3倍強の約10MB。実測は実装PRで行う |

## 5. 受け入れテスト

| テスト | 確認 |
| --- | --- |
| `test_stroke_single_segment_is_straight` | `segments=1` の `samples` が始点と終点を結ぶ一直線上にある（基準は2.1節） |
| `test_stroke_directions_cover_16_angles` | 16 方向の角度が 22.5 度刻みで、長さ段階の通りに端点が動く |
| `test_stroke_relative_direction_zero_is_straight` | 相対方向 0 の連続線分が、合計変位が同じ1本線と、2.1節の基準（座標1マス以内・IoU 0.95以上）で一致する |
| `test_stroke_outside_canvas_is_masked` | 外へ出る方向・長さの選択肢がマスされ、サンプルされない |
| `test_stroke_samples_are_deterministic` | 同じ選択から同じ `samples`（Replay と一致） |
| `test_stroke_continue_starts_at_previous_end` | `continue` が直前の終点から始まる。直前がなければ `new` だけ選べる |
| `test_stroke_policy_gradient_matches_numeric` | 新 head の勾配が数値微分と一致する |
| `test_stroke_mask_is_sequential_and_inclusive_of_border` | 確定済みの現在位置から逐次にマスされ、境界上の終点は有効 |
| `test_stroke_all_invalid_truncates_segments` | 全候補が無効のとき線分数を切り詰める。0本なら`stroke_v2`が選べない |
| `test_stroke_unused_heads_excluded_from_logprob_and_entropy` | 使わない`dir`・`len`・`start`が対数確率とentropyに入らない（線分数1〜Kで確認） |
| `test_stroke_diagonal_rounding_is_deterministic` | 16方向×全長さ段階で整数変位が仕様どおり、(0,0)は無効 |
| `test_stroke_sample_budget_rejects_small_stroke_samples` | `K+1 > stroke_samples` の設定を拒否。上限超過時は長い線分に点を配る |
| `test_checkpoint_rejects_old_stroke_head` | 旧構成のcheckpointを理由つきで拒否し、新構成は読める |
| `test_policy_stroke_has_no_target_input` | 新 head も目標画像・パスを受け取らない（既存の隔離テストを拡張） |

「直線が描けること」は上のとおり画素で確かめますが、**絵が良くなるか**は実学習で測る必要があり、この設計だけでは確認できません。

## 6. 実装手順

1. `policy_params`・`action_codec` に離散ストローク（head、mask、`build_stroke_samples`）を実装し、上のテストを通す。
2. `policy.py` の sample・log_prob・勾配を新 head に合わせる。ガウス σ の設定項目を廃止し、温度・entropy を使う。
3. `config` の既定値（G・K・長さ段階・方向数）と checkpoint の互換（旧 checkpoint は読み込み拒否し、理由を出す）。
4. スモーク学習で直線率（1 線分ストロークの割合と、折れ線の平均曲がり角）を `training/evaluation` の記録に追加する。

実装手順 1〜3 は 1 PR（500 行以下に収まらなければ 1 と 2 を分ける）、4 は別 PR。

## 7. 未確認事項

- 既定の G=32・K=4・方向 16・長さ 6 段階は **見積り**で、実測していない。
- 学習が収束するか、絵が実際にまとまるかは未確認。
- 筆圧の変化・消しゴム（`erase_v2`）は対象外のまま（design_trainer.md 10節）。

## 8. 実装メモ（設計との対応と補足）

- 実装は`image_drawer/training/stroke_geometry.py`（格子・方向・長さ・逐次マスの`walk_stroke`）、`action_codec.py`（`build_stroke_samples`、`resolve_stroke`）、`policy.py`（head、勾配）、`config.py`（`stroke_grid`・`stroke_max_segments`・`stroke_lengths`・`softmax_temperature`・`stroke_samples`既定12）。受け入れテストは`tests/training/test_stroke_action.py`。
- サンプル・対数確率・Action変換は同じ`walk_stroke`を通る。ActionChoiceは選択だけを持ち、マスは直前ストロークの終点（`PolicySample.stroke_prev_end`）から再計算する。
- `dir`・`len`は線分ごとに別head（`stroke_dir_0..K-1`、`stroke_len_0..K-1`）。出力数の合計は4節の見積りと同じ。
- 温度は全headのlogitsに掛け（`logits / T`）、対数確率・エントロピーの勾配に`1/T`を掛ける。既定1.0では従来のsoftmaxと同じ。評価時も設定値の温度で統一する（ガウスの`sigma_end`固定は廃止）。
- 直前ストロークの終点は、Policyが観測の`action_tail`（先頭が空なら新しいtrial、末尾が受理された`stroke_v2`なら提案した終点を確定）だけで追う。Observationの契約は変えていない。
- 方向反転（相対方向8）は常に有効なため、1本目が引ければ2本目以降が全候補無効になる構成は実際には起きない。切り詰めの処理はあるが、テストでは`valid_moves`を差し替えて確認している。線分を1本も引けない構成（G=1など）では`kind`マスで`stroke_v2`を選べない。
- 手順4のうち、`metrics.jsonl`に`stroke_count`・`stroke_single_segment_rate`・`stroke_truncated_count`を追加した。折れ線の平均曲がり角と`training/evaluation`の記録への追加は未実装（別PR）。
- checkpointのsidecar JSONに`stroke_head_hash`を保存し、旧checkpoint（キーなし）と構成違いは`ValueError`（`CheckpointMismatchError`）で拒否する。`policy_version`は`parametric-softmax-stroke-v2`、`RandomLegalPolicy`は`random-legal-v2`（登録ID`random-legal-v1`は変えていない）。
- 実測（パラメータ数・メモリ・学習の収束・絵の品質）は未実施で、見積りのまま。

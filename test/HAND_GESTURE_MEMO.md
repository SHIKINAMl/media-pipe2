# Hand Gesture 実装メモ

## 対象ファイル
- `test/mediapipe_hand_test.py`:
  - 手検出の共通ランナー (`run_hand_test`) を提供
  - Live Stream 推論、骨格描画、カメラ処理を担当
- `test/mediapipe_hand_gesture_test.py`:
  - 共通ランナーを import してジェスチャー判定だけを追加
  - `overlay_callback=draw_gesture_labels` で描画を差し込み
- `test/mediapipe_hand_open_close_test.py`:
  - 共通ランナーを import して「開閉のみ」判定を追加
  - `overlay_callback=draw_open_close_labels` で表示を差し込み

## 実装方針
- 既存の手検出処理は再利用し、ジェスチャー判定ロジックを分離
- 最小ルールベースで以下を判定:
  - `OPEN`
  - `FIST`
  - `PEACE`
  - それ以外は `OTHER`

## 判定ルールの概要
- 人差し指〜小指:
  - `tip.y < pip.y` なら「伸びている」とみなす
- 親指:
  - 右手: `tip.x < ip.x`
  - 左手: `tip.x > ip.x`
  - handedness 不明時は `abs(tip.x - ip.x) > 0.03` を仮判定
- 最終ラベル:
  - `index_up && middle_up && !ring_up && !pinky_up` -> `PEACE`
  - 伸びている指の数 `>= 4` -> `OPEN`
  - 伸びている指の数 `<= 1` -> `FIST`
  - それ以外 -> `OTHER`

## 既知の注意点
- ルールベースのため、手の向き・カメラ角度・遮蔽に弱い
- ラベル平滑化を入れているが、急激な切り替え時は遅延が出る

## 改善候補
- 親指判定を2D比較だけでなく距離・角度ベースに拡張
- 新ジェスチャーの追加

## ラベル平滑化の実装
- 対象:
  - `test/mediapipe_hand_gesture_test.py`
  - `test/mediapipe_hand_open_close_test.py`
- 方式:
  - 各手インデックスごとに、直近 `N` フレーム分のラベル履歴を保持
  - `Counter(...).most_common(1)` で多数決ラベルを採用
- 現在値:
  - `SMOOTHING_WINDOW = 5`
- 補足:
  - 手が画面外へ出たインデックスの履歴は削除

## 開閉のみ簡潔版について
- 目的:
  - 判定を最小化して、ゲーム入力向けに安定しやすい2値判定を行う
- 実装ファイル:
  - `test/mediapipe_hand_open_close_test.py`
- 判定ロジック:
  - 各指の伸展判定を行い、伸びている指の数 `up_count` を算出
  - `up_count == 5` を `OPEN`
  - `up_count == 0` を `CLOSE`
  - それ以外を `OTHER`
- 平滑化ロジック:
  - 直近 `N` フレームで多数決を実施
  - ただし多数決対象は `OPEN` / `CLOSE` のみ
  - 周辺フレームに `OPEN` / `CLOSE` がない場合:
    - `OTHER` を維持
- 位置平滑化:
  - 直近 `N` フレームのランドマーク座標 `x/y/z` を平均化
  - 平滑化済みランドマークを骨格描画と開閉判定の両方に使用
- 親指判定の実装:
  - `is_thumb_extended()` で親指だけ別ロジックを使用
  - 判定条件は以下の3つ
    - 親指 MCP 角度 `angle_mcp > 145`
    - 親指 IP 角度 `angle_ip > 150`
    - 親指 tip が掌中心から見て IP より十分遠い
  - 掌中心は `wrist`, `index_mcp`, `pinky_mcp` の平均で近似
  - 距離閾値には `EXTENSION_THRESHOLD` を使用
  - 目的:
    - 親指の第一関節だけを少し曲げた時の誤判定を減らす
    - 親指付け根だけの角度変化で `OPEN/CLOSE` が逆転しにくくする
- 実行:
  - `.\\.venv\\Scripts\\python.exe .\\test\\mediapipe_hand_open_close_test.py`

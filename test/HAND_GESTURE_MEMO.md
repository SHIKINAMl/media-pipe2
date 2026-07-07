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
- フレーム単体判定なのでラベルが揺れることがある

## 改善候補
- 直近 N フレームの多数決でラベル平滑化
- 親指判定を2D比較だけでなく距離・角度ベースに拡張
- 新ジェスチャーの追加

## 開閉のみ簡潔版について
- 目的:
  - 判定を最小化して、ゲーム入力向けに安定しやすい2値判定を行う
- 実装ファイル:
  - `test/mediapipe_hand_open_close_test.py`
- 判定ロジック:
  - 各指の伸展判定を行い、伸びている指の数 `up_count` を算出
  - `up_count >= 3` を `OPEN`
  - それ以外を `CLOSE`
- 実行:
  - `.\\.venv\\Scripts\\python.exe .\\test\\mediapipe_hand_open_close_test.py`

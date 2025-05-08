import mediapipe as mp
import cv2 as cv
import threading
import vgamepad as vg
import numpy as np

# 仮想GamePadの初期化
gamepad = vg.VDS4Gamepad()

# MediaPipeの定義
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
ImageFormat = mp.ImageFormat

# モデルのパス
model_path = r"C:\Users\nagashima\PycharmProjects\ResearchProject01\pose_landmarker_full.task"

# グローバル変数で結果を共有（スレッド間安全のためロックを使用）
latest_landmarks = None
lock = threading.Lock()

# 結果を受け取るコールバック関数
def result_callback(result, output_image, timestamp_ms):
    global latest_landmarks
    with lock:
        latest_landmarks = result.pose_landmarks

# PoseLandmarkerの設定
options = PoseLandmarkerOptions(
    base_options = BaseOptions(model_asset_path = model_path),
    running_mode = VisionRunningMode.LIVE_STREAM,
    result_callback = result_callback,
    min_pose_detection_confidence = 0.5,
    min_tracking_confidence = 0.7
)

pose_landmarker = PoseLandmarker.create_from_options(options)

# カメラ起動
cap = cv.VideoCapture(0)

# フレーム送信用のタイムスタンプ（ms単位）
timestamp = 0


def draw_landmarks_subset(frame, landmarks, indices_with_labels):
    """
    指定されたランドマークのみを描画する関数。

    Parameters:
        frame (np.ndarray): OpenCVの画像フレーム
        landmarks (List[NormalizedLandmark]): MediaPipeのランドマーク一覧（33個）
        indices_with_labels (dict): {インデックス: ラベル名} の辞書
    """
    h, w = frame.shape[:2]

    for idx, label in indices_with_labels.items():
        landmark = landmarks[idx]
        x_px = int(landmark.x * w)
        y_px = int(landmark.y * h)

        # 青い円を描画
        cv.circle(frame, (x_px, y_px), 7, (255, 0, 0), -1)

        # ラベルを描画（白文字）
        cv.putText(frame, label, (x_px + 5, y_px - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

target_indices = {
    0: "nose",
    2: "left_eye",
    7: "left_ear",
    8: "right_ear"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format = ImageFormat.SRGB, data = frame_rgb)

    # 推論リクエストを送信
    pose_landmarker.detect_async(mp_image, timestamp)
    timestamp += 33  # 約30fps分のms

    # ランドマークの描画（非同期で取得される）
    with lock:
        if latest_landmarks:
            # ランドマーク取得
            nose = latest_landmarks[0][0]
            left_ear = latest_landmarks[0][7]
            right_ear = latest_landmarks[0][8]
            left_eye = latest_landmarks[0][2]

            # 首の傾きを左右・上下で判定
            x_angle = (nose.x - left_ear.x) / (right_ear.x - left_ear.x) - 0.5
            y_angle = nose.y - left_eye.y

            # スティック座標に変換
            x_val = int(np.clip(x_angle * 2, -1.0, 1.0) * 32767) * -1
            y_val = int(np.clip(y_angle * 5, -1.0, 1.0) * 32767)

            # ゲームパッド反映
            gamepad.right_joystick(x_value=x_val, y_value=y_val)
            gamepad.update()

            draw_landmarks_subset(frame, latest_landmarks[0], target_indices)


    # 表示
    cv.imshow("Pose Estimation (LIVE_STREAM)", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

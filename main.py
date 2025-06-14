import mediapipe as mp
import cv2 as cv
import threading
import vgamepad as vg
import numpy as np
import keyboard

# 仮想GamePadの初期化
gamepad = vg.VDS4Gamepad()

# MediaPipe定義
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
ImageFormat = mp.ImageFormat

# モデルのパス
model_path = r"C:\Users\nagashima\PycharmProjects\ResearchProject01\pose_landmarker_full.task"

# グローバル共有用
latest_landmarks = None
lock = threading.Lock()

# 設定初期値
deadzone = 0.1
gain = 3.0
running = True

# コールバック関数
def result_callback(result, output_image, timestamp_ms):
    global latest_landmarks
    with lock:
        latest_landmarks = result.pose_landmarks

# PoseLandmarker設定
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=result_callback,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.7
)
pose_landmarker = PoseLandmarker.create_from_options(options)

# カメラ起動
cap = cv.VideoCapture(0)
timestamp = 0

# ランドマーク描画
def draw_landmarks_subset(frame, landmarks, indices_with_labels):
    h, w = frame.shape[:2]
    for idx, label in indices_with_labels.items():
        landmark = landmarks[idx]
        x_px = int(landmark.x * w)
        y_px = int(landmark.y * h)
        cv.circle(frame, (x_px, y_px), 7, (255, 0, 0), -1)
        cv.putText(frame, label, (x_px + 5, y_px - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

# スティック変換（デッドゾーン先判定・微小丸め・クリップ）
def compute_stick_value(angle, gain=3.0, deadzone=0.1, exponent=1.5):
    if abs(angle) < deadzone:
        return 0
    scaled = angle * gain
    value = int(np.sign(scaled) * (abs(scaled) ** exponent) * 32767)
    if abs(value) < 200:
        value = 0
    return int(np.clip(value, -32767, 32767))

# 対象ランドマーク
target_indices = {
    0: "nose",
    2: "left_eye",
    7: "left_ear",
    8: "right_ear"
}

# キーイベント登録
keyboard.on_press_key("up", lambda _: increase_deadzone())
keyboard.on_press_key("down", lambda _: decrease_deadzone())
keyboard.on_press_key("right", lambda _: increase_gain())
keyboard.on_press_key("left", lambda _: decrease_gain())
keyboard.on_press_key("q", lambda _: quit_program())

# 調整関数
def increase_deadzone():
    global deadzone
    deadzone = min(deadzone + 0.1, 1.0)
    print(f"Deadzone increased to {deadzone:.1f}")

def decrease_deadzone():
    global deadzone
    deadzone = max(deadzone - 0.1, 0.0)
    print(f"Deadzone decreased to {deadzone:.1f}")

def increase_gain():
    global gain
    gain = min(gain + 0.1, 10.0)
    print(f"Gain increased to {gain:.1f}")

def decrease_gain():
    global gain
    gain = max(gain - 0.1, 0.1)
    print(f"Gain decreased to {gain:.1f}")

def quit_program():
    global running
    running = False
    print("Program terminated by user.")

# メインループ
while cap.isOpened() and running:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.flip(frame, 1)
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=ImageFormat.SRGB, data=frame_rgb)

    pose_landmarker.detect_async(mp_image, timestamp)
    timestamp += 33

    with lock:
        if latest_landmarks:
            nose = latest_landmarks[0][0]

            # 🎯 鼻のx座標 - 画像中央（0.5）で相対位置を取得
            relative_x = nose.x - 0.5
            relative_y = 0.0  # y方向は無効

            x_val = compute_stick_value(relative_x, gain=gain, deadzone=deadzone)
            y_val = compute_stick_value(relative_y, gain=gain, deadzone=deadzone)

            if x_val == 0 and y_val == 0:
                gamepad.right_joystick(x_value=0, y_value=0)
            else:
                gamepad.right_joystick(x_value=-x_val, y_value=y_val)

            gamepad.update()
            draw_landmarks_subset(frame, latest_landmarks[0], target_indices)

    # 状態表示
    cv.putText(frame, f"Deadzone: {deadzone:.2f}", (10, 30),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv.putText(frame, f"Gain: {gain:.2f}", (10, 60),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv.imshow("Pose Estimation (LIVE_STREAM)", frame)
    if cv.waitKey(1) == 27:  # Escキーで終了
        break

cap.release()
cv.destroyAllWindows()

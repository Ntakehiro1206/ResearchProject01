import mediapipe as mp
import cv2 as cv
import threading
import numpy as np

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
        label_text = f"{label} ({x_px}, {y_px})"
        cv.putText(frame, label_text, (x_px + 5, y_px - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)

pose_landmark_names = {
    0:  "nose",
    1:  "left_eye_inner",
    2:  "left_eye",
    3:  "left_eye_outer",
    4:  "right_eye_inner",
    5:  "right_eye",
    6:  "right_eye_outer",
    7:  "left_ear",
    8:  "right_ear",
    9:  "mouth_left",
    10: "mouth_right",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    17: "left_pinky",
    18: "right_pinky",
    19: "left_index",
    20: "right_index",
    21: "left_thumb",
    22: "right_thumb",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index"
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

            #for idx, landmark in enumerate(latest_landmarks[0]):
                #x = int(landmark.x * frame.shape[1])
                #y = int(landmark.y * frame.shape[0])

            draw_landmarks_subset(frame, latest_landmarks[0], pose_landmark_names)


                # 表示
    cv.imshow("Pose Estimation (LIVE_STREAM)", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

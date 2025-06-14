import mediapipe as mp
import cv2 as cv
import threading
import vgamepad as vg


# ───────────────
# MediaPipe Tasks 定義
# ───────────────
BaseOptions       = mp.tasks.BaseOptions
PoseLandmarker    = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
ImageFormat       = mp.ImageFormat

# ───────────────
# モデルパス
# ───────────────
model_path = r"C:\Users\nagashima\PycharmProjects\ResearchProject01\pose_landmarker_full.task"

# ───────────────
# PoseLandmarker を VIDEO モードで初期化（同期検出）
# ───────────────
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.7
)
pose_landmarker = PoseLandmarker.create_from_options(options)

# ───────────────
# キャプチャ設定：640×480 にダウンサンプル
# ───────────────
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# ───────────────
# 仮想ゲームパッド初期化
# ───────────────
gamepad = vg.VDS4Gamepad()

# ───────────────
# グローバル変数
# ───────────────
latest_landmarks = None
lock = threading.Lock()

# ───────────────
# ランドマーク描画対象：鼻＋両肩
# ───────────────
LANDMARKS = { 0:"nose", 11:"left_shoulder", 12:"right_shoulder" }

# ───────────────
# 閾値（水平オフセットで左/右判定）
# ───────────────
threshold_left  = 0.12
threshold_right = 0.12

# ───────────────
# 描画ヘルパー
# ───────────────
def draw_landmarks_subset(frame, landmarks):
    h,w = frame.shape[:2]
    for idx,label in LANDMARKS.items():
        lm = landmarks[idx]
        x,y = int(lm.x*w), int(lm.y*h)
        cv.circle(frame,(x,y),7,(0,0,255),-1)
        cv.putText(frame,label,(x+5,y-5),cv.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1,cv.LINE_AA)

# ───────────────
# メインループ
# ───────────────
timestamp = 0
while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break

    # 鏡像反転＆RGB化
    frame = cv.flip(frame,1)
    rgb   = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=ImageFormat.SRGB, data=rgb)

    # 同期検出（VIDEO モード）
    result = pose_landmarker.detect_for_video(mp_image, timestamp)
    timestamp += 33

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]
        nose     = lm[0]
        l_sh, r_sh = lm[11], lm[12]

        # 水平オフセット計算
        sw = r_sh.x - l_sh.x
        mid = (l_sh.x + r_sh.x)/2 if abs(sw)>1e-6 else 0.5
        x_off = (nose.x - mid) / sw if abs(sw)>1e-6 else 0.0

        # デバッグログ
        print(f"[DEBUG] x_offset: {x_off:.3f}")

        # 三段階制御
        if x_off < -threshold_left:
            stick_x = 2000
        elif x_off > threshold_right:
            stick_x = 8000
        else:
            stick_x = 5000

        # 送信
        gamepad.right_joystick(x_value=stick_x, y_value=5000)
        gamepad.update()

        # 描画
        draw_landmarks_subset(frame, lm)
        cv.putText(frame, f"x_off: {x_off:.3f}", (10,30),
                   cv.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        cv.putText(frame, f"Stick X: {stick_x}", (10,60),
                   cv.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        cv.putText(frame, f"ThL(a/d):{threshold_left:.2f}", (10,90),
                   cv.FONT_HERSHEY_SIMPLEX,0.7,(255,150,150),2)
        cv.putText(frame, f"ThR(j/l):{threshold_right:.2f}", (10,120),
                   cv.FONT_HERSHEY_SIMPLEX,0.7,(150,150,255),2)

    cv.imshow("Shoulder+Nose Control", frame)
    if cv.waitKey(1)==27:
        break

cap.release()
cv.destroyAllWindows()

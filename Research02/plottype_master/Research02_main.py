import cv2

# --- 自作モジュール ----------------------------------------------------------
from Research02.plottype_master.perspective_transformer import warp_by_aruco, pixel_to_center
from Research02.plottype_master.Research02_object_marker_detector import object_marker_detector
from Research02.plottype_master.Research02_convert_input import GamepadController
# ----------------------------------------------------------------------------


def main() -> None:
    # ── デバイス初期化 ──────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError("カメラの接続失敗")

    gamepad = GamepadController(fps=120, target_id=4, ds4_mode=False)
    gamepad.start()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 透視変換（4 つの ArUco が見つからなければ None）
            img_trans = warp_by_aruco(frame)
            objects: list[dict] = []

            # -------------- 検出・表示 --------------
            if img_trans is not None and img_trans.size:
                marker_img, objects = object_marker_detector(img_trans, False)
                shape_for_pad = img_trans.shape

                # ここで中心原点座標を計算して描画
                for o in objects:
                    cx, cy = pixel_to_center(
                        o["center"], shape_for_pad[1], shape_for_pad[0]
                    )
                    cv2.putText(
                        marker_img,
                        f"({int(cx)},{int(cy)})",
                        o["center"],
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                cv2.imshow("Top View + detect marker", marker_img)
            else:
                cv2.imshow("WebCam", frame)
                shape_for_pad = frame.shape

            # -------------- ゲームパッド更新 --------------
            tgt = next((o for o in objects if o["id"] == gamepad.id), None)
            if tgt:
                gamepad.update_center(tgt["center"], shape_for_pad)
            else:
                gamepad.update_center(None, shape_for_pad)

            # ESC キーで終了
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        # ── 後片付け ──────────────────────────────────────────
        gamepad.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
import time, threading
import vgamepad as vg


def center_to_axis_int(center_px, img_w, img_h, invert_y=True):
    """
    画像座標 → vgamepad.int16（‐32 768～+32 767）へ線形マッピング
    * 画面中心 (W/2, H/2) が 0
    * 左/上 = 負、右/下 = 正（XInput 規格）
    """
    cx, cy = center_px
    # 正規化値 (-1.0～+1.0)
    x_norm =  (cx - img_w / 2) / (img_w / 2)
    y_norm =  (cy - img_h / 2) / (img_h / 2)
    if invert_y:            # 上を負に合わせる
        y_norm = -y_norm

    # 32 767 を掛けて int16 に丸める
    to_int16 = lambda v: int(max(-32767, min(32767, v * 32767)))
    return to_int16(x_norm), to_int16(y_norm)


class GamepadController:
    def __init__(self, fps=120, target_id=0, ds4_mode=False):
        self.pad   = vg.VDS4Gamepad() if ds4_mode else vg.VX360Gamepad()
        self.fps   = fps
        self.dt    = 1.0 / fps
        self.id    = target_id
        self._lock = threading.Lock()
        self._center_px = None
        self._img_size  = (1, 1)   # 初期化（0 除算防止）
        self._running   = False
        self._thread    = None

    # ---------- 外部 API ----------
    def update_center(self, center_px, img_shape):
        with self._lock:
            self._center_px = center_px
            if img_shape is not None:
                self._img_size = (img_shape[1], img_shape[0])  # (W, H)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join()
        self.pad.reset()
        self.pad.update()

    # ---------- 内部ループ ----------
    def _loop(self):
        while self._running:
            # 共有データをコピー
            with self._lock:
                center = self._center_px
                img_w, img_h = self._img_size  # ← ここに入ってくる

            # ----- ガードをここに追加 -----
            # 画像サイズが未設定 (1,1) の間はニュートラルを送信
            if img_w <= 1 or img_h <= 1:
                self.pad.right_joystick(0, 0)
                self.pad.update()
                time.sleep(self.dt)
                continue
            # --------------------------------

            if center is not None:
                x_axis, y_axis = center_to_axis_int(
                    center, img_w, img_h,
                    invert_y=not isinstance(self.pad, vg.VDS4Gamepad)
                )
                self.pad.right_joystick(x_axis, y_axis)
            else:
                self.pad.right_joystick(0, 0)

            self.pad.update()
            time.sleep(self.dt)
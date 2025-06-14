import cv2

def detect_blobs(img_trans, width_mm=210, height_mm=210, image_width_px=840):
    tmp = img_trans.copy()

    # (1) グレースケール変換
    tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)

    # (2) ぼかし処理
    tmp = cv2.GaussianBlur(tmp, (11, 11), 0)

    # (3) 二値化処理
    _, tmp = cv2.threshold(tmp, 130, 255, cv2.THRESH_BINARY_INV)

    # (4) ブロブ検出
    n, img_label, data, center = cv2.connectedComponentsWithStats(tmp)

    # (5) 検出結果の整理
    detected_obj = []
    tr_x = lambda x: x * width_mm / image_width_px
    tr_y = lambda y: y * height_mm / image_width_px
    img_trans_marked = img_trans.copy()

    for i in range(1, n):  # 0番は背景
        x, y, w, h, size = data[i]
        if size < 300:
            continue
        detected_obj.append(dict(
            x=tr_x(x),
            y=tr_y(y),
            w=tr_x(w),
            h=tr_y(h),
            cx=tr_x(center[i][0]),
            cy=tr_y(center[i][1])
        ))
        cv2.rectangle(img_trans_marked, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(img_trans_marked, (int(center[i][0]), int(center[i][1])), 5, (0, 0, 255), -1)
        text = f'({int(center[i][0])}, {int(center[i][1])})'
        cv2.putText(img_trans_marked, text, (int(center[i][0]), int(center[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_trans_marked, detected_obj
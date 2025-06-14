# Research02/perspective_transformer.py
import cv2
import numpy as np

__all__ = ["warp_by_aruco", "pixel_to_center"]


def warp_by_aruco(img, marker_ids=None, width=840, height=840):
    """
    4 つの ArUco で囲った領域を真俯瞰に射影する。
    射影先は通常の (0,0) ～ (width-1, height-1)。

    Parameters
    ----------
    img : np.ndarray (BGR)
    marker_ids : [LT, RT, RB, LB] に対応する 4 ID。None → [0,1,2,3]
    width, height : 出力画像サイズ

    Returns
    -------
    img_warp : np.ndarray | None
        warp 成功 → 俯瞰画像、失敗 → None
    """
    if marker_ids is None:
        marker_ids = [0, 1, 2, 3]

    aruco = cv2.aruco
    p_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    corners, ids, _ = aruco.detectMarkers(img, p_dict)

    if ids is None or len(ids) < 4:
        # 必要な 4 マーカーが見つからない
        return None

    # ID → corner の並び替え
    ordered = [None] * 4
    for i, c in zip(ids.ravel(), corners):
        if i in marker_ids:
            ordered[marker_ids.index(i)] = c

    if any(c is None for c in ordered):
        # 指定 ID が揃っていない
        return None

    # --------- 射影元 4 点（印刷向きに合わせて要調整） ---------
    src = np.float32(
        [
            ordered[0][0][2],  # 左上マーカーの右下
            ordered[1][0][3],  # 右上           左下
            ordered[2][0][0],  # 右下           左上
            ordered[3][0][1],  # 左下           右上
        ]
    )

    # --------- 射影先 4 点：通常のピクセル座標 ---------
    dst = np.float32(
        [
            [0, 0],  # 左上
            [width - 1, 0],  # 右上
            [width - 1, height - 1],  # 右下
            [0, height - 1],  # 左下
        ]
    )

    H = cv2.getPerspectiveTransform(src, dst)
    img_warp = cv2.warpPerspective(img, H, (width, height))
    return img_warp


# ------------------------------------------------------------ #
# 補助：warp 後ピクセル → 中心原点座標 (右 +X, 上 +Y)
# ------------------------------------------------------------ #
def pixel_to_center(pt_xy, width, height):
    """(px,py) → (cx,cy) へ変換"""
    px, py = pt_xy
    cx = px - width * 0.5
    cy = height * 0.5 - py
    return cx, cy
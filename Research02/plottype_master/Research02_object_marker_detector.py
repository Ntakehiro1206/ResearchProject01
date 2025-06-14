import cv2
from cv2 import aruco

from Research02.plottype_master.aruco_marker_detector import dict_aruco


def object_marker_detector(img_trans, debug=False):
    grayscale = cv2.cvtColor(img_trans, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(grayscale, dict_aruco)

    result = img_trans.copy()
    detected_obj = []

    if ids is not None:
        aruco.drawDetectedMarkers(result, corners, ids)

        for i, corner in enumerate(corners):
            marker_id = int(ids[i][0])
            center = corner[0].mean(axis=0)
            center_x, center_y = int(center[0]), int(center[1])
            detected_obj.append({"id": marker_id, "center": (center_x, center_y)})
    else:
        if debug:
            cv2.putText(result, "範囲内にマーカーの検出なし", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return result, detected_obj





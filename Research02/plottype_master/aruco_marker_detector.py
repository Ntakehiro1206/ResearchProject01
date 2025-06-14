import cv2
from cv2 import aruco

dict_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

def detect_markers(frame):
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(grayscale, dict_aruco)

    result = frame.copy()

    if ids is not None:
        aruco.drawDetectedMarkers(result, corners, ids)

        for i, corner in enumerate(corners):
            top_left = tuple(corner[0][0].astype(int))
            marker_id = int(ids[i][0])
            cv2.putText(result, f"ID:{marker_id}", top_left,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(result, "Cant detect markers",(10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2 )

    return result

import cv2

aruco = cv2.aruco

dic_aruco = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

marker_size = 100
for marker_id in range(5):
    marker_img = aruco.generateImageMarker(dic_aruco, marker_id, marker_size)
    cv2.imshow(f"id_{marker_id:02}", marker_img)
    cv2.imwrite(f"id_{marker_id:02}.png", marker_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

import cv2
import numpy as np
camera_mat = np.array([[307.23545322976406, 0.0, 810.1725676249877], [0.0, 306.8632611459552, 618.028423327481], [0.0, 0.0, 1.0]])
dist_coeff = np.array([[0.06905767483991472], [-0.007408864190620307], [-0.004361045823840081], [8.144610119937468e-05]])
frame_shape = None
def undistort(frame):
    global frame_shape
    if frame_shape is None:
        frame_shape = frame.shape[1::-1]
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            camera_mat, dist_coeff, np.eye(3, 3), camera_mat, frame_shape, cv2.CV_16SC2)
    undist_frame = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR);
    return undist_frame
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 1600)
    cap.set(4, 1200)
    if not cap.isOpened(): raise "camera open failed"
    ok, raw_frame = cap.read()
    if not ok: raise "image read failed"
    undist_frame = undistort(raw_frame)
    cv2.imwrite("sample.png", undist_frame)


import cv2
import numpy as np
camera_mat = np.array([[122.7938597429089, 0.0, 327.23470729912793], [0.0, 123.70781204711702, 245.88701007032986], [0.0, 0.0, 1.0]])
dist_coeff = np.array([[0.041843092832358535], [0.006039819981015406], [0.013091410556075288], [-0.008650444406630757]])
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
    if not cap.isOpened(): raise "camera open failed"
    ok, raw_frame = cap.read()
    if not ok: raise "image read failed"
    undist_frame = undistort(raw_frame)
    cv2.imwrite("sample.png", undist_frame)

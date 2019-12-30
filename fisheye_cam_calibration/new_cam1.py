
import cv2
import numpy as np
camera_mat = np.array([[306.2179554207244, 0.0, 827.6396074947415], [0.0, 306.4032729038873, 612.7686375589241], [0.0, 0.0, 1.0]])
dist_coeff = np.array([[0.061469387879628086], [-0.018668965639305285], [0.03781464752574166], [-0.02677018297407956]])
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

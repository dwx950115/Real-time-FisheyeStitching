
import cv2
import numpy as np
camera_mat = np.array([[118.503890023102, 0.0, 350.5807111179997], [0.0, 118.62339323954097, 246.1061709844143], [0.0, 0.0, 1.0]])
dist_coeff = np.array([[0.13698731869687505], [-0.11179250973001682], [0.03139233829470256], [0.004616830509215432]])
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


import cv2
import numpy as np
camera_mat = np.array([[305.8103327739443, 0.0, 868.4087733419921], [0.0, 306.2007353016409, 628.2346880123545], [0.0, 0.0, 1.0]])
dist_coeff = np.array([[0.05474630002513208], [-0.003933657882507959], [0.03314157850495621], [-0.03770454084099611]])
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
    cap = cv2.VideoCapture(1)
    cap.set(3, 1600)
    cap.set(4, 1200)
    if not cap.isOpened(): raise "camera open failed"
    ok, raw_frame = cap.read()
    if not ok: raise "image read failed"
    undist_frame = undistort(raw_frame)
    cv2.imwrite("sample.png", undist_frame)
